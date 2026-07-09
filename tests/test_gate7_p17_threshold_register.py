"""Gate 7 P17: ThresholdRegister focused tests.

Predeclaration: docs/validation/2026-07-06-gate7-p17-predeclaration.md.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.message import DecodeJob, DecodeResult
from decsim.switching import Switching, ThresholdRegister


def result(soft):
    return DecodeResult(0, 0, soft_output=soft)


def job(code):
    return DecodeJob(op_id=0, window_id=0, n_rounds=1, code=code)


def test_register_lookup_default_and_update_history():
    reg = ThresholdRegister(default=0.5, per_code={"d5": 1.0})
    assert reg.get("d5") == 1.0
    assert reg.get("d7") == 0.5
    reg.set("d7", 2.0)
    assert reg.get("d7") == 2.0
    assert reg.history == [(1, "d7", 0.5, 2.0)]


def test_scalar_semantics_unchanged_without_register():
    s = Switching(confidence_threshold=1.0)
    assert s.keep_weak_result(result(1.0))
    assert not s.keep_weak_result(result(0.9))
    assert s.keep_weak_result(result(1.0), job("d5"))   # job ignored


def test_register_overrides_scalar_per_code():
    reg = ThresholdRegister(default=1.0, per_code={"hard": 5.0})
    s = Switching(confidence_threshold=99.0, threshold_register=reg)
    assert s.keep_weak_result(result(1.5), job("easy"))     # default 1.0
    assert not s.keep_weak_result(result(1.5), job("hard")) # 5.0
    reg.set("hard", 1.0)                                    # live update
    assert s.keep_weak_result(result(1.5), job("hard"))
    # no job / no code -> scalar path preserved
    assert not s.keep_weak_result(result(1.5))
    assert not s.keep_weak_result(result(1.5), job(None))


# ---- legacy-override dispatch (P15-P18 review finding C2): every
# plausible override signature must be called correctly by _keep_weak.

def test_dispatch_legacy_single_arg_override():
    class Legacy(Switching):
        def keep_weak_result(self, result):
            return result.soft_output >= 2.0
    s = Legacy(confidence_threshold=0.0)
    assert s._keep_weak(result(2.0), job("d5"))
    assert not s._keep_weak(result(1.9), job("d5"))


def test_dispatch_varargs_override_receives_job():
    seen = {}
    class Star(Switching):
        def keep_weak_result(self, *args):
            seen["n"] = len(args)
            return True
    s = Star(confidence_threshold=0.0)
    assert s._keep_weak(result(1.0), job("d5"))
    assert seen["n"] == 2                      # job passed, not dropped


def test_dispatch_keyword_only_job_override():
    seen = {}
    class KwOnly(Switching):
        def keep_weak_result(self, result, *, job=None):
            seen["job"] = job
            return True
    s = KwOnly(confidence_threshold=0.0)
    j = job("d5")
    assert s._keep_weak(result(1.0), j)
    assert seen["job"] is j


def test_dispatch_kwargs_override():
    seen = {}
    class Kw(Switching):
        def keep_weak_result(self, result, **kwargs):
            seen.update(kwargs)
            return True
    s = Kw(confidence_threshold=0.0)
    j = job("d5")
    assert s._keep_weak(result(1.0), j)
    assert seen["job"] is j


def test_dispatch_two_positional_override():
    class Two(Switching):
        def keep_weak_result(self, result, job):
            return job.code == "keep"
    s = Two(confidence_threshold=0.0)
    assert s._keep_weak(result(0.0), job("keep"))
    assert not s._keep_weak(result(9.9), job("drop"))
