"""Gate 7 P17: ThresholdRegister focused tests.

Predeclaration: docs/validation/2026-07-06-gate7-p17-predeclaration.md.
"""
import ast
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
    assert s.keep_weak_result(result(1.0), None)
    assert not s.keep_weak_result(result(0.9), None)
    assert s.keep_weak_result(result(1.0), job("d5"))   # job ignored


def test_register_overrides_scalar_per_code():
    reg = ThresholdRegister(default=1.0, per_code={"hard": 5.0})
    s = Switching(confidence_threshold=99.0, threshold_register=reg)
    assert s.keep_weak_result(result(1.5), job("easy"))     # default 1.0
    assert not s.keep_weak_result(result(1.5), job("hard")) # 5.0
    reg.set("hard", 1.0)                                    # live update
    assert s.keep_weak_result(result(1.5), job("hard"))
    # no job / no code -> scalar path preserved
    assert not s.keep_weak_result(result(1.5), None)
    assert not s.keep_weak_result(result(1.5), job(None))


def test_canonical_override_receives_result_and_job_directly():
    class Two(Switching):
        def keep_weak_result(self, result, job):
            return job.code == "keep"
    s = Two(confidence_threshold=0.0)
    assert s.keep_weak_result(result(0.0), job("keep"))
    assert not s.keep_weak_result(result(9.9), job("drop"))


def test_every_controlled_threshold_override_has_the_exact_contract():
    root = pathlib.Path(__file__).resolve().parent.parent
    paths = [root / "decsim" / "switching.py", *sorted(
        (root / "tests").glob("test_*.py")
    )]
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "keep_weak_result":
                continue
            assert [argument.arg for argument in node.args.args] == [
                "self",
                "result",
                "job",
            ]
            assert node.args.posonlyargs == []
            assert node.args.vararg is None
            assert node.args.kwonlyargs == []
            assert node.args.kw_defaults == []
            assert node.args.kwarg is None
            assert node.args.defaults == []
            found.append((path, node.lineno))
    assert found
