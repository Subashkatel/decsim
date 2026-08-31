from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import inspect
import shutil
import subprocess
import zipfile

import pytest

from decsim.pauli_frame.pauli_frame import PauliFrame as RuntimePauliFrame
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.protocols import PauliFrame as PauliFramePort
import decsim.run_spec as run_spec_module
from decsim.run_spec import RunSpec
from decsim.windows.window_manager import WindowManager


class ManualEngine:
    def __init__(self, now=0):
        self.now = now
        self.scheduled = []

    def schedule(self, delay_ticks, callback, label=""):
        self.scheduled.append((self.now + delay_ticks, callback, label))

    def log_io(self, who, message):
        """The I/O trace is off in these tests; the frame still narrates."""

    def run_next(self):
        event_ticks, callback, _ = self.scheduled.pop(0)
        self.now = event_ticks
        callback()

    def run_all(self):
        while self.scheduled:
            self.run_next()


def request(run_sequence=0, *, tier="weak", operation_id=0, window_id=0):
    return SimpleNamespace(
        tier=SimpleNamespace(value=tier),
        run_sequence=run_sequence,
        operation_id=operation_id,
        window_id=window_id,
    )


def accept(frame, window_key, observables, *, run_sequence=0, callback=lambda: None):
    frame.commit_weak_correction(
        window_key=window_key,
        logical_observables=observables,
        request_key=request(run_sequence),
        on_committed=callback,
    )


def test_wide_elementwise_xor_cancels_without_a_word_size_limit():
    engine = ManualEngine()
    frame = RuntimePauliFrame(engine, commit_ticks=0)
    left = tuple(index % 2 for index in range(130))
    right = tuple((index // 3) % 2 for index in range(130))

    accept(frame, (41, 0), left)
    accept(frame, (41, 1), right)
    assert frame.frame_for(41) == tuple(a ^ b for a, b in zip(left, right))

    accept(frame, (41, 2), left)
    accept(frame, (41, 3), right)
    assert frame.frame_for(41) == (0,) * 130


def test_none_dominates_one_stream_without_affecting_other_streams():
    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    accept(frame, (8, 0), (1, 0, 1))
    accept(frame, (8, 1), None)
    accept(frame, (9, 0), (0, 1, 1))

    assert frame.frame_for(8) is None
    assert frame.frame_for(9) == (0, 1, 1)
    assert frame.frame_for(404) == ()


def test_changed_observable_arity_is_rejected_when_the_stream_is_read():
    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    accept(frame, (5, 0), (1, 0))
    accept(frame, (5, 1), (0, 1, 0))

    with pytest.raises(RuntimeError, match="changed observable arity"):
        frame.frame_for(5)


def test_pending_and_installed_duplicates_are_dropped_without_work_or_charge():
    engine = ManualEngine(now=17)
    frame = RuntimePauliFrame(engine, commit_ticks=11)
    continuations = []

    accept(frame, (3, 7), [1, 0], run_sequence=2,
           callback=lambda: continuations.append(("accepted", engine.now)))
    accept(frame, (3, 7), [0, 1], run_sequence=3,
           callback=lambda: continuations.append(("pending duplicate", engine.now)))

    pending = frame.snapshot()
    assert len(engine.scheduled) == 1
    assert pending.commit_count == 1
    assert pending.pending_write_count == 1
    assert pending.duplicate_drop_count == 1
    assert pending.charged_ticks == 11
    assert continuations == []

    engine.run_all()
    accept(frame, (3, 7), [1, 1], run_sequence=4,
           callback=lambda: continuations.append(("installed duplicate", engine.now)))

    settled = frame.snapshot()
    assert engine.scheduled == []
    assert continuations == [("accepted", 28)]
    assert settled.commit_count == 1
    assert settled.pending_write_count == 0
    assert settled.duplicate_drop_count == 2
    assert settled.charged_ticks == 11
    assert [drop.run_sequence for drop in settled.duplicate_drops] == [3, 4]


def test_positive_latency_installs_then_continues_exactly_once():
    engine = ManualEngine(now=100)
    frame = RuntimePauliFrame(engine, commit_ticks=9)
    calls = []

    accept(frame, (12, 4), (1, 1), callback=lambda: calls.append(engine.now))
    assert calls == []
    assert frame.frame_for(12) == ()
    assert frame.snapshot().pending_write_count == 1
    assert [(ticks, label) for ticks, _, label in engine.scheduled] == [
        (109, "pauli frame commit (12, 4)")
    ]

    engine.run_all()
    assert calls == [109]
    assert frame.frame_for(12) == (1, 1)
    assert frame.snapshot().pending_write_count == 0


def test_explicit_zero_installs_inline_and_schedules_nothing():
    engine = ManualEngine(now=23)
    frame = RuntimePauliFrame(engine, commit_ticks=0)
    calls = []

    accept(frame, (6, 2), (1,), callback=lambda: calls.append(engine.now))

    assert calls == [23]
    assert engine.scheduled == []
    assert frame.frame_for(6) == (1,)
    snapshot = frame.snapshot()
    assert snapshot.records[0].accepted_ticks == 23
    assert snapshot.records[0].committed_ticks == 23
    assert snapshot.charged_ticks == 0


def test_records_and_prior_snapshots_are_deeply_immutable_reports():
    engine = ManualEngine()
    frame = RuntimePauliFrame(engine, commit_ticks=0)
    caller_owned = [1, 0, 1]
    accept(frame, (20, 0), caller_owned)
    frozen_before = frame.snapshot()
    caller_owned[0] = 0
    accept(frame, (20, 1), (1, 1, 0))

    assert frozen_before.commit_count == 1
    assert frozen_before.frames == ((20, (1, 0, 1)),)
    assert frozen_before.records[0].logical_observables == (1, 0, 1)
    assert frame.snapshot().commit_count == 2
    with pytest.raises(FrozenInstanceError):
        frozen_before.commit_count = 99
    with pytest.raises(FrozenInstanceError):
        frozen_before.records[0].accepted_ticks = 99


def test_configuration_rejects_implicit_or_disappearing_costs():
    with pytest.raises(TypeError):
        PauliFrameConfig()
    for invalid in (-1.0, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            PauliFrameConfig(commit_us=invalid)
    with pytest.raises(ValueError, match="rounds to zero ticks"):
        PauliFrameConfig(commit_us=1e-12)
    with pytest.raises(ValueError, match="requires zero_commit_cost_justification"):
        PauliFrameConfig(commit_us=0.0)
    with pytest.raises(ValueError, match="requires zero_commit_cost_justification"):
        PauliFrameConfig(commit_us=0.0, zero_commit_cost_justification="")
    with pytest.raises(ValueError, match="only valid for zero"):
        PauliFrameConfig(commit_us=1.0, zero_commit_cost_justification="free")

    zero = PauliFrameConfig(
        commit_us=0.0,
        zero_commit_cost_justification="Idealized register write for this run.",
    )
    assert zero.commit_ticks() == 0
    assert zero.resolve(ManualEngine()).commit_ticks == 0
    assert PauliFrameConfig(commit_us=1.0).commit_ticks() > 0



def test_default_toggle_is_absent_and_zero_cost_matches_the_inline_off_path():
    assert RunSpec().pauli_frame is None

    job = SimpleNamespace(request_key=request(1), op_id=7, window_id=2)
    result = SimpleNamespace(logical_observables=(1, 0))
    off_calls = []
    off_manager = object.__new__(WindowManager)
    off_manager.pauli_frame = None
    off_manager._commit_decode_done = lambda actual_job, actual_result: off_calls.append(
        (actual_job, actual_result)
    )
    WindowManager._sink_weak_correction(off_manager, job, result)

    zero_engine = ManualEngine()
    zero_calls = []
    zero_manager = object.__new__(WindowManager)
    zero_manager.pauli_frame = RuntimePauliFrame(zero_engine, commit_ticks=0)
    zero_manager._commit_decode_done = lambda actual_job, actual_result: zero_calls.append(
        (actual_job, actual_result)
    )
    WindowManager._sink_weak_correction(zero_manager, job, result)

    assert off_calls == zero_calls == [(job, result)]
    assert zero_engine.scheduled == []


def test_final_results_use_the_sink_while_provisional_and_delivery_legs_bypass_it():
    job = SimpleNamespace(
        request_key=request(8), op_id=4, window_id=1,
        awaiting_strong_result=False,
    )
    result = SimpleNamespace(logical_observables=(1, 0))
    sink_calls = []
    weak_commits = []
    sink = SimpleNamespace(
        commit_weak_correction=lambda **kwargs: sink_calls.append(kwargs)
    )
    final_engine = ManualEngine(now=10)
    manager = object.__new__(WindowManager)
    manager.engine = final_engine
    manager.pauli_frame = sink
    manager.windows = {(4, 1): SimpleNamespace(t_done=None, k=1)}
    manager._ops = {4: SimpleNamespace(name="logical")}
    manager._window_link_arrival = lambda *args: final_engine.now + 4
    manager._commit_decode_done = lambda actual_job, actual_result: weak_commits.append(
        (actual_job, actual_result)
    )
    handed_on = []
    manager._hand_on_boundary = lambda *args: handed_on.append(final_engine.now)

    WindowManager.on_decode_done(manager, job, result)
    assert manager.windows[(4, 1)].t_done == 10
    assert handed_on == [10]          # boundary leaves at decode done, before WDO and the sink
    assert sink_calls == []
    assert weak_commits == []
    final_engine.run_all()
    assert sink_calls[0]["window_key"] == (4, 1)
    assert sink_calls[0]["logical_observables"] == (1, 0)
    assert sink_calls[0]["request_key"] is job.request_key
    sink_calls[0]["on_committed"]()
    assert weak_commits == [(job, result)]

    engine = ManualEngine(now=31)
    provisional = object.__new__(WindowManager)
    provisional.engine = engine
    provisional.pauli_frame = SimpleNamespace(
        commit_weak_correction=lambda **kwargs: pytest.fail("provisional weak reached sink")
    )
    provisional.windows = {(4, 1): SimpleNamespace(t_done=None)}
    provisional._commit_decode_done = lambda actual_job, actual_result: weak_commits.append(
        (actual_job, actual_result)
    )
    job.awaiting_strong_result = True
    WindowManager.on_decode_done(provisional, job, result)
    assert provisional.windows[(4, 1)].t_done == 31
    assert weak_commits[-1] == (job, result)
    assert engine.scheduled == []

    strong_calls = []
    strong = object.__new__(WindowManager)
    strong.engine = engine
    strong.pauli_frame = SimpleNamespace(
        commit_weak_correction=lambda **kwargs: pytest.fail(
            "the DO delivery leg must not touch the frame; the fold happens "
            "at the strong commit")
    )
    strong.windows = {(4, 1): SimpleNamespace(op_id=4, k=1)}
    strong._ops = {4: SimpleNamespace(name="logical")}
    strong._window_link_arrival = lambda *args: engine.now + 5
    strong._commit_strong_decode_done = lambda completion: strong_calls.append(completion)
    completion = SimpleNamespace(request_key=request(
        9, tier="strong", operation_id=4, window_id=1
    ))
    WindowManager.on_strong_decode_done(strong, completion)
    assert strong_calls == []
    engine.run_all()
    assert strong_calls == [completion]


def test_runtime_satisfies_the_declared_keyword_only_correction_seam():
    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    assert isinstance(frame, PauliFramePort)
    parameters = inspect.signature(PauliFramePort.commit_weak_correction).parameters
    assert tuple(parameters) == (
        "self", "window_key", "logical_observables", "request_key", "on_committed"
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in parameters.items()
        if name != "self"
    )
    assert frame.snapshot() == frame.snapshot()


def test_frame_owner_is_a_named_seed_root_and_snapshot_is_non_destructive():
    owner = object()
    roots = run_spec_module._seed_roots(pauli_frame=owner, metrics=())
    expected_path = (run_spec_module.RunSeedPathSegment("field", "pauli_frame"),)
    assert roots == ((expected_path, owner),)

    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    accept(frame, (2, 0), (1, 0))
    before = frame.snapshot()
    assert frame.snapshot() == before
    assert frame.frame_for(2) == (1, 0)
    assert frame.snapshot() == before


def test_short_provenance_header_round_trips_from_source_and_wheel(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    module_bytes = (project_root / "decsim" / "pauli_frame" / "pauli_frame.py").read_bytes()
    module_lines = module_bytes.decode("ascii").splitlines()
    expected_prefix = [
        "# Data-core semantics adapted from PECOS PauliFrameAccumulator and ObsMask.",
        "# Source: https://github.com/PECOS-packages/PECOS",
        "# Commit: 7c679509ec7e87410f99445c2ec5442eb91016fd",
        "# Files: crates/pecos-decoder-core/src/pauli_frame.rs:53-108; crates/pecos-decoder-core/src/obs_mask.rs:132-141",
        "# Copyright 2026 The PECOS Developers.",
        "# License: Apache-2.0. Full text: tmp/references/code/pecos/LICENSE.",
    ]
    assert module_lines[:6] == expected_prefix
    notice_lines = []
    for line in module_lines[6:12]:
        assert line.startswith("# NOTICE: ")
        notice_lines.append(line.removeprefix("# NOTICE: "))
    assert ("\n".join(notice_lines) + "\n").encode("ascii") == (
        project_root / "tmp/references/code/pecos/NOTICE"
    ).read_bytes()
    modification_line = (
        "# Modified for decsim: Python tuple/None semantics, stream and window keys, "
        "idempotent async commit transaction, immutable records, and simulated commit latency."
    )
    assert module_lines[12] == modification_line
    assert "BEGIN EMBEDDED APACHE" not in module_bytes.decode("ascii")

    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    subprocess.run(
        [
            uv, "run", "--no-project", "--isolated",
            "--with", "setuptools>=68", "--with", "wheel", "--with", "build",
            "python", "-m", "build", "--no-isolation", "--wheel",
            "-o", str(tmp_path), str(project_root),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_paths = list(tmp_path.glob("*.whl"))
    assert len(wheel_paths) == 1
    with zipfile.ZipFile(wheel_paths[0]) as wheel:
        shipped_module = wheel.read("decsim/pauli_frame/pauli_frame.py")
    assert shipped_module.startswith(("\n".join(module_lines[:13]) + "\n").encode("ascii"))
    assert b"BEGIN EMBEDDED APACHE" not in shipped_module



def test_escalated_strong_final_folds_into_the_frame_and_gates_the_commit():
    """The strong result is the window's final correction: it folds into the
    frame (the provisional weak bypassed it) and the priced write gates the
    rest of the commit."""
    engine = ManualEngine(now=50)
    manager = object.__new__(WindowManager)
    manager.engine = engine
    frame_calls = []
    manager.pauli_frame = SimpleNamespace(
        commit_weak_correction=lambda **kwargs: frame_calls.append(kwargs)
    )
    manager.windows = {(4, 1): SimpleNamespace(op_id=4, k=1)}
    manager._ops = {4: SimpleNamespace(id=4, name="logical")}
    manager.op_strong_commit_time = {}
    manager._selected_request_keys = None
    finished = []
    manager._finish_strong_commit = (
        lambda completion, key, result, window, op: finished.append(key))
    completion = SimpleNamespace(
        request_key=request(9, tier="strong", operation_id=4, window_id=1),
        result=SimpleNamespace(logical_observables=(1,)),
    )
    WindowManager._commit_strong_decode_done(manager, completion)
    assert frame_calls[0]["window_key"] == (4, 1)
    assert frame_calls[0]["logical_observables"] == (1,)
    assert frame_calls[0]["request_key"] is completion.request_key
    assert finished == []            # the priced frame write gates the commit
    frame_calls[0]["on_committed"]()
    assert finished == [(4, 1)]


def test_frameless_strong_commit_finishes_directly():
    engine = ManualEngine(now=50)
    manager = object.__new__(WindowManager)
    manager.engine = engine
    manager.pauli_frame = None
    manager.windows = {(4, 1): SimpleNamespace(op_id=4, k=1)}
    manager._ops = {4: SimpleNamespace(id=4, name="logical")}
    manager.op_strong_commit_time = {}
    manager._selected_request_keys = None
    finished = []
    manager._finish_strong_commit = (
        lambda completion, key, result, window, op: finished.append(key))
    completion = SimpleNamespace(
        request_key=request(9, tier="strong", operation_id=4, window_id=1),
        result=SimpleNamespace(logical_observables=(1,)),
    )
    WindowManager._commit_strong_decode_done(manager, completion)
    assert finished == [(4, 1)]


def test_final_result_rides_its_tiers_output_link():
    """WDO carries a weak final home; DO carries a strong-primary final."""
    from decsim.links.links import LinkPath
    from decsim.message import DecoderRequestKey, DecoderTier

    for tier, expected_path in ((DecoderTier.WEAK, LinkPath.WDO),
                                (DecoderTier.STRONG, LinkPath.DO)):
        engine = ManualEngine(now=10)
        manager = object.__new__(WindowManager)
        manager.engine = engine
        manager.pauli_frame = None
        manager.windows = {(4, 1): SimpleNamespace(t_done=None, k=1)}
        manager._ops = {4: SimpleNamespace(name="logical")}
        paths = []
        def arrival(path, window, op, request_key, paths=paths):
            paths.append(path)
            return engine.now + 4
        manager._window_link_arrival = arrival
        manager._hand_on_boundary = lambda *args: None
        manager._commit_decode_done = lambda job, result: None
        job = SimpleNamespace(
            request_key=DecoderRequestKey(4, 1, tier, 0),
            op_id=4, window_id=1, awaiting_strong_result=False,
        )
        WindowManager.on_decode_done(
            manager, job, SimpleNamespace(logical_observables=(0,)))
        assert paths == [expected_path], tier
