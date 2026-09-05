from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import inspect

import pytest

from decsim.pauli_frame.pauli_frame import PauliFrame as RuntimePauliFrame
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.ports import Frame as PauliFramePort
import decsim.machine as machine_module
from decsim.machine import MachineSettings
from decsim.message import RunSeedPathSegment


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
    frame.commit_correction(
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
    assert frame.frame_for_stream(41) == tuple(a ^ b for a, b in zip(left, right))

    accept(frame, (41, 2), left)
    accept(frame, (41, 3), right)
    assert frame.frame_for_stream(41) == (0,) * 130


def test_none_dominates_one_stream_without_affecting_other_streams():
    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    accept(frame, (8, 0), (1, 0, 1))
    accept(frame, (8, 1), None)
    accept(frame, (9, 0), (0, 1, 1))

    assert frame.frame_for_stream(8) is None
    assert frame.frame_for_stream(9) == (0, 1, 1)
    assert frame.frame_for_stream(404) == ()


def test_changed_observable_arity_is_rejected_when_the_stream_is_read():
    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    accept(frame, (5, 0), (1, 0))
    accept(frame, (5, 1), (0, 1, 0))

    with pytest.raises(RuntimeError, match="changed its number of observables"):
        frame.frame_for_stream(5)


def test_a_second_correction_for_a_window_is_refused_loudly():
    """One authoritative correction per window enters the frame; a second
    write for the same window, pending or installed, is a protocol
    violation and raises instead of being dropped, so no caller's
    continuation is ever silently lost. A Pauli frame applies each
    window's correction exactly once (Riesebos et al. DAC 2017; PECOS
    frame semantics)."""
    engine = ManualEngine(now=17)
    frame = RuntimePauliFrame(engine, commit_ticks=11)
    continuations = []

    accept(frame, (3, 7), [1, 0], run_sequence=2,
           callback=lambda: continuations.append(("accepted", engine.now)))
    with pytest.raises(RuntimeError, match="already has a"):
        accept(frame, (3, 7), [0, 1], run_sequence=3,
               callback=lambda: continuations.append(("pending duplicate", engine.now)))

    engine.run_all()
    assert continuations == [("accepted", 28)]
    with pytest.raises(RuntimeError, match="already has a"):
        accept(frame, (3, 7), [1, 1], run_sequence=4,
               callback=lambda: continuations.append(("installed duplicate", engine.now)))
    settled = frame.snapshot()
    assert settled.commit_count == 1
    assert settled.pending_write_count == 0
    assert settled.charged_ticks == 11


def test_positive_latency_installs_then_continues_exactly_once():
    engine = ManualEngine(now=100)
    frame = RuntimePauliFrame(engine, commit_ticks=9)
    calls = []

    accept(frame, (12, 4), (1, 1), callback=lambda: calls.append(engine.now))
    assert calls == []
    assert frame.frame_for_stream(12) == ()
    assert frame.snapshot().pending_write_count == 1
    assert [(ticks, label) for ticks, _, label in engine.scheduled] == [
        (109, "pauli frame commit (12, 4)")
    ]

    engine.run_all()
    assert calls == [109]
    assert frame.frame_for_stream(12) == (1, 1)
    assert frame.snapshot().pending_write_count == 0


def test_explicit_zero_installs_inline_and_schedules_nothing():
    engine = ManualEngine(now=23)
    frame = RuntimePauliFrame(engine, commit_ticks=0)
    calls = []

    accept(frame, (6, 2), (1,), callback=lambda: calls.append(engine.now))

    assert calls == [23]
    assert engine.scheduled == []
    assert frame.frame_for_stream(6) == (1,)
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
            PauliFrameConfig(commit_microseconds=invalid)
    with pytest.raises(ValueError, match="rounds to zero ticks"):
        PauliFrameConfig(commit_microseconds=1e-12)
    with pytest.raises(ValueError, match="needs zero_commit_cost_justification"):
        PauliFrameConfig(commit_microseconds=0.0)
    with pytest.raises(ValueError, match="needs zero_commit_cost_justification"):
        PauliFrameConfig(commit_microseconds=0.0, zero_commit_cost_justification="")
    with pytest.raises(ValueError, match="needs a free write"):
        PauliFrameConfig(commit_microseconds=1.0, zero_commit_cost_justification="free")

    zero = PauliFrameConfig(
        commit_microseconds=0.0,
        zero_commit_cost_justification="Idealized register write for this run.",
    )
    assert zero.commit_ticks() == 0
    assert zero.resolve(ManualEngine()).commit_ticks == 0
    assert PauliFrameConfig(commit_microseconds=1.0).commit_ticks() > 0



def test_runtime_satisfies_the_declared_keyword_only_correction_seam():
    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    assert isinstance(frame, PauliFramePort)
    parameters = inspect.signature(PauliFramePort.commit_correction).parameters
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
    roots = machine_module._seed_roots(pauli_frame=owner, metrics=())
    expected_path = (RunSeedPathSegment("field", "pauli_frame"),)
    assert roots == ((expected_path, owner),)

    frame = RuntimePauliFrame(ManualEngine(), commit_ticks=0)
    accept(frame, (2, 0), (1, 0))
    before = frame.snapshot()
    assert frame.snapshot() == before
    assert frame.frame_for_stream(2) == (1, 0)
    assert frame.snapshot() == before
