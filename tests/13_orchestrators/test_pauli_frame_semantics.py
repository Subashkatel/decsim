"""The Pauli frame behaves like a real frame and charges what it says.

Sources: PECOS pauli_frame.rs (a stream's frame is the XOR of its
corrections; folding is non-destructive); Riesebos, Pauli frames for
quantum computer architectures, DAC 2017 (a correction is applied exactly
once); Yang et al. 2605.04892 Fig. 1 (one frame update costs one cycle,
4 ns at 250 MHz, and the loop waits for it).
"""

import dataclasses
import enum

import pytest

from decsim.engine import Engine
from decsim.pauli_frame.pauli_frame import PauliFrame, PauliFrameConfig


class Tier(enum.Enum):
    WEAK = "weak"
    STRONG = "strong"


@dataclasses.dataclass(frozen=True)
class RequestKey:
    tier: Tier
    run_sequence: int


def frame_with_commit_ticks(commit_ticks):
    engine = Engine(verbose=False)
    frame = PauliFrame(engine, commit_ticks=commit_ticks)
    return engine, frame


def commit(frame, window_key, observables, tier=Tier.WEAK, on_committed=None):
    frame.commit_correction(
        window_key=window_key,
        logical_observables=observables,
        request_key=RequestKey(tier, run_sequence=window_key[1]),
        on_committed=on_committed or (lambda: None))


def test_a_streams_frame_is_the_xor_of_its_corrections():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1, 0, 1))
    commit(frame, ("stream", 1), (1, 1, 0))
    commit(frame, ("stream", 2), (0, 1, 1))
    assert frame.frame_for_stream("stream") == (0, 0, 0)
    assert frame.frame_for_stream("other stream") == ()


def test_reading_the_frame_does_not_change_it():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1, 1))
    first_read = frame.frame_for_stream("stream")
    second_read = frame.frame_for_stream("stream")
    assert first_read == second_read == (1, 1)


def test_a_second_correction_for_a_window_is_refused():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1,), tier=Tier.WEAK)
    with pytest.raises(RuntimeError, match="second write"):
        commit(frame, ("stream", 0), (0,), tier=Tier.STRONG)
    assert frame.frame_for_stream("stream") == (1,)


def test_the_write_cost_is_charged_before_the_caller_continues():
    engine, frame = frame_with_commit_ticks(4)
    continued_at = []
    engine.schedule(10, lambda: commit(
        frame, ("stream", 0), (1,),
        on_committed=lambda: continued_at.append(engine.now)))
    engine.run()
    assert continued_at == [14]
    record = frame.snapshot().records[0]
    assert (record.accepted_ticks, record.committed_ticks) == (10, 14)


def test_a_correction_without_observables_makes_the_fold_unknown():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1,))
    commit(frame, ("stream", 1), None)
    assert frame.frame_for_stream("stream") is None


def test_the_settings_refuse_a_free_write_without_a_reason():
    with pytest.raises(ValueError, match="justification"):
        PauliFrameConfig(commit_us=0.0)
    assert PauliFrameConfig(commit_us=0.004).commit_ticks() == 4000
