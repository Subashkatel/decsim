"""The Pauli frame behaves like a real frame and charges what it says.

Sources: PECOS pauli_frame.rs (a stream's frame is the XOR of its
corrections; folding is non-destructive); Riesebos, "Pauli Frames for
Quantum Computer Architectures", TU Delft MSc thesis CE-MS-2016,
Sec. 3.2 Table 3.1 (a flush applies a Pauli record's gates once and
resets the record to I); Yang et al. 2605.04892 Fig. 1 (one frame update
costs one cycle, 4 ns at 250 MHz, and the loop waits for it).
"""

import dataclasses
import enum

import pytest

import decsim.front.experiment as experiment
import decsim.machine as machine_module
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import tests.front.yaml_configs as yaml_configs
from decsim.engine import Engine
from decsim.pauli_frame.pauli_frame import PauliFrame, PauliFrameConfig


class Tier(enum.Enum):
    WEAK = "weak"
    STRONG = "strong"


@dataclasses.dataclass(frozen=True)
class RequestKey:
    tier: Tier
    run_sequence: int


def do_nothing():
    return None


def frame_with_commit_ticks(commit_ticks):
    engine = Engine()
    frame = PauliFrame(engine, commit_ticks=commit_ticks)
    return engine, frame


def commit(frame, window_key, observables, tier=Tier.WEAK, on_committed=None):
    request_key = RequestKey(tier, run_sequence=window_key[1])
    if on_committed is None:
        on_committed = do_nothing
    frame.commit_correction(
        window_key=window_key,
        logical_observables=observables,
        request_key=request_key,
        on_committed=on_committed,
    )


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

    def note_continuation():
        continued_at.append(engine.now)

    def commit_at_tick_ten():
        commit(frame, ("stream", 0), (1,), on_committed=note_continuation)

    engine.schedule(10, commit_at_tick_ten)
    engine.run()
    assert continued_at == [14]
    snapshot = frame.snapshot()
    record = snapshot.records[0]
    assert (record.accepted_ticks, record.committed_ticks) == (10, 14)


def test_a_correction_without_observables_makes_the_fold_unknown():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1,))
    commit(frame, ("stream", 1), None)
    assert frame.frame_for_stream("stream") is None


def test_the_settings_refuse_a_free_write_without_a_reason():
    with pytest.raises(ValueError, match="justification"):
        PauliFrameConfig(commit_microseconds=0.0)
    settings = PauliFrameConfig(commit_microseconds=0.004)
    assert settings.commit_ticks() == 4000


def test_a_stream_whose_corrections_change_width_is_refused_when_read():
    """One stream's observables are one width; two widths cannot be XORed.

    A window's correction is one bit per logical observable of its
    operation, so a change of width means two operations' results
    reached one stream, and folding them would silently drop bits.
    """
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1, 0))
    commit(frame, ("stream", 1), (0, 1, 0))
    with pytest.raises(RuntimeError, match="changed its number of observables"):
        frame.frame_for_stream("stream")


def test_a_second_correction_arriving_while_the_first_is_pending_is_refused():
    """The write in flight counts as a write: the refusal does not wait."""
    engine, frame = frame_with_commit_ticks(4)
    continued = []

    def note_continuation():
        continued.append(engine.now)

    commit(frame, ("stream", 0), (1,), on_committed=note_continuation)
    with pytest.raises(RuntimeError, match="second write"):
        commit(frame, ("stream", 0), (0,), tier=Tier.STRONG)
    engine.run()
    assert continued == [4]
    snapshot = frame.snapshot()
    assert snapshot.commit_count == 1
    assert snapshot.pending_write_count == 0


def test_the_frame_keeps_its_own_copy_of_the_observables_it_was_given():
    """A caller may reuse its buffer; the committed correction is fixed."""
    engine, frame = frame_with_commit_ticks(0)
    callers_buffer = [1, 0, 1]
    commit(frame, ("stream", 0), callers_buffer)
    callers_buffer[0] = 0
    assert frame.frame_for_stream("stream") == (1, 0, 1)


def test_a_snapshot_does_not_change_when_the_frame_does():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1, 0))
    before = frame.snapshot()
    commit(frame, ("stream", 1), (1, 1))
    after = frame.snapshot()
    assert before.commit_count == 1
    assert after.commit_count == 2


def test_a_write_cost_that_is_not_a_duration_is_refused():
    with pytest.raises(ValueError, match="finite and not negative"):
        PauliFrameConfig(commit_microseconds=-1.0)


def test_a_write_cost_that_rounds_to_no_ticks_is_refused():
    with pytest.raises(ValueError, match="rounds to zero ticks"):
        PauliFrameConfig(commit_microseconds=1e-12)


def test_a_justification_beside_a_priced_write_is_refused_as_stale():
    with pytest.raises(ValueError, match="needs a free write"):
        PauliFrameConfig(
            commit_microseconds=1.0,
            zero_commit_cost_justification="an idealized register write",
        )


def test_a_free_write_with_a_reason_is_accepted_and_charges_nothing():
    settings = PauliFrameConfig(
        commit_microseconds=0.0,
        zero_commit_cost_justification="an idealized register write",
    )
    assert settings.commit_ticks() == 0


class CountingFrame(pauli_frame_module.PauliFrame):
    """A frame row written outside decsim: it counts what it committed.

    Its constructor is the port's, the engine and the write cost in
    ticks, which is what the root gives every row of FRAMES.
    """

    def __init__(self, engine, *, commit_ticks: int) -> None:
        reference = super()
        reference.__init__(engine, commit_ticks=commit_ticks)
        self.committed_windows = []

    def commit_correction(
        self, *, window_key, logical_observables, request_key, on_committed
    ) -> None:
        """Remember the window, then commit as the shipped row does."""
        self.committed_windows.append(window_key)
        reference = super()
        reference.commit_correction(
            window_key=window_key,
            logical_observables=logical_observables,
            request_key=request_key,
            on_committed=on_committed,
        )


def test_a_frame_row_written_outside_decsim_runs_from_a_yaml(
    monkeypatch, tmp_path
):
    """One FRAMES row and one pauli_frame.kind is the whole edit."""
    monkeypatch.setitem(pauli_frame_module.FRAMES, "counting", CountingFrame)
    section = dict(yaml_configs.MINIMAL_CONFIG["pauli_frame"])
    section["kind"] = "counting"
    config_path = yaml_configs.write_config(tmp_path, {"pauli_frame": section})
    experiment_config = experiment.load_experiment(config_path)
    settings = experiment_config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert isinstance(machine.pauli_frame, CountingFrame)
    assert result.terminal_status == "complete"
    assert machine.pauli_frame.committed_windows != []


def test_a_frame_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    section = dict(yaml_configs.MINIMAL_CONFIG["pauli_frame"])
    section["kind"] = "not_a_row"
    config_path = yaml_configs.write_config(tmp_path, {"pauli_frame": section})
    with pytest.raises(ValueError, match="pauli_frame.kind 'not_a_row' is not"):
        experiment.load_experiment(config_path)
