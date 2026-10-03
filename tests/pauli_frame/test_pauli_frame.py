"""The Pauli frame behaves like a real frame and charges what it says.

Sources: PECOS pauli_frame.rs (a stream's frame is the XOR of its
corrections; folding is non-destructive); Skoric et al. 2209.08552
lines 102-105 (a window commits its final correction once); Yang et al.
2605.04892 Fig. 1 (one frame update costs one cycle, 4 ns at 250 MHz,
and the loop waits for it). The write
is charged from the frame clock's next edge, gem5's clockEdge
(gem5 src/sim/clocked_object.hh lines 174-186).
"""

import dataclasses
import enum

import pytest

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.settings as machine_settings
import examples.two_tiers as two_tiers
import tests.declared_run as declared_run


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
    """A frame on a one-tick period, so its write costs its ticks.

    Every tick is an edge on that clock, which keeps these tests about
    the fold and the refusals; the edge law has its own test below.
    """
    engine = engine_module.Engine()
    clock = config.Clock(1)
    frame = pauli_frame_module.PauliFrame(
        engine, clock=clock, write_cycles=commit_ticks
    )
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


def _strong_flips(snapshot) -> list:
    """The observables each strong-tier correction in the frame flipped."""
    flips = []
    for record in snapshot.records:
        if record.tier == "strong":
            flips.append(record.logical_observables)
    return flips


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


def test_a_second_correction_for_a_window_is_refused():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1,), tier=Tier.WEAK)
    with pytest.raises(RuntimeError):
        commit(frame, ("stream", 0), (0,), tier=Tier.STRONG)
    assert frame.frame_for_stream("stream") == (1,)


def test_a_correction_without_observables_makes_the_fold_unknown():
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1,))
    commit(frame, ("stream", 1), None)
    assert frame.frame_for_stream("stream") is None


def test_a_stream_whose_corrections_change_width_is_refused_when_read():
    """One stream's observables are one width; two widths cannot be XORed.

    A window's correction is one bit per logical observable of its
    operation, so a change of width means two operations' results
    reached one stream, and folding them would silently drop bits.
    """
    engine, frame = frame_with_commit_ticks(0)
    commit(frame, ("stream", 0), (1, 0))
    commit(frame, ("stream", 1), (0, 1, 0))
    with pytest.raises(RuntimeError):
        frame.frame_for_stream("stream")


def test_a_second_correction_arriving_while_the_first_is_pending_is_refused():
    """The write in flight counts as a write: the refusal does not wait."""
    engine, frame = frame_with_commit_ticks(4)
    continued = []

    def note_continuation():
        continued.append(engine.now)

    commit(frame, ("stream", 0), (1,), on_committed=note_continuation)
    with pytest.raises(RuntimeError):
        commit(frame, ("stream", 0), (0,), tier=Tier.STRONG)
    engine.run()
    assert continued == [4]
    snapshot = frame.snapshot()
    assert snapshot.commit_count == 1
    assert snapshot.pending_write_count == 0


def test_a_charged_write_with_no_clock_still_stops():
    """A run that names no clock cannot land the first window's write."""
    algorithm = decoders.PresetLatencyDecoder.Settings(1.0)
    weak = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    frame = pauli_frame_module.PauliFrameConfig(write_cycles=1, clock=None)
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak, pauli_frame=frame
    )
    machine = machine_module.Machine.build(settings, 0)

    with pytest.raises(AttributeError):
        machine.run()


class CountingFrame(pauli_frame_module.PauliFrame):
    """A frame written outside decsim: it counts what it committed."""

    def __init__(self, engine, *, clock, write_cycles: int) -> None:
        reference = super()
        reference.__init__(engine, clock=clock, write_cycles=write_cycles)
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


class CountingFrameConfig(pauli_frame_module.PauliFrameConfig):
    """The record that builds a CountingFrame."""

    def build(self, engine) -> CountingFrame:
        return CountingFrame(
            engine, clock=self.clock, write_cycles=self.write_cycles
        )


def test_a_frame_written_outside_decsim_runs_from_its_record():
    """One record that builds the frame is the whole edit."""
    base = machine_settings.weak_decoder_baseline(3, 0.001, 1.0)
    frame = base.pauli_frame
    counting = CountingFrameConfig(
        write_cycles=frame.write_cycles, clock=frame.clock
    )
    settings = dataclasses.replace(base, pauli_frame=counting)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert isinstance(machine.control.pauli_frame, CountingFrame)
    assert result.terminal_status == "complete"
    assert machine.control.pauli_frame.committed_windows != []


def test_two_windows_writes_are_charged_in_parallel_and_never_queued():
    """Two windows' writes overlap: the frame is a register, not a queue.

    A write is one XOR into a register, one clock cycle of the frame unit
    (Yang et al. 2605.04892 Fig. 1 measures 4 ns inside a 550 ns loop,
    one cycle at 250 MHz). Two windows are two registers, so the second
    write does not wait behind the first: each caller continues one write
    cost after its own correction arrived, not two after the first.
    """
    engine, frame = frame_with_commit_ticks(4)
    continued_at = []

    def note():
        continued_at.append(engine.now)

    def commit_both():
        commit(frame, ("stream", 0), (1,), on_committed=note)
        commit(frame, ("stream", 1), (1,), on_committed=note)

    engine.schedule(10, commit_both)
    engine.run()

    assert continued_at == [14, 14]


def test_the_frames_fold_is_the_reported_prediction_on_a_switching_run():
    """What the frame holds is what the run reports, escalations included.

    The frame takes one final correction per window, the strong answer
    for an escalated window and never its provisional weak one; the
    report folds the windows' contributions with the strong answer
    replacing the weak (Skoric et al. 2209.08552 lines 444-445: a
    stream's logical correction is the sum of its committed windows'
    effects). Both folds must agree.
    """
    distance_three = two_tiers.points[0]
    workload = machine_settings.memory_workload(3, 0.01, 30)
    settings = dataclasses.replace(distance_three.machine, workload=workload)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    (operation_result,) = result.operation_results
    snapshot = machine.control.pauli_frame.snapshot()
    strong_flips = _strong_flips(snapshot)
    frame = machine.control.pauli_frame.frame_for_stream(
        operation_result.operation_id
    )

    assert (1,) in strong_flips
    assert frame == tuple(operation_result.logical_observables)
