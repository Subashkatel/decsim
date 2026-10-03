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
import decsim.engine as engine_module
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import tests.experiments.yaml_configs as yaml_configs


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


def test_a_write_cost_refusal_names_its_yaml_path():
    clocks = config.ClockSettings({"fridge": 250.0})
    section = {"clock": "fridge", "write_cycles": -1}

    with pytest.raises(ValueError) as refusal:
        pauli_frame_module.PauliFrameConfig.from_yaml(section, clocks)
    assert str(refusal.value) == (
        "pauli_frame.write_cycles must not be negative: cycles must be "
        "nonnegative"
    )


def test_a_charged_write_without_a_clock_is_refused():
    engine = engine_module.Engine()

    with pytest.raises(ValueError):
        pauli_frame_module.PauliFrame(engine, clock=None, write_cycles=1)


@pytest.mark.parametrize("key", ["clock", "write_cycles"])
def test_a_section_without_a_required_key_is_refused_by_name(key):
    """A sweep that leaves a key out reads a sentence, not a KeyError."""
    clocks = config.ClockSettings({"fridge": 250.0})
    section = {"kind": "logical_register", "clock": "fridge"}
    section["write_cycles"] = 1
    del section[key]

    with pytest.raises(ValueError) as refusal:
        pauli_frame_module.PauliFrameConfig.from_yaml(section, clocks)
    assert str(refusal.value) == (
        f"pauli_frame needs the keys ['{key}']; configs/reference.yaml "
        "holds every key with its unit"
    )


def test_a_key_the_section_does_not_have_is_refused_by_name():
    """A misspelt key would otherwise leave its default silently."""
    clocks = config.ClockSettings({"fridge": 250.0})
    section = {"clock": "fridge", "write_cycles": 1, "write_cycle": 2}

    with pytest.raises(ValueError) as refusal:
        pauli_frame_module.PauliFrameConfig.from_yaml(section, clocks)
    assert str(refusal.value) == (
        "pauli_frame does not know ['write_cycle']; its keys are "
        "['kind', 'clock', 'write_cycles']"
    )


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


def test_a_frame_written_outside_decsim_runs_from_a_yaml(monkeypatch, tmp_path):
    """One FRAMES record and one pauli_frame.kind is the whole edit."""
    monkeypatch.setitem(
        pauli_frame_module.FRAMES, "counting", CountingFrameConfig
    )
    section = dict(yaml_configs.MINIMAL_CONFIG["pauli_frame"])
    section["kind"] = "counting"
    config_path = yaml_configs.write_config(tmp_path, {"pauli_frame": section})
    experiment_config = experiment.load_experiment(config_path)
    point = experiment_config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert isinstance(machine.control.pauli_frame, CountingFrame)
    assert result.terminal_status == "complete"
    assert machine.control.pauli_frame.committed_windows != []


def test_a_frame_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    section = dict(yaml_configs.MINIMAL_CONFIG["pauli_frame"])
    section["kind"] = "not_a_row"
    config_path = yaml_configs.write_config(tmp_path, {"pauli_frame": section})
    with pytest.raises(ValueError, match="pauli_frame.kind 'not_a_row' is not"):
        experiment.load_experiment(config_path)


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
    config_path = yaml_configs.CONFIGS_DIR / "examples/two_tiers.yaml"
    experiment_config = experiment.load_experiment(config_path)
    point = experiment_config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.01,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
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
