"""A d=3 flip stream with a protected region, read out at a chosen round.

The stream tests build the same machine: the memory stream owner, a
protect operation at round 1 and a readout scheduled at the last round,
over Stim's d=3 memory fragments (tests/qpu/memory_programs.py), with a
Pauli frame so every window commits a correction.
"""

import dataclasses

import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.memory_stream as memory_stream
import decsim.frontends.settings as workload_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.topology.graph as graph
import decsim.windows.schemes.parallel as parallel_scheme
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the stream owner's operation id, the one its windows are keyed by
STREAM_ID = 100
# a window of three rounds commits every 3 us; 4 us of decoding a window
# overloads the tier, so its reaction grows and misses the deadline
DEADLINE_MICROSECONDS = 3.0


@dataclasses.dataclass(frozen=True)
class StreamRun:
    """One stream's shape: its length, decode time, region, escalation."""

    rounds: int
    weak_decode_microseconds: float
    first_recorded_round: int
    escalates: bool


def stream_graph(run: StreamRun, folds: bool) -> graph.Machine:
    """The stream machine's graph; folds lets each window go."""
    program = memory_programs.memory_program(3, "Z", 0.01)
    workload = stream_workload(program, run.rounds)
    source = flip_stream_device.FlipStreamDevice.Settings()
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source, round_period_microseconds=1.0
    )
    weak = matching_pool(run.weak_decode_microseconds)
    observation = observe_settings.ObservationSettings(
        first_recorded_round=run.first_recorded_round,
        reaction_deadline_microseconds=DEADLINE_MICROSECONDS,
        backlog_trace=True,
        record_switching_windows=run.escalates,
        folds_windows=folds,
    )
    frame = pauli_frame.PauliFrameConfig()
    machine = machines.base(workload, qpu, frame, observation=observation)
    with_weak = machines.with_pool(machine, weak)
    if not run.escalates:
        return with_weak
    return switching(with_weak)


def on_parallel_windows(machine: graph.Machine) -> graph.Machine:
    """The machine on Skoric's parallel blocks (schemes/parallel.py)."""
    windows = machine.held("windows")
    scheme = parallel_scheme.ParallelWindowScheme.Settings()
    parallel_windows = dataclasses.replace(windows, scheme=scheme)
    return machine.with_held("windows", parallel_windows)


def stream_workload(program, rounds: int) -> workload_settings.WorkloadSettings:
    """The stream owner, protected from round 1, read out at the round."""
    memory = program_records.Operation(STREAM_ID, "memory", (0,), patches=(0,))
    return memory_stream.streams_of([(memory, program)], rounds)


def matching_pool(latency: float) -> decoder_settings.DecoderPoolSettings:
    """A PyMatching pool whose every decode takes the preset latency."""
    clock = config.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=latency
    )
    return decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
    )


def switching(machine: graph.Machine) -> graph.Machine:
    """A strong tier redoing the windows whose gap is under 12 dB."""
    strong_window = strong_window_shapes.RedoWindow.Settings()
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=12.0
    )
    switching_settings = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    strong = matching_pool(6.0)
    return machines.with_strong_tier(machine, strong, switching_settings)
