"""Shared fabric for the stabilization tests: every link and stage has a
declared tick value, every decoder a preset latency, so timing assertions
are exact arithmetic, never measured host time."""

from dataclasses import replace

import pytest

from decsim.config import TimingConfig, microseconds_to_ticks
from decsim.decoders.decoders import (SAMPLED_CONFIDENCE_SOURCE,
                                      PresetLatencyDecoder,
                                      SampledConfidenceDecoder,
                                      SwitchingRouter)
from decsim.decoders.weak_strong_switching import Switching, StrongOnly
from decsim.controller.policies import Held
from decsim.links.link_profiles import (logical_reference_profile,
                                        with_controller_to_weak_buffer_path,
                                        with_controller_to_strong_buffer_path)
from decsim.links.settings import ChannelSettings, PathSettings
from decsim.message import Operation
from decsim.decoders.decoder_memory import DecoderMemoryConfig
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import (SlidingTerminalPolicy,
                                              SlidingWindowScheme)


def sliding_scheme():
    return SlidingWindowScheme(
        terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD)

# The declared stage ticks of this suite, in microseconds. QEC cycle is
# 1.0 us. The packet assembly time is declared zero here; its per-round
# charge has its own test in test_controller_io_paths.py.
DECLARED_US = {
    "qpu_to_controller": 2.0, "binary": 3.0, "pack": 0.0, "controller_to_weak_buffer": 4.0, "controller_to_strong_buffer": 7.0,
    "weak_buffer_to_weak_decoder": 5.0, "weak_decoder_to_strong_decoder": 3.0, "strong_buffer_to_strong_decoder": 6.0, "weak": 10.0, "strong": 30.0,
    "weak_decoder_to_frame": 2.0, "decoder_to_decoder": 0.5, "strong_decoder_to_frame": 4.0, "frame": 1.0, "frame_to_controller": 2.0, "controller_to_qpu": 2.0,
}
ROUND_US = 1.0


def _declared_edge(base_edge, latency_us):
    channel = ChannelSettings(base_edge.channel.name, microseconds_to_ticks(latency_us),
                              None, "stabilization declared tick")
    return PathSettings(channel, base_edge.default_payload,
                        base_edge.actual_payload_source)


def declared_profile(*, controller_to_weak_buffer=True, controller_to_strong_buffer=True, strong_buffer_us=None):
    """The reference card with every latency replaced by a declared tick."""
    base = logical_reference_profile()
    profile = replace(
        base,
        qpu_to_controller=_declared_edge(base.qpu_to_controller, DECLARED_US["qpu_to_controller"]),
        weak_buffer_to_weak_decoder=_declared_edge(base.weak_buffer_to_weak_decoder, DECLARED_US["weak_buffer_to_weak_decoder"]),
        weak_decoder_to_strong_decoder=_declared_edge(base.weak_decoder_to_strong_decoder, DECLARED_US["weak_decoder_to_strong_decoder"]),
        strong_buffer_to_strong_decoder=_declared_edge(base.strong_buffer_to_strong_decoder, DECLARED_US["strong_buffer_to_strong_decoder"]),
        weak_decoder_to_frame=_declared_edge(base.weak_decoder_to_frame, DECLARED_US["weak_decoder_to_frame"]),
        decoder_to_decoder=_declared_edge(base.decoder_to_decoder, DECLARED_US["decoder_to_decoder"]),
        strong_decoder_to_frame=_declared_edge(base.strong_decoder_to_frame, DECLARED_US["strong_decoder_to_frame"]),
        frame_to_controller=_declared_edge(base.frame_to_controller, DECLARED_US["frame_to_controller"]),
        controller_to_qpu=_declared_edge(base.controller_to_qpu, DECLARED_US["controller_to_qpu"]),
        # the declared qc tick is wire time only; readout classification
        # prices the controller processing separately
        is_controller_processing_outside_qpu_to_controller=True,
    )
    if controller_to_weak_buffer:
        profile = with_controller_to_weak_buffer_path(
            profile, latency_microseconds=DECLARED_US["controller_to_weak_buffer"],
            aggregate_bits_per_microsecond=None, source="stabilization declared tick")
    if controller_to_strong_buffer:
        profile = with_controller_to_strong_buffer_path(
            profile, latency_microseconds=(DECLARED_US["controller_to_strong_buffer"] if strong_buffer_us is None else strong_buffer_us),
            aggregate_bits_per_microsecond=None, source="stabilization declared tick")
    return profile


def declared_timing(round_us=ROUND_US):
    return TimingConfig(round_us=round_us,
                        measurement_signal_to_classical_bits_us=DECLARED_US["binary"],
                        t_pack_us=DECLARED_US["pack"])


def memory_op(op_id=1, *, name=None, blocked_by=None, predecessors=(),
              requires_result_return=False):
    return Operation(
        id=op_id, name=name or f"mem{op_id}", qubits=(op_id,),
        patches=(op_id,), blocked_by=blocked_by, predecessors=predecessors,
        requires_result_return_to_qpu=requires_result_return)


def weak_only_run(*, rounds=6, ops=None, controller_to_weak_buffer=True, seed=0, io_trace=False,
                  frame=True, make_metrics=None):
    """Weak-only baseline on the declared fabric; d=3 sliding windows."""
    spec = RunSpec(
        ops=(ops if ops is not None else [memory_op(1)]),
        d=3, rounds_policy=FixedRounds(rounds),
        decoder=PresetLatencyDecoder(DECLARED_US["weak"]),
        links=declared_profile(controller_to_weak_buffer=controller_to_weak_buffer, controller_to_strong_buffer=False),
        timing=declared_timing(),
        pauli_frame=(PauliFrameConfig(commit_microseconds=DECLARED_US["frame"])
                     if frame else None),
        make_metrics=make_metrics,
        seed=seed)
    return spec.build(io_trace=io_trace)


def strong_only_run(*, rounds=6, ops=None, seed=0, io_trace=False,
                    record=False):
    """Strong-primary baseline: readiness listens to syndrome buffer 1."""
    spec = RunSpec(
        ops=(ops if ops is not None else [memory_op(1)]),
        d=3, rounds_policy=FixedRounds(rounds),
        decoder=PresetLatencyDecoder(DECLARED_US["strong"]),
        escalation_policy=StrongOnly(),
        links=declared_profile(controller_to_weak_buffer=True, controller_to_strong_buffer=True),
        timing=declared_timing(),
        pauli_frame=PauliFrameConfig(commit_microseconds=DECLARED_US["frame"]),
        record_switching_windows=record,
        seed=seed)
    return spec.build(io_trace=io_trace)


def switching_run(*, rounds=6, escalation_probability, ops=None,
                  run_both_at_once=False, double_window=False,
                  unit_pools=None, seed=0, io_trace=False,
                  probability_for=None, record=False, strong_buffer_us=None,
                  weak_memory_rounds=None, round_us=ROUND_US,
                  make_metrics=None):
    """Weak-primary switching on the declared fabric.

    escalation_probability 0.0 or 1.0 (or a per-job probability_for
    returning 0.0/1.0) keeps the run deterministic: the sampled gap is
    1.0 (keep weak) or 0.0 (escalate), against threshold 0.5.
    """
    weak = SampledConfidenceDecoder(
        PresetLatencyDecoder(DECLARED_US["weak"]),
        escalation_probability, probability_for=probability_for)
    router = SwitchingRouter(weak=weak,
                             strong=PresetLatencyDecoder(DECLARED_US["strong"]))
    policy = Switching(0.5, SAMPLED_CONFIDENCE_SOURCE,
                       run_both_at_once=run_both_at_once,
                       double_window=double_window)
    spec = RunSpec(
        ops=(ops if ops is not None else [memory_op(1)]),
        d=3, rounds_policy=FixedRounds(rounds), scheme=sliding_scheme(),
        router=router, escalation_policy=policy,
        # serial switching requires Held boundaries; double_window rejects
        # them (weak_strong_switching.validate_declared_run)
        boundary_policy=(None if double_window else Held()),
        unit_pools=(unit_pools if unit_pools is not None
                    else {"default": 1, "strong": 1}),
        links=declared_profile(controller_to_weak_buffer=True, controller_to_strong_buffer=True, strong_buffer_us=strong_buffer_us),
        timing=declared_timing(round_us),
        pauli_frame=PauliFrameConfig(commit_microseconds=DECLARED_US["frame"]),
        record_switching_windows=record,
        make_metrics=make_metrics,
        decoder_memory=(None if weak_memory_rounds is None else
                        DecoderMemoryConfig({"default": weak_memory_rounds})),
        seed=seed)
    return spec.build(io_trace=io_trace)


class OccupancyProbe:
    """Metric recording each store's live-round timeline, for hold-lifetime
    assertions (observe runs after every engine event)."""

    name = "hold_occupancy_probe"

    def __init__(self, window_manager):
        self.window_manager = window_manager
        self.buffer0_timeline = []
        self.sb1_timeline = []

    def observe(self, engine):
        live_upstream = self.window_manager.syndrome_buffer.metrics().live_allocations
        if not self.buffer0_timeline or self.buffer0_timeline[-1][1] != live_upstream:
            self.buffer0_timeline.append((engine.now, live_upstream))
        room_store = self.window_manager.syndrome_buffer_1
        if room_store is not None:
            live_room = room_store.store.metrics().live_allocations
            if not self.sb1_timeline or self.sb1_timeline[-1][1] != live_room:
                self.sb1_timeline.append((engine.now, live_room))

    def result(self):
        return {"buffer0": list(self.buffer0_timeline),
                "sb1": list(self.sb1_timeline)}


def occupancy_metrics():
    """make_metrics factory returning ONE probe; read it off CompletedRun."""
    probes = []

    def make(engine, window_manager, decoder_manager, execution_runtime, factory):
        probe = OccupancyProbe(window_manager)
        probes.append(probe)
        return [probe]

    return make, probes


def log_tick(log_lines, needle):
    """The tick (in decsim ticks) of the first log line containing needle."""
    for line in log_lines:
        if needle in line:
            stamp = line.split("]")[0].lstrip("[").strip()
            if not stamp.endswith("us"):
                raise AssertionError(f"unparsable log stamp: {line!r}")
            return microseconds_to_ticks(float(stamp[:-2].strip()))
    raise AssertionError(f"no log line contains {needle!r}")


def log_index(log_lines, needle):
    """The position of the first log line containing needle."""
    for index, line in enumerate(log_lines):
        if needle in line:
            return index
    raise AssertionError(f"no log line contains {needle!r}")


@pytest.fixture
def fabric():
    """The suite's helper namespace, one import point for every test."""
    return {
        "DECLARED_US": DECLARED_US, "ROUND_US": ROUND_US,
        "declared_profile": declared_profile,
        "declared_timing": declared_timing,
        "memory_op": memory_op,
        "weak_only_run": weak_only_run,
        "strong_only_run": strong_only_run,
        "switching_run": switching_run,
        "log_tick": log_tick,
        "log_index": log_index,
        "occupancy_metrics": occupancy_metrics,
    }
