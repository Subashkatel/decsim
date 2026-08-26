"""Differential trace lock for the core rewrite.

One RunSpec per feature the unit tests do not close over: strong escalation
(serial, parallel, bulk, double window, held boundaries), protected and
dynamic streams, finite syndrome buffer and decoder memory, multi-fragment
QLX rounds, every link path, the schemes, the idle policies, the observers,
Stim data through PyMatching. `record` runs each scenario and writes the
ordered engine trace plus the frozen end state as JSON; `check` reruns and
diffs against the recording, printing the first difference. Wall time is
never recorded, so a recording is stable across hosts.

    python -m experiments.refactor_lock record [name ...]
    python -m experiments.refactor_lock check  [name ...]
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "refactor_lock"


# ---- canonical JSON --------------------------------------------------------

def canonical(value, depth=0):
    """A JSON-ready view of a runtime value: dataclasses become dicts, enums
    their names, tuples lists, dict keys strings; anything else its repr."""
    if depth > 12:
        return "<depth>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, bytes):
        return value.hex()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: canonical(getattr(value, f.name), depth + 1)
                for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(canonical(k, depth + 1)): canonical(v, depth + 1)
                for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [canonical(v, depth + 1) for v in value]
        return sorted(items, key=json.dumps) if isinstance(value, (set, frozenset)) else items
    if hasattr(value, "tolist"):
        return canonical(value.tolist(), depth + 1)
    return repr(value)


WINDOW_FIELDS = ("op_id", "k", "commit_lo", "commit_hi", "buffer_hi", "n_rounds",
                 "buffer_lo", "closed_temporal_boundaries", "deps", "dependents",
                 "committed", "queued", "t_first_round", "t_data_complete",
                 "t_queued", "t_dispatch", "t_done")


def capture(completed) -> dict:
    """Everything a refactor must keep: the ordered log, the scientific result,
    the window table and the frozen owner snapshots."""
    wm = completed.window_manager
    state = {
        "log": list(completed.engine.log_lines),
        "result": canonical(completed.result),
        "windows": [{name: canonical(getattr(w, name)) for name in WINDOW_FIELDS}
                    for _, w in sorted(wm.windows.items())],
        "op_results": canonical(getattr(wm, "op_results", None)),
        "final_ticks": completed.engine.now,
    }
    for owner in ("pauli_frame", "syndrome_buffer"):
        obj = getattr(completed, owner, None)
        if obj is not None and hasattr(obj, "snapshot"):
            state[owner] = canonical(obj.snapshot())
    runtime = completed.execution_runtime
    state["runtime"] = {name: canonical(getattr(runtime, name))
                        for name in ("op_start_time", "body_done_time",
                                     "decode_release_time")
                        if hasattr(runtime, name)}
    dm = completed.decoder_manager
    state["decoder_manager"] = {name: canonical(getattr(dm, name))
                                for name in ("strong_needed", "strong_cancelled",
                                             "jobs_done", "decodes_started")
                                if hasattr(dm, name)}
    ctl = completed.controller
    state["controller"] = {name: canonical(getattr(ctl, name))
                          for name in ("idle_rounds_emitted",)
                          if hasattr(ctl, name)}
    from decsim.observe import run_views as views
    state["views"] = {
        "utilization": canonical(views.utilization_view(dm)),
        "backlog": canonical(views.backlog_view(wm, dm)),
        "window_latency": canonical(views.window_latency_view(wm)),
        "reaction": canonical(views.reaction_view(runtime)),
        "strong_work": canonical(views.strong_work_view(wm, dm)),
        "decoder_memory": canonical(views.decoder_memory_view(dm)),
        "switching_records": (canonical(views.switching_records_view(wm, dm))
                              if getattr(wm, "_selected_request_keys", None) is not None
                              else None),
    }
    return state


# ---- scenarios -------------------------------------------------------------

def _memory_op(op_id=0, patch=0):
    from decsim.message import Operation
    return Operation(op_id, "memory", (patch,), clifford=True, patches=(patch,))


def _links(**path_latency_ticks):
    """Isolated fixed-latency link cards, zero by default (the legacy
    conftest helper)."""
    from dataclasses import fields, replace
    from decsim.links.link_profiles import logical_reference_profile
    reference = logical_reference_profile()
    edges = {f.name: getattr(reference, f.name) for f in fields(reference)}
    for name, edge in edges.items():
        if hasattr(edge, "channel"):
            edges[name] = replace(edge, channel=replace(
                edge.channel, capacity=None,
                propagation_latency_ticks=path_latency_ticks.get(name, 0)))
    return replace(reference, **edges)


def _sliding():
    from decsim.windows.windowing_schemes import SlidingTerminalPolicy, SlidingWindowScheme
    return SlidingWindowScheme(
        terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD)


def _switching_pair(low_confidence_probability, weak_tau=0.1, strong_tau=10.0):
    from decsim.decoders.decoders import PerRoundDecoder, SampledConfidenceDecoder
    weak = SampledConfidenceDecoder(PerRoundDecoder(weak_tau),
                                    low_confidence_probability)
    strong = PerRoundDecoder(strong_tau)
    return weak, strong


def _switching_spec(*, rounds=27, probability=0.3, seed=1, **switching_kwargs):
    from decsim.decoders.decoders import SAMPLED_CONFIDENCE_SOURCE, SwitchingRouter
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.decoders.weak_strong_switching import Switching
    weak, strong = _switching_pair(probability)
    extra = {}
    for key in ("boundary_policy", "links", "make_metrics", "record_switching_windows"):
        if key in switching_kwargs:
            extra[key] = switching_kwargs.pop(key)
    return RunSpec(
        ops=[_memory_op()], num_units=1, d=3, rounds_policy=FixedRounds(rounds),
        round_us=1.0, scheme=_sliding(),
        escalation_policy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE,
                           confidence_threshold=0.5, **switching_kwargs),
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1}, seed=seed, **extra)


def baseline_memory():
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(21),
                   round_us=1.0, decoder=PresetLatencyDecoder(2.0), num_units=1,
                   scheme=_sliding(), seed=3)


def baseline_reference_links():
    """Two patches, two units, the reference link cards, so every card is
    priced and the CWB arbitration sees two sources."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.links.link_profiles import logical_reference_profile
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    return RunSpec(ops=[_memory_op(0, 0), _memory_op(1, 1)], d=3,
                   rounds_policy=FixedRounds(15), round_us=1.0,
                   decoder=PresetLatencyDecoder(1.5), num_units=2,
                   scheme=_sliding(), links=logical_reference_profile(), seed=5)


def bandwidth_limited_links():
    """The bandwidth-limited profile: link FIFO capacity and reserve paths,
    two patches contending on CWB."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.links.link_profiles import bandwidth_limited_profile
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    return RunSpec(ops=[_memory_op(0, 0), _memory_op(1, 1)], d=5,
                   rounds_policy=FixedRounds(20), round_us=1.0,
                   decoder=PresetLatencyDecoder(2.0), num_units=2, scheme=_sliding(),
                   links=bandwidth_limited_profile(capacity_scale=0.25), seed=59)


def stim_pymatching():
    """Real Stim data through PyMatching with the decoder engine stages,
    finite decoder memory and the Pauli frame (the baseline closed loop)."""
    import stim
    from decsim.qpu.stim_device import StimDevice
    from decsim.config import TimingConfig
    from decsim.decoders.decoder_engine import DecoderEngine, DecoderStage, DecoderTiming
    from decsim.decoders.decoder_memory import DecoderMemoryConfig
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.links.link_profiles import logical_reference_profile
    from decsim.message import Operation
    from decsim.decoders.mwpm.decoder import PyMatchingDecoder
    from decsim.pauli_frame.pauli_frame import PauliFrameConfig
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    p = 0.003
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=15, distance=3,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                          circuit=circuit)
    engine = DecoderEngine(
        PyMatchingDecoder(PresetLatencyDecoder(1.0)),
        DecoderTiming(before=(DecoderStage("fetch", cycles_per_round=4),),
                      after=(DecoderStage("release", cycles_per_job=8),),
                      frequency_mhz=250.0))
    return RunSpec(ops=[operation], d=3, rounds_policy=FixedRounds(15),
                   device=StimDevice(), decoder=engine, num_units=1,
                   timing=TimingConfig(round_us=1.0),
                   links=logical_reference_profile(),
                   decoder_memory=DecoderMemoryConfig({"default": 12}),
                   pauli_frame=PauliFrameConfig(commit_us=0.1), seed=11)


def switching_serial():
    return _switching_spec()


def switching_parallel():
    return _switching_spec(run_both_at_once=True)


def switching_bulk_strong():
    return _switching_spec(bulk_strong=True, probability=0.5)


def switching_double_window():
    return _switching_spec(double_window=True, probability=0.5, rounds=33)


def switching_held_boundaries():
    from decsim.controller.policies import Held
    return _switching_spec(boundary_policy=Held(), probability=0.5, links=_links(dd=0, wsd=0))


def switching_all_escalate():
    return _switching_spec(probability=1.0, rounds=21)


def switching_with_observers():
    from decsim.observe.metrics import (DecodeBacklog, DecoderUtilization,
                                DecoderMemoryOccupancy, ReadyQueueStats,
                                StrongDecoderBacklog, WindowLatencyBreakdown)
    def metrics(engine, wm, dm, runtime, factory):
        return [DecodeBacklog(wm, dm), DecoderUtilization(dm),
                DecoderMemoryOccupancy(dm), ReadyQueueStats(dm),
                StrongDecoderBacklog(wm, dm), WindowLatencyBreakdown(wm)]
    return _switching_spec(probability=0.5, make_metrics=metrics,
                           record_switching_windows=True)


def _live_stream_pair():
    from decsim.message import Operation
    stream = Operation(0, "live-stream", (0,), clifford=True, patches=(0,))
    first = Operation(1, "A:T(q0)", (0,), clifford=False, consumes_magic_state=False,
                      patches=(0,), stream_id=stream.id)
    second = Operation(2, "B:T(q0)", (0,), clifford=False, consumes_magic_state=False,
                       patches=(0,), predecessors=(first.id,),
                       decoder_boundary_predecessors=(first.id,),
                       stream_id=stream.id, blocked_by=first.id)
    return stream, [first, second]


def _live_stream_spec(idle_policy, mode="trailing_buffer", decode_us=2.0):
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.qpu.syndrome_devices import TimingOnlyDevice
    from decsim.qpu.round_policies import PerOpRounds
    from decsim.run_spec import RunSpec
    from decsim.windows.windowing_schemes import SlidingWindowScheme
    stream, operations = _live_stream_pair()
    rounds = {operations[0].id: 2, operations[1].id: 2}
    return RunSpec(ops=operations, dynamic_streams=[stream], idle_policy=idle_policy,
                   device=TimingOnlyDevice(),
                   code=SurfaceCodeModel(d=3, commit_rounds_override=2,
                                         buffer_rounds_override=1,
                                         window_floor_justification="lock scenario: a one-round buffer exercises the stream handoff, not accuracy"),
                   rounds_policy=PerOpRounds(rounds), scheme=SlidingWindowScheme(),
                   decoder=PresetLatencyDecoder(decode_us), num_units=1, round_us=1.0,
                   feedback_boundary_mode=mode, links=_links(), seed=7)


def live_stream_extend():
    from decsim.controller.policies import ExtendStream
    return _live_stream_spec(ExtendStream())


def live_stream_measurement_closed():
    from decsim.controller.policies import ExtendStream
    return _live_stream_spec(ExtendStream(), mode="measurement_closed", decode_us=0.0)


def _protected_chain(feedback: bool):
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation, ProtectedRegion
    from decsim.qpu.round_policies import PerOpRounds
    from decsim.run_spec import RunSpec
    stream = Operation(100, "protected stream", (0,), patches=(0,))
    start = Operation(1, "allocate", (0,), patches=(0,), emits_detector_data=False)
    source = Operation(2, "compute", (0,), patches=(0,), predecessors=(start.id,),
                       emits_detector_data=False)
    end = Operation(3, "conditional", (0,), patches=(0,), predecessors=(source.id,),
                    blocked_by=source.id if feedback else None,
                    emits_detector_data=False)
    source_rounds = 0 if feedback else 2
    rounds = {stream.id: 40, start.id: 1, source.id: source_rounds, end.id: 1}
    ops, streams = [start, source, end], [stream]
    regions = [ProtectedRegion(0, stream.id, start.id, end.id)]
    if not feedback:
        streams.append(Operation(101, "reopened stream", (0,), patches=(0,)))
        ops += [Operation(4, "reopen", (0,), predecessors=(end.id,), patches=(0,),
                          emits_detector_data=False),
                Operation(5, "finish", (0,), predecessors=(4,), patches=(0,),
                          emits_detector_data=False)]
        regions.append(ProtectedRegion(0, 101, 4, 5))
        rounds.update({101: 40, 4: 1, 5: 1})
    return RunSpec(ops=ops, dynamic_streams=streams, protected_regions=tuple(regions),
                   code=SurfaceCodeModel(d=3, commit_rounds_override=2,
                                         buffer_rounds_override=1,
                                         window_floor_justification="lock scenario: a one-round buffer exercises the stream handoff, not accuracy"),
                   rounds_policy=PerOpRounds(rounds),
                   decoder=PresetLatencyDecoder(5.0 if feedback else 0),
                   round_us=1.0, seed=9)


def protected_chain_reopen():
    return _protected_chain(feedback=False)


def protected_chain_feedback():
    return _protected_chain(feedback=True)


def _feedback_chain_spec(mode, idle_policy=None):
    from decsim.qpu.code_geometry import SurfaceCodeModel
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.frontends.circuit_frontend import CircuitFrontend
    from decsim.message import Operation
    from decsim.observe.metrics import BacklogTrajectory, ConditionalReactionTime
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec
    ops = CircuitFrontend([
        Operation(0, "T0", (0,), clifford=False, consumes_magic_state=False),
        Operation(1, "T1", (0,), clifford=False, consumes_magic_state=False,
                  blocked_by=0),
    ]).build()
    return RunSpec(ops=ops, num_units=1, rounds_policy=FixedRounds(3), round_us=1.0,
                   code=SurfaceCodeModel(d=3), scheme=_sliding(),
                   decoder=PresetLatencyDecoder(2.0), links=_links(),
                   make_metrics=lambda e, wm, dm, runtime, f: [
                       ConditionalReactionTime(runtime), BacklogTrajectory(runtime)],
                   feedback_boundary_mode=mode, idle_policy=idle_policy, seed=13)


def feedback_chain_trailing_buffer():
    return _feedback_chain_spec("trailing_buffer")


def feedback_chain_separate_decode_jobs():
    from decsim.controller.policies import SeparateDecodeJobs
    return _feedback_chain_spec("trailing_buffer", SeparateDecodeJobs())


def feedback_chain_extend_stream_fallback():
    """ExtendStream with no live stream falls back to memory rounds."""
    from decsim.controller.policies import ExtendStream
    return _feedback_chain_spec("trailing_buffer", ExtendStream())


def feedback_chain_measurement_closed():
    return _feedback_chain_spec("measurement_closed")


def parallel_ab_scheme():
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.windows.windowing_schemes import ParallelWindowScheme
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(30), round_us=1.0,
                   decoder=PresetLatencyDecoder(4.0), num_units=3,
                   scheme=ParallelWindowScheme(), links=_links(dd=100), seed=17)


def sandwich_scheme():
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.windows.windowing_schemes import TanSandwichScheme
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(30), round_us=1.0,
                   decoder=PresetLatencyDecoder(4.0), num_units=3,
                   scheme=TanSandwichScheme(), links=_links(dd=100), seed=19)


def naive_online_scheme():
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.windows.windowing_schemes import NaiveOnlineScheme
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(20), round_us=1.0,
                   decoder=PresetLatencyDecoder(1.0), num_units=1,
                   scheme=NaiveOnlineScheme(), seed=23)


def _finite_buffer_spec(policy=None, slots=8):
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.syndrome_buffer.syndrome_buffer import SyndromeBufferingConfig
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(24), round_us=1.0,
                   decoder=PresetLatencyDecoder(6.0), num_units=1, scheme=_sliding(),
                   syndrome_buffering=SyndromeBufferingConfig(upstream_packet_slots=slots),
                   syndrome_packing_policy=policy, seed=29)


def finite_syndrome_buffer_fail_stop():
    """A slow decoder against a small upstream buffer: Buffer 0 fills and the
    run fails stop; the recorded state is the failure and its tick."""
    return _finite_buffer_spec()


def finite_syndrome_buffer_drop_round():
    from decsim.controller.syndrome_packing import PackingOverflowPolicy, SyndromePackingPolicy
    return _finite_buffer_spec(SyndromePackingPolicy(overflow=PackingOverflowPolicy.DROP_ROUND))


def finite_syndrome_buffer_holds():
    """Buffer large enough that holds and releases happen without overflow."""
    return _finite_buffer_spec(slots=30)


def finite_decoder_memory():
    from decsim.decoders.decoder_memory import DecoderMemoryConfig
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(24), round_us=1.0,
                   decoder=PresetLatencyDecoder(3.0), num_units=2, scheme=_sliding(),
                   decoder_memory=DecoderMemoryConfig({"default": 9}), seed=31)


def dependency_dag_three_ops():
    """Three operations on two patches with a decode dependency and a
    predecessor edge: the execution runtime's readiness and completion paths."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import PerOpRounds
    a = Operation(0, "A", (0,), clifford=True, patches=(0,))
    b = Operation(1, "B", (1,), clifford=True, patches=(1,))
    c = Operation(2, "C:T", (0, 1), clifford=False, consumes_magic_state=False,
                  patches=(0, 1), predecessors=(0, 1), blocked_by=0)
    return RunSpec(ops=[a, b, c], d=3,
                   rounds_policy=PerOpRounds({0: 9, 1: 12, 2: 6}), round_us=1.0,
                   decoder=PresetLatencyDecoder(2.5), num_units=2, scheme=_sliding(),
                   seed=37)


def scheduler_and_pools():
    """Two decoders by code router, two pools, priority order under load."""
    from decsim.decoders.decoders import CodeRouter, PresetLatencyDecoder
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.decoders.schedulers import FifoScheduler
    return RunSpec(ops=[_memory_op(0, 0), _memory_op(1, 1), _memory_op(2, 2)], d=3,
                   rounds_policy=FixedRounds(18), round_us=1.0,
                   router=CodeRouter(PresetLatencyDecoder(5.0)),
                   scheduler=FifoScheduler(), num_units=1, scheme=_sliding(), seed=41)


def qlx_multi_fragment():
    """The frozen QLX mem_surface program on its own Stim circuit: the qlx
    frontend, native detector routing, two-fragment rounds and t_pack."""
    import json as _json
    from dataclasses import replace
    import stim
    from decsim.qpu.stim_device import StimDevice
    from decsim.config import TimingConfig
    from decsim.decoders.decoders import PerRoundDecoder
    from decsim.frontends.qlx_frontend import qlx_frontend
    from decsim.message import OpKind
    from decsim.qpu.round_policies import GateRounds
    from decsim.run_spec import RunSpec
    data = Path(__file__).resolve().parent.parent / "tests" / "data" / "qlx"
    load = lambda name: _json.loads((data / name).read_text(encoding="utf-8"))
    circuit = stim.Circuit.from_file(data / "mem_surface.stim")
    program = qlx_frontend(load("schedule_mem_surface.json"), physical_circuit=circuit,
                           detector_metadata=load("mem_surface_decoder_params.json"),
                           decode_operation_id=100)
    program.operations = [
        replace(op, kind=OpKind.MEASURE) if op.name.startswith("measure_syndrome[") else op
        for op in program.operations]
    program.decoder_operations = (replace(program.decoder_operations[0], kind=OpKind.MEMORY),)
    device = StimDevice(detector_rounds=program.detector_rounds_by_stream,
                        terminal_detector_ids=program.terminal_detector_ids_by_stream,
                        measurement_rounds=program.measurement_rounds_by_stream)
    return RunSpec(frontend=program, decode_ops=program.decoder_operations, device=device,
                   decoder=PerRoundDecoder(tau_us=0.5), rounds_policy=GateRounds(merge_steps=2),
                   d=8, timing=TimingConfig(round_us=1.0, t_pack_us=0.1), seed=28)


def two_fragment_stream():
    """A stream whose final round arrives as two fragments (an ordinary
    measure segment and a terminal data readout, as the QLX frontend emits
    them): fragment reassembly, packing, t_pack, finalize_stream_round."""
    import stim
    from decsim.qpu.stim_device import StimDevice
    from decsim.config import TimingConfig
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.detector_error_model.fault_model_contracts import GRAPHLIKE_FAULT_MODEL_REQUIRED
    from decsim.detector_error_model.window_model_builders import build_window_error_models
    from decsim.message import Operation
    from decsim.qpu.round_policies import PerOpRounds
    from decsim.run_spec import RunSpec
    rounds, distance, stream_id = 3, 3, 7
    circuit = stim.Circuit.generated("repetition_code:memory", rounds=rounds,
                                     distance=distance, before_round_data_depolarization=0.05)
    inferred = build_window_error_models(
        circuit, [(1, rounds, rounds)], round_count=rounds,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED, fault_exclusion_ranges=())[0]
    detector_rounds = {d: inferred.defect_positions[d][0] for d in inferred.detector_ids}
    final_rows = sorted(d for d, r in detector_rounds.items() if r == rounds)
    terminal_rows = tuple(final_rows[len(final_rows) // 2:])
    device = StimDevice(detector_rounds={stream_id: detector_rounds},
                        terminal_detector_ids={stream_id: terminal_rows})
    ops = []
    for offset in range(rounds):
        last = offset == rounds - 1
        ops.append(Operation(offset + 1, f"segment {offset + 1}", (0,), patches=(0,),
                             circuit=circuit, stream_id=stream_id, stream_offset=offset,
                             predecessors=(offset,) if offset else (),
                             syndrome_fragment_index=0 if last else None,
                             syndrome_fragment_count=2 if last else None))
    ops.append(Operation(rounds + 1, "mz", (0,), patches=(0,), circuit=circuit,
                         stream_id=stream_id, stream_offset=rounds - 1,
                         predecessors=(rounds,), finalizes_stream_round=True,
                         syndrome_fragment_index=1, syndrome_fragment_count=2))
    owner = Operation(stream_id, "stream", (0,), patches=(0,), circuit=circuit)
    per_op = {op.id: 1 for op in ops}
    per_op[rounds + 1] = 0
    per_op[stream_id] = rounds
    return RunSpec(ops=ops, decode_ops=[owner], device=device, d=distance,
                   decoder=PresetLatencyDecoder(1.0), rounds_policy=PerOpRounds(per_op),
                   timing=TimingConfig(round_us=1.0, t_pack_us=0.1), seed=43)


class _WeakBoundaryDecoder:
    """Deterministic weak decoder: the windows in `uncertain` report low
    confidence and a boundary defect; a later window's logical bit depends on
    the defect it received (the legacy speculative-recovery fixture)."""

    def __init__(self, uncertain=(1,), ticks=1):
        from decsim.detector_error_model.fault_model_contracts import NO_FAULT_MODEL_REQUIRED
        self.fault_model_requirement = NO_FAULT_MODEL_REQUIRED
        self.uncertain = set(uncertain)
        self.ticks = ticks

    def latency(self, job):
        return self.ticks

    def decode(self, job):
        from decsim.decoders.decoders import SAMPLED_CONFIDENCE_SOURCE
        from decsim.message import DecodeResult, SoftOutput
        has_defect = any(payload.bits is not None and any(int(b) for b in payload.bits)
                         for payload in job.payloads)
        return DecodeResult(
            job.op_id, job.window_id,
            logical_observables=(int(job.window_id >= 2 and has_defect),),
            soft_output=SoftOutput(gap=0.0 if job.window_id in self.uncertain else 1.0,
                                   source=SAMPLED_CONFIDENCE_SOURCE),
            boundary_defects=({job.window.commit_hi + 1: [1]}
                              if job.window_id in self.uncertain else None))


class _CorrectingStrongDecoder:
    """Strong decoder that retracts the weak boundary defect after the
    downstream weak chain has run, forcing an Eager replay."""

    def __init__(self, ticks=10_000_000):
        from decsim.detector_error_model.fault_model_contracts import NO_FAULT_MODEL_REQUIRED
        self.fault_model_requirement = NO_FAULT_MODEL_REQUIRED
        self.ticks = ticks

    def latency(self, job):
        return self.ticks

    def decode(self, job):
        from decsim.message import DecodeResult
        return DecodeResult(job.op_id, job.window_id, logical_observables=(0,),
                            boundary_defects={job.window.commit_hi + 1: [0]})


def _recovery_spec(boundary_policy, *, run_both_at_once=False, rounds=15, seed=47):
    from decsim.decoders.decoders import SAMPLED_CONFIDENCE_SOURCE, SwitchingRouter
    from decsim.run_spec import RunSpec
    from decsim.qpu.round_policies import FixedRounds
    from decsim.decoders.weak_strong_switching import Switching
    return RunSpec(ops=[_memory_op()], d=3, rounds_policy=FixedRounds(rounds),
                   scheme=_sliding(),
                   escalation_policy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE,
                                      confidence_threshold=0.5,
                                      run_both_at_once=run_both_at_once),
                   boundary_policy=boundary_policy,
                   router=SwitchingRouter(_WeakBoundaryDecoder(), _CorrectingStrongDecoder()),
                   unit_pools={"default": 1, "strong": 1}, links=_links(dd=0, wsd=0),
                   record_switching_windows=True, seed=seed)


def recovery_eager_replay():
    """Eager boundaries: the strong correction replays the transitive weak cone."""
    from decsim.controller.policies import Eager
    return _recovery_spec(Eager())


def recovery_eager_replay_parallel():
    from decsim.controller.policies import Eager
    return _recovery_spec(Eager(), run_both_at_once=True)


def recovery_held_boundaries():
    """Held boundaries: nothing ships until the result is final, no replay."""
    from decsim.controller.policies import Held
    return _recovery_spec(Held())


def magic_state_factory():
    """T operations that consume magic states from a distillation factory
    whose correction decodes share the decoder manager."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.qpu.magic_state_factories import DistillationFactory
    from decsim.message import Operation
    from decsim.qpu.round_policies import PerOpRounds
    from decsim.run_spec import RunSpec
    ops = [Operation(0, "T0", (0,), clifford=False, consumes_magic_state=True, patches=(0,)),
           Operation(1, "T1", (0,), clifford=False, consumes_magic_state=True, patches=(0,),
                     predecessors=(0,), blocked_by=0)]
    return RunSpec(ops=ops, d=3, rounds_policy=PerOpRounds({0: 6, 1: 6}), round_us=1.0,
                   decoder=PresetLatencyDecoder(1.0), num_units=2, scheme=_sliding(),
                   make_factory=lambda engine, dm: DistillationFactory(
                       engine, num_units=1, cycle_ticks=1_000_000, decode_service=dm,
                       corr_rounds=3, n_corr=3, return_ticks=500_000),
                   seed=53)


def input_staging_ping_pong():
    """Strong-only Tan windows on one unit with the depth-1 input staging
    slot: the next window's SBD transfer overlaps the current compute."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.decoders.weak_strong_switching import StrongOnly
    from decsim.links.link_profiles import logical_reference_profile
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec
    from decsim.windows.windowing_schemes import TanSandwichScheme
    return RunSpec(ops=[Operation(0, "memory", (0,), patches=(0,))], d=3,
                   rounds_policy=FixedRounds(27), round_us=1.0,
                   decoder=PresetLatencyDecoder(5.0), num_units=1,
                   scheme=TanSandwichScheme(), escalation_policy=StrongOnly(),
                   links=logical_reference_profile(), seed=3)


def sliding_boundary_at_decoder():
    """Strong-only sliding windows with DECODER boundary application and
    the depth-1 staging slot: raw rounds ship at data-complete, the mask
    lands at the decoder, and the chain runs at dd + max(sbd, decode)."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.decoders.weak_strong_switching import StrongOnly
    from decsim.links.link_profiles import logical_reference_profile
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec
    from decsim.windows.windowing_schemes import (SlidingTerminalPolicy,
                                                  SlidingWindowScheme)
    return RunSpec(ops=[Operation(0, "memory", (0,), patches=(0,))], d=3,
                   rounds_policy=FixedRounds(27), round_us=1.0,
                   decoder=PresetLatencyDecoder(5.0), num_units=1,
                   scheme=SlidingWindowScheme(
                       terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD),
                   escalation_policy=StrongOnly(),
                   links=logical_reference_profile(), seed=3)


def dma_setup_overhead():
    """The per-transfer DMA setup cost (Shao MICRO 2016 measured 400 ns per
    transaction) on the decoder-input paths, over a fast strong decoder so
    the overhead is a visible fraction of the transfer leg."""
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.decoders.weak_strong_switching import StrongOnly
    from decsim.links.link_profiles import (logical_reference_profile,
                                            with_transfer_overhead)
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec
    from decsim.windows.windowing_schemes import TanSandwichScheme
    links = with_transfer_overhead(
        logical_reference_profile(), overhead_us=0.4,
        source="Shao et al. MICRO 2016: 40 cycles at 100 MHz per DMA "
               "transaction, measured on the Zedboard")
    return RunSpec(ops=[Operation(0, "memory", (0,), patches=(0,))], d=3,
                   rounds_policy=FixedRounds(27), round_us=1.0,
                   decoder=PresetLatencyDecoder(1.0), num_units=1,
                   scheme=TanSandwichScheme(), escalation_policy=StrongOnly(),
                   links=links, seed=3)


SCENARIOS = {
    name: fn for name, fn in globals().items()
    if callable(fn) and not name.startswith("_")
    and fn.__module__ == __name__
    and not isinstance(fn, type)
    and name not in ("canonical", "capture", "record", "check", "main")
}


# ---- record / check --------------------------------------------------------

def _run(name):
    spec = SCENARIOS[name]()
    try:
        completed = spec.build(verbose=False)
    except Exception as error:
        return {"log": [], "final_ticks": None,
                "failure": {"type": type(error).__name__, "message": str(error),
                            "tick": getattr(error, "tick", None),
                            "status": getattr(error, "status", None)}}
    return capture(completed)


def _first_difference(path, old, new):
    if type(old) is not type(new):
        return f"{path}: type {type(old).__name__} -> {type(new).__name__}"
    if isinstance(old, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                return f"{path}.{key}: added"
            if key not in new:
                return f"{path}.{key}: removed"
            found = _first_difference(f"{path}.{key}", old[key], new[key])
            if found:
                return found
        return None
    if isinstance(old, list):
        for index, (a, b) in enumerate(zip(old, new)):
            found = _first_difference(f"{path}[{index}]", a, b)
            if found:
                return found
        if len(old) != len(new):
            return f"{path}: length {len(old)} -> {len(new)}"
        return None
    if old != new:
        return f"{path}: {old!r} -> {new!r}"
    return None


def record(names):
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in names:
        state = _run(name)
        (RESULTS / f"{name}.json").write_text(json.dumps(state, indent=0, sort_keys=True))
        print(f"recorded {name}: {len(state['log'])} log lines, {state['final_ticks']} ticks")


def check(names):
    failed = []
    for name in names:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            print(f"MISSING {name}")
            failed.append(name)
            continue
        old = json.loads(path.read_text())
        try:
            new = json.loads(json.dumps(_run(name), sort_keys=True))
        except Exception as error:  # a scenario that no longer builds is a difference
            print(f"FAIL {name}: {type(error).__name__}: {error}")
            failed.append(name)
            continue
        difference = _first_difference(name, old, new)
        if difference:
            print(f"FAIL {difference}")
            failed.append(name)
        else:
            print(f"ok   {name}")
    return failed


def main(argv):
    if len(argv) < 2 or argv[1] not in ("record", "check", "list"):
        print(__doc__)
        return 2
    names = argv[2:] or sorted(SCENARIOS)
    if argv[1] == "list":
        print("\n".join(sorted(SCENARIOS)))
        return 0
    if argv[1] == "record":
        record(names)
        return 0
    failed = check(names)
    print(f"{len(names) - len(failed)}/{len(names)} scenarios unchanged")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
