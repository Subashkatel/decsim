"""One sweep point -> one wired RunSpec. Nothing runs here.

Every function turns one card of the config into the core object it names;
`build_run` assembles them. The mode picks the decode path: weak_baseline
leaves the core's default policy (every window on the weak tier),
strong_only installs the StrongOnly policy (every window on the strong
tier, woken from syndrome buffer 1).
"""

from __future__ import annotations

from dataclasses import replace

import stim

from decsim.config import TimingConfig, us as us_ticks
from decsim.decoders.decoder_engine import (DecoderEngine, DecoderStage,
                                            DecoderTiming)
from decsim.decoders.decoder_memory import DecoderMemoryConfig
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.decoders.weak_strong_switching import StrongOnly
from decsim.links.link_profiles import (logical_reference_profile,
                                        with_controller_to_buffer_edge,
                                        with_csb_edge)
from decsim.links.links import (LinkCapacityConfig, LinkConfig,
                                LinkQuantityBasis, TransferOverheadConfig)
from decsim.message import Operation
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice
from decsim.run_spec import RunSpec
from decsim.syndrome_buffer.syndrome_buffer import SyndromeBufferingConfig
from decsim.windows.windowing_schemes import (NaiveOnlineScheme,
                                              ParallelWindowScheme,
                                              SlidingWindowScheme,
                                              TanSandwichScheme)

from experiments.experiment_config import ExperimentConfig

WINDOWING_SCHEMES = {
    "sliding": SlidingWindowScheme,
    "parallel": ParallelWindowScheme,
    "sandwich": TanSandwichScheme,
    "naive_online": NaiveOnlineScheme,
}


def memory_circuit(config: ExperimentConfig, physical_error_probability: float,
                   distance: int) -> stim.Circuit:
    """The real data: Stim's generated memory circuit, one physical error
    probability on all four of Stim's noise channels (as Stim's guide does)."""
    p = physical_error_probability
    return stim.Circuit.generated(
        config.code_task, rounds=config.rounds_per_shot.rounds_for(distance),
        distance=distance,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def decoder_engine(config: ExperimentConfig) -> DecoderEngine:
    """The mode's decoder unit built from its card: the algorithm between
    fetch cycles before it and release cycles after it, on the engine's
    named clock. A named algorithm decodes every window for real and charges
    its measured wall clock (pymatching = MWPM, the weak tier;
    belief_matching = the strong tier, Toshio arXiv 2510.25222); a number is
    a fixed core latency in us on the MWPM path. The algorithm card prices
    the algorithm stage only, never a total decoder latency."""
    unit = config.active_decoder
    if unit.algorithm == "pymatching":
        algorithm = PyMatchingDecoder(latency_model=None)
    elif unit.algorithm == "belief_matching":
        from decsim.decoders.belief_matching.decoder import (
            BeliefMatchingDecoder)
        algorithm = BeliefMatchingDecoder(latency_model=None)
    else:
        algorithm = PyMatchingDecoder(PresetLatencyDecoder(unit.algorithm))
    if config.verify_windows == "tesseract":
        algorithm = TesseractCheckedDecoder(algorithm)
    fetch = DecoderStage(
        "fetch", cycles_per_round=unit.engine.fetch_cycles_per_round)
    release = DecoderStage(
        "release", cycles_per_job=unit.engine.release_cycles_per_job)
    timing = DecoderTiming(before=(fetch,), after=(release,),
                           frequency_mhz=unit.engine.frequency_mhz)
    return DecoderEngine(algorithm, timing)


class TesseractCheckedDecoder:
    """The referee: after every tier decode, the official Tesseract backend
    re-decodes the same window input and the owned observable contributions
    are compared. Never priced: the engine reads timing from the inner
    decoder alone. Its linked fault models are built whole-circuit, with
    memory linear in circuit length since the sparse audit (bee219c);
    verified through d=9 x 1000 rounds."""

    def __init__(self, inner):
        from decsim.detector_error_model.fault_model_contracts import (
            LINKED_FAULT_MODELS_REQUIRED)
        from decsim.decoders.tesseract import TesseractWindowDecoder
        self.inner = inner
        self.referee = TesseractWindowDecoder()
        # the referee reads the physical view, the tier the graphlike one
        self.fault_model_requirement = LINKED_FAULT_MODELS_REQUIRED
        self.windows_checked = 0
        self.window_disagreements = 0

    @property
    def measures_wall_clock(self):
        return getattr(self.inner, "measures_wall_clock", False)

    @property
    def last_decode_ns(self):
        return self.inner.last_decode_ns

    def run_seed_children(self):
        from decsim.message import RunSeedChild, RunSeedPathSegment
        children = [RunSeedChild((RunSeedPathSegment("field", "referee"),),
                                 self.referee)]
        inner_children = getattr(self.inner, "run_seed_children", None)
        if inner_children is not None:
            for child in inner_children():
                children.append(RunSeedChild(
                    (RunSeedPathSegment("field", "inner"),)
                    + child.relative_path,
                    child.child))
        return tuple(children)

    def latency(self, job):
        return self.inner.latency(job)

    def decode(self, job):
        import numpy as np
        from decsim.decoders.window_decode_results import (BackendDecodeStatus,
                                                           payload_syndrome)
        from decsim.detector_error_model.fault_model_contracts import (
            FaultRepresentation)
        result = self.inner.decode(job)
        model = job.dem
        if model is None or result.logical_observables is None:
            return result
        outcome = self.referee.decode(model, payload_syndrome(job))
        if outcome.status is not BackendDecodeStatus.SUCCEEDED:
            return result
        physical = model.require_faults(FaultRepresentation.PHYSICAL)
        referee_correction = np.asarray(outcome.physical_correction,
                                        dtype=bool)
        # only this window's owned faults count toward its contribution
        owned_correction = referee_correction & physical.owned
        observable_flip_counts = (physical.observables.astype(np.int64)
                                  @ owned_correction.astype(np.int64))
        flip_counts = np.asarray(observable_flip_counts).ravel()
        referee_flips = tuple(int(count) % 2 for count in flip_counts)
        self.windows_checked += 1
        if referee_flips != tuple(result.logical_observables):
            self.window_disagreements += 1
        return result


def link_model(config: ExperimentConfig):
    """Every path's numbers from the config, on the reference card's payload
    sizes; a null card keeps the reference card's numbers for that path.
    CWB and csb are the two optional store hops."""
    source = f"experiments/configs/{config.name}.yaml links"
    cards = dict(config.links)
    profile = logical_reference_profile()
    cwb = cards.pop("cwb")
    if cwb is not None:
        profile = with_controller_to_buffer_edge(
            profile, latency_us=cwb.latency_us,
            aggregate_bits_per_us=cwb.bits_per_us, source=source)
    csb = cards.pop("csb")
    if csb is not None:
        profile = with_csb_edge(
            profile, latency_us=csb.latency_us,
            aggregate_bits_per_us=csb.bits_per_us, source=source)
    edge_overrides = {}
    for path, card in cards.items():
        if card is None:
            continue
        capacity = None
        if card.bits_per_us is not None:
            capacity = LinkCapacityConfig(card.bits_per_us,
                                          LinkQuantityBasis.DIRECT_AGGREGATE,
                                          None, source)
        channel = LinkConfig(us_ticks(card.latency_us), capacity, source)
        overhead = None
        if card.transfer_overhead_us:
            overhead = TransferOverheadConfig(
                us_ticks(card.transfer_overhead_us), source)
        edge_overrides[path] = replace(getattr(profile, path), channel=channel,
                                       transfer_overhead=overhead)
    # The config prices controller processing on its own line
    # (controller.t_binary_availability_us), so its qc card is link
    # propagation only; the attestation lets a nonzero processing cost run.
    return replace(profile, **edge_overrides,
                   profile_name=f"{config.name}.yaml",
                   qc_excludes_controller_processing=True)


def code_model(config: ExperimentConfig, distance: int) -> SurfaceCodeModel:
    return SurfaceCodeModel(
        d=distance,
        commit_rounds_override=config.windowing.commit_rounds,
        buffer_rounds_override=config.windowing.buffer_rounds)


def idle_policy(config: ExperimentConfig):
    """The named idle-round card resolved to its policy object."""
    from decsim.controller.policies import (ExtendStream, Ignore,
                                            SeparateDecodeJobs)
    return {"separate_decode_jobs": SeparateDecodeJobs,
            "ignore": Ignore,
            "extend_stream": ExtendStream}[config.idle_policy]()


def decoder_memory(config: ExperimentConfig):
    unit_buffer_size = config.active_decoder.unit_buffer_size
    if unit_buffer_size is None:
        return None
    return DecoderMemoryConfig({"default": unit_buffer_size})


def syndrome_buffering(config: ExperimentConfig) -> SyndromeBufferingConfig:
    """The syndrome-path store capacities, straight off the buffers card
    (rounds; None = unbounded)."""
    buffers = config.buffers
    return SyndromeBufferingConfig(
        upstream_packet_slots=buffers.buffer_0_size,
        sb1_packet_slots=buffers.buffer_1_size,
        packing_assembly_slots=buffers.packing_workspace_size)


def escalation_policy(config: ExperimentConfig):
    """weak_baseline: None, the core's default (every window weak, final).
    strong_only: every window decoded once on the strong tier."""
    if config.mode == "strong_only":
        return StrongOnly()
    return None


def build_run(config: ExperimentConfig, *, physical_error_probability: float,
              distance: int, round_period_us: float, seed: int):
    """The wired RunSpec for one sweep point and seed, plus its engine
    (the engine is returned so the measurement can read its stage records)."""
    circuit = memory_circuit(config, physical_error_probability, distance)
    operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                          circuit=circuit)
    engine = decoder_engine(config)
    timing = TimingConfig(
        round_us=round_period_us,
        t_binary_availability_us=config.controller.t_binary_availability_us,
        t_pack_us=config.controller.t_pack_us)
    spec = RunSpec(
        ops=[operation], code=code_model(config, distance),
        scheme=WINDOWING_SCHEMES[config.windowing.scheme](),
        rounds_policy=FixedRounds(config.rounds_per_shot.rounds_for(distance)),
        device=StimDevice(), decoder=engine,
        num_units=config.active_decoder.units,
        timing=timing, links=link_model(config),
        decoder_memory=decoder_memory(config),
        syndrome_buffering=syndrome_buffering(config),
        escalation_policy=escalation_policy(config),
        idle_policy=idle_policy(config),
        pauli_frame=PauliFrameConfig(commit_us=config.pauli_frame_commit_us),
        seed=seed)
    return spec, engine
