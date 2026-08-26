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
from decsim.decoders.decoder_engine import DecoderEngine, DecoderStage, DecoderTiming
from decsim.decoders.decoder_memory import DecoderMemoryConfig
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.decoders.weak_strong_switching import StrongOnly
from decsim.links.link_profiles import (logical_reference_profile,
                                        with_controller_to_buffer_edge,
                                        with_copy_out_edge)
from decsim.links.links import (LinkCapacityConfig, LinkConfig,
                                LinkQuantityBasis, TransferOverheadConfig)
from decsim.message import Operation
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import (NaiveOnlineScheme,
                                              ParallelWindowScheme,
                                              SlidingWindowScheme,
                                              TanSandwichScheme)

from experiments.experiment_config import MEASURED, ExperimentConfig

WINDOWING_SCHEMES = {
    "sliding": SlidingWindowScheme,
    "parallel": ParallelWindowScheme,
    "sandwich": TanSandwichScheme,
    "naive_online": NaiveOnlineScheme,
}


def memory_circuit(config: ExperimentConfig,
                   physical_error_probability: float) -> stim.Circuit:
    """The real data: Stim's generated memory circuit, one physical error
    probability on all four of Stim's noise channels (as Stim's guide does)."""
    p = physical_error_probability
    return stim.Circuit.generated(
        config.code_task, rounds=config.rounds_per_shot, distance=config.distance,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def decoder_engine(config: ExperimentConfig, algorithm_latency_us) -> DecoderEngine:
    """PyMatching inside the decoder engine: fetch cycles before it, release
    cycles after it, at the engine clock. The algorithm charges either its
    real wall clock (MEASURED) or the card's fixed core latency; a card
    prices the algorithm stage only, never a total decoder latency."""
    if algorithm_latency_us == MEASURED:
        algorithm = PyMatchingDecoder(latency_model=None)
    else:
        algorithm = PyMatchingDecoder(PresetLatencyDecoder(algorithm_latency_us))
    engine_card = config.decoder.engine
    fetch = DecoderStage("fetch", cycles_per_round=engine_card.fetch_cycles_per_round)
    release = DecoderStage("release", cycles_per_job=engine_card.release_cycles_per_job)
    timing = DecoderTiming(before=(fetch,), after=(release,),
                           frequency_mhz=engine_card.frequency_mhz)
    return DecoderEngine(algorithm, timing)


def link_model(config: ExperimentConfig):
    """Every path's numbers from the config, on the reference card's payload
    sizes; a null card keeps the reference card's numbers for that path.
    C2B and copy_out are the two optional store hops."""
    source = f"experiments/configs/{config.name}.yaml links"
    cards = dict(config.links)
    profile = logical_reference_profile()
    c2b = cards.pop("c2b")
    if c2b is not None:
        profile = with_controller_to_buffer_edge(
            profile, latency_us=c2b.latency_us,
            aggregate_bits_per_us=c2b.bits_per_us, source=source)
    copy_out = cards.pop("copy_out")
    if copy_out is not None:
        profile = with_copy_out_edge(
            profile, latency_us=copy_out.latency_us,
            aggregate_bits_per_us=copy_out.bits_per_us, source=source)
    channels = {}
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
        channels[path] = replace(getattr(profile, path), channel=channel,
                                 transfer_overhead=overhead)
    # The config prices controller processing on its own line
    # (controller.t_binary_availability_us), so its qc card is link
    # propagation only; the attestation lets a nonzero processing cost run.
    return replace(profile, **channels, profile_name=f"{config.name}.yaml",
                   qc_excludes_controller_processing=True)


def code_model(config: ExperimentConfig) -> SurfaceCodeModel:
    return SurfaceCodeModel(d=config.distance,
                            commit_rounds_override=config.windowing.commit_rounds,
                            buffer_rounds_override=config.windowing.buffer_rounds)


def decoder_memory(config: ExperimentConfig):
    if config.decoder_memory_rounds is None:
        return None
    return DecoderMemoryConfig({"default": config.decoder_memory_rounds})


def escalation_policy(config: ExperimentConfig):
    """weak_baseline: None, the core's default (every window weak, final).
    strong_only: every window decoded once on the strong tier."""
    if config.mode == "strong_only":
        return StrongOnly()
    return None


def build_run(config: ExperimentConfig, *, physical_error_probability: float,
              round_period_us: float, algorithm_latency_us, seed: int):
    """The wired RunSpec for one sweep point and seed, plus its engine
    (the engine is returned so the measurement can read its stage records)."""
    circuit = memory_circuit(config, physical_error_probability)
    operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                          circuit=circuit)
    engine = decoder_engine(config, algorithm_latency_us)
    timing = TimingConfig(
        round_us=round_period_us,
        t_binary_availability_us=config.controller.t_binary_availability_us,
        t_pack_us=config.controller.t_pack_us)
    spec = RunSpec(
        ops=[operation], code=code_model(config),
        scheme=WINDOWING_SCHEMES[config.windowing.scheme](),
        rounds_policy=FixedRounds(config.rounds_per_shot),
        device=StimDevice(), decoder=engine, num_units=config.decoder.units,
        timing=timing, links=link_model(config),
        decoder_memory=decoder_memory(config),
        escalation_policy=escalation_policy(config),
        pauli_frame=PauliFrameConfig(commit_us=config.pauli_frame_commit_us),
        seed=seed)
    return spec, engine
