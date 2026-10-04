"""The whole machine's settings: one record per part.

The aggregate object model of a run, kept apart from the build script
that reads it. gem5 draws the same line: a component's parameters are
its own Python class (src/python/m5/SimObject.py:204-205) and the
configuration script in configs/ reads them
(configs/deprecated/example/se.py), never the other way round. Each
part's record is defined by the package that owns it.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Union

import decsim.config as config
import decsim.controller.policies as idle_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.producers as producers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.layouts as layouts
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.windows as window_records
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


@dataclasses.dataclass(frozen=True)
class MachineSettings:
    """One settings record per part of the machine.

    Every field has a default, so a Python caller names only what
    differs from a timing-only run of three-qubit surface code patches
    with no decoder at all. links is the fabric card; the reference card
    prices propagation only. clock is the machine's clock, the one every
    part that names none of its own counts its cycles on; it is stated
    here alone, and a part on it names none (the controller's clock
    None).

    The decode side is three slots, each None when the run has no such
    part: weak_decoder alone decodes every window once on the weak
    decoder, strong_decoder alone on the strong one, and both with
    switching decode weak first and escalate a window to the strong
    decoder; none of the three is a run that plans no decoding. Two
    decoders without switching is refused.
    """

    clock: Optional[config.Clock] = None
    qpu: qpu_settings.QpuSettings = qpu_settings.QpuSettings()
    controller: controller_settings.ControllerSettings = (
        controller_settings.ControllerSettings()
    )
    idle_policy: Union[
        idle_policies.IgnoreSettings, idle_policies.SeparateDecodeJobsSettings
    ] = idle_policies.SeparateDecodeJobsSettings()
    detection_events: detection_event_settings.DetectionEventSettings = (
        detection_event_settings.DetectionEventSettings()
    )
    links: link_settings.FabricSettings = (
        link_profiles.logical_reference_profile()
    )
    weak_syndrome_buffer: Union[
        syndrome_buffer_module.SyndromeBufferSettings,
        ported_syndrome_buffer.PortedSyndromeBufferSettings,
    ] = syndrome_buffer_module.SyndromeBufferSettings()
    strong_syndrome_buffer: syndrome_buffer_module.SyndromeBufferSettings = (
        syndrome_buffer_module.SyndromeBufferSettings()
    )
    windows: window_settings.WindowSettings = window_settings.WindowSettings()
    weak_decoder: Optional[decoder_settings.DecoderPoolSettings] = None
    strong_decoder: Optional[decoder_settings.DecoderPoolSettings] = None
    decoder_manager: decoder_settings.DecoderManagerSettings = (
        decoder_settings.DecoderManagerSettings()
    )
    switching: Optional[escalation_settings.SwitchingSettings] = None
    pauli_frame: Optional[pauli_frame_module.PauliFrameConfig] = None
    workload: workload_settings.WorkloadSettings = (
        workload_settings.WorkloadSettings()
    )
    magic_state_factory: qpu_settings.FactorySettings = (
        magic_state_factories.InfiniteFactory.Settings()
    )
    observation: observe_settings.ObservationSettings = (
        observe_settings.ObservationSettings()
    )

    def __post_init__(self) -> None:
        _check_decode_slots(self)

    def decoder_settings_for(
        self, tier: str
    ) -> Optional[decoder_settings.DecoderPoolSettings]:
        """The pool of one decoder tier, by its tier word, weak or strong."""
        if tier == "weak":
            return self.weak_decoder
        assert tier == "strong", f"no decoder tier named {tier!r}"
        return self.strong_decoder

    @property
    def window_tier(self) -> window_records.DecoderTier:
        """The tier that decodes the plan's windows: weak if set, else strong.

        A run with no decoder plans no decoding, and its windows, if any,
        are the weak tier's.
        """
        if self.weak_decoder is None and self.strong_decoder is not None:
            return window_records.DecoderTier.STRONG
        return window_records.DecoderTier.WEAK

    @property
    def escalation_kind(self) -> str:
        """The word a trace names these slots' run by.

        The trace's process name and the plots that read it name a run
        by it: switching, strong_only or weak_baseline.
        """
        if self.switching is not None:
            return "switching"
        if self.window_tier is window_records.DecoderTier.STRONG:
            return "strong_only"
        return "weak_baseline"

    def point_facts(self) -> dict:
        """The point's facts a threshold reads, None where none is stated.

        The names are threshold_sources.POINT_FACTS: the qpu's distance
        and round period, the physical error probability the workload
        was made at, and the window sizes, None being the distance. A
        threshold reads them here, off the point, so a run states each
        once.
        """
        workload_record = self.workload.workload_record
        probability = None
        if workload_record is not None:
            probability = workload_record.physical_error_probability
        scheme = self.windows.scheme
        return {
            "distance": self.qpu.distance,
            "physical_error_probability": probability,
            "round_period_microseconds": self.qpu.round_period_microseconds,
            "commit_rounds": scheme.commit_rounds,
            "buffer_rounds": scheme.buffer_rounds,
        }

    def at_point(self) -> "MachineSettings":
        """The settings with the threshold their point's facts give.

        A calibration table's row is read here, once per point by the
        point's task (collect.Task), or by Machine.build for a record no
        task read; settings read at their point are their own reading,
        and any other threshold is the same at every point.
        """
        switching = self.switching
        if switching is None:
            return self
        facts = self.point_facts()
        threshold = switching.threshold.at_point(facts)
        if threshold is switching.threshold:
            return self
        resolved = dataclasses.replace(switching, threshold=threshold)
        return dataclasses.replace(self, switching=resolved)


# The shipped machines' two clock domains: the chip's, at 250 MHz
# (2605.04892 line 1063), and the host's, at the same rate, an estimate
# that sets edge rounding only.
FRIDGE_CLOCK = config.Clock.from_megahertz(250.0)
ROOM_CLOCK = config.Clock.from_megahertz(250.0)
# The hops the weak base prices: the weak chain, the decision and the
# pulse. The strong side keeps the reference card.
_WEAK_BASELINE_PATH_CLOCKS = {
    "qpu_to_controller": FRIDGE_CLOCK,
    "controller_to_weak_buffer": FRIDGE_CLOCK,
    "weak_buffer_to_weak_decoder": FRIDGE_CLOCK,
    "decoder_to_decoder": FRIDGE_CLOCK,
    "weak_decoder_to_frame": FRIDGE_CLOCK,
    "frame_to_controller": FRIDGE_CLOCK,
    "controller_to_qpu": FRIDGE_CLOCK,
}
# The hops the strong base prices: the readout, the weak store's write,
# the decision and the pulse on the chip; the strong store's write and
# the strong chain on the host. The weak chain and the escalation keep
# the reference card.
_STRONG_BASELINE_PATH_CLOCKS = {
    "qpu_to_controller": FRIDGE_CLOCK,
    "controller_to_weak_buffer": FRIDGE_CLOCK,
    "controller_to_strong_buffer": ROOM_CLOCK,
    "strong_buffer_to_strong_decoder": ROOM_CLOCK,
    "decoder_to_decoder": ROOM_CLOCK,
    "strong_decoder_to_frame": ROOM_CLOCK,
    "frame_to_controller": FRIDGE_CLOCK,
    "controller_to_qpu": FRIDGE_CLOCK,
}
# The strong side's four hops, all on the host.
_STRONG_SIDE_PATH_CLOCKS = {
    "controller_to_strong_buffer": ROOM_CLOCK,
    "weak_decoder_to_strong_decoder": ROOM_CLOCK,
    "strong_buffer_to_strong_decoder": ROOM_CLOCK,
    "strong_decoder_to_frame": ROOM_CLOCK,
}
# A base's shot is ten rounds per unit of distance.
_BASELINE_ROUNDS_PER_DISTANCE = 10
_BELIEF_MATCHING = belief_matching.BeliefMatchingDecoder.Settings(
    max_iterations=30, belief_propagation_method="product_sum"
)
# The bases' engine: a round a cycle in and ten cycles to write the
# correction out. No source states either count, so both are estimates.
_ESTIMATED_ENGINE = decoder_settings.EngineSettings(
    clock=FRIDGE_CLOCK,
    fetch_cycles_per_round=1,
    fetch_cycles_per_job=0,
    release_cycles_per_job=10,
    release_cycles_per_round=0,
)
# the same engine counted on the host's clock
_HOST_ESTIMATED_ENGINE = dataclasses.replace(
    _ESTIMATED_ENGINE, clock=ROOM_CLOCK
)
# Matching's answer at the latency of a d = 3 lookup-table core, 7 cycles
# at 250 MHz, 28 ns (LILLIPUT, 2108.06569 line 1070 and Table 4, text
# lines 1088-1100). The core is a lookup table, so the card borrows its
# time, not its answer.
_TIMED_MATCHING = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
    preset_latency_microseconds=0.028
)
_UNBOUNDED_UNIT_MEMORY = decoder_settings.UnitMemorySettings(
    bits=None, word_bits=None
)
# Belief matching on one host unit with an unbounded memory, charged its
# measured wall clock: the strong base's tier.
_HOST_BELIEF_MATCHING_POOL = decoder_settings.DecoderPoolSettings(
    algorithm=_BELIEF_MATCHING,
    unit_count=1,
    engine=_HOST_ESTIMATED_ENGINE,
    unit_memory=_UNBOUNDED_UNIT_MEMORY,
    copies_input=True,
    copies_boundary_fold=True,
    result_blocks_unit=False,
)


def weak_decoder_baseline(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> MachineSettings:
    """The weak decoder alone.

    Real PyMatching answers every window on one chip unit, charged
    LILLIPUT's 28 ns, and each hop it prices is one fridge cycle on an
    unbounded wire.
    """
    reference = link_profiles.logical_reference_profile()
    links = _one_cycle_paths(
        reference, _WEAK_BASELINE_PATH_CLOCKS, "the weak base's hops"
    )
    unit_memory = decoder_settings.UnitMemorySettings(bits=None, word_bits=None)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=_TIMED_MATCHING,
        unit_count=1,
        engine=_ESTIMATED_ENGINE,
        unit_memory=unit_memory,
        copies_input=True,
        copies_boundary_fold=True,
        result_blocks_unit=False,
    )
    qpu = _stim_qpu(distance, round_period_microseconds)
    rounds_per_shot = _BASELINE_ROUNDS_PER_DISTANCE * distance
    workload = memory_workload(
        distance, physical_error_probability, rounds_per_shot
    )
    return _baseline(qpu, workload, links, weak_decoder, None)


def strong_decoder_baseline(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> MachineSettings:
    """The strong decoder alone.

    Belief matching answers every window on one host unit, charged its
    measured wall clock. Each hop it prices is one cycle on an unbounded
    wire, of the fridge clock on the chip and the room clock on the
    host.
    """
    reference = link_profiles.logical_reference_profile()
    links = _one_cycle_paths(
        reference, _STRONG_BASELINE_PATH_CLOCKS, "the strong base's hops"
    )
    qpu = _stim_qpu(distance, round_period_microseconds)
    rounds_per_shot = _BASELINE_ROUNDS_PER_DISTANCE * distance
    workload = memory_workload(
        distance, physical_error_probability, rounds_per_shot
    )
    return _baseline(qpu, workload, links, None, _HOST_BELIEF_MATCHING_POOL)


def one_cycle_strong_side(
    links: link_settings.FabricSettings,
) -> link_settings.FabricSettings:
    """The card with the strong side's four hops one room cycle each.

    A run that switches on the weak base reaches the strong decoder over
    these hops, each an unbounded wire; every other path keeps its card.
    """
    return _one_cycle_paths(
        links, _STRONG_SIDE_PATH_CLOCKS, "the strong side's hops"
    )


def memory_workload(
    distance: int, physical_error_probability: float, rounds_per_shot: int
) -> workload_settings.WorkloadSettings:
    """One memory shot of Stim's rotated surface code, rounds_per_shot long.

    One probability on all four of Stim's noise channels
    (producers.memory_circuit).
    """
    workload = producers.memory_circuit(
        "surface_code:rotated_memory_z",
        rounds_per_shot,
        distance,
        physical_error_probability,
    )
    return workload_settings.WorkloadSettings.running(workload)


def _stim_qpu(
    distance: int, round_period_microseconds: float
) -> qpu_settings.QpuSettings:
    """Stim samples the rotated surface code at distance, on one layout."""
    source = stim_device.StimDevice.Settings()
    code_card = code_geometry.SurfaceCodeModel.Settings()
    layout = layouts.UniformLayout.Settings()
    return qpu_settings.QpuSettings(
        source=source,
        code_card=code_card,
        round_period_microseconds=round_period_microseconds,
        distance=distance,
        layout=layout,
        error_model_provider=None,
    )


def _baseline(
    qpu: qpu_settings.QpuSettings,
    workload: workload_settings.WorkloadSettings,
    links: link_settings.FabricSettings,
    weak_decoder: Optional[decoder_settings.DecoderPoolSettings],
    strong_decoder: Optional[decoder_settings.DecoderPoolSettings],
) -> MachineSettings:
    """What both bases share, every value written out, around one decoder.

    Both stores are unbounded and free, and the frame writes in one
    fridge cycle, 4 ns (2605.04892 Table I, lines 1051 and 1063).
    """
    controller = _baseline_controller()
    idle_policy = idle_policies.SeparateDecodeJobsSettings()
    detection_events = _baseline_detection_events()
    unbounded_buffer = syndrome_buffer_module.SyndromeBufferSettings(
        bits=None, clock=None, write_cycles=0, read_cycles=0
    )
    windows = _baseline_windows()
    decoder_manager = _baseline_decoder_manager()
    magic_state_factory = magic_state_factories.InfiniteFactory.Settings()
    observation = _baseline_observation()
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=1, clock=FRIDGE_CLOCK
    )
    return MachineSettings(
        # the controller and every part that names no clock of its own
        # count their cycles on the machine's clock
        clock=FRIDGE_CLOCK,
        qpu=qpu,
        controller=controller,
        idle_policy=idle_policy,
        detection_events=detection_events,
        links=links,
        weak_syndrome_buffer=unbounded_buffer,
        strong_syndrome_buffer=unbounded_buffer,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        decoder_manager=decoder_manager,
        switching=None,
        pauli_frame=pauli_frame,
        workload=workload,
        magic_state_factory=magic_state_factory,
        observation=observation,
    )


def _baseline_controller() -> controller_settings.ControllerSettings:
    """A controller whose per-round costs sit inside the round period."""
    return controller_settings.ControllerSettings(
        readout_to_bits_cycles=0,
        packing_cycles_per_round=0,
        decision_to_pulse_cycles=0,
        packing_rounds_in_flight=None,
    )


def _baseline_detection_events() -> (
    detection_event_settings.DetectionEventSettings
):
    """The controller forms every round's detection events, at no cost."""
    return detection_event_settings.DetectionEventSettings(
        formed_at=("controller",),
        clock=None,
        latency_cycles=0,
        cycles_per_round=0,
    )


def _baseline_windows() -> window_settings.WindowSettings:
    """Sliding windows of the code distance, issued free on the machine clock.

    The last window drains with a flush tail, each committed window
    ships its boundary at once, and the hand-off is a dense seam mask.
    """
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=None, buffer_rounds=None
    )
    eager = boundary_policies.Eager.Settings()
    dense_seam_mask = boundary_payloads.DenseSeamMask.Settings()
    return window_settings.WindowSettings(
        clock=None,
        decision_cycles=0,
        scheme=scheme,
        terminal_policy="flush",
        boundary_policy=eager,
        boundary_payload=dense_seam_mask,
    )


def _baseline_decoder_manager() -> decoder_settings.DecoderManagerSettings:
    """First in, first out, one region a decode, dispatch unpriced."""
    fifo = schedulers.FifoScheduler.Settings()
    return decoder_settings.DecoderManagerSettings(
        scheduler=fifo,
        bulk_strong=False,
        dispatch_cycles=0,
        clock=None,
    )


def _baseline_observation() -> observe_settings.ObservationSettings:
    """The results alone: no log, no trace, no extra counter."""
    return observe_settings.ObservationSettings(
        log="off",
        log_component_io=False,
        record_switching_windows=False,
        backlog_trace=False,
        trace="off",
        trace_shots=(0,),
        data_movement=False,
    )


def _one_cycle_paths(
    links: link_settings.FabricSettings, path_clocks: Mapping, hops: str
) -> link_settings.FabricSettings:
    """The card with each named path one cycle of its clock, the rest kept.

    Each named path is an unbounded wire of one lane, with no setup,
    header or protocol; hops names the set in the card's name.
    """
    source = f"{hops}: one cycle of the hop's clock, an idealized wire"
    cards = {}
    for path_name, clock in path_clocks.items():
        cards[path_name] = link_profiles.path_card(
            links,
            path_name,
            clock=clock,
            latency_cycles=1,
            bits_per_cycle=None,
            source=source,
            lane_count=1,
            setup_cycles_per_transfer=0,
            header_bits_per_transfer=0,
            protocol=None,
        )
    profile_name = f"{links.profile_name} with {hops} at one cycle"
    return dataclasses.replace(links, **cards, profile_name=profile_name)


def _check_decode_slots(settings: MachineSettings) -> None:
    """Without switching, one decoder; a second would run on ignored."""
    has_weak = settings.weak_decoder is not None
    has_strong = settings.strong_decoder is not None
    has_two_decoders = has_weak and has_strong
    if settings.switching is not None or not has_two_decoders:
        return
    raise ValueError(
        "weak_decoder and strong_decoder are both set and switching is "
        "not; a run with no switching decodes its windows on one decoder, "
        "so set switching or drop one decoder"
    )
