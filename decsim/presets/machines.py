"""The decoder bases, written as graphs, and a strong tier added to one.

Each base is every part a machine holds, every value written out, around
one decoder: the weak base's PyMatching on a chip unit, the strong base's
belief matching on a host unit. The parts are named as
topology/legacy.py's graph_of names them, so `--set weak.unit_count=2`
reaches the weak pool of any of them.

    weak_baseline      the weak decoder alone, its hops a fridge cycle
    strong_baseline    the strong decoder alone, its hops a cycle each
    with_strong_tier   a machine that escalates its weak decoder's
                       windows to a strong tier on the host
    with_escalation    a machine whose pool escalates on to one more
                       pool, a chain grown one edge at a time
    with_router        a machine whose entry store feeds a router to
                       its entry pool and more, a decoder per patch
    memory_workload    one memory shot of Stim's rotated surface code
"""

import dataclasses
from typing import Any, Optional

import decsim.controller.policies as idle_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.ports as ports
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.producers as producers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.layouts as layouts
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.routers as routers
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the parts' names, as topology/legacy.py's graph_of names them
QPU = "qpu"
CONTROLLER = "controller"
WINDOWS = "windows"
FRAME = "frame"
WEAK_STORE = "weak_buffer"
WEAK_POOL = "weak"
STRONG_STORE = "strong_buffer"
STRONG_POOL = "strong"
SWITCHING = "switching"
ROUTER = "router"
# the loop every machine closes: readout, decision, pulse
LOOP_WIRES = (
    graph.Wire("qpu.readout", "controller.readout"),
    graph.Wire("frame.decisions", "controller.decisions"),
    graph.Wire("controller.pulses", "qpu.pulses"),
)
WEAK_STORE_WRITE = graph.Wire("controller.rounds", "weak_buffer.rounds")
# each tier's store read and corrections
WEAK_TIER_WIRES = (
    graph.Wire("weak_buffer.reads", "weak.rounds"),
    graph.Wire("weak.corrections", "frame.corrections"),
)
STRONG_TIER_WIRES = (
    graph.Wire("controller.rounds", "strong_buffer.rounds"),
    graph.Wire("strong_buffer.reads", "strong.rounds"),
    graph.Wire("strong.corrections", "frame.corrections"),
)
# the weak pool's escalations reach the switching part, whose re-decodes
# reach the strong pool
ESCALATION_WIRES = (
    graph.Wire("weak.escalate", "switching.results"),
    graph.Wire("switching.redecode", "strong.escalated"),
)
# A base's shot is ten rounds per unit of distance.
BASELINE_ROUNDS_PER_DISTANCE = 10
_BELIEF_MATCHING = belief_matching.BeliefMatchingDecoder.Settings(
    max_iterations=30, belief_propagation_method="product_sum"
)
# The bases' engine: a round a cycle in and ten cycles to write the
# correction out. No source states either count, so both are estimates.
_ESTIMATED_ENGINE = decoder_settings.EngineSettings(
    clock=clocks.FRIDGE_CLOCK,
    fetch_cycles_per_round=1,
    fetch_cycles_per_job=0,
    release_cycles_per_job=10,
    release_cycles_per_round=0,
)
# the same engine counted on the host's clock
_HOST_ESTIMATED_ENGINE = dataclasses.replace(
    _ESTIMATED_ENGINE, clock=clocks.ROOM_CLOCK
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
# PyMatching on one chip unit with an unbounded memory, charged
# LILLIPUT's 28 ns: the weak base's tier.
WEAK_BASELINE_POOL = decoder_settings.DecoderPoolSettings(
    algorithm=_TIMED_MATCHING,
    unit_count=1,
    engine=_ESTIMATED_ENGINE,
    unit_memory=_UNBOUNDED_UNIT_MEMORY,
    copies_input=True,
    copies_boundary_fold=True,
    result_blocks_unit=False,
)
# Belief matching on one host unit with an unbounded memory, charged its
# measured wall clock: the strong base's tier.
STRONG_BASELINE_POOL = decoder_settings.DecoderPoolSettings(
    algorithm=_BELIEF_MATCHING,
    unit_count=1,
    engine=_HOST_ESTIMATED_ENGINE,
    unit_memory=_UNBOUNDED_UNIT_MEMORY,
    copies_input=True,
    copies_boundary_fold=True,
    result_blocks_unit=False,
)
# Both bases' stores are unbounded and free.
_UNBOUNDED_STORE = syndrome_buffer_module.SyndromeBufferSettings(
    bits=None, clock=None, write_cycles=0, read_cycles=0
)


def weak_baseline(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> graph.Machine:
    """The weak decoder alone.

    Real PyMatching answers every window on one chip unit, charged
    LILLIPUT's 28 ns, and each hop it prices is one fridge cycle on an
    unbounded wire.
    """
    base = _baseline(
        distance, physical_error_probability, round_period_microseconds
    )
    weak_pool = graph.Part(WEAK_POOL, WEAK_BASELINE_POOL, fabrics.FRIDGE)
    with_pool = base.with_part(weak_pool)
    wired = with_pool.with_wires(*WEAK_TIER_WIRES)
    return dataclasses.replace(wired, fabric=link_cards.WEAK_BASELINE_FABRIC)


def strong_baseline(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> graph.Machine:
    """The strong decoder alone.

    Belief matching answers every window on one host unit, charged its
    measured wall clock. Each hop it prices is one cycle on an unbounded
    wire, of the fridge clock on the chip and the room clock on the
    host. The controller still writes the weak store, which no decoder
    reads.
    """
    base = _baseline(
        distance, physical_error_probability, round_period_microseconds
    )
    strong_store = graph.Part(STRONG_STORE, _UNBOUNDED_STORE, fabrics.HOST)
    strong_pool = graph.Part(STRONG_POOL, STRONG_BASELINE_POOL, fabrics.HOST)
    with_store = base.with_part(strong_store)
    with_pool = with_store.with_part(strong_pool)
    wired = with_pool.with_wires(*STRONG_TIER_WIRES)
    return dataclasses.replace(wired, fabric=link_cards.STRONG_BASELINE_FABRIC)


def with_strong_tier(
    machine: graph.Machine,
    strong_pool: decoder_settings.DecoderPoolSettings,
    switching: escalation_settings.SwitchingSettings,
    fabric: Optional[ports.FabricRule] = None,
) -> graph.Machine:
    """The machine escalating its weak decoder's windows to a strong tier.

    The strong store and pool sit on the host, the weak pool escalates
    to the strong one under switching, and the windows take the strong
    window's shape (window_settings.switching_windows). fabric, when
    given, replaces the machine's: a base's own hops reach no strong
    side, so a study on a base names one (link_cards.strong_side).
    """
    windows = machine.part(WINDOWS)
    switching_windows = window_settings.switching_windows(
        windows.settings, switching.strong_window
    )
    store = syndrome_buffer_module.SyndromeBufferSettings()
    parts = (
        graph.Part(WINDOWS, switching_windows),
        graph.Part(STRONG_STORE, store, fabrics.HOST),
        graph.Part(STRONG_POOL, strong_pool, fabrics.HOST),
        graph.Part(SWITCHING, switching),
    )
    grown = machine
    for part in parts:
        grown = grown.with_part(part)
    wired = grown.with_wires(*STRONG_TIER_WIRES, *ESCALATION_WIRES)
    if fabric is None:
        return wired
    return dataclasses.replace(wired, fabric=fabric)


def with_escalation(
    machine: graph.Machine,
    source: str,
    name: str,
    pool: decoder_settings.DecoderPoolSettings,
    switching: escalation_settings.SwitchingSettings,
    site: str = fabrics.HOST,
) -> graph.Machine:
    """The machine whose source pool escalates on to a pool named name.

    The new pool reads a store of its own, name_buffer, and the switching
    part name_switching decides on the source pool's results; both sit
    at site. An escalated window's rounds come from the source pool's
    store over the hop source_to_name (topology/graph_layout.py). An
    edge that does not leave the entry pool takes the redo window; a
    confidence of two solves there has its source pool solve each
    escalated window once per logical class.
    """
    store_name = f"{name}_buffer"
    switching_name = f"{name}_switching"
    store = syndrome_buffer_module.SyndromeBufferSettings()
    parts = (
        graph.Part(store_name, store, site),
        graph.Part(name, pool, site),
        graph.Part(switching_name, switching),
    )
    wires = (
        graph.Wire(f"{CONTROLLER}.rounds", f"{store_name}.rounds"),
        graph.Wire(f"{store_name}.reads", f"{name}.rounds"),
        graph.Wire(f"{name}.corrections", f"{FRAME}.corrections"),
        graph.Wire(f"{source}.escalate", f"{switching_name}.results"),
        graph.Wire(f"{switching_name}.redecode", f"{name}.escalated"),
    )
    grown = machine
    for part in parts:
        grown = grown.with_part(part)
    return grown.with_wires(*wires)


def with_router(
    machine: graph.Machine,
    rule: Any,
    pools: tuple,
    entry_store: str = WEAK_STORE,
    entry_pool: str = WEAK_POOL,
    site: str = fabrics.FRIDGE,
) -> graph.Machine:
    """The machine whose entry store feeds a router to several entry pools.

    The router, named router, sends each window to the pool its rule
    names (windows/routers.py): the machine's entry pool, first, or one
    of pools, each (name, DecoderPoolSettings) at site, read from the
    entry store over its own hop, entry_store_to_name, and answering the
    frame. by_patch is VQD's decoder per logical qubit (Maurya and
    Tannu 2406.17995v2 Sec. 4.1).
    """
    router_settings = routers.Router.Settings(rule)
    router_part = graph.Part(ROUTER, router_settings)
    rewired = machine.without_wire(
        f"{entry_store}.reads", f"{entry_pool}.rounds"
    )
    grown = rewired.with_part(router_part)
    wires = [
        graph.Wire(f"{entry_store}.reads", f"{ROUTER}.rounds"),
        graph.Wire(f"{ROUTER}.windows", f"{entry_pool}.rounds"),
    ]
    for name, pool in pools:
        part = graph.Part(name, pool, site)
        grown = grown.with_part(part)
        windows = graph.Wire(f"{ROUTER}.windows", f"{name}.rounds")
        corrections = graph.Wire(f"{name}.corrections", f"{FRAME}.corrections")
        wires.extend((windows, corrections))
    return grown.with_wires(*wires)


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


def _baseline(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> graph.Machine:
    """What both bases share, every value written out, no decoder yet.

    The weak store is unbounded and free, and the frame writes in one
    fridge cycle, 4 ns (2605.04892 Table I, lines 1051 and 1063).
    """
    qpu = _stim_qpu(distance, round_period_microseconds)
    rounds_per_shot = BASELINE_ROUNDS_PER_DISTANCE * distance
    workload = memory_workload(
        distance, physical_error_probability, rounds_per_shot
    )
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=1, clock=clocks.FRIDGE_CLOCK
    )
    factory = magic_state_factories.InfiniteFactory.Settings()
    controller = _baseline_controller()
    idle_policy = idle_policies.SeparateDecodeJobsSettings()
    detection_events = _baseline_detection_events()
    windows = _baseline_windows()
    decoder_manager = _baseline_decoder_manager()
    parts = (
        graph.Part(QPU, qpu),
        graph.Part(CONTROLLER, controller),
        graph.Part("idle_policy", idle_policy),
        graph.Part("detection_events", detection_events),
        graph.Part(WINDOWS, windows),
        graph.Part("decoder_manager", decoder_manager),
        graph.Part("workload", workload),
        graph.Part("magic_state_factory", factory),
        graph.Part(FRAME, pauli_frame),
        graph.Part(WEAK_STORE, _UNBOUNDED_STORE, fabrics.FRIDGE),
    )
    observation = _baseline_observation()
    return graph.Machine(
        # the controller and every part that names no clock of its own
        # count their cycles on the machine's clock
        clock=clocks.FRIDGE_CLOCK,
        parts=parts,
        wires=(*LOOP_WIRES, WEAK_STORE_WRITE),
        observation=observation,
    )


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
    first_in_first_out = schedulers.FifoScheduler.Settings()
    return decoder_settings.DecoderManagerSettings(
        scheduler=first_in_first_out,
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
