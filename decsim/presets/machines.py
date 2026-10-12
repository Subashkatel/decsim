"""Machines written as graphs: the decoder bases, and the steps that grow one.

Each base is every part a machine holds, every value written out, around
one decoder: the weak base's PyMatching on a chip unit, the strong base's
belief matching on a host unit. The parts share their names, so
`--set weak.unit_count=2` reaches the weak pool of any of them.

    base               the parts a machine holds once, its loop wired
    with_pool          a machine with a pool reading a store of its own
    with_frame         a machine given a frame that every pool corrects
    weak_baseline      the weak decoder alone, its hops a fridge cycle
    strong_baseline    the strong decoder alone, its hops a cycle each
    strong_alone       a machine's weak pool replaced by a strong one
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

import decsim.config as config
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
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.part_kinds as part_kinds
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.routers as routers
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the parts' names
QPU = "qpu"
CONTROLLER = "controller"
WINDOWS = "windows"
FRAME = "frame"
WEAK_POOL = window_records.WEAK_POOL_ID.word
STRONG_POOL = window_records.STRONG_POOL_ID.word
SWITCHING = "switching"
ROUTER = "router"
# a store goes by its pool's name and this
STORE_SUFFIX = "_buffer"
WEAK_STORE = WEAK_POOL + STORE_SUFFIX
STRONG_STORE = STRONG_POOL + STORE_SUFFIX
_RECORDS_NOTHING_EXTRA = observe_settings.ObservationSettings()
_REFERENCE_FABRIC = fabrics.ReferenceFabric()
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


def base(
    *records: Any,
    clock: Optional[config.Clock] = None,
    observation: observe_settings.ObservationSettings = _RECORDS_NOTHING_EXTRA,
    fabric: ports.FabricRule = _REFERENCE_FABRIC,
) -> graph.Machine:
    """A machine of parts held once, its loop wired, no decoder yet.

    Each record is the part of its kind, named for the kind and the
    frame FRAME; a QPU or a controller not given is its kind's default,
    and any other kind not given runs on its default unnamed. The QPU
    reads out to the controller and takes its pulses, and a frame sends
    the controller its decisions.
    """
    qpu_kind = part_kinds.kind_named(QPU)
    controller_kind = part_kinds.kind_named(CONTROLLER)
    parts = {
        QPU: graph.Part(QPU, qpu_kind.default),
        CONTROLLER: graph.Part(CONTROLLER, controller_kind.default),
    }
    for record in records:
        name = _held_name(record)
        parts[name] = graph.Part(name, record)
    wires = [graph.Wire(f"{QPU}.readout", f"{CONTROLLER}.readout")]
    if FRAME in parts:
        decisions = graph.Wire(f"{FRAME}.decisions", f"{CONTROLLER}.decisions")
        wires.append(decisions)
    pulses = graph.Wire(f"{CONTROLLER}.pulses", f"{QPU}.pulses")
    wires.append(pulses)
    held_parts = parts.values()
    return graph.Machine(
        clock=clock,
        parts=tuple(held_parts),
        wires=tuple(wires),
        fabric=fabric,
        observation=observation,
    )


def with_pool(
    machine: graph.Machine,
    pool: decoder_settings.DecoderPoolSettings,
    name: str = WEAK_POOL,
    site: str = fabrics.FRIDGE,
    store: Optional[ports.SyndromeBufferSettings] = None,
) -> graph.Machine:
    """The machine with a pool that reads a store of its own, name_buffer.

    The controller writes the store and the pool corrects the frame,
    when the machine has one; both sit at site. store None is the
    default buffer.
    """
    if store is None:
        store = syndrome_buffer_module.SyndromeBufferSettings()
    store_name = f"{name}{STORE_SUFFIX}"
    controllers = machine.parts_of("controller")
    controller = controllers[0]
    store_part = graph.Part(store_name, store, site)
    pool_part = graph.Part(name, pool, site)
    grown = machine.with_part(store_part)
    grown = grown.with_part(pool_part)
    wires = [
        graph.Wire(f"{controller.name}.rounds", f"{store_name}.rounds"),
        graph.Wire(f"{store_name}.reads", f"{name}.rounds"),
    ]
    frames = machine.parts_of("pauli_frame")
    if frames:
        corrections = f"{frames[0].name}.corrections"
        correction_wire = graph.Wire(f"{name}.corrections", corrections)
        wires.append(correction_wire)
    return grown.with_wires(*wires)


def with_frame(
    machine: graph.Machine, frame: Optional[Any] = None
) -> graph.Machine:
    """The machine with a frame named FRAME, every pool correcting it.

    frame None is the default frame; the frame sends the controller its
    decisions.
    """
    if frame is None:
        frame = pauli_frame_module.PauliFrameConfig()
    controllers = machine.parts_of("controller")
    controller = controllers[0]
    wires = [graph.Wire(f"{FRAME}.decisions", f"{controller.name}.decisions")]
    for pool in machine.parts_of("decoder_pool"):
        corrections = graph.Wire(
            f"{pool.name}.corrections", f"{FRAME}.corrections"
        )
        wires.append(corrections)
    frame_part = graph.Part(FRAME, frame)
    grown = machine.with_part(frame_part)
    return grown.with_wires(*wires)


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
    machine = _baseline(
        distance, physical_error_probability, round_period_microseconds
    )
    with_weak = with_pool(machine, WEAK_BASELINE_POOL, store=_UNBOUNDED_STORE)
    fabric = link_cards.WEAK_BASELINE_FABRIC
    return dataclasses.replace(with_weak, fabric=fabric)


def strong_baseline(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
) -> graph.Machine:
    """The strong decoder alone.

    Belief matching answers every window on one host unit, charged its
    measured wall clock. Each hop it prices is one cycle on an unbounded
    wire, of the fridge clock on the chip and the room clock on the
    host.
    """
    machine = _baseline(
        distance, physical_error_probability, round_period_microseconds
    )
    with_strong = with_pool(
        machine,
        STRONG_BASELINE_POOL,
        STRONG_POOL,
        fabrics.HOST,
        _UNBOUNDED_STORE,
    )
    fabric = link_cards.STRONG_BASELINE_FABRIC
    return dataclasses.replace(with_strong, fabric=fabric)


def strong_alone(
    machine: graph.Machine, strong_pool: decoder_settings.DecoderPoolSettings
) -> graph.Machine:
    """The machine's weak pool and its store replaced by a strong pool.

    The strong pool reads a store of its own, and both sit on the host.
    """
    without_pool = machine.without_part(WEAK_POOL)
    without_store = without_pool.without_part(WEAK_STORE)
    return with_pool(without_store, strong_pool, STRONG_POOL, fabrics.HOST)


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
    windows = machine.held(WINDOWS)
    switching_windows = window_settings.switching_windows(
        windows, switching.strong_window
    )
    reshaped = machine.with_held(WINDOWS, switching_windows)
    grown = with_escalation(
        reshaped,
        WEAK_POOL,
        STRONG_POOL,
        strong_pool,
        switching,
        switching_name=SWITCHING,
    )
    if fabric is None:
        return grown
    return dataclasses.replace(grown, fabric=fabric)


def with_escalation(
    machine: graph.Machine,
    source: str,
    name: str,
    pool: decoder_settings.DecoderPoolSettings,
    switching: escalation_settings.SwitchingSettings,
    site: str = fabrics.HOST,
    switching_name: Optional[str] = None,
) -> graph.Machine:
    """The machine whose source pool escalates on to a pool named name.

    The new pool reads a store of its own, name_buffer (with_pool), and
    the switching part, name_switching unless switching_name names it,
    decides on the source pool's results; both sit at site. An escalated
    window's rounds come from the source pool's store over the hop
    source_to_name (topology/graph_layout.py). An edge that does not
    leave the entry pool takes the redo window; a confidence of two
    solves there has its source pool solve each escalated window once
    per logical class.
    """
    if switching_name is None:
        switching_name = f"{name}_switching"
    grown = with_pool(machine, pool, name, site)
    switching_part = graph.Part(switching_name, switching)
    grown = grown.with_part(switching_part)
    results = graph.Wire(f"{source}.escalate", f"{switching_name}.results")
    redecode = graph.Wire(f"{switching_name}.redecode", f"{name}.escalated")
    return grown.with_wires(results, redecode)


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

    The frame writes in one fridge cycle, 4 ns (2605.04892 Table I,
    lines 1051 and 1063); each base's store is unbounded and free.
    """
    qpu = _stim_qpu(distance, round_period_microseconds)
    # made at the task, so it follows an assigned qpu.distance
    workload = producers.MemorySettings(
        physical_error_probability,
        rounds_per_distance=BASELINE_ROUNDS_PER_DISTANCE,
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
    observation = _baseline_observation()
    return base(
        qpu,
        controller,
        idle_policy,
        detection_events,
        windows,
        decoder_manager,
        workload,
        factory,
        pauli_frame,
        # the controller and every part that names no clock of its own
        # count their cycles on the machine's clock
        clock=clocks.FRIDGE_CLOCK,
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


def _held_name(record: Any) -> str:
    """The name a part held once goes by: its kind's, the frame FRAME."""
    record_class = type(record)
    kind = part_kinds.kind_of(record_class)
    if kind is None or kind.is_many:
        class_name = record_class.__name__
        message = f"a {class_name} is no part a machine holds once"
        raise graph.GraphError(message)
    if kind.name == "pauli_frame":
        return FRAME
    return kind.name
