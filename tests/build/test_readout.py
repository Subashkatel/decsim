"""The readout part: where the events form, and the stores it builds.

Every path a round takes to a decoder crosses exactly one seat of the
detection event placement. The strong syndrome buffer exists only when
a tier reads from it, and a separate controller readout cost needs a
link card that leaves that cost out, a refusal that sits on the one card
it is about.
"""

import dataclasses

import pytest

import decsim.build.readout as readout_part
import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as event_settings
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.transfers as transfer_records
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests
import tests.topology.test_graph_layout as layout_tests


class _DeviceWithNoFormationTable:
    """A timing-only or synthetic source: it forms nothing."""


# The paths a round takes to a decoder (Layout.formation_paths), each
# seat named by its part: the weak pool alone, the weak pool escalating
# to the strong one, the strong pool alone.
WEAK_PATH = ("controller", "weak_buffer", "weak")
ESCALATION_PATH = ("controller", "weak_buffer", "strong_buffer", "strong")
STRONG_PATH = ("controller", "strong_buffer", "strong")
WEAK_BASELINE = (WEAK_PATH,)
SWITCHING = (WEAK_PATH, ESCALATION_PATH)
STRONG_ONLY = (STRONG_PATH,)


@pytest.mark.parametrize(
    "paths, formed_at",
    [
        (WEAK_BASELINE, ("controller",)),
        (WEAK_BASELINE, ("weak_buffer",)),
        (SWITCHING, ("weak_buffer",)),
        (SWITCHING, ("weak", "strong")),
        (SWITCHING, ("weak", "strong_buffer")),
        (STRONG_ONLY, ("strong_buffer",)),
    ],
)
def test_a_seat_list_every_path_crosses_once_is_built(paths, formed_at):
    detection_events = event_settings.DetectionEventSettings(
        formed_at=formed_at
    )
    device = _DeviceWithNoFormationTable()

    placement = readout_part.build_detection_events(
        detection_events, None, device, paths
    )

    assert placement.forms_at(formed_at[0])


@pytest.mark.parametrize(
    "paths, formed_at",
    [
        (WEAK_BASELINE, ("strong",)),
        (WEAK_BASELINE, ("controller", "weak")),
        (SWITCHING, ("weak",)),
        (SWITCHING, ("weak_buffer", "strong_buffer")),
        (STRONG_ONLY, ("weak_buffer",)),
    ],
)
def test_a_path_that_crosses_no_seat_or_two_is_refused(paths, formed_at):
    """None decodes raw outcomes; two form events of events."""
    detection_events = event_settings.DetectionEventSettings(
        formed_at=formed_at
    )
    device = _DeviceWithNoFormationTable()

    with pytest.raises(ValueError, match="a path with none decodes raw"):
        readout_part.build_detection_events(
            detection_events, None, device, paths
        )


@pytest.mark.parametrize(
    "paths, formed_at",
    [
        (WEAK_BASELINE, ("weak", "strong")),
        (STRONG_ONLY, ("weak_buffer", "strong")),
    ],
)
def test_a_seat_no_round_crosses_is_refused(paths, formed_at):
    """A seat of a part the machine does not hold would form nothing.

    The refusal lists the seats the rounds do cross, so a typo is fixed
    from the message.
    """
    detection_events = event_settings.DetectionEventSettings(
        formed_at=formed_at
    )
    device = _DeviceWithNoFormationTable()

    with pytest.raises(ValueError, match="which no round crosses") as refusal:
        readout_part.build_detection_events(
            detection_events, None, device, paths
        )

    seats = sorted({seat for path in paths for seat in path})
    assert f"the seats are {seats}" in str(refusal.value)


def test_a_placement_that_names_no_clock_forms_on_the_machines():
    """No preset clock ticks at 300 MHz, so a former on one fails here."""
    detection_events = event_settings.DetectionEventSettings(latency_cycles=5)
    machine_clock = config.Clock.from_megahertz(300.0)
    device = _DeviceWithNoFormationTable()

    placement = readout_part.build_detection_events(
        detection_events, machine_clock, device, WEAK_BASELINE
    )

    assert placement.clock == machine_clock


def test_a_controller_that_names_no_clock_runs_on_the_machines():
    """No preset clock ticks at 300 MHz, so a controller on one fails here."""
    machine_clock = config.Clock.from_megahertz(300.0)
    declared = declared_run.declared_controller()
    controller = dataclasses.replace(declared, clock=None)
    graph = declared_run.weak_only_graph(
        clock=machine_clock, controller=controller
    )

    machine = machine_module.Machine.build(graph, 0)

    assert machine.readout.controller.settings.clock == machine_clock


# A ported strong store: the default byte FIFO, and AFS's 32-bit word at
# four cycles an access (ported_syndrome_buffer.py).
PORTED_CLOCK = config.Clock(1000)
PORTED_STRONG_STORE = ported_syndrome_buffer.PortedSyndromeBufferSettings(
    clock=PORTED_CLOCK
)


def test_a_ported_strong_store_is_refused_at_build():
    """Its writes land unbooked, so its reads alone would be priced."""
    planned = _strong_only(PORTED_STRONG_STORE)

    with pytest.raises(ValueError, match="the strong_buffer takes"):
        machine_module.Machine.build(planned)


# One cost each, set on the plain store: the clock, a write, a read.
CHARGED_STRONG_STORES = [
    syndrome_buffer_module.SyndromeBufferSettings(clock=PORTED_CLOCK),
    syndrome_buffer_module.SyndromeBufferSettings(write_cycles=100),
    syndrome_buffer_module.SyndromeBufferSettings(read_cycles=100),
]


@pytest.mark.parametrize("strong_syndrome_buffer", CHARGED_STRONG_STORES)
def test_a_cost_on_the_strong_store_is_refused_at_build(
    strong_syndrome_buffer,
):
    """Its receiving end books no access, so the cost would go unpaid."""
    planned = _strong_only(strong_syndrome_buffer)

    with pytest.raises(ValueError, match="the strong_buffer stores"):
        machine_module.Machine.build(planned)


def test_a_strong_store_with_only_a_capacity_is_built():
    capacity_only = syndrome_buffer_module.SyndromeBufferSettings(bits=4096)
    planned = _strong_only(capacity_only)

    machine = machine_module.Machine.build(planned)

    assert machine.readout.store_named(machines.STRONG_STORE) is not None


def test_a_ported_strong_store_is_refused():
    """The strong store books no port access, so a ported card is refused."""
    ported_strong = ported_syndrome_buffer.PortedSyndromeBufferSettings()

    sentence = _strong_store_refusal(ported_strong)

    assert "not PortedSyndromeBufferSettings" in sentence


def test_a_strong_store_cost_is_refused():
    """The strong store charges nothing, so a cost set on it is refused."""
    costed_strong = syndrome_buffer_module.SyndromeBufferSettings(
        write_cycles=3
    )

    sentence = _strong_store_refusal(costed_strong)

    assert "which only the store the controller writes charges" in sentence


def _strong_store_refusal(strong_store) -> str:
    """The build refusal of a switching task given the strong store."""
    switching = shape_tests.weak_base_switching(3, 0.001, 1.0)
    stored = switching.with_record(machines.STRONG_STORE, strong_store)
    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(stored)
    return str(refusal.value)


def test_a_weak_only_run_reads_nothing_from_the_room_side():
    """No tier reads from the room side, so no store is built for it."""
    machine = declared_run.weak_only_run()
    readout = machine.readout

    weak = readout.store_parts(machines.WEAK_STORE)
    strong = readout.store_parts(machines.STRONG_STORE)

    assert readout.primary_output is weak.output
    assert strong is None


def test_a_readout_cost_on_the_controller_needs_a_card_that_excludes_it():
    """Otherwise the reference latency charges the same work twice."""
    graph = _graph_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=False
    )

    with pytest.raises(ValueError, match="a separate controller readout cost"):
        machine_module.Machine.build(graph)


def test_a_readout_cost_beside_a_card_that_excludes_it_is_allowed():
    graph = _graph_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )

    machine_module.Machine.build(graph)


@pytest.mark.parametrize(
    "processing_exclusions", [(False, True), (True, False)]
)
def test_every_readout_route_must_exclude_separately_charged_processing(
    processing_exclusions: tuple[bool, bool],
) -> None:
    """Each route's path carries the flag the controller's cost reads.

    A route is taken by a readout's whole footprint, so the two routes
    sit on the patches of two memory operations.
    """
    graph = _graph_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2),
    ]
    workload = declared_run.declared_workload(operations, 6)
    graph = graph.with_held("workload", workload)
    layout = layout_tests.laid_out(graph)
    readout = layout.card.path_settings(transfer_records.QPU_TO_CONTROLLER)
    first_path = dataclasses.replace(
        readout, excludes_receiver_processing=processing_exclusions[0]
    )
    second_path = dataclasses.replace(
        readout, excludes_receiver_processing=processing_exclusions[1]
    )
    first_route = link_settings.ReadoutRoute((1,), first_path)
    second_route = link_settings.ReadoutRoute((2,), second_path)
    routed = dataclasses.replace(
        graph, readout_routes=(first_route, second_route)
    )

    with pytest.raises(ValueError, match="a separate controller readout cost"):
        machine_module.Machine.build(routed)


def _graph_with_readout_cost(*, readout_to_bits_cycles, card_excludes_it):
    """A weak-only machine whose controller readout cost and card are set."""
    controller = declared_run.declared_controller()
    controller = dataclasses.replace(
        controller, readout_to_bits_cycles=readout_to_bits_cycles
    )
    graph = declared_run.weak_only_graph(controller=controller)
    fabric = graph.fabric
    readout = dataclasses.replace(
        fabric.readout, excludes_receiver_processing=card_excludes_it
    )
    flagged = dataclasses.replace(fabric, readout=readout)
    return dataclasses.replace(graph, fabric=flagged)


def _ported_run_with_a_rate(bits_per_microsecond):
    """A weak-only machine on a ported store, its read link rated."""
    storage_clock = config.Clock(1000)
    ported = ported_syndrome_buffer.PortedSyndromeBufferSettings(
        clock=storage_clock
    )
    graph = declared_run.weak_only_graph(weak_syndrome_buffer=ported)
    fabric = graph.fabric
    read = dataclasses.replace(
        fabric.read_written, capacity_bits_per_microsecond=bits_per_microsecond
    )
    rated = dataclasses.replace(fabric, read_written=read)
    return dataclasses.replace(graph, fabric=rated)


def test_a_rate_out_of_a_ported_store_is_refused_as_a_second_price():
    """The store's read port prices the bits; the link keeps its latency."""
    graph = _ported_run_with_a_rate(2000.0)

    with pytest.raises(ValueError, match="weak_buffer kind"):
        machine_module.Machine.build(graph)


def _strong_only(strong_store):
    """The strong pool alone decoding the windows, on this strong store."""
    decoder = decoders.PresetLatencyDecoder.Settings(1.0)
    tier = decoder_settings.DecoderPoolSettings(
        algorithm=decoder, engine=declared_run.DECLARED_ENGINE
    )
    weak_only = declared_run.weak_only_graph()
    strong_alone = machines.strong_alone(weak_only, tier)
    return strong_alone.with_record(machines.STRONG_STORE, strong_store)
