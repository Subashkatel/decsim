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
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests


class _DeviceWithNoFormationTable:
    """A timing-only or synthetic source: it forms nothing."""


def _settings_forming_at(formed_at=("controller",)):
    """A weak-only machine's settings, forming its events at the seats."""
    detection_events = event_settings.DetectionEventSettings(
        formed_at=formed_at
    )
    return _machine_settings(detection_events=detection_events)


# The two facts the paths of a run are read off: the tier that decodes
# the plan's windows, and whether regions escalate to the strong tier.
WEAK_BASELINE = (window_records.DecoderTier.WEAK, False)
SWITCHING = (window_records.DecoderTier.WEAK, True)
STRONG_ONLY = (window_records.DecoderTier.STRONG, False)


@pytest.mark.parametrize(
    "run_facts, formed_at",
    [
        (WEAK_BASELINE, ("controller",)),
        (WEAK_BASELINE, ("weak_syndrome_buffer",)),
        (WEAK_BASELINE, ("weak_decoder", "strong_decoder")),
        (SWITCHING, ("weak_syndrome_buffer",)),
        (SWITCHING, ("weak_decoder", "strong_decoder")),
        (SWITCHING, ("weak_decoder", "strong_syndrome_buffer")),
        (STRONG_ONLY, ("strong_syndrome_buffer",)),
        (STRONG_ONLY, ("weak_syndrome_buffer", "strong_decoder")),
    ],
)
def test_a_seat_list_every_path_crosses_once_is_built(run_facts, formed_at):
    settings = _settings_forming_at(formed_at)
    device = _DeviceWithNoFormationTable()

    placement = readout_part.build_detection_events(
        settings.detection_events, None, device, *run_facts
    )

    assert placement.forms_at(formed_at[0])


@pytest.mark.parametrize(
    "run_facts, formed_at",
    [
        (WEAK_BASELINE, ("strong_decoder",)),
        (WEAK_BASELINE, ("controller", "weak_decoder")),
        (SWITCHING, ("weak_decoder",)),
        (SWITCHING, ("weak_syndrome_buffer", "strong_syndrome_buffer")),
        (STRONG_ONLY, ("weak_syndrome_buffer",)),
    ],
)
def test_a_path_that_crosses_no_seat_or_two_is_refused(run_facts, formed_at):
    """None decodes raw outcomes; two form events of events."""
    settings = _settings_forming_at(formed_at)
    device = _DeviceWithNoFormationTable()

    with pytest.raises(ValueError, match="a path with none decodes raw"):
        readout_part.build_detection_events(
            settings.detection_events, None, device, *run_facts
        )


def test_a_placement_that_names_no_clock_forms_on_the_machines():
    """No preset clock ticks at 300 MHz, so a former on one fails here."""
    detection_events = event_settings.DetectionEventSettings(latency_cycles=5)
    machine_clock = config.Clock.from_megahertz(300.0)
    device = _DeviceWithNoFormationTable()

    placement = readout_part.build_detection_events(
        detection_events, machine_clock, device, *WEAK_BASELINE
    )

    assert placement.clock == machine_clock


def test_a_controller_that_names_no_clock_runs_on_the_machines():
    """No preset clock ticks at 300 MHz, so a controller on one fails here."""
    machine_clock = config.Clock.from_megahertz(300.0)
    declared = declared_run.declared_controller()
    controller = dataclasses.replace(declared, clock=None)
    settings = _machine_settings(clock=machine_clock, controller=controller)

    machine = machine_module.Machine.build(settings, 0)

    assert machine.readout.controller.settings.clock == machine_clock


# A ported strong store: the default byte FIFO, and AFS's 32-bit word at
# four cycles an access (ported_syndrome_buffer.py).
PORTED_CLOCK = config.Clock(1000)
PORTED_STRONG_STORE = ported_syndrome_buffer.PortedSyndromeBufferSettings(
    clock=PORTED_CLOCK
)


def test_a_ported_strong_store_is_refused_at_build():
    """Its writes land unbooked, so its reads alone would be priced."""
    settings = _machine_settings(strong_syndrome_buffer=PORTED_STRONG_STORE)
    planned = _strong_only(settings)

    with pytest.raises(ValueError, match="the strong syndrome buffer takes"):
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
    settings = _machine_settings(strong_syndrome_buffer=strong_syndrome_buffer)
    planned = _strong_only(settings)

    with pytest.raises(ValueError, match="the strong syndrome buffer stores"):
        machine_module.Machine.build(planned)


def test_a_strong_store_with_only_a_capacity_is_built():
    capacity_only = syndrome_buffer_module.SyndromeBufferSettings(bits=4096)
    settings = _machine_settings(strong_syndrome_buffer=capacity_only)
    planned = _strong_only(settings)

    machine = machine_module.Machine.build(planned)

    assert machine.readout.strong_syndrome_buffer is not None


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

    assert "which only the weak syndrome buffer charges" in sentence


def _strong_store_refusal(strong_store) -> str:
    """The build refusal of a switching task given the strong store."""
    settings = shape_tests.weak_base_switching(3, 0.001, 1.0)
    settings = dataclasses.replace(
        settings, strong_syndrome_buffer=strong_store
    )
    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(settings)
    return str(refusal.value)


def test_a_weak_only_run_reads_nothing_from_the_room_side():
    """No tier reads from the room side, so no store is built for it."""
    machine = declared_run.weak_only_run()
    readout = machine.readout

    assert readout.primary_output is readout.weak_output
    assert readout.strong_syndrome_buffer is None
    assert readout.strong_syndrome_round_receiver is None
    assert readout.strong_output is None


def test_a_readout_cost_on_the_controller_needs_a_card_that_excludes_it():
    """Otherwise the reference latency charges the same work twice."""
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=False
    )

    with pytest.raises(ValueError, match="a separate controller readout cost"):
        machine_module.Machine.build(settings)


def test_a_readout_cost_beside_a_card_that_excludes_it_is_allowed():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )

    machine_module.Machine.build(settings)


@pytest.mark.parametrize(
    "processing_exclusions", [(False, True), (True, False)]
)
def test_every_readout_route_must_exclude_separately_charged_processing(
    processing_exclusions: tuple[bool, bool],
) -> None:
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )
    first_path = dataclasses.replace(
        settings.links.qpu_to_controller,
        excludes_receiver_processing=processing_exclusions[0],
    )
    second_path = dataclasses.replace(
        settings.links.qpu_to_controller,
        excludes_receiver_processing=processing_exclusions[1],
    )
    first_route = link_settings.ReadoutRoute((0,), first_path)
    second_route = link_settings.ReadoutRoute((1,), second_path)
    links = dataclasses.replace(
        settings.links, readout_routes=(first_route, second_route)
    )
    settings = dataclasses.replace(settings, links=links)

    with pytest.raises(ValueError, match="a separate controller readout cost"):
        machine_module.Machine.build(settings)


def _settings_with_readout_cost(*, readout_to_bits_cycles, card_excludes_it):
    """A machine whose controller readout cost and card are set by hand."""
    controller = declared_run.declared_controller()
    controller = dataclasses.replace(
        controller, readout_to_bits_cycles=readout_to_bits_cycles
    )
    links = declared_run.declared_profile()
    card = links.qpu_to_controller
    links = _links_with_readout_flag(links, card, card_excludes_it)
    return _machine_settings(controller=controller, links=links)


def _links_with_readout_flag(links, card, excludes_receiver_processing):
    """The fabric settings with one card's flag replaced."""
    changed_card = dataclasses.replace(
        card, excludes_receiver_processing=excludes_receiver_processing
    )
    return dataclasses.replace(links, qpu_to_controller=changed_card)


def _machine_settings(**changes):
    """A weak-only machine's settings, with the named sections replaced."""
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    decoder = decoders.PresetLatencyDecoder.Settings(1.0)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=decoder, engine=declared_run.DECLARED_ENGINE
    )
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
        weak_decoder=weak_decoder,
    )
    return dataclasses.replace(settings, **changes)


def _ported_run_with_a_rate(bits_per_microsecond):
    """A weak-only machine on a ported store, its read link rated or not."""
    links = declared_run.declared_profile()
    path = links.weak_buffer_to_weak_decoder
    capacity = None
    if bits_per_microsecond is not None:
        capacity = link_settings.CapacitySettings(bits_per_microsecond, "card")
    channel = dataclasses.replace(path.channel, capacity=capacity)
    path = dataclasses.replace(path, channel=channel)
    links = dataclasses.replace(links, weak_buffer_to_weak_decoder=path)
    storage_clock = config.Clock(1000)
    ported = ported_syndrome_buffer.PortedSyndromeBufferSettings(
        clock=storage_clock
    )
    return _machine_settings(weak_syndrome_buffer=ported, links=links)


def test_a_rate_out_of_a_ported_store_is_refused_as_a_second_price():
    """The store's read port prices the bits; the link keeps its latency."""
    settings = _ported_run_with_a_rate(2000.0)

    with pytest.raises(ValueError, match="weak_syndrome_buffer kind"):
        machine_module.Machine.build(settings)


def _strong_only(settings):
    """The settings with the strong tier alone decoding the windows."""
    decoder = decoders.PresetLatencyDecoder.Settings(1.0)
    tier = decoder_settings.DecoderPoolSettings(
        algorithm=decoder, engine=declared_run.DECLARED_ENGINE
    )
    return dataclasses.replace(settings, weak_decoder=None, strong_decoder=tier)
