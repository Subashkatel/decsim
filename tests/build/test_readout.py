"""The readout part: where the events form, and the stores it builds.

Every path a round takes to a decoder crosses exactly one seat of the
detection event placement. The strong syndrome buffer exists only when
a tier reads from it, and a separate controller readout cost needs a
link card that leaves that cost out, a refusal that sits on the one card
it is about.
"""

import dataclasses
import pathlib
import shutil

import pytest

import decsim.build.readout as readout_part
import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.experiments.experiment as experiment
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import tests.declared_run as declared_run

THIS_FILE = pathlib.Path(__file__)
CONFIGS = THIS_FILE.parents[2] / "configs"
# the data-movement grid's fourth block, the one with the strong tier
DATA_MOVEMENT_SWITCHING_BLOCK = 3


class _DeviceWithNoFormationTable:
    """A timing-only or synthetic source: it forms nothing."""


class _CountingDetector:
    """A burst detector that records the rounds it is shown."""

    def __init__(self):
        self.observed = []

    def observe_round(self, operation_id, round_index, events):
        """One round's events."""
        del operation_id, events
        self.observed.append(round_index)


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


def test_a_source_that_does_not_answer_the_port_forms_nothing():
    settings = _settings_forming_at()
    device = _DeviceWithNoFormationTable()
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=1,
        bits=None,
        size_bits=8,
        fragment_index=0,
    )
    carried = (fragment,)

    placement = readout_part.build_detection_events(
        settings.detection_events, device, *WEAK_BASELINE
    )

    assert placement.form_at("controller", carried) == carried


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
        settings.detection_events, device, *run_facts
    )

    assert placement.forms_at(formed_at[0])


@pytest.mark.parametrize(
    "run_facts, formed_at, crossed",
    [
        (WEAK_BASELINE, ("strong_decoder",), "[]"),
        (WEAK_BASELINE, ("controller", "weak_decoder"), "['controller', "),
        (SWITCHING, ("weak_decoder",), "[]"),
        (
            SWITCHING,
            ("weak_syndrome_buffer", "strong_syndrome_buffer"),
            "['weak_syndrome_buffer', 'strong_syndrome_buffer']",
        ),
        (STRONG_ONLY, ("weak_syndrome_buffer",), "[]"),
    ],
)
def test_a_path_that_crosses_no_seat_or_two_is_refused(
    run_facts, formed_at, crossed
):
    """None decodes raw outcomes; two form events of events."""
    settings = _settings_forming_at(formed_at)
    device = _DeviceWithNoFormationTable()

    with pytest.raises(ValueError) as refusal:
        readout_part.build_detection_events(
            settings.detection_events, device, *run_facts
        )

    sentence = str(refusal.value)
    assert f"at {crossed}" in sentence
    assert "crosses exactly one seat" in sentence


def test_the_burst_detector_counts_at_the_primary_tiers_seat():
    """The escalated region's seat forms too, but is not counted twice."""
    settings = _settings_forming_at(("weak_decoder", "strong_decoder"))
    source = _OneRoundSource()
    detector = _CountingDetector()
    placement = readout_part.build_detection_events(
        settings.detection_events, source, *SWITCHING, detector
    )
    raw = (_OneRoundSource.fragment(),)

    placement.form_at("strong_decoder", raw)
    placement.form_at("weak_decoder", raw)

    assert detector.observed == [1]


def test_a_burst_detector_on_a_source_that_forms_nothing_is_refused():
    settings = _settings_forming_at()
    device = _DeviceWithNoFormationTable()
    detector = _CountingDetector()

    with pytest.raises(ValueError, match="burst_detector counts detection"):
        readout_part.build_detection_events(
            settings.detection_events, device, *WEAK_BASELINE, detector
        )


def test_the_weak_store_is_the_one_its_settings_build():
    settings = _ported_run_with_a_rate(None)

    machine = machine_module.Machine.build(settings)

    store = machine.readout.weak_syndrome_buffer
    assert type(store) is ported_syndrome_buffer.PortedSyndromeBuffer


# A ported strong store: the default byte FIFO, and AFS's 32-bit word at
# four cycles an access (ported_syndrome_buffer.py).
PORTED_CLOCK = config.Clock(1000)
PORTED_STORES = [
    ported_syndrome_buffer.PortedSyndromeBufferSettings(clock=PORTED_CLOCK),
    ported_syndrome_buffer.PortedSyndromeBufferSettings(
        clock=PORTED_CLOCK, word_bits=32, cycles_per_access=4
    ),
]


@pytest.mark.parametrize("strong_syndrome_buffer", PORTED_STORES)
def test_a_ported_strong_store_is_refused_at_build(strong_syndrome_buffer):
    """Its writes land unbooked, so its reads alone would be priced."""
    settings = _machine_settings(strong_syndrome_buffer=strong_syndrome_buffer)
    planned = _strong_only(settings)

    with pytest.raises(ValueError, match="ported_syndrome_buffer prices"):
        machine_module.Machine.build(planned)


def test_a_ported_strong_store_is_refused_alike_from_yaml_and_python(
    tmp_path,
):
    """Both routes reach the one check, so they refuse in one sentence."""
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    path = configs / "experiments" / "data_movement" / "data_movement.yaml"
    text = path.read_text()
    config = experiment.load_experiment(path)
    switching_block = config.sweep[DATA_MOVEMENT_SWITCHING_BLOCK]
    points = switching_block.points()
    values = dict(points[0])
    values["workload.arguments.physical_error_probability"] = 0.001
    point = config.point_task(values)
    settings = point.settings
    ported_strong = ported_syndrome_buffer.PortedSyndromeBufferSettings()
    python_built = dataclasses.replace(
        settings, strong_syndrome_buffer=ported_strong
    )
    ported_section = "strong_syndrome_buffer:\n  kind: ported_syndrome_buffer\n"
    ported_text = text + ported_section
    path.write_text(ported_text)
    ported_config = experiment.load_experiment(path)
    ported_point = ported_config.point_task(values)
    yaml_built = ported_point.settings

    with pytest.raises(ValueError) as yaml_refusal:
        machine_module.Machine.build(yaml_built)
    with pytest.raises(ValueError) as python_refusal:
        machine_module.Machine.build(python_built)

    yaml_sentence = str(yaml_refusal.value)
    python_sentence = str(python_refusal.value)
    assert yaml_sentence == python_sentence


def test_a_weak_only_run_reads_nothing_from_the_room_side():
    """No tier reads from the room side, so no store is built for it."""
    machine = declared_run.weak_only_run()
    readout = machine.readout

    assert readout.primary_output is readout.weak_output
    assert readout.strong_syndrome_buffer is None
    assert readout.strong_syndrome_round_receiver is None
    assert readout.strong_output is None


def test_a_run_that_may_escalate_reads_the_room_side():
    """The weak tier still decodes the plan's windows from its own store."""
    machine = declared_run.switching_run()
    readout = machine.readout

    assert readout.primary_output is readout.weak_output
    assert readout.strong_syndrome_buffer is not None
    assert readout.strong_syndrome_round_receiver is not None


def test_a_strong_primary_run_reads_the_room_side():
    machine = declared_run.strong_only_run()
    readout = machine.readout

    assert readout.strong_syndrome_buffer is not None
    assert readout.primary_output is readout.strong_output


def test_a_readout_cost_on_the_controller_needs_a_card_that_excludes_it():
    """Otherwise the reference latency charges the same work twice."""
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=False
    )

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(settings)

    sentence = str(refusal.value)
    assert "separate controller readout cost" in sentence
    assert "excludes that cost" in sentence


def test_a_readout_cost_beside_a_card_that_excludes_it_is_allowed():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )

    machine_module.Machine.build(settings)


def test_no_readout_cost_asks_nothing_of_the_card():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=0, card_excludes_it=False
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

    with pytest.raises(ValueError, match="latency excludes that cost"):
        machine_module.Machine.build(settings)


@pytest.mark.parametrize(
    "readout_cycles, excludes_processing", [(0, False), (6, True)]
)
def test_readout_routes_allow_processing_to_be_charged_once(
    readout_cycles: int, excludes_processing: bool
) -> None:
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=readout_cycles, card_excludes_it=True
    )
    path = dataclasses.replace(
        settings.links.qpu_to_controller,
        excludes_receiver_processing=excludes_processing,
    )
    route = link_settings.ReadoutRoute((0,), path)
    links = dataclasses.replace(settings.links, readout_routes=(route,))
    settings = dataclasses.replace(settings, links=links)

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
    decoder = decoders.PresetLatencyDecoder(1.0)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
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

    with pytest.raises(ValueError, match="set its bits_per_cycle to null"):
        machine_module.Machine.build(settings)


def test_a_ported_store_beside_an_unrated_link_is_accepted():
    settings = _ported_run_with_a_rate(None)

    machine_module.Machine.build(settings)


def _strong_only(settings):
    """The settings with the strong tier alone decoding the windows."""
    decoder = decoders.PresetLatencyDecoder(1.0)
    tier = decoder_settings.DecoderSettings(decoder=decoder)
    policy = escalation_policies.StrongOnly(escalation_policies.NO_CONFIDENCE)
    escalation = escalation_settings.EscalationSettings(policy=policy)
    return dataclasses.replace(
        settings, strong_decoder=tier, escalation=escalation
    )


class _OneRoundSource:
    """Recipes of a one-round operation: its one event is its first outcome."""

    def formation_table(self, operation_id):
        """The one table."""
        del operation_id
        recipe = detector_formation.DetectorRecipe(
            detector_index=0,
            round_index=1,
            kind=detector_formation.LayerKind.PREPARATION,
            records=((1, 0),),
            reference_parity=0,
            coordinates=(),
        )
        return detector_formation.FormationTable(
            round_count=1,
            packet_width_by_round={1: 1},
            readout_slot_start=None,
            detectors=(recipe,),
            observables=(),
        )

    @staticmethod
    def fragment():
        """The operation's one round, its outcome set."""
        return round_records.RetainedSyndromeFragment(
            operation_id=1,
            patch_ids=(0,),
            round_index=1,
            bits=(1,),
            size_bits=1,
            fragment_index=0,
        )
