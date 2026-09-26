"""Building the two syndrome buffers, the strong receiver and the frame.

The strong syndrome buffer exists only when a tier reads from it, and a
separate controller readout cost needs a link card that leaves that cost
out, a refusal that sits on the one card
it is about. A placement that forms the detection events on the weak
decoder chip needs the run's rounds to land in the weak syndrome buffer,
and is the one row that may carry that chip's formation charge.
"""

import dataclasses

import pytest

import decsim.build.parts as build_parts
import decsim.build.stores as store_build
import decsim.detector_error_model.detection_event_formation as formation
import decsim.engine as engine_module
import decsim.links.settings as link_settings
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.settings as store_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run


class _Policy:
    """The two facts the store build reads off an escalation policy."""

    def __init__(self, *, may_escalate, primary_tier) -> None:
        self.requires_strong_context = may_escalate
        self.primary_tier = primary_tier


def _parts(settings):
    """The fixtures a store seat's builder reads, and nothing else."""
    engine = engine_module.Engine()
    return build_parts.Parts(
        settings=settings,
        engine=engine,
        plan=None,
        escalation_policy=None,
        pool=None,
        detection_events=None,
    )


def test_buffer_zero_is_built_from_the_kind_the_section_names():
    settings = _machine_settings()

    parts = _parts(settings)
    store = store_build.build_weak_syndrome_buffer(parts)

    kind = settings.weak_syndrome_buffer.kind
    assert isinstance(store, syndrome_buffer_module.SYNDROME_BUFFERS[kind])


def test_a_syndrome_buffer_kind_that_names_no_row_is_refused():
    weak_syndrome_buffer = store_settings.SyndromeBufferSettings(kind="tape")
    settings = _machine_settings(weak_syndrome_buffer=weak_syndrome_buffer)

    with pytest.raises(ValueError) as refusal:
        store_build.check_store_kinds(settings)

    assert "weak_syndrome_buffer.kind" in str(refusal.value)


def test_a_room_side_kind_that_names_no_row_is_refused_though_unused():
    """A weak-only run builds no strong syndrome buffer but reads its yaml."""
    strong_syndrome_buffer = store_settings.SyndromeBufferSettings(kind="tape")
    settings = _machine_settings(strong_syndrome_buffer=strong_syndrome_buffer)

    with pytest.raises(ValueError) as refusal:
        store_build.check_store_kinds(settings)

    assert "strong_syndrome_buffer.kind" in str(refusal.value)


def test_a_weak_only_run_reads_nothing_from_the_room_side():
    """No tier reads from the room side, so no seat is built for it."""
    weak_only = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.WEAK
    )

    assert not store_build.uses_strong_store(weak_only)


def test_a_run_that_may_escalate_reads_the_room_side():
    switching = _Policy(
        may_escalate=True, primary_tier=window_records.DecoderTier.WEAK
    )

    assert store_build.uses_strong_store(switching)


def test_a_strong_primary_run_reads_the_room_side():
    strong_primary = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.STRONG
    )

    assert store_build.uses_strong_store(strong_primary)
    assert store_build.uses_the_room_side(strong_primary)


def test_forming_on_the_weak_chip_is_refused_when_no_round_reaches_it():
    """A strong-primary run writes its rounds into the strong store only."""
    settings = _machine_settings()
    strong_primary = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.STRONG
    )
    on_the_weak_chip = formation.WeakSyndromeBufferSideFormation(None, 0)

    with pytest.raises(ValueError) as refusal:
        store_build.check_weak_syndrome_buffer_formation(
            settings, strong_primary, on_the_weak_chip
        )

    sentence = str(refusal.value)
    assert "detection_events_formed_at weak_syndrome_buffer" in sentence
    assert "decoded on the strong tier" in sentence


def test_a_weak_chip_formation_charge_is_refused_under_another_row():
    """A run that forms elsewhere would pay for the conversion twice."""
    charged = store_settings.SyndromeBufferSettings(
        clock=declared_run.DECLARED_CLOCK, detection_event_cycles_per_round=5
    )
    settings = _machine_settings(weak_syndrome_buffer=charged)
    weak_primary = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.WEAK
    )
    at_the_controller = formation.ControllerSideFormation(None, 0)

    with pytest.raises(ValueError) as refusal:
        store_build.check_weak_syndrome_buffer_formation(
            settings, weak_primary, at_the_controller
        )

    sentence = str(refusal.value)
    assert "weak_syndrome_buffer.detection_event_cycles_per_round" in sentence
    assert "a conversion this run does elsewhere" in sentence


def test_a_weak_primary_run_may_form_and_charge_on_the_weak_chip():
    charged = store_settings.SyndromeBufferSettings(
        clock=declared_run.DECLARED_CLOCK, detection_event_cycles_per_round=5
    )
    settings = _machine_settings(weak_syndrome_buffer=charged)
    weak_primary = _Policy(
        may_escalate=False, primary_tier=window_records.DecoderTier.WEAK
    )
    on_the_weak_chip = formation.WeakSyndromeBufferSideFormation(None, 0)

    store_build.check_weak_syndrome_buffer_formation(
        settings, weak_primary, on_the_weak_chip
    )


def test_a_readout_cost_on_the_controller_needs_a_card_that_excludes_it():
    """Otherwise the reference latency charges the same work twice."""
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=False
    )

    with pytest.raises(ValueError) as refusal:
        store_build.check_readout_cost_is_priced(settings)

    sentence = str(refusal.value)
    assert "separate controller readout cost" in sentence
    assert "excludes that cost" in sentence


def test_a_readout_cost_beside_a_card_that_excludes_it_is_allowed():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=6, card_excludes_it=True
    )

    store_build.check_readout_cost_is_priced(settings)


def test_no_readout_cost_asks_nothing_of_the_card():
    settings = _settings_with_readout_cost(
        readout_to_bits_cycles=0, card_excludes_it=False
    )

    store_build.check_readout_cost_is_priced(settings)


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
        store_build.check_readout_cost_is_priced(settings)


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

    store_build.check_readout_cost_is_priced(settings)


def _settings_with_readout_cost(*, readout_to_bits_cycles, card_excludes_it):
    """A machine whose controller readout cost and card are set by hand."""
    controller = declared_run.declared_controller()
    controller = dataclasses.replace(
        controller, readout_to_bits_cycles=readout_to_bits_cycles
    )
    links = declared_run.declared_profile()
    card = links.qpu_to_controller
    links = _links_with_readout_flag(links, card, card_excludes_it)
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


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
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )
    return dataclasses.replace(settings, **changes)
