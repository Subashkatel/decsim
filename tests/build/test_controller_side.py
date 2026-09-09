"""The controller-side build: formation, the factory, and the strong route.

Each of these is one decision the root makes before a round is packed,
and each is checked here rather than inside the running machine, which
is where a yaml's mistake belongs (STYLE.md rule 4).
"""

import pytest

import decsim.build.controller_side as controller_side
import decsim.controller.settings as controller_settings
import decsim.decoders.decoders as decoders
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.settings as machine_settings
import tests.declared_run as declared_run


class _DeviceWithNoFormationTable:
    """A timing-only or synthetic source: it forms nothing."""


class _DeviceThatForms:
    def form_round(self, operation_id, round_index, raw_bits):
        """One round's detection events."""
        del operation_id, round_index
        return tuple(raw_bits)


def _settings(**changes):
    """A weak-only machine settings record with the given changes."""
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller(**changes)
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )


def test_a_source_with_no_formation_table_forms_nothing_either_way():
    settings = _settings()
    device = _DeviceWithNoFormationTable()

    formation = controller_side.build_detection_events(settings, device)

    assert formation.form_round is None


def test_the_row_says_where_a_forming_source_narrows_the_round():
    forms_here = _settings(detection_events_formed_at="controller")
    forms_later = _settings(detection_events_formed_at="decoder")
    device = _DeviceThatForms()

    at_the_controller = controller_side.build_detection_events(
        forms_here, device
    )
    at_the_decoder = controller_side.build_detection_events(forms_later, device)

    assert at_the_controller.at_the_controller is True
    assert at_the_decoder.at_the_controller is False
    assert at_the_controller.form_round == device.form_round


def test_a_formation_row_that_is_not_on_the_table_is_refused():
    settings = _settings(detection_events_formed_at="the fridge")
    device = _DeviceThatForms()

    with pytest.raises(ValueError) as refusal:
        controller_side.build_detection_events(settings, device)

    sentence = str(refusal.value)
    assert "controller.detection_events_formed_at" in sentence
    assert list(controller_settings.DETECTION_EVENT_FORMATION) is not None


def test_the_two_formation_rows_are_the_two_the_key_offers():
    rows = controller_settings.DETECTION_EVENT_FORMATION

    assert sorted(rows) == ["controller", "decoder"]


def test_one_decoder_for_both_tiers_is_refused_when_the_run_may_escalate():
    weak_and_strong_are_one = decoders.PresetLatencyDecoder(10.0)
    router = decoders.SwitchingRouter(
        weak=weak_and_strong_are_one, strong=weak_and_strong_are_one
    )
    may_escalate = _EscalatingPolicy()

    with pytest.raises(ValueError) as refusal:
        controller_side.check_strong_route(may_escalate, router)

    assert "routes to the same decoder as the weak tier" in str(refusal.value)


def test_a_run_that_never_escalates_is_asked_nothing_about_its_route():
    """The check is about a strong re-decode, which such a run never makes."""
    one_decoder = decoders.PresetLatencyDecoder(10.0)
    router = decoders.SwitchingRouter(weak=one_decoder, strong=one_decoder)
    never_escalates = _WeakOnlyPolicy()

    controller_side.check_strong_route(never_escalates, router)


def test_two_distinct_decoders_pass_the_strong_route_check():
    weak = decoders.PresetLatencyDecoder(10.0)
    strong = decoders.PresetLatencyDecoder(30.0)
    router = decoders.SwitchingRouter(weak=weak, strong=strong)

    escalating = _EscalatingPolicy()

    controller_side.check_strong_route(escalating, router)


def test_two_rows_with_different_cards_are_built_by_the_one_call():
    """One constructor signature, so a row's own keys ride in arguments."""
    plan = _PlanWithRoundTicks(1000)
    no_card = qpu_settings.FactorySettings(kind="infinite")
    with_a_card = qpu_settings.FactorySettings(
        kind="distillation",
        arguments={
            "unit_count": 1,
            "attempt_ticks": 1000,
            "correction_round_count": 1,
            "correction_decode_count": 0,
        },
    )

    always_in_stock = controller_side.build_factory(no_card, None, None, plan)
    fifteen_to_one = controller_side.build_factory(
        with_a_card, None, None, plan
    )

    assert isinstance(always_in_stock, magic_state_factories.InfiniteFactory)
    assert isinstance(fifteen_to_one, magic_state_factories.DistillationFactory)


def test_a_factory_kind_that_names_no_row_is_refused():
    plan = _PlanWithRoundTicks(1000)
    settings = qpu_settings.FactorySettings(kind="teleported")

    with pytest.raises(ValueError) as refusal:
        controller_side.build_factory(settings, None, None, plan)

    assert "magic_state_factory.kind" in str(refusal.value)


def test_the_collaborators_record_carries_the_runs_round_and_arguments():
    plan = _PlanWithRoundTicks(1234)
    settings = qpu_settings.FactorySettings(
        kind="infinite", arguments={"count": 2}
    )

    collaborators = magic_state_factories.FactoryCollaborators(
        engine=None,
        decode_service=None,
        round_ticks=plan.round_ticks,
        arguments=settings.arguments,
    )

    assert collaborators.round_ticks == 1234
    assert collaborators.arguments == {"count": 2}


def test_the_process_name_says_which_point_a_trace_is_of():
    settings = _settings()

    name = controller_side.process_name(settings, 7)

    assert name.startswith("decsim ")
    assert " d3 " in name
    assert name.endswith("seed7")


class _EscalatingPolicy:
    """The one fact check_strong_route reads."""

    requires_strong_context = True


class _WeakOnlyPolicy:
    requires_strong_context = False


class _PlanWithRoundTicks:
    """The one fact build_factory reads off the plan."""

    def __init__(self, round_ticks: int) -> None:
        self.round_ticks = round_ticks
