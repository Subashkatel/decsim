"""The controller-side build: formation, the factory, and the strong route.

Each of these is one decision the root makes before a round is packed,
and each is checked here rather than inside the running machine, which
is where a yaml's mistake belongs (STYLE.md rule 4).
"""

import dataclasses
import types

import pytest

import decsim.build.controller_side as controller_side
import decsim.build.parts as build_parts
import decsim.decoders.decoders as decoders
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
import decsim.machine as machine_module
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import tests.declared_run as declared_run


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


def _settings(formed_at=("controller",)):
    """A weak-only machine settings record forming at the given seats."""
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    detection_events = event_settings.DetectionEventSettings(
        formed_at=formed_at
    )
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        detection_events=detection_events,
        pauli_frame=frame,
    )


def _policy(primary_tier="weak", requires_strong_context=False):
    """The two facts the paths of a run are read off."""
    tier = window_records.DecoderTier(primary_tier)
    return types.SimpleNamespace(
        primary_tier=tier, requires_strong_context=requires_strong_context
    )


WEAK_BASELINE = _policy()
SWITCHING = _policy(requires_strong_context=True)
STRONG_ONLY = _policy("strong")


def test_a_source_that_does_not_answer_the_port_forms_nothing():
    settings = _settings()
    device = _DeviceWithNoFormationTable()
    carried = ("a fragment",)

    placement = controller_side.build_detection_events(
        settings, device, WEAK_BASELINE
    )

    assert placement.form_at("controller", carried) is carried


@pytest.mark.parametrize(
    "policy, formed_at",
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
def test_a_seat_list_every_path_crosses_once_is_built(policy, formed_at):
    settings = _settings(formed_at)
    device = _DeviceWithNoFormationTable()

    placement = controller_side.build_detection_events(settings, device, policy)

    assert placement.forms_at(formed_at[0])


@pytest.mark.parametrize(
    "policy, formed_at, crossed",
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
    policy, formed_at, crossed
):
    """None decodes raw outcomes; two form events of events."""
    settings = _settings(formed_at)
    device = _DeviceWithNoFormationTable()

    with pytest.raises(ValueError) as refusal:
        controller_side.build_detection_events(settings, device, policy)

    sentence = str(refusal.value)
    assert f"at {crossed}" in sentence
    assert "crosses exactly one seat" in sentence


def test_the_burst_detector_counts_at_the_primary_tiers_seat():
    """The escalated region's seat forms too, but is not counted twice."""
    settings = _settings(("weak_decoder", "strong_decoder"))
    source = _OneRoundSource()
    detector = _CountingDetector()
    placement = controller_side.build_detection_events(
        settings, source, SWITCHING, detector
    )
    raw = (_OneRoundSource.fragment(),)

    placement.form_at("strong_decoder", raw)
    placement.form_at("weak_decoder", raw)

    assert detector.observed == [1]


def test_a_burst_detector_on_a_source_that_forms_nothing_is_refused():
    settings = _settings()
    device = _DeviceWithNoFormationTable()
    detector = _CountingDetector()

    with pytest.raises(ValueError, match="burst_detector counts detection"):
        controller_side.build_detection_events(
            settings, device, WEAK_BASELINE, detector
        )


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


_DISTILLATION = magic_state_factories.DistillationFactory
_MULTI_LEVEL = magic_state_factories.MultiLevelDistillationFactory


def test_two_rows_with_different_cards_are_built_by_the_one_call():
    """One constructor signature, so a row's own keys ride in its Settings."""
    plan = _PlanWithRoundTicks(1000)
    no_card = qpu_settings.FactorySettings(kind="infinite")
    card = _DISTILLATION.Settings(
        unit_count=1,
        attempt_ticks=1000,
        correction_round_count=1,
        correction_decode_count=0,
    )
    with_a_card = qpu_settings.FactorySettings(
        kind="distillation", row_settings=card
    )

    without_a_card = _parts(no_card, plan)
    from_a_card = _parts(with_a_card, plan)
    always_in_stock = controller_side.build_factory(without_a_card)
    fifteen_to_one = controller_side.build_factory(from_a_card)

    assert isinstance(always_in_stock, magic_state_factories.InfiniteFactory)
    assert isinstance(fifteen_to_one, magic_state_factories.DistillationFactory)


_ONE_LEVEL = magic_state_factories.DistillLevel(unit_count=1, distance=3)
_NO_DECODE_CARD = _DISTILLATION.Settings(
    unit_count=1,
    attempt_ticks=1000,
    correction_round_count=1,
    correction_decode_count=0,
)
_ELEVEN_DECODE_CARD = _DISTILLATION.Settings(
    unit_count=1, attempt_ticks=1000, correction_round_count=1
)
_ONE_LEVEL_CARD = _MULTI_LEVEL.Settings(levels=(_ONE_LEVEL,))
_ROOT_BUILT_FACTORIES = (
    qpu_settings.FactorySettings(kind="infinite"),
    qpu_settings.FactorySettings(
        kind="distillation", row_settings=_NO_DECODE_CARD
    ),
    qpu_settings.FactorySettings(
        kind="distillation", row_settings=_ELEVEN_DECODE_CARD
    ),
    qpu_settings.FactorySettings(
        kind="multi_level", row_settings=_ONE_LEVEL_CARD
    ),
)


@pytest.mark.parametrize("factory_settings", _ROOT_BUILT_FACTORIES)
def test_every_factory_row_builds_and_binds_the_decode_queue_the_root_wires(
    factory_settings,
):
    """A row reads the collaborators it needs and ignores the rest.

    The root binds every row's decode queue to the run's decoder manager,
    so a row whose card asks for no correction decode runs beside it.
    """
    settings = machine_settings.MachineSettings(
        magic_state_factory=factory_settings
    )

    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    assert result.terminal_status == "complete"


def test_a_factory_kind_that_names_no_row_is_refused():
    plan = _PlanWithRoundTicks(1000)
    settings = qpu_settings.FactorySettings(kind="teleported")

    parts = _parts(settings, plan)

    with pytest.raises(ValueError) as refusal:
        controller_side.build_factory(parts)

    assert "magic_state_factory.kind" in str(refusal.value)


def test_the_collaborators_record_carries_the_runs_round_and_the_rows_keys():
    plan = _PlanWithRoundTicks(1234)
    card = _MULTI_LEVEL.Settings(levels=(_ONE_LEVEL,))
    settings = qpu_settings.FactorySettings(
        kind="multi_level", row_settings=card
    )
    parts = _parts(settings, plan)

    factory = controller_side.build_factory(parts)

    assert factory.card is card
    assert factory.preparation_ticks == 2 * 3 * 1234


def test_the_process_name_says_which_point_a_trace_is_of():
    settings = _settings()

    name = controller_side.process_name(settings, 7)

    assert name.startswith("decsim ")
    assert " d3 " in name
    assert name.endswith("seed7")


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
            max_record_span=0,
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


class _EscalatingPolicy:
    """The one fact check_strong_route reads."""

    requires_strong_context = True


class _WeakOnlyPolicy:
    requires_strong_context = False


def _parts(factory_settings, plan):
    """The fixtures the factory's builder reads: its section and the plan."""
    settings = _settings()
    settings = dataclasses.replace(
        settings, magic_state_factory=factory_settings
    )
    parts = build_parts.Parts(
        settings=settings,
        engine=None,
        plan=plan,
        escalation_policy=None,
        pool=None,
        detection_events=None,
    )
    return parts


class _PlanWithRoundTicks:
    """The one fact build_factory reads off the plan."""

    def __init__(self, round_ticks: int) -> None:
        self.round_ticks = round_ticks
