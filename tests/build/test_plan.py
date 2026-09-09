"""The run plan: the rows the yaml names and the defaults the plan derives.

I7 Part 1 slice 2 gave two window keys a null default whose meaning the
plan derives from the escalation row's declared facts, and slice 1
replaced every kind-string branch in this module with a declared fact.
Both are pinned here through build_plan, on settings shaped as a yaml
would leave them.
"""

import pytest

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.trace_source as trace_source
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run


def _plan(*, escalation=None, windows=None):
    """The plan of a six-round memory run with the given two sections."""
    if escalation is None:
        escalation = escalation_settings.EscalationSettings()
    if windows is None:
        windows = window_settings.WindowSettings()
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
        escalation=escalation,
        windows=windows,
    )
    policy = escalation_build.build_escalation_policy(escalation)
    return plan_build.build_plan(settings, policy)


def _switching():
    """A switching section with a fixed threshold and a gap signal."""
    return escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
    )


def test_a_run_that_never_escalates_gets_the_flush_tail():
    plan = _plan()

    assert plan.scheme.terminal_policy == "flush"
    assert plan.scheme.has_trailing_tail_context is False


def test_a_run_that_may_escalate_gets_the_lookahead_tail():
    """A strong recovery reads past the last window's commit."""
    switching = _switching()

    plan = _plan(escalation=switching)

    assert plan.scheme.terminal_policy == "lookahead"
    assert plan.scheme.has_trailing_tail_context is True


def test_a_terminal_policy_written_in_the_section_wins_over_the_default():
    windows = window_settings.WindowSettings(terminal_policy="lookahead")

    plan = _plan(windows=windows)

    assert plan.scheme.terminal_policy == "lookahead"


def test_a_run_that_never_escalates_ships_boundaries_eagerly():
    plan = _plan()

    assert isinstance(plan.boundary_policy, boundary_policies.Eager)


def test_a_serial_escalation_holds_its_boundaries():
    """Descendants wait out an escalation, so nothing ships until final."""
    switching = _switching()

    plan = _plan(escalation=switching)

    assert isinstance(plan.boundary_policy, boundary_policies.Held)


def test_a_boundaries_row_written_in_the_section_wins_over_the_default():
    windows = window_settings.WindowSettings(boundaries="held")

    plan = _plan(windows=windows)

    assert isinstance(plan.boundary_policy, boundary_policies.Held)


def test_the_workload_row_declares_whether_the_run_has_a_frontend():
    """The plan reads the declared fact, not the kind string."""
    rows = workload_settings.WORKLOADS

    for kind, row in rows.items():
        assert isinstance(row.has_frontend, bool), kind


def test_a_windows_kind_that_names_no_row_is_refused():
    windows = window_settings.WindowSettings(kind="diagonal")

    with pytest.raises(ValueError) as refusal:
        _plan(windows=windows)

    assert "windows.kind" in str(refusal.value)


def test_a_scheme_that_declares_none_of_the_three_facts_is_refused():
    """Every row answers what the plan and the policy read off it."""
    silent = _SilentScheme()
    windows = window_settings.WindowSettings(scheme=silent)

    with pytest.raises(ValueError) as refusal:
        _plan(windows=windows)

    assert "has_trailing_tail_context" in str(refusal.value)


class _SilentScheme:
    """A scheme row that declares nothing the plan reads."""

    def plan_operation(self, *arguments, **sizes):
        """Never reached: the plan refuses this row first."""
        del arguments, sizes
        raise AssertionError("the plan should have refused this row")


# ---- the default boundary row lives on the rows that decide it


class _OutsideBoundaryPolicy:
    """A boundary policy row written outside decsim; it ships when final."""

    ships_provisional_boundaries = False

    def on_commit(self, window, *, final: bool) -> bool:
        """Ship when the committing result is final."""
        del window
        return final


class _OutsideEscalation:
    """An escalation row written outside decsim that never escalates."""

    decides_on_a_confidence = False
    requires_strong_context = False
    primary_tier = window_records.DecoderTier.WEAK
    default_boundary_policy = "outside_boundaries"

    def check_plan(self, plan) -> None:
        """Every run shape is served."""
        del plan

    def tiers_for_ready_window(self, window) -> tuple:
        """The weak tier alone."""
        del window
        return (window_records.DecoderTier.WEAK,)

    def verdict_for_weak_result(self, job, result):
        """Every result is final."""
        del job
        del result
        return decoding_records.Verdict.KEEP

    def learn_from_strong_result(self, window_key, result) -> None:
        """Nothing is learned."""
        del window_key
        del result


class _OutsideShape:
    """A strong window shape written outside decsim that absorbs nothing."""

    absorbs_weak_windows = False
    pins_the_far_face = False
    default_boundary_policy = "outside_boundaries"
    window_absorbed = trace_source.SILENT

    def __init__(self, collaborators) -> None:
        del collaborators

    def plan(self, weak_job):
        """Never reached: the test builds the plan and runs nothing."""
        del weak_job
        raise AssertionError("the test never escalates a window")


def _outside_shape_switching():
    """A switching section whose strong window is the outside shape."""
    return escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
        strong_window="outside_shape",
    )


def test_an_outside_escalation_row_names_its_own_default_boundary_row(
    monkeypatch,
):
    """A row that never escalates names the default; plan.py holds none.

    The default used to be the literal "eager" inside plan.py, so no row
    of BOUNDARY_POLICIES written outside decsim could ever be it.
    """
    monkeypatch.setitem(
        window_settings.BOUNDARY_POLICIES,
        "outside_boundaries",
        _OutsideBoundaryPolicy,
    )
    policy = _OutsideEscalation()
    escalation = escalation_settings.EscalationSettings(policy=policy)

    plan = _plan(escalation=escalation)

    assert isinstance(plan.boundary_policy, _OutsideBoundaryPolicy)


def test_an_outside_strong_window_row_names_the_escalations_default(
    monkeypatch,
):
    """A row that may escalate leaves the default to its strong window."""
    monkeypatch.setitem(
        window_settings.BOUNDARY_POLICIES,
        "outside_boundaries",
        _OutsideBoundaryPolicy,
    )
    monkeypatch.setitem(
        escalation_settings.STRONG_WINDOW_SHAPES, "outside_shape", _OutsideShape
    )
    switching = _outside_shape_switching()

    plan = _plan(escalation=switching)

    assert isinstance(plan.boundary_policy, _OutsideBoundaryPolicy)
