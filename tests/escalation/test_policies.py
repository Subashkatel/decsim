"""The escalation policies: one law per port method, and the paper's identity.

Toshio et al. 2510.25222 Sec. IV C: the switching rate is the mass of the
window-gap distribution below the threshold, so on one run the windows
escalated equal the recorded weak gaps below g_th equal the strong
corrections in the frame (one identity, checked here on its threshold and
its complementary-gap signal, which the weak decoder computes from its
own two forced-class solves). The port is gem5's conditional predictor
(src/cpu/pred/conditional.hh): a row answers and is told; the root acts.
"""

import dataclasses
import math

import pytest
import stim

import decsim.build.escalation as escalation_build
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.policies as policies
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.front.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import tests.escalation.declared_fabric as fabric
import tests.front.yaml_configs as yaml_configs

SOURCE = decoding_records.SoftOutputSource(
    method="matching-gap",
    cluster_origin="decoder",
    growth_schedule="uniform",
    gap_units="natural-log",
    correction="minimum-weight",
    weight_step_natural_log=1.0,
    references=(),
)
OTHER_SOURCE = decoding_records.SoftOutputSource(
    method="cluster-gap",
    cluster_origin="union-find",
    growth_schedule="uniform",
    gap_units="decibels",
    correction="none",
    weight_step_natural_log=0.1,
    references=(),
)
WINDOW = window_records.Window(
    operation_id=1,
    window_index=1,
    commit_lo=4,
    commit_hi=6,
    buffer_hi=9,
    round_count=6,
)
JOB = decoding_records.DecodeJob(operation_id=1, window_id=1, round_count=6)
WEAK_TIER = (window_records.DecoderTier.WEAK,)
STRONG_TIER = (window_records.DecoderTier.STRONG,)
BOTH_TIERS = (
    window_records.DecoderTier.WEAK,
    window_records.DecoderTier.STRONG,
)


def _result(gap, source=SOURCE) -> decoding_records.DecodeResult:
    soft_output = None
    if gap is not None:
        soft_output = decoding_records.SoftOutput(gap=gap, source=source)
    return decoding_records.DecodeResult(
        1, 1, logical_observables=(0,), soft_output=soft_output
    )


def _switching(**arguments) -> policies.Switching:
    fixed = threshold_sources.FixedThreshold(2.0)
    collaborators = policies.EscalationCollaborators(
        threshold=fixed, expected_source=SOURCE, **arguments
    )
    return policies.Switching(collaborators)


def _always_auditing_online_threshold(
    *, threshold: float
) -> threshold_sources.OnlineThreshold:
    """An online source that audits every kept window (audit rate 1.0)."""
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=0.0, threshold=threshold, step=0.0
    )
    audit = threshold_sources.AuditLane(audit_rate=1.0)
    adjustment = threshold_sources.TargetAdjustment(
        kept_bad_budget=0.5,
        adjust_factor=2.0,
        min_escalation_rate=1e-5,
        max_escalation_rate=0.9,
    )
    controller = threshold_sources.OnlineThresholdController(
        tracker, audit, adjustment
    )
    draws = _Draws()
    return threshold_sources.OnlineThreshold(controller, draws)


# ---- the paper's identity, Sec. IV C


def test_escalations_equal_gaps_below_the_threshold_equal_strong_frame_writes():
    """One d=3, 30-round memory shot at p=0.005 and g_th = 15 dB (seed 1)."""
    threshold_nats = 15.0 * math.log(10.0) / 10.0
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=30,
        distance=3,
        after_clifford_depolarization=0.005,
        before_round_data_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(30)
    workload = workload_settings.WorkloadSettings(
        operations=[operation], rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=3, device=device)
    lookahead = window_records.WindowingSchemeCard(terminal_policy="lookahead")
    scheme = sliding_scheme.SlidingWindowScheme(lookahead)
    held = boundary_policies.Held()
    windows = window_settings.WindowSettings(
        scheme=scheme, boundary_policy=held
    )
    decoder_manager = decoder_settings.DecoderManagerSettings()
    weak_decoder = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=100.0
    )
    strong_decoder = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=100.0
    )
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        gap_threshold_decibels=15.0,
        gap_threshold_nats=threshold_nats,
    )
    pauli_frame = pauli_frame_module.PauliFrameConfig(commit_microseconds=0.004)
    observation = observe_settings.ObservationSettings(
        record_switching_windows=True
    )
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        decoder_manager=decoder_manager,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        escalation=escalation,
        pauli_frame=pauli_frame,
        observation=observation,
    )
    machine = machine_module.Machine.build(settings, 1)
    machine.run()
    weak_gaps = []
    for record in machine.observation.decode_records.requests:
        is_weak = record.request_key.tier is window_records.DecoderTier.WEAK
        if is_weak and record.soft_output is not None:
            weak_gaps.append(record.soft_output.gap)
    # one gap per window: the companion forced-class request carries none
    below = 0
    for gap in weak_gaps:
        if gap < threshold_nats:
            below += 1
    strong_frame_writes = 0
    frame = machine.pauli_frame.snapshot()
    for record in frame.records:
        if record.tier == "strong":
            strong_frame_writes += 1
    assert len(weak_gaps) == 10
    assert below == 3
    assert machine.decoder_manager.strong_requests.counts.needed == 3
    assert strong_frame_writes == 3


# ---- one law per port method


def test_baseline_keeps_every_weak_result():
    baseline = policies.Baseline(policies.NO_CONFIDENCE)
    assert baseline.primary_tier is window_records.DecoderTier.WEAK
    assert baseline.requires_strong_context is False
    assert baseline.tiers_for_ready_window(WINDOW) == WEAK_TIER
    unsure = _result(0.0)
    verdict = baseline.verdict_for_weak_result(JOB, unsure)
    assert verdict is decoding_records.Verdict.KEEP


def test_strong_only_decodes_every_window_on_the_strong_tier_once():
    strong_only = policies.StrongOnly(policies.NO_CONFIDENCE)
    assert strong_only.primary_tier is window_records.DecoderTier.STRONG
    assert strong_only.requires_strong_context is False
    assert strong_only.tiers_for_ready_window(WINDOW) == STRONG_TIER
    timing_only = _result(None)
    verdict = strong_only.verdict_for_weak_result(JOB, timing_only)
    assert verdict is decoding_records.Verdict.KEEP


def test_switching_keeps_at_the_threshold_and_escalates_below_it():
    switching = _switching()
    assert switching.tiers_for_ready_window(WINDOW) == WEAK_TIER
    at_threshold = _result(2.0)
    below_threshold = _result(1.999)
    timing_only = _result(None)
    kept = switching.verdict_for_weak_result(JOB, at_threshold)
    escalated = switching.verdict_for_weak_result(JOB, below_threshold)
    unsure = switching.verdict_for_weak_result(JOB, timing_only)
    assert kept is decoding_records.Verdict.KEEP
    assert escalated is decoding_records.Verdict.ESCALATE
    assert unsure is decoding_records.Verdict.ESCALATE


def test_switching_decodes_both_tiers_at_once_when_asked():
    parallel = _switching(run_both_at_once=True)
    assert parallel.tiers_for_ready_window(WINDOW) == BOTH_TIERS


def test_a_soft_output_from_another_signal_is_refused_with_a_sentence():
    switching = _switching()
    mismatched = _result(5.0, source=OTHER_SOURCE)
    with pytest.raises(
        ValueError,
        match="decoder confidence source does not match the switching "
        "threshold source",
    ):
        switching.verdict_for_weak_result(JOB, mismatched)


def test_a_strong_result_teaches_the_online_source():
    online = _always_auditing_online_threshold(threshold=0.0)
    collaborators = policies.EscalationCollaborators(
        threshold=online, expected_source=SOURCE
    )
    switching = policies.Switching(collaborators)
    confident = _result(5.0)
    # the kept window is audited, so its verdict escalates
    verdict = switching.verdict_for_weak_result(JOB, confident)
    assert verdict is decoding_records.Verdict.ESCALATE
    revised = decoding_records.DecodeResult(1, 1, logical_observables=(1,))
    switching.learn_from_strong_result((1, 1), revised)
    assert online.controller.raise_count == 1


def test_a_plan_that_contradicts_itself_is_refused_at_build_with_a_sentence():
    with pytest.raises(
        ValueError, match="the two policies contradict; pick one"
    ):
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            strong_window="forward",
            run_both_at_once=True,
        )


def test_a_refusal_names_the_strong_window_row_the_yaml_chose():
    """Four rows reach these refusals, so none of them may name one row.

    The run shape carries escalation.strong_window (RunShape), and the
    sentence a user reads names the row their yaml asked for.
    """
    with pytest.raises(
        ValueError,
        match="escalation.strong_window forward_seam_pinned defers the "
        "strong start until the far weak boundary exists",
    ):
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            strong_window="forward_seam_pinned",
            run_both_at_once=True,
        )


def _forward_window_settings(
    commit_rounds: int, buffer_rounds: int
) -> machine_settings.MachineSettings:
    """The gate's switching card with the forward window and the sizes.

    Every part is a table row, the way the yaml builds it.
    """
    windows = window_settings.WindowSettings(
        commit_rounds=commit_rounds, buffer_rounds=buffer_rounds
    )
    weak_decoder = decoder_settings.DecoderSettings(
        kind="pymatching", engine_megahertz=100.0
    )
    strong_decoder = decoder_settings.DecoderSettings(
        kind="belief_matching", engine_megahertz=100.0
    )
    escalation = escalation_settings.EscalationSettings(
        kind="switching", gap_threshold_nats=1.0, strong_window="forward"
    )
    return machine_settings.MachineSettings(
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        escalation=escalation,
    )


@pytest.mark.parametrize("commit_rounds, buffer_rounds", [(3, 4), (4, 3)])
def test_a_forward_window_crossing_a_later_commit_region_is_refused(
    commit_rounds, buffer_rounds
):
    """The slice note's ruling 5: the crossing shape is decided at build.

    The strong region is commit plus two buffers; when twice the buffer
    is not a multiple of the commit, it ends inside a later window's
    commit region, and the run used to die at its first escalation.
    """
    settings = _forward_window_settings(commit_rounds, buffer_rounds)
    with pytest.raises(
        ValueError,
        match="the strong region of forward, commit plus two buffers, "
        "must end inside its own commit region",
    ):
        machine_module.Machine.build(settings, 0)


@pytest.mark.parametrize("commit_rounds, buffer_rounds", [(3, 3), (4, 4)])
def test_a_forward_window_ending_on_a_commit_edge_builds(
    commit_rounds, buffer_rounds
):
    settings = _forward_window_settings(commit_rounds, buffer_rounds)
    machine = machine_module.Machine.build(settings, 0)
    assert machine.window_manager.strong_redecode is not None


def test_an_online_source_under_a_forward_window_is_refused_as_serial():
    """check_plan's serial-only law.

    An audit label compares one window's weak and strong committed
    observables, and a forward-window strong result owns a larger extent
    than the audited window.
    """
    online = _always_auditing_online_threshold(threshold=2.0)
    collaborators = policies.EscalationCollaborators(
        threshold=online, expected_source=decoders.SAMPLED_CONFIDENCE_SOURCE
    )
    policy = policies.Switching(collaborators)
    escalation = escalation_settings.EscalationSettings(
        policy=policy, strong_window="forward"
    )
    with pytest.raises(
        ValueError, match="online threshold calibration is serial-only"
    ):
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            strong_window="forward",
            escalation=escalation,
        )


def test_an_online_source_beside_run_both_at_once_is_refused():
    online = _always_auditing_online_threshold(threshold=2.0)
    collaborators = policies.EscalationCollaborators(
        threshold=online, expected_source=SOURCE, run_both_at_once=True
    )
    with pytest.raises(ValueError, match="nothing to audit"):
        policies.Switching(collaborators)


# ---- a row plugs in


class _AlwaysEscalate(policies.EscalationPolicyBase):
    """A fake row: every weak result is re-decoded by the strong tier."""

    requires_strong_context = True
    primary_tier = window_records.DecoderTier.WEAK

    def verdict_for_weak_result(self, job, result) -> decoding_records.Verdict:
        del job
        del result
        return decoding_records.Verdict.ESCALATE


def test_a_policy_row_added_to_the_table_runs_a_switching_point(monkeypatch):
    """One class and one ESCALATIONS row; the row declares its tier."""
    monkeypatch.setitem(
        escalation_settings.ESCALATIONS, "always_escalate", _AlwaysEscalate
    )
    escalation = escalation_settings.EscalationSettings(kind="always_escalate")
    machine = fabric.switching_machine(
        rounds=9, escalated_windows=set(), escalation=escalation
    )
    machine.run()
    assert machine.decoder_manager.strong_requests.counts.needed == 3
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "strong"),
        ((1, 1), "strong"),
        ((1, 2), "strong"),
    ]


class _ConfidentEscalation(policies.EscalationPolicyBase):
    """A fourth row: it escalates, and it decides on a confidence.

    Its threshold and the source it expects arrive in the one
    EscalationCollaborators record, so the build learns what this row
    needs from the two facts it declares and from nothing else.
    """

    decides_on_a_confidence = True
    requires_strong_context = True
    primary_tier = window_records.DecoderTier.WEAK

    def __init__(self, collaborators):
        self.threshold = collaborators.threshold
        self.expected_source = collaborators.expected_source

    def verdict_for_weak_result(self, job, result) -> decoding_records.Verdict:
        """Keep what the threshold keeps; a result with no gap escalates."""
        if result.soft_output is None:
            return decoding_records.Verdict.ESCALATE
        if self.threshold.decide_keep(job, result):
            return decoding_records.Verdict.KEEP
        return decoding_records.Verdict.ESCALATE


def _confident_config(tmp_path):
    """A yaml naming the fourth row, with no router, boundary or scheme."""
    return _switching_config(tmp_path, kind="confident")


def _switching_config(
    tmp_path,
    *,
    kind="switching",
    threshold_source=None,
    windows_kind=None,
    terminal_policy=None,
    boundaries=None,
):
    """A yaml with no router, no boundary policy and no windowing scheme."""
    escalation = {"kind": kind, "gap_threshold_db": 20.0}
    if threshold_source is not None:
        escalation["threshold_source"] = threshold_source
    weak_decoder = _weak_unit()
    workload = dict(yaml_configs.MINIMAL_CONFIG["workload"])
    workload["rounds_per_shot"] = 9
    sweep_point = {
        "physical_error_probability": [0.008],
        "distance": [3],
        "round_period_us": [1.0],
        "shots": 1,
    }
    strong_decoder = yaml_configs.strong_unit("belief_matching")
    card = {
        "escalation": escalation,
        "workload": workload,
        **weak_decoder,
        **strong_decoder,
        "sweep": [sweep_point],
    }
    card["windows"] = _windows_section(
        windows_kind, terminal_policy, boundaries
    )
    return yaml_configs.write_config(tmp_path, card)


def _weak_unit() -> dict:
    """One pymatching unit on the fridge clock, the switching weak tier."""
    return {
        "weak_decoder": {
            "kind": "pymatching",
            "units": 1,
            "unit_memory_rounds": None,
            "engine": {
                "clock": "fridge",
                "fetch_cycles_per_round": 1,
                "release_cycles_per_job": 1,
            },
        }
    }


def _windows_section(windows_kind, terminal_policy, boundaries) -> dict:
    """The minimal windows section, with the keys the caller named."""
    windows = dict(yaml_configs.MINIMAL_CONFIG["windows"])
    if windows_kind is not None:
        windows["kind"] = windows_kind
    if terminal_policy is not None:
        windows["terminal_policy"] = terminal_policy
    if boundaries is not None:
        windows["boundaries"] = boundaries
    return windows


def test_a_fourth_escalation_row_gets_the_boundaries_router_and_join(
    monkeypatch, tmp_path
):
    """The wiring reads the row's declared facts, not the kind's name.

    A row that escalates and decides on a confidence, named from a yaml
    that gives no router, no boundary policy and no windowing scheme,
    gets held boundaries, the two-pool router and the confidence join.
    """
    monkeypatch.setitem(
        escalation_settings.ESCALATIONS, "confident", _ConfidentEscalation
    )
    config_path = _confident_config(tmp_path)
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.008, distance=3, round_period_us=1.0
    )
    machine = machine_module.Machine.build(settings, 0)
    boundary_policy = machine.window_manager.courier.boundary_policy
    assert isinstance(boundary_policy, boundary_policies.Held)
    assert machine.window_manager.requester.gap_join is not None
    pool = machine.decoder_manager.pool
    assert sorted(pool.units_by_pool) == ["default", "strong"]
    strong_probe = decoding_records.DecodeJob(
        operation_id=-1,
        window_id=0,
        round_count=0,
        kind=decoding_records.DecodeJobKind.STRONG_REDECODE,
    )
    weak_probe = decoding_records.DecodeJob(
        operation_id=-1, window_id=0, round_count=0
    )
    router = pool.router
    assert router.route(strong_probe) is not router.route(weak_probe)
    assert machine.window_manager.planner.scheme.has_trailing_tail_context
    result = machine.run()
    assert result.terminal_status == "complete"


class _KeepEverything:
    """A threshold source written outside decsim: the port, and no more.

    Its one constructor argument is the sweep point's threshold in nats,
    which is what the root gives every row of the table.
    """

    audits_by_escalating = False
    reads_a_calibration_table = False
    built_per_sweep_point = False

    def __init__(self, threshold_nats: float) -> None:
        self.threshold_nats = threshold_nats

    def decide_keep(self, job, result) -> bool:
        """Every weak result is confident enough."""
        del job
        del result
        return True

    def learn_from_strong_result(self, window_key, result) -> None:
        """This source learns nothing."""
        del window_key
        del result


def test_a_threshold_source_written_outside_decsim_runs_from_a_yaml(
    monkeypatch, tmp_path
):
    """One THRESHOLD_SOURCES row and one yaml name, nothing else."""
    monkeypatch.setitem(
        escalation_settings.THRESHOLD_SOURCES,
        "keep_everything",
        _KeepEverything,
    )
    config_path = _switching_config(
        tmp_path, threshold_source="keep_everything"
    )
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.008, distance=3, round_period_us=1.0
    )
    policy = escalation_build.build_escalation_policy(settings.escalation)
    assert isinstance(policy.threshold, _KeepEverything)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert machine.decoder_manager.strong_requests.counts.needed == 0


class _Draws:
    """A uniform draw that always audits (0.0 is below every audit rate)."""

    def random(self) -> float:
        return 0.0


def _serial_switching_settings(
    boundary_policy,
) -> machine_settings.MachineSettings:
    """Serial switching (no forward window) with the boundary policy given."""
    lookahead = window_records.WindowingSchemeCard(terminal_policy="lookahead")
    scheme = sliding_scheme.SlidingWindowScheme(lookahead)
    windows = window_settings.WindowSettings(
        scheme=scheme, boundary_policy=boundary_policy
    )
    weak_decoder = decoder_settings.DecoderSettings(kind="pymatching")
    strong_decoder = decoder_settings.DecoderSettings(kind="belief_matching")
    escalation = escalation_settings.EscalationSettings(
        kind="switching", gap_threshold_nats=1.0
    )
    return machine_settings.MachineSettings(
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        escalation=escalation,
    )


def test_serial_switching_refuses_eager_boundaries_at_build():
    """A provisional boundary shipped eagerly is never corrected.

    Under serial switching the weak result may be revised by the strong
    decoder, so the boundary waits for the final result; Eager would
    hand a successor a correction the strong tier later replaces.
    """
    eager = boundary_policies.Eager()
    settings = _serial_switching_settings(eager)
    with pytest.raises(
        ValueError, match="serial switching requires held boundaries"
    ):
        machine_module.Machine.build(settings, 0)


def test_the_forward_window_refuses_held_boundaries_at_build():
    """The far boundary IS the restart window's weak commit.

    Toshio 2510.25222 Sec. III C: under the forward window the weak chain
    keeps committing while the strong region decodes, so holding the
    weak boundaries until the strong result arrives would deadlock the
    strong window on itself.
    """
    settings = _forward_window_settings(3, 3)
    held = boundary_policies.Held()
    windows = window_settings.WindowSettings(
        commit_rounds=3, buffer_rounds=3, boundary_policy=held
    )
    settings = dataclasses.replace(settings, windows=windows)
    with pytest.raises(
        ValueError,
        match="a boundary policy that holds provisional boundaries would",
    ):
        machine_module.Machine.build(settings, 0)


class DelegatingWindowScheme:
    """A windowing scheme row a study adds: sliding windows, wrapped.

    It fills the WindowingScheme port by holding a sliding scheme and
    passing every call to it, and it declares the three facts the port
    asks a row to declare. That is what a new row is: one class, the
    declarations, and nothing else changes (gem5's port API, arXiv
    2007.03152 lines 489-491).
    """

    has_trailing_tail_context = True
    commits_in_one_serial_chain = True
    supports_dynamic_streams = True

    def __init__(
        self,
        card: window_records.WindowingSchemeCard = (
            window_records.DEFAULT_SCHEME_CARD
        ),
    ) -> None:
        del card
        lookahead = window_records.WindowingSchemeCard(
            terminal_policy="lookahead"
        )
        self.inner = sliding_scheme.SlidingWindowScheme(lookahead)

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ):
        """The windows the sliding scheme lays out."""
        return self.inner.plan_operation(
            operation_id,
            round_count,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )

    def data_complete(self, window, *, readiness) -> bool:
        """Whether the window has every round it reads."""
        return self.inner.data_complete(window, readiness=readiness)

    def validate_buffer(self, geometry) -> None:
        """The sliding scheme's trailing floor."""
        self.inner.validate_buffer(geometry)


class UndeclaredWindowScheme:
    """The same kind of row with the port's declarations left off.

    It carries no port method either: the build refuses the row before
    it plans a window.
    """


def test_a_windowing_scheme_added_from_outside_runs_under_switching():
    """The forward window reads the row's declaration, not its class.

    A row that commits in one serial chain and keeps trailing tail
    context serves the forward strong window, whatever class it is.
    """
    scheme = DelegatingWindowScheme()
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        strong_window="forward",
        round_microseconds=4.0,
        scheme=scheme,
    )
    machine.run()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]


def test_a_windowing_scheme_named_in_a_yaml_runs_under_switching(
    monkeypatch, tmp_path
):
    """The yaml path reads the declaration too, not the kind's name.

    Before this round build/plan.py refused any windows.kind but sliding
    under switching, so a row that declares a trailing tail ran when it
    was handed in through Python and was refused when a yaml named it.
    """
    monkeypatch.setitem(
        window_settings.WINDOWING_SCHEMES,
        "delegating",
        DelegatingWindowScheme,
    )
    config_path = _switching_config(tmp_path, windows_kind="delegating")
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.008, distance=3, round_period_us=1.0
    )
    machine = machine_module.Machine.build(settings, 0)
    planner_scheme = machine.window_manager.planner.scheme

    assert isinstance(planner_scheme, DelegatingWindowScheme)
    result = machine.run()
    assert result.terminal_status == "complete"


def test_eager_boundaries_named_in_a_yaml_are_refused_under_switching(
    tmp_path,
):
    """The boundary policy's key reaches the escalation's own refusal.

    A yaml may name windows.boundaries; the escalation's check_plan is
    what refuses eager shipping against a serial escalation, by the fact
    the row declares.
    """
    config_path = _switching_config(tmp_path, boundaries="eager")
    with pytest.raises(ValueError, match="serial switching requires held"):
        config = experiment.load_experiment(config_path)
        settings = config.point_settings(
            physical_error_probability=0.008, distance=3, round_period_us=1.0
        )
        machine_module.Machine.build(settings, 0)


def test_held_boundaries_named_in_a_yaml_are_the_rows_the_run_gets(tmp_path):
    """The key is read, not only defaulted."""
    config_path = _switching_config(tmp_path, boundaries="held")
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.008, distance=3, round_period_us=1.0
    )
    machine = machine_module.Machine.build(settings, 0)
    boundary_policy = machine.window_manager.courier.boundary_policy

    assert settings.windows.boundaries == "held"
    assert isinstance(boundary_policy, boundary_policies.Held)


def test_a_flush_tail_named_in_a_yaml_is_refused_under_switching(tmp_path):
    """The policy's own refusal is the only one left."""
    config_path = _switching_config(tmp_path, terminal_policy="flush")
    with pytest.raises(ValueError, match="no trailing tail context"):
        config = experiment.load_experiment(config_path)
        settings = config.point_settings(
            physical_error_probability=0.008, distance=3, round_period_us=1.0
        )
        machine_module.Machine.build(settings, 0)


def test_a_windowing_scheme_without_the_declarations_is_refused_by_name():
    """A row that declares nothing is refused at build, by the fact it lacks."""
    scheme = UndeclaredWindowScheme()
    with pytest.raises(
        ValueError,
        match="UndeclaredWindowScheme does not declare "
        "has_trailing_tail_context",
    ):
        fabric.switching_machine(rounds=9, escalated_windows={1}, scheme=scheme)
