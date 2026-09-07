"""The escalation policies: one law per port method, and the paper's identity.

Toshio et al. 2510.25222 Sec. IV C: the switching rate is the mass of
the window-gap distribution below the threshold, so on one run the
windows escalated equal the recorded weak gaps below g_th equal the
strong corrections in the frame (the sandbox harness rowS1's identity,
on its threshold and its complementary-gap signal). The port is gem5's
conditional predictor (src/cpu/pred/conditional.hh): a row answers and
is told; the root acts.
"""

import math

import pytest
import stim

import decsim.confidence.complementary as complementary
import decsim.confidence.decoder as confidence_decoder
import decsim.controller.policies as boundary_policies
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.settings as decoder_settings
import decsim.escalation.policies as policies
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.message as message
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.windows.settings as window_settings
import decsim.windows.windowing_schemes as windowing_schemes
import tests.escalation.declared_fabric as fabric

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
    op_id=1, k=1, commit_lo=4, commit_hi=6, buffer_hi=9, n_rounds=6
)
JOB = decoding_records.DecodeJob(op_id=1, window_id=1, n_rounds=6)
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
    return policies.Switching(fixed, SOURCE, **arguments)


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
    weak_latency = decoders.PresetLatencyDecoder(1.0)
    weak_matching = mwpm.PyMatchingDecoder(weak_latency)
    signal = complementary.ComplementaryGap()
    weak = confidence_decoder.SoftOutputDecoder(weak_matching, signal)
    strong_latency = decoders.PresetLatencyDecoder(5.0)
    strong = mwpm.PyMatchingDecoder(strong_latency)
    router = decoders.SwitchingRouter(weak=weak, strong=strong)
    fixed = threshold_sources.FixedThreshold(threshold_nats)
    policy = policies.Switching(fixed, complementary.COMPLEMENTARY_GAP_SOURCE)
    operation = message.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(30)
    workload = workload_settings.WorkloadSettings(
        operations=[operation], rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=3, device=device)
    lookahead = windowing_schemes.SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
    scheme = windowing_schemes.SlidingWindowScheme(terminal_policy=lookahead)
    held = boundary_policies.Held()
    windows = window_settings.WindowSettings(
        scheme=scheme, boundary_policy=held
    )
    decoder_manager = decoder_settings.DecoderManagerSettings(
        router=router, unit_pools={"default": 1, "strong": 1}
    )
    escalation = decoder_settings.EscalationSettings(policy=policy)
    pauli_frame = pauli_frame_module.PauliFrameConfig(commit_microseconds=0.004)
    observation = observe_settings.ObservationSettings(
        record_switching_windows=True
    )
    settings = machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        decoder_manager=decoder_manager,
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
    baseline = policies.Baseline()
    assert baseline.primary_tier is window_records.DecoderTier.WEAK
    assert baseline.requires_strong_context is False
    assert baseline.tiers_for_ready_window(WINDOW) == WEAK_TIER
    unsure = _result(0.0)
    verdict = baseline.verdict_for_weak_result(JOB, unsure)
    assert verdict is decoding_records.Verdict.KEEP


def test_strong_only_decodes_every_window_on_the_strong_tier_once():
    strong_only = policies.StrongOnly()
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
    switching = policies.Switching(online, SOURCE)
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
            double_window=True,
            run_both_at_once=True,
        )


def _double_window_settings(
    commit_rounds: int, buffer_rounds: int
) -> machine_module.MachineSettings:
    """The gate's switching card with double_window on and the sizes given.

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
    escalation = decoder_settings.EscalationSettings(
        kind="switching", gap_threshold_nats=1.0, double_window=True
    )
    return machine_module.MachineSettings(
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        escalation=escalation,
    )


@pytest.mark.parametrize("commit_rounds, buffer_rounds", [(3, 4), (4, 3)])
def test_a_double_window_crossing_a_later_commit_region_is_refused_at_build(
    commit_rounds, buffer_rounds
):
    """The slice note's ruling 5: the crossing shape is decided at build.

    The strong region is commit plus two buffers; when twice the buffer
    is not a multiple of the commit, it ends inside a later window's
    commit region, and the run used to die at its first escalation.
    """
    settings = _double_window_settings(commit_rounds, buffer_rounds)
    with pytest.raises(
        ValueError,
        match="the double window's strong region, commit plus two buffers, "
        "must end inside its own commit region",
    ):
        machine_module.Machine.build(settings, 0)


@pytest.mark.parametrize("commit_rounds, buffer_rounds", [(3, 3), (4, 4)])
def test_a_double_window_ending_on_a_commit_edge_builds(
    commit_rounds, buffer_rounds
):
    settings = _double_window_settings(commit_rounds, buffer_rounds)
    machine = machine_module.Machine.build(settings, 0)
    assert machine.window_manager.strong_redecode is not None


def test_an_online_source_under_a_double_window_is_refused_as_serial_only():
    """check_plan's serial-only law.

    An audit label compares one window's weak and strong committed
    observables, and a double-window strong result owns a larger extent
    than the audited window.
    """
    online = _always_auditing_online_threshold(threshold=2.0)
    policy = policies.Switching(online, decoders.SAMPLED_CONFIDENCE_SOURCE)
    escalation = decoder_settings.EscalationSettings(
        policy=policy, double_window=True
    )
    with pytest.raises(
        ValueError, match="online threshold calibration is serial-only"
    ):
        fabric.switching_machine(
            rounds=9,
            escalated_windows=set(),
            double_window=True,
            escalation=escalation,
        )


def test_an_online_source_beside_run_both_at_once_is_refused():
    online = _always_auditing_online_threshold(threshold=2.0)
    with pytest.raises(ValueError, match="nothing to audit"):
        policies.Switching(online, SOURCE, run_both_at_once=True)


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
    """One class and its table rows.

    The kind goes in ESCALATIONS, and its tier in the settings'
    TIER_BY_ESCALATION_KIND, which the front reads.
    """
    monkeypatch.setitem(
        machine_module.ESCALATIONS, "always_escalate", _AlwaysEscalate
    )
    monkeypatch.setitem(
        decoder_settings.TIER_BY_ESCALATION_KIND, "always_escalate", "weak"
    )
    escalation = decoder_settings.EscalationSettings(kind="always_escalate")
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


class _Draws:
    """A uniform draw that always audits (0.0 is below every audit rate)."""

    def random(self) -> float:
        return 0.0
