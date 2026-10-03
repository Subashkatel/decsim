"""The threshold sources' laws: tables are read; online tracks and audits.

FixedThreshold's keep at g >= g_th (Toshio et al. 2510.25222 Sec. III
A, step 3) is checked through the built policy in test_policies.py.
OnlineThreshold's rate tracker is the adaptive conformal
recursion of Gibbs and Candes, arXiv:2106.00170, Eq. (2) at line 143,
which two tests here run beside the tracker; its audit lane is the
inverse-propensity estimate over a random sample of kept windows, and
its outer loop raises the target on one bad audit and relaxes it on a
rule-of-three clean quota.
"""

import dataclasses
import math
import pathlib
import random
import types

import pytest

import decsim.collect as collect
import decsim.escalation.threshold_sources as threshold_sources
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings
import decsim.windows.schemes.sliding as sliding_scheme
import tests.escalation.test_strong_window_shapes as shape_tests

SOURCE = object()  # opaque: the sources compare gaps, never sources


def _result(
    gap: float, logical_observables=(0,)
) -> decoding_records.DecodeResult:
    soft_output = decoding_records.SoftOutput(gap=gap, source=SOURCE)
    return decoding_records.DecodeResult(
        1, 4, logical_observables=logical_observables, soft_output=soft_output
    )


def _job(window_id: int = 4) -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1, window_id=window_id, round_count=3
    )


def _controller(
    *,
    target=0.5,
    threshold=2.0,
    step=0.1,
    audit_rate=1.0,
    kept_bad_budget=0.5,
    max_escalation_rate=0.9,
) -> threshold_sources.OnlineThresholdController:
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=threshold, step=step
    )
    audit = threshold_sources.AuditLane(audit_rate=audit_rate)
    adjustment = threshold_sources.TargetAdjustment(
        kept_bad_budget=kept_bad_budget,
        adjust_factor=2.0,
        min_escalation_rate=1e-5,
        max_escalation_rate=max_escalation_rate,
    )
    return threshold_sources.OnlineThresholdController(
        tracker, audit, adjustment
    )


def test_the_rate_tracker_pins_the_target_rate_over_a_random_gap_stream():
    """A property test: 20000 Gaussian gaps against the recursion's target."""
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=0.1, threshold=4.6, step=0.05
    )
    gap_stream = random.Random(7)
    for _ in range(20000):
        gap = gap_stream.gauss(9.0, 3.0)
        tracker.observe(gap)
    realized_rate = tracker.escalation_rate()
    assert realized_rate == pytest.approx(0.1, abs=0.01)
    assert tracker.threshold > 0.0


def test_the_audit_lane_estimate_is_inverse_propensity_weighted():
    """Each audited bad outcome stands for 1/audit_rate kept windows."""
    lane = threshold_sources.AuditLane(audit_rate=0.5)
    lane.record_audit(weak_was_bad=True)
    lane.record_audit(weak_was_bad=True)
    lane.record_audit(weak_was_bad=False)
    assert lane.audited_count == 3
    assert lane.audited_bad_count == 2
    assert lane.kept_bad_rate(total_window_count=200) == (2 / 0.5) / 200


def test_one_bad_audit_raises_the_target_and_a_clean_quota_relaxes_it():
    """Raising is immediate; a relax needs ceil(3 / 0.5) = 6 clean audits."""
    controller = _controller(target=0.1, kept_bad_budget=0.5, audit_rate=0.1)
    assert controller.relax_audit_quota() == 6
    controller.record_audit_outcome(weak_was_bad=True)
    assert controller.tracker.target_escalation_rate == pytest.approx(0.2)
    assert controller.raise_count == 1
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    controller.record_audit_outcome(weak_was_bad=False)
    assert controller.relax_count == 0
    controller.record_audit_outcome(weak_was_bad=False)
    assert controller.relax_count == 1
    assert controller.tracker.target_escalation_rate == pytest.approx(0.1)


def test_the_raised_target_leaves_room_under_the_cap_for_the_audits():
    """The strong duty, audits included, stays under the Theorem 1 cap.

    Toshio 2510.25222 lines 1333-1340 count every window the strong tier
    decodes, and an audit is one of them: the duty is the target plus
    audit_rate of the kept windows. A target raised to 1.2 stops at
    0.9 - 0.1 = 0.8, whose duty 0.8 + 0.1 (1 - 0.8) = 0.82 is under 0.9.
    """
    controller = _controller(
        target=0.6,
        kept_bad_budget=0.5,
        audit_rate=0.1,
        max_escalation_rate=0.9,
    )
    controller.record_audit_outcome(weak_was_bad=True)
    target = controller.tracker.target_escalation_rate
    kept_fraction = 1.0 - target
    strong_duty = target + 0.1 * kept_fraction
    assert target == pytest.approx(0.8)
    assert strong_duty == pytest.approx(0.82)


def test_an_online_threshold_audits_kept_windows_and_learns_from_the_strong():
    """An audited window escalates; the strong result labels it.

    The label is whether the strong observables revised the kept weak
    ones.
    """
    controller = _controller(target=0.0, threshold=0.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    assert online.audits_by_escalating is True
    # audit_rate 1.0: every kept window is audited, so it escalates
    fourth = _job(4)
    confident = _result(5.0, (1, 0))
    kept = online.decide_keep(fourth, confident)
    assert kept is False
    audited = online.summary()
    assert audited["pending_audits"] == 1
    clean_strong = decoding_records.DecodeResult(
        1, 4, logical_observables=(1, 0)
    )
    online.learn_from_strong_result((1, 4), clean_strong)
    assert controller.raise_count == 0
    labeled = online.summary()
    assert labeled["pending_audits"] == 0
    fifth = _job(5)
    online.decide_keep(fifth, confident)
    revised_strong = decoding_records.DecodeResult(
        1, 5, logical_observables=(0, 0)
    )
    online.learn_from_strong_result((1, 5), revised_strong)
    assert controller.raise_count == 1
    # a strong result that answers no audit (an ordinary escalation) is
    # ignored rather than mislabeled
    online.learn_from_strong_result((1, 6), clean_strong)
    assert controller.audit.audited_count == 2


def test_an_audited_window_reaches_the_strong_tier_outside_the_capped_target():
    """The target bounds the gap escalations; an audit escalates past it.

    Toshio 2510.25222 lines 1333-1340 count every window the strong tier
    decodes in the backlog, and an audited window is one of them, so the
    strong duty is the tracker's rate plus the audits, and the target's
    cap leaves room for them.
    """
    controller = _controller(target=0.0, threshold=0.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    fourth = _job(4)
    confident = _result(5.0)
    kept = online.decide_keep(fourth, confident)
    assert kept is False
    summary = online.summary()
    assert summary["escalated"] == 0
    assert summary["pending_audits"] == 1


def test_an_online_threshold_records_its_trajectory_for_the_experiments_layer():
    controller = _controller(target=0.0, threshold=10.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    assert online.trajectory == [(0, 10.0, "start")]
    fourth = _job(4)
    audited = _result(15.0)
    online.decide_keep(fourth, audited)
    assert online.trajectory[-1] == (1, 10.0, "audit")


def test_an_audit_needs_the_weak_observables_for_its_label():
    controller = _controller(target=0.0, threshold=0.0, step=0.0)
    draws = random.Random(0)
    online = threshold_sources.OnlineThreshold(controller, draws)
    soft_output = decoding_records.SoftOutput(gap=5.0, source=SOURCE)
    timing_only = types.SimpleNamespace(
        soft_output=soft_output, logical_observables=None
    )
    fourth = _job(4)
    with pytest.raises(ValueError):
        online.decide_keep(fourth, timing_only)


def test_the_rate_tracker_follows_the_papers_recursion_step_for_step():
    """The referent run beside the row.

    Gibbs and Candes arXiv:2106.00170 Eq. (2), line 143:
    alpha_{t+1} = alpha_t + gamma (alpha - err_t), with err_t the
    indicator of the event whose rate is being pinned. Written out here
    over the same gap stream, it reproduces the tracker's threshold at
    every one of five hundred steps.
    """
    target = 0.1
    step = 0.05
    start = 4.6
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=start, step=step
    )
    gap_stream = random.Random(11)
    reference = start
    tracked = []
    expected = []
    for _ in range(500):
        gap = gap_stream.gauss(9.0, 3.0)
        tracker.observe(gap)
        tracked.append(tracker.threshold)
        escalated = gap < reference
        error = float(escalated)
        reference = reference + step * (target - error)
        expected.append(reference)
    assert tracked == pytest.approx(expected)


def test_the_realized_rate_is_pinned_by_the_distance_the_threshold_moved():
    """Proposition 4.1's identity (2106.00170 lines 309-317).

    Summing the recursion gives
    (1/T) sum err_t - alpha = (alpha_1 - alpha_{T+1}) / (T gamma), so the
    realized escalation rate can only stray from the target as far as the
    threshold itself travelled, whatever the gaps did. That is what makes
    the loop self-correcting under drift.
    """
    target = 0.1
    step = 0.05
    start = 4.6
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=start, step=step
    )
    gap_stream = random.Random(23)
    window_count = 2000
    for _ in range(window_count):
        gap = gap_stream.gauss(9.0, 3.0)
        tracker.observe(gap)
    travelled = start - tracker.threshold
    drift = travelled / (window_count * step)
    realized = tracker.escalation_rate()
    assert realized - target == pytest.approx(drift)


def test_the_target_rate_holds_when_every_gap_is_tied_at_zero():
    """Proposition 4.1 assumes nothing of the gaps (2106.00170 lines 300-302).

    A window whose two classes weigh the same has a gap of zero. With
    every gap at zero the threshold spends nine windows in ten below
    zero, where nothing escalates, and the distance it travels stays
    within one step, so the realized rate is within 1/T of the target.
    """
    target = 0.1
    tracker = threshold_sources.EscalationRateTracker(
        target_escalation_rate=target, threshold=0.0, step=1.0
    )
    window_count = 1000
    tied_gaps = [0.0] * window_count

    _observe_each(tracker, tied_gaps)

    realized = tracker.escalation_rate()
    strayed = realized - target
    assert abs(strayed) <= 1.0 / window_count


def _observe_each(tracker, gaps: list) -> None:
    """Feed the tracker one gap per window, so a test body holds no loop."""
    for gap in gaps:
        tracker.observe(gap)


@pytest.mark.parametrize(
    "threshold_decibels", [float("nan"), math.inf, -1.0, True, "20"]
)
def test_a_threshold_that_is_no_nonnegative_number_is_refused(
    threshold_decibels,
):
    """Every gap is a weight difference at least zero (Sec. III A)."""
    key = "threshold_decibels"
    fixed = threshold_sources.FixedThreshold.Settings
    table = threshold_sources.TableThreshold.Settings
    online = threshold_sources.OnlineThreshold.Settings
    table_path = pathlib.Path("calibration.csv")

    with pytest.raises(ValueError, match=key):
        fixed(threshold_decibels)
    with pytest.raises(ValueError, match=key):
        table(
            threshold_decibels=threshold_decibels,
            table=table_path,
            column="gth_eq4_wilson",
        )
    with pytest.raises(ValueError, match=key):
        online(threshold_decibels)


def test_an_online_step_that_is_not_a_number_is_refused():
    online = threshold_sources.OnlineThreshold.Settings
    with pytest.raises(ValueError, match="step_decibels"):
        online(20.0, step_decibels=float("nan"))


@pytest.mark.parametrize(
    "field, value, sentence",
    [
        ("step_decibels", 0.0, "step_decibels must be positive"),
        ("audit_rate", 0.0, "audit_rate must be in"),
        ("kept_bad_budget", 1.5, "kept_bad_budget must be in"),
        ("adjust_factor", 1.0, "adjust_factor must exceed 1"),
    ],
)
def test_an_online_knob_outside_its_range_is_refused(field, value, sentence):
    """A step of 0 never moves the threshold, a negative one moves it back."""
    online = threshold_sources.OnlineThreshold.Settings
    knob = {field: value}

    with pytest.raises(ValueError, match=sentence):
        online(20.0, **knob)


def _write_table(tmp_path, text: str) -> pathlib.Path:
    table_path = tmp_path / "calibration.csv"
    table_path.write_text(text)
    return table_path


def _facts(**given) -> dict:
    """A point's facts by name, None where the point gives none."""
    facts = dict.fromkeys(threshold_sources.POINT_FACTS)
    facts.update(given)
    return facts


def test_a_table_threshold_is_the_first_row_that_holds_the_points_facts(
    tmp_path,
):
    """A float key matches within a relative 1e-9; the first row wins."""
    table_path = _write_table(
        tmp_path,
        "distance,physical_error_probability,gth_eq4_wilson\n"
        "5,0.003,19.5\n"
        "5,0.0030000000001,12.0\n",
    )

    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=5, physical_error_probability=0.003)

    threshold = table.at_point(facts)

    assert table.threshold_decibels is None
    assert threshold.threshold_decibels == 19.5
    assert threshold.table == table_path
    assert threshold.column == "gth_eq4_wilson"


def test_a_yaml_path_header_is_refused_naming_its_fact_header(tmp_path):
    """Read as a threshold column it would leave its key unmatched.

    Matched on distance alone, the point would take the first row's
    12 dB instead of its own 20 dB, with no sign of it.
    """
    table_path = _write_table(
        tmp_path,
        "distance,workload.arguments.physical_error_probability,"
        "gth_eq4_wilson\n"
        "3,0.002,12\n"
        "3,0.001,20\n",
    )

    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=3, physical_error_probability=0.002)

    with pytest.raises(ValueError) as refusal:
        table.at_point(facts)

    assert "workload.arguments.physical_error_probability" in str(refusal.value)


def test_an_integer_key_matches_its_row_exactly(tmp_path):
    """Only a float key reads back within a relative 1e-9 of its text.

    A whole number is written exactly, so 1000000001 finds no row keyed
    1000000000 although the two lie within 1e-9 of each other.
    """
    table_path = _write_table(
        tmp_path, "distance,gth_eq4_wilson\n1000000000,12.0\n"
    )
    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=1000000001)

    with pytest.raises(ValueError):
        table.at_point(facts)


def test_a_table_keyed_on_a_fact_the_point_does_not_give_is_refused(
    tmp_path,
):
    table_path = _write_table(
        tmp_path, "round_period_microseconds,gth_eq4_wilson\n1.0,12.0\n"
    )
    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=5, physical_error_probability=0.003)

    with pytest.raises(ValueError, match="round_period_microseconds"):
        table.at_point(facts)


def test_a_read_table_record_needs_its_file_no_more(tmp_path):
    """The point's row, read once, is the record's own threshold.

    A copy of the read record and the source it builds read no file, so
    the table may be gone by then.
    """
    table_path = _write_table(tmp_path, "distance,gth_eq4_wilson\n5,12.0\n")
    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=5)
    read = table.at_point(facts)
    table_path.unlink()

    copied = dataclasses.replace(read)
    source = copied.build()

    twelve_decibels = threshold_sources.decibels_to_nats(12.0)
    assert source.threshold_nats == twelve_decibels


def test_an_online_source_built_by_hand_is_seeded_by_the_points_facts():
    """The seed text is the one the experiments layer's yaml points use."""
    settings = threshold_sources.OnlineThreshold.Settings(20.0)
    facts = _facts(distance=5, physical_error_probability=0.003)

    source = settings.for_point(facts)

    expected = random.Random("online-threshold d=5 p=0.003")
    twenty_decibels = threshold_sources.decibels_to_nats(20.0)
    assert source.random_generator.getstate() == expected.getstate()
    assert source.controller.tracker.threshold == twenty_decibels


def test_an_online_source_with_no_error_probability_is_refused():
    settings = threshold_sources.OnlineThreshold.Settings(20.0)
    facts = _facts(distance=5)

    with pytest.raises(ValueError, match="physical_error_probability"):
        settings.for_point(facts)


# a table of two points, its three methods' thresholds in decibels; the
# distance-7 point has an empty Wilson entry, too little evidence
CALIBRATION_TABLE = (
    "distance,physical_error_probability,gth_brute_force,gth_eq4,"
    "gth_eq4_wilson\n"
    "3,0.008,2.5,18.0,19.5\n"
    "7,0.008,,20.0,\n"
)
GATE_FACTS = {"distance": 3, "physical_error_probability": 0.008}
# two rates a Stim circuit prints alike, so their circuits are one and
# only the calibrator's seed text tells the points apart
PRINTED_ALIKE = (0.001, 0.0010000000000000002)


def _gate_on(threshold) -> machine_settings.MachineSettings:
    """The gate's switching card, d 3 and p 0.008, on this threshold."""
    settings = shape_tests.gate_switching()
    switching = dataclasses.replace(settings.switching, threshold=threshold)
    return dataclasses.replace(settings, switching=switching)


def _online_gate(physical_error_probability=0.008):
    """The gate on an online threshold that audits often, so it learns."""
    online = threshold_sources.OnlineThreshold.Settings(
        15.0,
        audit_rate=0.3,
        target_escalation_rate=0.2,
        max_escalation_rate=0.5,
    )
    settings = _gate_on(online)
    workload = machine_settings.memory_workload(
        3, physical_error_probability, 30
    )
    return dataclasses.replace(settings, workload=workload)


@pytest.mark.parametrize(
    "cell, sentence",
    [
        ("-5.0", "must be finite and not negative"),
        ("abc", "could not convert string to float"),
    ],
)
def test_a_table_entry_that_is_no_nonnegative_decibel_count_is_refused(
    tmp_path, cell, sentence
):
    table_path = _write_table(tmp_path, f"distance,gth_eq4_wilson\n3,{cell}\n")
    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=3)

    with pytest.raises(ValueError, match=sentence):
        table.at_point(facts)


def test_an_empty_entry_refuses_its_point(tmp_path):
    """No threshold was certified there, so none is guessed."""
    table_path = _write_table(tmp_path, CALIBRATION_TABLE)
    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(distance=7, physical_error_probability=0.008)

    with pytest.raises(ValueError, match="entry is empty"):
        table.at_point(facts)


def test_a_table_with_no_key_column_is_refused(tmp_path):
    """Headers that name no point fact would match every point to row one."""
    table_path = _write_table(tmp_path, "d,p,gth_eq4_wilson\n3,0.008,19.5\n")
    table = threshold_sources.TableThreshold.Settings(table_path)
    facts = _facts(**GATE_FACTS)

    with pytest.raises(ValueError, match="has no key column"):
        table.at_point(facts)


def test_the_column_named_is_the_method_the_point_reads(tmp_path):
    table_path = _write_table(tmp_path, CALIBRATION_TABLE)
    wilson = threshold_sources.TableThreshold.Settings(table_path)
    brute = threshold_sources.TableThreshold.Settings(
        table_path, column="gth_brute_force"
    )
    facts = _facts(**GATE_FACTS)

    wilson_threshold = wilson.at_point(facts)
    brute_threshold = brute.at_point(facts)

    assert wilson_threshold.threshold_decibels == 19.5
    assert brute_threshold.threshold_decibels == 2.5


def test_a_round_period_sweep_finds_its_table_row_by_its_facts(tmp_path):
    """The round period is a point fact, read off the qpu's settings."""
    table_path = _write_table(
        tmp_path,
        "distance,physical_error_probability,round_period_microseconds,"
        "gth_eq4_wilson\n"
        "3,0.008,0.5,12.0\n"
        "3,0.008,1.0,13.0\n",
    )
    table = threshold_sources.TableThreshold.Settings(table_path)
    whole = _gate_on(table)
    half_qpu = dataclasses.replace(whole.qpu, round_period_microseconds=0.5)
    half = dataclasses.replace(whole, qpu=half_qpu)

    half_point = half.at_point()
    whole_point = whole.at_point()
    half_threshold = half_point.switching.threshold
    whole_threshold = whole_point.switching.threshold

    assert half_threshold.threshold_decibels == 12.0
    assert whole_threshold.threshold_decibels == 13.0


def test_a_window_only_sweep_finds_its_table_row_by_its_geometry(tmp_path):
    """A threshold is calibrated for one window geometry.

    The commit and buffer rounds are point facts a table keys on, so a
    sweep over the window alone finds each point's row.
    """
    table_path = _write_table(
        tmp_path,
        "distance,physical_error_probability,commit_rounds,buffer_rounds,"
        "gth_eq4_wilson\n"
        "3,0.008,2,2,12.0\n"
        "3,0.008,3,2,13.0\n",
    )
    table = threshold_sources.TableThreshold.Settings(table_path)
    gate = _gate_on(table)

    two = _threshold_at_commit_rounds(gate, 2)
    three = _threshold_at_commit_rounds(gate, 3)

    assert two.threshold_decibels == 12.0
    assert three.threshold_decibels == 13.0


def _threshold_at_commit_rounds(gate, commit_rounds: int):
    """The threshold the gate's point reads with these commit rounds."""
    scheme = sliding_scheme.SlidingWindowScheme.Settings(
        commit_rounds=commit_rounds, buffer_rounds=2
    )
    windows = dataclasses.replace(gate.windows, scheme=scheme)
    point = dataclasses.replace(gate, windows=windows)
    placed = point.at_point()
    return placed.switching.threshold


def test_a_table_read_from_two_folders_names_its_points_alike(tmp_path):
    """The folder a table sits in is no part of what its points run."""
    first_folder = tmp_path / "first"
    other_folder = tmp_path / "elsewhere"

    first_id = _point_id_with_its_table_in(first_folder)
    other_id = _point_id_with_its_table_in(other_folder)

    assert first_id == other_id


def _point_id_with_its_table_in(folder: pathlib.Path) -> str:
    """The id of the gate's point on the calibration table in this folder."""
    folder.mkdir()
    table_path = _write_table(folder, CALIBRATION_TABLE)
    table = threshold_sources.TableThreshold.Settings(table_path)
    gate = _gate_on(table)
    task = collect.Task(gate, {})
    return task.strong_id()


def test_a_target_that_leaves_no_room_for_the_audits_is_refused():
    """The audits reach the strong tier beside the target.

    Toshio 2510.25222 lines 1333-1340 count every strong decode in the
    backlog, so a target of 0.30 under a 0.30 cap with audit_rate 0.01
    would put the strong duty past the cap.
    """
    with pytest.raises(ValueError, match="max_escalation_rate - audit_rate"):
        threshold_sources.OnlineThreshold.Settings(
            20.0,
            target_escalation_rate=0.30,
            audit_rate=0.01,
            max_escalation_rate=0.30,
        )


def test_a_fixed_threshold_point_builds_no_calibrator():
    fixed = threshold_sources.FixedThreshold.Settings(20.0)
    gate = _gate_on(fixed)

    task = collect.Task(gate, {})

    assert task.online_threshold is None


def test_one_calibrator_learns_across_every_shot_of_its_point():
    """The point's task builds it once and every shot's machine takes it."""
    gate = _online_gate()
    task = collect.Task(gate, {})
    calibrator = task.online_threshold

    collect.run_shot(task, 0)
    after_one_shot = calibrator.summary()
    collect.run_shot(task, 1)
    after_two_shots = calibrator.summary()

    first_shot_windows = after_one_shot["windows"]
    assert first_shot_windows > 0
    assert after_two_shots["windows"] == 2 * first_shot_windows


def test_an_online_point_reproduces_its_decisions():
    """The calibrator's random stream is seeded by the point's facts.

    Rerunning the point reruns the same audits, so the same windows
    escalate and every operation ends with the same observables. The
    decoders are charged their measured wall clock, so the times are
    not compared.
    """
    first_run = _two_shots_of_a_fresh_online_point()
    second_run = _two_shots_of_a_fresh_online_point()

    assert first_run == second_run


def _two_shots_of_a_fresh_online_point() -> tuple:
    """Each shot's observables, and what the point's calibrator did."""
    gate = _online_gate()
    task = collect.Task(gate, {})
    first_shot = collect.run_shot(task, 0)
    second_shot = collect.run_shot(task, 1)
    calibrator = task.online_threshold
    return (
        first_shot.result.operation_results,
        second_shot.result.operation_results,
        calibrator.trajectory,
        calibrator.summary(),
    )


def test_two_online_points_whose_rates_print_alike_have_two_ids():
    """The seed text picks the windows the calibrator audits."""
    plain_rate, nudged_rate = PRINTED_ALIKE
    plain_gate = _online_gate(plain_rate)
    nudged_gate = _online_gate(nudged_rate)

    plain = collect.Task(plain_gate, {})
    nudged = collect.Task(nudged_gate, {})
    (plain_operation,) = plain.settings.workload.operations
    (nudged_operation,) = nudged.settings.workload.operations

    assert str(plain_operation.circuit) == str(nudged_operation.circuit)
    assert plain.strong_id() != nudged.strong_id()
