"""The threshold sources: where the switching policy's threshold comes from.

FixedThreshold keeps a weak result whose gap is at or above one value,
the paper's constant g_th (Toshio et al. 2510.25222 Sec. III A, step 3).
TableThreshold looks a point's facts up in an offline calibration csv
before the machine is built, then decides as FixedThreshold does.
OnlineThreshold starts at a value and adapts it across a sweep point's
shots with a rate tracker and an audit lane; one instance per point is
shared by its shots. A record reads the point's facts off the point's
settings, as a gem5 Parent.x proxy reads x off the object above it
(src/python/m5/proxy.py:116-148). A record takes decibels and hands out
nats, the unit a gap is compared in.
"""

import csv
import dataclasses
import fractions
import math
import pathlib
import random
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.records.decoding as decoding_records

# The point's facts a calibration table keys its rows on, each column
# headed by the fact's name: the code, the noise, the round clock, and
# the window geometry a threshold is calibrated for.
POINT_FACTS = (
    "distance",
    "physical_error_probability",
    "round_period_microseconds",
    "commit_rounds",
    "buffer_rounds",
)
# Decibels are 10 log10 of the likelihood ratio; matching weights are
# its natural log: nats = decibels * ln(10) / 10.
LN_TEN = math.log(10.0)


def decibels_to_nats(decibels: float) -> float:
    """A gap threshold in the paper's decibels as matching weight."""
    scaled = decibels * LN_TEN
    return scaled / 10.0


def nats_to_decibels(nats: float) -> float:
    """A matching weight in the paper's decibels: decibels_to_nats undone."""
    decibels_per_nat = 10.0 / LN_TEN
    return nats * decibels_per_nat


def checked_decibels(decibels: float, name: str) -> float:
    """A keep threshold is finite and not negative, wherever it is read.

    Every gap is at least zero, so a negative threshold keeps every
    window as 0 dB does, and an infinite or undefined one is no
    likelihood ratio.
    """
    is_finite = config.is_number(decibels) and math.isfinite(decibels)
    if not is_finite or decibels < 0.0:
        raise ValueError(
            f"{name} must be finite and not negative (got {decibels!r})"
        )
    return decibels


class FixedThreshold:
    """The paper's constant g_th: keep at gap >= threshold, escalate below.

    audits_by_escalating says whether the row learns from strong results
    it forces, which makes it serial-only.
    """

    audits_by_escalating = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The constant, in decibels."""

        threshold_decibels: float

        def __post_init__(self) -> None:
            checked_decibels(self.threshold_decibels, "threshold_decibels")

        @property
        def threshold_nats(self) -> float:
            """The threshold as the weight a gap is compared in."""
            return decibels_to_nats(self.threshold_decibels)

        def at_point(self, facts: Mapping) -> "FixedThreshold.Settings":
            """A constant is the same at every point."""
            del facts
            return self

        def for_point(self, facts: Mapping) -> None:
            """A constant builds nothing shared across a point's shots."""
            del facts

        def build(self) -> "FixedThreshold":
            """A fresh source at this threshold."""
            return FixedThreshold(self.threshold_nats)

    def __init__(self, threshold_nats: float) -> None:
        self.threshold_nats = threshold_nats

    def decide_keep(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> bool:
        """The gap against the threshold, both in nats."""
        del job
        return result.soft_output.gap >= self.threshold_nats

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """A fixed threshold learns nothing."""
        del window_key
        del result


class TableThreshold(FixedThreshold):
    """The calibration table's g_th for this sweep point.

    The table holds one row per point with the threshold in decibels,
    set as Toshio et al. 2510.25222 Sec. III B sets it (by brute force,
    or as Eq. (4)'s smallest g_th, line 890).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The csv and its column, and the threshold they give the point.

        table is a label and no part of a point's id: the number the
        table gives names the point. threshold_decibels is that number,
        None until the point's row is read (at_point), which the point's
        task does before it names the point.
        """

        table: pathlib.Path = dataclasses.field(compare=False)
        column: str = "gth_eq4_wilson"
        threshold_decibels: Optional[float] = None

        def __post_init__(self) -> None:
            if self.threshold_decibels is None:
                return
            checked_decibels(self.threshold_decibels, "threshold_decibels")

        @property
        def threshold_nats(self) -> Optional[float]:
            """The threshold as the weight a gap is compared in; None unread."""
            if self.threshold_decibels is None:
                return None
            return decibels_to_nats(self.threshold_decibels)

        def at_point(self, facts: Mapping) -> "TableThreshold.Settings":
            """The record holding the threshold of the point these facts name.

            The table's key columns are headed by the POINT_FACTS they
            match; every other column is one method's threshold in dB
            (Toshio et al. 2510.25222 Sec. III B: brute force over
            P_L(g_th), lines 855-863, or Eq. (4) at line 890). The first
            row that holds the point wins. A record already read is its
            own reading.
            """
            if self.threshold_decibels is not None:
                return self
            columns, rows = _table_rows(self.table)
            point = _point_of(columns, facts, self.table)
            row = _first_row_holding(rows, point)
            cell = row[self.column]
            threshold_decibels = _certified_decibels(
                cell, self.table, self.column, point
            )
            return dataclasses.replace(
                self, threshold_decibels=threshold_decibels
            )

        def for_point(self, facts: Mapping) -> None:
            """A table's constant builds nothing shared across the shots."""
            del facts

        def build(self) -> "TableThreshold":
            """A fresh source at the point's threshold, once it is read."""
            return TableThreshold(self.threshold_nats)


class EscalationRateTracker:
    """Inner calibration loop: pin the escalation fraction at a target.

    Every window reveals whether it escalated, so the fraction is fully
    observable. The update is the adaptive conformal recursion of Gibbs
    and Candes, arXiv:2106.00170, Eq. (2) at line 143, alpha_{t+1} =
    alpha_t + gamma (alpha - err_t), with alpha the target rate and
    err_t whether this window escalated: the threshold rises a little on
    every kept window and falls a lot on every escalation. Their
    Proposition 4.1 (lines 309-317) bounds the long-run average of err_t
    around alpha with no assumption on the gap distribution, and the
    bound needs the recursion unclipped (Lemma 4.1, lines 303-308): the
    threshold may dip below zero, where nothing escalates until kept
    windows raise it again.
    """

    def __init__(
        self, target_escalation_rate: float, threshold: float, step: float
    ) -> None:
        self.target_escalation_rate = target_escalation_rate
        self.threshold = threshold
        self.step = step
        self.window_count = 0
        self.escalated_count = 0

    def observe(self, gap: float) -> bool:
        """Consume one window's gap; True when it escalates."""
        escalated = gap < self.threshold
        self.window_count += 1
        self.escalated_count += int(escalated)
        move = self.target_escalation_rate - float(escalated)
        self.threshold += self.step * move
        return escalated

    def escalation_rate(self) -> float:
        """The escalated fraction of the windows seen so far."""
        if self.window_count == 0:
            return 0.0
        return self.escalated_count / self.window_count


class AuditLane:
    """Randomized strong-decoder audits of kept windows.

    A small fixed sample goes to the expensive path so ground truth
    keeps flowing whatever threshold is live, the move of set-dueling in
    caches (Qureshi et al., ISCA 2007). Sampling kept windows avoids the
    selective-labels problem (Lakkaraju et al., KDD 2017). kept_bad_rate
    estimates P(weak revised and kept) per window, the quantity Toshio
    et al. 2510.25222 bound with epsilon * PL_strong in Eq. (4), line
    890; each audited bad outcome counts 1/audit_rate kept windows. The
    label is disagreement with the strong result.
    """

    def __init__(self, audit_rate: float) -> None:
        self.audit_rate = audit_rate
        self.audited_count = 0
        self.audited_bad_count = 0
        self.weighted_bad_sum = 0.0

    def should_audit(self, unit_random: float) -> bool:
        """Whether this kept window is audited, from one uniform draw."""
        return unit_random < self.audit_rate

    def record_audit(self, weak_was_bad: bool) -> None:
        """One audit's label; a bad one weighs 1/audit_rate windows."""
        self.audited_count += 1
        if weak_was_bad:
            self.audited_bad_count += 1
            self.weighted_bad_sum += 1.0 / self.audit_rate

    def kept_bad_rate(self, total_window_count: int) -> float:
        """The inverse-propensity estimate of P(weak revised and kept)."""
        if total_window_count == 0:
            return 0.0
        return self.weighted_bad_sum / total_window_count


@dataclasses.dataclass(frozen=True)
class TargetAdjustment:
    """How the outer loop moves the escalation target on an audit label.

    One bad audit multiplies the target by adjust_factor; ceil(3 /
    kept_bad_budget) clean audits in a row divide it back; the target
    stays inside [min_escalation_rate, max_escalation_rate -
    audit_rate].
    """

    kept_bad_budget: float
    adjust_factor: float
    min_escalation_rate: float
    max_escalation_rate: float


class OnlineThresholdController:
    """Both calibration loops together: duty tracking steered by audits.

    Raise: one audited bad outcome weighs one over the audit rate,
    already strong evidence the budget is blown, so the target is
    multiplied by adjust_factor at once. Relax: certifying the bad rate
    is below budget needs the rule-of-three quota, about 3 /
    kept_bad_budget clean audits for a 95% bound. The switching rate of
    Toshio 2510.25222 Theorem 1 (lines 1272-1291) counts every window
    the strong tier decodes (lines 1333-1340), audits included, so the
    target stays at or under max_escalation_rate - audit_rate and the
    strong duty at or under max_escalation_rate, a number the caller
    gives.
    """

    def __init__(
        self,
        tracker: EscalationRateTracker,
        audit: AuditLane,
        adjustment: TargetAdjustment,
    ) -> None:
        self.tracker = tracker
        self.audit = audit
        self.adjustment = adjustment
        self.clean_audit_streak = 0
        self.raise_count = 0
        self.relax_count = 0

    def relax_audit_quota(self) -> int:
        """Clean audits needed before a relax is statistically earned."""
        quota = 3.0 / self.adjustment.kept_bad_budget
        rounded_up = math.ceil(quota)
        return int(rounded_up)

    def observe(self, gap: float, unit_random: float) -> tuple:
        """One window: (escalated, audited)."""
        escalated = self.tracker.observe(gap)
        audited = False
        if not escalated:
            audited = self.audit.should_audit(unit_random)
        return escalated, audited

    def record_audit_outcome(self, weak_was_bad: bool) -> None:
        """One audit's label moves the target: up at once, down on a quota."""
        self.audit.record_audit(weak_was_bad)
        if weak_was_bad:
            self.clean_audit_streak = 0
            self.raise_count += 1
            self._scale_target(self.adjustment.adjust_factor)
            return
        self.clean_audit_streak += 1
        quota = self.relax_audit_quota()
        if self.clean_audit_streak >= quota:
            self.clean_audit_streak = 0
            self.relax_count += 1
            relax_factor = 1.0 / self.adjustment.adjust_factor
            self._scale_target(relax_factor)

    def target_cap(self) -> float:
        """The largest target whose strong duty, audits included, fits."""
        return self.adjustment.max_escalation_rate - self.audit.audit_rate

    def _scale_target(self, factor: float) -> None:
        proposed = self.tracker.target_escalation_rate * factor
        cap = self.target_cap()
        if proposed > cap:
            proposed = cap
        if proposed < self.adjustment.min_escalation_rate:
            proposed = self.adjustment.min_escalation_rate
        self.tracker.target_escalation_rate = proposed


class OnlineThreshold:
    """Adapts the threshold during the run, from the escalation rate it sees.

    One instance persists across every shot of a sweep point, and its
    random stream is seeded once, so a rerun reproduces the same audits.
    An audited window still escalates, so an audit costs latency, never
    accuracy; its label is whether the strong answer revised the weak
    committed observables.
    """

    audits_by_escalating = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The starting threshold and the calibrator's knobs.

        threshold_decibels is where the rate tracker starts; it steps
        toward target_escalation_rate by step_decibels per window. The
        audit lane strong-decodes audit_rate of the kept windows; one
        revised audit multiplies the target by adjust_factor, and ceil(3
        / kept_bad_budget) clean audits divide it back, inside
        [min_escalation_rate, max_escalation_rate - audit_rate]. The
        defaults are the drift-replay configuration it was validated on.
        """

        threshold_decibels: float
        target_escalation_rate: float = 1e-3
        step_decibels: float = 0.25
        audit_rate: float = 0.01
        kept_bad_budget: float = 2e-4
        adjust_factor: float = 2.0
        min_escalation_rate: float = 1e-5
        max_escalation_rate: float = 0.30

        def __post_init__(self) -> None:
            checked_decibels(self.threshold_decibels, "threshold_decibels")
            _check_online_numbers(self)
            _check_online_steps(self)
            _check_online_rates(self)

        @property
        def threshold_nats(self) -> float:
            """Where the rate tracker starts, as the weight a gap is in."""
            return decibels_to_nats(self.threshold_decibels)

        def step_nats(self) -> float:
            """The step in natural-log weight units."""
            return decibels_to_nats(self.step_decibels)

        def at_point(self, facts: Mapping) -> "OnlineThreshold.Settings":
            """The card is the same at every point; its source is not."""
            del facts
            return self

        def for_point(self, facts: Mapping) -> "OnlineThreshold":
            """The one instance a sweep point's shots share, point-seeded.

            The random stream is seeded from the point's distance and
            physical error probability as written, so a rerun draws the
            same audits.
            """
            distance = facts["distance"]
            physical_error_probability = facts["physical_error_probability"]
            _refuse_a_seed_with_no_fact(distance, physical_error_probability)
            step_nats = self.step_nats()
            tracker = EscalationRateTracker(
                target_escalation_rate=self.target_escalation_rate,
                threshold=self.threshold_nats,
                step=step_nats,
            )
            audit = AuditLane(audit_rate=self.audit_rate)
            adjustment = TargetAdjustment(
                kept_bad_budget=self.kept_bad_budget,
                adjust_factor=self.adjust_factor,
                min_escalation_rate=self.min_escalation_rate,
                max_escalation_rate=self.max_escalation_rate,
            )
            controller = OnlineThresholdController(tracker, audit, adjustment)
            generator = random.Random(
                f"online-threshold d={distance} p={physical_error_probability}"
            )
            return OnlineThreshold(controller, generator)

        def build(self) -> "OnlineThreshold":
            """Refused: the source learns across a point's shots."""
            raise ValueError(
                "an online threshold learns across a sweep point's shots, "
                "so it is built once per point by for_point and shared: "
                "a point's task builds it, and Machine.build takes it as "
                "online_threshold for one machine"
            )

    def __init__(
        self, controller: OnlineThresholdController, random_generator
    ) -> None:
        self.controller = controller
        self.random_generator = random_generator
        # (operation id, window id) -> the weak observables under audit
        self._pending_audits: dict = {}
        # (window count, threshold, event) at the start, every hundredth
        # window, and every audit and its label
        self.trajectory: list = []
        self._record("start")

    def decide_keep(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> bool:
        """True keeps the weak result, False escalates.

        An audit escalates, with the weak observables kept for its
        label.
        """
        unit_random = self.random_generator.random()
        escalated, audited = self.controller.observe(
            result.soft_output.gap, unit_random
        )
        if self.controller.tracker.window_count % 100 == 0:
            self._record("sample")
        if audited:
            key = (job.operation_id, job.window_id)
            self._pending_audits[key] = tuple(result.logical_observables)
            self._record("audit")
            return False
        return not escalated

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """A strong result arrived; if it answers an audit, label it."""
        weak_observables = self._pending_audits.pop(window_key, None)
        if weak_observables is None:
            return
        strong_observables = tuple(result.logical_observables)
        weak_was_bad = strong_observables != weak_observables
        self.controller.record_audit_outcome(weak_was_bad)
        event = "audit_clean"
        if weak_was_bad:
            event = "audit_bad"
        self._record(event)

    def summary(self) -> dict:
        """The source's counters and its live threshold."""
        tracker = self.controller.tracker
        audit = self.controller.audit
        return {
            "windows": tracker.window_count,
            "escalated": tracker.escalated_count,
            "escalation_rate": tracker.escalation_rate(),
            "threshold": tracker.threshold,
            "target_escalation_rate": tracker.target_escalation_rate,
            "audited": audit.audited_count,
            "audited_bad": audit.audited_bad_count,
            "kept_bad_rate": audit.kept_bad_rate(tracker.window_count),
            "raises": self.controller.raise_count,
            "relaxes": self.controller.relax_count,
            "pending_audits": len(self._pending_audits),
        }

    def _record(self, event: str) -> None:
        tracker = self.controller.tracker
        self.trajectory.append((tracker.window_count, tracker.threshold, event))


def _check_online_numbers(online) -> None:
    """Every knob of the online card is a finite number."""
    for field in dataclasses.fields(online):
        value = getattr(online, field.name)
        is_finite = config.is_number(value) and math.isfinite(value)
        if not is_finite:
            raise ValueError(
                f"{field.name} must be a finite number (got {value!r})"
            )


def _check_online_steps(online) -> None:
    """Refuse a step, an audit share or a factor outside its range."""
    if online.step_decibels <= 0:
        raise ValueError(
            f"step_decibels must be positive (got {online.step_decibels})"
        )
    if not 0 < online.audit_rate < 1:
        raise ValueError(
            f"audit_rate must be in (0, 1) (got {online.audit_rate})"
        )
    if not 0 < online.kept_bad_budget < 1:
        raise ValueError(
            f"kept_bad_budget must be in (0, 1) (got {online.kept_bad_budget})"
        )
    if online.adjust_factor <= 1:
        raise ValueError(
            f"adjust_factor must exceed 1 (got {online.adjust_factor})"
        )


def _check_online_rates(online) -> None:
    low = online.min_escalation_rate
    high = online.max_escalation_rate
    if not 0 < low <= high <= 1:
        raise ValueError(
            "min_escalation_rate must be above 0 and at most "
            "max_escalation_rate, which is at most 1 "
            f"(got {low} and {high})"
        )
    target_cap = high - online.audit_rate
    if not low <= online.target_escalation_rate <= target_cap:
        raise ValueError(
            "target_escalation_rate must lie inside "
            "[min_escalation_rate, max_escalation_rate - audit_rate], "
            "since the audits reach the strong tier beside it "
            f"(got {online.target_escalation_rate}, cap {target_cap})"
        )


def _refuse_a_seed_with_no_fact(distance, physical_error_probability) -> None:
    """The seed text names both facts, so neither may be missing."""
    if distance is not None and physical_error_probability is not None:
        return
    raise ValueError(
        "an online threshold seeds its audits from the point's distance "
        "and physical_error_probability, and the point gives "
        f"distance={distance!r}, "
        f"physical_error_probability={physical_error_probability!r}"
    )


def _table_rows(table_path: pathlib.Path) -> tuple:
    """The table's columns and rows.

    A table with no key column would match every point to its first row,
    a wrong threshold with no sign of it, so it is refused.
    """
    with open(table_path, newline="") as table_file:
        reader = csv.DictReader(table_file)
        rows = list(reader)
        columns = reader.fieldnames or []
    _refuse_a_settings_path_header(columns, table_path)
    key_columns = _key_columns(columns)
    if not key_columns:
        raise ValueError(
            f"threshold_table {table_path} has no key column; a key column "
            f"is headed by the point fact it matches, one of {POINT_FACTS}"
        )
    return columns, rows


def _refuse_a_settings_path_header(columns: list, table_path) -> None:
    """A header that is a settings path is refused, naming the fact to write.

    Read as a threshold column, it would leave its key unmatched, and the
    point would take another row's threshold with no sign of it.
    """
    for column in columns:
        if "." not in column:
            continue
        sentence = _settings_path_header_sentence(column, table_path)
        raise ValueError(sentence)


def _settings_path_header_sentence(column: str, table_path) -> str:
    """The refusal of a settings path header, with its fact if it names one."""
    sentence = (
        f"threshold_table {table_path} heads a column with the settings path "
        f"{column}; a key column is headed by the point fact it matches, "
        f"one of {POINT_FACTS}"
    )
    _, _, last_name = column.rpartition(".")
    if last_name not in POINT_FACTS:
        return sentence
    return f"{sentence}, so write {last_name}"


def _key_columns(columns: list) -> list:
    """The columns headed by a point fact, which name a table's points."""
    return [column for column in columns if column in POINT_FACTS]


def _point_of(columns: list, facts: Mapping, table_path) -> dict:
    """The point's value at each of the table's key columns.

    A fact the point does not give is refused: its None would otherwise
    match a row whose cell reads "None".
    """
    point = {}
    for column in _key_columns(columns):
        value = facts[column]
        if value is None:
            raise ValueError(
                f"threshold_table {table_path} keys its rows on {column}, "
                f"and the point gives no {column}"
            )
        point[column] = value
    return point


def _first_row_holding(rows: list, point: dict) -> Optional[dict]:
    """The first row whose key cells hold the point, or None."""
    for row in rows:
        if _is_point(row, point):
            return row
    return None


def _is_point(row: dict, point: dict) -> bool:
    """Whether every key cell of the row holds the point's value."""
    for column, value in point.items():
        if not _cell_holds(row[column], value):
            return False
    return True


def _cell_holds(cell: str, value) -> bool:
    """Whether a cell holds a value: a float within a relative 1e-9.

    A float written out as text reads back within that. A whole number
    is written exactly, so it is compared exactly, and any other value
    is compared as its text.
    """
    if not config.is_number(value):
        return cell == str(value)
    if isinstance(value, int):
        return fractions.Fraction(cell) == value
    number = float(cell)
    return math.isclose(number, value, rel_tol=1e-9)


def _certified_decibels(
    cell: str, table_path: pathlib.Path, column: str, point: dict
) -> float:
    entry = f"threshold_table {table_path} entry {column} at {point}"
    gap_threshold_db = float(cell)
    return checked_decibels(gap_threshold_db, entry)
