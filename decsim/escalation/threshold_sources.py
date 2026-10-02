"""The threshold sources: where the switching policy's threshold comes from.

FixedThreshold keeps a weak result whose gap is at or above one value,
the paper's constant g_th (Toshio et al. 2510.25222 Sec. III A, step
3). OnlineThreshold starts there and adapts it across a sweep point's
shots with two loops: a rate tracker that pins the escalation fraction
at a target, and an audit lane that strong-decodes a random sample of
kept windows to learn whether the target is safe; it is one instance
per sweep point, shared by every shot. The third source, TableThreshold,
looks each sweep point up in an offline calibration csv
(TableThreshold.Settings.at_sweep_point), so at run time it decides as
FixedThreshold does. Every row fills the ThresholdSource port
(decsim/ports.py) and is built from its own Settings record, which the
switching settings hold. Thresholds and gaps are natural-log weight
(nats), the unit the decoder compares in; the yaml converts the paper's
decibels.
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
import decsim.tables as tables

# The resolved paths the online source's seed reads its two numbers from.
SEED_DISTANCE_PATH = "qpu.distance"
SEED_PROBABILITY_PATH = "workload.arguments.physical_error_probability"
# Decibels are 10 log10 of the likelihood ratio; matching weights are
# its natural log: nats = decibels * ln(10) / 10.
LN_TEN = math.log(10.0)
ONLINE_KEYS = (
    "target_escalation_rate",
    "step_db",
    "audit_rate",
    "kept_bad_budget",
    "adjust_factor",
    "min_escalation_rate",
    "max_escalation_rate",
)


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

    Every signal's gap is a weight difference or a growth spent, never
    below zero, so a negative threshold keeps every window, as 0 dB
    already does, and an infinite or undefined one is no likelihood
    ratio (Toshio et al. 2510.25222 Sec. III A, step 3: keep at g >=
    g_th, with g_th in decibels).
    """
    if not math.isfinite(decibels) or decibels < 0.0:
        raise ValueError(
            f"{name} must be finite and not negative (got {decibels!r})"
        )
    return decibels


class FixedThreshold:
    """The paper's constant g_th: keep at gap >= threshold, escalate below.

    The three declarations below are what the yaml boundary and the root
    read off a row instead of its name (escalation/settings.py,
    build/escalation.py): audits_by_escalating says the row learns from
    strong results it forces, which is why such a row is serial-only;
    reads_a_calibration_table says its number comes from a csv, which
    opens threshold_table and threshold_column and closes
    gap_threshold_db; built_per_sweep_point says the row builds its own
    instance for a sweep point, through for_sweep_point, which is what
    the online card configures, while a row that declares it False is
    built by the root from the point's threshold in nats, its one
    constructor argument.
    """

    audits_by_escalating = False
    reads_a_calibration_table = False
    built_per_sweep_point = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The constant, in nats (escalation.gap_threshold_db, converted)."""

        threshold_nats: float

        def at_sweep_point(
            self, resolved: Mapping
        ) -> "FixedThreshold.Settings":
            """A constant is the same at every sweep point."""
            del resolved
            return self

        def for_sweep_point(self, resolved: Mapping) -> None:
            """A constant builds nothing shared across a point's shots."""
            del resolved

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

    An offline calibration csv holds one row per (distance, p) with the
    threshold in decibels, set as Toshio et al. 2510.25222 Sec. III B
    sets it (by brute force, or as Eq. (4)'s smallest g_th, line 890);
    the experiments layer looks the point up and converts it before the
    machine is built (Settings.at_sweep_point), so at run time this row
    decides on a constant exactly as FixedThreshold does. What it
    declares that the fixed row does not is where its number came from,
    which is what the settings need to know to demand the csv.
    """

    reads_a_calibration_table = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The csv and its column; the point's number once looked up.

        table is escalation.threshold_table and column
        escalation.threshold_column. threshold_nats is None until the
        experiments layer looks the sweep point up (at_sweep_point).
        base_directory resolves a relative table; it is a label, no
        part of a point's id, since the number the table gives a point
        names the point and the folder the table sits in does not.
        """

        table: str
        column: str = "gth_eq4_wilson"
        threshold_nats: Optional[float] = None
        base_directory: Optional[pathlib.Path] = dataclasses.field(
            compare=False, default=None
        )

        def at_sweep_point(
            self, resolved: Mapping
        ) -> "TableThreshold.Settings":
            """The sweep point's threshold, looked up in the calibration csv.

            The point is refused when the table does not certify it,
            instead of guessed. The table's key columns are headed by
            yaml paths (qpu.distance,
            workload.arguments.physical_error_probability), each matched
            against the value at that path in the point's resolved
            sections, so a point that sweeps neither still finds its
            row; every other column is one method's threshold in dB
            (Toshio et al. 2510.25222 Sec. III B sets g_th by brute force
            over P_L(g_th), lines 855-863, or as the smallest g_th with
            P_L,th(g_th) <= epsilon P_L,strong, Eq. (4) at line 890). The
            first row that holds the point wins.
            """
            table_path = self._table_path()
            rows = _table_rows(table_path, self.column)
            for row in rows:
                point = _point_at(row, resolved, table_path)
                if _is_point(row, point):
                    cell = row[self.column]
                    threshold_nats = _certified_nats(
                        cell, table_path, self.column, point
                    )
                    return dataclasses.replace(
                        self, threshold_nats=threshold_nats
                    )
            _refuse_a_point_off_the_table(rows, resolved, table_path)

        def for_sweep_point(self, resolved: Mapping) -> None:
            """A table builds nothing shared across a point's shots."""
            del resolved

        def build(self) -> "TableThreshold":
            """A fresh source at the point's threshold, once looked up."""
            if self.threshold_nats is None:
                raise ValueError(
                    "escalation.threshold_source table resolves the "
                    "threshold per sweep point in the experiments layer "
                    "(ExperimentConfig.point_task); build the machine "
                    "through it, or give gap_threshold_db"
                )
            return TableThreshold(self.threshold_nats)

        def _table_path(self) -> pathlib.Path:
            table_path = pathlib.Path(self.table)
            is_relative = not table_path.is_absolute()
            if is_relative and self.base_directory is not None:
                table_path = self.base_directory / table_path
            if not table_path.exists():
                raise ValueError(f"threshold_table {table_path} does not exist")
            return table_path


def _table_rows(table_path: pathlib.Path, column: str) -> list:
    """The table's rows, which name the threshold column and a key column.

    A table with no key column would match every point to its first row,
    a wrong threshold with no sign of it, so it is refused.
    """
    with open(table_path, newline="") as table_file:
        reader = csv.DictReader(table_file)
        rows = list(reader)
    if not rows:
        return rows
    columns = list(rows[0])
    if column not in columns:
        raise ValueError(
            f"threshold_table {table_path} has no column {column!r}; its "
            f"columns are {sorted(columns)}"
        )
    key_columns = _key_columns(columns)
    if not key_columns:
        raise ValueError(
            f"threshold_table {table_path} has no key column; a key column "
            "is headed by the yaml path of the setting it matches, as "
            "qpu.distance"
        )
    return rows


def _key_columns(columns: list) -> list:
    """The columns headed by a yaml path, which name a table's points."""
    keys = []
    for column in columns:
        if "." in column:
            keys.append(column)
    return keys


def _point_at(row: dict, resolved: Mapping, table_path) -> dict:
    """The point's value at each of the table's key columns."""
    point = {}
    reader = f"threshold_table {table_path}"
    for column in _key_columns(list(row)):
        point[column] = config.setting_at(resolved, column, reader)
    return point


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


def _refuse_a_point_off_the_table(
    rows: list, resolved: Mapping, table_path
) -> None:
    """The point, and the points the table does certify."""
    point = {}
    calibrated = []
    for row in rows:
        point = _point_at(row, resolved, table_path)
        keys = {}
        for column in point:
            keys[column] = row[column]
        calibrated.append(keys)
    raise ValueError(
        f"threshold_table {table_path} has no row for {point}; its rows "
        f"are {calibrated}"
    )


def _certified_nats(
    cell: str, table_path: pathlib.Path, column: str, point: dict
) -> float:
    if cell == "":
        raise ValueError(
            f"threshold_table {table_path} refuses {point}: the {column} "
            "entry is empty (not enough evidence at calibration time)"
        )
    entry = f"threshold_table {table_path} entry {column} at {point}"
    try:
        gap_threshold_db = float(cell)
    except ValueError:
        raise ValueError(
            f"{entry} must be a number of decibels (got {cell!r})"
        ) from None
    checked = checked_decibels(gap_threshold_db, entry)
    return decibels_to_nats(checked)


class EscalationRateTracker:
    """Inner calibration loop: pin the escalation fraction at a target.

    Label-free: every window reveals whether it escalated
    (gap < threshold), so the escalation fraction is fully observable.
    The update is the adaptive conformal recursion of Gibbs and Candes,
    arXiv:2106.00170, Eq. (2) at line 143,
    alpha_{t+1} = alpha_t + gamma (alpha - err_t): here alpha is the
    target escalation rate, err_t whether this window escalated, and
    gamma the step, so the threshold rises a little on every kept window
    and falls a lot on every escalation. It balances at the target
    quantile of the live gap distribution, and their Proposition 4.1
    (lines 309-317) bounds the long-run average of err_t around alpha
    with no assumption on the data-generating distribution, which is
    what a drifting gap distribution needs. The bound rests on the
    recursion running unclipped (their Lemma 4.1, lines 303-308): the
    threshold may dip below zero, where no gap is below it and nothing
    escalates until the kept windows have raised it again. A floor at
    zero would let a stream of gaps tied at zero escalate far above the
    target. The same shape is old in
    hardware, where an update threshold is servoed by the balance of two
    event rates (Seznec's O-GEHL, CBP-1 2004).
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

    The lane dedicates a small fixed sample to the expensive path so
    ground truth keeps flowing whatever threshold is live, which is the
    move of set-dueling in caches (Qureshi et al., ISCA 2007). Sampling
    kept windows is what avoids the selective-labels problem (Lakkaraju
    et al., KDD 2017): without it every label comes from below the
    threshold and the kept region is pure extrapolation. Neither paper
    is on disk here, so neither is cited by line.

    kept_bad_rate estimates P(weak revised and kept) per window, the
    quantity Toshio et al. 2510.25222 bound with epsilon * PL_strong in
    Eq. (4), line 890. Each audited bad outcome counts 1/audit_rate kept
    windows (inverse propensity). The label is disagreement with the
    strong result, which is the reference that paper measures against
    too.
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

    kept_bad_budget is the bad rate the target may not exceed; one bad
    audit multiplies the target by adjust_factor, ceil(3 / kept_bad_
    budget) clean audits in a row divide it back; the target stays
    inside [min_escalation_rate, max_escalation_rate - audit_rate]
    (OnlineThresholdController).
    """

    kept_bad_budget: float
    adjust_factor: float
    min_escalation_rate: float
    max_escalation_rate: float


class OnlineThresholdController:
    """Both calibration loops together: duty tracking steered by audits.

    The two directions of the outer loop have asymmetric evidence costs
    and are handled asymmetrically. Raise: each audited bad outcome
    carries importance weight one over the audit rate, so one event is
    already strong evidence the budget is blown; the escalation target
    is multiplied by adjust_factor immediately. Relax: certifying the
    bad rate is below budget needs the rule-of-three quota, about
    3 / kept_bad_budget clean audits for a 95% upper bound at the
    budget; only a full clean quota shrinks the target.

    The switching rate of Toshio 2510.25222 Theorem 1 (lines
    1272-1291) counts every window the strong tier decodes (lines
    1333-1340), and the audited windows reach the strong tier beside
    the ones escalated on their gap: the strong duty is the target plus
    audit_rate of the kept windows, target + audit_rate (1 - target).
    So the target stays inside [min_escalation_rate,
    max_escalation_rate - audit_rate], whatever accuracy would prefer,
    and the strong duty stays at or under max_escalation_rate, the
    number the yaml writes for the theorem's bound. That number is the
    yaml's, and nothing here derives it from the theorem's inputs.
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

    The row of threshold_source online, asked at Switching's decision
    point.

    One instance persists across every shot of a sweep point, so the
    controller learns over the point's whole window stream; its random
    stream is seeded once at construction, so a rerun of the point
    reproduces the same audits. An audited window still escalates (the
    strong result commits, so the audit costs latency, never accuracy)
    and is remembered here; when its strong result arrives, the label is
    whether the strong answer revised the weak committed observables.
    """

    audits_by_escalating = True
    reads_a_calibration_table = False
    built_per_sweep_point = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The starting threshold and the calibrator's knobs.

        threshold_nats is where the rate tracker starts
        (escalation.gap_threshold_db, converted). Two loops move it: a
        rate tracker steps it toward target_escalation_rate on every
        window (step_db per event), and a randomized audit lane
        strong-decodes audit_rate of the kept windows; one revised audit
        multiplies the target by adjust_factor, and only ceil(3 /
        kept_bad_budget) consecutive clean audits divide it back. The
        target stays inside [min_escalation_rate, max_escalation_rate -
        audit_rate], because the audits reach the strong tier beside the
        target and max_escalation_rate bounds the strong duty they make
        together (OnlineThresholdController). Defaults are the validated
        drift-replay configuration.
        """

        threshold_nats: float
        target_escalation_rate: float = 1e-3
        step_db: float = 0.25
        audit_rate: float = 0.01
        kept_bad_budget: float = 2e-4
        adjust_factor: float = 2.0
        min_escalation_rate: float = 1e-5
        max_escalation_rate: float = 0.30

        def __post_init__(self) -> None:
            _check_online_steps(self)
            _check_online_rates(self)

        @classmethod
        def from_yaml(
            cls, section: Mapping, threshold_nats: float
        ) -> "OnlineThreshold.Settings":
            """The `online` card, every key optional, from its start."""
            if not isinstance(section, Mapping):
                raise ValueError(
                    "escalation.online must be a mapping of the "
                    f"calibrator's knobs (got {section!r})"
                )
            tables.refuse_unknown_keys(
                "escalation.online", section, ONLINE_KEYS
            )
            knobs = {}
            for key, default in _ONLINE_DEFAULTS.items():
                knobs[key] = config.finite_number(
                    section, "escalation.online", key, default
                )
            return cls(threshold_nats=threshold_nats, **knobs)

        def step_nats(self) -> float:
            """The step in natural-log weight units."""
            return decibels_to_nats(self.step_db)

        def at_sweep_point(
            self, resolved: Mapping
        ) -> "OnlineThreshold.Settings":
            """The start and the knobs are the same at every sweep point."""
            del resolved
            return self

        def for_sweep_point(self, resolved: Mapping) -> "OnlineThreshold":
            """The one instance a sweep point's shots share, point-seeded.

            Both loops are assembled here, where they are read: the rate
            tracker starting at the threshold in nats, the audit lane,
            and the target adjustment. The random stream is seeded from
            the point's distance and error rate alone, read by path from
            its resolved sections, so a rerun of the point draws the same
            audits.
            """
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
            reader = "the online threshold's seed"
            distance = config.setting_at(resolved, SEED_DISTANCE_PATH, reader)
            probability = config.setting_at(
                resolved, SEED_PROBABILITY_PATH, reader
            )
            generator = random.Random(
                f"online-threshold d={distance} p={probability}"
            )
            return OnlineThreshold(controller, generator)

        def build(self) -> "OnlineThreshold":
            """Refused: the source learns across a point's shots."""
            raise ValueError(
                "escalation.threshold_source online is built once per sweep "
                "point by the experiments layer "
                "(ExperimentConfig.point_task), which seeds it and "
                "shares it across the point's shots; build the machine "
                "through it"
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

        An audit escalates with the decision recorded for labeling.
        """
        unit_random = self.random_generator.random()
        escalated, audited = self.controller.observe(
            result.soft_output.gap, unit_random
        )
        if self.controller.tracker.window_count % 100 == 0:
            self._record("sample")
        if audited:
            if result.logical_observables is None:
                raise ValueError(
                    "online threshold calibration needs the weak decoder "
                    "to produce logical observables for audit labels; "
                    "the configured weak card is timing-only"
                )
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
        if result.logical_observables is None:
            raise ValueError(
                f"audited window {window_key}: the strong result carries no "
                "logical observables to compare against"
            )
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


# The online card's knobs and their defaults, in the order the card
# lists them.
_ONLINE_DEFAULTS = {
    "target_escalation_rate": 1e-3,
    "step_db": 0.25,
    "audit_rate": 0.01,
    "kept_bad_budget": 2e-4,
    "adjust_factor": 2.0,
    "min_escalation_rate": 1e-5,
    "max_escalation_rate": 0.30,
}


def _check_online_steps(online) -> None:
    """Refuse a step, an audit share or a factor outside its range."""
    if online.step_db <= 0:
        raise ValueError(
            f"escalation.online.step_db must be positive (got {online.step_db})"
        )
    if not 0 < online.audit_rate < 1:
        raise ValueError(
            "escalation.online.audit_rate must be in (0, 1) "
            f"(got {online.audit_rate})"
        )
    if not 0 < online.kept_bad_budget < 1:
        raise ValueError(
            "escalation.online.kept_bad_budget must be in (0, 1) "
            f"(got {online.kept_bad_budget})"
        )
    if online.adjust_factor <= 1:
        raise ValueError(
            "escalation.online.adjust_factor must exceed 1 "
            f"(got {online.adjust_factor})"
        )


def _check_online_rates(online) -> None:
    low = online.min_escalation_rate
    high = online.max_escalation_rate
    if not 0 < low <= high <= 1:
        raise ValueError(
            "escalation.online needs 0 < min_escalation_rate <= "
            f"max_escalation_rate <= 1 (got {low} and {high})"
        )
    target_cap = high - online.audit_rate
    if not low <= online.target_escalation_rate <= target_cap:
        raise ValueError(
            "escalation.online.target_escalation_rate must lie inside "
            "[min_escalation_rate, max_escalation_rate - audit_rate], "
            "since the audits reach the strong tier beside it "
            f"(got {online.target_escalation_rate}, cap {target_cap})"
        )
