"""The escalation section: when a window is decoded again, and on what.

One record per yaml `escalation` section, the table of escalation kinds
and the table of strong window shapes. A row of ESCALATIONS is the only
place a kind's facts are written: which tier decodes the plan's windows
(primary_tier) and whether the strong context is retained
(requires_strong_context) and whether it decides on a confidence
(decides_on_a_confidence, which is also what gates this section's
confidence keys) are read off the class, so a new kind is one class and
one row (sinter's BUILT_IN_DECODERS shape). A row of
STRONG_WINDOW_SHAPES is the geometry the strong tier re-decodes
(Toshio et al. arXiv 2510.25222).
"""

import csv
import dataclasses
import math
import pathlib
from collections.abc import Mapping
from numbers import Real
from typing import Optional

import decsim.config as config
import decsim.escalation.policies as escalation_policies
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.ports as ports
import decsim.tables as tables

# escalation.kind names one of these rows.
ESCALATIONS = {
    "weak_baseline": escalation_policies.Baseline,
    "strong_only": escalation_policies.StrongOnly,
    "switching": escalation_policies.Switching,
}
# escalation.strong_window names one of these rows: the shape of the
# window the strong tier re-decodes. The root resolves the name once and
# builds the row with the window components it needs.
STRONG_WINDOW_SHAPES = {
    "near_seam_pinned": strong_window_shapes.NearSeamWindow,
    "forward_seam_pinned": strong_window_shapes.ForwardSeamWindow,
}
# escalation.threshold_source names one of these rows: where the
# switching threshold of a sweep point comes from.
THRESHOLD_SOURCES = {
    "fixed": threshold_sources.FixedThreshold,
    "table": threshold_sources.TableThreshold,
    "online": threshold_sources.OnlineThreshold,
}
# How many of the strong region's buffer regions the restarted weak
# window re-reads under the forward strong window (Toshio 2510.25222
# Sec. III C). The text has the weak decoder resume once r_com + r_buf
# rounds are stored after the strong region (lines 1229-1235), which
# both values meet. 1, the default, reads its last buffer region as the
# restart window's past context, which is how Fig. 12 step 5 draws the
# restart window: that block is half assigned to the strong decoder and
# half the weak decoder's buffer. 0 reads nothing inside the region.
RESTART_REREAD_BUFFER_REGIONS = (0, 1)
ESCALATION_KEYS = (
    "kind",
    "clock",
    "threshold_cycles",
    "switch_cycles",
    "confidence",
    "confidence_walk_microseconds",
    "gap_threshold_db",
    "run_both_at_once",
    "strong_window",
    "restart_reread_buffer_regions",
    "threshold_source",
    "threshold_table",
    "threshold_column",
    "online",
)
ONLINE_KEYS = (
    "target_escalation_rate",
    "step_db",
    "audit_rate",
    "kept_bad_budget",
    "adjust_factor",
    "min_escalation_rate",
    "max_escalation_rate",
)
# Decibels are 10 log10 of the likelihood ratio; matching weights are
# its natural log: nats = decibels * ln(10) / 10.
LN_TEN = math.log(10.0)


@dataclasses.dataclass(frozen=True)
class OnlineThresholdSettings:
    """The online calibrator's knobs (threshold_source online).

    Two loops around the live threshold: a rate tracker steps it toward
    target_escalation_rate on every window (step_db per event), and
    a randomized audit lane strong-decodes audit_rate of the kept
    windows; one revised audit multiplies the target by adjust_factor,
    and only ceil(3 / kept_bad_budget) consecutive clean audits divide it
    back. The target stays inside [min_escalation_rate,
    max_escalation_rate - audit_rate], because the audits reach the strong
    tier beside the target and max_escalation_rate bounds the strong
    duty they make together (threshold_sources.py,
    OnlineThresholdController).
    Defaults are the validated drift-replay configuration.
    """

    target_escalation_rate: float = 1e-3
    step_db: float = 0.25
    audit_rate: float = 0.01
    kept_bad_budget: float = 2e-4
    adjust_factor: float = 2.0
    min_escalation_rate: float = 1e-5
    max_escalation_rate: float = 0.30

    @classmethod
    def from_yaml(cls, section: Mapping) -> "OnlineThresholdSettings":
        """The `online` card, every key optional."""
        if not isinstance(section, Mapping):
            raise ValueError(
                "escalation.online must be a mapping of the calibrator's "
                f"knobs (got {section!r})"
            )
        unknown = set(section) - set(ONLINE_KEYS)
        if unknown:
            listed = sorted(unknown)
            known = _listed(ONLINE_KEYS)
            raise ValueError(
                f"escalation.online does not know {listed}; its keys are "
                f"{known}"
            )
        target_escalation_rate = _online_float(
            section, "target_escalation_rate", 1e-3
        )
        step_db = _online_float(section, "step_db", 0.25)
        audit_rate = _online_float(section, "audit_rate", 0.01)
        kept_bad_budget = _online_float(section, "kept_bad_budget", 2e-4)
        adjust_factor = _online_float(section, "adjust_factor", 2.0)
        min_escalation_rate = _online_float(
            section, "min_escalation_rate", 1e-5
        )
        max_escalation_rate = _online_float(
            section, "max_escalation_rate", 0.30
        )
        settings = cls(
            target_escalation_rate=target_escalation_rate,
            step_db=step_db,
            audit_rate=audit_rate,
            kept_bad_budget=kept_bad_budget,
            adjust_factor=adjust_factor,
            min_escalation_rate=min_escalation_rate,
            max_escalation_rate=max_escalation_rate,
        )
        settings.check()
        return settings

    def check(self) -> None:
        """Refuse a knob outside its range."""
        if self.step_db <= 0:
            raise ValueError(
                "escalation.online.step_db must be positive "
                f"(got {self.step_db})"
            )
        if not 0 < self.audit_rate < 1:
            raise ValueError(
                "escalation.online.audit_rate must be in (0, 1) "
                f"(got {self.audit_rate})"
            )
        if not 0 < self.kept_bad_budget < 1:
            raise ValueError(
                "escalation.online.kept_bad_budget must be in (0, 1) "
                f"(got {self.kept_bad_budget})"
            )
        if self.adjust_factor <= 1:
            raise ValueError(
                "escalation.online.adjust_factor must exceed 1 "
                f"(got {self.adjust_factor})"
            )
        self._check_rates()

    def step_nats(self) -> float:
        """The step in natural-log weight units."""
        return decibels_to_nats(self.step_db)

    def _check_rates(self) -> None:
        low = self.min_escalation_rate
        high = self.max_escalation_rate
        if not 0 < low <= high <= 1:
            raise ValueError(
                "escalation.online needs 0 < min_escalation_rate <= "
                f"max_escalation_rate <= 1 (got {low} and {high})"
            )
        target_cap = high - self.audit_rate
        if not low <= self.target_escalation_rate <= target_cap:
            raise ValueError(
                "escalation.online.target_escalation_rate must lie inside "
                "[min_escalation_rate, max_escalation_rate - audit_rate], "
                "since the audits reach the strong tier beside it "
                f"(got {self.target_escalation_rate}, cap {target_cap})"
            )


@dataclasses.dataclass(frozen=True)
class EscalationSettings:
    """The yaml's `escalation` section.

    Table rows (ESCALATIONS, above): weak_baseline (every window on the
    weak tier, final), strong_only (every window decoded once on the
    strong tier, woken from the strong syndrome buffer), switching (weak first,
    escalate serially on a small complementary gap, Toshio 2510.25222
    Sec. III A without the parallel head start). The switching knobs:
    keep the weak result when its gap is at or above the threshold in
    decibels (the paper uses 20 dB); threshold_source fixed uses it as
    given, table looks the sweep point up in an offline calibration csv
    (threshold_column, calibrated for this run's window geometry),
    online starts there and adapts it across a point's shots, serial
    switching only; run_both_at_once is Sec. III A's Step 1, the strong
    decoder started with the weak one and cancelled on confidence
    (false, the default, is the same section's on-demand variant, lines
    631-640); strong_window names the shape of the window the strong
    tier re-decodes (STRONG_WINDOW_SHAPES in
    decsim/escalation/strong_window_shapes.py): near_seam_pinned, the
    default, re-decodes the escalated window's commit region with its
    past face pinned on the earlier neighbour's committed correction and
    one buffer ahead (Bombin et al. 2303.04846 lines 775-788 and
    1456-1458); forward_seam_pinned is the paper's Sec. III C scheme as
    it is stated, an r_com + 2 r_buf extent read with no context, both
    faces pinned (lines 1248-1259), and restart_reread_buffer_regions
    is how many of the strong region's buffer regions the restarted
    weak window re-reads under it.
    confidence names the signal the weak tier reports and the threshold
    decides on (confidence/signals.py), and the yaml refuses a
    weak decoder whose decode cannot produce that signal's evidence;
    confidence_walk_microseconds is the card that prices the signal's own
    computation on the weak unit, null leaving each row on its own cost
    model.
    The complementary gap's two forced-class solves are two ordinary
    jobs of the weak pool, so weak_decoder.units alone decides whether
    they overlap. A Python-built policy is used as it is. The
    threshold in nats and the online threshold source are set per sweep
    point by the experiments layer; base_directory resolves a relative
    threshold_table.
    """

    clock: Optional[config.Clock] = None
    threshold_cycles: int = 0
    switch_cycles: int = 0
    kind: str = "weak_baseline"
    confidence: str = "complementary_gap"
    confidence_walk_microseconds: Optional[float] = None
    gap_threshold_db: Optional[float] = None
    threshold_source: str = "fixed"
    threshold_table: Optional[str] = None
    threshold_column: Optional[str] = None
    online: Optional[OnlineThresholdSettings] = None
    run_both_at_once: bool = False
    strong_window: str = "near_seam_pinned"
    restart_reread_buffer_regions: int = 1
    policy: Optional[ports.EscalationPolicy] = None
    gap_threshold_nats: Optional[float] = None
    online_threshold: Optional[ports.ThresholdSource] = None
    base_directory: Optional[pathlib.Path] = None

    def __post_init__(self) -> None:
        config.check_cycles(
            "escalation.threshold_cycles", self.threshold_cycles
        )
        config.check_cycles("escalation.switch_cycles", self.switch_cycles)
        charged = self.threshold_cycles + self.switch_cycles
        if charged > 0 and self.clock is None:
            raise ValueError("charged escalation costs need a clock")

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        base_directory: Optional[pathlib.Path] = None,
        default_clock: Optional[config.Clock] = None,
    ) -> "EscalationSettings":
        """The common timing card and the policy's confidence knobs.

        Both are checked at this boundary so a row without confidence
        can accept timing keys while still refusing confidence knobs.
        """
        kind = section.get("kind", "weak_baseline")
        unknown = set(section) - set(ESCALATION_KEYS)
        if unknown:
            listed = sorted(unknown)
            known = _listed(ESCALATION_KEYS)
            raise ValueError(
                f"escalation does not know {listed}; its keys are {known}"
            )
        row = tables.row(ESCALATIONS, "escalation.kind", kind)
        clock = default_clock
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        confidence_keys = set(section) - {
            "kind",
            "clock",
            "threshold_cycles",
            "switch_cycles",
        }
        if not row.decides_on_a_confidence:
            if confidence_keys:
                listed = sorted(confidence_keys)
                raise ValueError(
                    f"escalation.kind {kind} decides on no confidence; "
                    f"drop {listed}"
                )
            threshold_cycles = section.get("threshold_cycles", 0)
            switch_cycles = section.get("switch_cycles", 0)
            return cls(
                kind=kind,
                clock=clock,
                threshold_cycles=threshold_cycles,
                switch_cycles=switch_cycles,
            )
        return _switching_settings(kind, section, base_directory, clock)

    def threshold_nats_for(self, resolved: Mapping) -> Optional[float]:
        """The sweep point's threshold in nats, per threshold_source.

        fixed and online read the card (online starts there and adapts);
        table looks the point up in the calibration csv and refuses a
        point the table does not certify, instead of guessing. The
        table's key columns are headed by yaml paths
        (qpu.distance, workload.arguments.physical_error_probability),
        each matched against the value at that path in the point's
        resolved sections, so a point that sweeps neither still finds
        its row; every other column is one method's threshold in dB
        (Toshio et al. 2510.25222 Sec. III B sets g_th by brute force
        over P_L(g_th), lines 855-863, or as the smallest g_th with
        P_L,th(g_th) <= epsilon P_L,strong, Eq. (4) at line 890).
        """
        if not self._decides_on_a_confidence():
            return None
        row = self._threshold_row()
        if not row.reads_a_calibration_table:
            return self.gap_threshold_nats
        table_path = self._table_path()
        rows = _table_rows(table_path, self.threshold_column)
        for row in rows:
            point = _point_at(row, resolved, table_path)
            if _is_point(row, point):
                cell = row[self.threshold_column]
                return _certified_nats(
                    cell, table_path, self.threshold_column, point
                )
        _refuse_a_point_off_the_table(rows, resolved, table_path)

    def online_threshold_for(
        self, resolved: Mapping
    ) -> Optional[ports.ThresholdSource]:
        """The row's own source for this sweep point, when it builds one.

        A row that declares built_per_sweep_point builds it, so what
        the experiments layer installs on the point's task is the row
        the table names.
        Every other row answers None: the root builds those from the
        point's threshold in nats instead. The instance is shared by
        every shot of the point, so the source learns over the point's
        whole window stream.
        """
        if not self._decides_on_a_confidence():
            return None
        row = self._threshold_row()
        if not row.built_per_sweep_point:
            return None
        return row.for_sweep_point(
            self.online, self.gap_threshold_nats, resolved
        )

    def _threshold_row(self):
        """The row escalation.threshold_source names."""
        return tables.row(
            THRESHOLD_SOURCES,
            "escalation.threshold_source",
            self.threshold_source,
        )

    def _decides_on_a_confidence(self) -> bool:
        """Whether this section's kind reads a confidence to decide keep.

        A Python-built policy answers for itself; otherwise the kind's
        row does, so a threshold is looked up for exactly the kinds
        whose section carries the threshold keys.
        """
        if self.policy is not None:
            return self.policy.decides_on_a_confidence
        row = tables.row(ESCALATIONS, "escalation.kind", self.kind)
        return row.decides_on_a_confidence

    def _table_path(self) -> pathlib.Path:
        table_path = pathlib.Path(self.threshold_table)
        if not table_path.is_absolute() and self.base_directory is not None:
            table_path = self.base_directory / table_path
        if not table_path.exists():
            raise ValueError(f"threshold_table {table_path} does not exist")
        return table_path


def decibels_to_nats(decibels: float) -> float:
    """A gap threshold in the paper's decibels as matching weight."""
    scaled = decibels * LN_TEN
    return scaled / 10.0


def nats_to_decibels(nats: float) -> float:
    """A matching weight in the paper's decibels: decibels_to_nats undone."""
    decibels_per_nat = 10.0 / LN_TEN
    return nats * decibels_per_nat


def _switching_settings(
    kind: str,
    section: Mapping,
    base_directory: Optional[pathlib.Path],
    clock: Optional[config.Clock],
) -> EscalationSettings:
    """The confidence knobs of an escalating kind, every rule checked once.

    Which keys the section may carry is the threshold row's to say, not
    its name's: reads_a_calibration_table opens threshold_table and
    threshold_column and closes gap_threshold_db, built_per_sweep_point
    opens the online card, and audits_by_escalating is what serial-only
    calibration means (threshold_sources.py).
    """
    threshold_source = section.get("threshold_source", "fixed")
    threshold_row = tables.row(
        THRESHOLD_SOURCES, "escalation.threshold_source", threshold_source
    )
    gap_threshold_db = _gap_threshold_db(
        section, threshold_source, threshold_row
    )
    gap_threshold_nats = None
    if gap_threshold_db is not None:
        gap_threshold_nats = decibels_to_nats(gap_threshold_db)
    threshold_column = None
    if threshold_row.reads_a_calibration_table:
        raw_column = section.get("threshold_column", "gth_eq4_wilson")
        threshold_column = str(raw_column)
    online = _online_settings(section, threshold_source, threshold_row)
    named_confidence = section.get("confidence", "complementary_gap")
    confidence = str(named_confidence)
    walk_microseconds = _confidence_walk_microseconds(section)
    run_both_at_once = config.boolean(section, "escalation", "run_both_at_once")
    strong_window = _strong_window(section)
    _check_serial_only(threshold_source, threshold_row, strong_window)
    reread_regions = _restart_reread_buffer_regions(section)
    threshold_table = section.get("threshold_table")
    threshold_cycles = section.get("threshold_cycles", 0)
    switch_cycles = section.get("switch_cycles", 0)
    return EscalationSettings(
        clock=clock,
        threshold_cycles=threshold_cycles,
        switch_cycles=switch_cycles,
        kind=kind,
        confidence=confidence,
        confidence_walk_microseconds=walk_microseconds,
        gap_threshold_db=gap_threshold_db,
        gap_threshold_nats=gap_threshold_nats,
        threshold_source=threshold_source,
        threshold_table=threshold_table,
        threshold_column=threshold_column,
        online=online,
        run_both_at_once=run_both_at_once,
        strong_window=strong_window,
        restart_reread_buffer_regions=reread_regions,
        base_directory=base_directory,
    )


def _restart_reread_buffer_regions(section: Mapping) -> int:
    """How far into the strong region the restart window re-reads."""
    regions = section.get("restart_reread_buffer_regions", 1)
    is_a_count = type(regions) is int
    if not is_a_count or regions not in RESTART_REREAD_BUFFER_REGIONS:
        raise ValueError(
            "escalation.restart_reread_buffer_regions must be 0, a "
            "restart on the rounds stored after the strong region, or 1, "
            "a re-read of the region's last buffer region as Toshio "
            f"2510.25222 Fig. 12 step 5 draws it; got {regions!r}"
        )
    return int(regions)


def _gap_threshold_db(
    section: Mapping, threshold_source: str, threshold_row
) -> Optional[float]:
    """The card's threshold; a row that reads a table computes it instead."""
    if threshold_row.reads_a_calibration_table:
        return _calibrated_threshold(section, threshold_source)
    has_table_key = "threshold_table" in section
    has_column_key = "threshold_column" in section
    if has_table_key or has_column_key:
        raise ValueError(
            "threshold_table/threshold_column belong to a "
            "threshold_source that reads a calibration table; the source "
            f"{threshold_source} does not"
        )
    if "gap_threshold_db" not in section:
        raise ValueError(
            "escalation.kind switching needs gap_threshold_db, the keep "
            "threshold in decibels"
        )
    value = section["gap_threshold_db"]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "escalation.gap_threshold_db must be a number of decibels "
            f"(got {value!r})"
        )
    return _checked_decibels(float(value), "escalation.gap_threshold_db")


def _calibrated_threshold(
    section: Mapping, threshold_source: str
) -> Optional[float]:
    """A row that reads a table carries the csv path and no card."""
    if "gap_threshold_db" in section:
        raise ValueError(
            f"threshold_source {threshold_source} computes the threshold "
            "from threshold_table; drop gap_threshold_db"
        )
    if "threshold_table" not in section:
        raise ValueError(
            f"threshold_source {threshold_source} needs threshold_table, "
            "the calibration csv path (calibrate offline for this run's "
            "window geometry)"
        )
    return None


def _online_settings(
    section: Mapping, threshold_source: str, threshold_row
) -> Optional[OnlineThresholdSettings]:
    """The online card, read by a row the experiments layer builds."""
    learns_across_a_point = threshold_row.built_per_sweep_point
    if "online" in section and not learns_across_a_point:
        raise ValueError(
            "the online card belongs to a threshold_source built once "
            f"per sweep point; the source is {threshold_source}"
        )
    if not learns_across_a_point:
        return None
    raw_online = section.get("online")
    if raw_online is None:
        raw_online = {}
    return OnlineThresholdSettings.from_yaml(raw_online)


def _strong_window(section: Mapping) -> str:
    """The strong window shape the section names, refused if not a row."""
    named = section.get("strong_window", "near_seam_pinned")
    rows = sorted(STRONG_WINDOW_SHAPES)
    if named not in rows:
        raise ValueError(
            f"escalation.strong_window {named!r} is not a row of its "
            f"table; the rows are {rows}"
        )
    return str(named)


def _check_serial_only(
    threshold_source: str, threshold_row, strong_window: str
) -> None:
    """A row that audits by escalating is validated for serial switching."""
    row = STRONG_WINDOW_SHAPES[strong_window]
    if not row.absorbs_weak_windows:
        return
    if threshold_row.audits_by_escalating:
        raise ValueError(
            f"threshold_source {threshold_source} is serial-only: an "
            "audit label compares one window's weak and strong committed "
            "observables, and a strong window that absorbs the weak "
            "windows it covers owns a larger extent than the audited "
            "window"
        )


def _online_float(section: Mapping, key: str, default: float) -> float:
    """One knob of the online card: a finite number, never a flag.

    A string is read as a number, because YAML 1.1 loads `1e-3`, the
    way reference.yaml writes the target, as text.
    """
    raw = section.get(key, default)
    sentence = f"escalation.online.{key} must be a finite number (got {raw!r})"
    if isinstance(raw, bool):
        raise ValueError(sentence)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(sentence) from None
    if not math.isfinite(value):
        raise ValueError(sentence)
    return value


def _listed(keys: tuple) -> str:
    """The keys as prose: a, b and c."""
    leading = ", ".join(keys[:-1])
    return f"{leading} and {keys[-1]}"


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
    """Whether a cell holds a value: a number within a relative 1e-9.

    A float written out as text reads back within that; any other value
    is compared as its text.
    """
    if not config.is_number(value):
        return cell == str(value)
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
    checked_decibels = _checked_decibels(gap_threshold_db, entry)
    return decibels_to_nats(checked_decibels)


def _checked_decibels(decibels: float, name: str) -> float:
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


def _confidence_walk_microseconds(section: Mapping) -> Optional[float]:
    """The card that prices a confidence signal's own computation."""
    if "confidence_walk_microseconds" not in section:
        return None
    value = section["confidence_walk_microseconds"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "escalation.confidence_walk_microseconds must be a number of "
            f"microseconds, or null to leave the signal on its own cost "
            f"model (got {value!r})"
        )
    microseconds = float(value)
    if not math.isfinite(microseconds) or microseconds < 0.0:
        raise ValueError(
            "escalation.confidence_walk_microseconds must be finite and "
            f"not negative (got {value!r})"
        )
    return microseconds
