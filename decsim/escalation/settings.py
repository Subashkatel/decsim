"""The escalation section: when a window is decoded again, and on what.

One record per yaml `escalation` section, the table of escalation kinds
and the table of strong window shapes. A row of ESCALATIONS is the only
place a kind's facts are written: which tier decodes the plan's windows
(primary_tier) and whether the strong context is retained
(requires_strong_context) are read off the class, so a new kind is one
class and one row (sinter's BUILT_IN_DECODERS shape). A row of
STRONG_WINDOW_SHAPES is the geometry the strong tier re-decodes
(Toshio et al. arXiv 2510.25222).
"""

import csv
import dataclasses
import math
import pathlib
import random
from collections.abc import Mapping
from numbers import Real
from typing import Optional

import decsim.escalation.policies as escalation_policies
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.ports as ports

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
    "two_sided_context": strong_window_shapes.ContextWindow,
    "forward": strong_window_shapes.ForwardWindow,
}
THRESHOLD_SOURCES = ("fixed", "table", "online")
# How many of the strong region's buffer regions the restarted weak
# window re-reads under the forward strong window (Toshio 2510.25222
# Sec. III C, Fig. 12). 0, the default, is the paper: the weak decoder
# resumes on the commit plus buffer rounds stored after the strong
# region and reads nothing inside it. 1 reads one buffer region of the
# strong region as
# the restart window's far-boundary context, which is what decsim's
# forward window did until 2026-09-07.
RESTART_REREAD_BUFFER_REGIONS = (0, 1)
ESCALATION_KEYS = (
    "kind",
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
    target_escalation_rate on every window (step_decibels per event), and
    a randomized audit lane strong-decodes audit_rate of the kept
    windows; one revised audit multiplies the target by adjust_factor,
    and only ceil(3 / kept_bad_budget) consecutive clean audits divide it
    back. The target stays inside [min_escalation_rate,
    max_escalation_rate]; the max is the Theorem 1 backlog cap. Defaults
    are the validated drift-replay configuration.
    """

    target_escalation_rate: float = 1e-3
    step_decibels: float = 0.25
    audit_rate: float = 0.01
    kept_bad_budget: float = 2e-4
    adjust_factor: float = 2.0
    min_escalation_rate: float = 1e-5
    max_escalation_rate: float = 0.30

    @classmethod
    def from_yaml(cls, section: Mapping) -> "OnlineThresholdSettings":
        """The `online` card, every key optional."""
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
        step_decibels = _online_float(section, "step_db", 0.25)
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
            step_decibels=step_decibels,
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
        if self.step_decibels <= 0:
            raise ValueError(
                "escalation.online.step_db must be positive "
                f"(got {self.step_decibels})"
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
        return decibels_to_nats(self.step_decibels)

    def _check_rates(self) -> None:
        low = self.min_escalation_rate
        high = self.max_escalation_rate
        if not 0 < low <= high <= 1:
            raise ValueError(
                "escalation.online needs 0 < min_escalation_rate <= "
                f"max_escalation_rate <= 1 (got {low} and {high})"
            )
        if not low <= self.target_escalation_rate <= high:
            raise ValueError(
                "escalation.online.target_escalation_rate must lie inside "
                "[min_escalation_rate, max_escalation_rate] "
                f"(got {self.target_escalation_rate})"
            )


@dataclasses.dataclass(frozen=True)
class EscalationSettings:
    """The yaml's `escalation` section.

    Table rows (ESCALATIONS, above): weak_baseline (every window on the
    weak tier, final), strong_only (every window decoded once on the
    strong tier, woken from syndrome buffer 1), switching (weak first,
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
    decsim/escalation/strong_window_shapes.py: two_sided_context, the
    default, or forward, the paper's Sec. III C scheme), and
    restart_reread_buffer_regions is how many of the strong region's
    buffer regions the restarted weak window re-reads under the forward
    shape.
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
    point by the front; base_directory resolves a relative
    threshold_table.
    """

    kind: str = "weak_baseline"
    confidence: str = "complementary_gap"
    confidence_walk_microseconds: Optional[float] = None
    gap_threshold_decibels: Optional[float] = None
    threshold_source: str = "fixed"
    threshold_table: Optional[str] = None
    threshold_column: Optional[str] = None
    online: Optional[OnlineThresholdSettings] = None
    run_both_at_once: bool = False
    strong_window: str = "two_sided_context"
    restart_reread_buffer_regions: int = 0
    policy: Optional[ports.EscalationPolicy] = None
    gap_threshold_nats: Optional[float] = None
    online_threshold: Optional[threshold_sources.OnlineThreshold] = None
    base_directory: Optional[pathlib.Path] = None

    @classmethod
    def from_yaml(
        cls, section: Mapping, base_directory: Optional[pathlib.Path] = None
    ) -> "EscalationSettings":
        """The `escalation` section: a kind, and the switching knobs."""
        kind = section.get("kind", "weak_baseline")
        unknown = set(section) - set(ESCALATION_KEYS)
        if unknown:
            listed = sorted(unknown)
            known = _listed(ESCALATION_KEYS)
            raise ValueError(
                f"escalation does not know {listed}; its keys are {known}"
            )
        switching_keys = set(section) - {"kind"}
        if kind != "switching":
            if switching_keys:
                listed = sorted(switching_keys)
                raise ValueError(
                    f"escalation.kind {kind} never escalates; drop {listed}"
                )
            return cls(kind=kind)
        return _switching_settings(section, base_directory)

    def threshold_nats_for(
        self, physical_error_probability: float, distance: int
    ) -> Optional[float]:
        """The sweep point's threshold in nats, per threshold_source.

        fixed and online read the card (online starts there and adapts);
        table looks the point up in the calibration csv
        (calibrate_threshold.py's calibration_table.csv: one row per
        distance and p, thresholds in dB) and refuses a point the table
        does not certify, instead of guessing.
        """
        if self.kind != "switching":
            return None
        if self.threshold_source in ("fixed", "online"):
            return self.gap_threshold_nats
        table_path = self._table_path()
        rows = _table_rows(table_path, self.threshold_column)
        for row in rows:
            if _is_point(row, physical_error_probability, distance):
                cell = row[self.threshold_column]
                return _certified_nats(
                    cell,
                    table_path,
                    self.threshold_column,
                    physical_error_probability,
                    distance,
                )
        calibrated_points = []
        for row in rows:
            calibrated_points.append((int(row["distance"]), float(row["p"])))
        calibrated_points.sort()
        raise ValueError(
            f"threshold_table {table_path} has no row for d={distance} "
            f"p={physical_error_probability}; calibrated points: "
            f"{calibrated_points}"
        )

    def online_threshold_for(
        self, physical_error_probability: float, distance: int
    ) -> Optional[threshold_sources.OnlineThreshold]:
        """One online threshold source per sweep point (source online).

        Shared by every shot of the point so the controller learns over
        the point's whole window stream; seeded by the point's identity,
        so a rerun reproduces the same audit draws.
        """
        if self.kind != "switching":
            return None
        if self.threshold_source != "online":
            return None
        online = self.online
        step_nats = online.step_nats()
        tracker = threshold_sources.EscalationRateTracker(
            target_escalation_rate=online.target_escalation_rate,
            threshold=self.gap_threshold_nats,
            step=step_nats,
        )
        audit = threshold_sources.AuditLane(audit_rate=online.audit_rate)
        adjustment = threshold_sources.TargetAdjustment(
            kept_bad_budget=online.kept_bad_budget,
            adjust_factor=online.adjust_factor,
            min_escalation_rate=online.min_escalation_rate,
            max_escalation_rate=online.max_escalation_rate,
        )
        controller = threshold_sources.OnlineThresholdController(
            tracker, audit, adjustment
        )
        generator = random.Random(
            f"online-threshold d={distance} p={physical_error_probability}"
        )
        return threshold_sources.OnlineThreshold(controller, generator)

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


def _switching_settings(
    section: Mapping, base_directory: Optional[pathlib.Path]
) -> EscalationSettings:
    """The switching knobs, every cross-key rule checked once."""
    threshold_source = section.get("threshold_source", "fixed")
    if threshold_source not in THRESHOLD_SOURCES:
        raise ValueError(
            "escalation.threshold_source must be one of "
            f"{THRESHOLD_SOURCES}, got {threshold_source!r}"
        )
    gap_threshold_decibels = _gap_threshold_decibels(section, threshold_source)
    gap_threshold_nats = None
    if gap_threshold_decibels is not None:
        gap_threshold_nats = decibels_to_nats(gap_threshold_decibels)
    threshold_column = None
    if threshold_source == "table":
        raw_column = section.get("threshold_column", "gth_eq4_wilson")
        threshold_column = str(raw_column)
    online = _online_settings(section, threshold_source)
    named_confidence = section.get("confidence", "complementary_gap")
    confidence = str(named_confidence)
    walk_microseconds = _confidence_walk_microseconds(section)
    run_both_at_once = _switching_boolean(section, "run_both_at_once")
    strong_window = _strong_window(section)
    _check_serial_only(threshold_source, strong_window)
    reread_regions = _restart_reread_buffer_regions(section)
    threshold_table = section.get("threshold_table")
    return EscalationSettings(
        kind="switching",
        confidence=confidence,
        confidence_walk_microseconds=walk_microseconds,
        gap_threshold_decibels=gap_threshold_decibels,
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


def _switching_boolean(section: Mapping, key: str) -> bool:
    """One of the switching section's on-or-off knobs, off when silent."""
    value = section.get(key, False)
    if value not in (True, False):
        raise ValueError(
            f"escalation.{key} must be true or false, got {value!r}"
        )
    return value


def _restart_reread_buffer_regions(section: Mapping) -> int:
    """How far into the strong region the restart window re-reads."""
    regions = section.get("restart_reread_buffer_regions", 0)
    is_a_count = type(regions) is int
    if not is_a_count or regions not in RESTART_REREAD_BUFFER_REGIONS:
        raise ValueError(
            "escalation.restart_reread_buffer_regions must be 0, the "
            "paper's restart on the rounds stored after the strong "
            "region, or 1, decsim's re-read of one buffer region of it "
            f"for the far boundary; got {regions!r}"
        )
    return int(regions)


def _gap_threshold_decibels(
    section: Mapping, threshold_source: str
) -> Optional[float]:
    """The card's threshold; the table source computes it instead."""
    if threshold_source == "table":
        if "gap_threshold_db" in section:
            raise ValueError(
                "threshold_source table computes the threshold from "
                "threshold_table; drop gap_threshold_db"
            )
        if "threshold_table" not in section:
            raise ValueError(
                "threshold_source table needs threshold_table, the "
                "calibration csv path (calibrate offline for this run's "
                "window geometry)"
            )
        return None
    has_table_key = "threshold_table" in section
    has_column_key = "threshold_column" in section
    if has_table_key or has_column_key:
        raise ValueError(
            "threshold_table/threshold_column belong to threshold_source "
            f"table; the source is {threshold_source}"
        )
    if "gap_threshold_db" not in section:
        raise ValueError(
            "escalation.kind switching needs gap_threshold_db, the keep "
            "threshold in decibels"
        )
    return float(section["gap_threshold_db"])


def _online_settings(
    section: Mapping, threshold_source: str
) -> Optional[OnlineThresholdSettings]:
    if "online" in section and threshold_source != "online":
        raise ValueError(
            "the online card belongs to threshold_source online; the "
            f"source is {threshold_source}"
        )
    if threshold_source != "online":
        return None
    raw_online = section.get("online") or {}
    return OnlineThresholdSettings.from_yaml(raw_online)


def _strong_window(section: Mapping) -> str:
    """The strong window shape the section names, refused if not a row."""
    named = section.get("strong_window", "two_sided_context")
    rows = sorted(STRONG_WINDOW_SHAPES)
    if named not in rows:
        raise ValueError(
            f"escalation.strong_window {named!r} is not a row of its "
            f"table; the rows are {rows}"
        )
    return str(named)


def _check_serial_only(threshold_source: str, strong_window: str) -> None:
    """Online calibration is validated for serial switching."""
    row = STRONG_WINDOW_SHAPES[strong_window]
    if not row.absorbs_weak_windows:
        return
    if threshold_source == "online":
        raise ValueError(
            "threshold_source online is serial-only: an audit label "
            "compares one window's weak and strong committed "
            "observables, and a strong window that absorbs the weak "
            "windows it covers owns a larger extent than the audited "
            "window"
        )


def _online_float(section: Mapping, key: str, default: float) -> float:
    raw = section.get(key, default)
    return float(raw)


def _listed(keys: tuple) -> str:
    """The keys as prose: a, b and c."""
    leading = ", ".join(keys[:-1])
    return f"{leading} and {keys[-1]}"


def _table_rows(table_path: pathlib.Path, column: str) -> list:
    with open(table_path, newline="") as table_file:
        reader = csv.DictReader(table_file)
        rows = list(reader)
    if rows and column not in rows[0]:
        columns = sorted(rows[0])
        raise ValueError(
            f"threshold_table {table_path} has no column {column!r}; its "
            f"columns are {columns}"
        )
    return rows


def _is_point(row: dict, physical_error_probability: float, distance) -> bool:
    row_distance = int(row["distance"])
    if row_distance != distance:
        return False
    row_probability = float(row["p"])
    return math.isclose(
        row_probability, physical_error_probability, rel_tol=1e-9
    )


def _certified_nats(
    cell: str,
    table_path: pathlib.Path,
    column: str,
    physical_error_probability: float,
    distance: int,
) -> float:
    if cell == "":
        raise ValueError(
            f"threshold_table {table_path} refuses d={distance} "
            f"p={physical_error_probability}: the {column} entry is empty "
            "(not enough evidence at calibration time)"
        )
    gap_threshold_decibels = float(cell)
    return decibels_to_nats(gap_threshold_decibels)


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
