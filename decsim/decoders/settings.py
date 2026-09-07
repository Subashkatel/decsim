"""The settings of the decoder tiers, their manager and the escalation.

A decoder tier is one unit pool: its algorithm, its unit count, its
input memory and the cycle-priced stages around the algorithm (Toshio
arXiv 2510.25222: lightweight decoders decode constantly, a separate
accurate decoder is invoked on demand). The escalation says whether and
when a window is decoded again by the strong tier.
"""

import csv
import dataclasses
import math
import pathlib
import random
from collections.abc import Mapping
from typing import Any, Optional, Union

import decsim.config as config
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.escalation.threshold_sources as threshold_sources
import decsim.ports as ports

THRESHOLD_SOURCES = ("fixed", "table", "online")
GAP_COMPUTATIONS = ("serial", "parallel_pair", "split_pair")
# How many of the strong region's buffer regions the restarted weak
# window re-reads under the double window (Toshio 2510.25222 Sec. III C,
# Fig. 12). 0, the default, is the paper: the weak decoder resumes on
# the commit plus buffer rounds stored after the strong region and reads
# nothing inside it. 1 reads one buffer region of the strong region as
# the restart window's far-boundary context, which is what decsim's
# forward window did until 2026-09-07.
RESTART_REREAD_BUFFER_REGIONS = (0, 1)
ESCALATION_KEYS = (
    "kind",
    "gap_threshold_db",
    "double_window",
    "restart_reread_buffer_regions",
    "gap_computation",
    "gap_units",
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
# The tier each escalation kind decodes the plan's windows on. Switching
# decodes every window on the weak tier first and escalates to the
# strong tier.
TIER_BY_ESCALATION_KIND = {
    "weak_baseline": "weak",
    "strong_only": "strong",
    "switching": "weak",
}


@dataclasses.dataclass(frozen=True)
class DecoderSettings:
    """The yaml's `weak_decoder` and `strong_decoder` sections.

    Table rows (decsim/machine.py): pymatching, belief_matching (the two
    tiers of the decoder-switching setting, decoded per window and
    charged their measured wall clock), or a number, a fixed core
    latency in microseconds on the MWPM path. The card prices the
    algorithm stage only; the fetch and release stages are cycles of the
    engine's clock, resolved to a frequency once at load.
    unit_memory_rounds is the input SRAM per unit (None is unbounded); a
    unit overlaps input transfer with compute only when two windows fit.
    kind None is no decoder at all, right for a run that plans no
    windows. A Python-built decoder is routed as it is, with no engine
    stages around it.
    """

    kind: Union[str, float, None] = None
    units: int = 1
    unit_memory_rounds: Optional[int] = None
    fetch_cycles_per_round: int = 1
    release_cycles_per_job: int = 1
    engine_megahertz: Optional[float] = None
    decoder: Optional[ports.Decoder] = None

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "DecoderSettings":
        """A tier section: kind, units, unit memory and the engine card."""
        engine = section["engine"]
        engine_megahertz = clocks.megahertz(engine["clock"])
        unit_memory_rounds = section["unit_memory_rounds"]
        if unit_memory_rounds is not None and unit_memory_rounds < 1:
            raise ValueError(
                "unit_memory_rounds must be at least one round, or null "
                f"for an unbounded unit memory (got {unit_memory_rounds})"
            )
        return cls(
            kind=section["kind"],
            units=section["units"],
            unit_memory_rounds=unit_memory_rounds,
            fetch_cycles_per_round=engine["fetch_cycles_per_round"],
            release_cycles_per_job=engine["release_cycles_per_job"],
            engine_megahertz=engine_megahertz,
        )


@dataclasses.dataclass(frozen=True)
class DecoderManagerSettings:
    """The decoder manager's Python-only knobs; the yaml has no section.

    A router picks the decoder for each job (CodeRouter by code name,
    SwitchingRouter by tier); given, it replaces the one the root builds
    from the two tier sections. The scheduler orders the ready queue
    (FifoScheduler by default), unit_pools names each pool's unit count
    (built from the tiers' units by default) and decoder_memory bounds
    each pool's input memory in rounds (built from the active tier's
    unit_memory_rounds by default). bulk_strong serves the strong pool's
    queued re-decodes as one merged batch (Toshio 2510.25222 Sec. III C,
    the strong decoder processes its assigned data in bulk); timing-only,
    since a batch carries no accuracy-bearing result, and serial only.
    """

    router: Optional[Any] = None
    scheduler: Optional[Any] = None
    unit_pools: Optional[Mapping[str, int]] = None
    decoder_memory: Optional[decoder_memory_module.DecoderMemoryConfig] = None
    bulk_strong: bool = False


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

    Table rows (decsim/machine.py): weak_baseline (every window on the
    weak tier, final), strong_only (every window decoded once on the
    strong tier, woken from syndrome buffer 1), switching (weak first,
    escalate serially on a small complementary gap, Toshio 2510.25222
    Sec. III A without the parallel head start). The switching knobs:
    keep the weak result when its gap is at or above the threshold in
    decibels (the paper uses 20 dB); threshold_source fixed uses it as
    given, table looks the sweep point up in an offline calibration csv
    (threshold_column, calibrated for this run's window geometry),
    online starts there and adapts it across a point's shots, serial
    switching only; double_window is the paper's Sec. III C scheme, and
    restart_reread_buffer_regions is how many of the strong region's
    buffer regions the restarted weak window re-reads under it;
    gap_computation is where the two forced solves run (serial on one
    core, parallel_pair on two cores in the unit, split_pair on its own
    pool of gap_units). A Python-built policy is used as it is. The
    threshold in nats and the online threshold source are set per sweep
    point by the front; base_directory resolves a relative
    threshold_table.
    """

    kind: str = "weak_baseline"
    gap_threshold_decibels: Optional[float] = None
    threshold_source: str = "fixed"
    threshold_table: Optional[str] = None
    threshold_column: Optional[str] = None
    online: Optional[OnlineThresholdSettings] = None
    double_window: bool = False
    restart_reread_buffer_regions: int = 0
    gap_computation: str = "serial"
    gap_units: int = 1
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

    @property
    def decodes_on(self) -> str:
        """The tier that decodes the plan's windows: weak or strong."""
        return TIER_BY_ESCALATION_KIND[self.kind]

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
    gap_computation, gap_units = _gap_computation(section)
    double_window = section.get("double_window", False)
    if double_window not in (True, False):
        raise ValueError(
            "escalation.double_window must be true or false, got "
            f"{double_window!r}"
        )
    _check_serial_only(gap_computation, threshold_source, double_window)
    reread_regions = _restart_reread_buffer_regions(section)
    threshold_table = section.get("threshold_table")
    return EscalationSettings(
        kind="switching",
        gap_threshold_decibels=gap_threshold_decibels,
        gap_threshold_nats=gap_threshold_nats,
        threshold_source=threshold_source,
        threshold_table=threshold_table,
        threshold_column=threshold_column,
        online=online,
        double_window=double_window,
        restart_reread_buffer_regions=reread_regions,
        gap_computation=gap_computation,
        gap_units=gap_units,
        base_directory=base_directory,
    )


def _restart_reread_buffer_regions(section: Mapping) -> int:
    """How far into the strong region the restart window re-reads."""
    regions = section.get("restart_reread_buffer_regions", 0)
    if regions not in RESTART_REREAD_BUFFER_REGIONS:
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


def _gap_computation(section: Mapping) -> tuple:
    """(gap_computation, gap_units); gap_units belongs to split_pair."""
    gap_computation = section.get("gap_computation", "serial")
    if gap_computation not in GAP_COMPUTATIONS:
        raise ValueError(
            "escalation.gap_computation must be one of "
            f"{GAP_COMPUTATIONS}, got {gap_computation!r}"
        )
    raw_units = section.get("gap_units", 1)
    gap_units = int(raw_units)
    if gap_units < 1:
        raise ValueError(
            f"escalation.gap_units needs at least 1 unit (got {gap_units})"
        )
    if "gap_units" in section and gap_computation != "split_pair":
        raise ValueError(
            "escalation.gap_units sizes the split_pair sibling pool; "
            f"gap_computation is {gap_computation!r}, so drop the key"
        )
    return gap_computation, gap_units


def _check_serial_only(
    gap_computation: str, threshold_source: str, double_window: bool
) -> None:
    """split_pair and online calibration are validated for serial switching."""
    if not double_window:
        return
    if gap_computation == "split_pair":
        raise ValueError(
            "split_pair is validated for serial switching only: a strong "
            "window absorbing windows whose gap join is still open is not "
            "supported yet"
        )
    if threshold_source == "online":
        raise ValueError(
            "threshold_source online is serial-only: an audit label "
            "compares one window's weak and strong committed "
            "observables, and a double-window strong result owns a "
            "larger extent than the audited window"
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
