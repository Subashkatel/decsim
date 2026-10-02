"""The switching settings: when a window is decoded again, and on what.

SwitchingSettings is the machine's switching slot, filled on a run that
decodes weak first and escalates a window to the strong decoder. The
yaml's `escalation` section is read here: its kind says which of the
machine's decode slots a run fills (ESCALATION_KINDS), and a switching
kind's keys become the record. A row of STRONG_WINDOW_SHAPES is the
geometry the strong tier re-decodes (Toshio et al. arXiv 2510.25222).
"""

import dataclasses
import pathlib
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any, Optional

import decsim.config as config
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.ports as ports
import decsim.tables as tables

# escalation.kind names one of these words: the decoder sections of the
# yaml a run of that kind keeps, the machine's slots it fills. A kind
# that keeps both decoders switches between them.
ESCALATION_KINDS = {
    "weak_baseline": ("weak_decoder",),
    "strong_only": ("strong_decoder",),
    "switching": ("weak_decoder", "strong_decoder"),
}
# The section keys every kind reads: the price of the verdict, which only
# a switching run charges.
_TIMING_KEYS = ("kind", "clock", "threshold_cycles", "switch_cycles")
# escalation.strong_window names one of these rows: the shape of the
# window the strong tier re-decodes. The yaml's name becomes the row's
# Settings record, which the switching part builds on the run's engine.
STRONG_WINDOW_SHAPES = {
    "redo_window": strong_window_shapes.RedoWindow,
    "double_window": strong_window_shapes.DoubleWindow,
}
# escalation.threshold_source names one of these rows: where the
# switching threshold of a sweep point comes from.
THRESHOLD_SOURCES = {
    "fixed": threshold_sources.FixedThreshold,
    "table": threshold_sources.TableThreshold,
    "online": threshold_sources.OnlineThreshold,
}
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


@dataclasses.dataclass(frozen=True)
class SwitchingSettings:
    """The machine's switching slot: weak first, escalate on low confidence.

    Read from the yaml's `escalation` section when its kind is switching
    (weak first, escalate serially on a small complementary gap, Toshio
    2510.25222 Sec. III A without the parallel head start).
    threshold is the Settings record of the row of THRESHOLD_SOURCES
    the yaml's threshold_source names, which keeps the weak result when
    its gap is at or above the threshold (the paper uses 20 dB): fixed
    holds it as given, table looks the sweep point up in an offline
    calibration csv (threshold_column, calibrated for this run's window
    geometry), online starts there and adapts it across a point's shots,
    serial switching only; run_both_at_once is Sec. III A's Step 1, the strong
    decoder started with the weak one and cancelled on confidence
    (false, the default, is the same section's on-demand variant, lines
    631-640); strong_window is the Settings record of the shape of the
    window the strong tier re-decodes, a row of STRONG_WINDOW_SHAPES
    (decsim/escalation/strong_window_shapes.py): redo_window, the
    default, re-decodes the escalated window's commit region with its
    past face pinned on the earlier neighbour's committed correction and
    one buffer ahead (Bombin et al. 2303.04846 lines 775-788 and
    1456-1458); double_window is the paper's Sec. III C scheme as
    it is stated, an r_com + 2 r_buf extent read with no context, both
    faces pinned (lines 1248-1259), whose record holds how many of the
    strong region's buffer regions the restarted weak window re-reads.
    confidence is the Settings record of the signal the weak tier
    reports and the threshold decides on (confidence/signals.py names
    the rows), and the build refuses a weak decoder whose decode cannot
    produce that signal's evidence; the yaml's
    confidence_walk_microseconds, the card that prices the signal's own
    computation on the weak unit, is a field of that record, null
    leaving each row on its own cost model.
    The complementary gap's two forced-class solves are two ordinary
    jobs of the weak pool, so weak_decoder.units alone decides whether
    they overlap. clock, threshold_cycles and switch_cycles price the
    verdict's threshold and switch logic; clock None is the machine's
    clock. The experiments layer sets the
    sweep point's threshold (at_sweep_point) and installs the point's
    online threshold source, the one instance every shot of the point
    shares (the threshold record's for_sweep_point,
    collect.Task.shot_settings).
    """

    # the confidence row's Settings record, opaque here: the record whose
    # build(weak_algorithm, threshold_nats) returns the signal
    confidence: Any
    # the threshold row's Settings record (threshold_sources.py), opaque
    # here: at_sweep_point, for_sweep_point and build answer for it
    threshold: Any
    clock: Optional[config.Clock] = None
    threshold_cycles: int = 0
    switch_cycles: int = 0
    run_both_at_once: bool = False
    # the strong window row's Settings record, opaque here: the record
    # whose build(engine) returns the shape
    strong_window: Any = strong_window_shapes.RedoWindow.Settings()
    # the burst detector row's Settings record, opaque here: the record
    # whose build(engine, circuits, round_period_microseconds,
    # machine_clock) returns the detector; None watches for no burst
    burst_detector: Any = None
    # the sweep point's live online source, shared by every shot of the
    # point and installed per shot by the experiments layer
    online_threshold: Optional[ports.ThresholdSource] = None

    def __post_init__(self) -> None:
        _check_verdict_cycles(self.threshold_cycles, self.switch_cycles)

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        base_directory: Optional[pathlib.Path],
        confidence_settings: Callable,
    ) -> Optional["SwitchingSettings"]:
        """The `escalation` section: the switching slot, None if it keeps one.

        Every key is checked at this boundary, so a kind that keeps one
        decoder still accepts the verdict's timing keys and refuses the
        confidence keys. confidence_settings turns escalation.confidence
        and its walk card into the row's record (confidence/signals.py
        confidence_settings), handed in by the caller, since the
        confidence package sits above this one.
        """
        kind = escalation_kind(section)
        clock = None
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        threshold_cycles = section.get("threshold_cycles", 0)
        switch_cycles = section.get("switch_cycles", 0)
        confidence_keys = set(section) - set(_TIMING_KEYS)
        if kind != "switching":
            _refuse_confidence_keys(kind, confidence_keys)
            _check_verdict_cycles(threshold_cycles, switch_cycles)
            return None
        return _switching_settings(
            section,
            base_directory,
            clock,
            threshold_cycles,
            switch_cycles,
            confidence_settings,
        )

    def at_sweep_point(self, resolved: Mapping) -> "SwitchingSettings":
        """The slot at one sweep point: a table's threshold looked up.

        The threshold record answers for itself (threshold_sources.py):
        a table finds the point's row in its csv, the other rows are
        the same at every point.
        """
        threshold = self.threshold.at_sweep_point(resolved)
        return dataclasses.replace(self, threshold=threshold)


def escalation_kind(section: Mapping) -> str:
    """The section's kind, its keys checked: which decode slots a run fills."""
    tables.refuse_unknown_keys("escalation", section, ESCALATION_KEYS)
    kind = section.get("kind", "weak_baseline")
    kinds = sorted(ESCALATION_KINDS)
    if isinstance(kind, (list, Mapping)) or kind not in ESCALATION_KINDS:
        raise ValueError(
            f"escalation.kind {kind!r} is not a row of its table; the rows "
            f"are {kinds}"
        )
    return kind


def _check_verdict_cycles(threshold_cycles, switch_cycles) -> None:
    """The verdict's two costs are whole cycle counts."""
    config.check_cycles("escalation.threshold_cycles", threshold_cycles)
    config.check_cycles("escalation.switch_cycles", switch_cycles)


def _refuse_confidence_keys(kind: str, confidence_keys: set) -> None:
    """A kind that keeps one decoder decides on no confidence."""
    if not confidence_keys:
        return
    listed = sorted(confidence_keys)
    raise ValueError(
        f"escalation.kind {kind} decides on no confidence; drop {listed}"
    )


def _switching_settings(
    section: Mapping,
    base_directory: Optional[pathlib.Path],
    clock: Optional[config.Clock],
    threshold_cycles: int,
    switch_cycles: int,
    confidence_settings: Callable,
) -> SwitchingSettings:
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
    threshold = _threshold(
        section, threshold_source, threshold_row, base_directory
    )
    named_confidence = section.get("confidence", "complementary_gap")
    walk_microseconds = _confidence_walk_microseconds(section)
    confidence = confidence_settings(named_confidence, walk_microseconds)
    run_both_at_once = config.boolean(section, "escalation", "run_both_at_once")
    named_window = section.get("strong_window", "redo_window")
    window_row = tables.row(
        STRONG_WINDOW_SHAPES, "escalation.strong_window", named_window
    )
    _check_serial_only(threshold_source, threshold_row, window_row)
    strong_window = _strong_window(section, named_window, window_row)
    return SwitchingSettings(
        confidence=confidence,
        threshold=threshold,
        clock=clock,
        threshold_cycles=threshold_cycles,
        switch_cycles=switch_cycles,
        run_both_at_once=run_both_at_once,
        strong_window=strong_window,
    )


def _threshold(
    section: Mapping,
    threshold_source: str,
    threshold_row,
    base_directory: Optional[pathlib.Path],
):
    """The Settings record of the threshold row the section names.

    A row that reads a calibration table takes the csv and its column,
    a row built once per sweep point takes the online card and its
    starting threshold, and any other row the threshold alone, in nats.
    """
    gap_threshold_db = _gap_threshold_db(
        section, threshold_source, threshold_row
    )
    online = _online_card(section, threshold_source, threshold_row)
    if threshold_row.reads_a_calibration_table:
        table = section["threshold_table"]
        raw_column = section.get("threshold_column", "gth_eq4_wilson")
        column = str(raw_column)
        return threshold_row.Settings(
            table=table, column=column, base_directory=base_directory
        )
    threshold_nats = threshold_sources.decibels_to_nats(gap_threshold_db)
    if threshold_row.built_per_sweep_point:
        return threshold_row.Settings.from_yaml(online, threshold_nats)
    return threshold_row.Settings(threshold_nats=threshold_nats)


def _strong_window(section: Mapping, named_window: str, window_row):
    """The strong window row's record, with the double window's width.

    Only a strong window that absorbs the weak windows it covers
    restarts the weak chain, so the key written beside any other row
    would be read by nothing and is refused.
    """
    if "restart_reread_buffer_regions" in section:
        _refuse_restart_without_absorption(window_row, named_window)
    if not window_row.absorbs_weak_windows:
        return window_row.Settings()
    regions = section.get("restart_reread_buffer_regions", 1)
    return window_row.Settings(restart_reread_buffer_regions=regions)


def _refuse_restart_without_absorption(row, strong_window: str) -> None:
    if row.absorbs_weak_windows:
        return
    raise ValueError(
        "escalation.restart_reread_buffer_regions sets how far the weak "
        f"window restarted past a strong region re-reads, and "
        f"escalation.strong_window {strong_window} restarts none; remove "
        "the key or name double_window"
    )


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
    return float(value)


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


def _online_card(
    section: Mapping, threshold_source: str, threshold_row
) -> Optional[Mapping]:
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
    return raw_online


def _check_serial_only(
    threshold_source: str, threshold_row, window_row
) -> None:
    """A row that audits by escalating is validated for serial switching."""
    if not window_row.absorbs_weak_windows:
        return
    if threshold_row.audits_by_escalating:
        raise ValueError(
            f"threshold_source {threshold_source} is serial-only: an "
            "audit label compares one window's weak and strong committed "
            "observables, and a strong window that absorbs the weak "
            "windows it covers owns a larger extent than the audited "
            "window"
        )


def _confidence_walk_microseconds(section: Mapping) -> Optional[float]:
    """The card that prices a confidence signal's own computation."""
    value = section.get("confidence_walk_microseconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "escalation.confidence_walk_microseconds must be a number of "
            f"microseconds, or null to leave the signal on its own cost "
            f"model (got {value!r})"
        )
    return float(value)
