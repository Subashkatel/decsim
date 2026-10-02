"""The event_count row: the simple count baseline.

The patch's detection events over the last patch_window_rounds, the
Hamming weight of one round's syndrome of Tan et al. (2406.18897 lines
956-966) widened to W rounds, and each position's events over the last
detector_window_rounds, Q3DE's active node counter (Suzuki et al.
2501.00331 lines 713-727). A count fires at the smallest value whose
tail under the usual rates, as TailLaw estimates it, is at most the
row's false-alarm rate per round. A firing round's onset is the round
less the window that fired (Q3DE lines 727-728).
"""

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy
import stim

import decsim.burst_detectors.burst_region as burst_region
import decsim.burst_detectors.burst_windows as burst_windows
import decsim.burst_detectors.event_count.tail_law as tail_law
import decsim.burst_detectors.flag_log as flag_log
import decsim.burst_detectors.layout as layout_module
import decsim.config as config
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.records.windows as window_records
import decsim.tables as tables
import decsim.trace_source as trace_source

# the fields that are a whole number of rounds, at least one
_ROUND_COUNT_KEYS = (
    "patch_window_rounds",
    "detector_window_rounds",
    "rate_tracking_rounds",
)
# Q3DE's counter confidence, 1 - alpha = 0.99 (2501.00331 lines
# 1076-1078): a position is in the flagged region when its count reaches
# the count its usual rate reaches this rarely.
REGION_FALSE_ALARMS = 0.01
# Operation ids are opaque identities chosen by the workload; Any names
# them in the port's signatures.


class EventCountBurstDetector:
    """Fires when a patch's or a position's event count outruns its usual rate.

    The simple count baseline. One counter set per operation, calibrated
    at build from that operation's own circuit, the calibration Q3DE
    assumes is known in advance (2501.00331 lines 1078-1080). Each
    position's usual rate is then tracked as an exponential average of
    its events and frozen while the detector fires, as Q3DE removes the
    flagged positions from its count for the burst's lifetime (lines
    730-733).

    Trace source: round_flagged(operation_id, round_index) when a round
    fires.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The event_count row's keys.

        patch_window_rounds is W of the patch count; detector_window_rounds
        is Q3DE's c_win (up to 300 in its evaluation, 2501.00331 lines
        1417-1420); false_alarms_per_round is the detector's budget per
        round per patch, half for each statistic, and the per-position
        half split evenly over the positions; rate_tracking_rounds is the
        time constant of each usual rate (DGR 2311.16214 lines 216-218:
        10,000 rounds estimate a rate); cycles_per_round on clock is what
        counting one round costs; raise_strong_priors gives a flagged
        strong window the burst priors.
        """

        patch_window_rounds: int = 4
        detector_window_rounds: int = 20
        false_alarms_per_round: float = 1e-6
        rate_tracking_rounds: int = 10_000
        clock: Optional[config.Clock] = None
        cycles_per_round: int = 0
        raise_strong_priors: bool = False
        # the word the yaml and the reports name this row by
        name = "event_count"

        def __post_init__(self) -> None:
            for key in _ROUND_COUNT_KEYS:
                value = getattr(self, key)
                config.check_whole_count(key, value, "rounds")
            _check_false_alarms_per_round(self.false_alarms_per_round)
            config.check_cycles("cycles_per_round", self.cycles_per_round)

        @classmethod
        def from_yaml(
            cls, section: Mapping, clocks: config.ClockSettings
        ) -> "EventCountBurstDetector.Settings":
            """A priced count names its clock; an unpriced one needs none."""
            clock = _count_clock(section, clocks)
            raise_priors = config.boolean(
                section, "burst_detector", "raise_strong_priors"
            )
            values = {
                **section,
                "clock": clock,
                "raise_strong_priors": raise_priors,
            }
            return tables.section_record("burst_detector", cls, values)

        def build(
            self,
            engine: engine_module.Engine,
            circuits: Mapping,
            round_period_microseconds: float,
            machine_clock: Optional[config.Clock],
        ) -> "EventCountBurstDetector":
            """The detector, calibrated from each scored operation's circuit.

            A count that names no clock counts on machine_clock.
            """
            clocked = config.with_machine_clock(self, machine_clock)
            return EventCountBurstDetector(
                clocked, engine, circuits, round_period_microseconds
            )

    def __init__(
        self,
        settings: "EventCountBurstDetector.Settings",
        engine: engine_module.Engine,
        circuits: Mapping,
        round_period_microseconds: float,
    ) -> None:
        """Counters for each operation: circuits maps an id to its circuit.

        Each entry is the operation's circuit and its round count. The
        budget is per round, so the round period does not enter.
        """
        del round_period_microseconds
        self.settings = settings
        self.engine = engine
        self.counts_by_operation = _counts_by_operation(circuits, settings)
        self.windows = burst_windows.BurstWindows(
            self.counts_by_operation, engine
        )
        self.trace = _TraceSources()

    def observe_round(
        self, operation_id: Any, round_index: int, events: Sequence[int]
    ) -> None:
        """Count one round; its verdict is published after the row's cost."""
        counts = self.counts_by_operation[operation_id]
        published_tick = counts.flags.publication_tick(
            self.engine.now,
            self.settings.clock,
            self.settings.cycles_per_round,
        )
        is_firing = counts.count_round(round_index, events, published_tick)
        if is_firing:
            self.trace.round_flagged.fire(operation_id, round_index)

    def is_burst_window(self, window: window_records.Window) -> bool:
        """Whether a flag published by now meets the window's rounds."""
        return self.windows.is_burst_window(window)

    def with_burst_priors(
        self,
        window: window_records.Window,
        model: fault_models.WindowErrorModel,
    ) -> fault_models.WindowErrorModel:
        """The model with the flagged region's priors raised, graph kept."""
        if not self.settings.raise_strong_priors:
            return model
        return self.windows.with_burst_priors(window, model)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the event_count row reports, as one member."""

    round_flagged: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _CountCalibration:
    """What the counters know of one operation before its first round.

    Its layout, and the tail laws of the patch count and each position's.
    """

    layout: layout_module.Layout
    patch_law: tail_law.TailLaw
    position_laws: tuple


class _OperationCounts:
    """One operation's counters, fed round by round."""

    def __init__(
        self,
        calibration: _CountCalibration,
        settings: EventCountBurstDetector.Settings,
    ) -> None:
        self.calibration = calibration
        self.settings = settings
        # one array of per-position event counts per counted round
        self.rows: list = []
        self.tracked_rates = numpy.array(calibration.layout.usual_rates)
        self.flags = flag_log.FlagLog(first_round=1)

    def count_round(
        self, round_index: int, events: Sequence[int], published_tick: int
    ) -> bool:
        """Add one round to the counters; whether it fires.

        A firing round's onset is the round less the longest window of
        a statistic firing on it, never before the first round.
        """
        next_round = len(self.rows) + 1
        assert round_index == next_round, (
            "an operation's rounds are counted in order"
        )
        layout = self.calibration.layout
        row = layout.position_counts(round_index, events)
        self.rows.append(row)
        firing_window = self._firing_window()
        if firing_window == 0:
            self.flags.record(None, published_tick)
            self._track(row)
            return False
        before_onset = round_index - firing_window
        onset = before_onset + 1
        flag = flag_log.Flag(max(onset, 1))
        self.flags.record(flag, published_tick)
        return True

    def region_of(self, episode: flag_log.Episode) -> burst_region.BurstRegion:
        """The positions anomalous at the flag's newest round, and their rate.

        Q3DE takes the anomalous region from the positions whose counter
        passes its threshold (2501.00331 lines 729-731); the rate is
        measured over the flagged rounds inside the counter's window.
        """
        newest_round = episode.last_firing_round
        position_counts = self._position_counts(newest_round)
        region_thresholds = self._position_thresholds(REGION_FALSE_ALARMS)
        is_in_region = position_counts >= region_thresholds
        detector_window = self.settings.detector_window_rounds
        before_window = newest_round - detector_window
        window_start = before_window + 1
        first_counted = max(episode.first_round, window_start)
        flagged_rows = self.rows[first_counted - 1 : newest_round]
        return burst_region.BurstRegion.of_episode(
            self.calibration.layout, episode, is_in_region, flagged_rows
        )

    def _firing_window(self) -> int:
        """The longest window of a statistic firing on the newest round."""
        firing_window = 0
        if self._patch_fires():
            firing_window = self.settings.patch_window_rounds
        if self._position_fires():
            detector_window = self.settings.detector_window_rounds
            firing_window = max(firing_window, detector_window)
        return firing_window

    def _patch_fires(self) -> bool:
        """Statistic (a): the patch's events over the last W rounds."""
        patch_window = self.settings.patch_window_rounds
        recent = self.rows[-patch_window:]
        patch_count = numpy.sum(recent)
        tracked_total = numpy.sum(self.tracked_rates)
        usual_rates = self.calibration.layout.usual_rates
        usual_total = numpy.sum(usual_rates)
        rate_scale = _rate_scales(tracked_total, usual_total)
        false_alarms = self.settings.false_alarms_per_round / 2
        law = self.calibration.patch_law
        threshold = law.threshold(false_alarms, rate_scale)
        return patch_count >= threshold

    def _position_fires(self) -> bool:
        """Statistic (b): any position's events over the last c_win rounds."""
        newest_round = len(self.rows)
        position_counts = self._position_counts(newest_round)
        position_count = len(self.calibration.layout.positions)
        budget = self.settings.false_alarms_per_round / 2
        false_alarms = budget / position_count
        thresholds = self._position_thresholds(false_alarms)
        is_anomalous = position_counts >= thresholds
        return is_anomalous.any()

    def _position_counts(self, newest_round: int):
        """Each position's events over the c_win rounds ending there."""
        detector_window = self.settings.detector_window_rounds
        window_start = newest_round - detector_window
        first_index = max(0, window_start)
        recent = self.rows[first_index:newest_round]
        return numpy.sum(recent, axis=0)

    def _position_thresholds(self, false_alarms: float):
        """Each position's threshold at its own tracked rate."""
        usual_rates = self.calibration.layout.usual_rates
        rate_scales = _rate_scales(self.tracked_rates, usual_rates)
        thresholds = []
        for law, rate_scale in zip(
            self.calibration.position_laws, rate_scales, strict=True
        ):
            threshold = law.threshold(false_alarms, rate_scale)
            thresholds.append(threshold)
        return numpy.asarray(thresholds)

    def _track(self, row) -> None:
        """Move each usual rate one step toward this round's events."""
        error = row - self.tracked_rates
        step = error / self.settings.rate_tracking_rounds
        self.tracked_rates += step


def _counts_by_operation(circuits: Mapping, settings) -> dict:
    counts_by_operation = {}
    for operation_id, (circuit, round_count) in circuits.items():
        _refuse_a_short_operation(operation_id, round_count, settings)
        circuit_text = str(circuit)
        calibration = _count_calibration(circuit_text, round_count, settings)
        counts_by_operation[operation_id] = _OperationCounts(
            calibration, settings
        )
    return counts_by_operation


def _refuse_a_short_operation(
    operation_id: Any, round_count: int, settings
) -> None:
    """A law is read off a slab of bulk rounds inside the operation.

    Round one has no bulk detector and the last round is kept off the
    slab, so the longer window needs two rounds more than itself.
    """
    longest_window = max(
        settings.patch_window_rounds, settings.detector_window_rounds
    )
    fewest_rounds = longest_window + 2
    if round_count >= fewest_rounds:
        return
    raise ValueError(
        f"operation {operation_id} has {round_count} rounds; the burst "
        f"detector's windows need at least {fewest_rounds}, two more "
        "than its longer window"
    )


@functools.lru_cache(maxsize=64)
def _count_calibration(
    circuit_text: str, round_count: int, settings
) -> _CountCalibration:
    """The layout and the laws of one circuit.

    Each law is read off one slab of bulk rounds centred in the
    operation, where a memory circuit's rounds are alike. The result
    depends on its arguments alone (the draws take a fixed seed), so a
    run's shots share it rather than draw it again.
    """
    circuit = stim.Circuit(circuit_text)
    layout = layout_module.Layout.from_circuit(circuit, round_count)
    patch_law = _patch_law(layout.faults, round_count, settings)
    position_laws = _position_laws(
        layout.faults_by_position, round_count, settings
    )
    return _CountCalibration(layout, patch_law, position_laws)


def _patch_law(faults: list, round_count: int, settings) -> tail_law.TailLaw:
    """Statistic (a)'s law: every position over W centred rounds."""
    patch_window = settings.patch_window_rounds
    first_round, last_round = _centred_slab(round_count, patch_window)
    priors, incidence = _slab(faults, first_round, last_round, None)
    false_alarms = settings.false_alarms_per_round / 2
    return tail_law.TailLaw.from_priors(priors, incidence, false_alarms)


def _position_laws(
    faults_by_position: list, round_count: int, settings
) -> tuple:
    """Statistic (b)'s laws: one position over c_win centred rounds."""
    detector_window = settings.detector_window_rounds
    first_round, last_round = _centred_slab(round_count, detector_window)
    budget = settings.false_alarms_per_round / 2
    position_count = len(faults_by_position)
    false_alarms = budget / position_count
    laws = []
    for position, faults in enumerate(faults_by_position):
        priors, incidence = _slab(faults, first_round, last_round, position)
        law = tail_law.TailLaw.from_priors(priors, incidence, false_alarms)
        laws.append(law)
    return tuple(laws)


def _centred_slab(round_count: int, window_rounds: int) -> tuple:
    rounds_total = round_count + window_rounds
    after_total = rounds_total + 1
    last_round = after_total // 2
    before_slab = last_round - window_rounds
    first_round = before_slab + 1
    return first_round, last_round


def _slab(faults: list, first_round: int, last_round: int, position) -> tuple:
    """The priors and 0/1 incidence of the faults over one slab.

    position None counts every position; otherwise that one alone.
    """
    column_by_slot: dict = {}
    priors = []
    columns_by_fault = []
    for fault in faults:
        counted = _counted_slots(fault, first_round, last_round, position)
        if not counted:
            continue
        columns = []
        for slot in counted:
            column = column_by_slot.setdefault(slot, len(column_by_slot))
            columns.append(column)
        priors.append(fault.prior)
        columns_by_fault.append(columns)
    incidence = _incidence(columns_by_fault, len(column_by_slot))
    return numpy.asarray(priors), incidence


def _counted_slots(
    fault: layout_module.Fault, first_round: int, last_round: int, position
) -> list:
    counted = []
    for slot in fault.slots:
        round_index, position_index = slot
        is_in_rounds = first_round <= round_index <= last_round
        is_at_position = position is None or position_index == position
        if is_in_rounds and is_at_position:
            counted.append(slot)
    return counted


def _incidence(columns_by_fault: list, column_count: int):
    fault_count = len(columns_by_fault)
    shape = (fault_count, column_count)
    incidence = numpy.zeros(shape, dtype=numpy.uint8)
    for row, columns in enumerate(columns_by_fault):
        incidence[row, columns] = 1
    return incidence


def _rate_scales(tracked_rates, calibrated_rates):
    """Each tracked rate over its calibrated one; one where that is zero.

    A calibrated rate of zero is a detector no fault reaches, whose law
    is the faultless one at any scale.
    """
    tracked = numpy.asarray(tracked_rates, dtype=numpy.float64)
    calibrated = numpy.asarray(calibrated_rates, dtype=numpy.float64)
    scales = numpy.ones_like(tracked)
    is_calibrated = calibrated > 0
    numpy.divide(tracked, calibrated, out=scales, where=is_calibrated)
    return scales


def _check_false_alarms_per_round(value) -> None:
    if config.is_number(value) and 0 < value < 1:
        return
    raise ValueError(
        "false_alarms_per_round must be a probability between 0 and 1, "
        f"written as a number (got {value!r})"
    )


def _count_clock(section: Mapping, clocks: config.ClockSettings):
    """event_count's clock; a priced count needs one."""
    name = section.get("clock")
    if name is not None:
        return clocks.clock(name)
    cycles = section.get("cycles_per_round", 0)
    if config.is_whole_count(cycles):
        raise ValueError(
            "burst_detector.cycles_per_round needs a clock: name the "
            "clocks domain its cycles are counted in"
        )
    return None
