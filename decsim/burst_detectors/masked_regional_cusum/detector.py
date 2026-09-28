"""The masked_regional_cusum row: a CUSUM bank over regions of the checks.

The bank (chart_bank.py) scores every round, and it alarms when any
chart reaches its group's threshold, the multichart CUSUM rule (Zhang
et al. 1410.8765 lines 338-350). The thresholds are read off quiet
shots of the operation's own circuit (thresholds.py). A flag's onset is
the CUSUM's own change point, the round after the leading chart was
last zero (Xie et al. 2104.04186 lines 323-325). AlarmLines is the same
bank offline: whole shots of detection events scored at once, at
several false alarm rates.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy
import stim

import decsim.burst_detectors.burst_region as burst_region
import decsim.burst_detectors.burst_windows as burst_windows
import decsim.burst_detectors.flag_log as flag_log
import decsim.burst_detectors.layout as layout_module
import decsim.burst_detectors.masked_regional_cusum.chart_bank as chart_bank
import decsim.burst_detectors.masked_regional_cusum.thresholds as thresholds
import decsim.config as config
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.records.windows as window_records
import decsim.trace_source as trace_source

# The quiet shots are a Monte Carlo integral over the circuit's own
# model, so they take one fixed seed and every shot of a run gets the
# same thresholds.
CALIBRATION_SEED = 0
# Quiet shots drawn and scored together, which bounds the memory the
# calibration holds.
CALIBRATION_BATCH_SHOTS = 1000
MICROSECONDS_PER_SECOND = 1e6
# AlarmLines.first_alarm_rounds' round for a line that never fires
NO_ALARM = -1
# Operation ids are opaque identities chosen by the workload; Any names
# them in the port's signatures.


class MaskedRegionalCusumBurstDetector:
    """Fires when a region's detection events outrun its usual rate.

    One chart bank per operation, calibrated at build from that
    operation's own circuit, the calibration Q3DE assumes is known in
    advance (2501.00331 lines 1078-1080).

    Trace source: round_flagged(operation_id, round_index) when a round
    fires.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The masked_regional_cusum row's keys, defaulting to the method's.

        mask_window_rounds is the window of each check's repeat count
        and each pair's joint count; mask_count is the count at which
        either flags a check, None for never, and mask_hold_rounds how
        long a flag masks it. region_radii are the disc radii in grid
        units (a Stim distance over sqrt(2)); the whole patch is always
        a region, and no radii leave it the only one.
        fault_rate_multipliers are the designs k: a check whose usual
        rate is q0 fires with (1 - (1 - 2 q0)^k) / 2 under design k.
        rate_tracking_rounds is the time constant of each region's usual
        count, held while at most unmasked_share_floor of its usual rate
        is unmasked. false_alarms_per_second is the bank's budget, a
        shot's share of it the shot's length in time;
        calibration_shot_count is the quiet shots the thresholds are read
        from. clock,
        datapath_count and pipeline_cycles are the timing card, and
        raise_strong_priors gives a flagged strong window the burst
        priors. The papers cited fix the method, not these numbers: the
        defaults are the values the method was tuned at, so a study that
        changes them departs from that tuning.
        """

        mask_window_rounds: int = 64
        mask_count: Optional[int] = 8
        mask_hold_rounds: int = 100
        region_radii: tuple = (0.0, 1.5, 2.3, 3.2)
        fault_rate_multipliers: tuple = (2.0, 4.0, 20.0)
        rate_tracking_rounds: int = 5000
        unmasked_share_floor: float = 0.2
        false_alarms_per_second: float = 0.03
        calibration_shot_count: int = 20_000
        clock: Optional[config.Clock] = None
        datapath_count: int = 1
        pipeline_cycles: int = 30
        raise_strong_priors: bool = False

        @classmethod
        def from_yaml(
            cls, section: Mapping, clocks: config.ClockSettings
        ) -> "MaskedRegionalCusumBurstDetector.Settings":
            """The timing card prices on a clock; a null mask_count is none."""
            method = _method_keys(section)
            clock = _bank_clock(section, clocks)
            datapath_count = config.whole_count(
                section, "burst_detector", "datapath_count", 1, "datapaths"
            )
            pipeline_cycles = section.get("pipeline_cycles", 30)
            config.check_cycles(
                "burst_detector.pipeline_cycles", pipeline_cycles
            )
            raise_priors = config.boolean(
                section, "burst_detector", "raise_strong_priors"
            )
            return cls(
                clock=clock,
                datapath_count=datapath_count,
                pipeline_cycles=pipeline_cycles,
                raise_strong_priors=raise_priors,
                **method,
            )

    def __init__(
        self,
        settings: "MaskedRegionalCusumBurstDetector.Settings",
        engine: engine_module.Engine,
        circuits: Mapping,
        round_period_microseconds: float,
    ) -> None:
        """A chart bank for each operation: circuits maps an id to its circuit.

        Each entry is the operation's circuit and its round count; the
        round period gives a shot's length in time.
        """
        self.settings = settings
        self.engine = engine
        self.charts_by_operation = _charts_by_operation(
            circuits, settings, round_period_microseconds
        )
        self.windows = burst_windows.BurstWindows(
            self.charts_by_operation, engine
        )
        self.trace = _TraceSources()

    def observe_round(
        self, operation_id: Any, round_index: int, events: Sequence[int]
    ) -> None:
        """Score one round; its verdict is published after the row's cost."""
        charts = self.charts_by_operation[operation_id]
        if not charts.scores(round_index):
            return
        bank = charts.calibration.bank
        cycles = bank.cycles_per_round(self.settings)
        published_tick = charts.flags.publication_tick(
            self.engine.now, self.settings.clock, cycles
        )
        is_firing = charts.score_round(round_index, events, published_tick)
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
class AlarmLines:
    """One circuit's chart bank, read at several false alarm rates at once.

    The row's own bank, offline: a line is one rate's group levels, all
    read off the same quiet shots, and a whole stream of detection
    events is scored at once rather than round by round through
    observe_round. The lines score streams that run on from shot to
    shot, so they are calibrated on quiet streams in that state.
    """

    layout: layout_module.Layout
    bank: chart_bank.ChartBank
    # (lines, groups): each line's level for every group
    levels: numpy.ndarray

    @classmethod
    def calibrated(
        cls,
        circuit: stim.Circuit,
        round_count: int,
        settings: MaskedRegionalCusumBurstDetector.Settings,
        round_period_microseconds: float,
        false_alarms_per_second: Sequence[float],
        warm_up_shots: int,
    ) -> "AlarmLines":
        """The bank and one line per rate, read off warm quiet streams.

        CALIBRATION_BATCH_SHOTS quiet streams each score warm_up_shots
        shots whose maxima are dropped, then give one block maximum per
        shot, their state carried on, until the blocks number
        settings.calibration_shot_count. The online row calibrates on
        cold shots instead, since each machine shot starts its detector
        afresh. Settings' own rate is unread.
        """
        shot_seconds = _shot_seconds(round_count, round_period_microseconds)
        target_shares = []
        for rate in false_alarms_per_second:
            target_share = _shot_target(rate, shot_seconds)
            target_shares.append(target_share)
        layout = layout_module.Layout.from_circuit(circuit, round_count)
        bank = chart_bank.ChartBank.for_layout(layout, settings)
        maxima = _warm_block_maxima(
            circuit, layout, bank, settings, warm_up_shots
        )
        line_levels = []
        for target_share in target_shares:
            levels = thresholds.bank_thresholds(maxima, target_share)
            line_levels.append(levels)
        stacked = numpy.stack(line_levels)
        is_positive = stacked > 0
        assert is_positive.all(), "a level is a score a quiet shot passes"
        return cls(layout, bank, stacked)

    @classmethod
    def with_levels(
        cls,
        circuit: stim.Circuit,
        round_count: int,
        settings: MaskedRegionalCusumBurstDetector.Settings,
        levels: numpy.ndarray,
    ) -> "AlarmLines":
        """The bank with levels an earlier calibration saved."""
        layout = layout_module.Layout.from_circuit(circuit, round_count)
        bank = chart_bank.ChartBank.for_layout(layout, settings)
        return cls(layout, bank, levels)

    def new_state(self, stream_count: int) -> chart_bank.BankState:
        """Every stream's charts at zero, as the row's before its first round.

        The state is the bank's own, which score_ratios carries on.
        """
        first_round = self.layout.first_bulk_round
        return self.bank.new_state(stream_count, first_round)

    def score_ratios(
        self, state: chart_bank.BankState, events: numpy.ndarray
    ) -> numpy.ndarray:
        """(streams, bulk rounds, lines): each round's top score over level.

        events is (streams, detectors), one shot of the circuit per
        stream, in Stim's detector order. Its bulk rounds continue the
        state's streams, so shots scored one after another on one state
        are one long stream per row. The join leaves out two detector
        layers that are not bulk: each shot's first round, which compares
        against the prepared state, and its closing layer, rebuilt from
        the data-qubit readout. A line fires on a round where its ratio
        is at least one, the round the row flags at that line's rate.
        """
        rows = _bulk_rows(self.layout, events)
        stream_count, round_count, _ = rows.shape
        line_count = len(self.levels)
        shape = (stream_count, round_count, line_count)
        ratios = numpy.zeros(shape)
        for offset in range(round_count):
            round_index = self.layout.first_bulk_round + state.scored_rounds
            round_rows = rows[:, offset]
            group_scores = self.bank.score_round(state, round_rows, round_index)
            stream_scores = group_scores[:, None, :]
            group_ratios = stream_scores / self.levels
            ratios[:, offset] = numpy.max(group_ratios, axis=2)
        return ratios

    def first_alarm_rounds(
        self, ratios: numpy.ndarray, earliest_round: int
    ) -> numpy.ndarray:
        """(streams, lines): each line's first firing round from earliest on.

        ratios are one shot's, so its offset i is round first_bulk_round
        + i of that shot; NO_ALARM marks a line that never fires.
        """
        is_firing = ratios >= 1.0
        first_offset = earliest_round - self.layout.first_bulk_round
        watched = is_firing[:, first_offset:]
        has_alarm = numpy.any(watched, axis=1)
        first_watched = numpy.argmax(watched, axis=1)
        first_rounds = first_watched + earliest_round
        return numpy.where(has_alarm, first_rounds, NO_ALARM)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the masked_regional_cusum row reports, as one member."""

    round_flagged: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _ChartCalibration:
    """What the charts know of one operation before its first round."""

    layout: layout_module.Layout
    bank: chart_bank.ChartBank
    # one level per group, from quiet shots of the operation's circuit
    thresholds: numpy.ndarray


class _OperationCharts:
    """One operation's chart bank, scored round by round."""

    def __init__(self, calibration: _ChartCalibration) -> None:
        self.calibration = calibration
        first_round = calibration.layout.first_bulk_round
        self.state = calibration.bank.new_state(1, first_round)
        # one row of per-position events per scored round
        self.rows: list = []
        self.flags = flag_log.FlagLog(first_round)

    def scores(self, round_index: int) -> bool:
        """Whether the round holds bulk detectors, the rounds scored."""
        return round_index >= self.calibration.layout.first_bulk_round

    def score_round(
        self, round_index: int, events: Sequence[int], published_tick: int
    ) -> bool:
        """Score one round; whether any group reaches its threshold."""
        layout = self.calibration.layout
        scored_round = layout.first_bulk_round + len(self.rows)
        assert round_index == scored_round, (
            "an operation's rounds are scored in order"
        )
        row = layout.position_counts(round_index, events)
        rows = row[None, :]
        bank = self.calibration.bank
        group_scores = bank.score_round(self.state, rows, round_index)
        flag = self._flag(group_scores[0])
        self.rows.append(row)
        self.flags.record(flag, published_tick)
        return flag is not None

    def region_of(self, episode: flag_log.Episode) -> burst_region.BurstRegion:
        """The leading chart's region, and the rate it measured since onset."""
        layout = self.calibration.layout
        incidence = self.calibration.bank.incidence
        is_in_region = incidence[:, episode.region] > 0
        first_row = episode.first_round - layout.first_bulk_round
        last_row = episode.last_firing_round - layout.first_bulk_round
        after_last_row = last_row + 1
        flagged_rows = self.rows[first_row:after_last_row]
        return burst_region.BurstRegion.of_episode(
            layout, episode, is_in_region, flagged_rows
        )

    def _flag(self, group_scores) -> Optional[flag_log.Flag]:
        """The round's flag when any group reaches its threshold."""
        levels = self.calibration.thresholds
        is_firing = group_scores >= levels
        if not is_firing.any():
            return None
        bank = self.calibration.bank
        design, region = bank.leading_chart(self.state, levels)
        last_zero = self.state.last_zero_round[0, design, region]
        onset_round = int(last_zero) + 1
        return flag_log.Flag(onset_round, int(region))


def _charts_by_operation(
    circuits: Mapping, settings, round_period_microseconds: float
) -> dict:
    charts_by_operation = {}
    for operation_id, (circuit, round_count) in circuits.items():
        circuit_text = str(circuit)
        shot_seconds = _shot_seconds(round_count, round_period_microseconds)
        calibration = _chart_calibration(
            circuit_text, round_count, settings, shot_seconds
        )
        charts_by_operation[operation_id] = _OperationCharts(calibration)
    return charts_by_operation


@functools.lru_cache(maxsize=64)
def _chart_calibration(
    circuit_text: str, round_count: int, settings, shot_seconds: float
) -> _ChartCalibration:
    """The layout, the bank and the thresholds of one circuit.

    The result depends on its arguments alone (the quiet shots take a
    fixed seed), so a run's shots share it rather than draw it again;
    the cache lives in memory only.
    """
    circuit = stim.Circuit(circuit_text)
    rate = settings.false_alarms_per_second
    target_share = _shot_target(rate, shot_seconds)
    layout, bank, maxima = _calibration_maxima(circuit, round_count, settings)
    levels = thresholds.bank_thresholds(maxima, target_share)
    return _ChartCalibration(layout, bank, levels)


def _shot_seconds(round_count: int, round_period_microseconds: float) -> float:
    shot_microseconds = round_count * round_period_microseconds
    return shot_microseconds / MICROSECONDS_PER_SECOND


def _shot_target(false_alarms_per_second: float, shot_seconds: float) -> float:
    """The share of quiet shots that may alarm: the rate times a shot's time.

    A shot is one calibration block: the bank alarms in a shot when any
    group reaches its level on any of its rounds. A rate of zero or
    below has no level to read, since the thresholds divide by it.
    """
    if false_alarms_per_second <= 0:
        raise ValueError(
            "burst_detector.false_alarms_per_second must be a rate above "
            f"zero (got {false_alarms_per_second!r})"
        )
    target_share = false_alarms_per_second * shot_seconds
    if target_share < 1.0:
        return target_share
    raise ValueError(
        "burst_detector.false_alarms_per_second times one shot's length, "
        f"{shot_seconds} s, is {target_share}: at least one false alarm a "
        "shot, which no threshold can promise; ask for a lower rate"
    )


def _calibration_maxima(
    circuit: stim.Circuit, round_count: int, settings
) -> tuple:
    """The circuit's layout, its bank, and the quiet shots' block maxima."""
    layout = layout_module.Layout.from_circuit(circuit, round_count)
    bank = chart_bank.ChartBank.for_layout(layout, settings)
    maxima = _quiet_block_maxima(circuit, layout, bank, settings)
    return layout, bank, maxima


def _quiet_block_maxima(circuit, layout, bank, settings):
    """(shots, groups): each quiet shot's largest group scores.

    The shots are drawn from the circuit itself, the background the
    source draws from; Kulldorff reads his scan's null law the same way,
    by Monte Carlo replicas (1997, section 5).
    """
    sampler = circuit.compile_detector_sampler(seed=CALIBRATION_SEED)
    batches = []
    remaining = settings.calibration_shot_count
    while remaining > 0:
        shot_count = min(remaining, CALIBRATION_BATCH_SHOTS)
        samples = sampler.sample(shot_count)
        rows = _bulk_rows(layout, samples)
        state = bank.new_state(shot_count, layout.first_bulk_round)
        maxima = bank.block_maxima(state, rows, layout.first_bulk_round)
        batches.append(maxima)
        remaining -= shot_count
    return numpy.concatenate(batches)


def _warm_block_maxima(circuit, layout, bank, settings, warm_up_shots: int):
    """(blocks, groups): warm quiet streams' shot maxima, warm-up dropped."""
    sampler = circuit.compile_detector_sampler(seed=CALIBRATION_SEED)
    state = bank.new_state(CALIBRATION_BATCH_SHOTS, layout.first_bulk_round)
    for _ in range(warm_up_shots):
        _next_shot_maxima(sampler, layout, bank, state)
    exact_shots = settings.calibration_shot_count / CALIBRATION_BATCH_SHOTS
    stream_shots = math.ceil(exact_shots)
    batches = []
    for _ in range(stream_shots):
        maxima = _next_shot_maxima(sampler, layout, bank, state)
        batches.append(maxima)
    return numpy.concatenate(batches)


def _next_shot_maxima(sampler, layout, bank, state) -> numpy.ndarray:
    """Each warm stream's maxima over its next quiet shot."""
    samples = sampler.sample(CALIBRATION_BATCH_SHOTS)
    rows = _bulk_rows(layout, samples)
    return bank.block_maxima(state, rows, layout.first_bulk_round)


def _bulk_rows(layout: layout_module.Layout, samples) -> numpy.ndarray:
    """(shots, bulk rounds, checks): each bulk detector's event in its slot."""
    detectors, rows_at, checks_at = _slot_columns(layout)
    shot_rounds = max(rows_at) + 1
    check_count = len(layout.positions)
    shot_count = len(samples)
    shape = (shot_count, shot_rounds, check_count)
    rows = numpy.zeros(shape, dtype=numpy.uint8)
    rows[:, rows_at, checks_at] = samples[:, detectors]
    return rows


def _slot_columns(layout: layout_module.Layout) -> tuple:
    """Each bulk detector, and its scored-round offset and check."""
    slot_by_detector = layout.slot_by_detector
    detectors = list(slot_by_detector)
    rows_at = []
    checks_at = []
    for detector in detectors:
        round_index, position_index = slot_by_detector[detector]
        row_at = round_index - layout.first_bulk_round
        rows_at.append(row_at)
        checks_at.append(position_index)
    return detectors, rows_at, checks_at


def _method_keys(section: Mapping) -> dict:
    """The CUSUM's method numbers, each checked once at the yaml."""
    return {
        "mask_window_rounds": config.whole_count(
            section, "burst_detector", "mask_window_rounds", 64, "rounds"
        ),
        "mask_count": _mask_count(section),
        "mask_hold_rounds": config.whole_count(
            section, "burst_detector", "mask_hold_rounds", 100, "rounds"
        ),
        "region_radii": _radii(section),
        "fault_rate_multipliers": _multipliers(section),
        "rate_tracking_rounds": config.whole_count(
            section, "burst_detector", "rate_tracking_rounds", 5000, "rounds"
        ),
        "unmasked_share_floor": _share_floor(section),
        "false_alarms_per_second": _false_alarms_per_second(section),
        "calibration_shot_count": config.whole_count(
            section, "burst_detector", "calibration_shot_count", 20_000, "shots"
        ),
    }


def _mask_count(section: Mapping) -> Optional[int]:
    """The mask's count; null never masks, the plain regional CUSUM."""
    if section.get("mask_count", 8) is None:
        return None
    return config.whole_count(
        section, "burst_detector", "mask_count", 8, "firings"
    )


def _is_list_at_least(values, bound: float) -> bool:
    """Whether values is a yaml list of numbers, each at least bound."""
    if not isinstance(values, list):
        return False
    for value in values:
        if not config.is_number(value):
            return False
        if value < bound:
            return False
    return True


def _radii(section: Mapping) -> tuple:
    """The disc radii; none leaves the whole patch the only region."""
    radii = section.get("region_radii", [0.0, 1.5, 2.3, 3.2])
    if _is_list_at_least(radii, 0.0):
        return tuple(float(radius) for radius in radii)
    raise ValueError(
        "burst_detector.region_radii must be a list of disc radii, each a "
        f"number at least zero (got {radii!r})"
    )


def _multipliers(section: Mapping) -> tuple:
    multipliers = section.get("fault_rate_multipliers", [2.0, 4.0, 20.0])
    is_listed = _is_list_at_least(multipliers, 1.0)
    is_above_one = is_listed and 1.0 not in multipliers
    if is_above_one and multipliers:
        return tuple(float(multiplier) for multiplier in multipliers)
    raise ValueError(
        "burst_detector.fault_rate_multipliers must be a list of at least "
        "one design, each a number above one: a design at one is no burst "
        f"(got {multipliers!r})"
    )


def _share_floor(section: Mapping) -> float:
    value = section.get("unmasked_share_floor", 0.2)
    if config.is_number(value) and 0 <= value < 1:
        return float(value)
    raise ValueError(
        "burst_detector.unmasked_share_floor must be a share from 0 up to, "
        f"not including, 1 (got {value!r})"
    )


def _false_alarms_per_second(section: Mapping) -> float:
    value = section.get("false_alarms_per_second", 0.03)
    if config.is_number(value) and value > 0:
        return float(value)
    raise ValueError(
        "burst_detector.false_alarms_per_second must be a rate above zero, "
        f"written as a number (got {value!r})"
    )


def _bank_clock(section: Mapping, clocks: config.ClockSettings):
    """The chart bank's clock; its card's constants need one to price on."""
    name = section.get("clock")
    if name is not None:
        return clocks.clock(name)
    for key in ("datapath_count", "pipeline_cycles"):
        if key in section:
            raise ValueError(
                f"burst_detector.{key} prices the chart bank, which needs "
                "a clock: name the clocks domain its cycles are counted in"
            )
    return None
