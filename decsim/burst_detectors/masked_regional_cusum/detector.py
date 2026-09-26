"""The masked_regional_cusum row: a CUSUM bank over regions of the checks.

The bank (chart_bank.py) scores every round, and it alarms when any
chart reaches its group's threshold, the multichart CUSUM rule (Zhang
et al. 1410.8765 lines 338-350). The thresholds are read off quiet
shots of the operation's own circuit (thresholds.py). A flag's onset is
the CUSUM's own change point, the round after the leading chart was
last zero (Xie et al. 2104.04186 lines 323-325).
"""

import dataclasses
import functools
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
        shot's share of it the shot's length in time; calibration_shots
        are the quiet shots the thresholds are read from. clock,
        datapaths and pipeline_cycles are the timing card, and
        raise_strong_priors gives a flagged strong window the burst
        priors.
        """

        mask_window_rounds: int = 64
        mask_count: Optional[int] = 8
        mask_hold_rounds: int = 100
        region_radii: tuple = (0.0, 1.5, 2.3, 3.2)
        fault_rate_multipliers: tuple = (2.0, 4.0, 20.0)
        rate_tracking_rounds: int = 5000
        unmasked_share_floor: float = 0.2
        false_alarms_per_second: float = 0.03
        calibration_shots: int = 20_000
        clock: Optional[config.Clock] = None
        datapaths: int = 1
        pipeline_cycles: int = 30
        raise_strong_priors: bool = False

        @classmethod
        def from_yaml(
            cls, section: Mapping, clocks: config.ClockSettings
        ) -> "MaskedRegionalCusumBurstDetector.Settings":
            """The timing card prices on a clock; a null mask_count is none."""
            method = _method_keys(section)
            clock = _bank_clock(section, clocks)
            datapaths = config.whole_count(
                section, "burst_detector", "datapaths", 1, "datapaths"
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
                datapaths=datapaths,
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
        after_last_row = episode.last_firing_round - layout.first_bulk_round + 1
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
        shot_microseconds = round_count * round_period_microseconds
        shot_seconds = shot_microseconds / MICROSECONDS_PER_SECOND
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
    layout = layout_module.Layout.from_circuit(circuit, round_count)
    bank = chart_bank.ChartBank.for_layout(layout, settings)
    target_share = _shot_target(settings, shot_seconds)
    maxima = _quiet_block_maxima(circuit, layout, bank, settings)
    levels = thresholds.bank_thresholds(maxima, target_share)
    return _ChartCalibration(layout, bank, levels)


def _shot_target(settings, shot_seconds: float) -> float:
    """The share of quiet shots that may alarm: the rate times a shot's time.

    A shot is one calibration block: the bank alarms in a shot when any
    group reaches its level on any of its rounds.
    """
    target_share = settings.false_alarms_per_second * shot_seconds
    if target_share < 1.0:
        return target_share
    raise ValueError(
        "burst_detector.false_alarms_per_second times one shot's length, "
        f"{shot_seconds} s, is {target_share}: at least one false alarm a "
        "shot, which no threshold can promise; ask for a lower rate"
    )


def _quiet_block_maxima(circuit, layout, bank, settings):
    """(shots, groups): each quiet shot's largest group scores.

    The shots are drawn from the circuit itself, the background the
    source draws from; Kulldorff reads his scan's null law the same way,
    by Monte Carlo replicas (1997, section 5).
    """
    sampler = circuit.compile_detector_sampler(seed=CALIBRATION_SEED)
    detectors, rows_at, checks_at = _slot_columns(layout)
    shot_rounds = max(rows_at) + 1
    check_count = len(layout.positions)
    batches = []
    remaining = settings.calibration_shots
    while remaining > 0:
        shot_count = min(remaining, CALIBRATION_BATCH_SHOTS)
        samples = sampler.sample(shot_count)
        shape = (shot_count, shot_rounds, check_count)
        rows = numpy.zeros(shape, dtype=numpy.uint8)
        rows[:, rows_at, checks_at] = samples[:, detectors]
        maxima = bank.block_maxima(rows, layout.first_bulk_round)
        batches.append(maxima)
        remaining -= shot_count
    return numpy.concatenate(batches)


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
        "calibration_shots": config.whole_count(
            section, "burst_detector", "calibration_shots", 20_000, "shots"
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
    for key in ("datapaths", "pipeline_cycles"):
        if key in section:
            raise ValueError(
                f"burst_detector.{key} prices the chart bank, which needs "
                "a clock: name the clocks domain its cycles are counted in"
            )
    return None
