"""The burst detectors: detection events scored against their usual rates.

A burst of errors raises the detection rate of the stabilisers it covers
for hundreds of rounds (Google 2408.13687 lines 386-391 and 2101-2119).
Both rows read each round's bulk detection events as they are formed,
keyed by stabiliser position.

event_count is the simple count baseline: the patch's detection events
over the last patch_window_rounds, the Hamming weight of one round's
syndrome of Tan et al. (2406.18897 lines 956-966) widened to W rounds,
and each position's events over the last detector_window_rounds, Q3DE's
active node counter (Suzuki et al. 2501.00331 lines 713-727). A count
fires at the smallest value whose tail under the usual rates, as
TailLaw estimates it, is at most the row's false-alarm rate per round.

masked_regional_cusum scores every round with a bank of Page CUSUMs,
W_n = (W_{n-1} + l(X_n))+ (Page 1954, as written in Xie et al.
2104.04186 lines 274-328), each on the events of one region: a disc
around a check at one of several radii, or the whole patch, the
circular zones of Kulldorff's spatial scan statistic (Commun. Stat.
26(6) 1997, section 2, zone collection 2). Each region is scored
against designs that multiply the fault rate by a known k, so a step is
Lucas's count-data term c log rho - (rho - 1) mu (as written in Ward et
al. 2208.01494 lines 274-296), and the bank alarms when any chart
reaches its group's threshold, the multichart CUSUM rule (Zhang et al.
1410.8765 lines 338-350). A check that behaves like a defect, firing in
consecutive rounds or with its pair too often, leaves the counts for a
hold time, as Q3DE removes detected positions for a burst's lifetime
(2501.00331 lines 729-733). The method and its numbers are the burst
study's mask_p8r8_D0_x2, its thresholds read by the study's rule off
quiet shots of the operation's own circuit.

A flag runs from its estimated onset to the last round the detector
still fires on: the count's onset is the firing round less the window
that fired (Q3DE lines 727-728), the CUSUM's is its own change point,
the round after the leading chart was last zero (2104.04186 lines
323-325). The switching policy escalates every window a published flag
meets, and the strong window model may raise the priors of the flagged
region's faults with its graph unchanged (IonQ 2608.25027 lines
334-340).
"""

import bisect
import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy
import scipy.special
import scipy.stats
import stim

import decsim.config as config
import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.engine as engine_module
import decsim.records.windows as window_records
import decsim.trace_source as trace_source

# Q3DE's counter confidence, 1 - alpha = 0.99 (2501.00331 lines
# 1076-1078): a position is in the flagged region when its count reaches
# the count its usual rate reaches this rarely.
REGION_FALSE_ALARMS = 0.01
# The draws per fault count that integrate a count's tail given that
# many faults; tests/escalation/test_burst_detectors.py holds the tail
# against Stim's sampled one and against exact tails.
DRAWS_PER_FAULT_COUNT = 4000
# The count's draws and the CUSUM's quiet shots are Monte Carlo
# integrals over the circuit's own model, so they take one fixed seed
# and every shot of a run gets the same thresholds.
CALIBRATION_SEED = 0
# The fault counts the law integrates cover a rate twice the calibrated
# one; past that the truncated remainder is counted as a firing, so the
# truncation can only raise the tail.
RATE_HEADROOM = 2.0
# The truncated remainder is kept this far below the smallest
# false-alarm rate the law answers for.
TRUNCATION_SHARE = 0.01
# Quiet shots drawn and scored together, which bounds the memory the
# CUSUM's calibration holds.
CALIBRATION_BATCH_SHOTS = 1000
# The study floors each usual rate here, so a check no fault reaches
# still has a finite design ratio (burst study pass4.py lines 125, 396).
SMALLEST_USUAL_RATE = 1e-6
# Region radii are in the study's grid units, where neighbouring checks
# sit sqrt(2) apart: Stim's rotated layout puts them 2 apart.
GRID_UNITS_PER_COORDINATE = math.sqrt(2) / 2
# A disc holds a check at its radius exactly, whatever the rounding.
RADIUS_ROUNDING = 1e-9
# On Stim's rotated surface code the two checks diagonal to a data
# qubit measure its basis (gen_surface_code.cc lines 220-262), so the
# pair one data-qubit flip fires together sits this far apart in both
# coordinates.
PAIR_OFFSET = 2.0
# A check never flagged is masked at no round.
NEVER_FLAGGED = -(10**9)
# The CUSUM calibration rule's constants (burst study harness/
# analysis.py GroupTail and bank_thresholds): the exponential tail is
# fitted to the largest TAIL_FIT_COUNT block maxima, a bank target is
# measured when the blocks hold MEASURED_ALARMS expected alarms, the
# shared level is bisected this many times, and a fitted tail is never
# flatter than SMALLEST_TAIL_SCALE.
TAIL_FIT_COUNT = 20
MEASURED_ALARMS = 5
SHARE_BISECTION_STEPS = 30
SMALLEST_TAIL_SCALE = 1e-9
MICROSECONDS_PER_SECOND = 1e6
# A prior is a flip probability and never passes one half.
MAXIMUM_PRIOR = 0.5
BISECTION_STEPS = 60


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

        @classmethod
        def from_yaml(
            cls, section: Mapping, clocks: config.ClockSettings
        ) -> "EventCountBurstDetector.Settings":
            """The section's keys, each refused by a sentence when wrong."""
            patch_window = _whole_count(
                section, "patch_window_rounds", 4, "rounds"
            )
            detector_window = _whole_count(
                section, "detector_window_rounds", 20, "rounds"
            )
            false_alarms = _false_alarms_per_round(section)
            tracking = _whole_count(
                section, "rate_tracking_rounds", 10_000, "rounds"
            )
            cycles = section.get("cycles_per_round", 0)
            config.check_cycles("burst_detector.cycles_per_round", cycles)
            clock = _count_clock(section, clocks, cycles)
            raise_priors = _boolean(section, "raise_strong_priors")
            return cls(
                patch_window_rounds=patch_window,
                detector_window_rounds=detector_window,
                false_alarms_per_round=false_alarms,
                rate_tracking_rounds=tracking,
                clock=clock,
                cycles_per_round=cycles,
                raise_strong_priors=raise_priors,
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
        self.trace = _TraceSources()

    def observe_round(
        self, operation_id: Any, round_index: int, events: Sequence[int]
    ) -> None:
        """Count one round; its verdict is published after the row's cost."""
        counts = self.counts_by_operation[operation_id]
        newest = counts.flags.newest_published_tick()
        published_tick = _publication_tick(
            self.engine,
            newest,
            self.settings.clock,
            self.settings.cycles_per_round,
        )
        is_firing = counts.count_round(round_index, events, published_tick)
        if is_firing:
            self.trace.round_flagged.fire(operation_id, round_index)

    def is_burst_window(self, window: window_records.Window) -> bool:
        """Whether a flag published by now meets the window's rounds."""
        counts = self.counts_by_operation[window.operation_id]
        episode = counts.flags.episode_meeting(window, self.engine.now)
        return episode is not None

    def with_burst_priors(self, window: window_records.Window, model):
        """The model with the flagged region's priors raised, graph kept."""
        if not self.settings.raise_strong_priors:
            return model
        counts = self.counts_by_operation[window.operation_id]
        episode = counts.flags.episode_meeting(window, self.engine.now)
        if episode is None:
            return model
        region = counts.region_of(episode)
        return region.raised(model)


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
        """The masked_regional_cusum row's keys, the study's values.

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
            """The section's keys, each refused by a sentence when wrong."""
            method = _method_keys(section)
            clock = _bank_clock(section, clocks)
            datapaths = _whole_count(section, "datapaths", 1, "datapaths")
            pipeline_cycles = section.get("pipeline_cycles", 30)
            config.check_cycles(
                "burst_detector.pipeline_cycles", pipeline_cycles
            )
            raise_priors = _boolean(section, "raise_strong_priors")
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
        self.trace = _TraceSources()

    def observe_round(
        self, operation_id: Any, round_index: int, events: Sequence[int]
    ) -> None:
        """Score one round; its verdict is published after the row's cost."""
        charts = self.charts_by_operation[operation_id]
        if not charts.scores(round_index):
            return
        newest = charts.flags.newest_published_tick()
        bank = charts.calibration.bank
        cycles = bank.cycles_per_round(self.settings)
        published_tick = _publication_tick(
            self.engine, newest, self.settings.clock, cycles
        )
        is_firing = charts.score_round(round_index, events, published_tick)
        if is_firing:
            self.trace.round_flagged.fire(operation_id, round_index)

    def is_burst_window(self, window: window_records.Window) -> bool:
        """Whether a flag published by now meets the window's rounds."""
        charts = self.charts_by_operation[window.operation_id]
        episode = charts.flags.episode_meeting(window, self.engine.now)
        return episode is not None

    def with_burst_priors(self, window: window_records.Window, model):
        """The model with the flagged region's priors raised, graph kept."""
        if not self.settings.raise_strong_priors:
            return model
        charts = self.charts_by_operation[window.operation_id]
        episode = charts.flags.episode_meeting(window, self.engine.now)
        if episode is None:
            return model
        region = charts.region_of(episode)
        return region.raised(model)


@dataclasses.dataclass(frozen=True)
class TailLaw:
    """The tail of one count under the usual rates.

    Each fault of prior p fires a Poisson number of times of rate
    -ln(1 - 2p) / 2, whose chance of being odd, (1 - e^(-2 rate)) / 2, is
    p: only a fault's parity reaches the detectors, so the counted
    detectors flip exactly as under Stim's independent faults (the
    odd-number law of Tan et al. 2406.18897 lines 956-960). The firings
    are one Poisson process of rate fault_rate, the rates' sum, and
    conditional_tails[n][k] is the chance that n firings, picked in
    proportion to the rates and XOR-ed onto the counted detectors, leave
    at least k of them flipped, so two faults that cancel on a shared
    detector count as Stim counts them. The tail is sum_n Poisson(n;
    rate) conditional_tails[n][k] plus the truncated remainder counted
    as a firing. It is exact but for two things: conditional_tails is
    drawn, DRAWS_PER_FAULT_COUNT draws per fault count, so the tail is
    an estimate with that sampling error, and the truncation raises it
    by at most the remainder. A law with no faults is a count that is
    always zero.
    """

    fault_rate: float
    conditional_tails: Any

    def tails(self, rate_scale: float) -> Any:
        """P(count >= k) for every k, with every fault's rate scaled.

        A scale s on the rates gives a fault of prior p the odd chance
        (1 - (1 - 2p)^s) / 2, the law of s copies of the fault; s = 1 is
        the calibrated law.
        """
        rate = self.fault_rate * rate_scale
        if rate == 0:
            return self.conditional_tails[0]
        fault_count_limit = len(self.conditional_tails)
        fault_counts = numpy.arange(fault_count_limit)
        # Poisson(n; rate) = exp(n log rate - rate - log n!), the form
        # that costs one vector of logs rather than scipy's per-call setup
        counts_plus_one = fault_counts + 1
        log_factorials = scipy.special.gammaln(counts_plus_one)
        log_rate = numpy.log(rate)
        log_weights = fault_counts * log_rate - rate - log_factorials
        weights = numpy.exp(log_weights)
        weight_total = numpy.sum(weights)
        # the weights past the last fault count, never below zero when
        # the sum rounds past one
        uncovered = 1.0 - weight_total
        remainder = max(uncovered, 0.0)
        drawn = weights @ self.conditional_tails
        return drawn + remainder

    def threshold(self, false_alarms: float, rate_scale: float) -> int:
        """The smallest count whose tail is at most false_alarms.

        One past the largest count when none is, a count no round
        reaches, so the statistic never fires.
        """
        tails = self.tails(rate_scale)
        is_rare_enough = tails <= false_alarms
        rare_counts = numpy.flatnonzero(is_rare_enough)
        if len(rare_counts) == 0:
            return len(tails)
        return int(rare_counts[0])


def tail_law(priors: Any, incidence: Any, smallest_false_alarms: float):
    """The tail law of one count, integrated by fault count.

    priors holds each fault's probability and incidence its row of 0/1
    over the counted detectors. The fault counts cover RATE_HEADROOM
    times the calibrated rate until the Poisson remainder is below
    TRUNCATION_SHARE of the smallest false-alarm rate asked for.
    """
    fault_rates = _parity_rates(priors)
    fault_rate = numpy.sum(fault_rates)
    if fault_rate == 0:
        return _faultless_law(incidence)
    headroom_rate = fault_rate * RATE_HEADROOM
    remainder_bound = smallest_false_alarms * TRUNCATION_SHARE
    largest_fault_count = _fault_count_bound(headroom_rate, remainder_bound)
    generator = numpy.random.default_rng(CALIBRATION_SEED)
    conditional_tails = _conditional_tails(
        fault_rates, incidence, largest_fault_count, generator
    )
    return TailLaw(fault_rate, conditional_tails)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event a burst detector reports, as one member."""

    round_flagged: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _Fault:
    """One error mechanism, as the detector sees it."""

    prior: float
    # (round, position index) of every bulk detector it flips
    slots: tuple


@dataclasses.dataclass(frozen=True)
class _Layout:
    """Where one operation's checks sit, and their usual rates.

    positions are the stabiliser positions, a bulk detector's first two
    Stim coordinates; position_by_value maps each round's events, in
    formation order, onto them, -1 for a detector that is not bulk.
    usual_rates and position_priors are one round's detection
    probability at each position and the priors of the faults behind it.
    """

    positions: tuple
    position_by_value: dict
    first_bulk_round: int
    usual_rates: Any
    position_priors: tuple

    def position_counts(self, round_index: int, events: Sequence[int]):
        """The round's bulk detection events, per position."""
        value_positions = self.position_by_value[round_index]
        assert len(events) == len(value_positions), (
            "the burst detector reads the round the source formed"
        )
        values = numpy.asarray(events, dtype=numpy.int64)
        is_bulk = value_positions >= 0
        bulk_positions = value_positions[is_bulk]
        bulk_values = values[is_bulk]
        position_count = len(self.positions)
        counts = numpy.zeros(position_count, dtype=numpy.int64)
        numpy.add.at(counts, bulk_positions, bulk_values)
        return counts

    def prior_scale(self, is_in_region: Any, measured_rate: float) -> float:
        """The factor on the region's priors that explains its event rate.

        Solved so the region's mean detection probability, (1 - prod(1 -
        2 s p)) / 2 over the faults behind each position (the odd-number
        marginal of Tan et al. 2406.18897 lines 956-960), equals the rate
        measured over the flagged rounds.
        """
        region_priors = self._region_priors(is_in_region)
        # a count flag may leave no position anomalous on its own: there
        # is then no region whose priors to raise
        if not region_priors:
            return 1.0
        usual_rate = _mean_detection_probability(region_priors, 1.0)
        # a region no fault reaches has no prior a scale could raise
        if usual_rate == 0:
            return 1.0
        if measured_rate <= usual_rate:
            return 1.0
        largest_scale = _saturating_scale(region_priors)
        saturated = _mean_detection_probability(region_priors, largest_scale)
        if measured_rate >= saturated:
            return largest_scale
        return _bisected_scale(region_priors, measured_rate, largest_scale)

    def _region_priors(self, is_in_region: Any) -> list:
        """The priors behind each position of the region."""
        region_priors = []
        for position, is_member in enumerate(is_in_region):
            if is_member:
                region_priors.append(self.position_priors[position])
        return region_priors


@dataclasses.dataclass(frozen=True)
class _Flag:
    """A firing round's estimated onset, and the region that fired.

    region is the CUSUM's leading chart's region; a count flag has none
    of its own and takes its region when the priors ask.
    """

    onset_round: int
    region: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class _Episode:
    """One run of firing rounds, from the estimated onset.

    is_open says the newest published round still fires, so the flag
    covers every later round too; region is the flag's region at the
    run's newest firing round.
    """

    first_round: int
    last_firing_round: int
    is_open: bool
    region: Optional[int]

    def meets(self, first_round: int, last_round: int) -> bool:
        """Whether the flag covers any round of [first_round, last_round]."""
        if self.first_round > last_round:
            return False
        return self.is_open or self.last_firing_round >= first_round


class _FlagLog:
    """One operation's verdicts and when each was published."""

    def __init__(self, first_round: int) -> None:
        self.first_round = first_round
        self.flags: list = []
        self.published_ticks: list = []

    def record(self, flag: Optional[_Flag], published_tick: int) -> None:
        self.flags.append(flag)
        self.published_ticks.append(published_tick)

    def newest_published_tick(self) -> int:
        """The tick the previous round was published at; 0 before any."""
        if not self.published_ticks:
            return 0
        return self.published_ticks[-1]

    def episode_meeting(
        self, window: window_records.Window, now: int
    ) -> Optional[_Episode]:
        """The newest published flag that meets the window, or None."""
        published_count = bisect.bisect_right(self.published_ticks, now)
        published = self.flags[:published_count]
        episodes = _episodes(published, self.first_round)
        for episode in reversed(episodes):
            if episode.meets(window.start_round, window.buffer_hi):
                return episode
        return None


@dataclasses.dataclass(frozen=True)
class _BurstRegion:
    """The positions and rounds whose faults get the burst priors."""

    positions: frozenset
    first_round: int
    # None while the flag is open: every later round is in the burst
    last_round: Optional[int]
    prior_scale: float

    def raised(self, model):
        """The model with every region fault's prior scaled, capped at 1/2.

        Only the prior vector changes: "we update only the corresponding
        entries of the prior vector ... No reconstruction of the Tanner
        graph" (IonQ 2608.25027 lines 334-340).
        """
        if self.prior_scale == 1.0:
            return model
        region_rows = self._region_rows(model)
        graphlike = self._raised_faults(model.graphlike_faults, region_rows)
        physical = self._raised_faults(model.physical_faults, region_rows)
        return dataclasses.replace(
            model, graphlike_faults=graphlike, physical_faults=physical
        )

    def _region_rows(self, model):
        """The model's rows at a region position inside the flagged rounds."""
        rows = []
        coordinates = model.detector_coordinates
        for row, detector_id in enumerate(model.detector_ids):
            position = model.defect_positions[detector_id]
            round_index = position[0]
            row_coordinates = coordinates[row]
            planar = tuple(row_coordinates[:2])
            if self._covers(planar, round_index):
                rows.append(row)
        return numpy.asarray(rows, dtype=numpy.int64)

    def _covers(self, planar: tuple, round_index: int) -> bool:
        if planar not in self.positions:
            return False
        if round_index < self.first_round:
            return False
        return self.last_round is None or round_index <= self.last_round

    def _raised_faults(self, faults, region_rows):
        if faults is None:
            return None
        region_check = faults.check[region_rows]
        hits = region_check.sum(axis=0)
        hit_counts = numpy.ravel(hits)
        is_touched = hit_counts > 0
        priors = numpy.array(faults.priors, dtype=numpy.float64)
        scaled = priors[is_touched] * self.prior_scale
        priors[is_touched] = numpy.minimum(scaled, MAXIMUM_PRIOR)
        return dataclasses.replace(faults, priors=priors)


def _publication_tick(
    engine: engine_module.Engine,
    newest_tick: int,
    clock: Optional[config.Clock],
    cycles: int,
) -> int:
    """One unit serves the rounds in order, each for cycles of its clock.

    A round starts when it is formed or when the round before it is
    published, whichever is later, and is published cycles later,
    charged from the edge at or after that tick (gem5's Clocked,
    decsim/config.py Clock). A unit with no clock or no cycles publishes
    at once.
    """
    start = max(engine.now, newest_tick)
    if clock is None or cycles == 0:
        return start
    return clock.edge(cycles, start)


@dataclasses.dataclass(frozen=True)
class _CountCalibration:
    """What the counters know of one operation before its first round.

    Its layout, and the tail laws of the patch count and each position's.
    """

    layout: _Layout
    patch_law: TailLaw
    position_laws: tuple


class _OperationCounts:
    """The counters of one operation, and what they published when."""

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
        self.flags = _FlagLog(first_round=1)

    def count_round(
        self, round_index: int, events: Sequence[int], published_tick: int
    ) -> bool:
        """Add one round to the counters; whether it fires.

        A firing round's onset is the round less the longest window of
        a statistic firing on it, never before the first round.
        """
        assert round_index == len(self.rows) + 1, (
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
        onset = round_index - firing_window + 1
        flag = _Flag(max(onset, 1))
        self.flags.record(flag, published_tick)
        return True

    def region_of(self, episode: _Episode) -> _BurstRegion:
        """The positions anomalous at the flag's newest round, and their rate.

        Q3DE takes the anomalous region from the positions whose counter
        passes its threshold (2501.00331 lines 729-731); the rate is
        measured over the flagged rounds inside the counter's window.
        """
        layout = self.calibration.layout
        newest_round = episode.last_firing_round
        position_counts = self._position_counts(newest_round)
        region_thresholds = self._position_thresholds(REGION_FALSE_ALARMS)
        is_in_region = position_counts >= region_thresholds
        detector_window = self.settings.detector_window_rounds
        window_start = newest_round - detector_window + 1
        first_counted = max(episode.first_round, window_start)
        flagged_rows = self.rows[first_counted - 1 : newest_round]
        measured_rate = _mean_rate(flagged_rows, is_in_region)
        prior_scale = layout.prior_scale(is_in_region, measured_rate)
        positions = _positions_where(layout.positions, is_in_region)
        last_round = episode.last_firing_round
        if episode.is_open:
            last_round = None
        return _BurstRegion(
            positions, episode.first_round, last_round, prior_scale
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
        step = (row - self.tracked_rates) / self.settings.rate_tracking_rounds
        self.tracked_rates += step


def _counts_by_operation(circuits: Mapping, settings) -> dict:
    """One calibrated counter set per operation."""
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
    table = _formation_table(circuit, round_count)
    positions, slot_by_detector = _bulk_slots(table)
    faults = _faults(circuit, slot_by_detector)
    faults_by_position = _faults_by_position(faults, len(positions))
    layout = _layout(table, positions, slot_by_detector, faults_by_position)
    patch_law = _patch_law(faults, round_count, settings)
    position_laws = _position_laws(faults_by_position, round_count, settings)
    return _CountCalibration(layout, patch_law, position_laws)


def _patch_law(faults: list, round_count: int, settings) -> TailLaw:
    """Statistic (a)'s law: every position over W centred rounds."""
    patch_window = settings.patch_window_rounds
    first_round, last_round = _centred_slab(round_count, patch_window)
    priors, incidence = _slab(faults, first_round, last_round, None)
    false_alarms = settings.false_alarms_per_round / 2
    return tail_law(priors, incidence, false_alarms)


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
        law = tail_law(priors, incidence, false_alarms)
        laws.append(law)
    return tuple(laws)


def _centred_slab(round_count: int, window_rounds: int) -> tuple:
    """The window_rounds rounds in the middle of the operation."""
    last_round = (round_count + window_rounds + 1) // 2
    first_round = last_round - window_rounds + 1
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
    fault: _Fault, first_round: int, last_round: int, position
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


def _fault_count_bound(rate: float, remainder_bound: float) -> int:
    """The fewest fault counts whose Poisson remainder is below the bound."""
    largest_fault_count = 1
    while scipy.stats.poisson.sf(largest_fault_count, rate) > remainder_bound:
        largest_fault_count += 1
    return largest_fault_count


def _parity_rates(priors):
    """Each fault's Poisson rate whose odd chance is its prior.

    A Poisson count of rate r is odd with chance (1 - e^(-2r)) / 2, which
    is p at r = -ln(1 - 2p) / 2.
    """
    survivals = 1.0 - 2.0 * priors
    log_survivals = numpy.log(survivals)
    return -log_survivals / 2.0


def _faultless_law(incidence) -> TailLaw:
    """The law of a count no fault reaches: zero, with tail one at k = 0."""
    column_count = incidence.shape[1]
    tail_width = column_count + 2
    tails = numpy.zeros((1, tail_width))
    tails[0, 0] = 1.0
    return TailLaw(0.0, tails)


def _conditional_tails(
    fault_rates, incidence, largest_fault_count: int, generator
):
    """P(count >= k | n firings) for n up to the bound, by drawing."""
    column_count = incidence.shape[1]
    packed = numpy.packbits(incidence, axis=1)
    choice_weights = fault_rates / numpy.sum(fault_rates)
    row_count = largest_fault_count + 1
    tail_width = column_count + 2
    tails = numpy.zeros((row_count, tail_width))
    tails[0, 0] = 1.0
    for fault_count in range(1, row_count):
        tails[fault_count] = _drawn_tail(
            packed, choice_weights, fault_count, tail_width, generator
        )
    return tails


def _drawn_tail(
    packed, choice_weights, fault_count: int, tail_width, generator
):
    """The tail of the flipped count when fault_count faults fire."""
    fault_total = len(choice_weights)
    draw_shape = (DRAWS_PER_FAULT_COUNT, fault_count)
    picks = generator.choice(fault_total, size=draw_shape, p=choice_weights)
    picked_rows = packed[picks]
    parity = numpy.bitwise_xor.reduce(picked_rows, axis=1)
    flipped_bits = numpy.unpackbits(parity, axis=1)
    # bincount converts its input to intp (numpy
    # _core/src/multiarray/compiled_base.c, arr_bincount); a sum of uint8
    # is uint64, which numpy 2.0 and 2.1 refuse to cast to intp.
    flipped_counts = flipped_bits.sum(axis=1, dtype=numpy.intp)
    histogram = numpy.bincount(flipped_counts, minlength=tail_width)
    reversed_histogram = histogram[::-1]
    reversed_tails = numpy.cumsum(reversed_histogram)
    at_least = reversed_tails[::-1]
    return at_least / DRAWS_PER_FAULT_COUNT


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


@dataclasses.dataclass
class _BankState:
    """The chart bank's state, one row per stream it scores.

    The rings hold the last mask_window_rounds rounds of each check's
    repeats and each pair's joint firings; scored_rounds places the next
    round in them.
    """

    previous_row: Any
    repeat_ring: Any
    joint_ring: Any
    repeat_counts: Any
    joint_counts: Any
    last_flagged_round: Any
    usual_counts: Any
    scores: Any
    last_zero_round: Any
    scored_rounds: int = 0


@dataclasses.dataclass(frozen=True)
class _ChartBank:
    """The regions, the designs and the pairs of one operation's checks.

    incidence is (checks, regions), region_scales names each region's
    radius (len(radii) for the whole patch), and pair_incidence is
    (pairs, checks). A group is one design and one radius: its score is
    the largest of its regions', design-major.
    """

    settings: MaskedRegionalCusumBurstDetector.Settings
    usual_rates: Any
    incidence: Any
    region_rates: Any
    region_scales: Any
    pair_incidence: Any
    log_ratios: Any
    ratio_excesses: Any

    def group_count(self) -> int:
        design_count = len(self.settings.fault_rate_multipliers)
        scale_count = len(self.settings.region_radii) + 1
        return design_count * scale_count

    def cycles_per_round(self, settings) -> int:
        """ceil(regions x designs / datapaths) + pipeline_cycles.

        A bank of U chart updates over P datapaths takes one frame a
        round when U / P + L fits the round (burst study hw/BUDGET.md K1
        and K10). U is the algorithm's own count, one update per region
        and design; P and L are the hardware's, pencil figures until a
        design is built.
        """
        region_count = self.incidence.shape[1]
        design_count = len(settings.fault_rate_multipliers)
        updates = region_count * design_count
        updates_per_datapath = updates / settings.datapaths
        update_cycles = math.ceil(updates_per_datapath)
        return update_cycles + settings.pipeline_cycles

    def new_state(self, stream_count: int, first_round: int) -> _BankState:
        """Every chart at zero before first_round, every mu its usual rate."""
        check_count, region_count = self.incidence.shape
        pair_count = self.pair_incidence.shape[0]
        window = self.settings.mask_window_rounds
        design_count = len(self.settings.fault_rate_multipliers)
        check_shape = (stream_count, check_count)
        pair_shape = (stream_count, pair_count)
        chart_shape = (stream_count, design_count, region_count)
        check_ring_shape = (window, *check_shape)
        pair_ring_shape = (window, *pair_shape)
        before_first_round = first_round - 1
        previous_row = numpy.zeros(check_shape, numpy.uint8)
        repeat_ring = numpy.zeros(check_ring_shape, numpy.uint8)
        joint_ring = numpy.zeros(pair_ring_shape, numpy.uint8)
        repeat_counts = numpy.zeros(check_shape, int)
        joint_counts = numpy.zeros(pair_shape, int)
        last_flagged_round = numpy.full(check_shape, NEVER_FLAGGED)
        usual_counts = numpy.tile(self.region_rates, (stream_count, 1))
        scores = numpy.zeros(chart_shape)
        last_zero_round = numpy.full(chart_shape, before_first_round)
        return _BankState(
            previous_row=previous_row,
            repeat_ring=repeat_ring,
            joint_ring=joint_ring,
            repeat_counts=repeat_counts,
            joint_counts=joint_counts,
            last_flagged_round=last_flagged_round,
            usual_counts=usual_counts,
            scores=scores,
            last_zero_round=last_zero_round,
        )

    def score_round(self, state: _BankState, rows, round_index: int):
        """One round of every stream: mask, then score; the group scores.

        rows is (streams, checks) of 0/1. The score is read after the
        update, and the usual count moves after the score (burst study
        PROTOCOL.md section 18, "The exact rules of mask_p8r8_D0_x2").
        """
        is_unmasked = self._unmasked(state, rows, round_index)
        self._score(state, rows, is_unmasked, round_index)
        return self._group_scores(state)

    def block_maxima(self, rows, first_round: int):
        """Each stream's largest group scores over its rounds, from cold.

        rows is (streams, rounds, checks); the streams start together
        at first_round.
        """
        stream_count, round_count, _ = rows.shape
        state = self.new_state(stream_count, first_round)
        group_count = self.group_count()
        maxima = numpy.zeros((stream_count, group_count))
        for offset in range(round_count):
            round_index = first_round + offset
            round_rows = rows[:, offset]
            group_scores = self.score_round(state, round_rows, round_index)
            numpy.maximum(maxima, group_scores, out=maxima)
        return maxima

    def leading_chart(self, state: _BankState, thresholds) -> tuple:
        """The first stream's chart with the largest score over its level."""
        design_count = len(self.settings.fault_rate_multipliers)
        scale_count = len(self.settings.region_radii) + 1
        group_levels = numpy.reshape(thresholds, (design_count, scale_count))
        chart_levels = group_levels[:, self.region_scales]
        ratios = state.scores[0] / chart_levels
        flat_index = numpy.argmax(ratios)
        return numpy.unravel_index(flat_index, ratios.shape)

    def _unmasked(self, state: _BankState, rows, round_index: int):
        """Which checks count this round: none flagged in the hold.

        A repeat is a firing now and in the round before; a pair's joint
        firing is both its checks now. Either count over the window at
        mask_count flags the check, and a flag masks it for this round
        and the next mask_hold_rounds. A mask_count of None masks none.
        """
        settings = self.settings
        if settings.mask_count is None:
            return numpy.ones(rows.shape, dtype=bool)
        slot = state.scored_rounds % settings.mask_window_rounds
        repeats = rows & state.previous_row
        state.repeat_counts += repeats
        state.repeat_counts -= state.repeat_ring[slot]
        state.repeat_ring[slot] = repeats
        pair_rows = rows @ self.pair_incidence.T
        joints = pair_rows == 2
        state.joint_counts += joints
        state.joint_counts -= state.joint_ring[slot]
        state.joint_ring[slot] = joints
        state.previous_row = rows
        is_repeating = state.repeat_counts >= settings.mask_count
        is_pair_flagged = state.joint_counts >= settings.mask_count
        pair_flags = is_pair_flagged @ self.pair_incidence
        is_flagged = is_repeating | (pair_flags > 0)
        state.last_flagged_round[is_flagged] = round_index
        rounds_since_flag = round_index - state.last_flagged_round
        return rounds_since_flag > settings.mask_hold_rounds

    def _score(self, state: _BankState, rows, is_unmasked, round_index):
        """Every chart's CUSUM step, then each region's usual count.

        c counts the region's unmasked firings and f is the unmasked
        share of its usual rate, so its expected count is mu f and the
        step is Lucas's c log rho - (rho - 1) mu f.
        """
        counted_rows = rows * is_unmasked
        counts = counted_rows @ self.incidence
        unmasked_rates = is_unmasked * self.usual_rates
        unmasked_region_rates = unmasked_rates @ self.incidence
        unmasked_shares = unmasked_region_rates / self.region_rates
        expected = state.usual_counts * unmasked_shares
        evidence = counts[:, None, :] * self.log_ratios
        drift = expected[:, None, :] * self.ratio_excesses
        steps = evidence - drift
        state.scores += steps
        is_at_zero = state.scores <= 0.0
        state.scores[is_at_zero] = 0.0
        state.last_zero_round[is_at_zero] = round_index
        self._track(state, counts, unmasked_shares)
        state.scored_rounds += 1

    def _track(self, state: _BankState, counts, unmasked_shares) -> None:
        """Move mu toward c / f by 1 / rate_tracking_rounds, when f > floor.

        The count is scaled back to the whole region; with too little of
        the region unmasked the scaled count is noise, so mu holds.
        """
        is_tracked = unmasked_shares > self.settings.unmasked_share_floor
        whole_counts = numpy.zeros_like(counts)
        numpy.divide(
            counts, unmasked_shares, out=whole_counts, where=is_tracked
        )
        errors = whole_counts - state.usual_counts
        steps = errors / self.settings.rate_tracking_rounds
        tracked_steps = steps * is_tracked
        state.usual_counts += tracked_steps

    def _group_scores(self, state: _BankState):
        """(streams, groups): each design's largest score at each radius."""
        stream_count, design_count, _ = state.scores.shape
        scale_count = len(self.settings.region_radii) + 1
        shape = (stream_count, design_count, scale_count)
        group_scores = numpy.zeros(shape)
        for scale in range(scale_count):
            is_at_scale = self.region_scales == scale
            scale_scores = state.scores[:, :, is_at_scale]
            group_scores[:, :, scale] = numpy.max(scale_scores, axis=2)
        group_count = design_count * scale_count
        return numpy.reshape(group_scores, (stream_count, group_count))


@dataclasses.dataclass(frozen=True)
class _ChartCalibration:
    """What the charts know of one operation before its first round."""

    layout: _Layout
    bank: _ChartBank
    # one level per group, from quiet shots of the operation's circuit
    thresholds: Any


class _OperationCharts:
    """The chart bank of one operation, and what it published when."""

    def __init__(self, calibration: _ChartCalibration) -> None:
        self.calibration = calibration
        first_round = calibration.layout.first_bulk_round
        self.state = calibration.bank.new_state(1, first_round)
        # one row of per-position events per scored round
        self.rows: list = []
        self.flags = _FlagLog(first_round)

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

    def region_of(self, episode: _Episode) -> _BurstRegion:
        """The leading chart's region, and the rate it measured since onset."""
        layout = self.calibration.layout
        incidence = self.calibration.bank.incidence
        is_in_region = incidence[:, episode.region] > 0
        first_row = episode.first_round - layout.first_bulk_round
        after_last_row = episode.last_firing_round - layout.first_bulk_round + 1
        flagged_rows = self.rows[first_row:after_last_row]
        measured_rate = _mean_rate(flagged_rows, is_in_region)
        prior_scale = layout.prior_scale(is_in_region, measured_rate)
        positions = _positions_where(layout.positions, is_in_region)
        last_round = episode.last_firing_round
        if episode.is_open:
            last_round = None
        return _BurstRegion(
            positions, episode.first_round, last_round, prior_scale
        )

    def _flag(self, group_scores) -> Optional[_Flag]:
        """The round's flag when any group reaches its threshold."""
        thresholds = self.calibration.thresholds
        is_firing = group_scores >= thresholds
        if not is_firing.any():
            return None
        bank = self.calibration.bank
        design, region = bank.leading_chart(self.state, thresholds)
        last_zero = self.state.last_zero_round[0, design, region]
        onset_round = int(last_zero) + 1
        return _Flag(onset_round, int(region))


def _charts_by_operation(
    circuits: Mapping, settings, round_period_microseconds: float
) -> dict:
    """One calibrated chart bank per operation."""
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
    table = _formation_table(circuit, round_count)
    positions, slot_by_detector = _bulk_slots(table)
    faults = _faults(circuit, slot_by_detector)
    faults_by_position = _faults_by_position(faults, len(positions))
    layout = _layout(table, positions, slot_by_detector, faults_by_position)
    bank = _chart_bank(layout, settings)
    target_share = _shot_target(settings, shot_seconds)
    maxima = _quiet_block_maxima(
        circuit, layout, bank, slot_by_detector, settings
    )
    thresholds = _bank_thresholds(maxima, target_share)
    return _ChartCalibration(layout, bank, thresholds)


def _chart_bank(layout: _Layout, settings) -> _ChartBank:
    """The regions, the pairs and the designs' log ratios.

    A design multiplies each fault's rate by k, so a check of usual rate
    q0 fires with q1 = (1 - (1 - 2 q0)^k) / 2 (k chances combined by
    parity), and a region's ratio is rho = sum q1 / sum q0.
    """
    usual_rates = numpy.maximum(layout.usual_rates, SMALLEST_USUAL_RATE)
    incidence, region_scales = _regions(layout.positions, settings)
    pair_incidence = _pair_incidence(layout.positions)
    region_rates = usual_rates @ incidence
    survivals = 1.0 - 2.0 * usual_rates
    multipliers = numpy.asarray(settings.fault_rate_multipliers)
    design_survivals = survivals[None, :] ** multipliers[:, None]
    design_rates = (1.0 - design_survivals) / 2.0
    design_region_rates = design_rates @ incidence
    ratios = design_region_rates / region_rates
    log_ratios = numpy.log(ratios)
    ratio_excesses = ratios - 1.0
    return _ChartBank(
        settings=settings,
        usual_rates=usual_rates,
        incidence=incidence,
        region_rates=region_rates,
        region_scales=region_scales,
        pair_incidence=pair_incidence,
        log_ratios=log_ratios,
        ratio_excesses=ratio_excesses,
    )


def _regions(positions: tuple, settings) -> tuple:
    """Discs of every radius around every check, then the whole patch.

    Returns incidence (checks, regions) of 0/1 and each region's radius
    index, in the study's order (burst study harness layouts.py
    disc_regions).
    """
    coordinates = numpy.asarray(positions, dtype=float)
    grid = coordinates * GRID_UNITS_PER_COORDINATE
    offsets = grid[:, None, :] - grid[None, :, :]
    squared_offsets = offsets**2
    squared_distances = numpy.sum(squared_offsets, axis=2)
    columns = []
    region_scales = []
    for scale, radius in enumerate(settings.region_radii):
        reach = radius * radius + RADIUS_ROUNDING
        discs = squared_distances <= reach
        columns.append(discs)
        disc_scales = [scale] * len(positions)
        region_scales.extend(disc_scales)
    whole_patch = numpy.ones((len(positions), 1), dtype=bool)
    columns.append(whole_patch)
    region_scales.append(len(settings.region_radii))
    incidence = numpy.concatenate(columns, axis=1)
    return incidence.astype(float), numpy.asarray(region_scales)


def _pair_incidence(positions: tuple):
    """(pairs, checks): the two same-basis checks beside each data qubit."""
    coordinates = numpy.asarray(positions, dtype=float)
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distances = numpy.abs(offsets)
    is_pair_offset = distances == PAIR_OFFSET
    is_diagonal = numpy.all(is_pair_offset, axis=2)
    upper = numpy.triu(is_diagonal)
    first_checks, second_checks = numpy.nonzero(upper)
    pair_count = len(first_checks)
    pair_indices = numpy.arange(pair_count)
    shape = (pair_count, len(positions))
    pair_incidence = numpy.zeros(shape, dtype=numpy.uint8)
    pair_incidence[pair_indices, first_checks] = 1
    pair_incidence[pair_indices, second_checks] = 1
    return pair_incidence


def _shot_target(settings, shot_seconds: float) -> float:
    """The share of quiet shots that may alarm: the rate times a shot's time.

    A shot is one calibration block, as the study counts a Willow shot
    as one (burst study PROTOCOL.md section 1).
    """
    target_share = settings.false_alarms_per_second * shot_seconds
    if target_share < 1.0:
        return target_share
    raise ValueError(
        "burst_detector.false_alarms_per_second times one shot's length, "
        f"{shot_seconds} s, is {target_share}: at least one false alarm a "
        "shot, which no threshold can promise; ask for a lower rate"
    )


def _quiet_block_maxima(circuit, layout, bank, slot_by_detector, settings):
    """(shots, groups): each quiet shot's largest group scores.

    The shots are drawn from the circuit itself, the background the
    source draws from; Kulldorff reads his scan's null law the same way,
    by Monte Carlo replicas (1997, section 5).
    """
    sampler = circuit.compile_detector_sampler(seed=CALIBRATION_SEED)
    detectors, rows_at, checks_at = _slot_columns(slot_by_detector, layout)
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


def _slot_columns(slot_by_detector: dict, layout: _Layout) -> tuple:
    """Each bulk detector, and its scored-round offset and check."""
    detectors = list(slot_by_detector)
    rows_at = []
    checks_at = []
    for detector in detectors:
        round_index, position_index = slot_by_detector[detector]
        row_at = round_index - layout.first_bulk_round
        rows_at.append(row_at)
        checks_at.append(position_index)
    return detectors, rows_at, checks_at


@dataclasses.dataclass(frozen=True)
class _GroupTail:
    """One group's block maxima: their observed tail and a fitted one.

    level(share) is the smallest observed maximum whose tail is at most
    share while the blocks reach that share (share x blocks at least the
    fitted count); past it, the exponential fitted to the largest
    maxima's excesses over the next one, u + beta ln(k / (n share))
    (burst study PROTOCOL.md section 2).
    """

    values: Any
    tails: Any
    block_count: int
    fitted_count: int
    fit_base: float
    fit_scale: float

    def level(self, share: float) -> float:
        expected_blocks = share * self.block_count
        if expected_blocks >= self.fitted_count:
            return self._observed_level(share)
        reach = self.fitted_count / expected_blocks
        return self.fit_base + self.fit_scale * math.log(reach)

    def _observed_level(self, share: float) -> float:
        negated_tails = -self.tails
        negated_share = -share
        index = numpy.searchsorted(negated_tails, negated_share, side="left")
        if index < len(self.values):
            return float(self.values[index])
        past_largest = self.values[-1] + 1.0
        return float(past_largest)


def _group_tail(maxima) -> _GroupTail:
    values, counts = numpy.unique(maxima, return_counts=True)
    block_count = len(maxima)
    reversed_counts = counts[::-1]
    reversed_tails = numpy.cumsum(reversed_counts)
    tails = reversed_tails[::-1] / block_count
    ascending = numpy.sort(maxima)
    descending = ascending[::-1]
    largest_fitted_count = block_count - 1
    fitted_count = min(TAIL_FIT_COUNT, largest_fitted_count)
    fit_base = descending[fitted_count]
    excesses = descending[:fitted_count] - fit_base
    mean_excess = numpy.mean(excesses)
    fit_scale = max(float(mean_excess), SMALLEST_TAIL_SCALE)
    return _GroupTail(
        values, tails, block_count, fitted_count, fit_base, fit_scale
    )


def _bank_thresholds(maxima, target_share: float):
    """One level per group, sharing one tail share, so the bank meets target.

    The shared share is bisected so the share of blocks where any group
    reaches its level is at most the target. A target the blocks hold
    fewer than MEASURED_ALARMS alarms at keeps the bank-to-group ratio
    found where they hold that many (burst study PROTOCOL.md section 2;
    harness/analysis.py bank_thresholds).
    """
    block_count, group_count = maxima.shape
    tails = []
    for group in range(group_count):
        tail = _group_tail(maxima[:, group])
        tails.append(tail)
    # one group is the bank, so its level is the target's own
    if group_count == 1:
        return _levels(tails, target_share)
    expected_alarms = target_share * block_count
    if expected_alarms >= MEASURED_ALARMS:
        group_share = _shared_share(maxima, tails, target_share)
        return _levels(tails, group_share)
    measured_share = MEASURED_ALARMS / block_count
    group_share = _shared_share(maxima, tails, measured_share)
    bank_ratio = measured_share / group_share
    extrapolated_share = target_share / bank_ratio
    return _levels(tails, extrapolated_share)


def _shared_share(maxima, tails: list, bank_share: float) -> float:
    """The largest group share whose bank alarms at most bank_share."""
    group_count = len(tails)
    low = bank_share / group_count
    high = bank_share
    while _bank_alarm_share(maxima, tails, low) > bank_share:
        low /= 2
    for _ in range(SHARE_BISECTION_STEPS):
        product = low * high
        middle = math.sqrt(product)
        if _bank_alarm_share(maxima, tails, middle) <= bank_share:
            low = middle
        else:
            high = middle
    return low


def _bank_alarm_share(maxima, tails: list, group_share: float) -> float:
    levels = _levels(tails, group_share)
    is_reached = maxima >= levels
    is_alarmed = numpy.any(is_reached, axis=1)
    alarmed_share = numpy.mean(is_alarmed)
    return float(alarmed_share)


def _levels(tails: list, group_share: float):
    levels = []
    for tail in tails:
        level = tail.level(group_share)
        levels.append(level)
    return numpy.asarray(levels)


def _formation_table(circuit: stim.Circuit, round_count: int):
    """The recipes a Stim source forms this circuit's rounds by.

    StimDevice builds the same table from the same circuit
    (qpu/stim_device.py _bind_source and _sample_shot), so the values
    a seat forms sit in this table's detector order.
    """
    detector_rounds = detector_chronology.resolve_detector_rounds(
        circuit, None, round_count
    )
    return detector_formation.build_formation_table(
        circuit, round_count, detector_rounds=detector_rounds
    )


def _bulk_slots(table: detector_formation.FormationTable) -> tuple:
    """The positions, and each bulk detector's (round, position index)."""
    bulk = detector_formation.LayerKind.BULK
    planar_by_detector = {}
    for recipe in table.detectors:
        if recipe.kind is bulk:
            planar_by_detector[recipe.detector_index] = _planar(recipe)
    if not planar_by_detector:
        raise ValueError(
            f"a circuit of {table.round_count} rounds has no bulk detector, "
            "a check compared with its round before, so the burst detector "
            "has no round to read; give the operation at least two rounds"
        )
    planar_positions = planar_by_detector.values()
    distinct = set(planar_positions)
    positions = tuple(sorted(distinct))
    index_by_position = {}
    for index, position in enumerate(positions):
        index_by_position[position] = index
    rounds = table.detector_rounds()
    slot_by_detector = {}
    for detector, planar in planar_by_detector.items():
        position_index = index_by_position[planar]
        slot_by_detector[detector] = (rounds[detector], position_index)
    return positions, slot_by_detector


def _planar(recipe: detector_formation.DetectorRecipe) -> tuple:
    """A detector's position: its first two Stim coordinates."""
    if len(recipe.coordinates) < 2:
        raise ValueError(
            f"detector {recipe.detector_index} has no position: the burst "
            "detector keys a counter by a detector's first two Stim "
            "coordinates, so the circuit must give every bulk detector them"
        )
    return (recipe.coordinates[0], recipe.coordinates[1])


def _layout(
    table, positions: tuple, slot_by_detector: dict, faults_by_position: list
) -> _Layout:
    """The positions, the value map, and one bulk round's usual rates.

    The rates are read at the operation's middle round, where a memory
    circuit's bulk rounds are alike.
    """
    position_by_value = _position_by_value(table, slot_by_detector)
    slots = slot_by_detector.values()
    first_slot = min(slots)
    first_bulk_round = first_slot[0]
    middle_round = (table.round_count + 1) // 2
    usual_round = max(middle_round, first_bulk_round)
    position_priors = _position_priors(faults_by_position, usual_round)
    usual_rates = _detection_probabilities(position_priors)
    return _Layout(
        positions=positions,
        position_by_value=position_by_value,
        first_bulk_round=first_bulk_round,
        usual_rates=usual_rates,
        position_priors=position_priors,
    )


def _position_by_value(table, slot_by_detector: dict) -> dict:
    """Each round's formation order mapped onto positions, -1 if not bulk."""
    position_by_value = {}
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        recipes = table.detectors_of_round(round_index)
        positions = []
        for recipe in recipes:
            slot = slot_by_detector.get(recipe.detector_index, (0, -1))
            positions.append(slot[1])
        position_by_value[round_index] = numpy.asarray(positions)
    return position_by_value


def _faults(circuit: stim.Circuit, slot_by_detector: dict) -> list:
    """Every error mechanism of the circuit's model, with its bulk slots."""
    model = circuit.detector_error_model(decompose_errors=False)
    flattened = model.flattened()
    faults = []
    for instruction in flattened:
        if instruction.type != "error":
            continue
        fault = _fault(instruction, slot_by_detector)
        faults.append(fault)
    return faults


def _fault(instruction, slot_by_detector: dict) -> _Fault:
    arguments = instruction.args_copy()
    slots = []
    for target in instruction.targets_copy():
        slot = None
        if target.is_relative_detector_id():
            slot = slot_by_detector.get(target.val)
        if slot is not None:
            slots.append(slot)
    return _Fault(arguments[0], tuple(slots))


def _faults_by_position(faults: list, position_count: int) -> list:
    """Each position's faults: those that flip one of its detectors."""
    faults_by_position = []
    for _ in range(position_count):
        faults_by_position.append([])
    for fault in faults:
        touched = set()
        for _round, position_index in fault.slots:
            touched.add(position_index)
        for position_index in touched:
            faults_by_position[position_index].append(fault)
    return faults_by_position


def _position_priors(faults_by_position: list, round_index: int) -> tuple:
    """The priors of the faults behind each position's detector that round."""
    arrays = []
    for position, faults in enumerate(faults_by_position):
        slot = (round_index, position)
        priors = _priors_flipping(faults, slot)
        arrays.append(priors)
    return tuple(arrays)


def _priors_flipping(faults: list, slot: tuple):
    priors = []
    for fault in faults:
        if slot in fault.slots:
            priors.append(fault.prior)
    return numpy.asarray(priors)


def _detection_probabilities(position_priors: tuple):
    """Each position's chance of an odd number of its faults firing."""
    probabilities = []
    for priors in position_priors:
        probability = _odd_probability(priors, 1.0)
        probabilities.append(probability)
    return numpy.asarray(probabilities)


def _odd_probability(priors, scale: float) -> float:
    """(1 - prod(1 - 2 s p)) / 2, s p capped at one half (Tan et al.)."""
    scaled_priors = priors * scale
    scaled = numpy.minimum(scaled_priors, MAXIMUM_PRIOR)
    survivals = 1.0 - 2.0 * scaled
    product = numpy.prod(survivals)
    return (1.0 - product) / 2.0


def _mean_detection_probability(region_priors: list, scale: float) -> float:
    probabilities = []
    for priors in region_priors:
        probability = _odd_probability(priors, scale)
        probabilities.append(probability)
    mean_probability = numpy.mean(probabilities)
    return float(mean_probability)


def _saturating_scale(region_priors: list) -> float:
    """The scale past which every region prior sits at one half.

    A noiseless position has no prior, so it bounds nothing; it still
    counts in the region's rate as a position that never fires.
    """
    every_prior = numpy.concatenate(region_priors)
    smallest_prior = numpy.min(every_prior)
    return MAXIMUM_PRIOR / float(smallest_prior)


def _bisected_scale(region_priors: list, measured_rate, largest_scale) -> float:
    """The scale whose mean detection probability is the measured rate."""
    low = 1.0
    high = largest_scale
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        probability = _mean_detection_probability(region_priors, middle)
        if probability < measured_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _mean_rate(flagged_rows: list, is_in_region) -> float:
    """Events per region position per round over the flagged rows.

    A count flag may leave the region empty, which measures nothing.
    """
    region_size = numpy.count_nonzero(is_in_region)
    if region_size == 0:
        return 0.0
    totals = numpy.sum(flagged_rows, axis=0)
    region_events = numpy.sum(totals[is_in_region])
    samples = region_size * len(flagged_rows)
    return float(region_events) / samples


def _positions_where(positions: tuple, is_in_region) -> frozenset:
    members = []
    for position, is_member in zip(positions, is_in_region, strict=True):
        if is_member:
            members.append(position)
    return frozenset(members)


def _episodes(flags: list, first_round: int) -> list:
    """The runs of firing rounds, each from its first round's onset.

    flags[i] is round first_round + i's flag, None when it did not fire.
    """
    runs = []
    for index, flag in enumerate(flags):
        if flag is None:
            continue
        round_index = first_round + index
        _extend_runs(runs, flag, round_index)
    newest_round = first_round + len(flags) - 1
    _open_the_newest_run(runs, newest_round)
    return runs


def _extend_runs(runs: list, flag: _Flag, round_index: int) -> None:
    """A firing round joins the run it follows, or starts a run."""
    if runs and runs[-1].last_firing_round == round_index - 1:
        runs[-1] = dataclasses.replace(
            runs[-1], last_firing_round=round_index, region=flag.region
        )
        return
    run = _Episode(flag.onset_round, round_index, False, flag.region)
    runs.append(run)


def _open_the_newest_run(runs: list, newest_round: int) -> None:
    """A run still firing on the newest round covers every later round."""
    if not runs:
        return
    if runs[-1].last_firing_round == newest_round:
        runs[-1] = dataclasses.replace(runs[-1], is_open=True)


def _method_keys(section: Mapping) -> dict:
    """The CUSUM's method numbers, each checked once at the yaml."""
    return {
        "mask_window_rounds": _whole_count(
            section, "mask_window_rounds", 64, "rounds"
        ),
        "mask_count": _mask_count(section),
        "mask_hold_rounds": _whole_count(
            section, "mask_hold_rounds", 100, "rounds"
        ),
        "region_radii": _radii(section),
        "fault_rate_multipliers": _multipliers(section),
        "rate_tracking_rounds": _whole_count(
            section, "rate_tracking_rounds", 5000, "rounds"
        ),
        "unmasked_share_floor": _share_floor(section),
        "false_alarms_per_second": _false_alarms_per_second(section),
        "calibration_shots": _whole_count(
            section, "calibration_shots", 20_000, "shots"
        ),
    }


def _whole_count(section: Mapping, key: str, default: int, unit: str) -> int:
    value = section.get(key, default)
    is_whole = isinstance(value, int) and not isinstance(value, bool)
    if is_whole and value >= 1:
        return value
    raise ValueError(
        f"burst_detector.{key} must be a whole number of {unit}, at least "
        f"one (got {value!r})"
    )


def _mask_count(section: Mapping) -> Optional[int]:
    """The mask's count; null never masks, the plain regional CUSUM."""
    if section.get("mask_count", 8) is None:
        return None
    return _whole_count(section, "mask_count", 8, "firings")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_list_at_least(values, bound: float) -> bool:
    """Whether values is a yaml list of numbers, each at least bound."""
    if not isinstance(values, list):
        return False
    for value in values:
        if not _is_number(value):
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
    if _is_number(value) and 0 <= value < 1:
        return float(value)
    raise ValueError(
        "burst_detector.unmasked_share_floor must be a share from 0 up to, "
        f"not including, 1 (got {value!r})"
    )


def _false_alarms_per_second(section: Mapping) -> float:
    value = section.get("false_alarms_per_second", 0.03)
    if _is_number(value) and value > 0:
        return float(value)
    raise ValueError(
        "burst_detector.false_alarms_per_second must be a rate above zero, "
        f"written as a number (got {value!r})"
    )


def _false_alarms_per_round(section: Mapping) -> float:
    value = section.get("false_alarms_per_round", 1e-6)
    if _is_number(value) and 0 < value < 1:
        return float(value)
    raise ValueError(
        "burst_detector.false_alarms_per_round must be a probability "
        f"between 0 and 1, written as a number (got {value!r})"
    )


def _count_clock(section: Mapping, clocks: config.ClockSettings, cycles: int):
    """event_count's clock; a priced count needs one."""
    name = section.get("clock")
    if name is not None:
        return clocks.clock(name)
    if cycles > 0:
        raise ValueError(
            "burst_detector.cycles_per_round needs a clock: name the "
            "clocks domain its cycles are counted in"
        )
    return None


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


def _boolean(section: Mapping, key: str) -> bool:
    value = section.get(key, False)
    if isinstance(value, bool):
        return value
    raise ValueError(f"burst_detector.{key} must be true or false")
