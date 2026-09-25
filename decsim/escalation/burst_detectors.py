"""The burst detector: detection events counted against their usual rates.

A burst of errors raises the detection rate of the stabilisers it covers
for hundreds of rounds (Google 2408.13687 lines 386-391 and 2101-2119).
The event_count row counts each round's bulk detection events as they
are formed, one counter per stabiliser position, and fires on (a) the
patch's count over the last patch_window_rounds, the Hamming weight of
one round's syndrome of Tan et al. (2406.18897 lines 956-966) widened to
W rounds, or (b) one position's count over the last
detector_window_rounds, Q3DE's active node counter (Suzuki et al.
2501.00331 lines 713-727). A threshold is the smallest count whose tail
under the usual rates, as TailLaw estimates it, is at most the row's
false-alarm rate per round.

A flag runs from the estimated onset, the firing round less the window
of the statistic that fired (Q3DE lines 727-728), to the last round the
detector still fires on. The switching policy escalates every window a
published flag meets, and the strong window model may raise the priors
of the flagged region's faults with its graph unchanged (IonQ 2608.25027
lines 334-340).
"""

import bisect
import dataclasses
import functools
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

# Q3DE's counter confidence, 1 - alpha = 0.99 (2501.00331 lines
# 1076-1078): a position is in the flagged region when its count reaches
# the count its usual rate reaches this rarely.
REGION_FALSE_ALARMS = 0.01
# The draws per fault count that integrate a count's tail given that
# many faults; tests/escalation/test_burst_detectors.py holds the tail
# against Stim's sampled one and against exact tails.
DRAWS_PER_FAULT_COUNT = 4000
# The draws are a numerical integral over the circuit's own model, so
# they take one fixed seed and every shot of a run gets the same law.
CALIBRATION_SEED = 0
# The fault counts the law integrates cover a rate twice the calibrated
# one; past that the truncated remainder is counted as a firing, so the
# truncation can only raise the tail.
RATE_HEADROOM = 2.0
# The truncated remainder is kept this far below the smallest
# false-alarm rate the law answers for.
TRUNCATION_SHARE = 0.01
# A prior is a flip probability and never passes one half.
MAXIMUM_PRIOR = 0.5
BISECTION_STEPS = 60


class EventCountBurstDetector:
    """Fires when a patch's detection events outrun its usual rates.

    One counter set per operation, calibrated at build from that
    operation's own circuit, the calibration Q3DE assumes is known in
    advance (2501.00331 lines 1078-1080). Each position's usual rate is
    then tracked as an exponential average of its events and frozen
    while the detector fires, as Q3DE removes the flagged positions from
    its count for the burst's lifetime (lines 730-733).
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
            patch_window = _whole_count(section, "patch_window_rounds", 4)
            detector_window = _whole_count(
                section, "detector_window_rounds", 20
            )
            false_alarms = _false_alarms(section)
            tracking = _whole_count(section, "rate_tracking_rounds", 10_000)
            cycles = section.get("cycles_per_round", 0)
            config.check_cycles("burst_detector.cycles_per_round", cycles)
            clock = _clock(section, clocks, cycles)
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
    ) -> None:
        """Counters for each operation: circuits maps an id to its circuit.

        Each entry is the operation's circuit and its round count.
        """
        self.settings = settings
        self.engine = engine
        self.counts_by_operation = _counts_by_operation(circuits, settings)

    def observe_round(
        self, operation_id: Any, round_index: int, events: Sequence[int]
    ) -> None:
        """Count one round; its verdict is published after the row's cost."""
        counts = self.counts_by_operation[operation_id]
        published_tick = self._published_tick(counts)
        counts.count_round(round_index, events, published_tick)

    def is_burst_window(self, window: window_records.Window) -> bool:
        """Whether a flag published by now meets the window's rounds."""
        counts = self.counts_by_operation[window.operation_id]
        episode = counts.episode_meeting(window, self.engine.now)
        return episode is not None

    def with_burst_priors(self, window: window_records.Window, model):
        """The model with the flagged region's priors raised, graph kept."""
        if not self.settings.raise_strong_priors:
            return model
        counts = self.counts_by_operation[window.operation_id]
        episode = counts.episode_meeting(window, self.engine.now)
        if episode is None:
            return model
        region = counts.region_of(episode)
        return region.raised(model)

    def _published_tick(self, counts: "_OperationCounts") -> int:
        """One counter bank serves the rounds in order, each on its clock.

        A round's count starts when it is formed or when the round
        before it is published, whichever is later, and takes
        cycles_per_round of the row's clock, charged from the edge at
        or after that tick (gem5's Clocked, decsim/config.py Clock).
        """
        now = self.engine.now
        newest = counts.newest_published_tick()
        start = max(now, newest)
        cycles = self.settings.cycles_per_round
        if cycles == 0:
            return start
        return self.settings.clock.edge(cycles, start)


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
class _Fault:
    """One error mechanism, as the counters see it."""

    prior: float
    # (round, position index) of every bulk detector it flips
    slots: tuple


@dataclasses.dataclass(frozen=True)
class _Calibration:
    """What the counters know of one operation before its first round.

    positions are the stabiliser positions, a bulk detector's first two
    Stim coordinates; position_by_value maps each round's events, in
    formation order, onto them, -1 for a detector that is not bulk.
    calibrated_rates and position_priors are one round's detection
    probability at each position and the priors of the faults behind it.
    """

    positions: tuple
    position_by_value: dict
    patch_law: TailLaw
    position_laws: tuple
    calibrated_rates: Any
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
        # a flag the patch count raised may leave no position anomalous
        # on its own: there is then no region whose priors to raise
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
class _Episode:
    """One run of firing rounds, from the estimated onset.

    is_open says the newest published round still fires, so the flag
    covers every later round too.
    """

    first_round: int
    last_firing_round: int
    is_open: bool

    def meets(self, first_round: int, last_round: int) -> bool:
        """Whether the flag covers any round of [first_round, last_round]."""
        if self.first_round > last_round:
            return False
        return self.is_open or self.last_firing_round >= first_round


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
        if not self.positions or self.prior_scale == 1.0:
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


class _OperationCounts:
    """The counters of one operation, and what they published when."""

    def __init__(
        self,
        calibration: _Calibration,
        settings: EventCountBurstDetector.Settings,
    ) -> None:
        self.calibration = calibration
        self.settings = settings
        # one array of per-position event counts per counted round
        self.rows: list = []
        self.tracked_rates = numpy.array(calibration.calibrated_rates)
        self.published_ticks: list = []
        # per round, the longest window of a statistic firing on it; 0
        # when none fires
        self.firing_windows: list = []

    def newest_published_tick(self) -> int:
        """The tick the previous round was published at; 0 before any."""
        if not self.published_ticks:
            return 0
        return self.published_ticks[-1]

    def count_round(
        self, round_index: int, events: Sequence[int], published_tick: int
    ) -> None:
        """Add one round to the counters and decide whether it fires."""
        assert round_index == len(self.rows) + 1, (
            "an operation's rounds are counted in order"
        )
        row = self.calibration.position_counts(round_index, events)
        self.rows.append(row)
        firing_window = self._firing_window()
        self.firing_windows.append(firing_window)
        self.published_ticks.append(published_tick)
        if firing_window == 0:
            self._track(row)

    def episode_meeting(
        self, window: window_records.Window, now: int
    ) -> Optional[_Episode]:
        """The newest published flag that meets the window, or None."""
        published_count = bisect.bisect_right(self.published_ticks, now)
        published = self.firing_windows[:published_count]
        episodes = _episodes(published)
        for episode in reversed(episodes):
            if episode.meets(window.start_round, window.buffer_hi):
                return episode
        return None

    def region_of(self, episode: _Episode) -> _BurstRegion:
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
        window_start = newest_round - detector_window + 1
        first_counted = max(episode.first_round, window_start)
        flagged_rows = self.rows[first_counted - 1 : newest_round]
        measured_rate = _mean_rate(flagged_rows, is_in_region)
        prior_scale = self.calibration.prior_scale(is_in_region, measured_rate)
        positions = _positions_where(self.calibration.positions, is_in_region)
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
        calibrated_total = numpy.sum(self.calibration.calibrated_rates)
        rate_scale = _rate_scales(tracked_total, calibrated_total)
        false_alarms = self.settings.false_alarms_per_round / 2
        law = self.calibration.patch_law
        threshold = law.threshold(false_alarms, rate_scale)
        return patch_count >= threshold

    def _position_fires(self) -> bool:
        """Statistic (b): any position's events over the last c_win rounds."""
        newest_round = len(self.rows)
        position_counts = self._position_counts(newest_round)
        position_count = len(self.calibration.positions)
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
        calibrated_rates = self.calibration.calibrated_rates
        rate_scales = _rate_scales(self.tracked_rates, calibrated_rates)
        thresholds = []
        for law, rate_scale in zip(self.calibration.position_laws, rate_scales):
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
        circuit_text = str(circuit)
        calibration = _calibrate(circuit_text, round_count, settings)
        counts_by_operation[operation_id] = _OperationCounts(
            calibration, settings
        )
    return counts_by_operation


@functools.lru_cache(maxsize=64)
def _calibrate(circuit_text: str, round_count: int, settings):
    """The positions, the laws and the usual rates of one circuit.

    Each law is read off one slab of bulk rounds centred in the
    operation, where a memory circuit's rounds are alike. The result
    depends on its arguments alone (the draws take a fixed seed), so a
    run's shots share it rather than draw it again.
    """
    circuit = stim.Circuit(circuit_text)
    table = _formation_table(circuit, round_count)
    positions, slot_by_detector = _bulk_slots(table)
    position_by_value = _position_by_value(table, slot_by_detector)
    faults = _faults(circuit, slot_by_detector)
    patch_law = _patch_law(faults, round_count, settings)
    faults_by_position = _faults_by_position(faults, len(positions))
    position_laws = _position_laws(faults_by_position, round_count, settings)
    middle_round = (round_count + 1) // 2
    position_priors = _position_priors(faults_by_position, middle_round)
    calibrated_rates = _detection_probabilities(position_priors)
    return _Calibration(
        positions=positions,
        position_by_value=position_by_value,
        patch_law=patch_law,
        position_laws=position_laws,
        calibrated_rates=calibrated_rates,
        position_priors=position_priors,
    )


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


def _patch_law(faults: list, round_count: int, settings) -> TailLaw:
    """Statistic (a)'s law: every position over W centred rounds."""
    patch_window = settings.patch_window_rounds
    first_round, last_round = _centred_slab(round_count, patch_window)
    priors, incidence = _slab(faults, first_round, last_round, None)
    false_alarms = settings.false_alarms_per_round / 2
    return tail_law(priors, incidence, false_alarms)


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


def _mean_detection_probability(region_priors: list, scale: float) -> float:
    probabilities = []
    for priors in region_priors:
        probability = _odd_probability(priors, scale)
        probabilities.append(probability)
    mean_probability = numpy.mean(probabilities)
    return float(mean_probability)


def _saturating_scale(region_priors: list) -> float:
    """The scale past which every region prior sits at one half."""
    smallest_priors = []
    for priors in region_priors:
        smallest = numpy.min(priors)
        smallest_priors.append(smallest)
    smallest_prior = min(smallest_priors)
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
    """Events per region position per round over the flagged rows."""
    region_size = numpy.count_nonzero(is_in_region)
    if region_size == 0:
        return 0.0
    totals = numpy.sum(flagged_rows, axis=0)
    region_events = numpy.sum(totals[is_in_region])
    samples = region_size * len(flagged_rows)
    return float(region_events) / samples


def _positions_where(positions: tuple, is_in_region) -> frozenset:
    members = []
    for position, is_member in zip(positions, is_in_region):
        if is_member:
            members.append(position)
    return frozenset(members)


def _episodes(firing_windows: list) -> list:
    """The runs of firing rounds, each from its estimated onset."""
    first_rounds = []
    last_rounds = []
    for index, firing_window in enumerate(firing_windows):
        if firing_window == 0:
            continue
        round_index = index + 1
        if last_rounds and last_rounds[-1] == index:
            last_rounds[-1] = round_index
            continue
        onset = round_index - firing_window + 1
        first_rounds.append(max(onset, 1))
        last_rounds.append(round_index)
    newest_round = len(firing_windows)
    episodes = []
    for first_round, last_round in zip(first_rounds, last_rounds):
        is_open = last_round == newest_round
        episode = _Episode(first_round, last_round, is_open)
        episodes.append(episode)
    return episodes


def _whole_count(section: Mapping, key: str, default: int) -> int:
    value = section.get(key, default)
    is_whole = isinstance(value, int) and not isinstance(value, bool)
    if is_whole and value >= 1:
        return value
    raise ValueError(
        f"burst_detector.{key} must be a whole number of rounds, at least "
        f"one (got {value!r})"
    )


def _false_alarms(section: Mapping) -> float:
    value = section.get("false_alarms_per_round", 1e-6)
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_number and 0 < value < 1:
        return float(value)
    raise ValueError(
        "burst_detector.false_alarms_per_round must be a probability "
        f"between 0 and 1, written as a number (got {value!r})"
    )


def _clock(section: Mapping, clocks: config.ClockSettings, cycles: int):
    name = section.get("clock")
    if name is not None:
        return clocks.clock(name)
    if cycles > 0:
        raise ValueError(
            "burst_detector.cycles_per_round needs a clock: name the "
            "clocks domain its cycles are counted in"
        )
    return None


def _boolean(section: Mapping, key: str) -> bool:
    value = section.get(key, False)
    if isinstance(value, bool):
        return value
    raise ValueError(f"burst_detector.{key} must be true or false")
