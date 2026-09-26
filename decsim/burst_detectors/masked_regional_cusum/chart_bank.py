"""The chart bank: one Page CUSUM per region and design, and the mask.

Each chart scores a region's detection events, W_n = (W_{n-1} +
l(X_n))+ (Page 1954, as written in Xie et al. 2104.04186 lines
274-328). A region is a disc around a check at one of several radii,
or the whole patch, the circular zones of Kulldorff's spatial scan
statistic (Commun. Stat. 26(6) 1997, section 2, zone collection 2).
Each region is scored against designs that multiply the fault rate by
a known k, so a step is Lucas's count-data term c log rho - (rho - 1)
mu (as written in Ward et al. 2208.01494 lines 274-296). A check that
behaves like a defect, firing in consecutive rounds or with its pair
too often, leaves the counts for a hold time, as Q3DE removes detected
positions for a burst's lifetime (2501.00331 lines 729-733).
"""

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy

import decsim.burst_detectors.layout as layout_module

if TYPE_CHECKING:
    import decsim.burst_detectors.masked_regional_cusum.detector as detector

# Each usual rate is floored here, so a check no fault reaches still
# has a finite design ratio.
_SMALLEST_USUAL_RATE = 1e-6
# Region radii are in grid units, where neighbouring checks sit
# sqrt(2) apart: Stim's rotated layout puts them 2 apart.
_GRID_UNITS_PER_COORDINATE = math.sqrt(2) / 2
# A disc holds a check at its radius exactly, whatever the rounding.
_RADIUS_ROUNDING = 1e-9
# On Stim's rotated surface code the two checks diagonal to a data
# qubit measure its basis (gen_surface_code.cc lines 220-262), so the
# pair one data-qubit flip fires together sits this far apart in both
# coordinates.
_PAIR_OFFSET = 2.0
# A check never flagged is masked at no round.
_NEVER_FLAGGED = -(10**9)


@dataclasses.dataclass(frozen=True)
class ChartBank:
    """One operation's CUSUM charts, one per region of its checks per design.

    incidence is (checks, regions), region_scales names each region's
    radius (len(radii) for the whole patch), and pair_incidence is
    (pairs, checks). A group is one design and one radius: its score is
    the largest of its regions', design-major.
    """

    settings: "detector.MaskedRegionalCusumBurstDetector.Settings"
    usual_rates: numpy.ndarray
    incidence: numpy.ndarray
    region_rates: numpy.ndarray
    region_scales: numpy.ndarray
    pair_incidence: numpy.ndarray
    log_ratios: numpy.ndarray
    ratio_excesses: numpy.ndarray

    @classmethod
    def for_layout(
        cls,
        layout: layout_module.Layout,
        settings: "detector.MaskedRegionalCusumBurstDetector.Settings",
    ) -> "ChartBank":
        """The regions, the pairs and the designs' log ratios.

        A design multiplies each fault's rate by k, so a check of usual
        rate q0 fires with q1 = (1 - (1 - 2 q0)^k) / 2 (k chances
        combined by parity), and a region's ratio is rho = sum q1 / sum
        q0.
        """
        usual_rates = numpy.maximum(layout.usual_rates, _SMALLEST_USUAL_RATE)
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
        return cls(
            settings=settings,
            usual_rates=usual_rates,
            incidence=incidence,
            region_rates=region_rates,
            region_scales=region_scales,
            pair_incidence=pair_incidence,
            log_ratios=log_ratios,
            ratio_excesses=ratio_excesses,
        )

    def group_count(self) -> int:
        """Designs times radii, the whole patch counted as one radius."""
        design_count = len(self.settings.fault_rate_multipliers)
        scale_count = len(self.settings.region_radii) + 1
        return design_count * scale_count

    def cycles_per_round(
        self,
        settings: "detector.MaskedRegionalCusumBurstDetector.Settings",
    ) -> int:
        """ceil(regions x designs / datapaths) + pipeline_cycles.

        A bank of U chart updates spread over P datapaths, each a
        pipeline L cycles deep, finishes a round in U / P + L cycles.
        U is the algorithm's own count, one update per region and
        design; P and L are the hardware's, pencil figures until a
        design is built.
        """
        region_count = self.incidence.shape[1]
        design_count = len(settings.fault_rate_multipliers)
        updates = region_count * design_count
        updates_per_datapath = updates / settings.datapaths
        update_cycles = math.ceil(updates_per_datapath)
        return update_cycles + settings.pipeline_cycles

    def new_state(self, stream_count: int, first_round: int) -> "_BankState":
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
        last_flagged_round = numpy.full(check_shape, _NEVER_FLAGGED)
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

    def score_round(
        self, state: "_BankState", rows: numpy.ndarray, round_index: int
    ) -> numpy.ndarray:
        """One round of every stream: mask, then score; the group scores.

        rows is (streams, checks) of 0/1. The score is read after the
        update, and the usual count moves after the score, so a round's
        own events never lower the bar they are scored against.
        """
        is_unmasked = self._unmasked(state, rows, round_index)
        self._score(state, rows, is_unmasked, round_index)
        return self._group_scores(state)

    def block_maxima(
        self, rows: numpy.ndarray, first_round: int
    ) -> numpy.ndarray:
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

    def leading_chart(
        self, state: "_BankState", thresholds: numpy.ndarray
    ) -> tuple:
        """The first stream's chart with the largest score over its level."""
        design_count = len(self.settings.fault_rate_multipliers)
        scale_count = len(self.settings.region_radii) + 1
        group_levels = numpy.reshape(thresholds, (design_count, scale_count))
        chart_levels = group_levels[:, self.region_scales]
        ratios = state.scores[0] / chart_levels
        flat_index = numpy.argmax(ratios)
        return numpy.unravel_index(flat_index, ratios.shape)

    def _unmasked(self, state: "_BankState", rows, round_index: int):
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

    def _score(self, state: "_BankState", rows, is_unmasked, round_index):
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

    def _track(self, state: "_BankState", counts, unmasked_shares) -> None:
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

    def _group_scores(self, state: "_BankState"):
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


@dataclasses.dataclass
class _BankState:
    """The chart bank's state, one row per stream it scores.

    The rings hold the last mask_window_rounds rounds of each check's
    repeats and each pair's joint firings; scored_rounds places the next
    round in them.
    """

    previous_row: numpy.ndarray
    repeat_ring: numpy.ndarray
    joint_ring: numpy.ndarray
    repeat_counts: numpy.ndarray
    joint_counts: numpy.ndarray
    last_flagged_round: numpy.ndarray
    usual_counts: numpy.ndarray
    scores: numpy.ndarray
    last_zero_round: numpy.ndarray
    scored_rounds: int = 0


def _regions(positions: tuple, settings) -> tuple:
    """Discs of every radius around every check, then the whole patch.

    Returns incidence (checks, regions) of 0/1 and each region's radius
    index: radius by radius, each radius's discs in check order, and the
    whole patch last.
    """
    coordinates = numpy.asarray(positions, dtype=float)
    grid = coordinates * _GRID_UNITS_PER_COORDINATE
    offsets = grid[:, None, :] - grid[None, :, :]
    squared_offsets = offsets**2
    squared_distances = numpy.sum(squared_offsets, axis=2)
    columns = []
    region_scales = []
    for scale, radius in enumerate(settings.region_radii):
        reach = radius * radius + _RADIUS_ROUNDING
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
    is_pair_offset = distances == _PAIR_OFFSET
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
