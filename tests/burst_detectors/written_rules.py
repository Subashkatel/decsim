"""The masked regional CUSUM's rules, written out one function per rule.

A reference the chart bank and the row are held against, with none of
the row's code:

  flag: a check whose repeats (fired this round and the one before),
    or whose joint firings with its same-basis pair, reach mask_count
    in the last 64 rounds is flagged that round;
  mask: a check flagged in the last 100 rounds is left out;
  regions: a disc of each radius around each check, then the patch;
  expected count: the region's usual count mu times f, the share of
    its usual rate still unmasked;
  step: for each design, c log rho - (rho - 1) mu f, the score kept
    at zero or above (Page 1954; Lucas 1985 for counts);
  tracking: mu moves toward c / f over 5000 rounds while f > 0.2;
  groups: each design's largest score at each radius;
  alarm: any group at or over its threshold (multichart CUSUM).
"""

import numpy

REFERENCE_RADII = (0.0, 1.5, 2.3, 3.2)
REFERENCE_DESIGNS = (2.0, 4.0, 20.0)
# the rules' windows: flag, mask hold, tracking, and the share floor
REFERENCE_WINDOW = 64
REFERENCE_HOLD = 100
REFERENCE_TRACKING = 5000.0
REFERENCE_SHARE_FLOOR = 0.2
# grid units: neighbouring checks sqrt(2) apart
GRID_UNIT = 0.7071067811865476


def _reference_regions(positions, radii):
    """Regions: discs of each radius around each check, then the patch."""
    coordinates = numpy.asarray(positions)
    grid = coordinates * GRID_UNIT
    members = []
    scales = []
    for scale, radius in enumerate(radii):
        for center in grid:
            offsets = grid - center
            distances = numpy.linalg.norm(offsets, axis=1)
            reach = radius + 1e-9
            is_inside = distances <= reach
            disc = numpy.flatnonzero(is_inside)
            members.append(disc)
            scales.append(scale)
    whole = numpy.arange(len(positions))
    members.append(whole)
    scales.append(len(radii))
    return members, scales


def _reference_flags(rows, pairs, mask_count):
    """Flag: each check's flag per round, over the last 64 rounds.

    A repeat at round t is a firing at t and t - 1; a pair's joint
    firing is both its checks at t. mask_count None flags nothing.
    """
    round_count, check_count = rows.shape
    flags = numpy.zeros(rows.shape, dtype=bool)
    if mask_count is None:
        return flags
    silent_row = numpy.zeros((1, check_count), dtype=int)
    previous = numpy.vstack([silent_row, rows[:-1]])
    repeats = rows & previous
    for round_index in range(round_count):
        first = _window_start(round_index)
        window_repeats = repeats[first : round_index + 1].sum(axis=0)
        flags[round_index] = window_repeats >= mask_count
        _flag_joint_pairs(flags, rows, pairs, round_index, mask_count)
    return flags


def _window_start(round_index):
    """The first of the 64 rounds that end at round_index, from zero."""
    start = round_index - REFERENCE_WINDOW + 1
    return max(0, start)


def _flag_joint_pairs(flags, rows, pairs, round_index, mask_count):
    first = _window_start(round_index)
    window = rows[first : round_index + 1]
    for first_check, second_check in pairs:
        joint = window[:, first_check] & window[:, second_check]
        if joint.sum() >= mask_count:
            flags[round_index, first_check] = True
            flags[round_index, second_check] = True


def _reference_kept(flags, round_index):
    """Mask: a check counts unless it was flagged in [t - 100, t]."""
    hold_start = round_index - REFERENCE_HOLD
    first = max(0, hold_start)
    recent = flags[first : round_index + 1]
    return ~recent.any(axis=0)


def reference_scores(rows, positions, pairs, usual_rates, mask_count=8):
    """Regions to groups, region by region, round by round.

    Returns (rounds, groups) design-major, and the masked check-rounds.
    """
    members, scales = _reference_regions(positions, REFERENCE_RADII)
    flags = _reference_flags(rows, pairs, mask_count)
    usual_counts = []
    for member in members:
        region_rate = usual_rates[member].sum()
        usual_counts.append(region_rate)
    scores = numpy.zeros((len(REFERENCE_DESIGNS), len(members)))
    group_scores = []
    masked = 0
    for round_index, row in enumerate(rows):
        kept = _reference_kept(flags, round_index)
        is_masked = ~kept
        masked += is_masked.sum()
        for region, member in enumerate(members):
            _reference_region_step(
                scores, usual_counts, region, member, row, kept, usual_rates
            )
        groups = _reference_groups(scores, scales)
        group_scores.append(groups)
    return numpy.asarray(group_scores), masked


def _reference_region_step(
    scores, usual_counts, region, member, row, kept, usual_rates
):
    """Expected count, step and tracking for one region, in that order."""
    kept_member = member[kept[member]]
    count = row[kept_member].sum()
    kept_rate = usual_rates[kept_member].sum()
    region_rate = usual_rates[member].sum()
    share = kept_rate / region_rate
    expected = usual_counts[region] * share
    for design, multiplier in enumerate(REFERENCE_DESIGNS):
        fired = (1 - (1 - 2 * usual_rates[member]) ** multiplier) / 2
        ratio = fired.sum() / region_rate
        step = count * numpy.log(ratio) - (ratio - 1) * expected
        score = scores[design, region] + step
        scores[design, region] = max(0.0, score)
    if share > REFERENCE_SHARE_FLOOR:
        error = count / share - usual_counts[region]
        usual_counts[region] += error / REFERENCE_TRACKING


def _reference_groups(scores, scales):
    """Groups: each design's largest region score at each radius."""
    scale_count = max(scales) + 1
    groups = []
    for design_scores in scores:
        for scale in range(scale_count):
            is_at_scale = numpy.asarray(scales) == scale
            at_scale = design_scores[is_at_scale]
            groups.append(max(at_scale))
    return groups


def reference_pairs(circuit):
    """For each data qubit, its two same-basis checks, the flag's pairs.

    On Stim's rotated surface code data qubits sit at odd coordinates
    and X checks carry an H, and a check beside a data qubit sits one
    unit off in both coordinates.
    """
    coordinates = circuit.get_final_qubit_coordinates()
    x_checks = _hadamard_targets(circuit)
    pairs = set()
    for data_x, data_y in coordinates.values():
        if data_x % 2 == 1 and data_y % 2 == 1:
            beside = _pairs_beside(coordinates, x_checks, data_x, data_y)
            pairs |= beside
    return pairs


def _hadamard_targets(circuit):
    targets = set()
    for instruction in circuit.flattened():
        if instruction.name != "H":
            continue
        for target in instruction.targets_copy():
            targets.add(target.value)
    return targets


def _pairs_beside(coordinates, x_checks, data_x, data_y):
    by_basis = {True: [], False: []}
    for qubit, (check_x, check_y) in coordinates.items():
        x_offset = check_x - data_x
        y_offset = check_y - data_y
        is_beside = abs(x_offset) == 1 and abs(y_offset) == 1
        if is_beside:
            by_basis[qubit in x_checks].append((check_x, check_y))
    pairs = set()
    for checks in by_basis.values():
        if len(checks) == 2:
            pair = frozenset(checks)
            pairs.add(pair)
    return pairs
