"""Where one operation's checks sit, their usual rates, and their faults.

Both rows key a round's detection events by stabiliser position, a bulk
detector's first two Stim coordinates, and read the usual rate of each
position off the circuit's own detector error model.
"""

import dataclasses
from collections.abc import Sequence

import numpy
import stim

import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.detector_error_model.detector_formation as detector_formation

# A prior is a flip probability and never passes one half.
MAXIMUM_PRIOR = 0.5
BISECTION_STEPS = 60


@dataclasses.dataclass(frozen=True)
class Fault:
    """One error mechanism, as the detector sees it."""

    prior: float
    # (round, position index) of every bulk detector it flips
    slots: tuple


@dataclasses.dataclass(frozen=True)
class Layout:
    """Where one operation's checks sit, with their usual rates.

    positions are the stabiliser positions, a bulk detector's first two
    Stim coordinates; position_by_value maps each round's events, in
    formation order, onto them, -1 for a detector that is not bulk.
    usual_rates and position_priors are one round's detection
    probability at each position and the priors of the faults behind it.
    slot_by_detector is each bulk detector's (round, position index),
    faults the model's error mechanisms and faults_by_position those
    that flip one of each position's detectors.
    """

    positions: tuple
    position_by_value: dict
    first_bulk_round: int
    usual_rates: numpy.ndarray
    position_priors: tuple
    slot_by_detector: dict
    faults: list
    faults_by_position: list

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, round_count: int) -> "Layout":
        """The positions, the value map, and one bulk round's usual rates.

        The rates are read at the operation's middle round, where a
        memory circuit's bulk rounds are alike.
        """
        table = _formation_table(circuit, round_count)
        positions, slot_by_detector = _bulk_slots(table)
        faults = _faults(circuit, slot_by_detector)
        faults_by_position = _faults_by_position(faults, len(positions))
        position_by_value = _position_by_value(table, slot_by_detector)
        slots = slot_by_detector.values()
        first_slot = min(slots)
        first_bulk_round = first_slot[0]
        after_last_round = table.round_count + 1
        middle_round = after_last_round // 2
        usual_round = max(middle_round, first_bulk_round)
        position_priors = _position_priors(faults_by_position, usual_round)
        usual_rates = _detection_probabilities(position_priors)
        return cls(
            positions=positions,
            position_by_value=position_by_value,
            first_bulk_round=first_bulk_round,
            usual_rates=usual_rates,
            position_priors=position_priors,
            slot_by_detector=slot_by_detector,
            faults=faults,
            faults_by_position=faults_by_position,
        )

    def position_counts(
        self, round_index: int, events: Sequence[int]
    ) -> numpy.ndarray:
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

    def prior_scale(
        self, is_in_region: numpy.ndarray, measured_rate: float
    ) -> float:
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

    def _region_priors(self, is_in_region: numpy.ndarray) -> list:
        region_priors = []
        for position, is_member in enumerate(is_in_region):
            if is_member:
                region_priors.append(self.position_priors[position])
        return region_priors


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


def _fault(instruction, slot_by_detector: dict) -> Fault:
    arguments = instruction.args_copy()
    slots = []
    for target in instruction.targets_copy():
        slot = None
        if target.is_relative_detector_id():
            slot = slot_by_detector.get(target.val)
        if slot is not None:
            slots.append(slot)
    return Fault(arguments[0], tuple(slots))


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
    doubled = 2.0 * scaled
    survivals = 1.0 - doubled
    product = numpy.prod(survivals)
    odd_share = 1.0 - product
    return odd_share / 2.0


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
        bracket_total = low + high
        middle = bracket_total / 2
        probability = _mean_detection_probability(region_priors, middle)
        if probability < measured_rate:
            low = middle
        else:
            high = middle
    bracket_total = low + high
    return bracket_total / 2
