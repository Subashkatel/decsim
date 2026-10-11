"""A plan entry, a window's rows, and a window placed on arrays.

Sources: Skoric et al. 2209.08552, section I.B (a window is a commit
region inside a decoding region) and the distance-3 rotated surface code
of stim.Circuit.generated, whose four-round circuit holds 4, 8, 8 and 12
detectors in rounds 1 to 4 (test_detector_chronology).

The array placement is checked against tests/detector_error_model/
placement_oracle.py, the Python placement decsim ran before, unchanged
in logic: the same check and observable matrices entry for entry, the
same priors, owned mask, column ids and handoffs, and the same faults
added to what earlier windows committed.
"""

import random

import numpy
import stim

import tests.detector_error_model.placement_oracle as placement_oracle
from decsim.detector_error_model import (
    detector_chronology,
    window_placement,
    window_slicer,
)
from decsim.records import fault_model_contracts

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE
# a fault of probability one half and one past it, as Stim's merged
# mechanisms can be, beside ordinary ones
PRIOR_CHOICES = (0.001, 0.01, 0.2, 0.5, 0.7)
RANDOM_SEED = 2026
RANDOM_WINDOW_COUNT = 2000
# a window is terminal, or names its owners, this often in the corpus
LAST_WINDOW_SHARE = 0.2
EXPLICIT_OWNER_SHARE = 0.3
SURFACE_ROUNDS = 12
SURFACE_STEP = 3


def surface_code_detectors_by_round(rounds):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, rounds
    )
    return detector_chronology.detectors_by_round(round_by_detector)


def test_a_three_bound_entry_starts_its_buffer_at_its_first_commit_round():
    bounds = window_placement.parse_window_entry((2, 3, 4))
    assert bounds == (2, 2, 3, 4)


def test_a_windows_rows_are_the_detectors_of_rounds_two_and_three():
    detectors_by_round = surface_code_detectors_by_round(4)
    rows = window_placement.detectors_in_window(detectors_by_round, 2, 3)
    assert rows == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def random_subset(rng, count: int, size: int) -> list:
    """Size of range(count) in random order, or all when size is larger."""
    population = range(count)
    taken = min(size, count)
    return rng.sample(population, taken)


def random_fault(rng, detector_count: int, observable_count: int) -> tuple:
    """Zero to four detectors in any order, any observables, a prior."""
    detector_total = rng.randint(0, 4)
    detectors = random_subset(rng, detector_count, detector_total)
    observable_total = rng.randint(0, observable_count)
    observables = random_subset(rng, observable_count, observable_total)
    observables.sort()
    prior = rng.choice(PRIOR_CHOICES)
    return tuple(detectors), tuple(observables), prior


def random_catalog(rng, detector_count: int, observable_count: int):
    """Up to forty random faults in one graphlike catalog."""
    fault_count = rng.randint(0, 40)
    detector_sets = []
    observable_sets = []
    priors = []
    for _ in range(fault_count):
        fault = random_fault(rng, detector_count, observable_count)
        detectors, observables, prior = fault
        detector_sets.append(detectors)
        observable_sets.append(observables)
        priors.append(prior)
    return fault_model_contracts.FaultCatalog(
        representation=GRAPHLIKE,
        detector_sets=tuple(detector_sets),
        observable_sets=tuple(observable_sets),
        priors=tuple(priors),
    )


def rounds_of(catalog, detector_rounds: list) -> tuple:
    """Each fault's rounds, one per detector, as the slicer indexes them."""
    fault_rounds = []
    for detectors in catalog.detector_sets:
        rounds = [detector_rounds[detector] for detector in detectors]
        fault_rounds.append(tuple(rounds))
    return tuple(fault_rounds)


def random_context(rng, detector_rounds: list, observable_count: int):
    """A window of ordered random bounds over the detectors' rounds."""
    round_count = detector_rounds[-1]
    bounds = []
    for _ in range(4):
        bound = rng.randint(1, round_count)
        bounds.append(bound)
    bounds.sort()
    first_buffer, first_commit, last_commit, last_buffer = bounds
    rows = []
    for detector, round_index in enumerate(detector_rounds):
        if first_buffer <= round_index <= last_buffer:
            rows.append(detector)
    row_by_detector = {detector: row for row, detector in enumerate(rows)}
    is_last = rng.random() < LAST_WINDOW_SHARE
    return window_placement.WindowPlacementContext(
        rows=rows,
        row_by_detector=row_by_detector,
        observable_count=observable_count,
        first_commit_round=first_commit,
        last_commit_round=last_commit,
        is_last=is_last,
    )


def random_owners(rng, fault_count: int) -> tuple:
    """An owner map and the prior map it comes with, or neither."""
    if rng.random() < EXPLICIT_OWNER_SHARE:
        owned_count = fault_count // 3
        prior_count = fault_count // 5
        owned = random_subset(rng, fault_count, owned_count)
        prior = random_subset(rng, fault_count, prior_count)
        return set(owned), set(prior)
    return None, None


def random_placement(rng) -> dict:
    """A catalog, a window over it, and what earlier windows committed."""
    round_count = rng.randint(1, 6)
    detector_count = rng.randint(1, 30)
    detector_rounds = []
    for _ in range(detector_count):
        round_index = rng.randint(1, round_count)
        detector_rounds.append(round_index)
    detector_rounds.sort()
    observable_count = rng.randint(1, 2)
    catalog = random_catalog(rng, detector_count, observable_count)
    fault_count = len(catalog.detector_sets)
    candidate_count = fault_count // 2
    committed_count = fault_count // 4
    candidates = random_subset(rng, fault_count, candidate_count)
    candidates.sort()
    committed = random_subset(rng, fault_count, committed_count)
    excluded_round = rng.randint(1, round_count)
    exclusion_count = rng.randint(0, 1)
    exclusions = ((excluded_round, excluded_round),) * exclusion_count
    owned, prior = random_owners(rng, fault_count)
    return {
        "catalog": catalog,
        "fault_rounds": rounds_of(catalog, detector_rounds),
        "context": random_context(rng, detector_rounds, observable_count),
        "candidate_faults": candidates,
        "committed_elsewhere": set(committed),
        "explicitly_owned_faults": owned,
        "explicitly_prior_faults": prior,
        "fault_exclusion_ranges": exclusions,
    }


def placed_both_ways(placement: dict) -> tuple:
    """The window placed on arrays and by the Python oracle, and each set."""
    arrays_committed = set(placement["committed_elsewhere"])
    oracle_committed = set(placement["committed_elsewhere"])
    arrays = window_placement.catalog_arrays(
        placement["catalog"], placement["fault_rounds"]
    )
    shared = {
        "catalog": placement["catalog"],
        "context": placement["context"],
        "candidate_faults": placement["candidate_faults"],
        "explicitly_owned_faults": placement["explicitly_owned_faults"],
        "explicitly_prior_faults": placement["explicitly_prior_faults"],
        "fault_exclusion_ranges": placement["fault_exclusion_ranges"],
    }
    placed = window_placement.placed_faults_for_window(
        arrays=arrays, committed_elsewhere=arrays_committed, **shared
    )
    expected = placement_oracle.placed_faults_for_window(
        fault_rounds=placement["fault_rounds"],
        committed_elsewhere=oracle_committed,
        **shared,
    )
    return placed, expected, arrays_committed, oracle_committed


def assert_same_matrix(matrix, expected) -> None:
    """Equal shapes, element types and sparse structure, entry for entry."""
    assert matrix.shape == expected.shape
    assert matrix.dtype == expected.dtype
    assert matrix.indices.dtype == expected.indices.dtype
    assert numpy.array_equal(matrix.indptr, expected.indptr)
    assert numpy.array_equal(matrix.indices, expected.indices)
    assert numpy.array_equal(matrix.data, expected.data)


def compare_one_placement(placement: dict) -> None:
    placed, expected, committed, oracle_committed = placed_both_ways(placement)
    assert_same_matrix(placed.check, expected.check)
    assert_same_matrix(placed.observables, expected.observables)
    assert numpy.array_equal(placed.priors, expected.priors)
    assert placed.priors.dtype == expected.priors.dtype
    assert numpy.array_equal(placed.owned, expected.owned)
    assert placed.owned.dtype == expected.owned.dtype
    assert placed.source_fault_ids == expected.source_fault_ids
    assert dict(placed.boundary_flips) == dict(expected.boundary_flips)
    assert committed == oracle_committed


def compare_excluded_faults(placement: dict) -> None:
    arrays = window_placement.catalog_arrays(
        placement["catalog"], placement["fault_rounds"]
    )
    every_fault = list(range(len(placement["fault_rounds"])))
    ranges = placement["fault_exclusion_ranges"]
    excluded = window_placement.faults_touching_excluded_rounds(
        arrays, ranges, every_fault
    )
    expected = placement_oracle.faults_touching_excluded_rounds(
        placement["fault_rounds"], ranges, every_fault
    )
    assert excluded == expected


def test_the_property_corpus_of_random_windows_matches_the_python_placement():
    """Two thousand random catalogs and windows, placed both ways."""
    rng = random.Random(RANDOM_SEED)
    for _ in range(RANDOM_WINDOW_COUNT):
        placement = random_placement(rng)
        compare_one_placement(placement)
        compare_excluded_faults(placement)


def surface_slicer() -> window_slicer.WindowSlicer:
    """The slicer of a d5 rotated memory of twelve rounds."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=SURFACE_ROUNDS,
        after_clifford_depolarization=0.003,
        before_measure_flip_probability=0.003,
    )
    representations = frozenset({GRAPHLIKE})
    requirement = fault_model_contracts.DecoderFaultModelRequirement(
        representations
    )
    return window_slicer.WindowSlicer(
        circuit, round_count=SURFACE_ROUNDS, fault_model_requirement=requirement
    )


def surface_window(slicer, first_commit: int, committed: set) -> dict:
    """The window committing three rounds from first_commit, buffered three."""
    last_commit = first_commit + SURFACE_STEP - 1
    buffer_end = last_commit + SURFACE_STEP
    last_buffer = min(buffer_end, SURFACE_ROUNDS)
    rows = window_placement.detectors_in_window(
        slicer.chronology.detectors_by_round, first_commit, last_buffer
    )
    row_by_detector = {detector: row for row, detector in enumerate(rows)}
    is_last = last_commit >= SURFACE_ROUNDS
    context = window_placement.WindowPlacementContext(
        rows=rows,
        row_by_detector=row_by_detector,
        observable_count=slicer.observable_count,
        first_commit_round=first_commit,
        last_commit_round=last_commit,
        is_last=is_last,
    )
    fault_rounds = slicer.fault_index.fault_rounds[GRAPHLIKE]
    every_fault = list(range(len(fault_rounds)))
    return {
        "catalog": slicer.catalogs.by_representation[GRAPHLIKE],
        "fault_rounds": fault_rounds,
        "context": context,
        "candidate_faults": every_fault,
        "committed_elsewhere": committed,
        "explicitly_owned_faults": None,
        "explicitly_prior_faults": None,
        "fault_exclusion_ranges": (),
    }


def test_the_property_corpus_of_d5_windows_matches_the_python_placement():
    """Every window of a d5 memory in time order, ownership carried on."""
    slicer = surface_slicer()
    committed = set()
    for first_commit in range(1, SURFACE_ROUNDS, SURFACE_STEP):
        placement = surface_window(slicer, first_commit, committed)
        compare_one_placement(placement)
        _placed, _expected, committed, _oracle = placed_both_ways(placement)
