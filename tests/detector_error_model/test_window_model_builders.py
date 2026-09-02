"""A plan is sliced in order, and a plan that cannot be decoded is refused.

Sources: qLDPC's SlidingWindowDecoder, which is run here on the same
detector error model (its windows are cut in time order, each committing
the region the next one starts after, and the last window commits
everything it holds): qLDPC's detection regions are asserted against
decsim's detector_ids, its commit regions against literals, and decsim's
committed identities against qLDPC's, since a WindowErrorModel carries no
commit region of its own; Skoric et al. 2209.08552, section I.B (a
window's graph holds every edge touching a defect in its rounds, and
only the correction edges in the commit region are taken as final) and
the last paragraph of its section III, Methods (text lines 693-700: in
both branches the last window commits through the last round, the last
B window of reduced size, or the last A window whose commit region runs
from the bottom of the regular commit region to the last round).
Exclusion ranges are decsim's own device (a strong re-decode leaves the
weak decoder's committed faults uncommitted,
decsim/decoders/strong_escalation) with no paper referent.
"""

import numpy
import pytest
import qldpc.decoders
import stim

import decsim.message as message
from decsim.detector_error_model import (
    detector_chronology,
    fault_model_contracts,
    stim_fault_catalog,
    window_model_builders,
)

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_model_contracts.FaultRepresentation.PHYSICAL
GRAPHLIKE_REQUIRED = fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
LINKED_REQUIRED = fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED


def surface_code_circuit(rounds):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )


def owned_faults(model):
    faults = model.require_faults(GRAPHLIKE)
    return [
        fault_index
        for fault_index, is_owned in zip(faults.source_fault_ids, faults.owned)
        if is_owned
    ]


def owned_identities(catalog, model):
    """The (detectors, observables) identity of every fault the window owns."""
    return {
        (
            catalog.detector_sets[fault_index],
            catalog.observable_sets[fault_index],
        )
        for fault_index in owned_faults(model)
    }


def column_identity(detector_matrix, observable_matrix, column):
    detector_column = detector_matrix[:, column]
    observable_column = observable_matrix[:, column]
    detectors = detector_column.indices.tolist()
    observables = observable_column.indices.tolist()
    return tuple(sorted(detectors)), tuple(sorted(observables))


class QldpcWindow:
    """One window as qLDPC cut it, its two regions plus its committed errors."""

    def __init__(self, detection_detectors, commit_detectors, committed):
        self.detection_detectors = tuple(sorted(detection_detectors))
        self.commit_detectors = tuple(sorted(commit_detectors))
        self.committed = committed


def qldpc_sliding_windows(circuit, rounds, window_size, stride):
    """The windows qLDPC cuts over the circuit's decomposed error model."""
    round_of_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, rounds
    )
    model = circuit.detector_error_model(decompose_errors=True)
    decoder = qldpc.decoders.SlidingWindowDecoder(
        window_size,
        stride,
        detector_to_time=lambda detector: round_of_detector[detector],
        decompose_errors=True,
    )
    compiled = decoder.compile_decoder_for_dem(model)
    detector_matrix = compiled.dem_arrays.detector_flip_matrix.tocsc()
    observable_matrix = compiled.dem_arrays.observable_flip_matrix.tocsc()
    windows = []
    for (detection, commit), errors in zip(
        decoder.windows, compiled.window_errors
    ):
        committed_columns = numpy.nonzero(errors[0])
        committed = {
            column_identity(detector_matrix, observable_matrix, column)
            for column in committed_columns[0]
        }
        window = QldpcWindow(detection, commit, committed)
        windows.append(window)
    return windows


def test_a_sliding_plan_owns_every_fault_exactly_once():
    circuit = surface_code_circuit(6)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 2, 3), (3, 4, 5), (5, 6, 6)],
        round_count=6,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    first_owns = owned_faults(models[0])
    second_owns = owned_faults(models[1])
    last_owns = owned_faults(models[2])
    owned_together = first_owns + second_owns + last_owns
    every_owned = sorted(owned_together)
    assert len(models) == 3
    assert len(first_owns) == 48
    assert len(second_owns) == 64
    assert len(last_owns) == 62
    # The graphlike catalog of the distance-3, six-round circuit.
    assert every_owned == list(range(174))
    assert len(models[2].detector_ids) == 20


def test_the_windows_match_qldpcs_sliding_window_rule_window_by_window():
    circuit = surface_code_circuit(8)
    catalogs, _ = stim_fault_catalog.prepare_fault_catalogs(
        circuit, GRAPHLIKE_REQUIRED
    )
    catalog = catalogs[GRAPHLIKE]
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 2, 3), (3, 4, 5), (5, 8, 8)],
        round_count=8,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    first, second, last = qldpc_sliding_windows(
        circuit, 8, window_size=3, stride=2
    )
    assert first.detection_detectors == models[0].detector_ids
    assert first.commit_detectors == tuple(range(0, 12))
    assert first.committed == owned_identities(catalog, models[0])
    assert second.detection_detectors == models[1].detector_ids
    assert second.commit_detectors == tuple(range(12, 28))
    assert second.committed == owned_identities(catalog, models[1])
    assert last.detection_detectors == models[2].detector_ids
    assert last.commit_detectors == tuple(range(28, 64))
    assert last.committed == owned_identities(catalog, models[2])
    assert len(first.committed) == 48
    assert len(second.committed) == 64
    assert len(last.committed) == 126


def test_a_partial_plan_is_not_terminal_and_leaves_faults_unowned():
    circuit = surface_code_circuit(4)
    # The second window's buffer reaches round 4 and its commit rounds
    # stop at 3: it sees the round-4 faults and owns none of them.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 2), (3, 3, 3, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    first_owns = owned_faults(models[0])
    second_owns = owned_faults(models[1])
    owned_together = first_owns + second_owns
    every_owned = sorted(owned_together)
    unowned = {60, 65, 71, 78, 80, 81, 84} | set(range(87, 110))
    assert models[1].detector_ids == tuple(range(12, 32))
    assert len(first_owns) == 48
    assert len(second_owns) == 32
    assert len(every_owned) == 80
    assert set(every_owned) == set(range(110)) - unowned


def test_a_gap_between_commit_regions_is_refused():
    circuit = surface_code_circuit(6)
    with pytest.raises(ValueError, match="contiguous"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3), (4, 5, 6)],
            round_count=6,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_overlapping_commit_regions_are_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="without gaps or overlaps"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3), (2, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_commit_region_past_the_last_round_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="exceeds round_count"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3), (3, 5, 5)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_an_unordered_window_entry_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="bounds are not ordered"):
        window_model_builders.build_window_error_models(
            circuit,
            [(3, 2, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_window_bound_that_is_not_positive_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="bounds must be positive"):
        window_model_builders.build_window_error_models(
            circuit,
            [(0, 2, 3)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_window_bound_that_is_not_a_built_in_int_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="built-in ints"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2.0, 3)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_an_inverted_exclusion_range_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="range 3-1 is inverted"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=((3, 1),),
        )


def test_an_exclusion_endpoint_that_is_not_a_built_in_int_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="built-in integer"):
        window_model_builders.build_single_window_error_model(
            circuit,
            (1, 1, 2, 3),
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            exclude_faults_touching=(1.0, 2),
        )


def test_a_single_window_keeps_excluded_faults_but_does_not_own_them():
    circuit = surface_code_circuit(4)
    plain = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
    )
    excluded = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        exclude_faults_touching=(1, 1),
    )
    plain_faults = plain.require_faults(GRAPHLIKE)
    excluded_faults = excluded.require_faults(GRAPHLIKE)
    plain_owned = owned_faults(plain)
    excluded_owned = owned_faults(excluded)
    # The faults that flip a round-1 detector (detectors 0 to 3).
    touching_round_one = {0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19}
    assert excluded_faults.source_fault_ids == plain_faults.source_fault_ids
    assert len(plain_owned) == 48
    assert len(excluded_owned) == 32
    assert set(plain_owned) - set(excluded_owned) == touching_round_one


def test_exclusion_ranges_that_cover_the_window_leave_nothing_owned():
    circuit = surface_code_circuit(4)
    builder = (
        window_model_builders.build_single_window_error_model_with_exclusions
    )
    excluded = builder(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=((1, 1), (2, 3)),
    )
    faults = excluded.require_faults(GRAPHLIKE)
    assert not faults.owned.any()
    assert faults.boundary_flips == {}


def test_only_a_window_whose_commit_rounds_reach_the_last_round_is_terminal():
    circuit = surface_code_circuit(4)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 4), (3, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    first_faults = models[0].require_faults(GRAPHLIKE)
    second_faults = models[1].require_faults(GRAPHLIKE)
    first_owns = owned_faults(models[0])
    second_owns = owned_faults(models[1])
    # The first window's buffer reaches round 4, so it sees the whole
    # catalog of 110 faults; its commit rounds stop at 2.
    assert models[0].detector_ids == tuple(range(32))
    assert len(first_faults.source_fault_ids) == 110
    assert len(first_owns) == 48
    # Fault 26 flips detector 12, a round-3 detector: the first window
    # sees it and does not own it; the second window does.
    assert first_faults.source_fault_ids[26] == 26
    assert not first_faults.owned[26]
    assert 26 in second_owns
    assert len(second_owns) == 62
    assert second_faults.owned.all()


def test_a_front_buffer_round_is_decoded_but_not_owned():
    circuit = surface_code_circuit(6)
    # The runtime's geometry: the buffer starts buffer_rounds before the
    # commit region, here two rounds before commit rounds 3 and 4.
    model = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 3, 4, 5),
        round_count=6,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
    )
    faults = model.require_faults(GRAPHLIKE)
    owned_count = faults.owned.sum()
    assert model.detector_ids == tuple(range(36))
    assert len(faults.source_fault_ids) == 144
    assert owned_count == 82
    # Fault 0 flips detector 0 only (round 1) and fault 5 flips detector
    # 4 only (round 2): both decoded, neither owned.
    assert faults.source_fault_ids[0] == 0
    assert not faults.owned[0]
    assert faults.source_fault_ids[5] == 5
    assert not faults.owned[5]


def test_the_terminal_window_owns_every_uncommitted_fault_it_sees():
    circuit = surface_code_circuit(4)
    # Commit rounds 3 and 4 reach the last round, so the window is
    # terminal; its front buffer starts at round 1.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    faults = models[0].require_faults(GRAPHLIKE)
    assert models[0].detector_ids == tuple(range(32))
    assert len(faults.source_fault_ids) == 110
    assert faults.owned.all()
    # Fault 0 flips detector 0 only, a round-1 detector in the front
    # buffer: nobody committed it before, so the terminal window does.
    assert faults.source_fault_ids[0] == 0
    assert faults.owned[0]


def test_the_terminal_window_owns_every_uncommitted_fault_with_edges_given():
    circuit = surface_code_circuit(4)
    # The runtime always hands the plan builder a tuple of edges, so its
    # ownership is compiled from the graph even when the tuple is empty.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=(),
    )
    faults = models[0].require_faults(GRAPHLIKE)
    assert models[0].detector_ids == tuple(range(32))
    assert len(faults.source_fault_ids) == 110
    assert faults.owned.all()
    assert faults.source_fault_ids[0] == 0
    assert faults.owned[0]


def test_a_plan_tiling_the_operation_owns_the_same_faults_on_both_paths():
    circuit = surface_code_circuit(4)
    in_plan_order = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 4), (1, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    from_the_graph = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 4), (1, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1),),
    )
    first_in_plan_order = owned_faults(in_plan_order[0])
    last_in_plan_order = owned_faults(in_plan_order[1])
    first_from_the_graph = owned_faults(from_the_graph[0])
    last_from_the_graph = owned_faults(from_the_graph[1])
    assert first_from_the_graph == first_in_plan_order
    assert last_from_the_graph == last_in_plan_order
    assert len(first_from_the_graph) == 48
    assert len(last_from_the_graph) == 62


def test_two_windows_with_front_buffers_partition_the_catalog_between_them():
    circuit = surface_code_circuit(4)
    # The runtime's geometry with two buffer rounds: both windows see the
    # whole operation; only the second is terminal.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 4), (1, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    first_faults = models[0].require_faults(GRAPHLIKE)
    last_faults = models[1].require_faults(GRAPHLIKE)
    first_owns = owned_faults(models[0])
    last_owns = owned_faults(models[1])
    assert len(first_faults.source_fault_ids) == 110
    assert len(first_owns) == 48
    assert len(last_faults.source_fault_ids) == 62
    assert len(last_owns) == 62
    assert set(last_owns) == set(range(110)) - set(first_owns)


def test_a_single_window_built_alone_is_never_terminal():
    circuit = surface_code_circuit(4)
    model = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 3, 4, 4),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
    )
    faults = model.require_faults(GRAPHLIKE)
    owned_count = faults.owned.sum()
    assert model.detector_ids == tuple(range(32))
    assert len(faults.source_fault_ids) == 110
    assert owned_count == 80
    # Fault 0 (detector 0, round 1) lies in the front buffer: decoded,
    # not owned, although the commit rounds reach the last round.
    assert faults.source_fault_ids[0] == 0
    assert not faults.owned[0]


def test_a_linked_plan_of_several_windows_with_an_exclusion_range_is_refused():
    circuit = surface_code_circuit(6)
    # The first window would commit the component that flips detector 13
    # of physical fault 167 while the excluded round 4 kept the fault
    # itself uncommitted; the second window would then hold the fault
    # without its component.
    with pytest.raises(
        ValueError,
        match="linked fault model requirement with fault_exclusion_ranges",
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 3, 4), (3, 4, 6, 6)],
            round_count=6,
            fault_model_requirement=(
                fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
            ),
            fault_exclusion_ranges=((4, 4),),
        )


def test_a_linked_plan_of_several_windows_without_a_range_is_accepted():
    circuit = surface_code_circuit(4)
    # The belief-matching runtime path: window_manager passes no range.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=(
            fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        ),
        fault_exclusion_ranges=(),
    )
    first_projection = models[0].physical_to_graphlike_detector_projection
    second_projection = models[1].physical_to_graphlike_detector_projection
    first_owns = owned_faults(models[0])
    last_owns = owned_faults(models[1])
    assert len(models) == 2
    assert first_projection.shape == (80, 336)
    assert second_projection.shape == (62, 227)
    assert len(first_owns) == 48
    assert len(last_owns) == 62


def test_a_graphlike_plan_of_several_windows_keeps_its_exclusion_range():
    circuit = surface_code_circuit(4)
    plain = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    excluded = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=((1, 1),),
    )
    first_plain = owned_faults(plain[0])
    first_excluded = owned_faults(excluded[0])
    last_plain = owned_faults(plain[1])
    last_excluded = owned_faults(excluded[1])
    # The 16 graphlike faults that flip a round-1 detector stay unowned
    # by the first window, and the terminal window never sees them.
    assert len(first_plain) == 48
    assert len(first_excluded) == 32
    assert len(last_plain) == 62
    assert len(last_excluded) == 62


def test_a_linked_plan_of_one_window_keeps_its_exclusion_range():
    circuit = surface_code_circuit(4)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 4, 4)],
        round_count=4,
        fault_model_requirement=(
            fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        ),
        fault_exclusion_ranges=((1, 1),),
    )
    graphlike = models[0].require_faults(GRAPHLIKE)
    # The 16 graphlike faults that flip a round-1 detector stay unowned.
    assert len(graphlike.source_fault_ids) == 110
    assert graphlike.owned.sum() == 94
    assert not graphlike.owned[0]


def test_an_exclusion_range_that_is_a_bare_int_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError, match="must be a pair of built-in integers, got 5"
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(5,),
        )


def test_exclusion_ranges_given_as_a_list_of_lists_build_the_same_model():
    circuit = surface_code_circuit(4)
    from_tuple = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=((1, 1),),
    )
    from_list = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 4, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=[[1, 1]],
    )
    tuple_faults = from_tuple[0].require_faults(GRAPHLIKE)
    list_faults = from_list[0].require_faults(GRAPHLIKE)
    assert list_faults.source_fault_ids == tuple_faults.source_fault_ids
    assert list_faults.owned.tolist() == tuple_faults.owned.tolist()
    assert list_faults.owned.sum() == 94


def test_exclusion_ranges_given_as_none_are_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError, match="must be a sequence of ranges, got None"
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=None,
        )


def test_an_exclusion_range_before_the_first_round_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError, match=r"range 0-0 lies outside rounds 1\.\.4"
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=((0, 0),),
        )


def test_an_exclusion_range_after_the_last_round_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError, match=r"range 9-9 lies outside rounds 1\.\.4"
    ):
        window_model_builders.build_single_window_error_model(
            circuit,
            (1, 1, 2, 3),
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            exclude_faults_touching=(9, 9),
        )


def test_a_last_buffer_round_past_round_count_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="buffer region exceeds round_count"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4, 9)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_single_window_whose_buffer_passes_round_count_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="buffer region exceeds round_count"):
        window_model_builders.build_single_window_error_model(
            circuit,
            (1, 3, 4, 9),
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
        )


def test_an_empty_plan_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="at least one window"):
        window_model_builders.build_window_error_models(
            circuit,
            [],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_window_entry_with_two_bounds_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="three or four bounds, got 2"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_window_entry_with_five_bounds_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="three or four bounds, got 5"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 3, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_window_bound_that_is_a_bool_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="window bounds must be built-in ints"):
        window_model_builders.build_window_error_models(
            circuit,
            [(True, 2, 3)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_an_exclusion_range_with_three_values_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError,
        match=r"must be a pair of built-in integers, got \(1, 2, 3\)",
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=((1, 2, 3),),
        )


def test_an_exclusion_endpoint_that_is_a_bool_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError,
        match=r"must be a pair of built-in integers, got \(True, 3\)",
    ):
        window_model_builders.build_single_window_error_model(
            circuit,
            (1, 1, 2, 3),
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            exclude_faults_touching=(True, 3),
        )


def test_an_exclusion_endpoint_that_is_a_bool_in_second_place_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError,
        match=r"must be a pair of built-in integers, got \(3, True\)",
    ):
        window_model_builders.build_single_window_error_model(
            circuit,
            (1, 1, 2, 3),
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            exclude_faults_touching=(3, True),
        )


def test_an_exclusion_endpoint_that_is_a_float_in_second_place_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError,
        match=r"must be a pair of built-in integers, got \(2, 1.0\)",
    ):
        window_model_builders.build_single_window_error_model(
            circuit,
            (1, 1, 2, 3),
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            exclude_faults_touching=(2, 1.0),
        )


def test_a_window_short_of_the_last_round_is_not_terminal_with_edges_given():
    circuit = surface_code_circuit(4)
    # The second window's buffer reaches round 4 and its commit rounds
    # stop at 3; ownership is compiled from the graph.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 3, 4)],
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1),),
    )
    first_owns = owned_faults(models[0])
    second_owns = owned_faults(models[1])
    owned_together = first_owns + second_owns
    # The 30 faults that flip only round-4 detectors.
    unowned = {60, 65, 71, 78, 80, 81, 84} | set(range(87, 110))
    assert models[1].detector_ids == tuple(range(12, 32))
    assert len(first_owns) == 48
    assert len(second_owns) == 32
    assert set(owned_together) == set(range(110)) - unowned


def test_a_linked_window_depending_on_the_terminal_window_is_refused():
    circuit = surface_code_circuit(6)
    # Window 1 is terminal and window 0 depends on it. Physical fault 11
    # touches rounds 1 and 2 and goes to window 0; its component 9 flips
    # a round-1 detector only, which no commit round reaches, so the
    # terminal window owns it, and window 0 would keep 11 without 9.
    with pytest.raises(
        ValueError,
        match=(
            "window 0 keeps physical fault 11 while window 1, which it "
            "depends on, owns graphlike component 9"
        ),
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3, 3), (1, 4, 6, 6)],
            round_count=6,
            fault_model_requirement=LINKED_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((1, 0),),
        )


def test_a_linked_chain_builds_with_every_component_beside_its_fault():
    circuit = surface_code_circuit(6)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 3, 3), (1, 4, 6, 6)],
        round_count=6,
        fault_model_requirement=LINKED_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1),),
    )
    first_projection = models[0].physical_to_graphlike_detector_projection
    second_projection = models[1].physical_to_graphlike_detector_projection
    first_owns = owned_faults(models[0])
    last_owns = owned_faults(models[1])
    assert len(models) == 2
    assert first_projection.shape == (80, 336)
    assert second_projection.shape == (94, 367)
    assert len(first_owns) == 80
    assert len(last_owns) == 94


def test_a_linked_sandwich_builds_with_every_component_beside_its_fault():
    circuit = surface_code_circuit(5)
    # Tan's edges under the generic protocol: the seam depends on both
    # neighbours, and every fault it keeps is whole.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 3, 3), (3, 4, 5, 5)],
        round_count=5,
        fault_model_requirement=LINKED_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1), (2, 1)),
    )
    before_projection = models[0].physical_to_graphlike_detector_projection
    seam_projection = models[1].physical_to_graphlike_detector_projection
    after_projection = models[2].physical_to_graphlike_detector_projection
    before_owns = owned_faults(models[0])
    seam_owns = owned_faults(models[1])
    after_owns = owned_faults(models[2])
    assert before_projection.shape == (80, 336)
    assert seam_projection.shape == (14, 31)
    assert after_projection.shape == (112, 476)
    assert len(before_owns) == 48
    assert len(seam_owns) == 14
    assert len(after_owns) == 80


def test_a_component_off_the_windows_rows_owned_by_an_ancestor_is_allowed():
    circuit = surface_code_circuit(6)
    # Window 0 decodes rounds 2 and 3 and depends on the terminal window.
    # Physical fault 11 touches rounds 1 and 2 and belongs to window 0;
    # its component 9 flips a round-1 detector only, which the terminal
    # window owns. That component lands on no row of window 0, so the
    # window's columns still add up and the plan builds.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(2, 2, 3, 3), (1, 4, 6, 6)],
        round_count=6,
        fault_model_requirement=LINKED_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((1, 0),),
    )
    dependent_projection = models[0].physical_to_graphlike_detector_projection
    terminal_projection = models[1].physical_to_graphlike_detector_projection
    dependent_owns = owned_faults(models[0])
    terminal_owns = owned_faults(models[1])
    assert models[0].detector_ids == tuple(range(4, 20))
    assert dependent_projection.shape == (55, 220)
    assert terminal_projection.shape == (174, 703)
    assert len(dependent_owns) == 55
    assert len(terminal_owns) == 119


def test_a_component_owned_by_an_indirect_ancestor_is_refused():
    circuit = surface_code_circuit(7)
    # Depths (0, 2, 0, 1, 4, 3): window 4 depends on window 5 alone and
    # reaches window 1 only through it. Physical fault 87 touches rounds
    # 2 and 3 and goes to window 0, which window 4 does not depend on;
    # its component 31 touches round 3 alone and goes to window 1.
    # Window 4 decodes rounds 2 to 6, keeps 87, and would drop 31, which
    # its direct parent does not own.
    with pytest.raises(
        ValueError,
        match=(
            "window 4 keeps physical fault 87 while window 1, which it "
            "depends on, owns graphlike component 31"
        ),
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [
                (1, 1, 2, 2),
                (3, 3, 3, 3),
                (4, 4, 4, 4),
                (5, 5, 5, 5),
                (2, 6, 6, 6),
                (7, 7, 7, 7),
            ],
            round_count=7,
            fault_model_requirement=LINKED_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((3, 1), (1, 5), (5, 4), (2, 3)),
        )


def test_a_component_nobody_owns_stays_beside_its_fault():
    circuit = surface_code_circuit(6)
    # No commit round reaches round 1 and no window is terminal, so the
    # faults of round 1 are owned by nobody. Window 0 depends on window 1
    # and keeps physical fault 0 (round 1), whose component 0 has no
    # owner, and physical fault 11 (rounds 1 and 2), whose component 9
    # (round 1) has no owner: nobody is not an ancestor, so both faults
    # are whole and the plan builds.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 2, 3, 3), (1, 4, 5, 5)],
        round_count=6,
        fault_model_requirement=LINKED_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((1, 0),),
    )
    dependent_projection = models[0].physical_to_graphlike_detector_projection
    parent_projection = models[1].physical_to_graphlike_detector_projection
    physical = models[0].require_faults(PHYSICAL)
    graphlike = models[0].require_faults(GRAPHLIKE)
    fault_0_column = physical.source_fault_ids.index(0)
    fault_11_column = physical.source_fault_ids.index(11)
    component_0_column = graphlike.source_fault_ids.index(0)
    component_9_column = graphlike.source_fault_ids.index(9)
    assert dependent_projection.shape == (62, 227)
    assert parent_projection.shape == (144, 616)
    assert not physical.owned[fault_0_column]
    assert physical.owned[fault_11_column]
    assert not graphlike.owned[component_0_column]
    assert not graphlike.owned[component_9_column]


def test_a_linked_terminal_window_with_empty_edges_builds():
    circuit = surface_code_circuit(4)
    # One window with an empty tuple of edges: ownership is compiled from
    # a graph with no edge, the window has no ancestor, so the linked
    # check refuses nothing, and the terminal law gives it every fault.
    # That empty edges compile from the graph rather than plan order is
    # pinned in test_window_ownership_dag by
    # test_a_fault_between_two_windows_of_the_same_depth_has_no_owner.
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 3, 4, 4)],
        round_count=4,
        fault_model_requirement=LINKED_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=(),
    )
    projection = models[0].physical_to_graphlike_detector_projection
    faults = models[0].require_faults(GRAPHLIKE)
    assert projection.shape == (110, 423)
    assert len(faults.source_fault_ids) == 110
    assert faults.owned.all()


def test_a_gap_is_reported_before_the_linked_check():
    circuit = surface_code_circuit(6)
    with pytest.raises(ValueError, match="contiguous"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3, 3), (1, 5, 6, 6)],
            round_count=6,
            fault_model_requirement=LINKED_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((1, 0),),
        )


def test_a_commit_past_round_count_is_reported_before_the_linked_check():
    circuit = surface_code_circuit(6)
    with pytest.raises(
        ValueError, match="window commit or buffer region exceeds round_count"
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3, 3), (1, 4, 7, 7)],
            round_count=6,
            fault_model_requirement=LINKED_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((1, 0),),
        )


def test_a_protocol_defect_is_reported_before_the_linked_check():
    circuit = surface_code_circuit(6)
    with pytest.raises(ValueError, match="requires exactly the graphlike"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 3, 3), (4, 4, 5, 5), (2, 6, 6, 6)],
            round_count=6,
            fault_model_requirement=LINKED_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((2, 1), (1, 3)),
            window_protocol=message.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
        )


def test_a_closed_window_defect_is_reported_before_the_linked_check():
    circuit = surface_code_circuit(6)
    with pytest.raises(ValueError, match="must be a dependency destination"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 3, 3), (4, 4, 5, 5), (2, 6, 6, 6)],
            round_count=6,
            fault_model_requirement=LINKED_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((2, 1), (1, 3)),
            closed_temporal_boundary_windows=(7,),
        )


def test_a_declared_detector_round_map_moves_a_fault_between_windows():
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=2,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )
    # Stim's coordinates put detectors 2 and 3 in round 2; the declared
    # map puts them in round 1. Fault 5 flips detector 2 only.
    from_coordinates = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 1, 1), (2, 2, 2, 2)],
        round_count=2,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    declared = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 1, 1), (2, 2, 2, 2)],
        round_count=2,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
        detector_rounds={0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2},
    )
    assert from_coordinates[0].detector_ids == (0, 1)
    assert from_coordinates[1].detector_ids == (2, 3, 4, 5)
    assert owned_faults(from_coordinates[0]) == [0, 1, 2, 3, 4]
    assert owned_faults(from_coordinates[1]) == [5, 6, 7, 8, 9, 10, 11, 12]
    assert declared[0].detector_ids == (0, 1, 2, 3)
    assert declared[1].detector_ids == (4, 5)
    assert owned_faults(declared[0]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert owned_faults(declared[1]) == [10, 11, 12]


def test_a_single_window_excluding_a_list_pair_builds_the_same_model():
    circuit = surface_code_circuit(4)
    from_tuple = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        exclude_faults_touching=(1, 1),
    )
    from_list = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        exclude_faults_touching=[1, 1],
    )
    tuple_faults = from_tuple.require_faults(GRAPHLIKE)
    list_faults = from_list.require_faults(GRAPHLIKE)
    assert list_faults.source_fault_ids == tuple_faults.source_fault_ids
    assert list_faults.owned.tolist() == tuple_faults.owned.tolist()
    assert list_faults.owned.sum() == 32


def test_a_single_window_with_list_ranges_builds_the_same_model():
    circuit = surface_code_circuit(4)
    builder = (
        window_model_builders.build_single_window_error_model_with_exclusions
    )
    from_tuple = builder(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=((1, 1), (3, 3)),
    )
    from_list = builder(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=[[1, 1], [3, 3]],
    )
    tuple_faults = from_tuple.require_faults(GRAPHLIKE)
    list_faults = from_list.require_faults(GRAPHLIKE)
    assert list_faults.source_fault_ids == tuple_faults.source_fault_ids
    assert list_faults.owned.tolist() == tuple_faults.owned.tolist()
    # Only the 14 faults that flip round-2 detectors alone stay owned.
    assert list_faults.owned.sum() == 14
