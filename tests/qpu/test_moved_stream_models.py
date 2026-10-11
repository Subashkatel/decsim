"""A stream's window models, each a short circuit's moved in time.

The referent is the whole circuit's catalog sliced in window order by
WindowSlicer, the way GrowingStimModels slices it: every window's rows,
columns, priors, observables, owned mask, handoffs, positions,
coordinates and bases must be the same, and a fault must keep one name
across every window that holds it (probe p1, Adroit config_proto/
probes8d, at d=3, 5 and 11).
"""

import numpy
import pytest

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.qpu.moved_stream_models as moved_stream_models
import decsim.records.fault_model_contracts as fault_models
import decsim.records.windows as window_records
import tests.qpu.memory_programs as memory_programs

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
BOTH = frozenset({GRAPHLIKE, PHYSICAL})
REQUIREMENT = fault_models.DecoderFaultModelRequirement(
    BOTH,
    require_physical_to_graphlike_link=True,
    detector_bases=True,
)
ROUNDS = 23
# (commit rounds, buffer rounds) of a window
SHAPES = [(5, 5), (1, 2), (4, 0), (3, 7)]


def _program():
    return memory_programs.memory_program(3, "Z", 0.006)


def _windows(commit_rounds: int, buffer_rounds: int) -> list:
    """The stream's windows: commit regions tiling its rounds in order."""
    windows = []
    commit_lo = 1
    while commit_lo <= ROUNDS:
        stride_end = commit_lo + commit_rounds - 1
        commit_hi = min(stride_end, ROUNDS)
        buffer_hi = commit_hi + buffer_rounds
        round_count = buffer_hi - commit_lo + 1
        window = window_records.Window(
            operation_id=100,
            window_index=len(windows),
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=buffer_hi,
            round_count=round_count,
            buffer_lo=commit_lo,
        )
        windows.append(window)
        commit_lo = commit_hi + 1
    return windows


def _window_at(window_index: int, commit_lo: int) -> window_records.Window:
    """A five-round commit region and five buffer rounds after it."""
    commit_hi = commit_lo + 4
    buffer_hi = commit_lo + 9
    return window_records.Window(
        operation_id=100,
        window_index=window_index,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
        round_count=10,
        buffer_lo=commit_lo,
    )


def _whole_slicer(program) -> window_slicer.WindowSlicer:
    circuit, measurement_rounds = program.assemble(ROUNDS)
    table = detector_formation.build_formation_table(
        circuit, ROUNDS, measurement_rounds=measurement_rounds
    )
    detector_rounds = table.detector_rounds()
    return window_slicer.WindowSlicer(
        circuit,
        round_count=ROUNDS,
        detector_rounds=detector_rounds,
        fault_model_requirement=REQUIREMENT,
    )


def _in_order(slicer, windows: list) -> list:
    """The whole circuit's models, sliced in window order."""
    models = []
    for window in windows:
        last_buffer_round = min(window.buffer_hi, ROUNDS)
        is_last = window.commit_hi == ROUNDS
        model = slicer.slice_window(
            window.start_round,
            window.commit_lo,
            window.commit_hi,
            last_buffer_round,
            is_last=is_last,
        )
        models.append(model)
    return models


def _assert_same(moved, whole, names: "_Names") -> None:
    """The same model, the whole circuit's positions named consistently."""
    assert moved.detector_ids == whole.detector_ids
    assert moved.detector_coordinates == whole.detector_coordinates
    assert moved.defect_positions == whole.defect_positions
    assert moved.first_commit_round == whole.first_commit_round
    assert dict(moved.detector_bases) == dict(whole.detector_bases)
    assert moved.observable_bases == whole.observable_bases
    moved_link = moved.physical_to_graphlike_detector_projection
    whole_link = whole.physical_to_graphlike_detector_projection
    assert (moved_link != whole_link).nnz == 0
    for representation in (GRAPHLIKE, PHYSICAL):
        moved_faults = moved.require_faults(representation)
        whole_faults = whole.require_faults(representation)
        assert moved_faults.check.shape == whole_faults.check.shape
        assert (moved_faults.check != whole_faults.check).nnz == 0
        observables_differ = (
            moved_faults.observables != whole_faults.observables
        )
        assert observables_differ.nnz == 0
        assert numpy.array_equal(moved_faults.owned, whole_faults.owned)
        numpy.testing.assert_allclose(
            moved_faults.priors, whole_faults.priors, rtol=0, atol=1e-12
        )
        assert moved_faults.boundary_flips == whole_faults.boundary_flips
        names.check(representation, whole_faults, moved_faults)


class _Names:
    """Each whole-circuit fault's name, which must be one and its own."""

    def __init__(self) -> None:
        self.name_by_position: dict = {}
        self.position_by_name: dict = {}

    def check(self, representation, whole_faults, moved_faults) -> None:
        pairs = zip(
            whole_faults.source_fault_ids,
            moved_faults.source_fault_ids,
            strict=True,
        )
        for position, name in pairs:
            position_key = (representation, position)
            name_key = (representation, name)
            known_name = self.name_by_position.setdefault(position_key, name)
            known_position = self.position_by_name.setdefault(
                name_key, position
            )
            assert known_name == name
            assert known_position == position


@pytest.mark.parametrize("commit_rounds, buffer_rounds", SHAPES)
def test_every_finished_windows_model_is_the_whole_circuits_in_order(
    commit_rounds: int, buffer_rounds: int
):
    program = _program()
    windows = _windows(commit_rounds, buffer_rounds)
    slicer = _whole_slicer(program)
    whole = _in_order(slicer, windows)
    models = moved_stream_models.MovedStimModels(program, REQUIREMENT)
    models.finish(ROUNDS)
    names = _Names()

    for window, whole_model in zip(windows, whole, strict=True):
        moved = models.for_window(window)
        _assert_same(moved, whole_model, names)


@pytest.mark.parametrize("commit_rounds, buffer_rounds", SHAPES)
def test_a_live_windows_model_is_the_longer_circuits(
    commit_rounds: int, buffer_rounds: int
):
    """Before the readout, every window the final round cannot reach."""
    program = _program()
    windows = _windows(commit_rounds, buffer_rounds)
    slicer = _whole_slicer(program)
    whole = _in_order(slicer, windows)
    models = moved_stream_models.MovedStimModels(program, REQUIREMENT)
    names = _Names()
    pairs = zip(windows, whole, strict=True)
    live_pairs = [
        (window, model)
        for window, model in pairs
        if window.buffer_hi + 2 < ROUNDS
    ]

    for window, whole_model in live_pairs:
        moved = models.for_window(window)
        _assert_same(moved, whole_model, names)
    assert len(live_pairs) > 2


@pytest.mark.parametrize("index", [3, -1])
def test_a_strong_window_takes_its_neighbours_commits_by_name(index: int):
    """Priors named by the weak window before it, and a fault excluded."""
    program = _program()
    windows = _windows(3, 4)
    slicer = _whole_slicer(program)
    whole = _in_order(slicer, windows)
    models = moved_stream_models.MovedStimModels(program, REQUIREMENT)
    models.finish(ROUNDS)
    window = windows[index]
    neighbour = windows[index - 1]
    neighbour_model = models.for_window(neighbour)
    whole_neighbour = whole[index - 1]
    exclusions = ((window.commit_lo, window.commit_lo),)
    named_priors = neighbour_model.owned_fault_ids()
    whole_priors = whole_neighbour.owned_fault_ids()

    moved = models.for_strong_window(window, exclusions, named_priors)
    last_buffer_round = min(window.buffer_hi, ROUNDS)
    is_last = window.commit_hi == ROUNDS
    expected = slicer.slice_window_alone(
        window.start_round,
        window.commit_lo,
        window.commit_hi,
        last_buffer_round,
        fault_exclusion_ranges=exclusions,
        explicitly_prior_faults=whole_priors,
        is_last=is_last,
    )
    names = _Names()
    _assert_same(neighbour_model, whole_neighbour, names)
    _assert_same(moved, expected, names)


def test_a_window_far_into_the_stream_moves_the_interior_model():
    """Round 10^6 costs what round 20 costs: one model, moved."""
    program = _program()
    models = moved_stream_models.MovedStimModels(program, REQUIREMENT)
    near = _window_at(3, 16)
    far = _window_at(199_999, 1_000_001)

    near_model = models.for_window(near)
    far_model = models.for_window(far)

    rounds_between = far.commit_lo - near.commit_lo
    near_faults = near_model.graphlike_faults
    far_faults = far_model.graphlike_faults
    first_row_moved = far_model.detector_ids[0] - near_model.detector_ids[0]
    first_name_moved = (
        far_faults.source_fault_ids[0] - near_faults.source_fault_ids[0]
    )
    checks_differ = near_faults.check != far_faults.check
    assert first_row_moved == 8 * rounds_between
    assert first_name_moved == (
        rounds_between * moved_stream_models.NAME_STRIDE
    )
    assert checks_differ.nnz == 0


def test_a_closed_boundary_and_a_second_finish_are_refused():
    program = _program()
    models = moved_stream_models.MovedStimModels(program, REQUIREMENT)
    window = window_records.Window(
        operation_id=100,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=3,
        round_count=3,
        closed_temporal_boundaries=True,
    )
    with pytest.raises(ValueError, match="trailing-buffer feedback"):
        models.for_window(window)
    models.finish(ROUNDS)
    with pytest.raises(RuntimeError, match="only finish once"):
        models.finish(ROUNDS)
