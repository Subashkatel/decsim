"""The decoder-facing records hold what they say and refuse the rest.

These tests check the module's own contract: a placed model and a window
model hand a decoder one column per fault as frozen uint8 csc matrices,
so a window shared between decoders cannot be edited by one of them; a
requirement joins with another; the module is a leaf that loads no
numeric library when it is imported on its own.
"""

import dataclasses
import pathlib
import subprocess
import sys

import numpy
import pytest
import scipy.sparse

from decsim.detector_error_model import fault_model_contracts

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_model_contracts.FaultRepresentation.PHYSICAL

LEAF_IMPORT_PROBE = """
import importlib.util
import sys

specification = importlib.util.spec_from_file_location("leaf", sys.argv[1])
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
loaded = sorted(name for name in sys.modules if name in ("numpy", "scipy"))
print(loaded)
"""


def placed_model():
    check = scipy.sparse.csc_matrix([[1, 0], [1, 1]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix([[0, 1]], dtype=numpy.uint8)
    return fault_model_contracts.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=check,
        priors=[0.1, 0.2],
        observables=observables,
        owned=[True, False],
        source_fault_ids=[4, 9],
        boundary_flips={0: [0, 1, 7]},
    )


def test_joining_two_requirements_asks_for_both_representations():
    joined = fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED.joined(
        fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    assert joined.representations == frozenset({GRAPHLIKE, PHYSICAL})
    assert joined.require_physical_to_graphlike_link is False


def test_joining_keeps_the_link_when_either_side_needs_it():
    joined = fault_model_contracts.NO_FAULT_MODEL_REQUIRED.joined(
        fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
    )
    assert joined == fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED


def test_a_link_needs_both_representations():
    only_graphlike = frozenset({GRAPHLIKE})
    with pytest.raises(ValueError, match="a physical-to-graphlike link"):
        fault_model_contracts.DecoderFaultModelRequirement(
            only_graphlike, require_physical_to_graphlike_link=True
        )


def test_frozen_sparse_columns_never_freezes_the_callers_matrix():
    # Already uint8 csc, so nothing but an explicit copy separates the
    # frozen matrix from the caller's.
    original = scipy.sparse.csc_matrix([[1, 0], [0, 1]], dtype=numpy.uint8)
    frozen = fault_model_contracts.frozen_sparse_columns(original)
    original.data[0] = 0
    assert original.data.flags.writeable is True
    assert frozen.data.flags.writeable is False
    assert frozen.dtype == numpy.uint8
    dense = frozen.toarray()
    assert dense.tolist() == [[1, 0], [0, 1]]


def test_a_placed_model_is_frozen_for_every_reader():
    placed = placed_model()
    assert placed.priors.flags.writeable is False
    assert placed.owned.flags.writeable is False
    assert placed.check.data.flags.writeable is False
    assert placed.observables.indices.flags.writeable is False
    assert placed.source_fault_ids == (4, 9)
    assert placed.boundary_flips == {0: (0, 1, 7)}
    with pytest.raises(TypeError, match="object does not support item"):
        placed.boundary_flips[1] = (2,)


def test_a_window_hands_out_the_representation_it_holds():
    graphlike = placed_model()
    check = scipy.sparse.csc_matrix([[1, 1], [0, 1]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix([[1, 0]], dtype=numpy.uint8)
    physical = fault_model_contracts.PlacedFaultModel(
        representation=PHYSICAL,
        check=check,
        priors=[0.3, 0.4],
        observables=observables,
        owned=[False, True],
        source_fault_ids=[2, 5],
        boundary_flips={1: [1]},
    )
    window = fault_model_contracts.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=None,
        defect_positions={0: (1, 0), 1: (1, 1)},
        first_commit_round=1,
        graphlike_faults=graphlike,
        physical_faults=physical,
    )
    physical_faults = window.require_faults(PHYSICAL)
    graphlike_faults = window.require_faults(GRAPHLIKE)
    assert physical_faults.representation is PHYSICAL
    assert physical_faults.source_fault_ids == (2, 5)
    assert graphlike_faults.representation is GRAPHLIKE
    assert graphlike_faults.source_fault_ids == (4, 9)


def test_a_window_refuses_a_representation_it_does_not_hold():
    placed = placed_model()
    window = fault_model_contracts.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=None,
        defect_positions={},
        first_commit_round=1,
        graphlike_faults=placed,
        physical_faults=None,
    )
    with pytest.raises(RuntimeError, match="window model does not contain"):
        window.require_faults(PHYSICAL)


def test_a_windows_crossing_faults_are_the_ones_reaching_behind_it():
    """What a window commits of the seam behind its commit region.

    The window that restarts the weak chain after a strong region owns
    the faults crossing that seam, and its own strong redo is pinned on
    exactly those and on no other fault of its correction (Toshio et al.
    2510.25222 lines 1248-1250).
    """
    check = scipy.sparse.csc_matrix([[1, 0], [1, 1]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix([[1, 1]], dtype=numpy.uint8)
    graphlike = fault_model_contracts.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=check,
        priors=[0.1, 0.2],
        observables=observables,
        owned=[True, True],
        source_fault_ids=[4, 9],
        boundary_flips={0: [0, 2], 1: [1]},
    )
    window = fault_model_contracts.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=None,
        # detector 2 sits on round 5, the round before the commit region
        defect_positions={0: (6, 0), 1: (6, 1), 2: (5, 0)},
        first_commit_round=6,
        graphlike_faults=graphlike,
        physical_faults=None,
    )
    crossing = window.crossing_fault_ids()
    assert crossing == {GRAPHLIKE: frozenset({4})}


def test_importing_the_contract_alone_loads_no_numeric_library():
    module_path = pathlib.Path(fault_model_contracts.__file__)
    probe = subprocess.run(
        [sys.executable, "-c", LEAF_IMPORT_PROBE, str(module_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "[]"


def consistent_window():
    """Two detectors with coordinates, read by the graphlike view."""
    graphlike = placed_model()
    return fault_model_contracts.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=((0.0,), (1.0,)),
        defect_positions={0: (1, 0), 1: (1, 1)},
        first_commit_round=1,
        graphlike_faults=graphlike,
        physical_faults=None,
    )


THREE_COLUMN_OBSERVABLES = scipy.sparse.csc_matrix(
    [[0, 1, 1]], dtype=numpy.uint8
)


@pytest.mark.parametrize(
    "change",
    [{"priors": [0.1]}, {"observables": THREE_COLUMN_OBSERVABLES}],
    ids=["one prior for two columns", "three observable columns for two"],
)
def test_every_fault_column_has_one_prior_and_one_observable_column(change):
    placed = placed_model()

    with pytest.raises(ValueError, match="needs one of each"):
        dataclasses.replace(placed, **change)


@pytest.mark.parametrize(
    "change, refusal",
    [
        ({"detector_ids": (0, 1, 2)}, "every row needs one id"),
        ({"detector_coordinates": ((0.0,),)}, "every id needs one"),
    ],
    ids=["three ids for two check rows", "one coordinate for two ids"],
)
def test_every_detector_id_has_one_check_row_and_one_coordinate(
    change, refusal
):
    window = consistent_window()

    with pytest.raises(ValueError, match=refusal):
        dataclasses.replace(window, **change)
