"""The union_find row against ldpc's UnionFindDecoder on the same graph.

With uniform priors every edge has the same length, so decsim's weighted
growth (Delfosse and Nickerson 1709.06218, Huang, Newman and Brown
2004.04693, both in tmp/uf-decoder-research/papers) is the paper's
uniform growth; ldpc's peeling decoder (ldpc.union_find_decoder) is the
referent. Every syndrome an error produces is satisfiable: both
decoders reproduce it, and on a single fault both name that fault.
ldpc's decoder does not return on an unsatisfiable syndrome, so the
unsatisfiable side is decsim's own
(tests/15_decoders/test_union_find_unsatisfiable.py).
"""

import random

import numpy
import scipy.sparse
from ldpc.union_find_decoder import UnionFindDecoder as LdpcUnionFind

import decsim.decoders.union_find.decoder as union_find
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message

# rows 0..3 and 4..7 are two boundaryless 4-cycles, rows 8..11 a chain
# with boundary edges at both ends
CHECK = [
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
]


def _model():
    check = numpy.asarray(CHECK, dtype=numpy.uint8)
    column_count = check.shape[1]
    placed = fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=numpy.full(column_count, 0.1),
        observables=numpy.zeros((1, column_count), dtype=numpy.uint8),
        owned=numpy.ones(column_count, dtype=bool),
        source_fault_ids=tuple(range(column_count)),
        boundary_flips={},
    )
    rows = tuple(range(check.shape[0]))
    return fault_models.WindowErrorModel(
        detector_ids=rows,
        detector_coordinates=None,
        defect_positions={row: (1, row) for row in rows},
        graphlike_faults=placed,
        physical_faults=None,
        physical_to_graphlike_detector_projection=None,
    )


def _job(model, syndrome) -> message.DecodeJob:
    bits = tuple(int(bit) for bit in syndrome)
    payload = message.SyndromePayload(
        operation_id=1,
        patch_id=0,
        round_index=1,
        bits=bits,
        code=None,
        n_fragments=1,
        fragment_index=0,
        size_bits=len(bits),
    )
    return message.DecodeJob(
        op_id=1,
        window_id=0,
        n_rounds=1,
        dem=model,
        payloads=[payload],
        label="W0",
    )


def _referee():
    check = scipy.sparse.csr_matrix(numpy.asarray(CHECK, dtype=numpy.uint8))
    return LdpcUnionFind(check, uf_method="peeling")


def test_both_decoders_reproduce_every_syndrome_an_error_produces():
    """A property test: 200 random errors, both corrections reproduce them."""
    model = _model()
    check = numpy.asarray(CHECK, dtype=numpy.uint8)
    referee = _referee()
    row = union_find.UnionFindDecoder(weight_step=0.1)
    rng = random.Random(11)
    for _ in range(200):
        error = numpy.array(
            [int(rng.random() < 0.2) for _ in range(check.shape[1])],
            dtype=numpy.uint8,
        )
        syndrome = (check @ error) % 2
        ldpc_correction = numpy.asarray(
            referee.decode(syndrome), dtype=numpy.uint8
        )
        assert numpy.array_equal((check @ ldpc_correction) % 2, syndrome)
        decoded = row.decode_with_growth_evidence(_job(model, syndrome))
        assert decoded.hard_evidence.unmatched_detectors == ()
        decsim_correction = numpy.asarray(
            decoded.hard_evidence.selected_faults, dtype=numpy.uint8
        )
        assert numpy.array_equal((check @ decsim_correction) % 2, syndrome)


def test_a_single_fault_is_named_by_both_decoders():
    model = _model()
    check = numpy.asarray(CHECK, dtype=numpy.uint8)
    referee = _referee()
    row = union_find.UnionFindDecoder(weight_step=0.1)
    for fault in range(check.shape[1]):
        syndrome = check[:, fault]
        ldpc_correction = numpy.asarray(
            referee.decode(syndrome), dtype=numpy.uint8
        )
        decoded = row.decode_with_growth_evidence(_job(model, syndrome))
        decsim_correction = numpy.asarray(
            decoded.hard_evidence.selected_faults, dtype=numpy.uint8
        )
        assert decsim_correction.tolist() == ldpc_correction.tolist()
        assert int(decsim_correction.sum()) == 1
