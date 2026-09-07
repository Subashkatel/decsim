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
import decsim.records.rounds as round_records

# rows 0..3 and 4..7 are two boundaryless 4-cycles, rows 8..11 a chain
# with boundary edges at both ends
CHECK = numpy.asarray(
    [
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
    ],
    dtype=numpy.uint8,
)
DETECTOR_COUNT, FAULT_COUNT = CHECK.shape
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE


def _model():
    priors = numpy.full(FAULT_COUNT, 0.1)
    observables = numpy.zeros((1, FAULT_COUNT), dtype=numpy.uint8)
    owned = numpy.ones(FAULT_COUNT, dtype=bool)
    placed = fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=CHECK,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(range(FAULT_COUNT)),
        boundary_flips={},
    )
    rows = tuple(range(DETECTOR_COUNT))
    defect_positions = {}
    for row in rows:
        defect_positions[row] = (1, row)
    return fault_models.WindowErrorModel(
        detector_ids=rows,
        detector_coordinates=None,
        defect_positions=defect_positions,
        graphlike_faults=placed,
        physical_faults=None,
        physical_to_graphlike_detector_projection=None,
    )


def _job(model, syndrome) -> message.DecodeJob:
    bits = []
    for bit in syndrome:
        bits.append(int(bit))
    bits = tuple(bits)
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=1,
        bits=bits,
        size_bits=len(bits),
        fragment_index=0,
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
    check = scipy.sparse.csr_matrix(CHECK)
    return LdpcUnionFind(check, uf_method="peeling")


def _reproduces(correction, syndrome) -> bool:
    reproduced = CHECK @ correction
    reproduced = reproduced % 2
    same = numpy.array_equal(reproduced, syndrome)
    return bool(same)


def _random_error(rng) -> numpy.ndarray:
    bits = []
    for _ in range(FAULT_COUNT):
        draw = rng.random()
        is_fault = draw < 0.2
        bits.append(int(is_fault))
    return numpy.array(bits, dtype=numpy.uint8)


def test_both_decoders_reproduce_every_syndrome_an_error_produces():
    """A property test: 200 random errors, both corrections reproduce them."""
    model = _model()
    referee = _referee()
    row = union_find.UnionFindDecoder(weight_step=0.1)
    rng = random.Random(11)
    for _ in range(200):
        error = _random_error(rng)
        syndrome = CHECK @ error
        syndrome = syndrome % 2
        ldpc_correction = referee.decode(syndrome)
        ldpc_correction = numpy.asarray(ldpc_correction, dtype=numpy.uint8)
        assert _reproduces(ldpc_correction, syndrome)
        job = _job(model, syndrome)
        decoded = row.decode_with_growth_evidence(job)
        assert decoded.hard_evidence.unmatched_detectors == ()
        decsim_correction = numpy.asarray(
            decoded.hard_evidence.selected_faults, dtype=numpy.uint8
        )
        assert _reproduces(decsim_correction, syndrome)


def test_a_single_fault_is_named_by_both_decoders():
    model = _model()
    referee = _referee()
    row = union_find.UnionFindDecoder(weight_step=0.1)
    for fault in range(FAULT_COUNT):
        syndrome = CHECK[:, fault]
        ldpc_correction = referee.decode(syndrome)
        ldpc_correction = numpy.asarray(ldpc_correction, dtype=numpy.uint8)
        job = _job(model, syndrome)
        decoded = row.decode_with_growth_evidence(job)
        decsim_correction = numpy.asarray(
            decoded.hard_evidence.selected_faults, dtype=numpy.uint8
        )
        assert decsim_correction.tolist() == ldpc_correction.tolist()
        selected_count = decsim_correction.sum()
        assert int(selected_count) == 1
