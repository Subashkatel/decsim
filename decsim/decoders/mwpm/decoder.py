"""The PyMatching adapter: minimum-weight perfect matching on one window.

PyMatching's Matching.from_check_matrix on the window's graphlike faults
with log-odds weights, parallel faults merged as independent errors
(the convention of Stim detector error models and of PyMatching's DEM
loader), one matching per live window model, warmed on three columns
before the first timed call. Higgott and Gidney, Sparse Blossom
(2303.15933, tmp/uf-decoder-research/papers) is the algorithm behind
the call.
"""

import numpy
import pymatching

import decsim.decoders.decoder as decoder_module
import decsim.decoders.mwpm.weights as weights
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models

WARM_UP_COLUMNS = 3


class PyMatchingDecoder(decoder_module.WindowDecoderBase):
    """Decode one window with PyMatching.

    Measured mode times ``matching.decode`` only, not syndrome extraction
    or result construction. Measurements are of one call on one thread:
    with several units they do not prove parallel hardware.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE

    def compile(self, faults, model=None):
        """The matching graph of one placed model, validated and warm."""
        del model
        fault_identity.validate_graphlike_matrices(
            faults.check,
            faults.observables,
            location="PyMatching window model",
        )
        # PyMatching normalises the matrix it is given in place; the placed
        # matrix is frozen, so it gets a copy (one per model, cached).
        # Faults sharing the same detector endpoints are independent error
        # mechanisms and merge into one edge with the combined probability,
        # the convention of Stim detector error models and of PyMatching's
        # DEM loader; a window restriction folds distinct faults onto the
        # same in-window endpoints, so a window graph always has some.
        check = faults.check.copy()
        edge_weights = self._weights_for(faults)
        error_probabilities = weights.finite_priors(faults.priors)
        matching = pymatching.Matching.from_check_matrix(
            check,
            weights=edge_weights,
            error_probabilities=error_probabilities,
            merge_strategy="independent",
        )
        _warm_up(matching, faults)
        return matching

    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """The matching's correction, or an empty one marked invalid.

        PyMatching raises when a syndrome has odd parity in a boundaryless
        component; no valid plan produces one, so that case is reported as
        an empty correction with INVALID_CORRECTION rather than ending the
        run.
        """
        del model
        try:
            return backend.decode(syndrome), None
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            fault_count = faults.check.shape[1]
            empty = numpy.zeros(fault_count, dtype=numpy.uint8)
            invalid = (
                window_decode_results.BackendDecodeStatus.INVALID_CORRECTION
            )
            return empty, invalid

    def _weights_for(self, faults):
        return weights.matching_weights(faults.priors)


class UnweightedPyMatchingDecoder(PyMatchingDecoder):
    """Weight-oblivious MWPM: same matching graph, every edge at weight 1.

    A deliberately coarse weak tier (a hardware matcher without
    weighted-edge support): at circuit-level noise it decodes measurably
    worse than weighted MWPM because hook-error paths are no longer
    penalized.
    """

    def _weights_for(self, faults):
        fault_count = len(faults.priors)
        return numpy.ones(fault_count)


def _warm_up(matching, faults) -> None:
    """Decode a few one-column syndromes so the graph is built and warm.

    PyMatching builds its internal graph lazily and finishes warming
    only once it has matched real defects; a running software decoder
    has the window graph prebuilt and warm, so decode a few defects
    before the first timed call. Each warm-up syndrome is the detector
    set of one column, which that column alone explains, so it is
    satisfiable on any graph (a boundaryless toric component would
    reject an arbitrary detector pair).
    """
    check = faults.check
    detector_count = check.shape[0]
    warmed = 0
    for column in range(check.shape[1]):
        rows = _column_rows(check, column)
        if rows.size == 0:
            continue
        syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
        syndrome[rows] = 1
        matching.decode(syndrome)
        warmed += 1
        if warmed == WARM_UP_COLUMNS:
            return


def _column_rows(check, column: int):
    start = check.indptr[column]
    end = check.indptr[column + 1]
    return check.indices[start:end]
