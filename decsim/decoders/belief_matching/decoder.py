"""The belief-matching adapter: the strong tier's decoder on one window.

Belief propagation on the physical hyperedges, then matching on the
graphlike edges with posterior weights: Higgott et al., beliefmatching's
BeliefMatching.decode (.pydeps/beliefmatching/belief_matching.py:344-358).
ldpc's BpDecoder gives the hyperedge posteriors, the window's projection
maps them onto matching edges, PyMatching matches with -log(posterior)
weights. The posterior clamp is 1e-15 here and 1e-14 there. Toshio et
al. 2510.25222 (tmp/papers) run belief matching as the accurate decoder
invoked on demand.
"""

from typing import Optional

import ldpc
import numpy
import pymatching
import scipy.sparse
import scipy.special

import decsim.decoders.decoder as decoder_module
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models

POSTERIOR_FLOOR = 1e-15
POSTERIOR_CEILING = 1.0 - POSTERIOR_FLOOR


class BeliefMatchingDecoder(decoder_module.WindowDecoderBase):
    """Decode one hyperedge-bearing window with belief matching.

    Measured mode times the BP-plus-matching call only, never the
    one-time window-model construction (validation, BP build), the same
    contract as PyMatchingDecoder's warm-up.
    """

    fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        max_iterations: int = 30,
        belief_propagation_method: str = "product_sum",
    ):
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        self.max_iterations = max_iterations
        self.belief_propagation_method = belief_propagation_method

    def compile(self, faults, model) -> tuple:
        """The model's BP decoder and sparse hyperedge-to-edge map, warm.

        The first decode on a model builds ldpc's message-passing state;
        one decode of the empty syndrome takes that outside the timed
        call.
        """
        physical = model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        projection = model.physical_to_graphlike_detector_projection
        if projection is None:
            raise ValueError(
                "belief matching needs the physical-to-graphlike link"
            )
        fault_identity.validate_belief_matching_matrices(
            faults.check,
            faults.observables,
            physical.check,
            physical.priors,
            projection,
            location="belief-matching window model",
        )
        physical_check = scipy.sparse.csr_matrix(physical.check)
        error_channel = list(physical.priors)
        belief_propagation = ldpc.BpDecoder(
            physical_check,
            error_channel=error_channel,
            max_iter=self.max_iterations,
            bp_method=self.belief_propagation_method,
            input_vector_type="syndrome",
        )
        projection = projection.astype(numpy.float64)
        edge_from_hyperedge = scipy.sparse.csr_matrix(projection)
        backend = (belief_propagation, edge_from_hyperedge)
        detector_count = physical.check.shape[0]
        empty_syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
        self.decode_window(backend, model, faults, empty_syndrome)
        return backend

    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """BP on the hyperedges, then a matching with posterior weights.

        PyMatching raises on odd parity in a boundaryless component (see
        mwpm); that case is an empty correction marked invalid.
        """
        del model
        belief_propagation, edge_from_hyperedge = backend
        edge_posteriors = _edge_posteriors(
            belief_propagation, edge_from_hyperedge, syndrome
        )
        edge_weights = -numpy.log(edge_posteriors)
        check = faults.check.copy()
        matching = pymatching.Matching.from_check_matrix(
            check, weights=edge_weights
        )
        try:
            selected = matching.decode(syndrome)
        except ValueError as error:
            if "perfect matching" not in str(error):
                raise
            fault_count = faults.check.shape[1]
            empty = numpy.zeros(fault_count, dtype=numpy.uint8)
            invalid = (
                window_decode_results.BackendDecodeStatus.INVALID_CORRECTION
            )
            return empty, invalid
        return numpy.asarray(selected, dtype=numpy.uint8), None


def _edge_posteriors(belief_propagation, edge_from_hyperedge, syndrome):
    """Run BP and map hyperedge posteriors onto matching-edge posteriors."""
    belief_propagation.decode(syndrome)
    log_likelihood_ratios = numpy.asarray(
        belief_propagation.log_prob_ratios, dtype=float
    )
    log_likelihood_ratios = numpy.nan_to_num(log_likelihood_ratios, nan=0.0)
    hyperedge_posteriors = scipy.special.expit(-log_likelihood_ratios)
    edge_posteriors = edge_from_hyperedge @ hyperedge_posteriors
    return numpy.clip(edge_posteriors, POSTERIOR_FLOOR, POSTERIOR_CEILING)
