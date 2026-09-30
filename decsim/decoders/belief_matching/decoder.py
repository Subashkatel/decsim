"""The belief-matching adapter: the strong tier's decoder on one window.

Belief propagation on the physical hyperedges, then matching on the
graphlike edges with posterior weights: Higgott et al., beliefmatching's
BeliefMatching.decode (.pydeps/beliefmatching/belief_matching.py:344-358).
ldpc's BpDecoder gives the hyperedge posteriors, one per distinct
detector set as beliefmatching's matrices hold them (lines 104-136), the
window's projection maps them onto matching edges, PyMatching matches
with -log(posterior) weights. The posterior clamp is 1e-15 here and
1e-14 there. Toshio et al. 2510.25222 run belief matching as the
accurate decoder invoked on demand.
"""

from typing import Optional

import ldpc
import numpy
import pymatching
import scipy.sparse
import scipy.special

import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.stim_fault_catalog as stim_fault_catalog
import decsim.records.decoding as decoding_records

POSTERIOR_FLOOR = 1e-15
POSTERIOR_CEILING = 1.0 - POSTERIOR_FLOOR


class BeliefMatchingDecoder(decoder_module.WindowDecoderBase):
    """Decode one hyperedge-bearing window with belief matching.

    Measured mode times the BP-plus-matching call only, never the
    one-time BP build, the same contract as PyMatchingDecoder's warm-up.
    """

    fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    missing_evidence_reasons = {
        decoding_records.DecoderEvidence.FORCED_CLASS_WEIGHT: (
            "belief matching matches on the posterior graph belief "
            "propagation reweights, so its forced pair must be built on "
            "that graph and not on the window's raw priors (Gidney et "
            "al. arXiv:2312.04522 lines 843-846); that pair is not "
            "built yet, and a gap from another graph is not this "
            "decoder's confidence"
        )
    }

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
        columns, error_channel = _distinct_hyperedges(physical)
        physical_check = scipy.sparse.csr_matrix(physical.check)
        hyperedge_check = physical_check[:, columns]
        belief_propagation = ldpc.BpDecoder(
            hyperedge_check,
            error_channel=error_channel,
            max_iter=self.max_iterations,
            bp_method=self.belief_propagation_method,
            input_vector_type="syndrome",
        )
        hyperedge_projection = projection[:, columns]
        hyperedge_projection = hyperedge_projection.astype(numpy.float64)
        edge_from_hyperedge = scipy.sparse.csr_matrix(hyperedge_projection)
        backend = (belief_propagation, edge_from_hyperedge)
        detector_count = physical.check.shape[0]
        empty_syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
        self.decode_window(backend, model, faults, empty_syndrome)
        return backend

    def decode_window(self, backend, model, faults, syndrome):
        """BP on the hyperedges, then a matching with posterior weights.

        PyMatching raises on odd parity in a boundaryless component (see
        the PyMatching adapter); that case is an empty correction marked
        invalid, with no correction's reason.
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
            return backend_outcome.no_correction_decode(
                decoding_records.BackendDecodeStatus.INVALID_CORRECTION,
                decoding_records.BackendFailureReason.NO_PERFECT_MATCHING,
                fault_count,
            )
        correction = numpy.asarray(selected, dtype=numpy.uint8)
        return decoding_records.WindowDecode(correction)


def _distinct_hyperedges(physical) -> tuple:
    """One column per distinct detector set, and its combined prior.

    stim's decomposed model lists a hyperedge once per decomposition and
    the linked catalog keeps each as a column. beliefmatching gives BP
    one mechanism per detector set, the instructions' priors combined as
    independent faults, its edges the first decomposition's
    (belief_matching.py lines 104-112 and 135-136), so the first column
    of each set stands for it here.
    """
    check = physical.check
    position_by_detectors = {}
    columns = []
    priors = []
    for column in range(check.shape[1]):
        detectors = basis_split.column_rows(check, column)
        prior = float(physical.priors[column])
        position = position_by_detectors.get(detectors)
        if position is None:
            position_by_detectors[detectors] = len(columns)
            columns.append(column)
            priors.append(prior)
            continue
        earlier = priors[position]
        priors[position] = stim_fault_catalog.merge_probability(earlier, prior)
    return columns, priors


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
