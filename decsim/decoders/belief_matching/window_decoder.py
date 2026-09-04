"""Belief matching over one window.

Belief propagation on the physical hyperedges, then matching on the
graphlike edges with posterior weights: Higgott et al., beliefmatching's
BeliefMatching.decode (.pydeps/beliefmatching/belief_matching.py:344-358).
ldpc's BpDecoder gives the hyperedge posteriors, the window's projection
maps them onto matching edges, PyMatching matches with -log(posterior)
weights. The posterior clamp is 1e-15 here and 1e-14 there.
"""

import weakref

import ldpc
import numpy
import pymatching
import scipy.sparse
import scipy.special

import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models

POSTERIOR_FLOOR = 1e-15
POSTERIOR_CEILING = 1.0 - POSTERIOR_FLOOR


def belief_matching_window_decoder(
    max_iterations: int = 30, belief_propagation_method: str = "product_sum"
):
    """A belief-matching callable over WindowErrorModel inputs.

    One BP decoder per live window model, evicted with the model.
    """
    cache: dict = {}

    def decode(model, syndrome):
        _require_belief_matching_model(model)
        graphlike = model.require_faults(
            fault_models.FaultRepresentation.GRAPHLIKE
        )
        physical = model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        syndrome = numpy.asarray(syndrome, dtype=numpy.uint8)
        if physical.check.shape[1] == 0:  # empty window
            edge_count = graphlike.check.shape[1]
            return numpy.zeros(edge_count, dtype=numpy.uint8)
        belief_propagation, edge_from_hyperedge = _cache_entry(
            model, cache, max_iterations, belief_propagation_method
        )
        edge_posteriors = _edge_posteriors(
            belief_propagation, edge_from_hyperedge, syndrome
        )
        edge_weights = -numpy.log(edge_posteriors)
        check = graphlike.check.copy()
        matching = pymatching.Matching.from_check_matrix(
            check, weights=edge_weights
        )
        selected = matching.decode(syndrome)
        return numpy.asarray(selected, dtype=numpy.uint8)

    return decode


def _require_belief_matching_model(model) -> None:
    """Fail if the window model lacks hyperedge data for belief matching."""
    model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    if model.physical_to_graphlike_detector_projection is None:
        raise ValueError(
            "belief_matching_window_decoder needs a physical-to-graphlike link"
        )


def _cache_entry(
    model, cache: dict, max_iterations: int, belief_propagation_method: str
) -> tuple:
    """The model's BP decoder and sparse hyperedge-to-edge map, built once."""
    model_identity = id(model)
    entry = cache.get(model_identity)
    if entry is not None:
        return entry
    graphlike = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    fault_identity.validate_belief_matching_matrices(
        graphlike.check,
        graphlike.observables,
        physical.check,
        physical.priors,
        model.physical_to_graphlike_detector_projection,
        location="belief-matching window model",
    )
    physical_check = scipy.sparse.csr_matrix(physical.check)
    error_channel = list(physical.priors)
    belief_propagation = ldpc.BpDecoder(
        physical_check,
        error_channel=error_channel,
        max_iter=max_iterations,
        bp_method=belief_propagation_method,
        input_vector_type="syndrome",
    )
    projection = model.physical_to_graphlike_detector_projection
    projection = projection.astype(numpy.float64)
    edge_from_hyperedge = scipy.sparse.csr_matrix(projection)
    entry = (belief_propagation, edge_from_hyperedge)
    cache[model_identity] = entry
    # id() values are recycled by CPython; evict on GC so a fresh model
    # cannot alias a dead one's key and receive a stale decoder
    weakref.finalize(model, cache.pop, model_identity, None)
    return entry


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
