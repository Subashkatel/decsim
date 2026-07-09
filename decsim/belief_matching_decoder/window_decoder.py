"""Inner belief-matching decoder used by decode_windowed tests and references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detector_error_model import WindowErrorModel


def _require_belief_matching_model(model: "WindowErrorModel") -> None:
    """Fail if the window model lacks hyperedge data for belief matching."""
    if model.h_check is None:
        raise ValueError("belief_matching_window_decoder needs models built with "
                         "build_window_error_models(..., belief_matching=True)")


def _cache_entry(model: "WindowErrorModel", cache: dict,
                 max_iter: int, bp_method: str):
    """Return cached BP decoder and sparse hyperedge-to-edge map."""
    import numpy as np
    from ldpc import BpDecoder
    from scipy.sparse import csr_matrix

    entry = cache.get(id(model))
    if entry is not None:
        return entry

    bp = BpDecoder(csr_matrix(model.h_check),
                   error_channel=list(model.h_priors),
                   max_iter=max_iter,
                   bp_method=bp_method,
                   input_vector_type="syndrome")
    edge_from_hyperedge = csr_matrix(model.h2e.astype(np.float64))
    entry = (bp, edge_from_hyperedge)
    cache[id(model)] = entry
    return entry


def _edge_posteriors(bp, edge_from_hyperedge, syndrome):
    """Run BP and map hyperedge posteriors onto matching-edge posteriors."""
    import numpy as np
    from scipy.special import expit

    bp.decode(syndrome)
    log_likelihood_ratios = np.nan_to_num(
        np.asarray(bp.log_prob_ratios, dtype=float),
        nan=0.0)
    hyperedge_posteriors = expit(-log_likelihood_ratios)
    return np.clip(edge_from_hyperedge @ hyperedge_posteriors,
                   1e-15, 1.0 - 1e-15)


def belief_matching_window_decoder(max_iter: int = 30, bp_method: str = "product_sum"):
    """Build a belief-matching callable for WindowErrorModel inputs."""
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        import numpy as np
        import pymatching

        _require_belief_matching_model(model)
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if model.h_check.shape[1] == 0:                # empty window
            return np.zeros(model.check.shape[1], dtype=np.uint8)

        bp, edge_from_hyperedge = _cache_entry(model, cache, max_iter, bp_method)
        edge_posteriors = _edge_posteriors(bp, edge_from_hyperedge, syndrome)
        matching = pymatching.Matching.from_check_matrix(
            model.check,
            weights=-np.log(edge_posteriors))
        return np.asarray(matching.decode(syndrome), dtype=np.uint8)

    return decode
