from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.window_error_models import WindowErrorModel


def belief_matching_window_decoder(max_iter: int = 30, bp_method: str = "product_sum"):
    """A BELIEF-MATCHING inner decoder for ``decode_windowed`` (Higgott & Gidney,
    arXiv:2203.04948 -- the strong decoder of the decoder-switching paper arXiv:2510.25222).
    Build the window models with ``build_window_error_models(..., belief_matching=True)`` so
    each carries ``h_check`` / ``h_priors`` / ``h2e``.

    Per shot, mirrors the canonical `beliefmatching` package's reweight path: run BP on the
    window's UNDECOMPOSED hyperedge graph -> posterior edge probabilities ps_e = h2e @ ps_h
    -> reweight the matching (edge) graph with -log(ps_e) -> MWPM. Returns the EDGE selection,
    so ``decode_windowed``'s owned-commit + artificial-defect handoff are unchanged.

    Validated against the `beliefmatching` package as the GLOBAL oracle, and shown to track
    global belief-matching through windowing within error bars at d=3,5,7 (the residual is
    buffer-independent, i.e. not a boundary artifact) -- see
    experiments/windowed-belief-matching/. Caches one ``ldpc.BpDecoder`` + the sparse h2e per
    model; the pymatching graph is rebuilt per shot because its weights are the per-shot BP
    posteriors (this per-shot BP cost is exactly why belief-matching is the *slow* strong
    decoder). Faithfulness note: we always reweight-then-match; the package additionally
    short-circuits to the BP correction when BP converges, a ~0.5%-of-shots variant that leaves
    the logical error rate unchanged within error bars."""
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        import numpy as np
        import pymatching
        from ldpc import BpDecoder
        from scipy.sparse import csr_matrix
        from scipy.special import expit
        if model.h_check is None:
            raise ValueError("belief_matching_window_decoder needs models built with "
                             "build_window_error_models(..., belief_matching=True)")
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if model.h_check.shape[1] == 0:                # empty window
            return np.zeros(model.check.shape[1], dtype=np.uint8)
        entry = cache.get(id(model))
        if entry is None:                              # cache BP + the SPARSE edge<-hyper map
            bp = BpDecoder(csr_matrix(model.h_check), error_channel=list(model.h_priors),
                           max_iter=max_iter, bp_method=bp_method,
                           input_vector_type="syndrome")
            entry = (bp, csr_matrix(model.h2e.astype(np.float64)))
            cache[id(model)] = entry
        bp, h2e = entry
        bp.decode(syndrome)
        llrs = np.nan_to_num(np.asarray(bp.log_prob_ratios, dtype=float), nan=0.0)
        ps_h = expit(-llrs)                            # = 1/(1+exp(llrs)), overflow-safe
        ps_e = np.clip(h2e @ ps_h, 1e-15, 1.0 - 1e-15)
        m = pymatching.Matching.from_check_matrix(model.check, weights=-np.log(ps_e))
        return np.asarray(m.decode(syndrome), dtype=np.uint8)

    return decode
