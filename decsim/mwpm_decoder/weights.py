"""Shared prior -> matching-weight conversion for the MWPM decoders."""

from __future__ import annotations


def matching_weights(priors):
    """log((1-p)/p) with validated, boundary-clipped priors.

    Malformed priors (NaN/inf or outside [0, 1]) are model-construction bugs
    and raise. Exact 0/1 priors are legitimate degenerate inputs (e.g. a
    deterministic injected fault) and are clipped to [1e-12, 1-1e-12]: the
    raw weights would be +-inf, and -inf hard-crashes pymatching.
    """
    import numpy as np

    priors = np.asarray(priors, dtype=float)
    if not np.isfinite(priors).all() or (priors < 0).any() or (priors > 1).any():
        raise ValueError(
            "window error model priors must be finite probabilities in "
            f"[0, 1]; got range [{priors.min()}, {priors.max()}]")
    clipped = np.clip(priors, 1e-12, 1 - 1e-12)
    return np.log((1 - clipped) / clipped)
