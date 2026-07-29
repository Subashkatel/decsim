"""Shared prior -> matching-weight conversion for the MWPM decoders."""

from __future__ import annotations


def matching_weights(priors):
    """Return finite log-odds while preserving strict-interior priors.

    Malformed priors (NaN/inf or outside [0, 1]) are model-construction bugs
    and raise. Exact 0/1 priors are legitimate degenerate inputs (e.g. a
    deterministic injected fault), but their infinite raw weights cannot be
    passed to PyMatching.
    """
    import numpy as np

    priors = np.asarray(priors, dtype=float)
    if not np.isfinite(priors).all() or (priors < 0).any() or (priors > 1).any():
        raise ValueError(
            "window error model priors must be finite probabilities in "
            f"[0, 1]; got range [{priors.min()}, {priors.max()}]")
    finite_priors = priors.copy()
    finite_priors[priors == 0] = 1e-12
    finite_priors[priors == 1] = 1 - 1e-12
    return np.log((1 - finite_priors) / finite_priors)
