"""Shared prior -> matching-weight conversion for the MWPM decoders."""

from __future__ import annotations


def finite_priors(priors):
    """Priors moved strictly inside (0, 1) so their log-odds are finite.

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
    interior = priors.copy()
    interior[priors == 0] = 1e-12
    interior[priors == 1] = 1 - 1e-12
    return interior


def matching_weights(priors):
    """Finite log-odds log((1 - p) / p) of the priors, PyMatching's weight."""
    import numpy as np

    interior = finite_priors(priors)
    return np.log1p(-interior) - np.log(interior)
