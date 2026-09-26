"""Prior to matching-weight conversion, shared by the MWPM decoders."""

import numpy


def finite_priors(priors):
    """Priors moved strictly inside (0, 1) so their log-odds are finite.

    Malformed priors (NaN, inf, or outside [0, 1]) are model-construction
    bugs and raise. Exact 0 and 1 priors are legitimate degenerate inputs
    (a deterministic injected fault), but their infinite raw weights
    cannot be passed to PyMatching. The referents disagree on the bound:
    beliefmatching 0.2.0 clips its edge probabilities into
    [1e-14, 1 - 1e-14] before it takes the log (belief_matching.py,
    decode), and PyMatching 2.4.0's add_edge documents the weight as
    log((1-p)/p) and names no bound. The 1e-12 here is decsim's own.
    """
    priors = numpy.asarray(priors, dtype=float)
    if not _are_probabilities(priors):
        lowest = priors.min()
        highest = priors.max()
        raise ValueError(
            "window error model priors must be finite probabilities in "
            f"[0, 1]; got range [{lowest}, {highest}]"
        )
    interior = priors.copy()
    interior[priors == 0] = 1e-12
    interior[priors == 1] = 1 - 1e-12
    return interior


def matching_weights(priors):
    """Finite log-odds log((1 - p) / p) of the priors, PyMatching's weight."""
    interior = finite_priors(priors)
    log_survival = numpy.log1p(-interior)
    log_prior = numpy.log(interior)
    return log_survival - log_prior


def _are_probabilities(priors) -> bool:
    is_finite = numpy.isfinite(priors)
    if not is_finite.all():
        return False
    at_least_zero = priors >= 0
    at_most_one = priors <= 1
    inside = at_least_zero & at_most_one
    all_inside = inside.all()
    return bool(all_inside)
