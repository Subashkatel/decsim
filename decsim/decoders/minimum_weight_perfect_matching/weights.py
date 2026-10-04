"""Prior to matching-weight conversion, shared by the MWPM decoders."""

import numpy


def finite_priors(priors: numpy.ndarray) -> numpy.ndarray:
    """Priors moved strictly inside (0, 1) so their log-odds are finite.

    NaN, infinite or out-of-range priors raise. Exact 0 and 1 are
    legitimate (a deterministic injected fault) but give infinite
    weights PyMatching cannot take. beliefmatching 0.2.0 clips into
    [1e-14, 1 - 1e-14] before the log (belief_matching.py, decode), and
    PyMatching 2.4.0's add_edge names no bound; the 1e-12 here is
    decsim's own.
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


def matching_weights(priors: numpy.ndarray) -> numpy.ndarray:
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
