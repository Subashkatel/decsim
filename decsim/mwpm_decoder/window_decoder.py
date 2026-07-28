"""Inner PyMatching decoder used by decode_windowed tests and references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detector_error_model import WindowErrorModel


def matching_window_decoder():
    """Build a cached PyMatching callable for WindowErrorModel inputs.

    The cache is keyed by object identity and evicted when a model is
    garbage-collected: id() values are recycled by CPython, so without the
    eviction a fresh model can alias a dead one's key and receive a stale
    matching (wrong graph -> shape errors, or silently wrong corrections).
    """
    import weakref

    import pymatching

    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        matching = cache.get(id(model))
        if matching is None:
            from ..detector_error_model import validate_graphlike_matrices
            from .weights import matching_weights

            validate_graphlike_matrices(
                model.check,
                model.obs,
                location="PyMatching window model",
            )
            matching = pymatching.Matching.from_check_matrix(
                model.check, weights=matching_weights(model.priors))
            cache[id(model)] = matching
            weakref.finalize(model, cache.pop, id(model), None)
        return matching.decode(syndrome)

    return decode
