"""Inner PyMatching decoder over one WindowErrorModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...detector_error_model.fault_model_contracts import WindowErrorModel


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
        from ...detector_error_model.fault_model_contracts import FaultRepresentation

        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        matching = cache.get(id(faults))
        if matching is None:
            from ...detector_error_model.fault_identity_validation import (
                validate_graphlike_matrices,
            )
            from .weights import matching_weights

            validate_graphlike_matrices(
                faults.check,
                faults.observables,
                location="PyMatching window model",
            )
            matching = pymatching.Matching.from_check_matrix(
                faults.check.copy(), weights=matching_weights(faults.priors))
            cache[id(faults)] = matching
            weakref.finalize(faults, cache.pop, id(faults), None)
        return matching.decode(syndrome)

    return decode
