"""Inner PyMatching decoder used by decode_windowed tests and references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.window_error_models import WindowErrorModel


def matching_window_decoder():
    """Build a cached PyMatching callable for WindowErrorModel inputs."""
    import numpy as np
    import pymatching
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        matching = cache.get(id(model))
        if matching is None:
            weights = np.log((1 - model.priors) / model.priors)
            matching = pymatching.Matching.from_check_matrix(model.check, weights=weights)
            cache[id(model)] = matching
        return matching.decode(syndrome)

    return decode
