from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.window_error_models import WindowErrorModel


def matching_window_decoder():
    """A PyMatching inner decoder for decode_windowed, caching one Matching per
    WindowErrorModel (the matrices are shot-independent). Boundary edges arise from
    single-detector columns; weights are the standard log((1-p)/p)."""
    import numpy as np
    import pymatching
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        m = cache.get(id(model))
        if m is None:
            weights = np.log((1 - model.priors) / model.priors)
            m = pymatching.Matching.from_check_matrix(model.check, weights=weights)
            cache[id(model)] = m
        return m.decode(syndrome)

    return decode
