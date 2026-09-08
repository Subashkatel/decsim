"""The window settings: the scheme and its commit and buffer sizes."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.ports as ports
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager
import decsim.windows.windowing_schemes as windowing_schemes

# windows.kind names one of these rows: how the stream is cut into
# windows.
WINDOWING_SCHEMES = {
    "sliding": windowing_schemes.SlidingWindowScheme,
    "parallel": windowing_schemes.ParallelWindowScheme,
    "sandwich": windowing_schemes.TanSandwichScheme,
    "naive_online": windowing_schemes.NaiveOnlineScheme,
}
# windows.boundary_payload names one of these rows: how the hand-off
# between two windows is written on decoder_to_decoder.
BOUNDARY_PAYLOADS = {
    "dense_seam_mask": boundary_payloads.DenseSeamMask,
    "sparse_seam_list": boundary_payloads.SparseSeamList,
}


@dataclasses.dataclass(frozen=True)
class WindowSettings:
    """The yaml's `windows` section.

    Table rows (WINDOWING_SCHEMES, above): sliding, parallel,
    sandwich,
    naive_online. commit_rounds and buffer_rounds size every window; None
    is the code distance. boundary_payload names a row of
    BOUNDARY_PAYLOADS (above): how the
    hand-off between windows is written on decoder_to_decoder. A
    Python-built scheme, boundary policy or window interaction is used as
    it is; the root's defaults are the sliding scheme, Eager shipping and
    the default interaction.
    """

    kind: str = "sliding"
    commit_rounds: Optional[int] = None
    buffer_rounds: Optional[int] = None
    boundary_payload: str = "dense_seam_mask"
    scheme: Optional[ports.WindowingScheme] = None
    boundary_policy: Optional[window_manager.BoundaryPolicy] = None
    window_interaction: Optional[window_interactions.WindowInteraction] = None

    @classmethod
    def from_yaml(cls, section: Mapping) -> "WindowSettings":
        """The `windows` section: a kind of the table, two sizes, the wire."""
        boundary_payload = section.get("boundary_payload", "dense_seam_mask")
        return cls(
            kind=section["kind"],
            commit_rounds=section["commit_rounds"],
            buffer_rounds=section["buffer_rounds"],
            boundary_payload=boundary_payload,
        )
