"""The window settings: the scheme and its commit and buffer sizes."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.ports as ports
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.schemes.naive_online as naive_online_scheme
import decsim.windows.schemes.parallel as parallel_scheme
import decsim.windows.schemes.sandwich as sandwich_scheme
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager

# windows.kind names one of these rows: how the stream is cut into
# windows.
WINDOWING_SCHEMES = {
    "sliding": sliding_scheme.SlidingWindowScheme,
    "parallel": parallel_scheme.ParallelWindowScheme,
    "sandwich": sandwich_scheme.TanSandwichScheme,
    "naive_online": naive_online_scheme.NaiveOnlineScheme,
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
