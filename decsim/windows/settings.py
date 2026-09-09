"""The window settings: the scheme and its commit and buffer sizes."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.tables as tables
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.naive_online as naive_online_scheme
import decsim.windows.schemes.parallel as parallel_scheme
import decsim.windows.schemes.sandwich as sandwich_scheme
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.window_interactions as window_interactions

# windows.kind names one of these rows: how the stream is cut into
# windows.
WINDOWING_SCHEMES = {
    "sliding": sliding_scheme.SlidingWindowScheme,
    "parallel": parallel_scheme.ParallelWindowScheme,
    "sandwich": sandwich_scheme.TanSandwichScheme,
    "naive_online": naive_online_scheme.NaiveOnlineScheme,
}
# windows.boundaries names one of these rows: when a committed window
# ships its boundary to the windows after it.
BOUNDARY_POLICIES = {
    "eager": boundary_policies.Eager,
    "held": boundary_policies.Held,
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
    hand-off between windows is written on decoder_to_decoder.
    terminal_policy is flush or lookahead (TERMINAL_POLICIES,
    records/windows.py), how a finite stream drains its last buffered
    window; null, the default, leaves the row on flush, and on lookahead
    when the escalation may escalate, since a strong recovery needs a
    last window that reads past its own commit. boundaries names a row of
    BOUNDARY_POLICIES (above): when a committed window ships its boundary
    to the windows after it; null, the default, is the row the
    escalation policy declares, or, when that policy may escalate, the
    row its strong window shape declares (default_boundary_policy on
    escalation/policies.py and escalation/strong_window_shapes.py):
    held when the escalation may escalate and the strong window does not
    absorb the weak windows it covers, and eager otherwise. A
    Python-built scheme, boundary policy or window interaction is used as
    it is; the root's defaults are the sliding scheme, Eager shipping and
    the default interaction.
    """

    kind: str = "sliding"
    commit_rounds: Optional[int] = None
    buffer_rounds: Optional[int] = None
    boundary_payload: str = "dense_seam_mask"
    terminal_policy: Optional[str] = None
    boundaries: Optional[str] = None
    scheme: Optional[ports.WindowingScheme] = None
    boundary_policy: Optional[ports.BoundaryPolicy] = None
    window_interaction: Optional[window_interactions.WindowInteraction] = None

    @classmethod
    def from_yaml(cls, section: Mapping) -> "WindowSettings":
        """The `windows` section: a kind of the table, two sizes, the wire."""
        boundary_payload = section.get("boundary_payload", "dense_seam_mask")
        terminal_policy = section.get("terminal_policy")
        _check_terminal_policy(terminal_policy)
        boundaries = section.get("boundaries")
        if boundaries is not None:
            tables.row(BOUNDARY_POLICIES, "windows.boundaries", boundaries)
        return cls(
            kind=section["kind"],
            commit_rounds=section["commit_rounds"],
            buffer_rounds=section["buffer_rounds"],
            boundary_payload=boundary_payload,
            terminal_policy=terminal_policy,
            boundaries=boundaries,
        )


def _check_terminal_policy(terminal_policy) -> None:
    """windows.terminal_policy is one of the two words, or absent."""
    if terminal_policy is None:
        return
    if terminal_policy in window_records.TERMINAL_POLICIES:
        return
    listed = list(window_records.TERMINAL_POLICIES)
    raise ValueError(
        f"windows.terminal_policy is one of {listed}, got {terminal_policy!r}"
    )
