"""The window scheme, its sizes, and its decision cost on a named clock."""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

import decsim.config as config
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
# The keys every row of the windows section shares; any other key is the
# scheme row's own (its Settings, decsim/tables.py row_settings).
WINDOWS_KEYS = (
    "kind",
    "clock",
    "decision_cycles",
    "commit_rounds",
    "buffer_rounds",
    "boundary_payload",
    "terminal_policy",
    "boundaries",
)
# The keys the section must name; the rest have a default.
_REQUIRED_WINDOWS_KEYS = ("kind", "commit_rounds", "buffer_rounds")


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
    the default interaction. row_settings is the scheme row's own
    Settings, read from the section's keys outside WINDOWS_KEYS, or None
    for a row that declares none.
    """

    clock: Optional[config.Clock] = None
    decision_cycles: int = 0
    kind: str = "sliding"
    commit_rounds: Optional[int] = None
    buffer_rounds: Optional[int] = None
    boundary_payload: str = "dense_seam_mask"
    terminal_policy: Optional[str] = None
    boundaries: Optional[str] = None
    scheme: Optional[ports.WindowingScheme] = None
    boundary_policy: Optional[ports.BoundaryPolicy] = None
    window_interaction: Optional[window_interactions.WindowInteraction] = None
    # the scheme row's own Settings record, opaque to the section
    row_settings: Optional[Any] = None

    def __post_init__(self) -> None:
        config.check_cycles("windows.decision_cycles", self.decision_cycles)
        if self.decision_cycles > 0 and self.clock is None:
            raise ValueError("windows.decision_cycles needs a clock")
        _check_window_rounds("windows.commit_rounds", self.commit_rounds, 1)
        _check_window_rounds("windows.buffer_rounds", self.buffer_rounds, 0)

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        default_clock: Optional[config.Clock] = None,
    ) -> "WindowSettings":
        """The `windows` section: a kind of the table, two sizes, the wire."""
        _check_required_keys(section)
        kind = section["kind"]
        row = tables.row(WINDOWING_SCHEMES, "windows.kind", kind)
        row_settings = tables.row_settings(
            row, "windows", section, WINDOWS_KEYS
        )
        boundary_payload = section.get("boundary_payload", "dense_seam_mask")
        tables.row(
            BOUNDARY_PAYLOADS, "windows.boundary_payload", boundary_payload
        )
        terminal_policy = section.get("terminal_policy")
        _check_terminal_policy(terminal_policy)
        boundaries = section.get("boundaries")
        if boundaries is not None:
            tables.row(BOUNDARY_POLICIES, "windows.boundaries", boundaries)
        clock = default_clock
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        decision_cycles = section.get("decision_cycles", 0)
        return cls(
            clock=clock,
            decision_cycles=decision_cycles,
            kind=kind,
            commit_rounds=section["commit_rounds"],
            buffer_rounds=section["buffer_rounds"],
            boundary_payload=boundary_payload,
            terminal_policy=terminal_policy,
            boundaries=boundaries,
            row_settings=row_settings,
        )


def _check_required_keys(section: Mapping) -> None:
    """The section names the scheme and both window sizes."""
    missing = set(_REQUIRED_WINDOWS_KEYS) - set(section)
    if not missing:
        return
    listed = sorted(missing)
    raise ValueError(
        f"windows needs the keys {listed}; configs/reference.yaml holds "
        "every key with its meaning"
    )


def _check_window_rounds(key: str, rounds, least: int) -> None:
    """A window size is a whole count of rounds, or null for the code's.

    A window is a commit region of ncom rounds and a buffer region of
    nbuf (Skoric et al. 2209.08552 lines 194-197, nW = ncom + nbuf). A
    window that commits no round never moves the stream on, so ncom is
    at least one; a buffer may be empty. YAML reads `true` as a
    boolean, which Python counts as an int, so a flag is refused by name
    as config.check_cycles refuses it.
    """
    if rounds is None:
        return
    if _is_round_count(rounds, least):
        return
    raise ValueError(
        f"{key} is a whole number of rounds, at least {least}, or null "
        f"for the code's own size (got {rounds!r})"
    )


def _is_round_count(rounds, least: int) -> bool:
    if isinstance(rounds, bool):
        return False
    if not isinstance(rounds, int):
        return False
    return rounds >= least


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
