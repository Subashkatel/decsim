"""The window scheme, its sizes, and its decision cost on a named clock."""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Protocol

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
# The keys every row of the windows section shares; any other key is a
# field of the scheme row's own Settings record.
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


class SchemeSettings(Protocol):
    """A windowing scheme row's settings record (WINDOWING_SCHEMES, above).

    It holds the window sizes, None being the code distance, and builds
    the scheme with the terminal policy that drains a finite stream.
    """

    name: str
    commit_rounds: Optional[int]
    buffer_rounds: Optional[int]

    def build(self, terminal_policy: str) -> ports.WindowingScheme:
        """A fresh scheme of these sizes."""


class BoundaryPolicySettings(Protocol):
    """A boundary row's settings record (BOUNDARY_POLICIES, above)."""

    def build(self) -> ports.BoundaryPolicy:
        """A fresh policy."""


class BoundaryPayloadSettings(Protocol):
    """A payload row's settings record (BOUNDARY_PAYLOADS, above)."""

    def build(self) -> ports.BoundaryPayload:
        """A fresh payload."""


@dataclasses.dataclass(frozen=True)
class WindowSettings:
    """How the rounds are cut into decode windows, and what a window ships.

    scheme is a windowing scheme row's Settings record (WINDOWING_SCHEMES,
    above: sliding, parallel, sandwich, naive_online), which holds the
    window sizes, commit_rounds and buffer_rounds, None being the code
    distance. terminal_policy is flush or lookahead (TERMINAL_POLICIES,
    records/windows.py), how a finite stream drains its last buffered
    window; the rows that lay their own tail read none.
    boundary_policy is a boundary row's Settings record
    (BOUNDARY_POLICIES, above: eager or held), when a committed window
    ships its boundary to the windows after it. A switching run's policy
    refuses a tail or a boundary row its strong window cannot serve.
    boundary_payload is a payload row's Settings record
    (BOUNDARY_PAYLOADS, above: dense_seam_mask or sparse_seam_list), how
    the hand-off between windows is written on decoder_to_decoder. clock and
    decision_cycles price issuing one decode request; clock None is the
    machine's clock.
    """

    clock: Optional[config.Clock] = None
    decision_cycles: int = 0
    scheme: SchemeSettings = sliding_scheme.SlidingWindowScheme.Settings()
    terminal_policy: str = "flush"
    boundary_policy: BoundaryPolicySettings = boundary_policies.Eager.Settings()
    boundary_payload: BoundaryPayloadSettings = (
        boundary_payloads.DenseSeamMask.Settings()
    )

    def __post_init__(self) -> None:
        config.check_cycles("windows.decision_cycles", self.decision_cycles)
        _check_terminal_policy(self.terminal_policy)

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        switching=None,
    ) -> "WindowSettings":
        """The `windows` section: a scheme of the table, its sizes, the wire.

        A section that leaves terminal_policy or boundaries null gets the
        value its run's shape takes: on a switching run the lookahead tail,
        since a strong recovery reads past the last window's commit, and
        the boundary row its strong window shape declares
        (default_boundary_policy on escalation/strong_window_shapes.py);
        otherwise the flush tail and eager boundaries. switching is the
        run's switching slot, None for a run that keeps one decoder.
        """
        _check_required_keys(section)
        kind = section["kind"]
        row = tables.row(WINDOWING_SCHEMES, "windows.kind", kind)
        scheme = _scheme_settings(section, row)
        terminal_policy = _terminal_policy(section, switching)
        payload_word = section.get("boundary_payload", "dense_seam_mask")
        payload_row = tables.row(
            BOUNDARY_PAYLOADS, "windows.boundary_payload", payload_word
        )
        boundary_payload = payload_row.Settings()
        boundary_policy = _boundary_policy(section, switching)
        clock = None
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        decision_cycles = section.get("decision_cycles", 0)
        return cls(
            clock=clock,
            decision_cycles=decision_cycles,
            scheme=scheme,
            terminal_policy=terminal_policy,
            boundary_policy=boundary_policy,
            boundary_payload=boundary_payload,
        )


def _scheme_settings(section: Mapping, row):
    """The scheme row's record, from the section's keys it declares."""
    declared_keys = tables.row_keys(row)
    own_keys = []
    for key in declared_keys:
        if key not in WINDOWS_KEYS:
            own_keys.append(key)
    known_keys = WINDOWS_KEYS + tuple(own_keys)
    tables.refuse_unknown_keys("windows", section, known_keys)
    values = {}
    for key in declared_keys:
        if key in section:
            values[key] = section[key]
    return tables.section_record("windows", row.Settings, values)


def _terminal_policy(section: Mapping, switching) -> str:
    """The section's terminal tail, or the one its run's shape takes."""
    terminal_policy = section.get("terminal_policy")
    if terminal_policy is not None:
        return terminal_policy
    if switching is not None:
        return "lookahead"
    return "flush"


def _boundary_policy(section: Mapping, switching):
    """The boundary row the section names, or the one its run's shape takes.

    A run with no switching never revises a committed window and ships
    every boundary at its commit; a switching run hands the default to
    its strong window shape, whose absorption is what decides.
    """
    boundaries = section.get("boundaries")
    if boundaries is None and switching is None:
        boundaries = "eager"
    if boundaries is None:
        boundaries = switching.strong_window.default_boundary_policy
    row = tables.row(BOUNDARY_POLICIES, "windows.boundaries", boundaries)
    return row.Settings()


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


def _check_terminal_policy(terminal_policy) -> None:
    """windows.terminal_policy is one of the two words."""
    if terminal_policy in window_records.TERMINAL_POLICIES:
        return
    listed = list(window_records.TERMINAL_POLICIES)
    raise ValueError(
        f"windows.terminal_policy is one of {listed}, got {terminal_policy!r}"
    )
