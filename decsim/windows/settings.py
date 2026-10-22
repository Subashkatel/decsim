"""The window scheme, its sizes, and its decision cost on a named clock."""

import dataclasses
from typing import Optional, Protocol

import decsim.config as config
import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.release_rules as release_rules
import decsim.windows.schemes.sliding as sliding_scheme


class SchemeSettings(Protocol):
    """A windowing scheme row's settings record (windows/schemes/).

    It holds the window sizes, None being the code distance, and builds
    the scheme with the terminal policy that drains a finite stream.
    """

    name: str
    commit_rounds: Optional[int]
    buffer_rounds: Optional[int]

    def build(self, terminal_policy: str) -> ports.WindowingScheme:
        """A fresh scheme of these sizes."""


class ReleaseRuleSettings(Protocol):
    """A release rule row's settings record (windows/release_rules.py)."""

    name: str
    # whether a blocker waits for operations before it, not itself only
    awaits_ancestors: bool

    def build(self, operations: tuple) -> release_rules.ReleaseRule:
        """The rule over the workload's operations."""


class BoundaryPayloadSettings(Protocol):
    """A payload row's settings record (windows/boundary_payloads.py)."""

    def build(self) -> ports.BoundaryPayload:
        """A fresh payload."""


@dataclasses.dataclass(frozen=True)
class WindowSettings:
    """The window side's settings.

    scheme is a windowing scheme row's Settings record (windows/schemes/),
    which holds the window sizes, commit_rounds and buffer_rounds, None
    being the code distance. terminal_policy is flush or lookahead
    (TERMINAL_POLICIES, records/windows.py), how a finite stream drains
    its last buffered window; the rows that lay their own tail read none.
    boundary_policy is a boundary row's Settings record
    (windows/boundary_policies.py), when a committed window ships its
    boundary to the windows after it. A switching run's policy refuses a
    tail or a boundary row its strong window cannot serve.
    boundary_payload is a payload row's Settings record
    (windows/boundary_payloads.py), how the hand-off between windows is
    written on the hop between their decoders. clock and decision_cycles price
    issuing one decode request; clock None is the machine's clock.
    recorded_confidence is a confidence row's Settings record
    (decsim/confidence/) that a run with no switching computes on every
    window and records, deciding nothing, so a decoder run alone gives
    each window's gap beside its answer; its work is charged as on a
    switching run. None computes none; a switching run records its own
    edge's signal. release_rule is a release rule row's Settings record
    (windows/release_rules.py), the decodes a blocking operation's
    outcome waits for; None is release_rule_for's choice by the run.
    """

    clock: Optional[config.Clock] = None
    decision_cycles: int = 0
    scheme: SchemeSettings = sliding_scheme.SlidingWindowScheme.Settings()
    terminal_policy: str = "flush"
    boundary_policy: ports.BoundaryPolicySettings = (
        boundary_policies.Eager.Settings()
    )
    boundary_payload: BoundaryPayloadSettings = (
        boundary_payloads.DenseSeamMask.Settings()
    )
    recorded_confidence: Optional[ports.ConfidenceSettings] = None
    release_rule: Optional[ReleaseRuleSettings] = None

    def __post_init__(self) -> None:
        config.check_cycles("windows.decision_cycles", self.decision_cycles)
        _check_terminal_policy(self.terminal_policy)


def switching_windows(
    windows: WindowSettings, strong_window: ports.StrongWindowBoundaries
) -> WindowSettings:
    """The windows with the tail and boundary row a switching run takes.

    A strong recovery reads past the last window's commit, so the tail is
    lookahead. The boundary row is the record the strong window gives
    (ports.StrongWindowBoundaries).
    """
    boundary_policy = strong_window.boundary_policy
    return dataclasses.replace(
        windows, terminal_policy="lookahead", boundary_policy=boundary_policy
    )


def release_rule_for(
    windows: WindowSettings, patch_lives: tuple
) -> ReleaseRuleSettings:
    """The release rule row the run takes: the one set, else by its shape.

    A lattice-surgery run waits for its decoded ancestors, as SWIPER and
    Triage do, since a merge's outcome is read in a Pauli frame that
    earlier merges with other patches set; any other run waits for its
    own streams, which is then the whole of its outcome's history.
    """
    if windows.release_rule is not None:
        return windows.release_rule
    if patch_lives:
        return release_rules.DecodedAncestors.Settings()
    return release_rules.OwnStreams.Settings()


def _check_terminal_policy(terminal_policy) -> None:
    """windows.terminal_policy is one of the two words."""
    if terminal_policy in window_records.TERMINAL_POLICIES:
        return
    listed = list(window_records.TERMINAL_POLICIES)
    raise ValueError(
        f"windows.terminal_policy is one of {listed}, got {terminal_policy!r}"
    )
