"""The window scheme, its sizes, and its decision cost on a named clock."""

import dataclasses
from typing import Optional, Protocol

import decsim.config as config
import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
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


class BoundaryPayloadSettings(Protocol):
    """A payload row's settings record (windows/boundary_payloads.py)."""

    def build(self) -> ports.BoundaryPayload:
        """A fresh payload."""


@dataclasses.dataclass(frozen=True)
class WindowSettings:
    """How the rounds are cut into decode windows, and what a window ships.

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
    written on decoder_to_decoder. clock and decision_cycles price
    issuing one decode request; clock None is the machine's clock.
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


def _check_terminal_policy(terminal_policy) -> None:
    """windows.terminal_policy is one of the two words."""
    if terminal_policy in window_records.TERMINAL_POLICIES:
        return
    listed = list(window_records.TERMINAL_POLICIES)
    raise ValueError(
        f"windows.terminal_policy is one of {listed}, got {terminal_policy!r}"
    )
