"""The window settings: the scheme and its commit and buffer sizes."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.ports as ports
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager


@dataclasses.dataclass(frozen=True)
class WindowSettings:
    """The yaml's `windows` section.

    Table rows (decsim/machine.py): sliding, parallel, sandwich,
    naive_online. commit_rounds and buffer_rounds size every window; None
    is the code distance. A Python-built scheme, boundary policy or
    window interaction is used as it is; the root's defaults are the
    sliding scheme, Eager shipping and the default interaction.
    """

    kind: str = "sliding"
    commit_rounds: Optional[int] = None
    buffer_rounds: Optional[int] = None
    scheme: Optional[ports.WindowingScheme] = None
    boundary_policy: Optional[window_manager.BoundaryPolicy] = None
    window_interaction: Optional[window_interactions.WindowInteraction] = None

    @classmethod
    def from_yaml(cls, section: Mapping) -> "WindowSettings":
        """The `windows` section: a kind of the table and two sizes."""
        return cls(
            kind=section["kind"],
            commit_rounds=section["commit_rounds"],
            buffer_rounds=section["buffer_rounds"],
        )
