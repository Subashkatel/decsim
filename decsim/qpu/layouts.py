"""The layout: which code every patch and every operation runs on.

A uniform layout gives every patch the same code card and claims one
qubit-exclusivity resource per operation. That is the only layout today;
the seam it fills is LayoutModel in decsim/protocols.py.
"""

from typing import Any

import decsim.message as message
import decsim.protocols as protocols

# A patch identity is opaque to the layout; Any stands for it below.


class UniformLayout:
    """Every patch uses the same code card."""

    def __init__(self, code: protocols.CodeModel):
        self.code = code

    def code_for_patch(self, patch_id: Any) -> protocols.CodeModel:
        """The one code, whatever the patch."""
        del patch_id
        return self.code

    def code_for_op(
        self, operation: message.OperationPlanningView
    ) -> protocols.CodeModel:
        """The one code, whatever the operation."""
        del operation
        return self.code

    def spatial_nodes_for(
        self,
        operation: message.OperationPlanningView,
        *,
        base_spatial_node_count: int,
    ) -> int:
        """The base node count, unchanged."""
        del operation
        return base_spatial_node_count

    def patch_spatial_nodes_for(
        self, patch_identity: Any, *, base_spatial_node_count: int
    ) -> int:
        """The base node count, unchanged."""
        del patch_identity
        return base_spatial_node_count

    def resources_for(
        self, operation: message.OperationPlanningView
    ) -> list[message.ResourceClaim]:
        """Return one qubit exclusivity claim."""
        qubits = frozenset(operation.qubits)
        return [message.ResourceClaim("qubits", qubits)]

    def codes(self) -> list[protocols.CodeModel]:
        """The one code, as the list the seam asks for."""
        return [self.code]
