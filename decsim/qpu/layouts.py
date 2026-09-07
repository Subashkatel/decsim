"""The layout: which code every patch and every operation runs on.

A uniform layout gives every patch the same code card and claims one
qubit-exclusivity resource per operation. That is the only layout today;
the seam it fills is LayoutModel below.
"""

from typing import Any, Protocol, runtime_checkable

import decsim.qpu.code_geometry as code_geometry
import decsim.records.program as program_records

# A patch identity is opaque to the layout; Any stands for it below.


@runtime_checkable
class LayoutModel(Protocol):
    """Which code every patch and every operation runs on, and its claims.

    codes, code_for_op and code_for_patch stay stable for a build; every
    selector returns the one declared run code.
    """

    def code_for_op(self, operation: program_records.OperationPlanningView):
        """The code the operation runs on."""

    def code_for_patch(self, patch_id: Any):
        """The code the patch runs on."""

    def codes(self) -> list:
        """Every code the layout declares."""

    def spatial_nodes_for(
        self,
        operation: program_records.OperationPlanningView,
        *,
        base_spatial_node_count: int,
    ) -> int:
        """Decoding-graph nodes per round of the operation."""

    def patch_spatial_nodes_for(
        self, patch_identity: Any, *, base_spatial_node_count: int
    ) -> int:
        """Decoding-graph nodes per round of the patch."""

    def resources_for(
        self, operation: program_records.OperationPlanningView
    ) -> list[program_records.ResourceClaim]:
        """The resources the operation holds while it runs."""


class UniformLayout:
    """Every patch uses the same code card."""

    def __init__(self, code: code_geometry.CodeModel):
        self.code = code

    def code_for_patch(self, patch_id: Any) -> code_geometry.CodeModel:
        """The one code, whatever the patch."""
        del patch_id
        return self.code

    def code_for_op(
        self, operation: program_records.OperationPlanningView
    ) -> code_geometry.CodeModel:
        """The one code, whatever the operation."""
        del operation
        return self.code

    def spatial_nodes_for(
        self,
        operation: program_records.OperationPlanningView,
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
        self, operation: program_records.OperationPlanningView
    ) -> list[program_records.ResourceClaim]:
        """Return one qubit exclusivity claim."""
        qubits = frozenset(operation.qubits)
        return [program_records.ResourceClaim("qubits", qubits)]

    def codes(self) -> list[code_geometry.CodeModel]:
        """The one code, as the list the seam asks for."""
        return [self.code]
