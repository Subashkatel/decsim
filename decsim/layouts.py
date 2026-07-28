"""QPU layout and code assignment models."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Operation
    from .protocols import CodeModel


class UniformLayout:
    """Every patch uses the same code model."""

    def __init__(self, code: CodeModel):
        self.code = code

    def run_manifest_config(self):
        return {"kind": "uniform"}

    @property
    def name(self) -> str:
        return f"uniform[{self.code.name}]"

    @property
    def distance(self) -> int:
        return self.code.distance

    def code_for_patch(self, patch_id: Any) -> CodeModel:
        return self.code

    def code_for_op(self, op: Operation) -> CodeModel:
        return self.code

    def spatial_nodes_for(
        self,
        operation: Operation,
        *,
        base_spatial_node_count: int,
    ) -> int:
        return base_spatial_node_count

    def patch_spatial_nodes_for(
        self,
        patch_identity,
        *,
        base_spatial_node_count: int,
    ) -> int:
        return base_spatial_node_count

    def resources_for(self, op: Operation) -> list:
        """Typed exclusivity claims (v1.0: qubit claims only, parity-safe)."""
        from .message import ResourceClaim
        return [ResourceClaim("qubits", frozenset(op.qubits))]

    def codes(self) -> list:
        return [self.code]
