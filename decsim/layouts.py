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
 
    def spatial_nodes_for(self, op: Operation) -> int:
        num_patches = len(op.patches) if op.patches else len(op.qubits)
        return self.code.spatial_nodes(num_patches)
 
    def codes(self) -> list:
        return [self.code]


class ZonedLayout:
    """Heterogeneous layout built from patch-to-code assignments."""

    def __init__(self, assignment: dict, default: CodeModel):
        self.assignment = dict(assignment)
        self.default = default
 
    @property
    def name(self) -> str:
        names = sorted({c.name for c in self.codes()})
        return "zoned[" + " + ".join(names) + "]"
 
    @property
    def distance(self) -> int:
        return self.default.distance
 
    def code_for_patch(self, patch_id: Any) -> CodeModel:
        return self.assignment.get(patch_id, self.default)
 
    def _operation_patches(self, op: Operation):
        """Return explicit patches, or fall back to operation qubits."""
        return op.patches if op.patches else op.qubits
 
    def code_for_op(self, op: Operation) -> CodeModel:
        codes = [self.code_for_patch(patch) for patch in self._operation_patches(op)]
        if not codes:
            return self.default
        return max(codes, key=lambda code: code.distance)
 
    def spatial_nodes_for(self, op: Operation) -> int:
        patch_count_by_code: dict = {}
        for patch in self._operation_patches(op):
            code = self.code_for_patch(patch)
            patch_count_by_code[code] = patch_count_by_code.get(code, 0) + 1
        return sum(code.spatial_nodes(count)
                   for code, count in patch_count_by_code.items())
 
    def codes(self) -> list:
        seen_ids = set()
        unique_codes = []
        for code in list(self.assignment.values()) + [self.default]:
            if id(code) in seen_ids:
                continue
            seen_ids.add(id(code))
            unique_codes.append(code)
        return unique_codes
