"""The layout: which code every patch and every operation runs on.

The uniform layout, the only one, gives every patch the same card and
claims one qubit-exclusivity resource per operation.
"""

import dataclasses
from typing import Any

import decsim.ports as ports
import decsim.records.program as program_records

# A patch identity is opaque to the layout; Any stands for it below.


class UniformLayout:
    """Every patch uses the same code card."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The uniform layout has no keys: every patch takes the run's card."""

        def build(self, code: ports.CodeModel) -> "UniformLayout":
            """A fresh layout over the run's card."""
            return UniformLayout(code)

    def __init__(self, code: ports.CodeModel):
        self.code = code

    def code_for_patch(
        self,
        patch_id: Any,  # an opaque identity
    ) -> ports.CodeModel:
        """The one code, whatever the patch."""
        del patch_id
        return self.code

    def code_for_operation(
        self, operation: program_records.OperationPlanningView
    ) -> ports.CodeModel:
        """The one code, whatever the operation."""
        del operation
        return self.code

    def resources_for(
        self, operation: program_records.OperationPlanningView
    ) -> list[program_records.ResourceClaim]:
        """Return one qubit exclusivity claim."""
        qubits = frozenset(operation.qubits)
        return [program_records.ResourceClaim("qubits", qubits)]

    def codes(self) -> list[ports.CodeModel]:
        """The one code, as the list the seam asks for."""
        return [self.code]
