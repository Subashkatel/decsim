"""Clocked readout source: emit each operation on the round clock.

The final round and ``on_body_done`` occur in the same event.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ..seeding import _RandomSeedConsumer
from ..message import (
    Operation,
    RunSeedChild,
    RunSeedPathSegment,
    QPUReadout,
)

if TYPE_CHECKING:
    from ..protocols import CodeModel


def _stream_payload_target(op: Operation, round_index: int) -> tuple:
    """(decode_op_id, global_round) for standalone ops or stream segments."""
    return (op.stream_id if op.stream_id is not None else op.id,
            round_index + (op.stream_offset or 0))


class TimingOnlyDevice:
    """Emit payloads with no syndrome bits for timing-only studies
    (the default RunSpec device)."""

    operation_circuit_scope = "none"

    def begin_operation(self, op: Operation, segment_round_count: int,
                        source_round_count: int) -> None:
        return None

    def round_payloads(self, op: Operation, round_index: int) -> list:
        target, global_round = _stream_payload_target(op, round_index)
        return [QPUReadout(target,
                              op.patches[0] if op.patches else op.qubits[0],
                              global_round)]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list:
        return [QPUReadout(stream_id, patch, global_round)]

    def finalize_stream_round(self, op: Operation,
                              source_round_count: int) -> list:
        raise ValueError("TimingOnlyDevice cannot finalize a physical stream")

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, fault_model_requirement):
        return None

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        return None

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int, *, fault_model_requirement,
                                    fault_exclusion_ranges: tuple,
                                    window_protocol) -> list:
        return []

    def window_model_for_stream(self, stream_id, window):
        return None

    def strong_window_model_for_operation(self, op: Operation, window,
                                          round_count: int,
                                          *, fault_model_requirement,
                                          exclude_faults_touching=None):
        return None

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        fault_model_requirement, fault_exclusion_ranges: tuple,
    ):
        return None


class SyndromeBitDevice(_RandomSeedConsumer):
    """Emit deterministic fake bits to exercise the payload path."""

    operation_circuit_scope = "none"

    def __init__(self, code: CodeModel, seed: Optional[int] = None,
                 max_bits: int = 8,
                 per_patch: bool = False):
        self.code = code
        self.max_bits = max_bits
        self.per_patch = per_patch
        self._initialize_run_seed_state(seed)

    def run_seed_children(self):
        """Expose the code model that determines payload shape."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "code"),),
                self.code,
            ),
        )

    def begin_operation(self, op: Operation, segment_round_count: int,
                        source_round_count: int) -> None:
        return None

    def _bits(self, num_patches: int) -> list:
        """Fake bits for one payload covering this many patches."""
        bit_count = min(self.code.syndrome_bits_per_round(num_patches), self.max_bits)
        self._mark_stochastic_use()
        return [self._rng.randint(0, 1) for _ in range(bit_count)]

    def round_payloads(self, op: Operation, round_index: int) -> list[QPUReadout]:
        """One payload per patch when per_patch=True; else the single aggregated payload."""
        target, global_round = _stream_payload_target(op, round_index)
        if not self.per_patch:
            num_patches = len(op.patches) if op.patches else len(op.qubits)
            bits = self._bits(num_patches)
            return [QPUReadout(
                target,
                op.patches[0] if op.patches else op.qubits[0],
                global_round,
                bits=bits,
                code=self.code.name,
                size_bits=len(bits))]
        patches = op.patches if op.patches else op.qubits
        payloads = []
        for patch in patches:
            bits = self._bits(1)
            payloads.append(QPUReadout(
                target, patch, global_round, bits=bits, code=self.code.name,
                size_bits=len(bits)))
        return payloads

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[QPUReadout]:
        """Emit one fake-bit payload for a feedback-idle stream round."""
        bits = self._bits(1)
        return [QPUReadout(stream_id, patch, global_round, bits=bits,
                                code=self.code.name, size_bits=len(bits))]

    def finalize_stream_round(self, op: Operation,
                              source_round_count: int) -> list[QPUReadout]:
        raise ValueError("SyndromeBitDevice cannot finalize a physical stream")

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, fault_model_requirement):
        """Fake-bit streams have no fixed detector-model length."""
        return None

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        return None

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int, *, fault_model_requirement,
                                    fault_exclusion_ranges: tuple,
                                    window_protocol) -> list:
        """Fake-bit decode jobs carry no detector error model."""
        return []

    def window_model_for_stream(self, stream_id, window):
        """Fake-bit dynamic stream windows carry no detector error model."""
        return None

    def strong_window_model_for_operation(self, op: Operation, window, round_count: int,
                                          *, fault_model_requirement,
                                          exclude_faults_touching=None):
        """Fake-bit strong re-decodes carry no detector error model."""
        return None

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        fault_model_requirement, fault_exclusion_ranges: tuple,
    ):
        """Fake-bit strong re-decodes carry no detector error model."""
        return None
