"""Device-side syndrome sources."""


from __future__ import annotations

from typing import TYPE_CHECKING

from .message import Operation, SyndromePayload

if TYPE_CHECKING:
    from .protocols import CodeModel


def _stream_payload_target(op: Operation, round_index: int) -> tuple:
    """Return (decode_op_id, global_round) for standalone ops or stream segments."""
    return (op.stream_id if op.stream_id is not None else op.id,
            round_index + (op.stream_offset or 0))


class TimingOnlyDevice:
    """Emit payloads with no syndrome bits for timing-only studies."""

    def begin_operation(self, op: Operation) -> None:
        """Nothing to set up for this device."""
        return None

    def round_payloads(self, op: Operation, round_index: int) -> list[SyndromePayload]:
        """Emit this operation round as one timing-only payload."""
        target, global_round = _stream_payload_target(op, round_index)
        return [SyndromePayload(target, op.patches[0] if op.patches else op.qubits[0],
                                global_round)]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[SyndromePayload]:
        """Emit one timing-only payload for a feedback-idle stream round."""
        return [SyndromePayload(stream_id, patch, global_round)]

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, belief_matching: bool = False):
        """Timing-only streams have no fixed detector-model length."""
        return None

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        """Timing-only streams can seal at any runtime length."""
        return None

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int,
                                    *, belief_matching: bool = False) -> list:
        """Timing-only decode jobs carry no detector error model."""
        return []

    def window_model_for_stream(self, stream_id, window, *, is_last: bool):
        """Timing-only dynamic stream windows carry no detector error model."""
        return None


class SyndromeBitDevice:
    """Emit deterministic fake bits to exercise the payload path."""

    def __init__(self, code: CodeModel, seed: int = 0, max_bits: int = 8,
                 per_patch: bool = False):
        import random
        self.code = code
        self.max_bits = max_bits
        self.per_patch = per_patch
        self.rng = random.Random(seed)

    def begin_operation(self, op: Operation) -> None:
        """Nothing to set up."""
        return None

    def _bits(self, num_patches: int) -> list:
        """Fake bits for one payload covering this many patches."""
        bit_count = min(self.code.syndrome_bits_per_round(num_patches), self.max_bits)
        return [self.rng.randint(0, 1) for _ in range(bit_count)]

    def round_payloads(self, op: Operation, round_index: int) -> list[SyndromePayload]:
        """One payload per patch when per_patch=True; else the single aggregated payload."""
        target, global_round = _stream_payload_target(op, round_index)
        if not self.per_patch:
            return [SyndromePayload(
                target,
                op.patches[0] if op.patches else op.qubits[0],
                global_round,
                bits=self._bits(len(op.qubits)),
                code=self.code.name)]
        patches = op.patches if op.patches else op.qubits
        return [SyndromePayload(target, p, global_round, bits=self._bits(1),
                                code=self.code.name)
                for p in patches]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[SyndromePayload]:
        """Emit one fake-bit payload for a feedback-idle stream round."""
        return [SyndromePayload(stream_id, patch, global_round,
                                bits=self._bits(1), code=self.code.name)]

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, belief_matching: bool = False):
        """Fake-bit streams have no fixed detector-model length."""
        return None

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        """Fake-bit streams can seal at any runtime length."""
        return None

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int,
                                    *, belief_matching: bool = False) -> list:
        """Fake-bit decode jobs carry no detector error model."""
        return []

    def window_model_for_stream(self, stream_id, window, *, is_last: bool):
        """Fake-bit dynamic stream windows carry no detector error model."""
        return None
