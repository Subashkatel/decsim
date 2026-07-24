"""Clocked syndrome source (port 2): per-op round emission on the round clock.

Part module: the round-emission half of the QPU seam (the control half is
chip.py's Chip). The Chip drives it via start(op, ...) and receives
on_body_done(op) at the final round — in the SAME event (Contract 3 rule 1).
"""

from __future__ import annotations

from typing import Callable

from .message import Operation, SyndromePayload


def _stream_payload_target(op: Operation, round_index: int) -> tuple:
    """(decode_op_id, global_round) for standalone ops or stream segments."""
    return (op.stream_id if op.stream_id is not None else op.id,
            round_index + (op.stream_offset or 0))


class TimingOnlyDevice:
    """Emit payloads with no syndrome bits for timing-only studies
    (the default RunSpec device)."""

    def begin_operation(self, op: Operation) -> None:
        return None

    def round_payloads(self, op: Operation, round_index: int) -> list:
        target, global_round = _stream_payload_target(op, round_index)
        return [SyndromePayload(target,
                              op.patches[0] if op.patches else op.qubits[0],
                              global_round)]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list:
        return [SyndromePayload(stream_id, patch, global_round)]

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, belief_matching: bool = False):
        return None

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        return None

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int,
                                    *, belief_matching: bool = False) -> list:
        return []

    def window_model_for_stream(self, stream_id, window, *, is_last: bool):
        return None

    def strong_window_model_for_operation(self, op: Operation, window,
                                          round_count: int,
                                          *, belief_matching: bool = False,
                                          exclude_faults_touching=None):
        return None

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        belief_matching: bool = False, fault_exclusion_ranges: tuple,
    ):
        return None


class ClockedDevice:
    """Emit one syndrome round per round-tick per running operation."""

    def __init__(self, engine, device, controller, cluster, rounds_for):
        self.engine = engine
        self.device = device            # round_payloads / begin_operation / ...
        self.controller = controller    # relay path (t_qc + t_cd)
        self.cluster = cluster          # on_syndrome_arrival sink
        self.rounds_for = rounds_for    # op -> total rounds (shared with planner)

    def start(self, operation, round_ticks: int,
              on_body_done: Callable) -> None:
        """Begin an operation's stream: round 1 fires one round-tick from now."""
        self.device.begin_operation(operation)
        self.engine.schedule(
            round_ticks,
            lambda: self._round(operation, 1, round_ticks, on_body_done),
            label=f"round1({operation.name})")

    def _round(self, operation, round_index: int, round_ticks: int,
               on_body_done: Callable) -> None:
        """Emit one round; the final round triggers body-done in this event."""
        total_rounds = self.rounds_for(operation)
        payloads = self.device.round_payloads(operation, round_index)
        self.engine.log("Chip", f"{operation.name} fires round "
                                f"{round_index}/{total_rounds}")
        self.relay_payloads(payloads)
        if round_index < total_rounds:
            self.engine.schedule(
                round_ticks,
                lambda: self._round(operation, round_index + 1, round_ticks,
                                    on_body_done),
                label=f"round{round_index + 1}({operation.name})")
        else:
            on_body_done(operation)

    def relay_payloads(self, payloads) -> None:
        """Send all fragments from one syndrome round through the controller."""
        for payload in payloads:
            payload.n_fragments = len(payloads)
            self.controller.relay_syndrome(payload,
                                           self.cluster.on_syndrome_arrival)

    def idle_round_payloads(self, operation, stream_id, global_round, patch):
        """Idle-round payloads for extend_stream mode (delegates to the device)."""
        return self.device.idle_round_payloads(operation, stream_id,
                                               global_round, patch)


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
            num_patches = len(op.patches) if op.patches else len(op.qubits)
            bits = self._bits(num_patches)
            return [SyndromePayload(
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
            payloads.append(SyndromePayload(
                target, patch, global_round, bits=bits, code=self.code.name,
                size_bits=len(bits)))
        return payloads

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[SyndromePayload]:
        """Emit one fake-bit payload for a feedback-idle stream round."""
        bits = self._bits(1)
        return [SyndromePayload(stream_id, patch, global_round, bits=bits,
                                code=self.code.name, size_bits=len(bits))]

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

    def strong_window_model_for_operation(self, op: Operation, window, round_count: int,
                                          *, belief_matching: bool = False,
                                          exclude_faults_touching=None):
        """Fake-bit strong re-decodes carry no detector error model."""
        return None

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        belief_matching: bool = False, fault_exclusion_ranges: tuple,
    ):
        """Fake-bit strong re-decodes carry no detector error model."""
        return None
