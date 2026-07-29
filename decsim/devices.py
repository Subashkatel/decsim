"""Clocked syndrome source (port 2): per-op round emission on the round clock.

Part module: the round-emission half of the QPU seam (the control half is
chip.py's Chip). The Chip drives it via start(op, ...) and receives
on_body_done(op) at the final round — in the SAME event (Contract 3 rule 1).
"""

from __future__ import annotations

from dataclasses import replace
import random
import threading
from types import MappingProxyType
from typing import Callable, Optional

from .message import (
    Operation,
    RunSeedChild,
    RunSeedPathSegment,
    RunSeedReservation,
    SyndromePayload,
)


def _stream_payload_target(op: Operation, round_index: int) -> tuple:
    """(decode_op_id, global_round) for standalone ops or stream segments."""
    return (op.stream_id if op.stream_id is not None else op.id,
            round_index + (op.stream_offset or 0))


class TimingOnlyDevice:
    """Emit payloads with no syndrome bits for timing-only studies
    (the default RunSpec device)."""

    operation_circuit_scope = "none"

    def begin_operation(
        self,
        op: Operation,
        resolved_round_count: int,
    ) -> None:
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

    def __init__(
        self,
        engine,
        device,
        controller,
        cluster,
        round_count_by_operation_id,
    ):
        self.engine = engine
        self.device = device            # round_payloads / begin_operation / ...
        self.controller = controller    # relay path (t_qc + t_cd)
        self.cluster = cluster          # on_syndrome_arrival sink
        self._round_count_by_operation_id = MappingProxyType(
            dict(round_count_by_operation_id)
        )

    def start(self, operation, round_ticks: int,
              on_body_done: Callable) -> None:
        """Begin an operation's stream: round 1 fires one round-tick from now."""
        try:
            total_rounds = self._round_count_by_operation_id[operation.id]
        except KeyError as error:
            raise ValueError(
                f"operation {operation.id} has no resolved round count"
            ) from error
        self.device.begin_operation(operation, total_rounds)
        self.engine.schedule(
            round_ticks,
            lambda: self._round(
                operation,
                1,
                total_rounds,
                round_ticks,
                on_body_done,
            ),
            label=f"round1({operation.name})")

    def _round(
        self,
        operation,
        round_index: int,
        total_rounds: int,
        round_ticks: int,
        on_body_done: Callable,
    ) -> None:
        """Emit one round; the final round triggers body-done in this event."""
        payloads = self.device.round_payloads(operation, round_index)
        self.engine.log("Chip", f"{operation.name} fires round "
                                f"{round_index}/{total_rounds}")
        self.relay_payloads(payloads)
        if round_index < total_rounds:
            self.engine.schedule(
                round_ticks,
                lambda: self._round(
                    operation,
                    round_index + 1,
                    total_rounds,
                    round_ticks,
                    on_body_done,
                ),
                label=f"round{round_index + 1}({operation.name})")
        else:
            on_body_done(operation)

    def relay_payloads(self, payloads) -> None:
        """Send all fragments from one syndrome round through the controller."""
        fragment_count = len(payloads)
        for payload in payloads:
            relayed_payload = replace(
                payload,
                n_fragments=fragment_count,
            )
            self.controller.relay_syndrome(relayed_payload,
                                           self.cluster.on_syndrome_arrival)

    def idle_round_payloads(self, operation, stream_id, global_round, patch):
        """Idle-round payloads for extend_stream mode (delegates to the device)."""
        return self.device.idle_round_payloads(operation, stream_id,
                                               global_round, patch)


class SyndromeBitDevice:
    """Emit deterministic fake bits to exercise the payload path."""

    operation_circuit_scope = "none"

    def __init__(self, code: CodeModel, seed: Optional[int] = None,
                 max_bits: int = 8,
                 per_patch: bool = False):
        self.code = code
        self.max_bits = max_bits
        self.per_patch = per_patch
        self._explicit_seed = seed
        self._rng = random.Random(seed)
        self._run_seed_lock = threading.Lock()
        self._pending_run_seed = None
        self._run_seed_claimed = False
        self._stochastic_use_started = False

    def run_seed_children(self):
        """Expose the code model that determines payload shape."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "code"),),
                self.code,
            ),
        )

    def reserve_run_seed(self, seed: Optional[int]) -> RunSeedReservation:
        """Prepare a replacement RNG without advancing the active one."""
        if seed is not None and (
            type(seed) is not int or not 0 <= seed < (1 << 64)
        ):
            raise TypeError(
                "SyndromeBitDevice run root must be an unsigned 64-bit "
                f"built-in integer or None; got {seed!r}"
            )
        with self._run_seed_lock:
            if self._stochastic_use_started:
                raise ValueError(
                    "SyndromeBitDevice was already used and cannot be rebound"
                )
            if self._run_seed_claimed:
                raise ValueError(
                    "SyndromeBitDevice is already claimed by a built run"
                )
            if self._pending_run_seed is not None:
                raise ValueError(
                    "SyndromeBitDevice already has a pending run-seed "
                    "reservation"
                )
            if seed is not None and self._explicit_seed is not None:
                raise ValueError(
                    "SyndromeBitDevice has an explicit seed that conflicts "
                    f"with numeric run root {seed}"
                )
            if seed is not None:
                seed_source = "derived"
                effective_seed = seed
            elif self._explicit_seed is not None:
                if type(self._explicit_seed) is not int:
                    raise TypeError(
                        "SyndromeBitDevice explicit seed must be a built-in "
                        "integer for run provenance"
                    )
                seed_source = "explicit_local"
                effective_seed = self._explicit_seed
            else:
                seed_source = "entropy"
                effective_seed = None
            reservation = RunSeedReservation(
                proposed_seed_source=seed_source,
                proposed_seed=effective_seed,
                prepared_state=random.Random(effective_seed),
            )
            self._pending_run_seed = reservation
            return reservation

    def cancel_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._run_seed_lock:
            if self._pending_run_seed is reservation:
                self._pending_run_seed = None

    def commit_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._run_seed_lock:
            if self._pending_run_seed is not reservation:
                raise ValueError(
                    "SyndromeBitDevice can commit only its exact pending "
                    "run-seed reservation"
                )
            self._rng = reservation.prepared_state
            self._pending_run_seed = None
            self._run_seed_claimed = True

    def begin_operation(
        self,
        op: Operation,
        resolved_round_count: int,
    ) -> None:
        """Nothing to set up."""
        return None

    def _bits(self, num_patches: int) -> list:
        """Fake bits for one payload covering this many patches."""
        bit_count = min(self.code.syndrome_bits_per_round(num_patches), self.max_bits)
        with self._run_seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    "SyndromeBitDevice cannot draw while a run-seed "
                    "reservation is pending"
                )
            self._stochastic_use_started = True
        return [self._rng.randint(0, 1) for _ in range(bit_count)]

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
