"""Syndrome sources without a circuit: timing-only and fake-bit readout.

A syndrome source fills the SyndromeDevice seam (decsim/ports.py) and
is driven by the QPU cycle clock (cycle_clock.py): one payload list per
operation round, one per idle stream round. Neither source here has a
circuit, so both answer every detector-error-model question with nothing.

TimingOnlyDevice emits payloads that carry no bits, for runs that price
timing alone; it is the default source of a run. SyndromeBitDevice emits
seeded random bits sized by the code card's syndrome width, to exercise
the payload path end to end without Stim.
"""

from typing import Any, Optional

import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.qpu.code_geometry as code_geometry
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

# Stream ids and patches are opaque identities chosen by the workload; Any
# stands for them in every signature below.


class TimingOnlyDevice:
    """Emits payloads without bits, so a run prices timing alone."""

    operation_circuit_scope = "none"
    # nothing is sampled here, so the port's shot source never fires
    shot_sampled = trace_source.SILENT

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """This source draws no shot, so it knows no truth."""
        del operation_id
        return None

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Nothing to sample."""

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """One bitless payload on the operation's first patch."""
        target, global_round = _stream_target_and_global_round(
            operation, round_index
        )
        patch = _first_patch_of(operation)
        return [round_records.QPUReadout(target, patch, global_round)]

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> list[round_records.QPUReadout]:
        """One bitless payload for the idle stream round."""
        del operation
        return [round_records.QPUReadout(stream_id, patch, global_round)]

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Refused: a stream without a circuit has no final readout."""
        del operation, source_round_count
        raise ValueError("TimingOnlyDevice cannot finalize a physical stream")

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        """No circuit, so no fixed stream length."""

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """No circuit, so any length is fine."""

    def window_models_for_operation(
        self,
        operation: program_records.Operation,
        windows: list[window_records.Window],
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
        window_protocol: window_records.WindowProtocol,
    ) -> list[fault_models.WindowErrorModel]:
        """No circuit, so no window has an error model."""
        del operation, windows, round_count
        del fault_model_requirement, fault_exclusion_ranges, window_protocol
        return []

    def window_model_for_stream(
        self, stream_id: Any, window: window_records.Window
    ) -> None:
        """No circuit, so no stream window has an error model."""

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        exclude_faults_touching: Optional[tuple] = None,
        prior_faults: Optional[dict] = None,
    ) -> None:
        """No circuit, so no strong re-decode has an error model."""

    def strong_window_model_for_operation_with_exclusions(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict] = None,
    ) -> None:
        """No circuit, so no strong re-decode has an error model."""


class SyndromeBitDevice(seeding._RandomSeedConsumer):
    """Emits seeded random bits shaped like the code card's syndrome."""

    operation_circuit_scope = "none"
    # the bits are drawn per round, not per shot, so nothing fires here
    shot_sampled = trace_source.SILENT

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """This source draws no shot, so it knows no truth."""
        del operation_id
        return None

    def __init__(
        self,
        code: code_geometry.CodeModel,
        seed: Optional[int] = None,
        max_bit_count: int = 8,
        one_payload_per_patch: bool = False,
    ):
        self.code = code
        self.max_bit_count = max_bit_count
        self.one_payload_per_patch = one_payload_per_patch
        self._initialize_run_seed_state(seed)

    def run_seed_children(self) -> tuple[seed_records.RunSeedChild, ...]:
        """The code card, which shapes every payload."""
        segment = seed_records.RunSeedPathSegment("field", "code")
        return (seed_records.RunSeedChild((segment,), self.code),)

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Nothing to sample ahead of time."""

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """One payload per patch, or one payload covering every patch."""
        target, global_round = _stream_target_and_global_round(
            operation, round_index
        )
        if self.one_payload_per_patch:
            return self._payload_per_patch(operation, target, global_round)
        patch_count = len(operation.patches)
        if not operation.patches:
            patch_count = len(operation.qubits)
        bits = self._fake_bits(patch_count)
        patch = _first_patch_of(operation)
        return [self._payload(target, patch, global_round, bits)]

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> list[round_records.QPUReadout]:
        """One fake-bit payload for the idle stream round."""
        del operation
        bits = self._fake_bits(1)
        return [self._payload(stream_id, patch, global_round, bits)]

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Refused: a stream without a circuit has no final readout."""
        del operation, source_round_count
        raise ValueError("SyndromeBitDevice cannot finalize a physical stream")

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        """No circuit, so no fixed stream length."""

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """No circuit, so any length is fine."""

    def window_models_for_operation(
        self,
        operation: program_records.Operation,
        windows: list[window_records.Window],
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
        window_protocol: window_records.WindowProtocol,
    ) -> list[fault_models.WindowErrorModel]:
        """No circuit, so no window has an error model."""
        del operation, windows, round_count
        del fault_model_requirement, fault_exclusion_ranges, window_protocol
        return []

    def window_model_for_stream(
        self, stream_id: Any, window: window_records.Window
    ) -> None:
        """No circuit, so no stream window has an error model."""

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        exclude_faults_touching: Optional[tuple] = None,
        prior_faults: Optional[dict] = None,
    ) -> None:
        """No circuit, so no strong re-decode has an error model."""

    def strong_window_model_for_operation_with_exclusions(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict] = None,
    ) -> None:
        """No circuit, so no strong re-decode has an error model."""

    def _fake_bits(self, patch_count: int) -> list:
        syndrome_bit_count = self.code.syndrome_bits_per_round(patch_count)
        bit_count = min(syndrome_bit_count, self.max_bit_count)
        self._mark_stochastic_use()
        return [self._rng.randint(0, 1) for _ in range(bit_count)]

    def _payload(
        self, target: Any, patch: Any, global_round: int, bits: list
    ) -> round_records.QPUReadout:
        return round_records.QPUReadout(
            target,
            patch,
            global_round,
            bits=bits,
            code=self.code.name,
            size_bits=len(bits),
        )

    def _payload_per_patch(
        self,
        operation: program_records.Operation,
        target: Any,
        global_round: int,
    ) -> list[round_records.QPUReadout]:
        patches = operation.patches
        if not patches:
            patches = operation.qubits
        payloads = []
        for patch in patches:
            bits = self._fake_bits(1)
            payload = self._payload(target, patch, global_round, bits)
            payloads.append(payload)
        return payloads


def _stream_target_and_global_round(
    operation: program_records.Operation, round_index: int
) -> tuple:
    """The decode identity and global round of one operation round.

    A stream segment folds into its stream at its offset; a standalone
    operation is its own stream.
    """
    target = operation.id
    if operation.stream_id is not None:
        target = operation.stream_id
    stream_offset = 0
    if operation.stream_offset is not None:
        stream_offset = operation.stream_offset
    global_round = round_index + stream_offset
    return target, global_round


def _first_patch_of(operation: program_records.Operation) -> Any:
    if operation.patches:
        return operation.patches[0]
    return operation.qubits[0]
