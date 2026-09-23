"""Syndrome sources without a circuit: timing-only and fake-bit readout.

A syndrome source fills the SyndromeSource port (decsim/ports.py) and
is driven by the QPU cycle clock (cycle_clock.py): one payload list per
operation round, one per idle stream round. Neither source here has a
circuit, so neither builds a detector error model.

TimingOnlyDevice emits payloads that carry no bit values and state the
code card's syndrome width, for runs that price timing alone; it is the
default source of a run. A timing simulation models a transfer's size
and not its content: gem5's packet trace records a tick, a command, an
address and a size and no data (gem5 src/proto/packet.proto), and its
network tester says "No need to do functional simulation / We just do
timing simulation of the network" (gem5 src/cpu/testers/
garnet_synthetic_traffic/GarnetSyntheticTraffic.cc). SyndromeBitDevice
emits seeded random bits sized by the same width, to exercise the
payload path end to end without Stim. Both name NO_WINDOW_MODELS as
their window model source, which answers every model question with
nothing.
"""

import random
from typing import Any, Optional

import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.ports as ports
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

# Stream ids and patches are opaque identities chosen by the workload; Any
# stands for them in every signature below.


class TimingOnlyDevice:
    """Emits payloads with a size and no bit values: timing alone.

    A round's size is the code card's: a rotated surface code "requires
    d2 - 1 syndrome qubits" a round (Barber et al. 2309.05558 lines
    947-951), so every link and every memory the round crosses can
    price it.
    """

    operation_circuit_scope = "none"
    takes_code_card = True
    # nothing is sampled here, so the port's shot source never fires
    shot_sampled = trace_source.SILENT

    def __init__(self, code: ports.CodeModel) -> None:
        self.code = code

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
        *,
        round_period_ticks: int,
    ) -> None:
        """Nothing to sample."""

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """One valueless payload attributed to the operation's footprint."""
        target, global_round = _stream_target_and_global_round(
            operation, round_index
        )
        patches = program_records.patches_of(operation)
        patch_count = _patch_count_of(operation)
        size_bits = self.code.syndrome_bits_per_round(patch_count)
        readout = round_records.QPUReadout(
            target, patches, global_round, size_bits=size_bits
        )
        return [readout]

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """One valueless payload for the idle stream round."""
        del is_final
        del round_period_ticks
        patches = program_records.patches_of(operation)
        patch_count = _patch_count_of(operation)
        size_bits = self.code.syndrome_bits_per_round(patch_count)
        readout = round_records.QPUReadout(
            stream_id, patches, global_round, size_bits=size_bits
        )
        return [readout]

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Refused: a stream without a circuit has no final readout."""
        del operation, source_round_count
        raise ValueError("TimingOnlyDevice cannot finalize a physical stream")

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> None:
        """No physical circuit constrains this stream's length."""

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """No circuit, so any length is fine."""

    def readout_departure_tick(
        self, readout: round_records.QPUReadout, readout_tick: int
    ) -> int:
        """The readout leaves the chip at the boundary it was read out at."""
        del readout
        return readout_tick

    def window_model_source(self) -> "NoWindowModels":
        """No circuit, so no window has a model to build."""
        return NO_WINDOW_MODELS


class SyndromeBitDevice(seeding._AtomicRunSeedConsumer):
    """Emits seeded random bits shaped like the code card's syndrome.

    Each payload draws from a generator of its own, seeded by the
    device's seed, the stream, the round and the patches, the way the
    Stim source samples each stream under its own substream
    (stim_device.py, _sample_seed_for). A round's bits therefore depend
    on the seed and the round alone, and not on how many rounds another
    operation drew before it, which another component's timing decides.
    """

    operation_circuit_scope = "none"
    takes_code_card = True
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
        code: ports.CodeModel,
        seed: Optional[int] = None,
        one_payload_per_patch: bool = False,
    ):
        self.code = code
        self.one_payload_per_patch = one_payload_per_patch
        self._seed = seed
        self._initialize_run_seed_binding(seed)

    def run_seed_children(self) -> tuple[seed_records.RunSeedChild, ...]:
        """The code card, which shapes every payload."""
        segment = seed_records.RunSeedPathSegment("field", "code")
        return (seed_records.RunSeedChild((segment,), self.code),)

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
        *,
        round_period_ticks: int,
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
        patch_count = _patch_count_of(operation)
        patches = program_records.patches_of(operation)
        bits = self._fake_bits(target, global_round, patches, patch_count)
        return [self._payload(target, patches, global_round, bits)]

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """One fake-bit payload for the idle stream round."""
        del is_final
        del round_period_ticks
        if self.one_payload_per_patch:
            return self._payload_per_patch(operation, stream_id, global_round)
        patches = program_records.patches_of(operation)
        patch_count = len(patches)
        bits = self._fake_bits(stream_id, global_round, patches, patch_count)
        return [self._payload(stream_id, patches, global_round, bits)]

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Refused: a stream without a circuit has no final readout."""
        del operation, source_round_count
        raise ValueError("SyndromeBitDevice cannot finalize a physical stream")

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> None:
        """No physical circuit constrains this stream's length."""

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """No circuit, so any length is fine."""

    def readout_departure_tick(
        self, readout: round_records.QPUReadout, readout_tick: int
    ) -> int:
        """The readout leaves the chip at the boundary it was read out at."""
        del readout
        return readout_tick

    def window_model_source(self) -> "NoWindowModels":
        """No circuit, so no window has a model to build."""
        return NO_WINDOW_MODELS

    def _install_run_seed_state(self, prepared_state) -> None:
        self._seed = prepared_state

    def _fake_bits(
        self, target: Any, global_round: int, patches: tuple, patch_count: int
    ) -> list:
        bit_count = self.code.syndrome_bits_per_round(patch_count)
        self._mark_stochastic_use()
        generator = self._payload_generator(target, global_round, patches)
        return [generator.randint(0, 1) for _ in range(bit_count)]

    def _payload_generator(
        self, target: Any, global_round: int, patches: tuple
    ) -> random.Random:
        """The generator of one payload; an unseeded device draws entropy.

        random.Random hashes a text seed with SHA-512, so the same text
        gives the same bits in every process (Python's random module,
        seed version 2).
        """
        if self._seed is None:
            return random.Random()
        payload_identity = f"{self._seed}|{target!r}|{global_round}|{patches!r}"
        return random.Random(payload_identity)

    def _payload(
        self, target: Any, patches: tuple, global_round: int, bits: list
    ) -> round_records.QPUReadout:
        return round_records.QPUReadout(
            target,
            patches,
            global_round,
            bits=bits,
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
            patch_ids = (patch,)
            bits = self._fake_bits(target, global_round, patch_ids, 1)
            payload = self._payload(target, patch_ids, global_round, bits)
            payloads.append(payload)
        return payloads


class NoWindowModels:
    """The window model source of a source with no circuit: no model at all.

    One shared component answers every model question with nothing, so a
    circuit-less source fills SyndromeSource alone and names this for its
    models (ports.SyndromeSource.window_model_source).
    """

    # nothing here reads an operation's circuit
    operation_circuit_scope = "none"

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        """No circuit, so no fixed stream length."""

    def finalize_stream_models(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        """No decoder model has a terminal boundary to fix."""
        del stream_operation
        del stream_round_count
        return False

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


# The one no-model component every circuit-less source names.
NO_WINDOW_MODELS = NoWindowModels()


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


def _patch_count_of(operation: program_records.Operation) -> int:
    """The patches whose syndrome one round of the operation reads out."""
    if operation.patches:
        return len(operation.patches)
    return len(operation.qubits)
