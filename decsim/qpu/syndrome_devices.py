"""Syndrome sources without a circuit: timing-only and fake-bit readout.

TimingOnlyDevice, the default source, emits payloads with a size and no
bit values. A timing simulation models a transfer's size, not its
content: gem5's packet trace records no data (src/proto/packet.proto),
and its network tester does "timing simulation of the network" only
(GarnetSyntheticTraffic.cc). SyndromeBitDevice emits seeded random bits
of the same width, to exercise the payload path without Stim. Both
state a round's raw width and its event width layer by layer
(_round_widths), so each seat prices what it holds, and both state a
split-off data readout in finalize_stream_round.
"""

import dataclasses
import random
from collections.abc import Mapping
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


def boundary_departure_tick(
    readout: round_records.QPUReadout, readout_tick: int
) -> int:
    """The readout leaves the chip at the boundary it was read out at.

    A source with a fixed delay or jitter names a later tick of its own
    (ports.SyndromeSource.readout_departure_tick).
    """
    del readout
    return readout_tick


class CircuitlessSource:
    """What a syndrome source with no circuit answers, whatever it emits.

    No shot, so no truth; no stream length or window model; the readout
    leaves at its boundary; each operation's last round is kept by decode
    identity.
    """

    operation_circuit_scope = "none"
    # nothing is sampled per shot, so the port's shot source never fires
    shot_sampled = trace_source.SILENT

    def __init__(self, code: ports.CodeModel) -> None:
        self.code = code
        # decode identity -> the source's rounds, its last read out
        self.round_count_by_identity: dict = {}

    def logical_observable_truth(
        self,
        operation_id: Any,  # an opaque identity
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
        """Nothing to sample; the round the data qubits are read out on."""
        del segment_round_count, round_period_ticks
        target = program_records.decode_identity(operation)
        self.round_count_by_identity[target] = source_round_count

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

    readout_departure_tick = staticmethod(boundary_departure_tick)

    def window_model_source(self) -> "NoWindowModels":
        """No circuit, so no window has a model to build."""
        return NO_WINDOW_MODELS


class TimingOnlyDevice(CircuitlessSource):
    """Emits payloads with a size and no bit values: timing alone.

    A rotated surface code "requires d2 - 1 syndrome qubits" a round
    (Barber et al. 2309.05558 lines 947-951), and an operation's last round
    adds its data qubits (_round_widths).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The timing-only source has no keys: the card states every width."""

        # the word the reports name this row by
        name = "timing_only"

        def build(
            self, code: ports.CodeModel, circuit_arguments: Mapping
        ) -> "TimingOnlyDevice":
            """A fresh source shaped by the run's card."""
            del circuit_arguments
            return TimingOnlyDevice(code)

    emits_bit_values = False

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """One valueless payload attributed to the operation's footprint."""
        target = program_records.decode_identity(operation)
        global_round = program_records.global_round(operation, round_index)
        last_round = self.round_count_by_identity.get(target)
        has_data_readout = _reads_the_data_out(
            operation, global_round, last_round, 1
        )
        patches = program_records.patches_of(operation)
        patch_count = _patch_count_of(operation)
        readout = _valueless_readout(
            self.code,
            target,
            patches,
            global_round,
            has_data_readout,
            patch_count,
        )
        return [readout]

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,  # an opaque identity
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """One valueless payload for the idle stream round."""
        del round_period_ticks
        patches = program_records.patches_of(operation)
        patch_count = _patch_count_of(operation)
        readout = _valueless_readout(
            self.code, stream_id, patches, global_round, is_final, patch_count
        )
        return [readout]

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """The stream's final data readout as its own valueless fragment."""
        del source_round_count
        target = program_records.decode_identity(operation)
        final_round = program_records.global_round(operation, 1)
        patches = program_records.patches_of(operation)
        patch_count = _patch_count_of(operation)
        raw_bits, event_bits = _data_readout_widths(self.code, patch_count)
        readout = round_records.QPUReadout(
            target,
            patches,
            final_round,
            size_bits=raw_bits,
            event_bits=event_bits,
        )
        return [readout]


class SyndromeBitDevice(CircuitlessSource, seeding._AtomicRunSeedConsumer):
    """Emits seeded random bits shaped like the code card's syndrome.

    Each payload's generator is seeded by the substream of its stream, round
    and patches (seeding.substream_seed), so a round's bits do not depend on
    how many rounds another operation drew first, which timing decides.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The random-bit source has no keys: the card states every width."""

        # the word the reports name this row by
        name = "syndrome_bits"

        def build(
            self, code: ports.CodeModel, circuit_arguments: Mapping
        ) -> "SyndromeBitDevice":
            """A fresh source shaped by the run's card."""
            del circuit_arguments
            return SyndromeBitDevice(code)

    emits_bit_values = True

    def __init__(
        self,
        code: ports.CodeModel,
        seed: Optional[int] = None,
        one_payload_per_patch: bool = False,
    ):
        CircuitlessSource.__init__(self, code)
        self.one_payload_per_patch = one_payload_per_patch
        self._seed = seed
        self._initialize_run_seed_binding(seed)

    def run_seed_children(self) -> tuple[seed_records.RunSeedChild, ...]:
        """The code card, which shapes every payload."""
        segment = seed_records.RunSeedPathSegment("field", "code")
        return (seed_records.RunSeedChild((segment,), self.code),)

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """One payload per patch, or one payload covering every patch."""
        target = program_records.decode_identity(operation)
        global_round = program_records.global_round(operation, round_index)
        last_round = self.round_count_by_identity.get(target)
        groups = self._patch_groups(operation)
        payload_count = len(groups)
        has_data_readout = _reads_the_data_out(
            operation, global_round, last_round, payload_count
        )
        return self._payloads(operation, target, global_round, has_data_readout)

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,  # an opaque identity
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """One fake-bit payload for the idle stream round."""
        del round_period_ticks
        return self._payloads(operation, stream_id, global_round, is_final)

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """The final data readout in the round's own payload shape."""
        del source_round_count
        target = program_records.decode_identity(operation)
        final_round = program_records.global_round(operation, 1)
        payloads = []
        for patch_ids, patch_count in self._patch_groups(operation):
            widths = _data_readout_widths(self.code, patch_count)
            payload = self._payload(target, patch_ids, final_round, widths)
            payloads.append(payload)
        return payloads

    def _install_run_seed_state(self, prepared_state) -> None:
        self._seed = prepared_state

    def _fake_bits(
        self,
        target: Any,  # an opaque identity
        global_round: int,
        patches: tuple,
        bit_count: int,
    ) -> list:
        self._mark_stochastic_use()
        generator = self._payload_generator(target, global_round, patches)
        return [generator.randint(0, 1) for _ in range(bit_count)]

    def _payload_generator(
        self,
        target: Any,  # an opaque identity
        global_round: int,
        patches: tuple,
    ) -> random.Random:
        """The generator of one payload; an unseeded device draws entropy."""
        if self._seed is None:
            return random.Random()
        keys = (target, global_round, *patches)
        payload_seed = seeding.substream_seed(self._seed, keys)
        return random.Random(payload_seed)

    def _payloads(
        self,
        operation: program_records.Operation,
        target: Any,  # an opaque identity
        global_round: int,
        has_data_readout: bool,
    ) -> list[round_records.QPUReadout]:
        """The round's payloads: one per patch, or one over every patch."""
        payloads = []
        for patch_ids, patch_count in self._patch_groups(operation):
            widths = _round_widths(
                self.code, patch_count, global_round, has_data_readout
            )
            payload = self._payload(target, patch_ids, global_round, widths)
            payloads.append(payload)
        return payloads

    def _patch_groups(self, operation: program_records.Operation) -> tuple:
        """(patch ids, patch count) of each payload: one per patch, or one."""
        if not self.one_payload_per_patch:
            patches = program_records.patches_of(operation)
            patch_count = _patch_count_of(operation)
            return ((patches, patch_count),)
        patches = operation.patches
        if not patches:
            patches = operation.qubits
        groups = []
        for patch in patches:
            groups.append(((patch,), 1))
        return tuple(groups)

    def _payload(
        self,
        target: Any,  # an opaque identity
        patches: tuple,
        global_round: int,
        widths: tuple,
    ) -> round_records.QPUReadout:
        """Random raw bits at the (raw, event) widths, its events' stated."""
        raw_bits, event_bits = widths
        bits = self._fake_bits(target, global_round, patches, raw_bits)
        return round_records.QPUReadout(
            target,
            patches,
            global_round,
            bits=bits,
            size_bits=raw_bits,
            event_bits=event_bits,
        )


class NoWindowModels:
    """The window model source of a circuit-less source: no model at all."""

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
        self,
        stream_id: Any,  # an opaque identity
        window: window_records.Window,
    ) -> None:
        """No circuit, so no stream window has an error model."""

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple = (),
        prior_faults: Optional[dict] = None,
    ) -> None:
        """No circuit, so no strong re-decode has an error model."""


# The one no-model component every circuit-less source names.
NO_WINDOW_MODELS = NoWindowModels()


def _patch_count_of(operation: program_records.Operation) -> int:
    """The patches whose syndrome one round of the operation reads out."""
    if operation.patches:
        return len(operation.patches)
    return len(operation.qubits)


def _round_widths(
    code: ports.CodeModel,
    patch_count: int,
    global_round: int,
    has_data_readout: bool,
) -> tuple:
    """(raw bits, event bits) of one memory round, by its layer.

    Stim's three layers (detector_formation.py): c/2 events on the first
    round, c on the middle ones, c + c/2 on the last with the data readout:
    4, 8 and 12 at d=3, with 8 raw bits a round and 17 on the last.
    """
    check_bits = code.syndrome_bits_per_round(patch_count)
    half_the_checks = check_bits // 2
    raw_bits = check_bits
    event_bits = check_bits
    if global_round == 1:
        event_bits = half_the_checks
    if has_data_readout:
        raw_bits += code.data_bits_per_readout(patch_count)
        event_bits += half_the_checks
    return raw_bits, event_bits


def _data_readout_widths(code: ports.CodeModel, patch_count: int) -> tuple:
    """(raw bits, event bits) of a last round's data readout on its own.

    The data qubits leave raw and close the other half of the checks.
    """
    raw_bits = code.data_bits_per_readout(patch_count)
    check_bits = code.syndrome_bits_per_round(patch_count)
    event_bits = check_bits // 2
    return raw_bits, event_bits


def _reads_the_data_out(
    operation: program_records.Operation,
    global_round: int,
    last_round: Optional[int],
    payload_count: int,
) -> bool:
    """Whether this round of the operation carries the data readout.

    The last round does, unless the program split it and the terminal
    fragment brings the readout (finalize_stream_round).
    """
    if global_round != last_round:
        return False
    is_first_half = operation.syndrome_fragment_index == 0
    split_fragment_count = 2 * payload_count
    has_two_halves = operation.syndrome_fragment_count == split_fragment_count
    return not (is_first_half and has_two_halves)


def _valueless_readout(
    code: ports.CodeModel,
    target: Any,  # an opaque identity
    patches: tuple,
    global_round: int,
    has_data_readout: bool,
    patch_count: int,
) -> round_records.QPUReadout:
    """One readout with the round's widths and no bit values."""
    raw_bits, event_bits = _round_widths(
        code, patch_count, global_round, has_data_readout
    )
    return round_records.QPUReadout(
        target,
        patches,
        global_round,
        size_bits=raw_bits,
        event_bits=event_bits,
    )
