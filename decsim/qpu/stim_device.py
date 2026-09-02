"""The Stim syndrome source: sampled measurement bits, one packet per round.

The device samples one shot of an operation's Stim circuit, or replays a
recorded shot, and emits the raw measurement bits round by round, the way
readout electronics do: every round carries one bit per measure qubit,
and the final round of a memory circuit also carries the data-qubit
readout (Stim, src/stim/gen/gen_surface_code.cc: MR on the measure qubits
every round, M on the data qubits at the end). Detection events are
formed later, at the decoder input, by form_round from the same formation
table (detector_error_model/detector_formation.py).

One shot is sampled per stream identity and reused by every segment of
the stream. With a numeric seed every identity gets its own substream, a
blake2b hash of the root seed and the identity, so a run is reproducible
across processes; with no seed Stim picks its own entropy. The device
also builds the window error models of its circuits through the detector
error model package.
"""

import dataclasses
import hashlib
import numbers
from collections.abc import Sequence
from typing import Any, Optional

import numpy

import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.window_model_builders as window_models
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.message as message
import decsim.seeding as seeding

# Stream ids, operation ids and patches are opaque identities chosen by
# the workload; Any stands for them in every signature below.


class StimDevice(seeding._AtomicRunSeedConsumer):
    """Streams one sampled Stim shot as raw measurement packets, by round.

    The optional maps are keyed by stream identity and use one-based
    emitted rounds. detector_rounds says which round each detector belongs
    to when the circuit does not follow Stim's generator layout;
    measurement_rounds declares the QPU's packet schedule for the same
    reason. A non-empty terminal_detector_ids entry says the stream's
    final data readout arrives as its own fragment, through
    finalize_stream_round.
    """

    operation_circuit_scope = "per_operation"

    def __init__(
        self,
        seed: Optional[numbers.Integral] = None,
        detector_rounds: Optional[dict] = None,
        terminal_detector_ids: Optional[dict] = None,
        measurement_rounds: Optional[dict] = None,
    ):
        self._seed = _validated_seed(seed)
        self._initialize_run_seed_binding(self._seed)
        detector_rounds = detector_rounds or {}
        self._detector_rounds_override = {
            key: dict(rounds_map) for key, rounds_map in detector_rounds.items()
        }
        terminal_detector_ids = terminal_detector_ids or {}
        self._terminal_detector_ids = {
            key: tuple(detector_ids)
            for key, detector_ids in terminal_detector_ids.items()
        }
        measurement_rounds = measurement_rounds or {}
        self._measurement_rounds_override = {
            key: dict(rounds_map)
            for key, rounds_map in measurement_rounds.items()
        }
        # A shot sits under its sample key (the stream id, or the operation
        # id of a standalone operation) and under every operation id that
        # replays it.
        self._shot_by_key: dict = {}
        self._stream_model_by_id: dict = {}
        self._source_binding_by_key: dict = {}

    def sampled_truth(self) -> dict:
        """Every sampled observable-flip vector, by operation or stream."""
        truth_by_key = {}
        for key, shot in self._shot_by_key.items():
            truth_by_key[key] = _as_int_bits(shot.truth)
        return truth_by_key

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """Stim's sampled observable-flip vector, when the shot exists."""
        shot = self._shot_by_key.get(operation_id)
        if shot is None:
            return None
        return _as_int_bits(shot.truth)

    def sampled_detection_events(self, operation_id: Any) -> tuple[bool, ...]:
        """The shot's detection events, in the circuit's detector order."""
        shot = self._shot_by_key.get(operation_id)
        if shot is None:
            raise KeyError(
                f"no sampled detection events for identity {operation_id!r}; "
                "the operation has not begun"
            )
        return tuple(bool(bit) for bit in shot.detection_events)

    def begin_operation(
        self,
        operation: message.Operation,
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Sample one fresh shot, or reuse the stream's shot for a segment."""
        if operation.circuit is None:
            raise ValueError("StimDevice operations require a circuit")
        _check_segment(operation, segment_round_count, source_round_count)
        key = _sample_key_of(operation)
        if self._seed is not None:
            _check_sample_key(key)
        detector_rounds = self._bind_source(
            key, operation.circuit, source_round_count
        )
        is_stream = operation.stream_id is not None
        if is_stream and operation.stream_offset:
            self._shot_by_key[operation.id] = self._shot_by_key[key]
            return
        self._sample_shot(key, operation, source_round_count, detector_rounds)

    def form_round(
        self, operation_id: Any, round_index: int, raw_bits: Sequence[int]
    ) -> tuple[int, ...]:
        """The detection events of one complete round, in detector order.

        Formed at the decoder input from the round's raw packet.
        """
        shot = self._shot_by_key.get(operation_id)
        if shot is None:
            raise KeyError(
                f"no detector formation state for identity {operation_id!r}; "
                "the operation has not begun"
            )
        events, _ = shot.former.feed_packet(round_index, raw_bits)
        return tuple(value for _, value in events)

    def round_payloads(
        self, operation: message.Operation, round_index: int
    ) -> list[message.QPUReadout]:
        """This operation round as one raw measurement packet."""
        key = _sample_key_of(operation)
        stream_offset = 0
        if operation.stream_offset is not None:
            stream_offset = operation.stream_offset
        global_round = round_index + stream_offset
        bits = self._round_packet_bits(key, global_round)
        patch = _first_patch_or_zero(operation)
        return [
            message.QPUReadout(
                key, patch, global_round, bits=bits, size_bits=len(bits)
            )
        ]

    def finalize_stream_round(
        self, operation: message.Operation, source_round_count: int
    ) -> list[message.QPUReadout]:
        """The stream's final data readout as its own raw fragment."""
        key = _sample_key_of(operation)
        table = self._check_finalizer(key, operation, source_round_count)
        shot = self._shot_by_key[key]
        final_packet = shot.packets[table.round_count]
        bits = final_packet[table.readout_slot_start :]
        patch = _first_patch_of(operation)
        final_round = operation.stream_offset + 1
        return [
            message.QPUReadout(
                key, patch, final_round, bits=bits, size_bits=len(bits)
            )
        ]

    def idle_round_payloads(
        self,
        operation: message.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> list[message.QPUReadout]:
        """This idle stream round as one raw measurement packet."""
        del operation
        if stream_id not in self._shot_by_key:
            raise RuntimeError("idle emission requires a sampled bound stream")
        binding = self._source_binding_by_key[stream_id]
        is_int = type(global_round) is int
        if not is_int or not 1 <= global_round <= binding.round_count:
            raise ValueError("idle round is outside the finite source")
        bits = self._round_packet_bits(stream_id, global_round)
        return [
            message.QPUReadout(
                stream_id, patch, global_round, bits=bits, size_bits=len(bits)
            )
        ]

    def register_dynamic_stream(
        self,
        stream_operation: message.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> Optional[int]:
        """Prepare the window model source of one finite Stim stream.

        Returns the stream's fixed round count, or None without a circuit.
        """
        if stream_operation.circuit is None:
            return None
        detector_rounds = self._bind_source(
            stream_operation.id, stream_operation.circuit, round_count
        )
        slicer = window_slicer.WindowSlicer(
            stream_operation.circuit,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
        )
        stream_model = _StreamModel(round_count, slicer)
        self._stream_model_by_id[stream_operation.id] = stream_model
        return round_count

    def validate_stream_length(
        self, stream_operation: message.Operation, stream_round_count: int
    ) -> None:
        """Refuse a stream whose runtime length differs from its circuit."""
        stream_model = self._stream_model_by_id.get(stream_operation.id)
        if stream_model is None:
            return
        finite_round_count = stream_model.round_count
        if stream_round_count == finite_round_count:
            return
        raise RuntimeError(
            f"{stream_operation.name} sealed at {stream_round_count} rounds, "
            f"but its Stim circuit was registered for {finite_round_count} "
            "rounds. Real-syndrome live streams need an exact finite "
            "circuit. Use a timing-only stream for unknown feedback length, "
            "or build the Stim circuit after the stream length is known."
        )

    def window_models_for_operation(
        self,
        operation: message.Operation,
        windows: list[message.Window],
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
        window_protocol: message.WindowProtocol,
    ) -> list[fault_models.WindowErrorModel]:
        """The detector error models of one finite Stim operation's windows."""
        if operation.circuit is None or not windows:
            return []
        key = _sample_key_of(operation)
        detector_rounds = self._bind_source(key, operation.circuit, round_count)
        model_plan = [_window_span(window, round_count) for window in windows]
        index_by_key = {
            window.key: index for index, window in enumerate(windows)
        }
        dependency_edges = _dependency_edges(windows, index_by_key)
        closed_windows = _closed_temporal_boundary_windows(windows)
        return window_models.build_window_error_models(
            operation.circuit,
            model_plan,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges,
            dependency_edges=dependency_edges,
            closed_temporal_boundary_windows=closed_windows,
            window_protocol=window_protocol,
        )

    def window_model_for_stream(
        self, stream_id: Any, window: message.Window
    ) -> Optional[fault_models.WindowErrorModel]:
        """The detector error model of one dynamic stream window.

        The window whose commit region reaches the stream's last round is
        the terminal one.
        """
        stream_model = self._stream_model_by_id.get(stream_id)
        if stream_model is None:
            return None
        is_last = window.commit_hi == stream_model.round_count
        return stream_model.slicer.slice_window(
            window.start_round,
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
            is_last=is_last,
        )

    def strong_window_model_for_operation(
        self,
        operation: message.Operation,
        window: message.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        exclude_faults_touching: Optional[tuple] = None,
    ) -> Optional[fault_models.WindowErrorModel]:
        """An independent two-sided context model for a strong re-decode.

        One optional inclusive range is assigned to another seam side.
        """
        if operation.circuit is None:
            return None
        key = _sample_key_of(operation)
        detector_rounds = self._bind_source(key, operation.circuit, round_count)
        span = _window_span(window, round_count)
        return window_models.build_single_window_error_model(
            operation.circuit,
            span,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            exclude_faults_touching=exclude_faults_touching,
        )

    def strong_window_model_for_operation_with_exclusions(
        self,
        operation: message.Operation,
        window: message.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
    ) -> Optional[fault_models.WindowErrorModel]:
        """A strong re-decode model with several non-owned inclusive ranges."""
        if operation.circuit is None:
            return None
        key = _sample_key_of(operation)
        detector_rounds = self._bind_source(key, operation.circuit, round_count)
        span = _window_span(window, round_count)
        return window_models.build_single_window_error_model_with_exclusions(
            operation.circuit,
            span,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges,
        )

    def _prepare_run_seed_state(self, effective_seed):
        return (effective_seed, {}, {}, {})

    def _install_run_seed_state(self, prepared_state) -> None:
        (
            self._seed,
            self._shot_by_key,
            self._stream_model_by_id,
            self._source_binding_by_key,
        ) = prepared_state

    def _sample_seed_for(self, key) -> int:
        """A substream seed that is stable across processes for one identity."""
        key_type_tag = b"str"
        if type(key) is int:
            key_type_tag = b"int"
        root_text = str(self._seed)
        key_text = str(key)
        root_bytes = root_text.encode()
        key_bytes = key_text.encode()
        hash_input = b"\0".join(
            (root_bytes, b"stim_device", key_type_tag, key_bytes)
        )
        hasher = hashlib.blake2b(hash_input, digest_size=8)
        digest = hasher.digest()
        return int.from_bytes(digest, "big")

    def _sampler_for(self, key, circuit):
        shot = self._shot_by_key.get(key)
        if shot is not None:
            return shot.sampler
        if self._seed is None:
            return circuit.compile_sampler()
        sample_seed = self._sample_seed_for(key)
        return circuit.compile_sampler(seed=sample_seed)

    def _sample_shot(
        self,
        key,
        operation: message.Operation,
        source_round_count: int,
        detector_rounds: dict,
    ) -> None:
        sampler = self._sampler_for(key, operation.circuit)
        self._mark_stochastic_use()
        measurement_row = self._measurement_row(sampler)
        measurement_rounds = self._measurement_rounds_override.get(key)
        table = detector_formation.build_formation_table(
            operation.circuit,
            source_round_count,
            measurement_rounds=measurement_rounds,
            detector_rounds=detector_rounds,
        )
        packets = detector_formation.split_measurements_into_packets(
            table, measurement_row
        )
        # The whole-shot formation is the oracle for the truth and for
        # sampled_detection_events; the streaming former is what the
        # decoder input runs, one packet at a time.
        formed_events, formed_truth = detector_formation.form_shot(
            table, packets
        )
        former = detector_formation.StreamingDetectorFormer(table)
        shot = _SampledShot(
            sampler, packets, table, former, formed_events, formed_truth
        )
        self._shot_by_key[key] = shot
        self._shot_by_key[operation.id] = shot

    def _measurement_row(self, sampler) -> tuple[int, ...]:
        """One shot of raw measurement bits in circuit measurement order."""
        shots = sampler.sample(shots=1)
        row = shots[0]
        return _as_int_bits(row)

    def _round_packet_bits(self, key, global_round: int) -> tuple[int, ...]:
        """The raw bits the QPU emits for one round.

        When the stream declares a terminal fragment, the folded readout
        bits of the final round stay behind for finalize_stream_round.
        """
        shot = self._shot_by_key[key]
        packet = shot.packets[global_round]
        table = shot.table
        terminal_ids = self._terminal_detector_ids.get(key)
        is_final_round = global_round == table.round_count
        has_terminal_fragment = bool(terminal_ids)
        has_folded_readout = table.readout_slot_start is not None
        readout_arrives_separately = is_final_round and has_terminal_fragment
        if readout_arrives_separately and has_folded_readout:
            return packet[: table.readout_slot_start]
        return packet

    def _check_finalizer(
        self, key, operation: message.Operation, source_round_count: int
    ):
        """The formation table of a stream whose finalizer is well formed."""
        if key not in self._shot_by_key:
            raise RuntimeError("terminal finalizer requires a sampled stream")
        binding = self._source_binding_by_key[key]
        if source_round_count != binding.round_count:
            raise ValueError(
                "finalizer source duration differs from its binding"
            )
        circuit_text = None
        if operation.circuit is not None:
            circuit_text = str(operation.circuit)
        if circuit_text != binding.circuit_text:
            raise ValueError(
                "finalizer circuit differs from its source binding"
            )
        final_round = operation.stream_offset + 1
        if final_round != source_round_count:
            raise ValueError("finalizer is not at the final source round")
        if not self._terminal_detector_ids.get(key):
            raise ValueError("terminal finalizer has no declared detector ids")
        shot = self._shot_by_key[key]
        table = shot.table
        if table.readout_slot_start is None:
            raise ValueError("terminal finalizer has no folded readout bits")
        return table

    def _bind_source(self, key, circuit, source_round_count: int) -> dict:
        """Bind one finite circuit, duration and detector chronology per key."""
        detector_rounds_override = self._detector_rounds_override.get(key)
        resolved = detector_chronology.resolve_detector_rounds(
            circuit, detector_rounds_override, source_round_count
        )
        circuit_text = str(circuit)
        binding = self._source_binding_by_key.get(key)
        if binding is None:
            detector_rounds = dict(resolved)
            self._source_binding_by_key[key] = _SourceBinding(
                circuit_text, source_round_count, detector_rounds
            )
            return detector_rounds
        if binding.circuit_text != circuit_text:
            raise ValueError("circuit differs from the bound finite source")
        if binding.round_count != source_round_count:
            raise ValueError(
                "source duration differs from the bound finite source"
            )
        return binding.detector_rounds


class RecordedStimDevice(StimDevice):
    """Replays recorded raw measurements (hardware data) instead of sampling.

    measurements is a (shots, measurements) bool array in the measurement
    order of the operation's circuit (the measurements.b8 of a released
    experiment); shot selects the row. Detection events, observable truth,
    round chronology and window error models come from the circuit
    exactly as for sampled data.
    """

    def __init__(self, measurements: numpy.ndarray, shot: int, **settings: Any):
        StimDevice.__init__(self, **settings)
        self.measurements = measurements
        self.shot = shot

    @staticmethod
    def detector_rounds_from_coordinates(
        circuit: Any, round_count: int
    ) -> dict[int, int]:
        """The one-based emitted round of every detector of a hardware circuit.

        Google's released memory experiments write detector coordinates as
        concatenated (x, y, t) triples; a detector belongs to its latest t.
        """
        rounds = {}
        coordinates_by_detector = circuit.get_detector_coordinates()
        for detector_id, coordinates in coordinates_by_detector.items():
            layer = int(max(coordinates[2::3]))
            if layer >= round_count:
                rounds[detector_id] = round_count
            else:
                rounds[detector_id] = layer + 1
        return rounds

    def _measurement_row(self, sampler) -> tuple[int, ...]:
        del sampler
        row = self.measurements[self.shot]
        return _as_int_bits(row)


@dataclasses.dataclass(frozen=True)
class _SampledShot:
    """One shot of a circuit: its sampler, packets and formed events."""

    sampler: Any
    packets: dict
    table: detector_formation.FormationTable
    former: detector_formation.StreamingDetectorFormer
    detection_events: tuple
    truth: tuple


@dataclasses.dataclass(frozen=True)
class _SourceBinding:
    """The finite circuit one stream identity is bound to."""

    circuit_text: str
    round_count: int
    detector_rounds: dict


@dataclasses.dataclass(frozen=True)
class _StreamModel:
    """The window slicer of one registered dynamic stream."""

    round_count: int
    slicer: window_slicer.WindowSlicer


def _sample_key_of(operation: message.Operation):
    if operation.stream_id is not None:
        return operation.stream_id
    return operation.id


def _validated_seed(seed) -> Optional[int]:
    """The seed under Stim's public unsigned 64-bit contract, or None."""
    if seed is None:
        return None
    if not isinstance(seed, numbers.Integral):
        raise ValueError(
            f"seed must be None or a 64-bit unsigned integer; got {seed!r}"
        )
    root_seed = int(seed)
    seed_limit = 1 << 64
    if not 0 <= root_seed < seed_limit:
        raise ValueError(
            f"seed must be None or a 64-bit unsigned integer; got {seed!r}"
        )
    return root_seed


def _check_sample_key(key) -> None:
    """Refuse an identity whose equality could alias a legal cache key."""
    key_type = type(key)
    if key_type not in (int, str):
        raise TypeError(
            f"stream_id must be an int or str so the run's sampling is "
            f"reproducible across processes; sample key {key!r} is a "
            f"{key_type.__name__}"
        )


def _check_segment(
    operation: message.Operation,
    segment_round_count: int,
    source_round_count: int,
) -> None:
    if operation.stream_id is None:
        has_offset = operation.stream_offset is not None
        if has_offset or segment_round_count != source_round_count:
            raise ValueError(
                "standalone duration must equal its source duration"
            )
        return
    segment_end = operation.stream_offset + segment_round_count
    if segment_end > source_round_count:
        raise ValueError("stream segment extends beyond its finite source")


def _as_int_bits(bits) -> tuple[int, ...]:
    return tuple(int(bit) for bit in bits)


def _first_patch_of(operation: message.Operation):
    if operation.patches:
        return operation.patches[0]
    return operation.qubits[0]


def _first_patch_or_zero(operation: message.Operation):
    if operation.patches:
        return operation.patches[0]
    if operation.qubits:
        return operation.qubits[0]
    return 0


def _window_span(window: message.Window, round_count: int) -> tuple:
    last_read_round = min(window.buffer_hi, round_count)
    return (
        window.start_round,
        window.commit_lo,
        window.commit_hi,
        last_read_round,
    )


def _dependency_edges(windows: list, index_by_key: dict) -> tuple:
    edges = []
    for destination_index, window in enumerate(windows):
        source_indexes = _local_dependencies(window, index_by_key)
        for source_index in source_indexes:
            edges.append((source_index, destination_index))
    return tuple(edges)


def _local_dependencies(window, index_by_key: dict) -> list:
    return [
        index_by_key[dependency]
        for dependency in window.deps
        if dependency in index_by_key
    ]


def _closed_temporal_boundary_windows(windows: list) -> tuple:
    closed = []
    for index, window in enumerate(windows):
        if window.closed_temporal_boundaries:
            closed.append(index)
    return tuple(closed)
