"""Stim-backed syndrome source for real-decoding runs.

The device samples raw measurement bits and emits them as one packet per
round, the way readout electronics do; detection events are formed later,
at the decoder input, by ``form_round`` from the same formation table.
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Optional

from ..detector_error_model.detector_formation import (
    StreamingDetectorFormer,
    build_formation_table,
    form_shot,
    split_measurements_into_packets,
)
from ..message import Operation, QPUReadout
from ..seeding import _AtomicRunSeedConsumer


class StimDevice(_AtomicRunSeedConsumer):
    """Sample Stim circuits and stream raw measurement packets by round.

    Numeric seeds create stable per-identity substreams; ``None`` lets Stim
    choose entropy.
    """

    operation_circuit_scope = "per_operation"

    def __init__(
        self,
        seed: Optional[Integral] = None,
        detector_rounds: Optional[dict] = None,
        terminal_detector_ids: Optional[dict] = None,
        measurement_rounds: Optional[dict] = None,
    ):
        """Configure sampling and explicit chronology or terminal metadata.

        Detector-round and measurement-round maps use one-based emitted
        rounds; measurement rounds declare the QPU's packet schedule when the
        circuit does not follow the Stim-generator layout. A non-empty
        terminal_detector_ids entry says the stream's final data readout
        arrives as its own fragment through finalize_stream_round.
        """
        self._initialize_run_seed_binding(seed)
        self._seed = seed
        self._detector_rounds_override = {
            key: dict(rounds_map)
            for key, rounds_map in (detector_rounds or {}).items()}
        self._terminal_detector_ids = {
            key: tuple(detector_ids)
            for key, detector_ids in (terminal_detector_ids or {}).items()
        }
        self._measurement_rounds_override = {
            key: dict(rounds_map)
            for key, rounds_map in (measurement_rounds or {}).items()}
        self._samplers: dict = {}
        self._packets: dict = {}      # key -> {round: raw bits}
        self._tables: dict = {}       # key -> FormationTable
        self._formers: dict = {}      # key -> StreamingDetectorFormer (decoder-input state)
        self._dets: dict = {}         # key -> formed detection events of the whole shot
        self._truth: dict = {}
        self._stream_models: dict = {}
        self._source_bindings: dict = {}

    def _explicit_run_seed(self):
        return self._validated_root_seed()

    def _prepare_run_seed_state(self, effective_seed):
        return (effective_seed, {}, {}, {}, {}, {}, {}, {}, {})

    def _install_run_seed_state(self, prepared_state) -> None:
        (
            self._seed,
            self._samplers,
            self._packets,
            self._tables,
            self._formers,
            self._dets,
            self._truth,
            self._stream_models,
            self._source_bindings,
        ) = prepared_state

    @staticmethod
    def _key(op: Operation):
        return op.stream_id if op.stream_id is not None else op.id

    def sampled_truth(self) -> dict:
        """Every sampled observable-flip vector by operation or stream identity."""
        return {key: tuple(int(bit) for bit in bits) for key, bits in self._truth.items()}

    def logical_observable_truth(
        self, operation_id: int
    ) -> Optional[tuple[int, ...]]:
        """Return Stim's sampled observable-flip vector, when available."""
        sampled = self._truth.get(operation_id)
        if sampled is None:
            return None
        return tuple(int(bit) for bit in sampled)

    def sampled_detection_events(self, operation_id) -> tuple[bool, ...]:
        """The detection events sampled for this operation or stream identity,
        in the circuit's detector order, as an immutable tuple of bools."""
        sampled_events = self._dets.get(operation_id)
        if sampled_events is None:
            raise KeyError(
                f"no sampled detection events for identity {operation_id!r}; "
                "the operation has not begun")
        return tuple(bool(bit) for bit in sampled_events)

    @staticmethod
    def _validate_sample_key(key) -> None:
        """Reject identities whose equality can alias a legal cache key."""
        sample_key_type = type(key)
        if sample_key_type not in (int, str):
            raise TypeError(
                f"stream_id must be an int or str so the run's sampling is "
                f"reproducible across processes; sample key {key!r} is a "
                f"{sample_key_type.__name__}")

    def _validated_root_seed(self) -> int:
        """Return the seed under Stim's public unsigned-64-bit contract."""
        if not isinstance(self._seed, Integral):
            raise ValueError(
                f"seed must be None or a 64-bit unsigned integer; "
                f"got {self._seed!r}")
        root_seed = int(self._seed)
        if not 0 <= root_seed < (1 << 64):
            raise ValueError(
                f"seed must be None or a 64-bit unsigned integer; "
                f"got {self._seed!r}")
        return root_seed

    def _sample_seed(self, key) -> int:
        """Derive a cross-process-stable substream from the shot-reuse identity."""
        self._validate_sample_key(key)
        root_seed = self._validated_root_seed()
        sample_key_type = type(key)
        key_type_tag = b"int" if sample_key_type is int else b"str"
        hash_input = b"\0".join((
            str(root_seed).encode(),
            b"stim_device",
            key_type_tag,
            str(key).encode(),
        ))
        digest = hashlib.blake2b(
            hash_input,
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big")

    def begin_operation(
        self,
        op: Operation,
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Sample one fresh shot, or reuse the stream shot for later segments."""
        if op.circuit is None:
            raise ValueError("StimDevice operations require a circuit")
        if op.stream_id is None:
            if op.stream_offset is not None or segment_round_count != source_round_count:
                raise ValueError("standalone duration must equal its source duration")
        else:
            if op.stream_offset + segment_round_count > source_round_count:
                raise ValueError("stream segment extends beyond its finite source")
        key = self._key(op)
        if self._seed is not None:
            self._validate_sample_key(key)
        detector_rounds = self._source_rounds(
            key, op.circuit, source_round_count
        )
        if op.stream_id is not None and op.stream_offset:
            self._dets[op.id] = self._dets[key]
            self._truth[op.id] = self._truth[key]
            return
        sampler = self._samplers.get(key)
        if sampler is None:
            if self._seed is None:
                sampler = op.circuit.compile_sampler()
            else:
                sample_seed = self._sample_seed(key)
                sampler = op.circuit.compile_sampler(seed=sample_seed)
            self._samplers[key] = sampler
        self._mark_stochastic_use()
        measurement_row = self._measurement_row(key, sampler)
        table = build_formation_table(
            op.circuit, source_round_count,
            measurement_rounds=self._measurement_rounds_override.get(key),
            detector_rounds=detector_rounds)
        packets = split_measurements_into_packets(table, measurement_row)
        # the whole-shot formation is the oracle for truth and for
        # sampled_detection_events; the streaming former below is what the
        # decoder input actually runs, one packet at a time
        formed_events, formed_truth = form_shot(table, packets)
        self._tables[key] = table
        self._packets[key] = packets
        self._formers[key] = StreamingDetectorFormer(table)
        self._dets[key] = formed_events
        self._truth[key] = formed_truth
        self._dets[op.id] = formed_events
        self._truth[op.id] = formed_truth

    def _measurement_row(self, key, sampler) -> tuple[int, ...]:
        """One shot of raw measurement bits in circuit measurement order."""
        return tuple(int(bit) for bit in sampler.sample(shots=1)[0])

    def _round_packet_bits(self, key, global_round: int) -> tuple[int, ...]:
        """The raw bits the QPU emits for this round. When the stream declares
        a terminal fragment, the folded readout bits stay behind for
        finalize_stream_round."""
        packet = self._packets[key][global_round]
        table = self._tables[key]
        readout_arrives_separately = (
            global_round == table.round_count
            and bool(self._terminal_detector_ids.get(key))
            and table.readout_slot_start is not None)
        if readout_arrives_separately:
            return packet[:table.readout_slot_start]
        return packet

    def form_round(self, operation_id, round_index: int, raw_bits) -> tuple[int, ...]:
        """Detection events of one complete round, in detector order, formed
        at the decoder input from the round's raw packet."""
        former = self._formers.get(operation_id)
        if former is None:
            raise KeyError(
                f"no detector formation state for identity {operation_id!r}; "
                "the operation has not begun")
        events, _ = former.feed_packet(round_index, raw_bits)
        return tuple(value for _, value in events)

    def round_payloads(self, op: Operation, round_index: int) -> list[QPUReadout]:
        """Emit this operation round as one raw measurement packet."""
        key = self._key(op)
        global_round = round_index + (op.stream_offset or 0)
        bits = self._round_packet_bits(key, global_round)
        patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
        target = op.stream_id if op.stream_id is not None else op.id
        return [QPUReadout(target, patch, global_round, bits=bits,
                                size_bits=len(bits))]

    def finalize_stream_round(
        self, op: Operation, source_round_count: int,
    ) -> list[QPUReadout]:
        """Emit the stream's final data readout as its own raw fragment."""
        key = self._key(op)
        if key not in self._packets:
            raise RuntimeError("terminal finalizer requires a sampled stream")
        binding = self._source_bindings.get(key)
        if binding is None:
            raise RuntimeError("sampled stream has no source binding")
        circuit_text, bound_round_count, _ = binding
        if source_round_count != bound_round_count:
            raise ValueError("finalizer source duration differs from its binding")
        if op.circuit is None or str(op.circuit) != circuit_text:
            raise ValueError("finalizer circuit differs from its source binding")
        if op.stream_offset + 1 != source_round_count:
            raise ValueError("finalizer is not at the final source round")
        if not self._terminal_detector_ids.get(key):
            raise ValueError("terminal finalizer has no declared detector ids")
        table = self._tables[key]
        if table.readout_slot_start is None:
            raise ValueError("terminal finalizer has no folded readout bits")
        bits = self._packets[key][table.round_count][table.readout_slot_start:]
        patch = op.patches[0] if op.patches else op.qubits[0]
        return [QPUReadout(
            key,
            patch,
            op.stream_offset + 1,
            bits=bits,
            size_bits=len(bits),
        )]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[QPUReadout]:
        """Emit this feedback-idle stream round as one raw measurement packet."""
        binding = self._source_bindings.get(stream_id)
        if stream_id not in self._packets or binding is None:
            raise RuntimeError("idle emission requires a sampled bound stream")
        if type(global_round) is not int or not 1 <= global_round <= binding[1]:
            raise ValueError("idle round is outside the finite source")
        bits = self._round_packet_bits(stream_id, global_round)
        return [QPUReadout(stream_id, patch, global_round, bits=bits,
                                size_bits=len(bits))]

    def _source_rounds(self, key, circuit, source_round_count: int) -> dict:
        """Bind one finite circuit, duration, and detector chronology per key."""
        from ..detector_error_model.detector_chronology import resolve_detector_rounds

        resolved = resolve_detector_rounds(
            circuit,
            self._detector_rounds_override.get(key),
            source_round_count,
        )
        circuit_text = str(circuit)
        binding = self._source_bindings.get(key)
        if binding is None:
            private_rounds = dict(resolved)
            self._source_bindings[key] = (
                circuit_text, source_round_count, private_rounds
            )
            return private_rounds
        if binding[0] != circuit_text:
            raise ValueError("circuit differs from the bound finite source")
        if binding[1] != source_round_count:
            raise ValueError("source duration differs from the bound finite source")
        return binding[2]

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, fault_model_requirement):
        """Prepare the window model source for one finite Stim-backed stream."""
        if stream_op.circuit is None:
            return None

        from ..detector_error_model.window_slicer import WindowSlicer

        detector_rounds = self._source_rounds(
            stream_op.id, stream_op.circuit, round_count)
        self._stream_models[stream_op.id] = {
            "round_count": round_count,
            "slicer": WindowSlicer(
                stream_op.circuit,
                round_count=round_count,
                detector_rounds=detector_rounds,
                fault_model_requirement=fault_model_requirement),
        }
        return round_count

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        """Reject a Stim stream whose runtime length differs from its finite circuit."""
        stream_model = self._stream_models.get(stream_op.id)
        if stream_model is None:
            return

        finite_round_count = stream_model["round_count"]
        if stream_round_count == finite_round_count:
            return

        raise RuntimeError(
            f"{stream_op.name} sealed at {stream_round_count} rounds, but its Stim "
            f"circuit was registered for {finite_round_count} rounds. Real-syndrome "
            "live streams need an exact finite circuit. Use a timing-only stream for "
            "unknown feedback length, or build the Stim circuit after the stream length "
            "is known.")

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int, *, fault_model_requirement,
                                    fault_exclusion_ranges: tuple,
                                    window_protocol) -> list:
        """Build detector error models for one finite Stim operation."""
        if op.circuit is None or not windows:
            return []

        from ..detector_error_model.window_model_builders import (
            build_window_error_models,
        )

        detector_rounds = self._source_rounds(
            self._key(op), op.circuit, round_count)
        model_plan = [
            (window.start_round, window.commit_lo, window.commit_hi,
             min(window.buffer_hi, round_count))
            for window in windows
        ]
        local_index = {
            window.key: window_index
            for window_index, window in enumerate(windows)
        }
        dependency_edges = tuple(
            (local_index[dependency], destination_index)
            for destination_index, window in enumerate(windows)
            for dependency in window.deps
            if dependency in local_index
        )
        return build_window_error_models(
            op.circuit,
            model_plan,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges,
            dependency_edges=dependency_edges,
            closed_temporal_boundary_windows=tuple(
                window_index
                for window_index, window in enumerate(windows)
                if window.closed_temporal_boundaries
            ),
            window_protocol=window_protocol,
        )

    def window_model_for_stream(self, stream_id, window):
        """Build the detector error model for one dynamic stream window; the
        window whose commit region reaches the stream's last round is the
        terminal one."""
        stream_model = self._stream_models.get(stream_id)
        if stream_model is None:
            return None

        buffer_lo = window.start_round
        return stream_model["slicer"].slice_window(
            buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi,
            is_last=window.commit_hi == stream_model["round_count"])

    def strong_window_model_for_operation(self, op: Operation, window, round_count: int,
                                          *, fault_model_requirement,
                                          exclude_faults_touching=None):
        """Build an independent two-sided context model for a strong re-decode
        with one optional inclusive range assigned to another seam side."""
        if op.circuit is None:
            return None

        from ..detector_error_model.window_model_builders import (
            build_single_window_error_model,
        )

        detector_rounds = self._source_rounds(
            self._key(op), op.circuit, round_count)
        return build_single_window_error_model(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            exclude_faults_touching=exclude_faults_touching)

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        fault_model_requirement, fault_exclusion_ranges: tuple,
    ):
        """Build a strong model with multiple non-owned inclusive ranges."""
        if op.circuit is None:
            return None

        from ..detector_error_model.window_model_builders import (
            build_single_window_error_model_with_exclusions,
        )

        detector_rounds = self._source_rounds(
            self._key(op), op.circuit, round_count)
        return build_single_window_error_model_with_exclusions(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges)


class RecordedStimDevice(StimDevice):
    """Replay recorded raw measurements (hardware data) instead of sampling.

    ``measurements`` is a (shots, measurements) bool array in the measurement
    order of ``op.circuit`` (the measurements.b8 of a released experiment);
    ``shot`` selects the row. Detection events, observable truth, round
    chronology and window error models come from the circuit exactly as for
    sampled data.
    """

    def __init__(self, measurements, shot: int, **kwargs):
        super().__init__(**kwargs)
        self.measurements = measurements
        self.shot = shot

    @staticmethod
    def detector_rounds_from_coordinates(circuit, round_count: int) -> dict:
        """One-based emitted round per detector for hardware circuits whose
        detector coordinates are concatenated (x, y, t) triples (Google's
        released memory experiments): the detector belongs to its latest t."""
        rounds = {}
        for detector_id, coords in circuit.get_detector_coordinates().items():
            layer = int(max(coords[2::3]))
            rounds[detector_id] = round_count if layer >= round_count else layer + 1
        return rounds

    def _measurement_row(self, key, sampler) -> tuple[int, ...]:
        return tuple(int(bit) for bit in self.measurements[self.shot])
