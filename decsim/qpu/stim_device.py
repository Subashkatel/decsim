"""The Stim syndrome source: one sampled shot, emitted as raw bits by round.

Every round carries one bit per measure qubit and the final round of a
memory circuit also carries the data-qubit readout (Stim,
src/stim/gen/gen_surface_code.cc); detection events are formed wherever
detection_events.formed_at seats the former, from the recipes
formation_table reads off the circuit. One shot is sampled per stream identity,
under that identity's substream of the root seed (seeding.substream_seed)
so a run is reproducible across processes, and reused by every segment
of the stream.
"""

import bisect
import dataclasses
import math
import numbers
from collections.abc import Mapping
from typing import Any, Optional

import numpy
import stim

import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.window_model_builders as window_models
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

# Stream ids, operation ids and patches are opaque identities chosen by
# the workload; Any stands for them in every signature below.

# The noise a burst raises, each channel named for the place Stim's
# generator puts one of its four noise parameters (Stim
# src/stim/gen/circuit_gen_params.cc): gate is after_clifford_depolarization
# after a unitary (append_unitary_1, append_unitary_2), idle is
# before_round_data_depolarization after the round's first TICK
# (append_begin_round_tick), measurement is before_measure_flip_probability
# before a measurement (append_measure) and reset is
# after_reset_flip_probability after a reset (append_reset,
# append_measure_reset).
BURST_CHANNELS = ("gate", "idle", "measurement", "reset")


class StimDevice(seeding._AtomicRunSeedConsumer):
    """Streams one sampled Stim shot as raw measurement packets, by round.

    The model maps are keyed by stream identity and use one-based
    emitted rounds. detector_rounds says which round each detector belongs
    to when the circuit does not follow Stim's generator layout;
    measurement_rounds declares the QPU's packet schedule for the same
    reason. A non-empty terminal_detector_ids entry says the stream's
    final data readout arrives as its own fragment, through
    finalize_stream_round. Readout partitions are keyed by the emitting
    operation id, then by the one-based stream round. Separate syndrome
    and data emitters can therefore declare different acquisition groups.

    Trace source: shot_sampled(operation, detection_events) once per
    fresh shot, with the whole-circuit detection events a reference
    decode reads.
    """

    operation_circuit_scope = "per_operation"
    # the circuit states every round's width
    takes_code_card = False

    def __init__(
        self,
        seed: Optional[numbers.Integral] = None,
        detector_rounds: Optional[dict] = None,
        terminal_detector_ids: Optional[dict] = None,
        measurement_rounds: Optional[dict] = None,
        readout_partitions: Optional[dict] = None,
    ) -> None:
        self._seed = validated_seed(seed)
        self._initialize_run_seed_binding(self._seed)
        detector_rounds_override = _rounds_by_key(detector_rounds)
        terminal_ids = _detector_ids_by_key(terminal_detector_ids)
        measurement_rounds_override = _rounds_by_key(measurement_rounds)
        self._readout_partitions = _partitions_by_key(readout_partitions)
        self._shots = _ShotTable(
            detector_rounds_override=detector_rounds_override,
            terminal_detector_ids=terminal_ids,
            measurement_rounds_override=measurement_rounds_override,
        )
        self.shot_sampled = trace_source.TraceSource()

    def sampled_truth(self) -> dict:
        """Every sampled observable-flip vector, by operation or stream."""
        truth_by_key = {}
        for key, shot in self._shots.shot_by_key.items():
            truth_by_key[key] = _as_int_bits(shot.truth)
        operation_ids = self._shots.sample_key_by_operation_id.items()
        for operation_id, key in operation_ids:
            if operation_id in truth_by_key:
                continue
            shot = self._shots.shot_by_key[key]
            truth_by_key[operation_id] = _as_int_bits(shot.truth)
        return truth_by_key

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """Stim's sampled observable-flip vector, when the shot exists."""
        shot = self._shot_for(operation_id)
        if shot is None:
            return None
        return _as_int_bits(shot.truth)

    def readout_departure_tick(
        self, readout: round_records.QPUReadout, readout_tick: int
    ) -> int:
        """The readout leaves the chip at the boundary it was read out at."""
        del readout
        return readout_tick

    def window_model_source(self) -> "StimDevice":
        """This source: the circuit it samples is the window models' too."""
        return self

    def sampled_detection_events(self, operation_id: Any) -> tuple[bool, ...]:
        """The shot's detection events, in the circuit's detector order."""
        shot = self._shot_for(operation_id)
        if shot is None:
            raise KeyError(
                f"no sampled detection events for identity {operation_id!r}; "
                "the operation has not begun"
            )
        return tuple(bool(bit) for bit in shot.detection_events)

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
        *,
        round_period_ticks: int,
    ) -> None:
        """Sample one fresh shot, or reuse the stream's shot for a segment."""
        del round_period_ticks
        if operation.circuit is None:
            raise ValueError("StimDevice operations require a circuit")
        _check_segment(operation, segment_round_count, source_round_count)
        key = program_records.decode_identity(operation)
        if self._seed is not None:
            _check_sample_key(key)
        detector_rounds = self._bind_source(
            key, operation.circuit, source_round_count
        )
        is_stream = operation.stream_id is not None
        is_later_segment = is_stream and operation.stream_offset > 0
        if is_later_segment:
            # A later segment reads its stream's shot under its own id.
            assert key in self._shots.shot_by_key, (
                "a segment begins after its head"
            )
            self._shots.sample_key_by_operation_id[operation.id] = key
            return
        assert key not in self._shots.shot_by_key, f"{key!r} has already begun"
        self._sample_shot(key, operation, source_round_count, detector_rounds)

    def formation_table(
        self, operation_id: Any
    ) -> detector_formation.FormationTable:
        """The recipes the operation's rounds are formed by, off its circuit."""
        shot = self._shot_for(operation_id)
        if shot is None:
            raise KeyError(
                f"no detector formation table for identity {operation_id!r}; "
                "the operation has not begun"
            )
        return shot.table

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """This operation round in its declared raw measurement partitions."""
        key = program_records.decode_identity(operation)
        global_round = program_records.global_round(operation, round_index)
        bits = self._round_packet_bits(key, global_round)
        return self._readouts(key, operation, global_round, bits)

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """The stream's final data readout as its own raw fragment.

        The refusals are the contract with the frontend that declared the
        stream: a wrong declaration would stamp another round's bits as
        the final readout.
        """
        key = program_records.decode_identity(operation)
        shot = self._shots.shot_by_key[key]
        binding = self._shots.source_binding_by_key[key]
        circuit_text = str(operation.circuit)
        if circuit_text != binding.circuit_text:
            raise RuntimeError(
                "finalizer circuit differs from its source binding"
            )
        final_round = operation.stream_offset + 1
        if final_round != source_round_count:
            raise RuntimeError("finalizer is not at the final source round")
        if not self._shots.terminal_detector_ids.get(key):
            raise RuntimeError(
                "terminal finalizer has no declared detector ids"
            )
        table = shot.table
        if table.readout_slot_start is None:
            raise RuntimeError("terminal finalizer has no folded readout bits")
        final_packet = shot.packets[table.round_count]
        bits = final_packet[table.readout_slot_start :]
        return self._readouts(key, operation, final_round, bits)

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """This idle stream round in the owner's measurement partitions."""
        del is_final
        del round_period_ticks
        binding = self._shots.source_binding_by_key[stream_id]
        if not 1 <= global_round <= binding.round_count:
            raise ValueError("idle round is outside the finite source")
        bits = self._round_packet_bits(stream_id, global_round)
        return self._readouts(stream_id, operation, global_round, bits)

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> Optional[int]:
        """Bind the finite physical circuit without sampling a shot."""
        if stream_operation.circuit is None:
            return None
        self._bind_source(
            stream_operation.id, stream_operation.circuit, round_count
        )
        return round_count

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
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
        self._shots.stream_model_by_id[stream_operation.id] = stream_model
        return round_count

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Refuse a stream whose runtime length differs from its circuit."""
        binding = self._shots.source_binding_by_key.get(stream_operation.id)
        if binding is None:
            return
        _check_finite_stream_length(
            stream_operation, stream_round_count, binding.round_count
        )

    def finalize_stream_models(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        """A finite model's terminal boundary is fixed at registration.

        A seal at another length stops the run where the decoder memory
        checks each input against its window model's rows.
        """
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
        """The detector error models of one finite Stim operation's windows."""
        if operation.circuit is None or not windows:
            return []
        key = program_records.decode_identity(operation)
        detector_rounds = self._bind_source(key, operation.circuit, round_count)
        model_plan = [_window_span(window) for window in windows]
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
        self, stream_id: Any, window: window_records.Window
    ) -> Optional[fault_models.WindowErrorModel]:
        """The detector error model of one dynamic stream window.

        The window whose commit region reaches the stream's last round is
        the terminal one.
        """
        stream_model = self._shots.stream_model_by_id.get(stream_id)
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
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        exclude_faults_touching: Optional[tuple] = None,
        prior_faults: Optional[dict] = None,
    ) -> Optional[fault_models.WindowErrorModel]:
        """An independent window model for a strong re-decode.

        One optional inclusive range is assigned to another seam side,
        and a pinned face's neighbour supplies the faults it has already
        committed, which are no columns of this model.
        """
        if operation.circuit is None:
            return None
        key = program_records.decode_identity(operation)
        detector_rounds = self._bind_source(key, operation.circuit, round_count)
        span = _window_span(window)
        return window_models.build_single_window_error_model(
            operation.circuit,
            span,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            exclude_faults_touching=exclude_faults_touching,
            prior_faults=prior_faults,
        )

    def strong_window_model_for_operation_with_exclusions(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict] = None,
    ) -> Optional[fault_models.WindowErrorModel]:
        """A strong re-decode model with several non-owned inclusive ranges."""
        if operation.circuit is None:
            return None
        key = program_records.decode_identity(operation)
        detector_rounds = self._bind_source(key, operation.circuit, round_count)
        span = _window_span(window)
        return window_models.build_single_window_error_model_with_exclusions(
            operation.circuit,
            span,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges,
            prior_faults=prior_faults,
        )

    def _prepare_run_seed_state(self, effective_seed):
        return effective_seed

    def _install_run_seed_state(self, prepared_state) -> None:
        """A fresh run keeps the declarations and drops what it sampled."""
        self._seed = prepared_state
        self._shots = dataclasses.replace(
            self._shots,
            shot_by_key={},
            sample_key_by_operation_id={},
            stream_model_by_id={},
            source_binding_by_key={},
        )

    def _sampler_for(
        self, key, circuit: stim.Circuit
    ) -> stim.CompiledMeasurementSampler:
        if self._seed is None:
            return circuit.compile_sampler()
        sample_seed = seeding.substream_seed(self._seed, (key,))
        return circuit.compile_sampler(seed=sample_seed)

    def _readouts(self, key, operation, round_index, bits):
        patches = program_records.patches_of(operation)
        readout = round_records.QPUReadout(
            key, patches, round_index, bits=bits, size_bits=len(bits)
        )
        by_round = self._readout_partitions.get(operation.id, {})
        partitions = by_round.get(round_index, ())
        return round_records.partition_measurements(readout, partitions)

    def _sample_shot(
        self,
        key,
        operation: program_records.Operation,
        source_round_count: int,
        detector_rounds: dict,
    ) -> None:
        measurement_rounds = self._shots.measurement_rounds_override.get(key)
        table = detector_formation.build_formation_table(
            operation.circuit,
            source_round_count,
            measurement_rounds=measurement_rounds,
            detector_rounds=detector_rounds,
        )
        sampled_circuit = self._sampled_circuit(operation.circuit, table)
        sampler = self._sampler_for(key, sampled_circuit)
        self._mark_stochastic_use()
        measurement_row = self._measurement_row(sampler)
        packets = detector_formation.split_measurements_into_packets(
            table, measurement_row
        )
        # The whole-shot formation is the oracle for the truth and for
        # sampled_detection_events; each seat that forms the rounds runs
        # the same table one packet at a time.
        formed_events, formed_truth = detector_formation.form_shot(
            table, packets
        )
        shot = _SampledShot(packets, table, formed_events, formed_truth)
        self._shots.shot_by_key[key] = shot
        self._shots.sample_key_by_operation_id[operation.id] = key
        self.shot_sampled.fire(operation, formed_events)

    def _sampled_circuit(
        self, circuit: stim.Circuit, table: detector_formation.FormationTable
    ) -> stim.Circuit:
        """The circuit the shot is drawn from: the operation's own."""
        del table
        return circuit

    def _measurement_row(
        self, sampler: stim.CompiledMeasurementSampler
    ) -> tuple[int, ...]:
        """One shot of raw measurement bits in circuit measurement order."""
        shots = sampler.sample(shots=1)
        row = shots[0]
        return _as_int_bits(row)

    def _shot_for(self, identity):
        """The shot behind a replaying operation id, or behind a sample key."""
        key = self._shots.sample_key_by_operation_id.get(identity, identity)
        return self._shots.shot_by_key.get(key)

    def _round_packet_bits(self, key, global_round: int) -> tuple[int, ...]:
        """The raw bits the QPU emits for one round.

        When the stream declares a terminal fragment, the folded readout
        bits of the final round stay behind for finalize_stream_round.
        """
        shot = self._shots.shot_by_key[key]
        packet = shot.packets[global_round]
        table = shot.table
        terminal_ids = self._shots.terminal_detector_ids.get(key)
        is_final_round = global_round == table.round_count
        has_terminal_fragment = bool(terminal_ids)
        has_folded_readout = table.readout_slot_start is not None
        readout_arrives_separately = is_final_round and has_terminal_fragment
        if readout_arrives_separately and has_folded_readout:
            return packet[: table.readout_slot_start]
        return packet

    def _bind_source(
        self, key, circuit: stim.Circuit, source_round_count: int
    ) -> dict:
        """Bind one finite circuit, duration and detector chronology per key."""
        detector_rounds_override = self._shots.detector_rounds_override.get(key)
        resolved = detector_chronology.resolve_detector_rounds(
            circuit, detector_rounds_override, source_round_count
        )
        circuit_text = str(circuit)
        binding = self._shots.source_binding_by_key.get(key)
        if binding is None:
            detector_rounds = dict(resolved)
            self._shots.source_binding_by_key[key] = _SourceBinding(
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

    def __init__(
        self,
        measurements: numpy.ndarray,
        shot: int,
        seed: Optional[numbers.Integral] = None,
        detector_rounds: Optional[dict] = None,
        terminal_detector_ids: Optional[dict] = None,
        measurement_rounds: Optional[dict] = None,
        readout_partitions: Optional[dict] = None,
    ) -> None:
        StimDevice.__init__(
            self,
            seed=seed,
            detector_rounds=detector_rounds,
            terminal_detector_ids=terminal_detector_ids,
            measurement_rounds=measurement_rounds,
            readout_partitions=readout_partitions,
        )
        self.measurements = measurements
        self.shot = shot

    @staticmethod
    def detector_rounds_from_coordinates(
        circuit: stim.Circuit, round_count: int
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

    def _measurement_row(
        self, sampler: stim.CompiledMeasurementSampler
    ) -> tuple[int, ...]:
        del sampler
        row = self.measurements[self.shot]
        return _as_int_bits(row)


class BurstStimDevice(StimDevice):
    """Samples every shot with one error burst the decoders are not told of.

    The shot is drawn from burst_circuit, the operation's circuit with the
    burst's extra noise; detection events are formed, and window models
    built, from the operation's own circuit, as on hardware, where the
    decoders' error model is the calibrated one. The public
    qec-burst-scaling code samples its bursts the same way, with the
    decoder weights from the background circuit (qecburst/circuit.py
    inject_burst_profile; qecburst/simulate.py). One burst per shot:
    bursts come every 10 s on Sycamore (McEwen 2104.05219, "λ = 1/(10
    s)") and about once an hour on Willow (2408.13687, "once every
    hour"), far apart next to a shot, so the rate joins when the shots
    are read, as Q3DE weighs a burst shot's logical error by the time
    bursts take up (2501.00331, equation (1)).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The burst: its rise and decay, where, how strong, which noise.

        burst_onset_round is the first one-based round with extra noise.
        It climbs to burst_error_probability over burst_rise_rounds,
        (i + 1) / rise of it in the i-th round from the onset, then
        decays from that peak as exp(-(rounds since the peak) /
        burst_decay_rounds), the exponential recovery McEwen measures ("a
        typical ~25 ms exponential decay", 2104.05219) and
        qec-burst-scaling samples (qecburst/circuit.py
        exponential_decay_profile); a burst_decay_rounds of None holds the
        peak's probability to the shot's end. A rise of 1 is a step,
        McEwen's and qec-burst-scaling's shape; the six largest bursts in
        the repetition-code data released with 2408.13687 peak about 3
        rounds after their onset, as Kurilovich's T1 transient of about
        10 us would (2506.18228 lines 312-315). The region is every qubit
        whose first two Stim coordinates lie within burst_radius of
        burst_center, the midpoint of the qubits' coordinates when None;
        a burst_radius of None is every qubit. burst_channels names the
        noise it raises (BURST_CHANNELS). A burst_error_probability of 0
        is no burst: the shot is drawn from the operation's own circuit.
        """

        burst_onset_round: int = 1
        burst_rise_rounds: int = 1
        burst_decay_rounds: Optional[float] = None
        burst_radius: Optional[float] = None
        burst_center: Optional[tuple] = None
        burst_error_probability: float = 0.0
        burst_channels: tuple = BURST_CHANNELS

        def __post_init__(self) -> None:
            _check_onset_round(self.burst_onset_round)
            _check_rise_rounds(self.burst_rise_rounds)
            _check_decay_rounds(self.burst_decay_rounds)
            _check_radius(self.burst_radius)
            _check_center(self.burst_center)
            _check_burst_probability(self.burst_error_probability)
            _check_channels(self.burst_channels)

        @classmethod
        def from_yaml(cls, section: Mapping) -> "BurstStimDevice.Settings":
            """The qpu section's burst keys; an absent key is its default.

            A yaml list, the centre or the channels, reads as a tuple.
            """
            values = dict(section)
            for key in ("burst_center", "burst_channels"):
                value = values.get(key)
                if isinstance(value, list):
                    values[key] = tuple(value)
            return cls(**values)

    def __init__(
        self,
        settings: Optional[Settings] = None,
        seed: Optional[numbers.Integral] = None,
        detector_rounds: Optional[dict] = None,
        terminal_detector_ids: Optional[dict] = None,
        measurement_rounds: Optional[dict] = None,
        readout_partitions: Optional[dict] = None,
    ) -> None:
        StimDevice.__init__(
            self,
            seed=seed,
            detector_rounds=detector_rounds,
            terminal_detector_ids=terminal_detector_ids,
            measurement_rounds=measurement_rounds,
            readout_partitions=readout_partitions,
        )
        if settings is None:
            settings = BurstStimDevice.Settings()
        self.burst = settings

    def _sampled_circuit(
        self, circuit: stim.Circuit, table: detector_formation.FormationTable
    ) -> stim.Circuit:
        """The burst circuit; the operation's own when there is no burst."""
        if self.burst.burst_error_probability == 0:
            return circuit
        return burst_circuit(circuit, table, self.burst)


def validated_seed(seed) -> Optional[int]:
    """The seed under Stim's public unsigned 64-bit contract, or None.

    Every Stim-backed source draws under the same contract.
    """
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


def burst_circuit(
    circuit: stim.Circuit,
    table: detector_formation.FormationTable,
    burst: BurstStimDevice.Settings,
) -> stim.Circuit:
    """The flattened circuit with one burst's extra noise in its rounds.

    A round is decsim's own: the instructions after the previous round's
    last measurement through its own last one, as the formation table's
    packet widths lay the measurements out, so no TICK count is assumed.
    In a burst round every noise instruction of a named channel that
    touches the region is preceded by a copy of itself on the region's
    targets at the round's extra probability; a two-qubit channel keeps a
    pair when either qubit is in the region. The idle copy lands right
    after the round's first TICK, where qec-burst-scaling inserts its
    one DEPOLARIZE1 per round (qecburst/circuit.py inject_burst_profile).
    A patch the region misses is returned as it is, so its shot is the
    one stim_device draws. A region on the patch with none of the named
    noise to raise is refused: qec-burst-scaling locates its burst on the
    background noise too, and refuses a circuit without it
    (qecburst/geometry.py get_data_qubits, "Ensure p0 > 0";
    circuit.py build_background_circuit, "expects 0 < p0").

    Raises:
        ValueError: the burst starts after the shot's last round, or its
            region covers the patch and finds no noise of its channels,
            so no shot would run with the burst the settings name.
    """
    if burst.burst_onset_round > table.round_count:
        _refuse_a_late_onset(burst, table.round_count)
    region = _burst_region(circuit, burst)
    round_ends = _round_ends(table)
    flattened = circuit.flattened()
    instructions = list(flattened)
    sampled = stim.Circuit()
    inserted_count = 0
    segment_start = 0
    measurement_count = 0
    for index, instruction in enumerate(instructions):
        round_index = _round_of(round_ends, measurement_count)
        probability = _burst_probability(burst, round_index)
        copy = _burst_copy(instructions, index, region, probability, burst)
        measurement_count += instruction.num_measurements
        if copy is None:
            continue
        # A slice joins the circuit in Stim's own code; appending each
        # instruction from Python costs ten times as long on a
        # 2,000-round memory.
        sampled += flattened[segment_start:index]
        sampled.append(copy)
        inserted_count += 1
        segment_start = index
    if inserted_count > 0:
        sampled += flattened[segment_start:]
        return sampled
    if region:
        _refuse_a_burst_with_no_noise_to_raise(burst)
    return circuit


@dataclasses.dataclass(frozen=True)
class _SampledShot:
    """One shot of a circuit: its packets and formed events."""

    packets: dict
    table: detector_formation.FormationTable
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


def _check_finite_stream_length(
    operation, actual_round_count, expected_round_count
):
    if actual_round_count == expected_round_count:
        return
    raise RuntimeError(
        f"{operation.name} sealed at {actual_round_count} rounds, "
        f"but its Stim circuit was registered for {expected_round_count} "
        "rounds. Real-syndrome live streams need an exact finite "
        "circuit. Use a timing-only stream for unknown feedback length, "
        "or build the Stim circuit after the stream length is known."
    )


def _check_sample_key(key) -> None:
    """Refuse an identity whose equality could alias a legal cache key."""
    key_type = type(key)
    if key_type not in (int, str):
        raise ValueError(
            f"stream_id must be an int or str so the run's sampling is "
            f"reproducible across processes; sample key {key!r} is a "
            f"{key_type.__name__}"
        )


def _check_segment(
    operation: program_records.Operation,
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


def _window_span(window: window_records.Window) -> tuple:
    return (
        window.start_round,
        window.commit_lo,
        window.commit_hi,
        window.buffer_hi,
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


@dataclasses.dataclass(frozen=True)
class _ShotTable:
    """Every per-shot and per-stream map of one device, as one member.

    A shot sits under its sample key: the stream id, or the operation id
    of a standalone operation. An operation that replays a stream's shot
    is known by its own id, which maps to the sample key. The three
    override maps are the caller's declarations about the circuit; the
    four registries are what sampling fills in. Grouping them is gem5's
    move for a component's many members
    (gem5 src/base/stats/group.hh:60-92).
    """

    detector_rounds_override: dict
    terminal_detector_ids: dict
    measurement_rounds_override: dict
    shot_by_key: dict = dataclasses.field(default_factory=dict)
    sample_key_by_operation_id: dict = dataclasses.field(default_factory=dict)
    stream_model_by_id: dict = dataclasses.field(default_factory=dict)
    source_binding_by_key: dict = dataclasses.field(default_factory=dict)


def _partitions_by_key(declared: Optional[dict]) -> dict:
    declared = declared or {}
    copied = {}
    for key, by_round in declared.items():
        copied[key] = {
            index: tuple(partitions) for index, partitions in by_round.items()
        }
    return copied


def _rounds_by_key(declared: Optional[dict]) -> dict:
    """One rounds map per stream key, copied out of the caller's mapping."""
    declared = declared or {}
    copied = {}
    for key, rounds_map in declared.items():
        copied[key] = dict(rounds_map)
    return copied


def _detector_ids_by_key(declared: Optional[dict]) -> dict:
    """One terminal detector tuple per stream key."""
    declared = declared or {}
    copied = {}
    for key, detector_ids in declared.items():
        copied[key] = tuple(detector_ids)
    return copied


def _check_onset_round(value) -> None:
    is_count = isinstance(value, int) and not isinstance(value, bool)
    if is_count and value >= 1:
        return
    raise ValueError(
        f"qpu.burst_onset_round is a one-based round, at least 1; got {value!r}"
    )


def _check_rise_rounds(value) -> None:
    is_count = isinstance(value, int) and not isinstance(value, bool)
    if is_count and value >= 1:
        return
    raise ValueError(
        f"qpu.burst_rise_rounds is a whole number of rounds, at least 1 "
        f"(1 is a step); got {value!r}"
    )


def _check_decay_rounds(value) -> None:
    if value is None:
        return
    if _is_finite_number(value) and value > 0:
        return
    raise ValueError(
        f"qpu.burst_decay_rounds is a number of rounds above 0 or null; "
        f"got {value!r}"
    )


def _check_radius(value) -> None:
    """Zero is one qubit's burst, the TLS case of 2408.13687."""
    if value is None:
        return
    if _is_finite_number(value) and value >= 0:
        return
    raise ValueError(
        f"qpu.burst_radius is a distance of 0 or more in Stim qubit "
        f"coordinates, or null; got {value!r}"
    )


def _check_center(value) -> None:
    if value is None:
        return
    is_pair = isinstance(value, tuple) and len(value) == 2
    if is_pair and all(_is_finite_number(number) for number in value):
        return
    raise ValueError(
        f"qpu.burst_center is [x, y] in Stim qubit coordinates or null; "
        f"got {value!r}"
    )


def _check_burst_probability(value) -> None:
    """At most 3/4, the largest DEPOLARIZE1 probability Stim accepts."""
    if _is_finite_number(value) and 0 <= value <= 0.75:
        return
    raise ValueError(
        "qpu.burst_error_probability is a number from 0 to 0.75 (YAML "
        f"reads 1e-3 as text; write 1.0e-3); got {value!r}"
    )


def _check_channels(value) -> None:
    is_named = isinstance(value, tuple) and len(value) > 0
    if is_named and set(value) <= set(BURST_CHANNELS):
        return
    raise ValueError(
        f"qpu.burst_channels is a non-empty list drawn from "
        f"{list(BURST_CHANNELS)}; got {value!r}"
    )


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _burst_region(
    circuit: stim.Circuit, burst: BurstStimDevice.Settings
) -> frozenset:
    """The qubits within burst_radius of the centre, or every qubit."""
    if burst.burst_radius is None:
        every_qubit = range(circuit.num_qubits)
        return frozenset(every_qubit)
    coordinates = circuit.get_final_qubit_coordinates()
    center = burst.burst_center
    if center is None:
        center = _coordinate_midpoint(coordinates)
    region = set()
    for qubit, position in coordinates.items():
        position_x, position_y = _planar(position)
        offset_x = position_x - center[0]
        offset_y = position_y - center[1]
        distance = math.hypot(offset_x, offset_y)
        if distance <= burst.burst_radius:
            region.add(qubit)
    return frozenset(region)


def _coordinate_midpoint(coordinates: dict) -> tuple:
    """The centre of the box the qubits' first two coordinates span."""
    positions = [_planar(position) for position in coordinates.values()]
    x_values = [position_x for position_x, _ in positions]
    y_values = [position_y for _, position_y in positions]
    middle_x = (min(x_values) + max(x_values)) / 2
    middle_y = (min(y_values) + max(y_values)) / 2
    return (middle_x, middle_y)


def _planar(position) -> tuple:
    """A qubit's first two coordinates; a line's qubits sit at y = 0."""
    if len(position) < 2:
        return (position[0], 0.0)
    return (position[0], position[1])


def _round_ends(table: detector_formation.FormationTable) -> list:
    """The measurement count at each round's end, round 1 first."""
    ends = []
    total = 0
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        total += table.packet_width_by_round[round_index]
        ends.append(total)
    return ends


def _round_of(round_ends: list, measurement_count: int) -> int:
    """The round of the instruction that follows measurement_count records.

    Its round is the first whose end lies beyond them; what follows the
    last measurement stays in the last round.
    """
    rounds_ended = bisect.bisect_right(round_ends, measurement_count)
    round_index = rounds_ended + 1
    return min(round_index, len(round_ends))


def _burst_probability(burst: BurstStimDevice.Settings, round_index) -> float:
    """The extra probability in one round, zero before the onset."""
    rounds_since_onset = round_index - burst.burst_onset_round
    if rounds_since_onset < 0:
        return 0.0
    rounds_climbed = rounds_since_onset + 1
    climbed_share = rounds_climbed / burst.burst_rise_rounds
    rise_share = min(climbed_share, 1.0)
    amplitude = burst.burst_error_probability * rise_share
    if burst.burst_decay_rounds is None:
        return amplitude
    rounds_past_peak = rounds_climbed - burst.burst_rise_rounds
    rounds_since_peak = max(rounds_past_peak, 0)
    exponent = -rounds_since_peak / burst.burst_decay_rounds
    decay = math.exp(exponent)
    return amplitude * decay


def _burst_copy(
    instructions: list,
    index: int,
    region: frozenset,
    probability: float,
    burst: BurstStimDevice.Settings,
) -> Optional[stim.CircuitInstruction]:
    """The copy one noise instruction gets in a burst round, or None."""
    if probability == 0:
        return None
    channel = _noise_channel(instructions, index)
    if channel not in burst.burst_channels:
        return None
    instruction = instructions[index]
    targets = _region_targets(instruction, region)
    if not targets:
        return None
    return stim.CircuitInstruction(instruction.name, targets, [probability])


def _noise_channel(instructions: list, index: int) -> Optional[str]:
    """Which of BURST_CHANNELS one instruction is, by its neighbours."""
    name = instructions[index].name
    previous_index = index - 1
    following_index = index + 1
    previous_gate = _gate_at(instructions, previous_index)
    following_gate = _gate_at(instructions, following_index)
    if name in ("DEPOLARIZE1", "DEPOLARIZE2"):
        return _depolarizing_channel(name, previous_gate)
    if name in ("X_ERROR", "Z_ERROR"):
        return _flip_channel(previous_gate, following_gate)
    return None


def _depolarizing_channel(name: str, previous_gate) -> Optional[str]:
    if previous_gate is None:
        return None
    is_round_start = previous_gate.name == "TICK"
    if is_round_start and name == "DEPOLARIZE1":
        return "idle"
    if previous_gate.is_unitary:
        return "gate"
    return None


def _flip_channel(previous_gate, following_gate) -> Optional[str]:
    if previous_gate is not None and previous_gate.is_reset:
        return "reset"
    if following_gate is not None and following_gate.produces_measurements:
        return "measurement"
    return None


def _gate_at(instructions: list, index: int) -> Optional[stim.GateData]:
    """Stim's data on the gate at index, None past either end."""
    if not 0 <= index < len(instructions):
        return None
    name = instructions[index].name
    return stim.gate_data(name)


def _region_targets(
    instruction: stim.CircuitInstruction, region: frozenset
) -> list:
    """The instruction's targets in the region; a pair goes whole."""
    targets = instruction.targets_copy()
    if instruction.name == "DEPOLARIZE2":
        return _pairs_touching(targets, region)
    return [target for target in targets if target.value in region]


def _pairs_touching(targets: list, region: frozenset) -> list:
    kept = []
    firsts = targets[::2]
    seconds = targets[1::2]
    for first, second in zip(firsts, seconds, strict=True):
        if first.value in region or second.value in region:
            kept.extend((first, second))
    return kept


def _refuse_a_burst_with_no_noise_to_raise(
    burst: BurstStimDevice.Settings,
) -> None:
    channels = " or ".join(burst.burst_channels)
    raise ValueError(
        f"the burst region covers qubits of the patch, and the circuit "
        f"has no {channels} noise on them from round "
        f"{burst.burst_onset_round}; a burst raises the circuit's own "
        "noise, so give the circuit that noise or name a channel it has "
        "in qpu.burst_channels"
    )


def _refuse_a_late_onset(
    burst: BurstStimDevice.Settings, round_count: int
) -> None:
    raise ValueError(
        f"qpu.burst_onset_round {burst.burst_onset_round} is after the "
        f"shot's last round, {round_count}; the burst would never start"
    )
