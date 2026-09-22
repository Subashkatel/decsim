"""Execute a repeated Stim memory only when the QPU requests a round.

Stim.TableauSimulator.do retains the quantum state and measurement record.
The controller chooses the final round; model lookahead never executes it.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import stim

import decsim.config as config
import decsim.detector_error_model.detector_formation as formation
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.qpu.stim_device as stim_device
import decsim.qpu.stim_stream_models as stream_models
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

# Any denotes opaque operation, stream and patch identities in port signatures.


class StreamingStimDevice(seeding._AtomicRunSeedConsumer):
    """Keep one physical memory history until its actual final readout.

    Programs are keyed by dynamic-stream owner. Only trailing-buffer feedback
    is supported. The source retains the executed circuit and raw record for
    reproduction, so host memory grows with the actual number of rounds.
    shot_sampled fires once at finalization with that complete circuit.
    """

    operation_circuit_scope = "none"

    def __init__(
        self,
        programs: Mapping[Any, circuit_records.RepeatedStimCircuit],
        seed: Optional[int] = None,
    ) -> None:
        self._seed = stim_device.validated_seed(seed)
        self._initialize_run_seed_binding(self._seed)
        self._programs = _copied_programs(programs)
        self._streams_by_id: dict[Any, _Stream] = {}
        self._models_by_stream_id: dict[
            Any, stream_models.GrowingStimModels
        ] = {}
        self._stream_id_by_operation: dict = {}
        self.shot_sampled = trace_source.TraceSource()

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> Optional[int]:
        """Create a physical history independently of decoder models."""
        del round_count
        program = self._program_for(stream_operation)
        stream_id = stream_operation.id
        seed = _stream_seed(self._seed, stream_id)
        history = _History(seed)
        period_ticks = _declared_round_period(program)
        self._streams_by_id[stream_id] = _Stream(
            stream_operation, program, history, period_ticks
        )
        return None

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> Optional[int]:
        """Create model state without declaring or resetting physical state."""
        del round_count
        program = self._program_for(stream_operation)
        self._models_by_stream_id[stream_operation.id] = (
            stream_models.GrowingStimModels(program, fault_model_requirement)
        )
        return None

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
        *,
        round_period_ticks: int,
    ) -> None:
        """Bind a segment to its live stream without sampling ahead."""
        del segment_round_count
        del source_round_count
        stream_id = _stream_id_of(operation)
        if stream_id not in self._streams_by_id:
            raise ValueError("live Stim memory requires a registered stream")
        stream = self._streams_by_id[stream_id]
        segment_patches = set(operation.patches)
        owner_patches = set(stream.owner.patches)
        if segment_patches != owner_patches:
            raise ValueError("live Stim segment patches must match its owner")
        _check_round_period(stream, round_period_ticks)
        self._stream_id_by_operation[operation.id] = stream_id

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """Execute a segment round without introducing a physical boundary."""
        stream_id = _stream_id_of(operation)
        stream_offset = operation.stream_offset
        if stream_offset is None:
            stream_offset = 0
        global_round = stream_offset + round_index
        return self._emit(stream_id, global_round, False)

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """Execute protection or final readout at the QPU's declared cadence."""
        del operation
        stream = self._stream_for(stream_id)
        _check_round_period(stream, round_period_ticks)
        return self._emit(stream_id, global_round, is_final)

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Refuse a separate fragment after the final round's folded readout."""
        del operation
        del source_round_count
        raise ValueError("live Stim memory folds readout into its final round")

    def form_round(
        self, operation_id: Any, round_index: int, raw_bits: Sequence[int]
    ) -> tuple[int, ...]:
        """Form arrived records using retained measurements across rounds."""
        stream = self._stream_for(operation_id)
        table = stream.history.table
        assert table is not None, "formation follows physical execution"
        if stream.former is None:
            stream.former = formation.StreamingDetectorFormer(table)
        else:
            stream.former.extend_table(table)
        events, _ = stream.former.feed_packet(round_index, raw_bits)
        return tuple(value for _, value in events)

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """Return truth only after the actual destructive readout executes."""
        stream_id = self._stream_id_by_operation.get(operation_id, operation_id)
        stream = self._streams_by_id.get(stream_id)
        if stream is None:
            return None
        if not stream.history.is_final:
            return None
        _, truth = stream.history.formed_shot()
        if not truth:
            return None
        return truth

    def sampled_detection_events(self, operation_id: Any) -> tuple[int, ...]:
        """Convert the executed prefix, without hypothetical future rounds."""
        stream = self._stream_for(operation_id)
        events, _ = stream.history.formed_shot()
        return events

    def sampled_measurements(self, operation_id: Any) -> tuple[int, ...]:
        """Copy the executed raw record in absolute measurement order."""
        stream = self._stream_for(operation_id)
        measurements = stream.history.simulator.current_measurement_record()
        return tuple(int(bit) for bit in measurements)

    def executed_circuit(self, stream_id: Any) -> stim.Circuit:
        """Copy the physical history for independent replay."""
        stream = self._stream_for(stream_id)
        return stream.history.circuit.copy()

    def measurement_rounds_for_stream(self, stream_id: Any) -> dict[int, int]:
        """Copy the explicit schedule needed to replay the executed circuit."""
        stream = self._stream_for(stream_id)
        return dict(stream.history.measurement_rounds)

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Require the sealed length to include the actual final readout."""
        stream = self._stream_for(stream_operation.id)
        if not stream.history.is_final:
            raise RuntimeError("live Stim stream sealed without final readout")
        if stream.history.round_count != stream_round_count:
            raise RuntimeError(
                "sealed length differs from executed Stim rounds"
            )

    def window_model_for_stream(
        self, stream_id: Any, window: window_records.Window
    ) -> fault_models.WindowErrorModel:
        """Describe a decoder window without advancing physical execution."""
        models = self._models_by_stream_id[stream_id]
        return models.for_window(window)

    def finalize_stream_models(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        """Fix decoder models at the validated physical terminal boundary."""
        models = self._models_by_stream_id[stream_operation.id]
        models.finish(stream_round_count)
        return True

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
        """Refuse static windows for a memory with a runtime-selected length."""
        del operation
        del round_count
        del fault_model_requirement
        del fault_exclusion_ranges
        del window_protocol
        if windows:
            raise ValueError("live Stim memory requires dynamic stream windows")
        return []

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        exclude_faults_touching: Optional[tuple] = None,
        prior_faults: Optional[dict] = None,
    ) -> fault_models.WindowErrorModel:
        """Build a strong context with one range assigned to its neighbour."""
        exclusions = ()
        if exclude_faults_touching is not None:
            exclusions = (exclude_faults_touching,)
        return self.strong_window_model_for_operation_with_exclusions(
            operation,
            window,
            round_count,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=exclusions,
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
    ) -> fault_models.WindowErrorModel:
        """Keep strong priors in the stream's stable fault namespace."""
        del round_count
        del fault_model_requirement
        stream_id = _stream_id_of(operation)
        models = self._models_by_stream_id[stream_id]
        return models.for_strong_window(
            window, fault_exclusion_ranges, prior_faults
        )

    def _install_run_seed_state(self, prepared_state) -> None:
        self._seed = prepared_state
        self._streams_by_id.clear()
        self._models_by_stream_id.clear()
        self._stream_id_by_operation.clear()

    def _program_for(self, operation):
        if operation.feedback_boundary_mode == "measurement_closed":
            raise ValueError(
                "live Stim memory requires trailing-buffer feedback"
            )
        if not operation.patches:
            raise ValueError("live Stim memory requires a nonempty footprint")
        if operation.id not in self._programs:
            raise ValueError(f"no repeated circuit for stream {operation.id!r}")
        return self._programs[operation.id]

    def _stream_for(self, identity):
        stream_id = self._stream_id_by_operation.get(identity, identity)
        return self._streams_by_id[stream_id]

    def _emit(self, stream_id, global_round, is_final):
        stream = self._stream_for(stream_id)
        history = stream.history
        if history.is_final:
            raise RuntimeError("live Stim stream is already finalized")
        next_round = history.round_count + 1
        if global_round != next_round:
            raise RuntimeError("live Stim rounds must execute consecutively")
        fragment = stream.program.round_circuit(global_round, is_final)
        self._mark_stochastic_use()
        bits = history.append(fragment, is_final)
        if is_final:
            self._report_finished_shot(stream)
        payload = round_records.QPUReadout(
            stream_id,
            stream.owner.patches,
            global_round,
            bits=bits,
            size_bits=len(bits),
        )
        return [payload]

    def _report_finished_shot(self, stream):
        circuit = stream.history.circuit.copy()
        operation = dataclasses.replace(stream.owner, circuit=circuit)
        events, _ = stream.history.formed_shot()
        self.shot_sampled.fire(operation, events)


@dataclasses.dataclass
class _Stream:
    """The state owned by one registered source stream."""

    owner: program_records.Operation
    program: circuit_records.RepeatedStimCircuit
    history: "_History"
    round_period_ticks: Optional[int]
    former: Optional[formation.StreamingDetectorFormer] = None


class _History:
    """The physical instructions and measurements executed so far."""

    def __init__(self, seed: Optional[int]) -> None:
        self.simulator = stim.TableauSimulator(seed=seed)
        self.circuit = stim.Circuit()
        self.measurement_rounds: dict[int, int] = {}
        self.round_count = 0
        self.is_final = False
        self.table: Optional[formation.FormationTable] = None

    def append(self, fragment: stim.Circuit, is_final: bool) -> tuple[int, ...]:
        first_measurement = self.circuit.num_measurements
        self.simulator.do(fragment)
        self.circuit += fragment
        self.round_count += 1
        self.is_final = is_final
        after_last_measurement = self.circuit.num_measurements
        for index in range(first_measurement, after_last_measurement):
            self.measurement_rounds[index] = self.round_count
        self.table = formation.build_formation_table(
            self.circuit,
            self.round_count,
            measurement_rounds=self.measurement_rounds,
        )
        measurements = self.simulator.current_measurement_record()
        new_measurements = measurements[first_measurement:]
        return tuple(int(bit) for bit in new_measurements)

    def formed_shot(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.table is None:
            return (), ()
        measurements = self.simulator.current_measurement_record()
        packets = formation.split_measurements_into_packets(
            self.table, measurements
        )
        return formation.form_shot(self.table, packets)


def _copied_programs(programs):
    copied = {}
    for stream_id, program in programs.items():
        identity_type = type(stream_id)
        if identity_type not in (int, str):
            raise ValueError("live Stim stream identities must be int or str")
        copied[stream_id] = dataclasses.replace(program)
    return copied


def _stream_seed(seed, stream_id):
    if seed is None:
        return None
    kind = "string_key"
    if type(stream_id) is int:
        kind = "integer_key"
    segment = seed_records.RunSeedPathSegment(kind, stream_id)
    return seeding.derive_component_seed(seed, (segment,))


def _stream_id_of(operation):
    if operation.stream_id is not None:
        return operation.stream_id
    return operation.id


def _declared_round_period(program):
    if program.round_period_microseconds is None:
        return None
    return config.microseconds_to_ticks(program.round_period_microseconds)


def _check_round_period(stream, period_ticks):
    declared_ticks = stream.round_period_ticks
    if declared_ticks is None:
        return
    if period_ticks != declared_ticks:
        raise ValueError("physical circuit period differs from the QPU cadence")
