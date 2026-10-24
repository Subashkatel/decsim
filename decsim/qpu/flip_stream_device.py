"""A live Stim memory sampled a round at a time, in memory that stays flat.

The live source (streaming_stim_device.py) runs a TableauSimulator and
keeps the circuit run and every measurement, so it grows with the
rounds. This one runs Stim's FlipSimulator, which tracks the Pauli
frame against the noiseless circuit: its do() carries the frame and the
measurement record from one call to the next, so the program's first
round, then each repeated round, then its final round run one call
each (stim 1.16 FlipSimulator.do; probe p2, Adroit config_proto/probes5:
bit-identical to do() of the whole circuit under one seed).

Every rollover_rounds rounds it lets go of what it has run, keeping the
frame and the measurement flips a later round's records reach back to
(to_numpy, clear, broadcast_pauli_errors, append_measurement_flips; probe
p3: 1.0 MB over 10^5 rounds at d=11, against 796 MB without). With
stabilizer randomization off the rollover changes no bit (probe p4);
with it on, clear() draws a new random frame, so the shot differs but
is drawn from the same distribution. Ten rounds is the default because
reading a round's flips copies the record kept since the last rollover
(probe p8: 30.6 us a round at d=11 every 10 rounds, 43.5 every 100,
118.4 every round).

A drift multiplies every noise probability of the rounds after
drift_round by drift_factor: the noise moves while the decoders keep
the window models of the program as written, as a device whose error
rates drift away from its last calibration does (Toshio et al.
2510.25222 Sec. III B calibrates once). Under a noise row linear in p,
such as SD6, a factor k is the memory at k p.

A round's raw bits are its flips XOR the noiseless reference record,
read once off a short circuit of the program, as the flips are against
it (FlipSimulator: "measurement flips ... relative to the noiseless
reference"). The formation table is the template's, moved in time
(detector_error_model/periodic_formation.py). The truth is the
observables' flips, kept across rollovers. A window's model is a short
circuit's, moved in time likewise (moved_stream_models.py).

While someone listens to checkpoint_reached, each round also closes a
copy of the frame and the records the readout reads with the readout
fragment, noiseless, as Tan et al.'s copies end a run early
(2209.09219 lines 1003-1037): the stream as if it ended there
(records/checkpoints.py). Probe: the copy's readout detectors and
observables equal Stim's own circuit of that many rounds, with a
noiseless readout, bit for bit (studies a_virtual_readout_probe.py,
200 of 200 seeds).
"""

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Any, Optional

import numpy
import stim

import decsim.config as config
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.periodic_formation as periodic_formation
import decsim.ports as ports
import decsim.qpu.moved_stream_models as moved_stream_models
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.checkpoints as checkpoint_records
import decsim.records.circuits as circuit_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.formation as formation_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

# probe p8: the rollover period with the least time a round
DEFAULT_ROLLOVER_ROUNDS = 10
FRAGMENT_NAMES = (
    "first_round",
    "repeated_round",
    "final_round",
    "single_round",
)


class FlipStreamDevice(seeding.AtomicRunSeedConsumer):
    """Runs each live stream's rounds on a FlipSimulator, kept flat.

    Programs are keyed by stream, as the live source's are. It keeps no
    whole shot, so shot_sampled is silent and a reference decode of the
    whole shot has nothing to read. checkpoint_reached(snapshot) fires
    after each round while someone listens.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """rollover_rounds is how often the simulator lets go of its past.

        stabilizer_randomization draws the random outcomes of the first
        measurement of each stabilizer, as Stim does by default; off, the
        rollover changes no bit.
        """

        rollover_rounds: int = DEFAULT_ROLLOVER_ROUNDS
        stabilizer_randomization: bool = True
        drift_round: Optional[int] = None
        drift_factor: float = 1.0
        # the word the reports name this row by
        name = "flip_stream"

        def __post_init__(self) -> None:
            """Every value checked, with the sentence the caller reads."""
            config.check_whole_count(
                "flip_stream.rollover_rounds", self.rollover_rounds, "rounds"
            )
            config.check_boolean(
                "flip_stream.stabilizer_randomization",
                self.stabilizer_randomization,
            )
            if self.drift_round is not None:
                config.check_whole_count(
                    "flip_stream.drift_round", self.drift_round, "rounds"
                )
            factor = self.drift_factor
            if not config.is_number(factor) or not 0 <= factor < math.inf:
                raise ValueError(
                    "flip_stream.drift_factor must be a finite number of at "
                    f"least 0, got {factor!r}"
                )

        def build(
            self, code: ports.CodeModel, circuit_arguments: Mapping
        ) -> "FlipStreamDevice":
            """A fresh source over the workload's live fragments."""
            del code
            return FlipStreamDevice(**circuit_arguments, settings=self)

    operation_circuit_scope = "none"
    emits_bit_values = True
    shot_sampled = trace_source.SILENT
    errors_sampled = trace_source.SILENT
    readout_departure_tick = staticmethod(
        syndrome_devices.boundary_departure_tick
    )

    def __init__(
        self,
        programs: Mapping[  # opaque identity keys
            Any, circuit_records.RepeatedStimCircuit
        ],
        seed: Optional[int] = None,
        settings: Optional["FlipStreamDevice.Settings"] = None,
    ) -> None:
        self._seed = stim_device.validated_seed(seed)
        seeding.AtomicRunSeedConsumer.__init__(self, self._seed)
        if settings is None:
            settings = FlipStreamDevice.Settings()
        self.settings = settings
        self._programs = dict(programs)
        self._streams_by_id: dict = {}
        self._models_by_stream_id: dict = {}
        self.checkpoint_reached = trace_source.TraceSource()

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> Optional[int]:
        """Start the stream's simulator; its length is the controller's."""
        del round_count
        program = self._program_for(stream_operation)
        stream_id = stream_operation.id
        seed = None
        if self._seed is not None:
            seed = seeding.substream_seed(self._seed, (stream_id,))
        settings = self.settings
        drift = None
        if settings.drift_round is not None:
            drift = (settings.drift_round, settings.drift_factor)
        plan = RoundPlan(program, settings.rollover_rounds, drift)
        is_randomized = self.settings.stabilizer_randomization
        history = FlipHistory(plan, seed, is_randomized)
        self._streams_by_id[stream_id] = _Stream(stream_operation, history)
        return None

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
    ) -> Optional[int]:
        """The stream's window models, made as its windows open."""
        del round_count
        program = self._program_for(stream_operation)
        self._models_by_stream_id[stream_operation.id] = (
            moved_stream_models.MovedStimModels(
                program, fault_model_requirement
            )
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
        """A segment of a declared stream runs on its simulator."""
        del segment_round_count, source_round_count, round_period_ticks
        stream_id = program_records.decode_identity(operation)
        if stream_id not in self._streams_by_id:
            raise ValueError("a flip stream's segment needs a declared stream")

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """Run one round of the segment's stream; its raw bits."""
        stream_id = program_records.decode_identity(operation)
        global_round = program_records.global_round(operation, round_index)
        return self._emit(stream_id, global_round, False)

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,  # an opaque identity
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """Run a protection round, or the final readout when is_final."""
        del operation, round_period_ticks
        return self._emit(stream_id, global_round, is_final)

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Refuse a separate fragment: the final round folds the readout."""
        del operation, source_round_count
        raise ValueError("a flip stream folds its readout into its final round")

    def formation_table(
        self,
        stream_id: Any,  # an opaque identity
    ) -> formation_records.TableReads:
        """The recipes of the rounds run so far, moved off the template."""
        stream = self._stream_for(stream_id)
        return stream.history.table()

    def logical_observable_truth(
        self,
        operation_id: Any,  # an opaque identity
    ) -> Optional[tuple[int, ...]]:
        """The observables' flips, once the final readout has run."""
        stream = self._streams_by_id.get(operation_id)
        if stream is None or not stream.history.is_final:
            return None
        return stream.history.truth()

    def window_model_source(self) -> "FlipStreamDevice":
        """This source: its program gives the window models."""
        return self

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Require the sealed length to end at the final readout run."""
        stream = self._stream_for(stream_operation.id)
        history = stream.history
        if not history.is_final:
            raise RuntimeError(
                f"flip stream {stream_operation.id!r} sealed before its "
                "final readout ran"
            )
        if history.round_count != stream_round_count:
            raise RuntimeError("sealed length differs from the rounds run")

    def window_model_for_stream(
        self,
        stream_id: Any,  # an opaque identity
        window: window_records.Window,
    ) -> fault_models.WindowErrorModel:
        """A window's model, made from the program without running it."""
        return self._models_by_stream_id[stream_id].for_window(window)

    def finalize_stream_models(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        """Fix the models at the final readout's round."""
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
        """Refuse a static plan: a stream's windows open as it runs."""
        del operation, round_count, fault_model_requirement
        del fault_exclusion_ranges, window_protocol
        if windows:
            raise ValueError(
                "not implemented yet: a flip stream's windows are planned as "
                "its rounds run, so it takes no static window plan"
            )
        return []

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement: fault_models.DecoderFaultModelRequirement,
        fault_exclusion_ranges: tuple = (),
        prior_faults: Optional[dict] = None,
    ) -> fault_models.WindowErrorModel:
        """A strong window's model in the stream's own fault names."""
        del round_count, fault_model_requirement
        stream_id = program_records.decode_identity(operation)
        models = self._models_by_stream_id[stream_id]
        return models.for_strong_window(
            window, fault_exclusion_ranges, prior_faults
        )

    def _install_run_seed_state(self, prepared_state) -> None:
        self._seed = prepared_state
        self._streams_by_id.clear()
        self._models_by_stream_id.clear()

    def _program_for(self, operation: program_records.Operation):
        if operation.id not in self._programs:
            raise ValueError(f"no repeated circuit for stream {operation.id!r}")
        return self._programs[operation.id]

    def _stream_for(self, identity) -> "_Stream":
        return self._streams_by_id[identity]

    def _emit(self, stream_id, global_round: int, is_final: bool) -> list:
        stream = self._stream_for(stream_id)
        history = stream.history
        if history.is_final:
            raise RuntimeError("a flip stream runs nothing after its readout")
        if global_round != history.round_count + 1:
            raise RuntimeError("a flip stream runs its rounds in order")
        self._mark_stochastic_use()
        bits = history.run_round(is_final)
        # a copy costs a FlipSimulator and a readout, so only for a listener
        if self.checkpoint_reached.has_listeners:
            snapshot = history.snapshot(stream_id)
            self.checkpoint_reached.fire(snapshot)
        payload = round_records.QPUReadout(
            stream_id,
            stream.owner.patches,
            global_round,
            bits=bits,
            size_bits=len(bits),
        )
        return [payload]


class FlipHistory:
    """One stream's simulator, the rounds it has run and their truth.

    The plan is the program's, shared by every stream of it.
    observable_flips holds the observables' flips let go at each
    rollover, folded into the simulator's own at the end.
    """

    def __init__(
        self, plan: "RoundPlan", seed: Optional[int], is_randomized: bool
    ) -> None:
        self.simulator = stim.FlipSimulator(
            batch_size=1,
            seed=seed,
            disable_stabilizer_randomization=not is_randomized,
        )
        self.plan = plan
        self.round_count = 0
        self.is_final = False
        self.observable_flips: list = []

    def run_round(self, is_final: bool) -> tuple[int, ...]:
        """Run the next round; its raw bits, flips against the reference."""
        self.round_count += 1
        name = _fragment_name(self.round_count, is_final)
        fragment = self.plan.fragment(name, self.round_count)
        self.simulator.do(fragment)
        self.is_final = is_final
        width = fragment.num_measurements
        reference = self.plan.references[name]
        flips = self.simulator.get_measurement_flips()
        round_flips = flips[flips.shape[0] - width :, 0]
        bits = numpy.logical_xor(round_flips, reference)
        if self.plan.rolls_over_after(self.round_count) and not is_final:
            self._roll_over()
        return tuple(int(bit) for bit in bits)

    def table(self) -> formation_records.TableReads:
        """The formation table of the rounds run so far."""
        live_reach = None
        if not self.is_final:
            live_reach = self.plan.live_reach
        return self.plan.template.table(
            self.round_count, is_final=self.is_final, live_reach=live_reach
        )

    def truth(self) -> tuple[int, ...]:
        """Each observable's flip over the whole stream."""
        flips = self.simulator.get_observable_flips()
        current = [int(flip) for flip in flips[:, 0]]
        flips_over_the_stream = _xor_lists(self.observable_flips, current)
        return tuple(flips_over_the_stream)

    def snapshot(
        self,
        stream_id: Any,  # an opaque identity
    ) -> checkpoint_records.CheckpointSnapshot:
        """The stream as if its readout came after the round just run.

        After the real readout it is that readout's. Before, a copy runs
        the noiseless readout and the stream runs on untouched.
        """
        if self.is_final:
            readout_count = self.plan.readout.num_detectors
            events = _last_detector_flips(self.simulator, readout_count)
            truth = self.truth()
        else:
            events, truth = self._closed_copy()
        return checkpoint_records.CheckpointSnapshot(
            stream_id,
            self.round_count,
            events,
            truth,
            self.is_final,
            self.plan.template.program,
        )

    def _closed_copy(self) -> tuple:
        """The readout's events and the truth, off a copy read out.

        The copy holds the frame and the records the readout reads, the
        rollover's own steps, with no random frame of its own; the
        readout is noiseless, so the copy draws nothing.
        """
        simulator = self.simulator
        frame_x, frame_z = _frame_of(simulator)
        flips = simulator.get_measurement_flips()
        reach = self.plan.readout_reach
        read = flips[flips.shape[0] - reach :]
        copy = stim.FlipSimulator(
            batch_size=1,
            num_qubits=simulator.num_qubits,
            disable_stabilizer_randomization=True,
            seed=0,
        )
        copy.broadcast_pauli_errors(pauli="X", mask=frame_x)
        copy.broadcast_pauli_errors(pauli="Z", mask=frame_z)
        copy.append_measurement_flips(read)
        copy.do(self.plan.readout)
        detectors = copy.get_detector_flips()
        observables = copy.get_observable_flips()
        events = tuple(int(bit) for bit in detectors[:, 0])
        read_out = [int(flip) for flip in observables[:, 0]]
        stream_truth = self.truth()
        stream_flips = list(stream_truth)
        truth = _xor_lists(stream_flips, read_out)
        return events, tuple(truth)

    def _roll_over(self) -> None:
        """Let go of the past, keeping the frame and the records reached."""
        simulator = self.simulator
        kept_count = self.plan.kept_measurements
        frame_x, frame_z = _frame_of(simulator)
        flips = simulator.get_measurement_flips()
        kept = flips[flips.shape[0] - kept_count :]
        observables = simulator.get_observable_flips()
        let_go = [int(flip) for flip in observables[:, 0]]
        self.observable_flips = _xor_lists(self.observable_flips, let_go)
        simulator.clear()
        fresh_x, fresh_z = _frame_of(simulator)
        x_change = frame_x ^ fresh_x
        z_change = frame_z ^ fresh_z
        simulator.broadcast_pauli_errors(pauli="X", mask=x_change)
        simulator.broadcast_pauli_errors(pauli="Z", mask=z_change)
        simulator.append_measurement_flips(kept)


class RoundPlan:
    """What every round of one program reads: fragments, references, table.

    live_reach is the live source's: how many rounds back a round after
    the first reads (streaming_stim_device._program_reach). drift is
    (round, factor): every round after round runs its fragment with each
    noise probability times factor; None runs the program as written.
    """

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        rollover_rounds: int,
        drift: Optional[tuple] = None,
    ) -> None:
        self.fragment_eras = _fragment_eras(program, drift)
        self.template = periodic_formation.StreamTemplate(program)
        self.references = _references(program, self.template)
        self.rollover_rounds = rollover_rounds
        repeated_back = detector_formation.measurements_read_back(
            program.repeated_round
        )
        final_back = detector_formation.measurements_read_back(
            program.final_round
        )
        self.kept_measurements = max(repeated_back, final_back)
        self.live_reach = _program_reach(program)

    def fragment(self, name: str, round_index: int) -> stim.Circuit:
        """The named fragment as round round_index runs it."""
        for last_round, fragments in self.fragment_eras:
            if last_round is None or round_index <= last_round:
                return fragments[name]
        raise AssertionError("the last era runs every later round")

    def rolls_over_after(self, round_index: int) -> bool:
        """Whether the simulator lets go of its past after this round."""
        return round_index % self.rollover_rounds == 0

    @functools.cached_property
    def readout(self) -> stim.Circuit:
        """The final round's readout, noiseless: what follows its round.

        Made when a checkpoint first asks, so a program whose final
        round is not its repeated round and a readout runs unasked.
        """
        return _noiseless_readout(self.template.program)

    @functools.cached_property
    def readout_reach(self) -> int:
        """The measurements before the readout that it reads."""
        return detector_formation.measurements_read_back(self.readout)


def scaled_noise(circuit: stim.Circuit, factor: float) -> stim.Circuit:
    """The circuit with every noise probability times factor.

    A noisy gate's arguments are its probabilities (stim gate_data
    is_noisy_gate), a measurement's its flip probability among them;
    Stim refuses a product past a channel's range.
    """
    scaled = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = instruction.body_copy()
            scaled_body = scaled_noise(body, factor)
            block = stim.CircuitRepeatBlock(
                instruction.repeat_count, scaled_body
            )
            scaled.append(block)
            continue
        gate = stim.gate_data(instruction.name)
        if not gate.is_noisy_gate:
            scaled.append(instruction)
            continue
        arguments = instruction.gate_args_copy()
        scaled_arguments = [argument * factor for argument in arguments]
        targets = instruction.targets_copy()
        scaled.append(instruction.name, targets, scaled_arguments)
    return scaled


@dataclasses.dataclass(frozen=True)
class _Stream:
    """One declared stream: its owner and what it has run."""

    owner: program_records.Operation
    history: "FlipHistory"


def _fragment_eras(program, drift: Optional[tuple]) -> tuple:
    """(last round or None, fragments) in order: as written, then drifted."""
    fragments = {name: getattr(program, name) for name in FRAGMENT_NAMES}
    if drift is None:
        return ((None, fragments),)
    drift_round, factor = drift
    drifted = {}
    for name, fragment in fragments.items():
        drifted[name] = scaled_noise(fragment, factor)
    return ((drift_round, fragments), (None, drifted))


def _references(program, template) -> dict:
    """The reference record of the first, a repeated and the final round.

    Read off the template's circuits; every repeated round's must be the
    same, or no one record serves the stream.
    """
    rounds = template.round_count
    live_record, live_rounds = _reference_by_round(program, rounds, False)
    final_record, final_rounds = _reference_by_round(program, rounds, True)
    repeated = [live_record[round_index] for round_index in live_rounds[1:]]
    for record in repeated[1:]:
        if not numpy.array_equal(record, repeated[0]):
            raise ValueError(
                "not implemented yet: a flip stream reads one reference "
                "record for every repeated round, and this program's "
                "repeated rounds differ in their noiseless outcomes"
            )
    single_record = program.single_round.reference_sample()
    return {
        "first_round": live_record[live_rounds[0]],
        "repeated_round": repeated[0],
        "final_round": final_record[final_rounds[-1]],
        "single_round": single_record,
    }


def _fragment_name(round_index: int, is_final: bool) -> str:
    """The fragment a round runs, as RepeatedStimCircuit.round_circuit picks."""
    if round_index == 1:
        if is_final:
            return "single_round"
        return "first_round"
    if is_final:
        return "final_round"
    return "repeated_round"


def _program_reach(program: circuit_records.RepeatedStimCircuit) -> int:
    """How many rounds back any round after the first reads."""
    round_width = program.repeated_round.num_measurements
    repeated_reach = detector_formation.rounds_read_back(
        program.repeated_round, round_width
    )
    final_reach = detector_formation.rounds_read_back(
        program.final_round, round_width
    )
    return max(repeated_reach, final_reach)


def _reference_by_round(program, round_count: int, is_final: bool) -> tuple:
    """The assembled circuit's reference sample, cut into its rounds."""
    circuit = stim.Circuit()
    starts = []
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        is_last_final = is_final and round_index == round_count
        fragment = program.round_circuit(round_index, is_last_final)
        starts.append(circuit.num_measurements)
        circuit += fragment
    sample = circuit.reference_sample()
    ends = [*starts[1:], circuit.num_measurements]
    by_round = {}
    for round_index, (start, end) in enumerate(
        zip(starts, ends, strict=True), 1
    ):
        by_round[round_index] = sample[start:end]
    return by_round, list(by_round)


def _noiseless_readout(
    program: circuit_records.RepeatedStimCircuit,
) -> stim.Circuit:
    """The instructions the final round adds to the repeated round."""
    repeated = program.repeated_round
    final = program.final_round
    length = len(repeated)
    head = final[:length]
    if head != repeated:
        raise ValueError(
            "not implemented yet: a checkpoint closes a copy with the "
            "readout the final round adds to the repeated round, and this "
            "program's final round does not begin with its repeated round"
        )
    readout = final[length:]
    return readout.without_noise()


def _last_detector_flips(simulator: stim.FlipSimulator, count: int) -> tuple:
    """The flips of the last count detectors the simulator ran."""
    detectors = simulator.get_detector_flips()
    last = detectors[detectors.shape[0] - count :, 0]
    return tuple(int(bit) for bit in last)


def _frame_of(simulator: stim.FlipSimulator) -> tuple:
    """The X and Z parts of the simulator's Pauli frame."""
    frames = simulator.to_numpy(output_xs=True, output_zs=True)
    return frames[0], frames[1]


def _xor_lists(first: list, second: list) -> list:
    """Two flip lists XORed, the shorter read as zeros past its end."""
    length = max(len(first), len(second))
    padded_first = first + [0] * (length - len(first))
    padded_second = second + [0] * (length - len(second))
    return [
        left ^ right
        for left, right in zip(padded_first, padded_second, strict=True)
    ]
