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

A burst (BurstSettings) raises the noise on a disk of qubits from its
onset round on, by a factor that rises over a few rounds and then
decays, as a particle's hit does on a chip (Google 2408.13687 p.4-5).
Each round from the onset runs its fragment cut at the disk's edge and
joined again at that round's factor (BurstFragment). The decoders are
not told: the reference record, the truth and the window models are
the program's, as with the drift.

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
import decsim.qpu.burst_shapes as burst_shapes
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


@dataclasses.dataclass(frozen=True)
class BurstSettings:
    """One noise burst on a disk of qubits, which the decoders are not told of.

    From onset_round on, every noise probability of an instruction on a
    qubit inside the disk is multiplied by 1 + (peak_factor - 1) f: f
    climbs to 1 over rise_rounds, (i + 1) / rise_rounds in the i-th
    round, then decays as exp(-(rounds since the peak) / decay_rounds)
    to the stream's end, the shot burst's shape
    (burst_shapes.peak_share). Scaling the circuit's own probabilities
    reads no noise names, so it raises Deltakit's SD6 and Stim's
    generated noise alike; Tan et al. 2406.18897 raise every channel of
    a burst by one factor too.

    The disk is the qubit_count qubits, data and measure, nearest center
    in their first two Stim coordinates (the patch's midpoint when
    None), the lower qubit index first at equal distance; a patch of
    fewer qubits is raised whole. A burst has a physical size, so its
    count stays the same at every distance. Every stream of the source
    runs the burst on its own patch.

    The defaults are Willow's large burst in its error-correction data
    (2408.13687 p.4-5 and p.20): about 30 qubits, decaying over 0.4 to
    0.7 ms, 550 rounds at 1.1 us a round, about once an hour. The rise
    of about 3 rounds is our fit of 39 bursts in Willow's data
    (specs/adaptive-switching/bursts.md), as the six largest bursts of
    2408.13687 peak about 3 rounds after onset. peak_factor has no
    source; a study sweeps it.
    """

    onset_round: int
    peak_factor: float
    rise_rounds: int = 3
    decay_rounds: float = 550.0
    qubit_count: int = 30
    center: Optional[tuple] = None

    def __post_init__(self) -> None:
        """Every value checked, with the sentence the caller reads."""
        config.check_whole_count(
            "flip_stream.burst.onset_round", self.onset_round, "rounds"
        )
        _check_factor("flip_stream.burst.peak_factor", self.peak_factor)
        config.check_whole_count(
            "flip_stream.burst.rise_rounds", self.rise_rounds, "rounds"
        )
        _check_decay_rounds(self.decay_rounds)
        config.check_whole_count(
            "flip_stream.burst.qubit_count", self.qubit_count, "qubits"
        )
        stim_device.check_burst_center("flip_stream.burst.center", self.center)


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
        rollover changes no bit. burst is one noise burst in every
        stream, None for none.
        """

        rollover_rounds: int = DEFAULT_ROLLOVER_ROUNDS
        stabilizer_randomization: bool = True
        drift_round: Optional[int] = None
        drift_factor: float = 1.0
        burst: Optional[BurstSettings] = None
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
            _check_factor("flip_stream.drift_factor", self.drift_factor)

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
        plan = RoundPlan(
            program, settings.rollover_rounds, drift, settings.burst
        )
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
    burst raises the noise on its disk from its onset on; None is no
    burst.
    """

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        rollover_rounds: int,
        drift: Optional[tuple] = None,
        burst: Optional[BurstSettings] = None,
    ) -> None:
        self.noise = _NoiseSchedule(program, drift, burst)
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
        return self.noise.fragment(name, round_index)

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


class BurstFragment:
    """One fragment cut at a burst's disk, joined again at any factor.

    Each noisy instruction on the disk is cut into runs of its target
    groups, in order: a one-qubit gate's qubits, a two-qubit gate's
    pairs, any other noisy instruction (a correlated error, a Pauli
    product) whole. A group is inside when any of its qubits is in the
    disk, so a pair across the disk's edge is raised, as the shot burst
    keeps a pair (stim_device._pairs_touching). The runs keep the
    targets' order, so a measurement's record, and every detector that
    reads it, is unchanged.

    A run inside is written as text at its raised probabilities and read
    by Stim, and joined to kept segments that stay Stim circuits: Stim's
    parser reads Python's repr of a float exactly, while Stim's own text
    rounds it. Joining fuses equal neighbours, so at factor 1 the
    fragment comes back as it was. On stim 1.16 appending one run
    through stim.Circuit.append costs up to 1 ms; a d = 9 SD6 round
    joined so costs about 200 us, against about 12 us to run it.
    """

    def __init__(self, fragment: stim.Circuit, disk: frozenset) -> None:
        kept_segments, raised_runs = _cut_at_disk(fragment, disk)
        self.kept_segments = kept_segments
        self.raised_runs = raised_runs

    def raised(self, factor: float) -> stim.Circuit:
        """The fragment with each probability on the disk times factor."""
        first_segment = self.kept_segments[0]
        raised = first_segment.copy()
        later_segments = self.kept_segments[1:]
        runs_and_segments = zip(self.raised_runs, later_segments, strict=True)
        for run, segment in runs_and_segments:
            run_circuit = run.circuit(factor)
            raised += run_circuit
            raised += segment
        return raised


class _NoiseSchedule:
    """The circuit each round runs: as written, drifted, or burst.

    The eras are (last round or None, fragments) in order: as written,
    then drifted. A burst raises each era's fragments on its disk from
    its onset on, by a factor that changes every round, so a burst
    round's circuit is joined as it runs rather than kept.
    """

    def __init__(
        self,
        program: circuit_records.RepeatedStimCircuit,
        drift: Optional[tuple],
        burst: Optional[BurstSettings],
    ) -> None:
        self.eras = _fragment_eras(program, drift)
        self.burst = burst
        self.burst_eras = _burst_eras(program, self.eras, burst)

    def fragment(self, name: str, round_index: int) -> stim.Circuit:
        """The named fragment as round round_index runs it."""
        era_index = self._era_index(round_index)
        _last_round, fragments = self.eras[era_index]
        if self.burst is None or round_index < self.burst.onset_round:
            return fragments[name]
        burst_fragments = self.burst_eras[era_index]
        factor = _burst_factor(self.burst, round_index)
        burst_fragment = burst_fragments[name]
        return burst_fragment.raised(factor)

    def _era_index(self, round_index: int) -> int:
        for index, (last_round, _fragments) in enumerate(self.eras):
            if last_round is None or round_index <= last_round:
                return index
        raise AssertionError("the last era runs every later round")


@dataclasses.dataclass(frozen=True)
class _RaisedRun:
    """One run of targets inside the disk: its gate, probabilities, text."""

    name: str
    probabilities: tuple
    targets_text: str

    def circuit(self, factor: float) -> stim.Circuit:
        """The run with its probabilities times factor, read from text."""
        scaled = [probability * factor for probability in self.probabilities]
        written = [repr(probability) for probability in scaled]
        arguments = ",".join(written)
        text = f"{self.name}({arguments}) {self.targets_text}"
        return stim.Circuit(text)


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


def _burst_eras(program, eras: tuple, burst: Optional[BurstSettings]) -> tuple:
    """Each era's fragments cut at the burst's disk; none without a burst."""
    if burst is None:
        return ()
    disk = _burst_disk(program, burst)
    burst_eras = []
    for _last_round, fragments in eras:
        cut = {
            name: BurstFragment(fragment, disk)
            for name, fragment in fragments.items()
        }
        burst_eras.append(cut)
    return tuple(burst_eras)


def _burst_disk(program, burst: BurstSettings) -> frozenset:
    """The burst's qubit_count qubits nearest its centre (BurstSettings).

    The qubits' coordinates are the first round's, where Stim's
    generator and Deltakit's export declare them.
    """
    first_round = program.first_round
    coordinates = first_round.get_final_qubit_coordinates()
    if not coordinates:
        raise ValueError(
            "flip_stream.burst finds its disk by the qubits' coordinates, "
            "and this program's first round declares none (QUBIT_COORDS)"
        )
    center = burst.center
    if center is None:
        center = burst_shapes.coordinate_midpoint(coordinates)
    distances = burst_shapes.planar_distances(coordinates, center)
    ranked = sorted((distance, qubit) for qubit, distance in distances.items())
    nearest = ranked[: burst.qubit_count]
    return frozenset(qubit for _distance, qubit in nearest)


def _burst_factor(burst: BurstSettings, round_index: int) -> float:
    """1 + (peak_factor - 1) f: the disk's noise in calm units this round."""
    share = burst_shapes.peak_share(
        round_index, burst.onset_round, burst.rise_rounds, burst.decay_rounds
    )
    extra = burst.peak_factor - 1
    raised = extra * share
    return 1 + raised


def _cut_at_disk(fragment: stim.Circuit, disk: frozenset) -> tuple:
    """(kept segments, raised runs), one run between each two segments."""
    kept_segments = [stim.Circuit()]
    raised_runs = []
    for instruction in fragment:
        if not _raises_on(instruction, disk):
            kept_segments[-1].append(instruction)
            continue
        runs = _target_runs(instruction, disk)
        _place_runs(instruction, runs, kept_segments, raised_runs)
    return kept_segments, raised_runs


def _raises_on(instruction: stim.CircuitInstruction, disk: frozenset) -> bool:
    """Whether the instruction has a probability on a qubit of the disk."""
    gate = stim.gate_data(instruction.name)
    arguments = instruction.gate_args_copy()
    if not gate.is_noisy_gate or not arguments:
        return False
    targets = instruction.targets_copy()
    return _is_inside(targets, disk)


def _target_runs(instruction: stim.CircuitInstruction, disk: frozenset) -> list:
    """(is inside, targets) runs of the instruction's groups, in order."""
    groups = _target_groups(instruction)
    runs = []
    for group in groups:
        is_inside = _is_inside(group, disk)
        _extend_runs(runs, is_inside, group)
    return runs


def _target_groups(instruction: stim.CircuitInstruction) -> list:
    """A one-qubit gate's targets one by one, a two-qubit gate's in pairs.

    Any other noisy instruction, a correlated error or a Pauli product,
    is one group, raised whole.
    """
    gate = stim.gate_data(instruction.name)
    targets = instruction.targets_copy()
    if gate.is_single_qubit_gate:
        return [[target] for target in targets]
    if not gate.is_two_qubit_gate:
        return [targets]
    firsts = targets[::2]
    seconds = targets[1::2]
    return [
        [first, second] for first, second in zip(firsts, seconds, strict=True)
    ]


def _is_inside(targets: list, disk: frozenset) -> bool:
    """Whether any of the targets is a qubit of the disk."""
    qubits = [target.qubit_value for target in targets]
    return not disk.isdisjoint(qubits)


def _extend_runs(runs: list, is_inside: bool, group: list) -> None:
    """The group joins the last run on its side of the edge, or opens one."""
    if runs and runs[-1][0] == is_inside:
        runs[-1][1].extend(group)
        return
    runs.append((is_inside, list(group)))


def _place_runs(
    instruction: stim.CircuitInstruction,
    runs: list,
    kept_segments: list,
    raised_runs: list,
) -> None:
    """A run outside joins the last kept segment; one inside opens the next."""
    name = instruction.name
    arguments = instruction.gate_args_copy()
    for is_inside, targets in runs:
        if not is_inside:
            kept_segments[-1].append(name, targets, arguments)
            continue
        targets_text = _targets_text(name, targets, arguments)
        run = _RaisedRun(name, tuple(arguments), targets_text)
        raised_runs.append(run)
        next_segment = stim.Circuit()
        kept_segments.append(next_segment)


def _targets_text(name: str, targets: list, arguments: list) -> str:
    """The targets as Stim writes them, after the gate and its probabilities."""
    instruction = stim.CircuitInstruction(name, targets, arguments)
    written = str(instruction)
    _head, _parenthesis, targets_text = written.partition(") ")
    return targets_text


def _check_factor(name: str, value: object) -> None:
    """A factor on the noise is a finite number of at least 0."""
    if config.is_number(value) and 0 <= value < math.inf:
        return
    raise ValueError(
        f"{name} must be a finite number of at least 0, got {value!r}"
    )


def _check_decay_rounds(value: object) -> None:
    if config.is_number(value) and 0 < value < math.inf:
        return
    raise ValueError(
        "flip_stream.burst.decay_rounds must be a finite number of rounds "
        f"above 0, got {value!r}"
    )


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
