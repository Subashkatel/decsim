"""Run a Deltakit memory or a protected wait through the existing machine.

The protection workload uses one finite history and a fixed scheduled end,
not result-dependent quantum branching. Numeric decoder cards execute real
PyMatching with declared service time (build/decoders.py).
"""

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Optional

import stim

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.settings as machine_settings


def main() -> None:
    """Write the input circuit, settings, result and trace into one folder."""
    arguments = _arguments()
    _resolve_memory_parameters(arguments)
    if arguments.mode == "protection" and arguments.family == "repetition":
        raise ValueError(
            "the protection example requires rotated_surface memory"
        )
    started = time.perf_counter()
    circuit, measurement_rounds = _circuit(arguments)
    workload = memory_workload(circuit, arguments.rounds, arguments.patch)
    if arguments.mode == "protection":
        workload = protection_workload(
            circuit, arguments.rounds, arguments.prefix_rounds, arguments.patch
        )
    settings = supplied_settings(
        circuit,
        measurement_rounds,
        workload,
        distance=arguments.distance,
        round_count=arguments.rounds,
        period_microseconds=arguments.period_microseconds,
        feedback_microseconds=arguments.feedback_microseconds,
        decoder_microseconds=arguments.decoder_microseconds,
    )
    if arguments.family == "repetition":
        code = RepetitionMemory(arguments.distance, arguments.rounds)
        qpu = dataclasses.replace(settings.qpu, distance=None, code=code)
        settings = dataclasses.replace(settings, qpu=qpu)
    machine = machine_module.Machine.build(settings, arguments.seed)
    prepared = time.perf_counter()
    setup_seconds = prepared - started
    result = machine.run()
    _write_run(arguments, circuit, measurement_rounds, machine, result)
    setup_path = arguments.output / "setup_seconds.json"
    _write_json(setup_path, setup_seconds)
    print(f"complete: {arguments.output}; setup {setup_seconds:.6f} seconds")


def memory_workload(
    circuit: stim.Circuit, round_count: int, patch: str
) -> workload_settings.WorkloadSettings:
    """One finite memory, including its final data measurement."""
    operation = program_records.Operation(
        1, "memory", (patch,), patches=(patch,), circuit=circuit
    )
    policy = round_policies.FixedRounds(round_count)
    return workload_settings.WorkloadSettings(
        operations=(operation,), rounds_policy=policy
    )


def protection_workload(
    circuit: stim.Circuit, round_count: int, prefix_round_count: int, patch: str
) -> workload_settings.WorkloadSettings:
    """One history protected until a declared horizon after a feedback wait.

    The continuation is an identity operation with a scheduling dependency
    on the prefix's decoded result. The physical circuit remains memory.
    A late continuation exhausts the finite source loudly.
    """
    if not 1 <= prefix_round_count < round_count:
        raise ValueError("prefix rounds must lie inside the finite horizon")
    owner = program_records.Operation(
        100, "protected-memory", (patch,), patches=(patch,), circuit=circuit
    )
    prefix = program_records.Operation(
        1,
        "prefix",
        (patch,),
        patches=(patch,),
        circuit=circuit,
        stream_id=100,
        stream_offset=0,
    )
    begin = program_records.Operation(
        2,
        "begin-protection",
        (patch,),
        patches=(patch,),
        predecessors=(1,),
        emits_detector_data=False,
    )
    resume = program_records.Operation(
        3,
        "continue-memory",
        (patch,),
        patches=(patch,),
        predecessors=(2,),
        blocked_by=1,
        emits_detector_data=False,
    )
    finish = program_records.Operation(
        4,
        "final-readout",
        (patch,),
        patches=(patch,),
        predecessors=(3,),
        scheduled_start_round=round_count,
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 2, 4)
    counts = {100: round_count, 1: prefix_round_count, 2: 0, 3: 1, 4: 0}
    policy = round_policies.PerOperationRounds(counts)
    return workload_settings.WorkloadSettings(
        operations=(prefix, begin, resume, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def supplied_settings(
    circuit: stim.Circuit,
    measurement_rounds: dict[int, int],
    workload: workload_settings.WorkloadSettings,
    *,
    distance: int,
    round_count: int,
    period_microseconds: float,
    feedback_microseconds: float,
    decoder_microseconds: float = 0.1,
) -> machine_settings.MachineSettings:
    """Build the ordinary supplied-circuit path, regardless of its producer."""
    if circuit.num_observables != 1:
        raise ValueError("the memory example requires one logical observable")
    declared_rounds = measurement_rounds.values()
    if max(declared_rounds) != round_count:
        raise ValueError(
            "the declared horizon must equal the final readout round"
        )
    table = detector_formation.build_formation_table(
        circuit, round_count, measurement_rounds=measurement_rounds
    )
    detector_rounds = table.detector_rounds()
    owners = workload.operations
    if workload.dynamic_streams:
        owners = workload.dynamic_streams
    measurements = {owner.id: measurement_rounds for owner in owners}
    detectors = {owner.id: detector_rounds for owner in owners}
    source = stim_device.StimDevice(
        measurement_rounds=measurements, detector_rounds=detectors
    )
    qpu = qpu_settings.QpuSettings(
        device=source,
        distance=distance,
        round_period_microseconds=period_microseconds,
    )
    clock = config.Clock(1000)
    decoder = decoder_settings.DecoderSettings(
        kind=decoder_microseconds, engine_clock=clock
    )
    links = _feedback_links(feedback_microseconds)
    observation = observation_settings.ObservationSettings(
        trace="chrome",
        data_movement=True,
    )
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=decoder,
        links=links,
        observation=observation,
    )


@dataclasses.dataclass(frozen=True)
class RepetitionMemory:
    """Whole-shot repetition card, with one check per neighbouring pair.

    This example chooses a single finite decode window. It asserts no
    surface-code buffering floor or repetition window threshold.
    """

    distance: int
    round_count: int
    name: str = "repetition memory"
    window_floor_justification: Optional[str] = None

    def rounds_per_logical_cycle(self) -> int:
        """Use one distance of rounds per logical cycle."""
        return self.distance

    def round_period_us(self) -> Optional[float]:
        """Use the cadence declared in QpuSettings."""
        return None

    def commit_rounds(self) -> int:
        """Decode the complete finite memory as one window."""
        return self.round_count

    def buffer_rounds(self) -> int:
        """The whole-shot problem needs no look-ahead."""
        return 0

    def buffering_floor(self) -> tuple[int, int]:
        """Declare no floor for this whole-shot example."""
        return (0, 0)

    def spatial_nodes(self, num_patches: int) -> int:
        """Count the check nodes in one repetition round."""
        return self.syndrome_bits_per_round(num_patches)

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """One stabilizer measurement per neighbouring data pair."""
        checks_per_patch = self.distance - 1
        return num_patches * checks_per_patch


def _feedback_links(feedback_microseconds: float):
    links = link_profiles.logical_reference_profile()
    ticks = config.microseconds_to_ticks(feedback_microseconds)
    channel = dataclasses.replace(
        links.frame_to_controller.channel, propagation_latency_ticks=ticks
    )
    path = dataclasses.replace(links.frame_to_controller, channel=channel)
    return dataclasses.replace(links, frame_to_controller=path)


def _circuit(arguments) -> tuple[stim.Circuit, dict[int, int]]:
    if arguments.input is None:
        return deltakit.memory_circuit(
            arguments.family,
            arguments.distance,
            arguments.rounds,
            arguments.basis,
            arguments.probability,
        )
    circuit_path = arguments.input / "circuit.stim"
    circuit = stim.Circuit.from_file(str(circuit_path))
    mapping_path = arguments.input / "measurement_rounds.json"
    mapping_text = mapping_path.read_text()
    mapping = json.loads(mapping_text)
    return circuit, {int(index): value for index, value in mapping.items()}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("memory", "protection"), default="memory"
    )
    parser.add_argument("--distance", type=int)
    parser.add_argument(
        "--family",
        choices=("rotated_surface", "repetition"),
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--prefix-rounds", type=int, default=3)
    parser.add_argument("--basis", choices=("X", "Z"))
    parser.add_argument("--probability", type=float)
    parser.add_argument("--period-microseconds", type=float, default=1.1)
    parser.add_argument("--feedback-microseconds", type=float, default=4.0)
    parser.add_argument("--decoder-microseconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch", default="memory-patch")
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _resolve_memory_parameters(arguments) -> None:
    """Keep the physical input's geometry and horizon when replaying it."""
    parameters = {
        "family": "rotated_surface",
        "distance": 3,
        "rounds": 24,
        "basis": "Z",
        "probability": 0.001,
    }
    if arguments.input is not None:
        path = arguments.input / "arguments.json"
        text = path.read_text()
        recorded = json.loads(text)
        parameters = {name: recorded[name] for name in parameters}
    for name, value in parameters.items():
        selected = getattr(arguments, name)
        _check_replay_parameter(arguments.input, name, selected, value)
        if selected is None:
            setattr(arguments, name, value)


def _check_replay_parameter(input_folder, name, selected, recorded) -> None:
    if input_folder is None or selected is None:
        return
    if selected != recorded:
        raise ValueError(f"{name} differs from the exported circuit parameters")


def _write_run(arguments, circuit, measurement_rounds, machine, result) -> None:
    folder = arguments.output
    folder.mkdir(parents=True, exist_ok=True)
    circuit_path = folder / "circuit.stim"
    circuit.to_file(str(circuit_path))
    mapping_path = folder / "measurement_rounds.json"
    _write_json(mapping_path, measurement_rounds)
    result_value = dataclasses.asdict(result)
    result_path = folder / "result.json"
    _write_json(result_path, result_value)
    command_values = [
        _command_value(event)
        for event in machine.observation.command_events.events
    ]
    commands_path = folder / "commands.json"
    _write_json(commands_path, command_values)
    trace_path = folder / "trace.json"
    machine.observation.trace_writer.write(str(trace_path))
    argument_values = vars(arguments)
    arguments_path = folder / "arguments.json"
    _write_json(arguments_path, argument_values)


def _command_value(event) -> dict:
    return {
        "kind": event.kind,
        "tick": event.tick,
        "operation_id": event.command.operation.id,
    }


def _write_json(path: pathlib.Path, value) -> None:
    text = json.dumps(value, indent=2, default=str)
    document = text + "\n"
    path.write_text(document)


if __name__ == "__main__":
    main()
