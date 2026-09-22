"""Run memory until decoded feedback permits an actual final readout.

Four canonical Stim fragments reproduce the same live history without the
optional producer. Runtime feedback changes how long the data stays live.
"""

import argparse
import dataclasses
import json
import pathlib
from typing import Optional

import stim

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.results as result_records
import decsim.settings as machine_settings


def main() -> None:
    """Save canonical inputs and the physical history selected by feedback."""
    arguments = _arguments()
    parameters = _physical_parameters(arguments)
    program = _program(arguments.input, parameters)
    source = streaming_stim_device.StreamingStimDevice(programs={100: program})
    settings = live_settings(
        source,
        distance=parameters["distance"],
        round_period_microseconds=parameters["round_period_microseconds"],
        prefix_round_count=arguments.prefix_round_count,
        patch=arguments.patch,
        feedback_microseconds=arguments.feedback_microseconds,
        decoder_microseconds=arguments.decoder_microseconds,
    )
    machine = machine_module.Machine.build(settings, arguments.seed)
    result = machine.run()
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_inputs(arguments.output, program, parameters)
    _write_execution(arguments.output, source, machine, result)
    selected_arguments = vars(arguments)
    argument_values = dict(selected_arguments)
    argument_values.update(parameters)
    argument_path = arguments.output / "arguments.json"
    _write_json(argument_path, argument_values)
    print(f"complete: {arguments.output}")


def live_settings(
    source: streaming_stim_device.StreamingStimDevice,
    *,
    distance: int,
    round_period_microseconds: float,
    prefix_round_count: int,
    patch: str,
    feedback_microseconds: float,
    decoder_microseconds: float,
) -> machine_settings.MachineSettings:
    """Use functional PyMatching with caller-declared service and link times."""
    workload = protection_workload(prefix_round_count, patch)
    qpu = qpu_settings.QpuSettings(
        distance=distance,
        device=source,
        round_period_microseconds=round_period_microseconds,
    )
    clock = config.Clock(1000)
    decoder = decoder_settings.DecoderSettings(
        kind=decoder_microseconds, engine_clock=clock
    )
    links = _feedback_links(feedback_microseconds)
    observation = observation_settings.ObservationSettings(
        trace="chrome", data_movement=True
    )
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=decoder,
        links=links,
        observation=observation,
    )


def protection_workload(
    prefix_round_count: int, patch: str
) -> workload_settings.WorkloadSettings:
    """Keep the complete dependency graph together to expose its ownership.

    The prefix requests decoding while the region protects the same stream.
    Decoded release permits resume, whose completion requests final readout.
    These operation and stream identities are local to this example.
    """
    if prefix_round_count < 1:
        raise ValueError("prefix_round_count must be positive")
    owner = program_records.Operation(100, "memory", (patch,), patches=(patch,))
    prefix = program_records.Operation(
        1, "prefix", (patch,), patches=(patch,), stream_id=100, stream_offset=0
    )
    begin = program_records.Operation(
        2,
        "protect",
        (patch,),
        patches=(patch,),
        predecessors=(1,),
        emits_detector_data=False,
    )
    resume = program_records.Operation(
        3,
        "resume",
        (patch,),
        patches=(patch,),
        predecessors=(2,),
        blocked_by=1,
        emits_detector_data=False,
    )
    finish = program_records.Operation(
        4,
        "readout",
        (patch,),
        patches=(patch,),
        predecessors=(3,),
        emits_detector_data=False,
    )
    region = program_records.ProtectedRegion(100, 2, 4)
    counts = {100: 0, 1: prefix_round_count, 2: 0, 3: 1, 4: 0}
    policy = round_policies.PerOperationRounds(counts)
    return workload_settings.WorkloadSettings(
        operations=(prefix, begin, resume, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _physical_arguments(parser)
    parser.add_argument(
        "--prefix-rounds", dest="prefix_round_count", type=int, default=3
    )
    parser.add_argument("--feedback-microseconds", type=float, default=4.0)
    parser.add_argument("--decoder-microseconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patch", default="memory-patch")
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _physical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--distance", type=int)
    parser.add_argument("--basis", choices=("X", "Z"))
    parser.add_argument("--physical-error-probability", type=float)
    parser.add_argument("--round-period-microseconds", type=float)
    parser.add_argument("--noise-model", choices=("sd6", "physical"))
    parser.add_argument("--relaxation-time-microseconds", type=float)
    parser.add_argument("--dephasing-time-microseconds", type=float)


def _physical_parameters(arguments: argparse.Namespace) -> dict:
    defaults = {
        "distance": 3,
        "basis": "Z",
        "physical_error_probability": 0.001,
        "round_period_microseconds": 1.1,
        "noise_model": "sd6",
        "relaxation_time_microseconds": None,
        "dephasing_time_microseconds": None,
    }
    recorded = defaults
    if arguments.input is not None:
        path = arguments.input / "physical_parameters.json"
        text = path.read_text()
        recorded = json.loads(text)
    parameters = {}
    for name in defaults:
        selected = getattr(arguments, name)
        parameters[name] = _selected_physical_parameter(
            arguments.input, name, selected, recorded[name]
        )
    return parameters


def _selected_physical_parameter(
    input_folder: Optional[pathlib.Path],
    name: str,
    selected: object,
    recorded: object,
) -> object:
    if selected is None:
        return recorded
    if input_folder is not None and selected != recorded:
        raise ValueError(
            f"{name} differs from the exported physical parameters"
        )
    return selected


def _program(
    input_folder: Optional[pathlib.Path], parameters: dict
) -> circuit_records.RepeatedStimCircuit:
    if input_folder is not None:
        return _load_program(input_folder, parameters)
    import decsim.frontends.deltakit as deltakit

    return deltakit.memory_rounds(
        "rotated_surface",
        parameters["distance"],
        parameters["basis"],
        parameters["physical_error_probability"],
        round_period_microseconds=parameters["round_period_microseconds"],
        noise_model=parameters["noise_model"],
        relaxation_time_microseconds=parameters["relaxation_time_microseconds"],
        dephasing_time_microseconds=parameters["dephasing_time_microseconds"],
    )


def _load_program(
    folder: pathlib.Path, parameters: dict
) -> circuit_records.RepeatedStimCircuit:
    fragments = {}
    for name in (
        "first_round",
        "repeated_round",
        "final_round",
        "single_round",
    ):
        path = folder / f"{name}.stim"
        fragments[name] = stim.Circuit.from_file(str(path))
    return circuit_records.RepeatedStimCircuit(
        **fragments,
        round_period_microseconds=parameters["round_period_microseconds"],
    )


def _feedback_links(
    feedback_microseconds: float,
) -> link_settings.FabricSettings:
    links = link_profiles.logical_reference_profile()
    ticks = config.microseconds_to_ticks(feedback_microseconds)
    channel = dataclasses.replace(
        links.frame_to_controller.channel, propagation_latency_ticks=ticks
    )
    path = dataclasses.replace(links.frame_to_controller, channel=channel)
    return dataclasses.replace(links, frame_to_controller=path)


def _write_inputs(
    folder: pathlib.Path,
    program: circuit_records.RepeatedStimCircuit,
    parameters: dict,
) -> None:
    for name in (
        "first_round",
        "repeated_round",
        "final_round",
        "single_round",
    ):
        circuit = getattr(program, name)
        path = folder / f"{name}.stim"
        circuit.to_file(str(path))
    path = folder / "physical_parameters.json"
    _write_json(path, parameters)


def _write_execution(
    folder: pathlib.Path,
    source: streaming_stim_device.StreamingStimDevice,
    machine: machine_module.Machine,
    result: result_records.RunResult,
) -> None:
    circuit = source.executed_circuit(100)
    circuit_path = folder / "executed.stim"
    circuit.to_file(str(circuit_path))
    measurement_rounds = source.measurement_rounds_for_stream(100)
    mapping_path = folder / "measurement_rounds.json"
    _write_json(mapping_path, measurement_rounds)
    measurements = source.sampled_measurements(100)
    measurement_path = folder / "measurements.json"
    _write_json(measurement_path, measurements)
    result_values = dataclasses.asdict(result)
    result_path = folder / "result.json"
    _write_json(result_path, result_values)
    command_values = [
        _command_value(event)
        for event in machine.observation.command_events.events
    ]
    commands_path = folder / "commands.json"
    _write_json(commands_path, command_values)
    trace_path = folder / "trace.json"
    machine.observation.trace_writer.write(str(trace_path))


def _command_value(event: cycle_clock.QPUCommandEvent) -> dict:
    return {
        "kind": event.kind,
        "tick": event.tick,
        "operation_id": event.command.operation.id,
    }


def _write_json(path: pathlib.Path, value: object) -> None:
    text = json.dumps(value, indent=2, default=str)
    document = text + "\n"
    path.write_text(document)


if __name__ == "__main__":
    main()
