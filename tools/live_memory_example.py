"""Run memory until decoded feedback permits an actual final readout.

Four canonical Stim fragments reproduce the same live history without the
optional producer. Runtime feedback changes how long the data stays live.
The fragments are read and written in the files row's form: a fragments
folder with the four .stim files and physical.json, which a run folder
keeps in inputs/<id>/fragments.
"""

import argparse
import json
import pathlib

import decsim.collect as collect
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.experiments.run_command as run_command
import decsim.experiments.run_folder as run_folder
import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.producers as producers
import decsim.qpu.settings as qpu_settings
import decsim.records.circuits as circuit_records
import decsim.settings as machine_settings


def main() -> None:
    """Save canonical inputs and the physical history selected by feedback.

    The output folder is a run folder (decsim/experiments/run_folder.py):
    its manifest, every value of the run in resolved/, its workload in
    inputs/, the shot's files as decsim run writes them and the finished
    flag, beside the executed history and the arguments.
    """
    arguments = _arguments()
    recorded = _recorded_values(arguments.input)
    parameters = _physical_parameters(arguments, recorded)
    program = _program(arguments, parameters)
    if arguments.prefix_round_count is None:
        arguments.prefix_round_count = parameters["distance"]
    settings = live_settings(
        program,
        distance=parameters["distance"],
        round_period_microseconds=parameters["round_period_microseconds"],
        prefix_round_count=arguments.prefix_round_count,
        patch=arguments.patch,
        feedback_microseconds=arguments.feedback_microseconds,
        decoder_microseconds=arguments.decoder_microseconds,
    )
    machine = machine_module.Machine.build(settings, arguments.seed)
    started_utc = run_folder.start_run(None, arguments.output)
    metadata = {
        "physical_error_probability": parameters["physical_error_probability"],
        "distance": parameters["distance"],
        "round_period_microseconds": parameters["round_period_microseconds"],
    }
    seeds = [(arguments.seed, 1)]
    task = collect.Task(settings, 1, metadata)
    run_folder.record_point(arguments.output, task, seeds)
    result = machine.run()
    label = f"seed{arguments.seed}"
    run_command.write_shot(machine, settings, arguments.output, label, result)
    _write_execution(arguments.output, machine)
    selected_arguments = vars(arguments)
    argument_values = dict(selected_arguments)
    argument_values.update(parameters)
    argument_path = arguments.output / "arguments.json"
    argument_json = collect.json_value(argument_values)
    run_folder.write_json(argument_path, argument_json)
    run_folder.finish_run(None, arguments.output, started_utc)
    print(f"complete: {arguments.output}")


def live_settings(
    program: circuit_records.RepeatedStimCircuit,
    *,
    distance: int,
    round_period_microseconds: float,
    prefix_round_count: int,
    patch: str,
    feedback_microseconds: float,
    decoder_microseconds: float,
) -> machine_settings.MachineSettings:
    """Use functional PyMatching with caller-declared service and link times.

    The workload is decsim.producers live_memory on the fragments, the
    one a yaml names, and the source is built from them as a yaml's
    streaming_stim is (decsim/build/plan.py _syndrome_source).
    """
    workload = producers.live_memory(program, prefix_round_count, patch)
    section = workload_settings.WorkloadSettings()
    lowered = section.running(workload)
    qpu = qpu_settings.QpuSettings(
        kind="streaming_stim",
        distance=distance,
        round_period_microseconds=round_period_microseconds,
    )
    clock = config.Clock(1000)
    decoder = decoder_settings.DecoderSettings(
        kind=decoder_microseconds, engine_clock=clock
    )
    reference = link_profiles.logical_reference_profile()
    feedback_ticks = config.microseconds_to_ticks(feedback_microseconds)
    links = link_profiles.with_path_latency(
        reference, "frame_to_controller", feedback_ticks
    )
    observation = observation_settings.ObservationSettings(
        trace="chrome", data_movement=True
    )
    return machine_settings.MachineSettings(
        workload=lowered,
        qpu=qpu,
        weak_decoder=decoder,
        links=links,
        observation=observation,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _physical_arguments(parser)
    # a prefix ends on a window boundary; the default is one window, the
    # distance's rounds, so it follows the distance
    parser.add_argument("--prefix-rounds", dest="prefix_round_count", type=int)
    parser.add_argument("--feedback-microseconds", type=float, default=4.0)
    parser.add_argument("--decoder-microseconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patch", default="memory-patch")
    # a fragments folder of the files row, such as a run's
    # inputs/<id>/fragments
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


def _recorded_values(fragments) -> dict:
    """The physical values recorded beside a fragments folder.

    physical.json's period when it binds one, and, for a run folder's
    inputs/<id>/fragments, the point's values in resolved/<id>.json, so
    a rerun from them runs the recorded point.
    """
    if fragments is None:
        return {}
    physical_path = fragments / circuit_frontend.PHYSICAL_FILE_NAME
    physical = _read_json(physical_path)
    recorded = {}
    for name, value in physical.items():
        if value is not None:
            recorded[name] = value
    point_folder = fragments.parent
    run_dir = point_folder.parent.parent
    resolved_path = (
        run_dir / run_folder.RESOLVED_FOLDER / f"{point_folder.name}.json"
    )
    if resolved_path.exists():
        record = _read_json(resolved_path)
        recorded.update(record["metadata"])
    return recorded


def _physical_parameters(arguments: argparse.Namespace, recorded: dict) -> dict:
    """The command line's physical parameters over the recorded ones.

    The defaults stand where neither names a value.
    """
    defaults = {
        "distance": 3,
        "basis": "Z",
        "physical_error_probability": 0.001,
        "round_period_microseconds": 1.1,
        "noise_model": "sd6",
        "relaxation_time_microseconds": None,
        "dephasing_time_microseconds": None,
    }
    parameters = {}
    for name, default in defaults.items():
        selected = getattr(arguments, name)
        parameters[name] = recorded.get(name, default)
        if selected is not None:
            parameters[name] = selected
    return parameters


def _program(
    arguments: argparse.Namespace, parameters: dict
) -> circuit_records.RepeatedStimCircuit:
    if arguments.input is not None:
        return circuit_frontend.read_fragments(arguments.input)
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


def _write_execution(
    folder: pathlib.Path, machine: machine_module.Machine
) -> None:
    """The history feedback selected: the executed circuit and its readout."""
    source = machine.syndrome_source
    stream_id = producers.LIVE_STREAM_ID
    circuit = source.executed_circuit(stream_id)
    circuit_path = folder / "executed.stim"
    circuit.to_file(str(circuit_path))
    measurement_rounds = source.measurement_rounds_for_stream(stream_id)
    mapping_path = folder / "measurement_rounds.json"
    run_folder.write_json(mapping_path, measurement_rounds)
    measurements = source.sampled_measurements(stream_id)
    measurement_path = folder / "measurements.json"
    run_folder.write_json(measurement_path, measurements)


def _read_json(path: pathlib.Path) -> object:
    text = path.read_text()
    return json.loads(text)


if __name__ == "__main__":
    main()
