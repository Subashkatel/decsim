"""Run memory until decoded feedback permits an actual final readout.

Four canonical Stim fragments reproduce the same live history without the
optional producer. Runtime feedback changes how long the data stays live.
The fragments are read and written in the files row's form: a fragments
folder with the four .stim files and physical.json, which a results
folder keeps in points/<name>/inputs/fragments.
"""

import argparse
import json
import pathlib

import decsim.collect as collect
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.experiments.run_folder as run_folder
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.producers as producers
import decsim.qpu.settings as qpu_settings
import decsim.records.circuits as circuit_records
import decsim.settings as machine_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# The one point this example runs, which names its folder in points/.
POINT_NAME = "shot"
# The yaml path each physical value of a point sits at: the run's
# metadata names its values by it, as a yaml sweep's does.
METADATA_PATHS = {
    "physical_error_probability": (
        "workload.arguments.physical_error_probability"
    ),
    "distance": "qpu.distance",
    "round_period_microseconds": "qpu.round_period_microseconds",
}
# The values a loaded fragments folder's circuit already holds. A replay
# reads them from the record beside the fragments, as unknown when none
# names them, and refuses a flag for one, which could only relabel a
# circuit it does not change.
CIRCUIT_VALUES = (
    "basis",
    "physical_error_probability",
    "noise_model",
    "relaxation_time_microseconds",
    "dephasing_time_microseconds",
)


def main() -> None:
    """Save canonical inputs and the physical history selected by feedback.

    The output folder is a results folder
    (decsim/experiments/run_folder.py): its run.json, every value of the
    run and its workload under points/shot/, the shot's files as decsim
    run writes them and the finished time, beside the executed history
    and the arguments.
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
    metadata = {}
    for name, path in METADATA_PATHS.items():
        metadata[path] = parameters[name]
    task = collect.Task(settings, metadata)
    point_ids = [task.strong_id()]
    started_utc = run_folder.start_run(arguments.output, None, point_ids)
    seeds = [(arguments.seed, 1)]
    run_folder.record_point(arguments.output, POINT_NAME, task, seeds)
    result = machine.run()
    label = f"seed{arguments.seed}"
    run_folder.write_shot(machine, settings, arguments.output, label, result)
    _write_execution(arguments.output, machine)
    selected_arguments = vars(arguments)
    argument_values = dict(selected_arguments)
    argument_values.update(parameters)
    argument_path = arguments.output / "arguments.json"
    argument_json = collect.json_value(argument_values)
    run_folder.write_json(argument_path, argument_json)
    run_folder.finish_run(arguments.output, None, point_ids, started_utc)
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
    engine = decoder_settings.EngineSettings(clock=clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=decoder_microseconds
    )
    decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching, engine=engine
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
    # points/<name>/inputs/fragments
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
    points/<name>/inputs/fragments, the arguments the run that made them
    recorded and the point's values in its machine.json, so a rerun from them
    runs, and names, the recorded point.
    """
    if fragments is None:
        return {}
    physical_path = fragments / workload_files.PHYSICAL_FILE_NAME
    physical = _read_json(physical_path)
    inputs_dir = fragments.parent
    point_folder = inputs_dir.parent
    points_dir = point_folder.parent
    run_dir = points_dir.parent
    recorded = _recorded_arguments(run_dir)
    for name, value in physical.items():
        if value is not None:
            recorded[name] = value
    record_path = point_folder / run_folder.RECORD_FILE
    if record_path.exists():
        record = _read_json(record_path)
        metadata_values = _metadata_values(record["metadata"])
        recorded.update(metadata_values)
    return recorded


def _recorded_arguments(run_dir: pathlib.Path) -> dict:
    """The circuit's values the run that saved the fragments recorded."""
    arguments_path = run_dir / "arguments.json"
    if not arguments_path.exists():
        return {}
    arguments = _read_json(arguments_path)
    values = {}
    for name in CIRCUIT_VALUES:
        if name in arguments:
            values[name] = arguments[name]
    return values


def _metadata_values(metadata: dict) -> dict:
    """The physical values a recorded point's metadata holds, by name."""
    values = {}
    for name, path in METADATA_PATHS.items():
        if path in metadata:
            values[name] = metadata[path]
    return values


def _physical_parameters(arguments: argparse.Namespace, recorded: dict) -> dict:
    """The command line's physical parameters over the recorded ones.

    The defaults stand where neither names a value; loaded fragments
    take no default for a value their circuit holds.
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
    if arguments.input is not None:
        _refuse_a_circuit_flag(arguments)
        for name in CIRCUIT_VALUES:
            defaults[name] = None
    parameters = {}
    for name, default in defaults.items():
        selected = getattr(arguments, name)
        parameters[name] = recorded.get(name, default)
        if selected is not None:
            parameters[name] = selected
    return parameters


def _refuse_a_circuit_flag(arguments: argparse.Namespace) -> None:
    """A flag for a value the loaded circuit holds would relabel it."""
    for name in CIRCUIT_VALUES:
        if getattr(arguments, name) is None:
            continue
        flag = name.replace("_", "-")
        raise SystemExit(
            f"--{flag} cannot change the fragments --input loads, whose "
            "circuit already holds it; the record beside them names it"
        )


def _program(
    arguments: argparse.Namespace, parameters: dict
) -> circuit_records.RepeatedStimCircuit:
    if arguments.input is not None:
        return workload_files.read_fragments(arguments.input)
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
    source = machine.qpu.syndrome_source
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
