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

import decsim.collect as collect
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.experiments.run_folder as run_folder
import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.settings as observation_settings
import decsim.producers as producers
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.records.workload as workload_records
import decsim.settings as machine_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# The one point this example runs, which names its folder in points/.
POINT_NAME = "shot"


def main() -> None:
    """Write the input circuit, settings, result and trace into one folder.

    The folder is a results folder (decsim/experiments/run_folder.py):
    its run.json, every value of the run and its workload under
    points/shot/, the shot's files as decsim run writes them and the
    finished time, beside the arguments and the setup time.
    """
    arguments = _arguments()
    _resolve_memory_parameters(arguments)
    if arguments.mode == "protection" and arguments.family == "repetition":
        raise ValueError(
            "the protection example requires rotated_surface memory"
        )
    started = time.perf_counter()
    circuit, measurement_rounds = _circuit(arguments)
    workload = memory_workload(
        circuit, measurement_rounds, arguments.rounds, arguments.patch
    )
    if arguments.mode == "protection":
        workload = protection_workload(
            circuit,
            measurement_rounds,
            arguments.rounds,
            arguments.prefix_rounds,
            arguments.patch,
        )
    settings = supplied_settings(
        workload,
        distance=arguments.distance,
        round_count=arguments.rounds,
        period_microseconds=arguments.period_microseconds,
        feedback_microseconds=arguments.feedback_microseconds,
        decoder_microseconds=arguments.decoder_microseconds,
    )
    if arguments.family == "repetition":
        code_card = RepetitionMemory.Settings(
            arguments.distance, arguments.rounds
        )
        qpu = dataclasses.replace(
            settings.qpu, distance=None, code_card=code_card
        )
        settings = dataclasses.replace(settings, qpu=qpu)
    machine = machine_module.Machine.build(settings, arguments.seed)
    prepared = time.perf_counter()
    setup_seconds = prepared - started
    task = _point_task(arguments, settings)
    point_ids = [task.strong_id()]
    started_utc = run_folder.start_run(arguments.output, None, point_ids)
    seeds = [(arguments.seed, 1)]
    run_folder.record_point(arguments.output, POINT_NAME, task, seeds)
    result = machine.run()
    label = f"seed{arguments.seed}"
    run_folder.write_shot(machine, settings, arguments.output, label, result)
    argument_path = arguments.output / "arguments.json"
    selected_arguments = vars(arguments)
    argument_values = collect.json_value(selected_arguments)
    run_folder.write_json(argument_path, argument_values)
    setup_path = arguments.output / "setup_seconds.json"
    run_folder.write_json(setup_path, setup_seconds)
    run_folder.finish_run(arguments.output, None, point_ids, started_utc)
    print(f"complete: {arguments.output}; setup {setup_seconds:.6f} seconds")


def memory_workload(
    circuit: stim.Circuit,
    measurement_rounds: dict[int, int],
    round_count: int,
    patch: str,
) -> workload_records.Workload:
    """One finite memory, including its final data measurement."""
    operation = program_records.Operation(
        1, "memory", (patch,), patches=(patch,)
    )
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    return workload_records.Workload((operation,), {1: round_count}, physical)


def protection_workload(
    circuit: stim.Circuit,
    measurement_rounds: dict[int, int],
    round_count: int,
    prefix_round_count: int,
    patch: str,
) -> workload_records.Workload:
    """One history protected until a declared horizon after a feedback wait.

    The continuation is an identity operation with a scheduling dependency
    on the prefix's decoded result. The physical circuit remains memory.
    A late continuation exhausts the finite source loudly. The stream's
    owner, its protected region and its round counts are derived when
    the workload is lowered (decsim/frontends/circuit_frontend.py).
    """
    if not 1 <= prefix_round_count < round_count:
        raise ValueError("prefix rounds must lie inside the finite horizon")
    patches = (patch,)
    prefix = program_records.Operation(
        1,
        "prefix",
        patches,
        patches=patches,
        stream_id=producers.LIVE_STREAM_ID,
        stream_offset=0,
    )
    begin = program_records.Operation(
        2,
        "begin-protection",
        patches,
        patches=patches,
        emits_detector_data=False,
    )
    resume = program_records.Operation(
        3,
        "continue-memory",
        patches,
        patches=patches,
        blocked_by=1,
        emits_detector_data=False,
    )
    finish = program_records.Operation(
        4,
        "final-readout",
        patches,
        patches=patches,
        scheduled_start_round=round_count,
        emits_detector_data=False,
    )
    operations = (prefix, begin, resume, finish)
    round_counts = {1: prefix_round_count, 2: 0, 3: 1, 4: 0}
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    return workload_records.Workload(operations, round_counts, physical)


def supplied_settings(
    workload: workload_records.Workload,
    *,
    distance: int,
    round_count: int,
    period_microseconds: float,
    feedback_microseconds: float,
    decoder_microseconds: float = 0.1,
) -> machine_settings.MachineSettings:
    """The machine that samples a finite circuit on the Stim source.

    The Stim source is built from the workload's circuit and round map,
    whatever made them.
    """
    physical = workload.physical
    if physical.circuit.num_observables != 1:
        raise ValueError("the memory example requires one logical observable")
    schedule = dict(physical.measurement_rounds)
    declared_rounds = schedule.values()
    if max(declared_rounds) != round_count:
        raise ValueError(
            "the declared horizon must equal the final readout round"
        )
    lowered = workload_settings.WorkloadSettings.running(workload)
    stim_source = stim_device.StimDevice.Settings()
    qpu = qpu_settings.QpuSettings(
        source=stim_source,
        distance=distance,
        round_period_microseconds=period_microseconds,
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
    links = link_profiles.with_path_latency(
        reference, "frame_to_controller", feedback_microseconds
    )
    observation = observation_settings.ObservationSettings(
        trace="chrome",
        data_movement=True,
    )
    return machine_settings.MachineSettings(
        workload=lowered,
        qpu=qpu,
        weak_decoder=decoder,
        links=links,
        observation=observation,
    )


@dataclasses.dataclass(frozen=True)
class RepetitionMemory:
    """Whole-shot repetition card, with one check per neighbouring pair.

    This example chooses a single finite decode window.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The card's own keys."""

        distance: int
        round_count: int

        def build(
            self,
            distance: Optional[int],
            commit_rounds_override: Optional[int],
            buffer_rounds_override: Optional[int],
        ) -> "RepetitionMemory":
            """The whole-shot card; the qpu sets no distance or window size."""
            del distance, commit_rounds_override, buffer_rounds_override
            return RepetitionMemory(self.distance, self.round_count)

    distance: int
    round_count: int
    name: str = "repetition memory"

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

    def spatial_nodes(self, patch_count: int) -> int:
        """Count the check nodes in one repetition round."""
        return self.syndrome_bits_per_round(patch_count)

    def syndrome_bits_per_round(self, patch_count: int) -> int:
        """One stabilizer measurement per neighbouring data pair."""
        checks_per_patch = self.distance - 1
        return patch_count * checks_per_patch


def _circuit(arguments) -> tuple[stim.Circuit, dict[int, int]]:
    if arguments.input is None:
        return deltakit.memory_circuit(
            arguments.family,
            arguments.distance,
            arguments.rounds,
            arguments.basis,
            arguments.probability,
        )
    points_folder = arguments.input / run_folder.POINTS_FOLDER
    (point_folder,) = points_folder.iterdir()
    inputs_folder = point_folder / run_folder.INPUTS_FOLDER
    operations_path = inputs_folder / "operations.json"
    circuit_path = inputs_folder / "circuit.stim"
    rounds_path = inputs_folder / "measurement_rounds.json"
    workload = workload_files.read_workload(
        operations_path, circuit_path, rounds_path
    )
    physical = workload.physical
    return physical.circuit, physical.measurement_rounds


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
    # a prefix ends on a window boundary; the default is one window, the
    # distance's rounds, so it follows the distance
    parser.add_argument("--prefix-rounds", type=int)
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
    if arguments.prefix_rounds is None:
        arguments.prefix_rounds = arguments.distance


def _check_replay_parameter(input_folder, name, selected, recorded) -> None:
    if input_folder is None or selected is None:
        return
    if selected != recorded:
        raise ValueError(f"{name} differs from the exported circuit parameters")


def _point_task(arguments, settings) -> collect.Task:
    """The run's one point, its values the command line's."""
    metadata = {
        "workload.arguments.physical_error_probability": arguments.probability,
        "qpu.distance": arguments.distance,
        "qpu.round_period_microseconds": arguments.period_microseconds,
    }
    return collect.Task(settings, metadata)


if __name__ == "__main__":
    main()
