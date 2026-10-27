"""The basic case's saved shots: one dataset per distance and error rate.

Each point's dataset holds the shot the error-model source draws for
every root seed from 0 to config.SHOTS (decsim/qpu/shot_datasets.py),
written in chunks of config.CHUNK_SHOTS root seeds, one Slurm task a chunk.

Usage
-----

From this folder, with the environment that has the deltakit extra and
this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python make_datasets.py write 11 0.005 3     # chunk 3 of d = 11, p = 0.005
python make_datasets.py seal 11 0.005        # the manifest over its chunks
python make_datasets.py sbatch > datasets.sbatch && sbatch datasets.sbatch
```

A point sealed over its first chunk alone is a smaller dataset the pilot
can read; sealing again after the rest are written covers them all.
"""

import argparse
import os
import pathlib
import sys

import decsim.experiments.sweeps as sweeps
import decsim.frontends.deltakit as deltakit
import decsim.producers as producers
import decsim.qpu.shot_datasets as shot_datasets

HERE = pathlib.Path(__file__)
config = sweeps.load_beside(__file__, "config.py")

# the slowest chunk, 50,000 draws at d = 11 at about 40 ms each
CHUNK_HOURS = 2


def main() -> None:
    """Write a chunk, seal a point, or print the Slurm array for all."""
    parser = _parser()
    parsed = parser.parse_args()
    if parsed.command == "sbatch":
        _print_sbatch()
        return
    distance = parsed.distance
    probability = parsed.physical_error_probability
    workload = _memory_workload(distance, probability)
    operation = workload.operations[0]
    folder_name = config.dataset_folder(distance, probability)
    folder = pathlib.Path(folder_name)
    if parsed.command == "write":
        _write(folder, operation, parsed.chunk)
        return
    description = _description(distance, probability)
    shot_datasets.seal(folder, operation.circuit, operation.id, description)


def _memory_workload(distance: int, probability: float):
    """The memory the machine runs (config.memory), made the same way."""
    rounds = config.constants.rounds_per_distance * distance
    sd6 = deltakit.SD6Noise()
    return producers.memory_circuit(
        config.CODE_TASK, rounds, distance, probability, sd6, True
    )


def _write(folder: pathlib.Path, operation, chunk: int) -> None:
    chunk_shots = config.CHUNK_SHOTS
    first_root_seed = chunk * chunk_shots
    folder.mkdir(parents=True, exist_ok=True)
    shot_datasets.write_chunk(
        folder, operation.circuit, operation.id, first_root_seed, chunk_shots
    )


def _description(distance: int, probability: float) -> dict:
    rounds = config.constants.rounds_per_distance * distance
    return {
        "code_task": config.CODE_TASK,
        "distance": distance,
        "physical_error_probability": probability,
        "rounds": rounds,
        "noise": "sd6",
        "noiseless_time_boundaries": True,
    }


def _print_sbatch() -> None:
    """One array task per chunk of every point; each seals nothing.

    The tasks run chunk by chunk, so every point's first chunk, which the
    pilot reads, is written before any point's second.
    """
    max_shots = config.SHOTS
    chunk_shots = config.CHUNK_SHOTS
    chunk_count = max_shots // chunk_shots
    probability_count = len(config.PHYSICAL_ERROR_PROBABILITIES)
    point_count = len(config.DISTANCES) * probability_count
    last_task = point_count * chunk_count - 1
    distances = " ".join(str(distance) for distance in config.DISTANCES)
    probabilities = " ".join(
        str(probability) for probability in config.PHYSICAL_ERROR_PROBABILITIES
    )
    python = os.path.normpath(sys.executable)
    checkout = HERE.parent.parent.parent
    logs = checkout.parent / "logs"
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=basic_case_datasets",
        f"#SBATCH --array=0-{last_task}",
        "#SBATCH --partition=cpu",
        "#SBATCH --ntasks=1 --cpus-per-task=1 --mem=8G",
        f"#SBATCH --time={CHUNK_HOURS}:00:00",
        f"#SBATCH --output={logs}/basic_case_datasets_%A_%a.out",
        f"distances=({distances})",
        f"probabilities=({probabilities})",
        f"chunk=$((SLURM_ARRAY_TASK_ID / {point_count}))",
        f"point=$((SLURM_ARRAY_TASK_ID % {point_count}))",
        f"distance=${{distances[$((point / {probability_count}))]}}",
        f"probability=${{probabilities[$((point % {probability_count}))]}}",
        f"cd {HERE.parent}",
        f"export PYTHONPATH={checkout}",
        f"{python} make_datasets.py write $distance $probability $chunk",
    ]
    text = "\n".join(lines)
    print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="draw and save one chunk")
    seal = commands.add_parser("seal", help="write a point's manifest")
    for command in (write, seal):
        command.add_argument("distance", type=int)
        command.add_argument("physical_error_probability", type=float)
    write.add_argument("chunk", type=int)
    commands.add_parser("sbatch", help="print the Slurm array for every chunk")
    return parser


if __name__ == "__main__":
    main()
