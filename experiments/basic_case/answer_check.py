r"""Each window's weak and strong answers, checked, on the basic case.

The basic case's switching machine (config.py) on the flip stream, the
source that copies its state each round, with every window's two
answers checked against the truth (observe/answer_checks.py). One
stream gives, on the same shot, union-find alone (the weak chain),
belief matching alone (the strong chain), the switching machine's own
outcome, the per-shot best-of-two line (the shot fails only when both
alone fail; Sheth et al. 1905.02345 lines 186-196) and the per-window
line (the best-of-two chain). A stream is 10 d rounds, as a basic case
shot is, and each starts its own machine.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md), one
process a block of seeds; answer_check_plots.py reads the folder:

```
export PYTHONPATH=$PWD/../..
OUT=$PWD/../../results/basic_case_answer_check; mkdir -p $OUT
python answer_check.py 7 0.003 0 100 $OUT/d7_p0.003_0.jsonl
```
"""

import dataclasses
import json
import pathlib
import sys
import time

import decsim
import decsim.experiments.stream_command as stream_command
import decsim.experiments.sweeps as sweeps
import decsim.observe.settings as observe_settings
import decsim.qpu.flip_stream_device as flip_stream_device

config = sweeps.load_beside(__file__, "config.py")
PATHS = observe_settings.ANSWER_CHECK_PATHS


def task(distance: int, physical_error_probability: float) -> decsim.Task:
    """The switching machine on the flip stream, its answers checked."""
    machine = config.machine("switching", distance, physical_error_probability)
    qpu = machine.held("qpu")
    source = flip_stream_device.FlipStreamDevice.Settings()
    streamed = dataclasses.replace(qpu, source=source)
    observation = dataclasses.replace(
        machine.observation, answer_check_paths=PATHS
    )
    machine = machine.with_held("qpu", streamed)
    machine = dataclasses.replace(machine, observation=observation)
    return decsim.Task("switching_answer_check", machine)


def run(
    distance: int,
    physical_error_probability: float,
    first_seed: int,
    count: int,
    out: str,
) -> None:
    """One json line a stream: each outcome, and each path's window rows."""
    rounds = config.constants.rounds_per_distance * distance
    path = pathlib.Path(out)
    after_last_seed = first_seed + count
    with path.open("w") as handle:
        for seed in range(first_seed, after_last_seed):
            # a fresh task starts a fresh machine for each stream
            stream_task = task(distance, physical_error_probability)
            started = time.perf_counter()
            record = stream_command.stream_record(stream_task, rounds, 0, seed)
            seconds = time.perf_counter() - started
            line = _line(record, distance, physical_error_probability, seed)
            line["seconds"] = round(seconds, 3)
            handle.write(json.dumps(line) + "\n")
            handle.flush()


def _line(
    record: dict, distance: int, physical_error_probability: float, seed: int
) -> dict:
    measurement = record["measurement"]
    chains = {row["path"]: row for row in measurement["answer_checks"]}
    return {
        "distance": distance,
        "p": physical_error_probability,
        "seed": seed,
        "rounds": measurement["rounds"],
        "switching_failed": measurement["logical_failure"],
        "escalated_windows": measurement["escalated_windows"],
        "failed": {path: chains[path]["failed"][0] for path in PATHS},
        "windows": {path: chains[path]["windows"][0] for path in PATHS},
    }


if __name__ == "__main__":
    run(
        int(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5],
    )
