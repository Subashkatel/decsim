"""Warm seconds a shot of this checkout against a base one, on one node.

    python tools/speed_check.py <base checkout> [--repeats 3]
        [--slowdown 0.2]

design.md section 8.3 asks for a speed test that fails on a slowdown
past 20%. One tree timed in two Slurm jobs on nodes of one type differed
by up to 20% (stage7_plan.md), so a stored baseline cannot hold that
line. This check times the base checkout and this one on the same node,
one after the other, `repeats` times each, and compares the medians.
Run it as one Slurm job, for example
`srun -c 2 python tools/speed_check.py ../decsim_base`.

Each timing is a fresh process run in the checkout it times. The
task's first shot builds its window models and is left out, as a long
piece spreads it over its shots; the timing is the mean wall seconds of
the warm shots, built, run and measured as a piece runs them
(decsim/experiments/collect.py, run_unit). The check exits 1 when a
task's median here passes its base median by more than the slowdown.
"""

import argparse
import os
import pathlib
import statistics
import subprocess
import sys
import time

TOOL_FILE = pathlib.Path(__file__)
TOOL_PATH = TOOL_FILE.resolve()
CHECKOUT = TOOL_PATH.parent.parent
# run file, task, shots: the tasks stage7_plan.md profiled, one of each
# kind of machine, each run for one first shot and its warm ones
TASKS = (
    ("experiments/switching_baseline/run.py", "weak_alone_d5_p0.003", 6),
    ("experiments/switching_baseline/run.py", "weak_alone_d13_p0.003", 4),
    ("experiments/switching_baseline/run.py", "switching_d5_p0.003", 4),
    ("experiments/switching_baseline/run.py", "switching_d11_p0.003", 3),
    (
        "experiments/switching_per_window/run.py",
        "union_find_alone_d9_p0.003",
        5,
    ),
    (
        "experiments/switching_per_window/run.py",
        "relay_bp5_alone_d9_p0.003",
        3,
    ),
    ("experiments/data_movement/run.py", "every_hop_copies_d7", 5),
)
BASE = "base"
HERE = "here"


def main(argv: list) -> None:
    """Time a task in this process, or compare the two checkouts."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    if parsed.time is not None:
        run_file, task_name, shots = parsed.time
        shot_count = int(shots)
        seconds = warm_seconds_per_shot(run_file, task_name, shot_count)
        print(seconds)
        return
    base_path = pathlib.Path(parsed.base)
    base = base_path.resolve()
    timings = alternating_timings(base, parsed.repeats)
    table = timing_table(timings)
    print(table)
    slower = slower_tasks(timings, parsed.slowdown)
    if slower:
        raise SystemExit(
            f"slower than the base by more than the slowdown: {slower}"
        )


def warm_seconds_per_shot(run_file: str, task_name: str, shots: int) -> float:
    """The mean wall seconds of the task's shots after its first."""
    import decsim.experiments.collect as collect
    import decsim.experiments.experiment as experiment
    import decsim.experiments.measure as measure
    import decsim.windows.built_window_models as built_window_models

    study = experiment.load_one_task(run_file, task_name)
    (task,) = study.tasks
    built = built_window_models.BuiltWindowModels()
    warm_seconds = []
    for seed in range(shots):
        start = time.perf_counter()
        shot = collect.run_shot(task, seed, built_models=built)
        measure.measure_shot(shot, None)
        seconds = time.perf_counter() - start
        warm_seconds.append(seconds)
    return statistics.mean(warm_seconds[1:])


def alternating_timings(base: pathlib.Path, repeats: int) -> dict:
    """Each task's timings in the base checkout and here, taken in turn."""
    timings = {}
    for run_file, task_name, shots in TASKS:
        task = (run_file, task_name, shots)
        timings[task_name] = _alternating_timings_of(base, task, repeats)
    return timings


def slower_tasks(timings: dict, slowdown: float) -> list:
    """The tasks whose median here passes the base median by the slowdown."""
    slower = []
    for task_name, sides in timings.items():
        base_median = statistics.median(sides[BASE])
        here_median = statistics.median(sides[HERE])
        allowed = base_median * (1 + slowdown)
        if here_median > allowed:
            slower.append(task_name)
    return slower


def timing_table(timings: dict) -> str:
    """One line a task: its median seconds a shot in the base and here."""
    lines = ["task  base_seconds_per_shot  here_seconds_per_shot"]
    for task_name, sides in timings.items():
        base_median = statistics.median(sides[BASE])
        here_median = statistics.median(sides[HERE])
        lines.append(f"{task_name}  {base_median:.4f}  {here_median:.4f}")
    return "\n".join(lines)


def _alternating_timings_of(
    base: pathlib.Path, task: tuple, repeats: int
) -> dict:
    """One task timed in the base checkout, then here, repeats times."""
    sides = {BASE: [], HERE: []}
    for _ in range(repeats):
        base_seconds = _timed_in(base, *task)
        sides[BASE].append(base_seconds)
        here_seconds = _timed_in(CHECKOUT, *task)
        sides[HERE].append(here_seconds)
    return sides


def _timed_in(
    checkout: pathlib.Path, run_file: str, task_name: str, shots: int
) -> float:
    """One timing, in a fresh process importing decsim from the checkout."""
    environment = dict(os.environ, PYTHONPATH=str(checkout))
    command = [
        sys.executable,
        str(TOOL_PATH),
        "--time",
        run_file,
        task_name,
        str(shots),
    ]
    completed = subprocess.run(
        command,
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = completed.stdout.splitlines()
    return float(lines[-1])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools/speed_check.py")
    parser.add_argument("base", nargs="?", help="the checkout to time against")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slowdown", type=float, default=0.2)
    parser.add_argument(
        "--time",
        nargs=3,
        metavar=("RUN_FILE", "TASK", "SHOTS"),
        help="time one task in this process and print its warm seconds",
    )
    return parser


if __name__ == "__main__":
    main(sys.argv[1:])
