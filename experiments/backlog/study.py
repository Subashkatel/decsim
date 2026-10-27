"""Run the backlog study: one long stream a point and seed.

    python study.py RESULTS PROCESSES

Two studies on config.py's machine. switching: every tau_weak and
gamma_switch of constants.yaml, r_com by App. C's rule, as Fig. 21
(2510.25222). weak_only: no escalation, r_com = r_buf = d, tau_weak
around Eq. (7)'s bound tau_gen / 2. A window's backlog is the rounds
measured and not yet committed when it commits: its d buffer rounds and
its reaction, data complete to commit, in rounds. RESULTS/streams.csv
holds a row a stream; RESULTS/trajectories.csv every window of seed 0.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
sbatch -N1 -c32 --mem=96G -t 2:00:00 --wrap "python study.py RESULTS 32"
python plots.py RESULTS
```
"""

import concurrent.futures
import csv
import itertools
import pathlib
import statistics
import sys

import decsim
import decsim.config as decsim_config
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
STREAM_FIELDS = (
    "study",
    "weak_fraction",
    "gamma",
    "window_rate",
    "seed",
    "commit_rounds",
    "load",
    "windows",
    "escalated",
    "mean_backlog_rounds",
    "max_backlog_rounds",
    "final_backlog_rounds",
)
TRAJECTORY_FIELDS = (
    "study",
    "weak_fraction",
    "gamma",
    "window",
    "committed_us",
    "tier",
    "backlog_rounds",
)


def main(results: pathlib.Path, processes: int) -> None:
    """Run every stream and write the two tables."""
    results.mkdir(parents=True, exist_ok=True)
    points = _points()
    streams = []
    trajectories = []
    with concurrent.futures.ProcessPoolExecutor(processes) as pool:
        for stream, trajectory in pool.map(run_stream, points):
            streams.append(stream)
            trajectories.extend(trajectory)
    streams_path = results / "streams.csv"
    trajectories_path = results / "trajectories.csv"
    _write(streams_path, STREAM_FIELDS, streams)
    _write(trajectories_path, TRAJECTORY_FIELDS, trajectories)


def _points() -> list:
    """(study, tau_weak / tau_gen, gamma, seed) of every stream."""
    seeds = range(constants.seeds)
    grid = itertools.product(
        constants.weak_decode_fractions, constants.switching_rates, seeds
    )
    points = []
    for fraction, gamma, seed in grid:
        points.append(("switching", fraction, gamma, seed))
    for fraction in constants.weak_only_fractions:
        points.append(("weak_only", fraction, 0.0, 0))
    return points


def run_stream(point: tuple) -> tuple:
    """One stream's row, and its windows' rows when it is seed 0."""
    study, fraction, gamma, seed = point
    commit = constants.distance
    if study == "switching":
        commit = config.commit_rounds(fraction)
    rate = config.window_rate(gamma, commit)
    machine = config.machine(fraction, rate, commit)
    task = decsim.Task(study, machine, {})
    shot = collect.run_shot(task, seed)
    windows = measure.decoded_windows(shot.machine.observation)
    trajectory = _trajectory(point, windows)
    backlogs = [row[-1] for row in trajectory]
    tiers = [row[-2] for row in trajectory]
    escalated = tiers.count("strong")
    if seed != 0:
        trajectory = []
    load = config.load(fraction, rate, commit)
    mean_backlog = statistics.fmean(backlogs)
    stream = (
        study,
        fraction,
        gamma,
        rate,
        seed,
        commit,
        load,
        len(windows),
        escalated,
        mean_backlog,
        max(backlogs),
        backlogs[-1],
    )
    return stream, trajectory


def _trajectory(point: tuple, windows: list) -> list:
    """A row a window: its commit, tier and backlog in rounds."""
    study, fraction, gamma, _seed = point
    rows = []
    for window, frame in windows:
        backlog = _backlog_rounds(window, frame)
        committed_us = decsim_config.ticks_to_microseconds(
            frame.committed_ticks
        )
        row = (study, fraction, gamma, window.window_index)
        rows.append((*row, committed_us, frame.tier, backlog))
    return rows


def _backlog_rounds(window, frame) -> float:
    """Rounds measured, not committed: the buffer and the reaction."""
    reaction_ticks = frame.committed_ticks - window.t_data_complete
    reaction_us = decsim_config.ticks_to_microseconds(reaction_ticks)
    reaction_rounds = reaction_us / constants.round_period_microseconds
    return constants.distance + reaction_rounds


def _write(path: pathlib.Path, fields: tuple, rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    process_count = int(sys.argv[2])
    main(folder, process_count)
