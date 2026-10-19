"""The rate controller against the static threshold, on the same shots.

    python controller_plots.py CONTROLLER_FOLDER HELDOUT_FOLDER

CONTROLLER_FOLDER is rate_controller.py's results, HELDOUT_FOLDER
threshold_heldout.py's. Writes into CONTROLLER_FOLDER/plots/:
  - controller.csv: per point, target and start, the controller's
    escalated share and failures beside the static threshold's for the
    same target on the same seeds, and the shots only one of them failed
    (a paired comparison: McNemar's test reads those two counts);
  - tracking.png: each controller's threshold against the windows it
    has seen, a row per d and a column per p, the calibrated threshold
    dashed.
"""

import collections
import csv
import pathlib
import sys

import matplotlib

import decsim.experiments.sweeps as sweeps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

calibration = sweeps.load_beside(__file__, "calibration.py")
PROBABILITY = "workload.arguments.physical_error_probability"
DISTANCE = "qpu.distance"
FIELDS = (
    "d",
    "p",
    "target",
    "start",
    "shots",
    "controller_share",
    "static_share",
    "controller_failures",
    "static_failures",
    "only_controller_failed",
    "only_static_failed",
    "final_threshold_db",
    "calibrated_db",
)
STYLES = {
    (0.05, "calibrated"): ("tab:blue", "-"),
    (0.05, "cold"): ("tab:blue", ":"),
    (0.1, "calibrated"): ("tab:orange", "-"),
    (0.1, "cold"): ("tab:orange", ":"),
}


def main(controller: pathlib.Path, heldout: pathlib.Path) -> None:
    """The table and the tracking figure."""
    folder = controller / "plots"
    folder.mkdir(exist_ok=True)
    controller_shots_path = controller / "shots.csv"
    heldout_shots_path = heldout / "shots.csv"
    summary_path = controller / "threshold_summary.csv"
    controller_shots = _read(controller_shots_path)
    heldout_shots = _read(heldout_shots_path)
    summaries = _read(summary_path)
    write_table(controller_shots, heldout_shots, summaries, folder)
    plot_tracking(controller, folder)


def write_table(controller_shots, heldout_shots, summaries, folder) -> None:
    """One row per controller task, beside the static run on its seeds."""
    controllers = _shots_by_task(controller_shots, _controller_key)
    statics = _shots_by_task(heldout_shots, _static_key)
    finals = {_controller_key(row): row["threshold_db"] for row in summaries}
    rows = []
    for key in sorted(controllers):
        distance, probability, target, start = key
        controller_by_seed = controllers[key]
        static_by_seed = statics[(distance, probability, target)]
        seeds = sorted(controller_by_seed)
        paired = _paired_counts(controller_by_seed, static_by_seed, seeds)
        calibrated = calibration.CALIBRATED[(distance, probability)][target]
        final = round(float(finals[key]), 2)
        rows.append((*key, len(seeds), *paired, final, calibrated))
    path = folder / "controller.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"wrote {path}")


def _paired_counts(controller: dict, static: dict, seeds: list) -> tuple:
    """Shares, failures, and the shots only one of the two failed."""
    windows = 0
    controller_escalated = 0
    static_escalated = 0
    controller_failed = 0
    static_failed = 0
    only_controller = 0
    only_static = 0
    for seed in seeds:
        mine = controller[seed]
        theirs = static[seed]
        windows += int(mine["decoded_windows"])
        controller_escalated += int(mine["escalated_windows"])
        static_escalated += int(theirs["escalated_windows"])
        mine_failed = mine["logical_failure"] == "True"
        theirs_failed = theirs["logical_failure"] == "True"
        controller_failed += mine_failed
        static_failed += theirs_failed
        only_controller += mine_failed and not theirs_failed
        only_static += theirs_failed and not mine_failed
    controller_share = controller_escalated / windows
    static_share = static_escalated / windows
    return (
        controller_share,
        static_share,
        controller_failed,
        static_failed,
        only_controller,
        only_static,
    )


def plot_tracking(controller: pathlib.Path, folder: pathlib.Path) -> None:
    """Each controller's threshold over the windows, a panel per point."""
    trajectories = _trajectories(controller)
    by_point = collections.defaultdict(dict)
    for key, samples in trajectories.items():
        by_point[key[:2]][key] = samples
    points = sorted(by_point)
    distances = sorted({distance for distance, _ in points})
    probabilities = sorted({probability for _, probability in points})
    rows, columns = len(distances), len(probabilities)
    size = (4 * columns, 3.2 * rows)
    figure, axes_grid = plt.subplots(rows, columns, figsize=size, squeeze=False)
    for point in points:
        distance, probability = point
        row = distances.index(distance)
        column = probabilities.index(probability)
        axes = axes_grid[row][column]
        for key, samples in by_point[point].items():
            _draw_trajectory(axes, key, samples)
        axes.set_title(f"d = {distance}, p = {probability:g}")
        axes.set_xlabel("windows")
        axes.set_ylabel("threshold (dB)")
        axes.grid(True, alpha=0.3)
    axes_grid[0][0].legend(fontsize=7)
    figure.tight_layout()
    path = folder / "tracking.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def _draw_trajectory(axes, key: tuple, samples: list) -> None:
    distance, probability, target, start = key
    color, style = STYLES[(target, start)]
    windows = [window for window, _ in samples]
    thresholds = [threshold for _, threshold in samples]
    label = f"{target:g} from {start}"
    axes.plot(windows, thresholds, style, color=color, lw=1, label=label)
    calibrated = calibration.CALIBRATED[(distance, probability)][target]
    axes.axhline(calibrated, color=color, ls="--", lw=0.8)


def _trajectories(controller: pathlib.Path) -> dict:
    """(window count, threshold dB) of every task, by its key."""
    trajectories = collections.defaultdict(list)
    paths = controller.glob("online_threshold_*.csv")
    for path in sorted(paths):
        for row in _read(path):
            key = _controller_key(row)
            window_count = int(row["window_count"])
            threshold = float(row["threshold_db"])
            trajectories[key].append((window_count, threshold))
    return trajectories


def _shots_by_task(rows: list, key_of) -> dict:
    grouped = collections.defaultdict(dict)
    for row in rows:
        key = key_of(row)
        seed = int(row["seed"])
        grouped[key][seed] = row
    return grouped


def _controller_key(row: dict) -> tuple:
    return (*_static_key(row), row["start"])


def _static_key(row: dict) -> tuple:
    distance = int(row[DISTANCE])
    probability = float(row[PROBABILITY])
    target = float(row["target"])
    return (distance, probability, target)


def _read(path: pathlib.Path) -> list:
    with path.open() as rows_file:
        reader = csv.DictReader(rows_file)
        return list(reader)


if __name__ == "__main__":
    controller_folder = pathlib.Path(sys.argv[1])
    heldout_folder = pathlib.Path(sys.argv[2])
    main(controller_folder, heldout_folder)
