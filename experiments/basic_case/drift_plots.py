"""The drift run's figure and table.

    python drift_plots.py DRIFT_FOLDER CALIBRATION_CSV

DRIFT_FOLDER holds drift.py's stream records, CALIBRATION_CSV is
threshold_plots.py's calibration.csv. Writes into DRIFT_FOLDER/plots/:
  - drift.png: per d, the escalated share of each hundred windows against
    the round, static, controller and loop, each seed thin and their mean
    thick, the target dotted and q* before and after the drift grey
    (top); each moving threshold against the round, the thresholds
    calibrated for the target at the two p grey (bottom);
  - drift.csv: per task, means over its seeds: the escalated share before
    and after the drift, and the drifted half's mean reaction (a
    window's last round to its commit) and its weak tier's reaction
    growth per us of stream, above 0 when the chain falls behind.
"""

import collections
import csv
import json
import pathlib
import statistics
import sys

import matplotlib

import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

drift = sweeps.load_beside(__file__, "drift.py")
CONTROLS = {"static": "tab:red", "rate": "tab:blue", "loop": "tab:green"}
# the controls whose threshold moves
MOVING = ("rate", "loop")
# the lines of p before the drift and after it
GUIDE_STYLES = ("--", "-.")
FIELDS = (
    "d",
    "control",
    "seeds",
    "share_before",
    "share_after",
    "reaction_after_us",
    "weak_growth_after",
    "logical_failures",
)


def main(folder: pathlib.Path, calibration_path: pathlib.Path) -> None:
    """The figure and the table."""
    out = folder / "plots"
    out.mkdir(exist_ok=True)
    records = _records(folder)
    calibration = _calibration(calibration_path)
    plot_drift(records, calibration, out)
    write_table(records, out)


def plot_drift(records: dict, calibration: dict, out: pathlib.Path) -> None:
    """Share on top, the controller's threshold below, a column per d."""
    distances = sorted({distance for distance, _ in records})
    columns = len(distances)
    size = (5 * columns, 6)
    figure, axes_grid = plt.subplots(2, columns, figsize=size, sharex=True)
    for column, distance in enumerate(distances):
        share_axes = axes_grid[0][column]
        threshold_axes = axes_grid[1][column]
        for control in CONTROLS:
            runs = records[(distance, control)]
            _draw_shares(share_axes, control, runs)
        _draw_share_guides(share_axes, calibration, distance)
        for control in MOVING:
            runs = records[(distance, control)]
            _draw_thresholds(threshold_axes, control, runs)
        _draw_threshold_guides(threshold_axes, calibration, distance)
        share_axes.set_title(f"d = {distance}")
        share_axes.set_ylabel("windows escalated")
        threshold_axes.set_ylabel("threshold (dB)")
        threshold_axes.set_xlabel("round")
    axes_grid[0][0].legend(fontsize=7, ncol=2)
    axes_grid[1][0].legend(fontsize=7)
    figure.tight_layout()
    path = out / "drift.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def _draw_shares(axes, control: str, runs: list) -> None:
    """Each seed's share per hundred windows, thin, and their mean."""
    color = CONTROLS[control]
    curves = [_shares(run) for run in runs]
    for rounds, shares in curves:
        axes.plot(rounds, shares, color=color, lw=0.4, alpha=0.4)
    rounds, _ = curves[0]
    columns = zip(*(shares for _, shares in curves), strict=False)
    means = [statistics.fmean(column) for column in columns]
    count = len(means)
    axes.plot(rounds[:count], means, color=color, lw=1.6, label=control)


def _draw_share_guides(axes, calibration: dict, distance: int) -> None:
    """The target dotted, q* at each p dashed, the drift a grey line."""
    axes.axhline(drift.TARGET, color="black", ls=":", lw=0.8, label="target")
    probabilities = _probabilities()
    guides = zip(probabilities, GUIDE_STYLES, strict=True)
    for probability, style in guides:
        q_star = calibration[(distance, probability)]["q_star"]
        label = f"q* at p = {probability:g}"
        axes.axhline(q_star, color="grey", ls=style, lw=0.8, label=label)
    axes.axvline(drift.DRIFT_ROUND, color="grey", lw=0.8)
    axes.grid(True, alpha=0.3)


def _draw_thresholds(axes, control: str, runs: list) -> None:
    """Each seed's moving threshold against the round."""
    for run in runs:
        rounds, decibels = _thresholds(run)
        axes.plot(rounds, decibels, color=CONTROLS[control], lw=0.6)


def _draw_threshold_guides(axes, calibration: dict, distance: int) -> None:
    """The threshold calibrated for the target at each p, dotted."""
    column = f"target_{drift.TARGET:g}"
    probabilities = _probabilities()
    guides = zip(probabilities, GUIDE_STYLES, strict=True)
    for probability, style in guides:
        decibels = calibration[(distance, probability)][column]
        label = f"calibrated at p = {probability:g}"
        axes.axhline(decibels, color="grey", ls=style, lw=0.8, label=label)
    axes.axvline(drift.DRIFT_ROUND, color="grey", lw=0.8)
    axes.grid(True, alpha=0.3)


def write_table(records: dict, out: pathlib.Path) -> None:
    """One row per task, its seeds' means."""
    keys = sorted(records)
    rows = [_row(key, records[key]) for key in keys]
    path = out / "drift.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"wrote {path}")


def _row(key: tuple, runs: list) -> tuple:
    """A task's seeds' mean shares, reaction and growth; failures summed."""
    per_run = [_run_numbers(run) for run in runs]
    columns = zip(*per_run, strict=True)
    means = [statistics.fmean(column) for column in columns]
    failures = sum(run["measurement"]["logical_failure"] for run in runs)
    return (*key, len(runs), *means, failures)


def _run_numbers(run: dict) -> tuple:
    """A run's shares before and after, and its drifted half's reaction."""
    before, after = _split_shares(run)
    measurement = run["measurement"]
    reaction = measurement["values"]["buffer0_ready_to_frame"]["mean"]
    growth = measurement["weak_reaction_growth_rate"]
    return (before, after, reaction, growth)


def _shares(run: dict) -> tuple:
    """(rounds, share escalated) of each hundred windows."""
    series = run["measurement"]["escalation_series"]
    rounds = []
    shares = []
    previous_windows, previous_escalated = 0, 0
    for windows, escalated, first_round in series:
        share = (escalated - previous_escalated) / (windows - previous_windows)
        rounds.append(first_round)
        shares.append(share)
        previous_windows, previous_escalated = windows, escalated
    return rounds, shares


def _split_shares(run: dict) -> tuple:
    """The share escalated before the drift round, and after it."""
    series = run["measurement"]["escalation_series"]
    before = [sample for sample in series if sample[2] <= drift.DRIFT_ROUND]
    windows, escalated, _ = before[-1]
    total_windows, total_escalated, _ = series[-1]
    after_windows = total_windows - windows
    after_escalated = total_escalated - escalated
    return escalated / windows, after_escalated / after_windows


def _thresholds(run: dict) -> tuple:
    """(rounds, dB) of the controller, at the windows the series sampled."""
    series = run["measurement"]["escalation_series"]
    round_at = {windows: first_round for windows, _, first_round in series}
    round_at[0] = 0
    rounds = []
    decibels = []
    for windows, nats, _event in run["online_threshold"]["trajectory"]:
        if windows not in round_at:
            continue
        rounds.append(round_at[windows])
        threshold = threshold_sources.nats_to_decibels(nats)
        decibels.append(threshold)
    return rounds, decibels


def _probabilities() -> tuple:
    before = drift.PROBABILITY
    drifted = before * drift.DRIFT_FACTOR
    return (before, round(drifted, 6))


def _records(folder: pathlib.Path) -> dict:
    """Every stream record, by (d, control), in seed order."""
    records = collections.defaultdict(list)
    paths = folder.glob("drift_*.json")
    for path in sorted(paths):
        text = path.read_text()
        record = json.loads(text)
        _, distance_word, control = record["task"].split("_")
        distance = int(distance_word[1:])
        records[(distance, control)].append(record)
    return records


def _calibration(path: pathlib.Path) -> dict:
    """calibration.csv's numbers by (d, p)."""
    table = {}
    with path.open() as rows_file:
        for row in csv.DictReader(rows_file):
            point = (int(row["d"]), float(row["p"]))
            table[point] = {
                key: float(value)
                for key, value in row.items()
                if key not in ("d", "p") and value != ""
            }
    return table


if __name__ == "__main__":
    drift_folder = pathlib.Path(sys.argv[1])
    calibration_file = pathlib.Path(sys.argv[2])
    main(drift_folder, calibration_file)
