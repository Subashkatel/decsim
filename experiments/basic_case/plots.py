"""The basic case's figures, from a run folder of run.py or pilot.py.

    python plots.py RESULTS_FOLDER

Writes into RESULTS_FOLDER/plots/:
  - logical_error_rate.png: per shot against p, a line per d, weak alone
    dotted, strong alone dashed, switching solid (Toshio et al.
    2510.25222 Fig. 15);
  - switching_rate.png: the shots with an escalated window, and the
    windows escalated, against p (Fig. 16);
  - latency.png: each window's time from its last round to its frame
    commit, median and 99th percentile, against d;
  - backlog.png: the time from a shot's last round to its last commit,
    mean and largest, against d: a decoder that keeps up holds it flat;
  - windows.png and windows.csv: every window of the switching runs,
    kept by union find, or escalated and answered by BP or by the
    matching after BP did not converge, and how many of each were wrong;
  - gap.png: the cluster gap's histogram and each tier's window failure
    rate given the gap (Figs. 4 to 6).

window_outcomes.csv, which windows and gap read, is written when the run
turns on constants.observation.record_window_outcomes.
"""

import collections
import csv
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROBABILITY = "workload.arguments.physical_error_probability"
DISTANCE = "qpu.distance"
DECODINGS = ("weak_alone", "strong_alone", "switching")
LINE_STYLES = {"weak_alone": ":", "strong_alone": "--", "switching": "-"}
# the latency point of one window: its last round read out to its commit
WINDOW_LATENCY = "qpu_last_round_to_frame"
# a gap in natural-log units as the paper's decibels, 10 log10(e^g)
DECIBELS_PER_NAT = 10 / math.log(10)
GAP_BIN_DECIBELS = 2.0
KINDS = ("kept by union find", "BP converged", "matching after BP failed")
# the smallest and largest distance the colours span
COLOR_SPAN = (5, 13)


def main(results: pathlib.Path) -> None:
    """Every figure the folder's files hold."""
    folder = results / "plots"
    folder.mkdir(exist_ok=True)
    shots_path = results / "shots.csv"
    shots = _read(shots_path)
    plot_logical_error_rate(shots, folder)
    plot_switching_rate(shots, folder)
    plot_backlog(shots, folder)
    samples_path = results / "window_samples.csv"
    samples = _read(samples_path)
    plot_latency(samples, folder)
    outcomes_path = results / "window_outcomes.csv"
    if not outcomes_path.exists():
        print("no window_outcomes.csv: windows and gap are not drawn")
        return
    outcomes = _read(outcomes_path)
    switching = [row for row in outcomes if row["decoding"] == "switching"]
    write_windows(switching, folder)
    plot_gap(switching, folder)


def _read(path: pathlib.Path) -> list:
    with path.open() as rows_file:
        reader = csv.DictReader(rows_file)
        return list(reader)


def _point(row: dict) -> tuple:
    distance = int(row[DISTANCE])
    probability = float(row[PROBABILITY])
    return row["decoding"], distance, probability


def _grouped(rows: list) -> dict:
    """The rows of each (decoding, d, p)."""
    groups = collections.defaultdict(list)
    for row in rows:
        point = _point(row)
        groups[point].append(row)
    return groups


def _distances(groups: dict) -> list:
    return sorted({point[1] for point in groups})


def _probabilities(groups: dict) -> list:
    return sorted({point[2] for point in groups})


def plot_logical_error_rate(shots: list, folder: pathlib.Path) -> None:
    """Fig. 15: failures per shot, with a binomial standard error."""
    groups = _grouped(shots)
    figure, axes = plt.subplots(figsize=(6, 4.5))
    for distance in _distances(groups):
        color = _color_of(distance)
        for decoding in DECODINGS:
            points = _rate_points(groups, decoding, distance, _failed)
            _draw_rates(axes, points, color, decoding, f"d={distance}")
    _finish(axes, "physical error rate p", "logical error rate per shot")
    _save(figure, folder, "logical_error_rate.png")


def plot_switching_rate(shots: list, folder: pathlib.Path) -> None:
    """Fig. 16: the shots and the windows the switching runs escalated."""
    groups = _grouped(shots)
    figure, (per_shot, per_window) = plt.subplots(1, 2, figsize=(11, 4.5))
    for distance in _distances(groups):
        color = _color_of(distance)
        label = f"d={distance}"
        shot_points = _rate_points(groups, "switching", distance, _escalated)
        _draw_rates(per_shot, shot_points, color, "switching", label)
        window_points = _window_rates(groups, distance)
        _draw_rates(per_window, window_points, color, "switching", label)
    _finish(per_shot, "physical error rate p", "shots with a window escalated")
    _finish(per_window, "physical error rate p", "windows escalated")
    _save(figure, folder, "switching_rate.png")


def _failed(row: dict) -> bool:
    return row["logical_failure"] == "True"


def _escalated(row: dict) -> bool:
    return int(row["escalated_windows"]) > 0


def _rate_points(groups: dict, decoding: str, distance: int, counts) -> list:
    """(p, rate, standard error) of the shots counts says yes to."""
    points = []
    for probability in _probabilities(groups):
        rows = groups.get((decoding, distance, probability), [])
        hits = sum(1 for row in rows if counts(row))
        trials = len(rows)
        point = _binomial(probability, hits, trials)
        points.append(point)
    return [point for point in points if point is not None]


def _window_rates(groups: dict, distance: int) -> list:
    """(p, rate, standard error) of the switching runs' escalated windows."""
    points = []
    for probability in _probabilities(groups):
        rows = groups.get(("switching", distance, probability), [])
        escalated = sum(int(row["escalated_windows"]) for row in rows)
        windows = sum(int(row["decoded_windows"]) for row in rows)
        point = _binomial(probability, escalated, windows)
        points.append(point)
    return [point for point in points if point is not None]


def _binomial(x: float, hits: int, trials: int):
    """The rate and its standard error; None with no trial or no hit."""
    if trials == 0 or hits == 0:
        return None
    rate = hits / trials
    variance = rate * (1 - rate) / trials
    error = math.sqrt(variance)
    return x, rate, error


def _draw_rates(axes, points: list, color, decoding: str, label: str) -> None:
    if not points:
        return
    xs, rates, errors = zip(*points, strict=True)
    style = LINE_STYLES[decoding]
    name = f"{label} {decoding}"
    axes.errorbar(
        xs, rates, yerr=errors, color=color, linestyle=style, label=name
    )
    axes.set_yscale("log")


def plot_latency(samples: list, folder: pathlib.Path) -> None:
    """Each window's last round to its commit: median and 99th percentile."""
    latency = [row for row in samples if row["name"] == WINDOW_LATENCY]
    groups = _grouped(latency)
    figure, panels = _panels(groups)
    for axes, probability in panels:
        for decoding in DECODINGS:
            rows = _quantile_rows(groups, decoding, probability)
            _draw_pair(axes, rows, decoding, ("median", "99th"))
        axes.set_title(f"p = {probability}")
        _finish(axes, "code distance d", "window latency (us)")
    _save(figure, folder, "latency.png")


def _quantile_rows(groups: dict, decoding: str, probability: float) -> list:
    """(d, median, 99th percentile) of the windows' latency."""
    rows = []
    for distance in _distances(groups):
        samples = groups.get((decoding, distance, probability), [])
        if not samples:
            continue
        median = _quantile(samples, 0.5)
        tail = _quantile(samples, 0.99)
        rows.append((distance, median, tail))
    return rows


def _quantile(samples: list, fraction: float) -> float:
    """The value below which this fraction of the counted windows fall."""
    counted = [(float(row["value_us"]), int(row["count"])) for row in samples]
    counted.sort()
    total = sum(count for _, count in counted)
    wanted = fraction * total
    seen = 0
    for value, count in counted:
        seen += count
        if seen >= wanted:
            return value
    return counted[-1][0]


def plot_backlog(shots: list, folder: pathlib.Path) -> None:
    """The last round to the last commit, mean and largest over shots."""
    groups = _grouped(shots)
    figure, panels = _panels(groups)
    for axes, probability in panels:
        for decoding in DECODINGS:
            rows = _reaction_rows(groups, decoding, probability)
            _draw_pair(axes, rows, decoding, ("mean", "largest"))
        axes.set_title(f"p = {probability}")
        _finish(axes, "code distance d", "last round to last commit (us)")
    _save(figure, folder, "backlog.png")


def _reaction_rows(groups: dict, decoding: str, probability: float) -> list:
    """(d, mean, largest) of the shots' last round to last commit."""
    rows = []
    for distance in _distances(groups):
        shots = groups.get((decoding, distance, probability), [])
        values = [float(row["end_of_stream_reaction_mean_us"]) for row in shots]
        if not values:
            continue
        mean = sum(values) / len(values)
        largest = max(values)
        rows.append((distance, mean, largest))
    return rows


def _panels(groups: dict) -> tuple:
    """A figure with one panel per p, and (panel, p) pairs."""
    probabilities = _probabilities(groups)
    count = len(probabilities)
    width = 5 * count
    figure, axes_row = plt.subplots(1, count, figsize=(width, 4.5))
    axes_list = _as_list(axes_row)
    panels = zip(axes_list, probabilities, strict=True)
    return figure, panels


def _draw_pair(axes, rows: list, decoding: str, names: tuple) -> None:
    """Two series against d: circles for the first, triangles the second."""
    if not rows:
        return
    distances, first, second = zip(*rows, strict=True)
    style = LINE_STYLES[decoding]
    first_label = f"{decoding} {names[0]}"
    second_label = f"{decoding} {names[1]}"
    first_format = "o" + style
    second_format = "^" + style
    axes.plot(distances, first, first_format, label=first_label)
    axes.plot(distances, second, second_format, label=second_label)
    axes.set_yscale("log")


def write_windows(outcomes: list, folder: pathlib.Path) -> None:
    """Every switching window by who answered it, and how many were wrong."""
    counts = _window_counts(outcomes)
    points = sorted({key[0] for key in counts})
    table = folder / "windows.csv"
    with table.open("w", newline="") as table_file:
        writer = csv.writer(table_file)
        writer.writerow(["d", "p", "kind", "windows", "wrong"])
        for point in points:
            _write_point(writer, counts, point)
    _plot_windows(counts, points, folder)


def _window_counts(outcomes: list) -> collections.Counter:
    """The windows of each (point, kind), and (point, kind, "wrong")."""
    counts = collections.Counter()
    for row in outcomes:
        point = _point(row)
        kind = _kind_of(row)
        counts[(point, kind)] += 1
        counts[(point, kind, "wrong")] += row["is_right"] == "False"
    return counts


def _write_point(writer, counts: collections.Counter, point: tuple) -> None:
    for kind in KINDS:
        windows = counts[(point, kind)]
        wrong = counts[(point, kind, "wrong")]
        writer.writerow([point[1], point[2], kind, windows, wrong])


def _kind_of(row: dict) -> str:
    """Who answered the window: union find, BP, or the matching."""
    if row["is_escalated"] != "True":
        return KINDS[0]
    evidence = json.loads(row["strong_evidence"])
    if evidence["matched"] == 1.0:
        return KINDS[2]
    return KINDS[1]


def _plot_windows(counts: collections.Counter, points: list, folder) -> None:
    """Stacked shares of the windows; the wrong ones of each, hatched."""
    width = max(6, len(points))
    figure, axes = plt.subplots(figsize=(width, 4.5))
    labels = [f"d={point[1]}\np={point[2]}" for point in points]
    positions = list(range(len(points)))
    bottoms = [0.0] * len(points)
    for kind in KINDS:
        shares, wrong_shares = _shares(counts, points, kind)
        bars = axes.bar(positions, shares, bottom=bottoms, label=kind)
        first_bar = bars.patches[0]
        color = first_bar.get_facecolor()
        axes.bar(
            positions,
            wrong_shares,
            bottom=bottoms,
            color=color,
            hatch="xx",
            edgecolor="black",
        )
        stacked = zip(bottoms, shares, strict=True)
        bottoms = [low + share for low, share in stacked]
    axes.set_xticks(positions, labels)
    axes.set_ylabel("share of all windows (hatched: wrong)")
    axes.legend()
    _save(figure, folder, "windows.png")


def _shares(counts: collections.Counter, points: list, kind: str) -> tuple:
    """Each point's share of windows of this kind, and of wrong ones."""
    shares = []
    wrong_shares = []
    for point in points:
        total = sum(counts[(point, each)] for each in KINDS)
        share = counts[(point, kind)] / total
        wrong_share = counts[(point, kind, "wrong")] / total
        shares.append(share)
        wrong_shares.append(wrong_share)
    return shares, wrong_shares


def plot_gap(outcomes: list, folder: pathlib.Path) -> None:
    """Figs. 4 to 6 per p: the gap's spread, and each tier's failures."""
    groups = _grouped(outcomes)
    probabilities = _probabilities(groups)
    count = len(probabilities)
    width = 5 * count
    figure, axes = plt.subplots(2, count, figsize=(width, 8), squeeze=False)
    for column, probability in enumerate(probabilities):
        spread = axes[0][column]
        failures = axes[1][column]
        for distance in _distances(groups):
            rows = groups.get(("switching", distance, probability), [])
            _draw_gap(spread, failures, rows, distance)
        spread.set_title(f"p = {probability}")
        spread.set_yscale("log")
        _finish(spread, "cluster gap (dB)", "windows")
        _finish(failures, "cluster gap (dB)", "window failure rate")
    _save(figure, folder, "gap.png")


def _draw_gap(spread, failures, rows: list, distance: int) -> None:
    """One distance's histogram, and its tiers' failures given the gap."""
    if not rows:
        return
    color = _color_of(distance)
    label = f"d={distance}"
    decibels = [_decibels(row) for row in rows]
    spread.hist(decibels, bins=50, histtype="step", color=color, label=label)
    weak = _failure_by_gap(rows, _weak_wrong)
    _draw_failure(failures, weak, ":", color, f"{label} union find")
    escalated = [row for row in rows if row["is_escalated"] == "True"]
    strong = _failure_by_gap(escalated, _strong_wrong)
    _draw_failure(failures, strong, "-", color, f"{label} belief matching")


def _decibels(row: dict) -> float:
    nats = float(row["gap_nats"])
    return nats * DECIBELS_PER_NAT


def _weak_wrong(row: dict) -> bool:
    """Union find's answer against the label of the window's commit rounds."""
    if row["is_escalated"] == "True":
        return row["weak_answer"] != row["weak_label"]
    return row["is_right"] == "False"


def _strong_wrong(row: dict) -> bool:
    return row["is_right"] == "False"


def _failure_by_gap(rows: list, wrong) -> list:
    """(bin centre in dB, failure rate) per 2 dB bin that holds a failure."""
    failed = collections.Counter()
    totals = collections.Counter()
    for row in rows:
        decibels = _decibels(row)
        index = decibels // GAP_BIN_DECIBELS
        failed[index] += wrong(row)
        totals[index] += 1
    points = []
    for index in sorted(totals):
        if not failed[index]:
            continue
        centre = (index + 0.5) * GAP_BIN_DECIBELS
        rate = failed[index] / totals[index]
        points.append((centre, rate))
    return points


def _draw_failure(axes, points: list, style: str, color, label: str) -> None:
    if not points:
        return
    centres, rates = zip(*points, strict=True)
    marker_format = "o" + style
    axes.plot(centres, rates, marker_format, color=color, label=label)
    axes.set_yscale("log")


def _color_of(distance: int):
    palette = plt.get_cmap("viridis")
    low, high = COLOR_SPAN
    position = (distance - low) / (high - low)
    return palette(position)


def _as_list(axes) -> list:
    if isinstance(axes, plt.Axes):
        return [axes]
    return list(axes)


def _finish(axes, x_label: str, y_label: str) -> None:
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=6)


def _save(figure, folder: pathlib.Path, name: str) -> None:
    path = folder / name
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    folder_argument = pathlib.Path(sys.argv[1])
    main(folder_argument)
