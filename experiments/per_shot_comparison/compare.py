"""Which decoder fails on which shots: union-find alone, Relay-BP-5 alone, switching.

`python compare.py <switching baseline run> <strong-only run> <out folder>`
reads every saved piece's shots.csv in the two experiment folders and
joins the runs on (distance, physical error rate, seed). Every run draws
the same Stim shot for the same seed, so one seed is one shot in all of
them; the join checks this on the switching shots with no escalation,
which must fail exactly where union-find alone fails.

It writes comparison.csv, one row per setting with the shots all three
runs share and how many of them each combination of decoders failed,
and plots/: the failure rate of each decoder on the shared shots, and
the shots only switching failed, split by whether one of their Relay-BP
windows did not converge. A setting needs MIN_SHARED_SHOTS shared shots
to be drawn.
"""

import csv
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as pyplot

# configuration id prefix -> the run's name in this comparison
RUN_NAMES = {
    "45a586ca": "union_find",
    "1d1aa311": "switching",
    "37cc8bd5": "relay_bp5",
}
SHOWN_NAME = {
    "union_find": "union-find alone",
    "relay_bp5": "Relay-BP-5 alone",
    "switching": "switching",
}
MIN_SHARED_SHOTS = 500
FIGURE_SIZE_INCHES = (9.6, 3.6)
DOTS_PER_INCH = 150
GRID_ALPHA = 0.3
LEGEND_FONT_SIZE = 8
BAR_WIDTH = 0.27


def main(switching_run: pathlib.Path, strong_run: pathlib.Path, out: pathlib.Path) -> None:
    """Join the runs, write comparison.csv and the two figures."""
    shots = {name: defaultdict(dict) for name in RUN_NAMES.values()}
    union_find_pieces = []
    for run_folder in (switching_run, strong_run):
        for name, point, first_seed, piece in pieces_of(run_folder):
            if name == "union_find":
                union_find_pieces.append((point, first_seed, piece))
                continue
            read_shots(piece, shots[name][point])
    # union-find alone ran millions of shots; read only the seeds the others ran
    for point, first_seed, piece in union_find_pieces:
        other_seeds = list(shots["switching"][point]) + list(shots["relay_bp5"][point])
        if other_seeds and first_seed <= max(other_seeds):
            read_shots(piece, shots["union_find"][point])
    check_same_shots(shots)
    rows = comparison_rows(shots)
    write_rows(rows, out / "comparison.csv")
    plots_folder = out / "plots"
    plots_folder.mkdir(exist_ok=True)
    drawn = [row for row in rows if row["shared_shots"] >= MIN_SHARED_SHOTS]
    failure_rate_figure(drawn, plots_folder / "failure_rate_per_decoder.png")
    only_switching_figure(drawn, plots_folder / "only_switching_failed.png")


def pieces_of(run_folder: pathlib.Path):
    """(run name, point, first seed, folder) of every saved piece."""
    for point_folder in sorted((run_folder / "pieces").iterdir()):
        point = point_of(run_folder, point_folder.name)
        for piece in sorted(point_folder.iterdir()):
            piece_record_path = piece / "piece.json"
            shots_path = piece / "shots.csv"
            if not piece_record_path.exists() or not shots_path.exists():
                continue
            piece_record = json.loads(piece_record_path.read_text())
            name = RUN_NAMES.get(piece_record["configuration_id"][:8])
            if name is None:
                continue
            first_seed = int(piece.name.split("-")[0])
            yield name, point, first_seed, piece


def point_of(run_folder: pathlib.Path, point_id: str) -> tuple:
    """(distance, physical error rate) of a point, from its resolved record."""
    record_path = run_folder / "resolved" / f"{point_id}.json"
    metadata = json.loads(record_path.read_text())["metadata"]
    distance = metadata["qpu.distance"]
    error_rate = metadata["workload.arguments.physical_error_probability"]
    return distance, error_rate


def read_shots(piece: pathlib.Path, shots_by_seed: dict) -> None:
    """Each shot of the piece: failed, escalated, nonconverged, weak weight."""
    with open(piece / "shots.csv") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            shots_by_seed[seed] = (
                row["logical_failure"] == "True",
                int(row["escalated_windows"] or 0),
                int(row["nonconverged_windows"] or 0),
                row["weak_syndrome_weight_mean"],
            )


def check_same_shots(shots: dict) -> None:
    """A switching shot that never escalated is union-find alone's shot."""
    matched = 0
    for point, switching_shots in shots["switching"].items():
        union_find_shots = shots["union_find"][point]
        for seed, switching_shot in switching_shots.items():
            union_find_shot = union_find_shots.get(seed)
            escalated = switching_shot[1]
            if union_find_shot is None or escalated:
                continue
            same_failure = switching_shot[0] == union_find_shot[0]
            same_syndromes = switching_shot[3] == union_find_shot[3]
            if not (same_failure and same_syndromes):
                raise SystemExit(f"seed {seed} at {point} is not the same shot")
            matched += 1
    print(f"{matched} switching shots with no escalation match union-find alone")


def comparison_rows(shots: dict) -> list:
    """One row per setting: the shared shots and each failure combination."""
    rows = []
    for point in sorted(shots["relay_bp5"]):
        union_find = shots["union_find"][point]
        relay = shots["relay_bp5"][point]
        switching = shots["switching"][point]
        shared_seeds = set(union_find) & set(relay) & set(switching)
        if not shared_seeds:
            continue
        counts = defaultdict(int)
        for seed in shared_seeds:
            union_find_failed = union_find[seed][0]
            relay_failed = relay[seed][0]
            switching_failed = switching[seed][0]
            counts[(union_find_failed, relay_failed, switching_failed)] += 1
            only_switching = switching_failed and not (union_find_failed or relay_failed)
            switching_nonconverged = switching[seed][2] > 0
            if only_switching and switching_nonconverged:
                counts["only_switching_nonconverged"] += 1
        rows.append(row_of(point, len(shared_seeds), counts))
    return rows


def row_of(point: tuple, shared_shots: int, counts: dict) -> dict:
    """A setting's counts, named; the key is (union-find, Relay-BP, switching) failed."""
    def failed(union_find, relay, switching):
        return counts[(union_find, relay, switching)]

    union_find_total = sum(failed(True, r, s) for r in (0, 1) for s in (0, 1))
    relay_total = sum(failed(u, True, s) for u in (0, 1) for s in (0, 1))
    switching_total = sum(failed(u, r, True) for u in (0, 1) for r in (0, 1))
    distance, error_rate = point
    return {
        "distance": distance,
        "physical_error_rate": error_rate,
        "shared_shots": shared_shots,
        "union_find_failed": union_find_total,
        "relay_bp5_failed": relay_total,
        "switching_failed": switching_total,
        "all_three_failed": failed(True, True, True),
        "union_find_and_relay_bp5_failed": failed(True, True, False),
        "union_find_and_switching_failed": failed(True, False, True),
        "relay_bp5_and_switching_failed": failed(False, True, True),
        "only_union_find_failed": failed(True, False, False),
        "only_relay_bp5_failed": failed(False, True, False),
        "only_switching_failed": failed(False, False, True),
        "only_switching_failed_nonconverged": counts["only_switching_nonconverged"],
        "none_failed": failed(False, False, False),
    }


def write_rows(rows: list, path: pathlib.Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def setting_label(row: dict) -> str:
    return f"d={row['distance']}\n{row['physical_error_rate']}"


def new_figure(title: str, rows: list) -> tuple:
    """One panel, a group of bars per setting."""
    figure, axis = pyplot.subplots(figsize=FIGURE_SIZE_INCHES, layout="constrained")
    axis.set_title(title)
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([setting_label(row) for row in rows], fontsize=8)
    axis.set_xlabel("distance and physical error rate")
    axis.grid(axis="y", alpha=GRID_ALPHA)
    return figure, axis


def save(figure, axis, path: pathlib.Path) -> None:
    axis.legend(fontsize=LEGEND_FONT_SIZE)
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def failure_rate_figure(rows: list, path: pathlib.Path) -> None:
    """Share of the shared shots each decoder failed, a bar each."""
    figure, axis = new_figure("Shots each decoder fails, on the same shots", rows)
    positions = range(len(rows))
    for index, name in enumerate(["union_find", "relay_bp5", "switching"]):
        offset = (index - 1) * BAR_WIDTH
        shares = [100 * row[f"{name}_failed"] / row["shared_shots"] for row in rows]
        lefts = [position + offset for position in positions]
        axis.bar(lefts, shares, BAR_WIDTH, label=SHOWN_NAME[name])
    axis.set_ylabel("shots failed (%)")
    save(figure, axis, path)


def only_switching_figure(rows: list, path: pathlib.Path) -> None:
    """Shots only switching failed, split by a Relay-BP window not converging."""
    figure, axis = new_figure("Shots only switching fails", rows)
    positions = range(len(rows))
    nonconverged = [
        100 * row["only_switching_failed_nonconverged"] / row["shared_shots"]
        for row in rows
    ]
    converged = [
        100 * (row["only_switching_failed"] - row["only_switching_failed_nonconverged"])
        / row["shared_shots"]
        for row in rows
    ]
    axis.bar(positions, converged, 0.6, label="every Relay-BP window converged")
    axis.bar(positions, nonconverged, 0.6, bottom=converged,
             label="a Relay-BP window did not converge")
    axis.set_ylabel("shots failed (%)")
    save(figure, axis, path)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
