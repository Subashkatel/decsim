"""The ler figure against distance: both tiers at one p from sweep.csv.

The figure's contract: measured points carry Wilson bars, a
zero-failure point is left off (a log axis cannot hold zero, so the
curve ends at the last distance that saw failures), and a run with no
point where the where values hold is refused rather than silently
dropped. The x setting and the kept p are named by their metadata
paths, as sinter's plot reads a point's json_metadata
(sinter/_command/_main_plot.py:21-35).

The two folders here are `decsim collect` folders written by hand: the
columns are the first ten of sweep.csv, in the order report.summarize
writes them, and one resolved/ record with the escalation kind, which
is every value the figure reads. Writing them rather than running the
sweep is what lets one tier hold a zero-failure point at d 5.
"""

import csv
import json

import matplotlib.figure
import pytest

import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder

# The first nine columns of sweep.csv (decsim/experiments/report.py
# summarize_point), which is all the figure reads.
SWEEP_FIELDS = [
    "point_id",
    "metadata",
    "algorithm",
    "shots",
    "windows_per_shot",
    "logical_failures",
    "logical_error_rate",
    "ler_wilson_low",
    "ler_wilson_high",
]


def write_sweep_csv(run_dir, algorithm, points):
    """points: (distance, p, shots, failures, rate, low, high) rows."""
    run_dir.mkdir()
    sweep_path = run_dir / "sweep.csv"
    with open(sweep_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for distance, p, shots, failures, rate, low, high in points:
            metadata = {"distance": distance, "physical_error_probability": p}
            metadata_text = json.dumps(metadata, sort_keys=True)
            writer.writerow(
                {
                    "point_id": f"{algorithm}-{distance}-{p}",
                    "metadata": metadata_text,
                    "algorithm": algorithm,
                    "shots": shots,
                    "windows_per_shot": distance,
                    "logical_failures": failures,
                    "logical_error_rate": rate,
                    "ler_wilson_low": low,
                    "ler_wilson_high": high,
                }
            )
    return run_dir


def write_resolved(run_dir, escalation_kind):
    """One point's resolved/ record: its sweep point and escalation kind."""
    resolved_dir = run_dir / run_folder.RESOLVED_FOLDER
    resolved_dir.mkdir()
    metadata = {
        "physical_error_probability": 0.001,
        "distance": 3,
        "round_period_microseconds": 1.0,
    }
    settings = {"escalation": {"kind": escalation_kind}}
    record = {"id": "point", "metadata": metadata, "settings": settings}
    record_text = json.dumps(record)
    record_path = resolved_dir / "point.json"
    record_path.write_text(record_text)


def legend_labels(run_dirs, tmp_path, monkeypatch) -> list:
    """The labels of the figure's legend, read as the figure is saved."""
    figure_path = tmp_path / "ler.png"
    labels = []

    def record_the_labels(figure, path, **options):
        del path, options
        axis = figure.axes[0]
        _, axis_labels = axis.get_legend_handles_labels()
        labels.extend(axis_labels)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record_the_labels)
    selection = at_probability(0.001)
    plots.ler_plot(run_dirs, selection, figure_path)
    return labels


def two_tier_runs(tmp_path):
    weak_run_dir = tmp_path / "weak"
    weak_points = [
        (3, 0.001, 1000, 7, 7e-3, 3e-3, 1.4e-2),
        (5, 0.001, 1000, 1, 1e-3, 2e-4, 6e-3),
    ]
    weak = write_sweep_csv(weak_run_dir, "0.028", weak_points)
    write_resolved(weak, "weak_baseline")
    strong_run_dir = tmp_path / "strong"
    strong_points = [
        (3, 0.001, 1000, 6, 6e-3, 2e-3, 1.2e-2),
        (5, 0.001, 1000, 0, 0.0, 0.0, 3.8e-3),
    ]
    strong = write_sweep_csv(strong_run_dir, "belief_matching", strong_points)
    write_resolved(strong, "strong_only")
    return weak, strong


def at_probability(probability: float) -> plots.Selection:
    """The figure against distance at one p, as `decsim plot` asks for it."""
    where = {"physical_error_probability": probability}
    return plots.Selection("distance", None, where)


def test_figure_written_with_zero_failure_point_left_off(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler.png"
    selection = at_probability(0.001)
    plots.ler_plot([weak, strong], selection, figure_path)
    assert figure_path.exists()


def test_the_y_axis_names_no_round_count_the_sweep_did_not_write(
    tmp_path, monkeypatch
):
    """A shot runs the yaml's rounds_per_shot, not always 10 d rounds."""
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler.png"
    labels = []

    def record_the_label(figure, path, **options):
        del path, options
        axis = figure.axes[0]
        label = axis.get_ylabel()
        labels.append(label)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record_the_label)
    selection = at_probability(0.001)
    plots.ler_plot([weak, strong], selection, figure_path)

    assert labels == ["Logical error rate per shot"]


def test_a_strong_pymatching_run_is_labelled_strong(tmp_path, monkeypatch):
    """The tier is the escalation row's, not guessed from the card name."""
    strong_run_dir = tmp_path / "strong_pymatching"
    strong_points = [(3, 0.001, 1000, 6, 6e-3, 2e-3, 1.2e-2)]
    strong = write_sweep_csv(strong_run_dir, "pymatching", strong_points)
    write_resolved(strong, "strong_only")

    labels = legend_labels([strong], tmp_path, monkeypatch)

    assert labels == ["pymatching (strong)"]


def test_a_curve_per_run_and_group_value_leaves_zero_failures_off(
    tmp_path, monkeypatch
):
    """Two runs of two distances grouped by distance: four curves.

    The strong run's d 5 point saw no failure, so its curve is empty.
    """
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler.png"
    drawn = []

    def record_the_curves(figure, path, **options):
        del path, options
        axis = figure.axes[0]
        for container in axis.containers:
            line = container.lines[0]
            label = container.get_label()
            x_values = line.get_xdata()
            drawn.append((label, list(x_values)))

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record_the_curves)
    where = {"physical_error_probability": 0.001}
    selection = plots.Selection("physical_error_probability", "distance", where)
    plots.ler_plot([weak, strong], selection, figure_path)

    assert drawn == [
        ("pymatching (weak), distance=3", [0.001]),
        ("pymatching (weak), distance=5", [0.001]),
        ("belief matching (strong), distance=3", [0.001]),
        ("belief matching (strong), distance=5", []),
    ]


def test_run_without_the_requested_p_is_refused(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler.png"
    selection = at_probability(0.002)
    with pytest.raises(ValueError, match="physical_error_probability=0.002"):
        plots.ler_plot([weak, strong], selection, figure_path)


def test_a_run_whose_points_set_no_x_is_refused(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler.png"
    selection = plots.Selection("qpu.distance", None, {})
    with pytest.raises(refusal.RefusalError, match="set no qpu.distance"):
        plots.ler_plot([weak, strong], selection, figure_path)


def test_a_run_without_sweep_csv_is_refused(tmp_path):
    empty_run = tmp_path / "no_sweep"
    empty_run.mkdir()
    figure_path = tmp_path / "ler.png"
    selection = at_probability(0.001)
    with pytest.raises(refusal.RefusalError, match="sweep.csv"):
        plots.ler_plot([empty_run], selection, figure_path)
