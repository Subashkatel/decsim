"""latency_samples.csv: measured algorithm wall clock per window.

One sample per decoded window, the algorithm stage only (the measured
quantity; the priced stages are in shots.csv), recorded only for a
decoder named by a table row, each sample beside its window's
inter-arrival, the deadline a decode must beat. decsim draws no latency
figure; these rows are what one is drawn from, and what it computes
from them (log times, densities, medians) is not stored.
"""

import collections
import csv

import decsim.experiments.collect_command as collect_command
import decsim.experiments.report as report
import tests.experiments.run_files as run_files

# PyMatching's measured wall clock on every window, two shots a point
WALL_CLOCK = {
    "machine_arguments": {"weak_decoder": "pymatching"},
    "collection": {"max_shots": 2},
}


def wall_clock_axes(distances: tuple) -> dict:
    return {
        run_files.DISTANCE_PATH: distances,
        run_files.ROUND_PERIOD_PATH: (1.0,),
        run_files.ERROR_RATE_PATH: (0.001,),
    }


def test_every_decoded_window_contributes_one_latency_sample():
    measurement = run_files.measured_shot(
        axes=wall_clock_axes((3,)), **WALL_CLOCK
    )
    samples = measurement.samples["algorithm"]
    assert len(samples) == measurement.decoded_windows
    assert all(sample > 0 for sample in samples)


def test_a_latency_sample_names_the_tier_that_decoded_its_window():
    """A weak-only run's windows are all the weak tier's."""
    measurement = run_files.measured_shot(
        axes=wall_clock_axes((3,)), **WALL_CLOCK
    )
    rows = report.latency_sample_rows([measurement])
    tiers = [row["tier"] for row in rows]
    assert tiers == ["weak"] * measurement.decoded_windows


def test_a_wall_clock_run_records_every_windows_sample_and_deadline(
    tmp_path, monkeypatch
):
    """A sample per decoded window, beside its point and its deadline.

    Each point holds as many samples as its shots decoded windows. The
    deadline is the window's inter-arrival, commit rounds (d when null)
    times the round period: 3 us at d 3 and 5 us at d 5.
    """
    monkeypatch.chdir(tmp_path)
    run_path = run_files.write_run_file(
        tmp_path, axes=wall_clock_axes((3, 5)), **WALL_CLOCK
    )
    run_dir, rows = collect_command.run_experiment(run_path)
    samples_path = run_dir / "latency_samples.csv"
    with open(samples_path, newline="") as handle:
        reader = csv.DictReader(handle)
        samples = list(reader)
    shots_path = run_dir / "shots.csv"
    with open(shots_path, newline="") as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)

    point_ids = [row["point_id"] for row in samples]
    sample_counts = collections.Counter(point_ids)
    deadlines = {
        (row["point_id"], row["qpu.distance"], row["window_period_us"])
        for row in samples
    }
    point_of = {row["qpu.distance"]: row["point_id"] for row in shots}
    assert sample_counts == _windows_by_point(shots)
    assert deadlines == {
        (point_of["3"], "3", "3.0"),
        (point_of["5"], "5", "5.0"),
    }


def test_a_latency_card_run_records_no_samples(tmp_path, monkeypatch):
    """The fixed-latency card is flat by construction: no raw samples."""
    monkeypatch.chdir(tmp_path)
    card_path = run_files.write_run_file(tmp_path, axes=wall_clock_axes((3, 5)))
    card_run_dir, rows = collect_command.run_experiment(card_path)
    assert not (card_run_dir / "latency_samples.csv").exists()


def _windows_by_point(shots: list) -> collections.Counter:
    """The windows every point's shots decoded, off shots.csv."""
    windows = collections.Counter()
    for row in shots:
        windows[row["point_id"]] += int(row["decoded_windows"])
    return windows
