"""latency_samples.csv: measured algorithm wall clock per window.

One sample per decoded window, the algorithm stage only (the measured
quantity; the priced stages are in shots.csv), recorded only for a
decoder named by a table row, each sample beside its window's
inter-arrival, the deadline a latency figure draws. decsim draws no
latency figure; these rows are everything one needs.
"""

import collections
import csv

import decsim.experiments.collect_command as collect_command
import decsim.experiments.experiment as experiment
import tests.experiments.yaml_configs as yaml_configs


def wall_clock_config(tmp_path, distances):
    return yaml_configs.write_config(
        tmp_path,
        {
            "weak_decoder": {
                **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
                "kind": "pymatching",
            },
            "sweep": [
                {
                    "axes": {
                        "workload.arguments.physical_error_probability": [
                            0.001
                        ],
                        "qpu.distance": distances,
                        "qpu.round_period_microseconds": [1.0],
                    },
                    "shots": 2,
                }
            ],
        },
    )


def test_every_decoded_window_contributes_one_latency_sample(tmp_path):
    config_path = wall_clock_config(tmp_path, [3])
    config = experiment.load_experiment(config_path)
    measurement = yaml_configs.measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )
    samples = measurement.samples["algorithm"]
    assert len(samples) == measurement.windows
    assert all(sample > 0 for sample in samples)


def test_a_wall_clock_run_records_every_windows_sample_and_deadline(
    tmp_path, monkeypatch
):
    """A sample per decoded window, beside its point and its deadline.

    Each point holds as many samples as its shots decoded windows. The
    deadline is the window's inter-arrival, commit rounds (d when null)
    times the round period: 3 us at d 3 and 5 us at d 5.
    """
    monkeypatch.chdir(tmp_path)
    config_path = wall_clock_config(tmp_path, [3, 5])
    run_dir, rows = collect_command.run_experiment(config_path)
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
    card_path = yaml_configs.write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "axes": {
                        "workload.arguments.physical_error_probability": [
                            0.001
                        ],
                        "qpu.distance": [3, 5],
                        "qpu.round_period_microseconds": [1.0],
                    },
                    "shots": 1,
                }
            ]
        },
    )
    card_run_dir, rows = collect_command.run_experiment(card_path)
    assert not (card_run_dir / "latency_samples.csv").exists()


def _windows_by_point(shots: list) -> collections.Counter:
    """The windows every point's shots decoded, off shots.csv."""
    windows = collections.Counter()
    for row in shots:
        windows[row["point_id"]] += int(row["windows"])
    return windows
