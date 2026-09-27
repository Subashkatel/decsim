"""The latency figure: measured algorithm wall clock per window.

The figure's contract: one sample per decoded window, the algorithm
stage only (the measured quantity; priced stages belong to the
stage-breakdown figure), recorded only for wall-clock algorithms, each
sample with its window's inter-arrival, the deadline the figure draws.
"""

import csv
import json

import pytest

import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal
from decsim.experiments.experiment import load_experiment
from tests.experiments.yaml_configs import (
    MINIMAL_CONFIG,
    measure_point_shot,
    strong_unit,
    write_config,
)


def wall_clock_config(tmp_path, distances):
    return write_config(
        tmp_path,
        {
            "weak_decoder": {
                **MINIMAL_CONFIG["weak_decoder"],
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
    config = load_experiment(config_path)
    measurement = measure_point_shot(
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

    The deadline is the window's inter-arrival, commit rounds (d when
    null) times the round period: 3 us at d 3 and 5 us at d 5.
    """
    from decsim.experiments.collect_command import run_experiment

    monkeypatch.chdir(tmp_path)
    config_path = wall_clock_config(tmp_path, [3, 5])
    run_dir, rows = run_experiment(config_path)
    samples_path = run_dir / "latency_samples.csv"
    with open(samples_path, newline="") as handle:
        reader = csv.DictReader(handle)
        samples = list(reader)
    shots_path = run_dir / "shots.csv"
    with open(shots_path, newline="") as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)

    deadlines = {}
    for sample in samples:
        metadata = json.loads(sample["metadata"])
        deadlines[metadata["qpu.distance"]] = float(sample["window_period_us"])
    windows = 0
    for shot in shots:
        windows += int(shot["windows"])
    assert len(samples) == windows
    assert deadlines == {3: 3.0, 5: 5.0}


def test_a_latency_card_run_records_no_samples(tmp_path, monkeypatch):
    """The fixed-latency card is flat by construction: no raw samples."""
    from decsim.experiments.collect_command import run_experiment

    monkeypatch.chdir(tmp_path)
    card_path = write_config(
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
    card_run_dir, rows = run_experiment(card_path)
    assert not (card_run_dir / "latency_samples.csv").exists()


def test_combined_figure_reads_two_runs_sample_files(tmp_path, monkeypatch):
    """The cross-tier figure: two runs' latency_samples.csv on one axes."""
    from decsim.experiments.collect_command import run_experiment
    from decsim.experiments.plots import combined_latency_plot

    monkeypatch.chdir(tmp_path)

    weak_config_path = wall_clock_config(tmp_path, [3, 5])
    weak_run_dir, rows = run_experiment(weak_config_path)
    strong_decoder_section = strong_unit("belief_matching")
    strong_path = write_config(
        tmp_path,
        {
            "escalation": {"kind": "strong_only"},
            **strong_decoder_section,
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
            ],
        },
    )
    strong_run_dir, rows = run_experiment(strong_path)

    combined = tmp_path / "latency_combined.png"
    weak_samples_path = weak_run_dir / "latency_samples.csv"
    strong_samples_path = strong_run_dir / "latency_samples.csv"
    sample_paths = [weak_samples_path, strong_samples_path]
    selection = plots.Selection("qpu.distance", None, {})
    combined_latency_plot(sample_paths, selection, combined)
    assert combined.exists()


def test_a_run_without_latency_samples_is_refused(tmp_path):
    empty_run = tmp_path / "timing_only"
    empty_run.mkdir()
    figure_path = tmp_path / "figure.png"
    selection = plots.Selection("qpu.distance", None, {})
    with pytest.raises(refusal.RefusalError, match="latency_samples.csv"):
        plots.figure("latency", [empty_run], figure_path, selection)
