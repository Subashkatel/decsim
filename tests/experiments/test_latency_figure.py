"""The latency-vs-d figure: measured algorithm wall clock per window.

The figure's contract: one sample per decoded window, the algorithm
stage only (the measured quantity; priced stages belong to the
stage-breakdown figure), drawn only for wall-clock algorithms across
more than one distance.
"""

import pytest

import decsim.experiments.experiment as experiment
import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal
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
                    "physical_error_probability": [0.001],
                    "distance": distances,
                    "round_period_microseconds": [1.0],
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


def test_latency_samples_pool_over_shots_per_distance(tmp_path):
    config_path = wall_clock_config(tmp_path, [3, 5])
    config = experiment.load_experiment(config_path)
    measurements = [
        yaml_configs.measure_point_shot(
            config,
            physical_error_probability=0.001,
            distance=distance,
            round_period_microseconds=1.0,
            seed=seed,
        )
        for distance in (3, 5)
        for seed in range(2)
    ]
    pooled = plots.latency_samples_by_distance(measurements)
    assert list(pooled) == [3, 5]
    for distance in (3, 5):
        decoded_windows = sum(
            measurement.windows
            for measurement in measurements
            if measurement.distance == distance
        )
        assert len(pooled[distance]) == decoded_windows


def test_latency_figure_written_only_for_wall_clock_multi_distance(
    tmp_path, monkeypatch
):
    import decsim.experiments.collect_command as collect_command

    monkeypatch.chdir(tmp_path)

    config_path = wall_clock_config(tmp_path, [3, 5])
    run_dir, rows = collect_command.run_experiment(config_path)
    assert (run_dir / "latency.png").exists()
    assert (run_dir / "latency_samples.csv").exists()

    # the fixed-latency card is flat in d by construction: no figure,
    # no raw samples
    card_path = yaml_configs.write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3, 5],
                    "round_period_microseconds": [1.0],
                    "shots": 1,
                }
            ]
        },
    )
    card_run_dir, rows = collect_command.run_experiment(card_path)
    assert not (card_run_dir / "latency.png").exists()
    assert not (card_run_dir / "latency_samples.csv").exists()


def test_combined_figure_reads_two_runs_sample_files(tmp_path, monkeypatch):
    """The cross-tier figure: two runs' latency_samples.csv on one axes."""
    import decsim.experiments.collect_command as collect_command

    monkeypatch.chdir(tmp_path)

    weak_config_path = wall_clock_config(tmp_path, [3, 5])
    weak_run_dir, rows = collect_command.run_experiment(weak_config_path)
    strong_decoder_section = yaml_configs.strong_unit("belief_matching")
    strong_path = yaml_configs.write_config(
        tmp_path,
        {
            "escalation": {"kind": "strong_only"},
            **strong_decoder_section,
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3, 5],
                    "round_period_microseconds": [1.0],
                    "shots": 1,
                }
            ],
        },
    )
    strong_run_dir, rows = collect_command.run_experiment(strong_path)

    combined = tmp_path / "latency_combined.png"
    weak_samples_path = weak_run_dir / "latency_samples.csv"
    strong_samples_path = strong_run_dir / "latency_samples.csv"
    sample_paths = [weak_samples_path, strong_samples_path]
    plots.combined_latency_plot(sample_paths, combined)
    assert combined.exists()


def test_a_run_without_latency_samples_is_refused(tmp_path):
    empty_run = tmp_path / "timing_only"
    empty_run.mkdir()
    figure_path = tmp_path / "figure.png"
    with pytest.raises(refusal.RefusalError, match="latency_samples.csv"):
        plots.figure("latency", [empty_run], figure_path)
