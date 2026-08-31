"""The latency-vs-d figure: measured algorithm wall clock per window.

The figure's contract: one sample per decoded window, the algorithm
stage only (the measured quantity; priced stages belong to the
stage-breakdown figure), drawn only for wall-clock algorithms across
more than one distance.
"""

from experiments.experiment_config import load_experiment
from experiments.measure_shot import measure_shot
from experiments.plots import latency_samples_by_distance, plots

from test_decoder_units import MINIMAL_CONFIG, strong_unit, write_config


def wall_clock_config(tmp_path, distances):
    return write_config(tmp_path, {
        "decoder": {"weak": {**MINIMAL_CONFIG["decoder"]["weak"],
                             "algorithm": "pymatching"}},
        "sweep": [{"physical_error_probability": [0.001],
                   "distance": distances, "round_period_us": [1.0],
                   "shots": 2}]})


def test_every_decoded_window_contributes_one_latency_sample(tmp_path):
    config = load_experiment(wall_clock_config(tmp_path, [3]))
    measurement = measure_shot(config, physical_error_probability=0.001,
                               distance=3, round_period_us=1.0, seed=0)
    samples = measurement.samples["algorithm"]
    assert len(samples) == measurement.windows
    assert all(sample > 0 for sample in samples)


def test_latency_samples_pool_over_shots_per_distance(tmp_path):
    config = load_experiment(wall_clock_config(tmp_path, [3, 5]))
    measurements = [
        measure_shot(config, physical_error_probability=0.001,
                     distance=distance, round_period_us=1.0, seed=seed)
        for distance in (3, 5) for seed in range(2)]
    pooled = latency_samples_by_distance(measurements)
    assert list(pooled) == [3, 5]
    for distance in (3, 5):
        decoded_windows = sum(measurement.windows
                              for measurement in measurements
                              if measurement.distance == distance)
        assert len(pooled[distance]) == decoded_windows


def test_latency_figure_written_only_for_wall_clock_multi_distance(
        tmp_path, monkeypatch):
    from experiments.run import run_experiment
    monkeypatch.chdir(tmp_path)

    config_path = wall_clock_config(tmp_path, [3, 5])
    run_dir, rows = run_experiment(config_path)
    assert (run_dir / "latency.png").exists()
    assert (run_dir / "latency_samples.csv").exists()

    # the fixed-latency card is flat in d by construction: no figure,
    # no raw samples
    card_path = write_config(tmp_path, {
        "sweep": [{"physical_error_probability": [0.001],
                   "distance": [3, 5], "round_period_us": [1.0],
                   "shots": 1}]})
    card_run_dir, rows = run_experiment(card_path)
    assert not (card_run_dir / "latency.png").exists()
    assert not (card_run_dir / "latency_samples.csv").exists()


def test_combined_figure_reads_two_runs_sample_files(tmp_path, monkeypatch):
    """The cross-tier figure: two runs' latency_samples.csv on one axes."""
    from experiments.plots import combined_latency_plot
    from experiments.run import run_experiment
    monkeypatch.chdir(tmp_path)

    weak_run_dir, rows = run_experiment(wall_clock_config(tmp_path, [3, 5]))
    strong_path = write_config(tmp_path, {
        "mode": "strong_only",
        "decoder": strong_unit("belief_matching"),
        "sweep": [{"physical_error_probability": [0.001],
                   "distance": [3, 5], "round_period_us": [1.0],
                   "shots": 1}]})
    strong_run_dir, rows = run_experiment(strong_path)

    combined = tmp_path / "latency_combined.png"
    combined_latency_plot([weak_run_dir / "latency_samples.csv",
                           strong_run_dir / "latency_samples.csv"], combined)
    assert combined.exists()
