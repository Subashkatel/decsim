"""The closed-loop runner measures every point on real Stim data, in both
modes, from a yaml config alone."""

from pathlib import Path

import pytest

from experiments.experiment_config import load_experiment
from experiments.measure_shot import POINTS, measure_shot
from experiments.run import run_sweep
from experiments.sweep_report import summarize, write_report

CONFIGS = Path(__file__).parent.parent / "configs"

SMALL_SWEEP = """\
extends: {base}.yaml
rounds_per_shot: 15
sweep:
  - physical_error_probability: [0.001]
    round_period_us: [1.0]
    algorithm_latency_us: [{algorithm}]
    shots: 2
"""


def small_config(tmp_path_factory, base: str, algorithm: float):
    """The repo config with a tiny sweep; extends resolves in configs/."""
    folder = tmp_path_factory.mktemp(base)
    target = folder / f"{base}.yaml"
    target.write_text((CONFIGS / f"{base}.yaml").read_text())
    small = folder / f"small_{base}.yaml"
    small.write_text(SMALL_SWEEP.format(base=base, algorithm=algorithm))
    return load_experiment(small)


@pytest.fixture(scope="module")
def weak_config(tmp_path_factory):
    return small_config(tmp_path_factory, "weak_baseline", 0.028)


@pytest.fixture(scope="module")
def strong_config(tmp_path_factory):
    return small_config(tmp_path_factory, "strong_only", 5.0)


@pytest.fixture(scope="module")
def weak_shot(weak_config):
    return measure_shot(weak_config, physical_error_probability=0.001,
                        round_period_us=1.0, algorithm_latency_us=0.028, seed=0)


@pytest.fixture(scope="module")
def strong_shot(strong_config):
    return measure_shot(strong_config, physical_error_probability=0.001,
                        round_period_us=1.0, algorithm_latency_us=5.0, seed=0)


def test_every_point_is_measured_and_positive_where_it_must_be(weak_shot):
    assert weak_shot.windows >= 3
    for point in POINTS:
        assert point in weak_shot.means and point in weak_shot.maxes
    for charged in ("c2b_per_round", "fetch", "algorithm", "release",
                    "input_link_per_window", "output_link_per_window",
                    "frame_commit"):
        assert weak_shot.means[charged] > 0, charged


def test_decoder_service_is_transfer_plus_the_engine_stages(weak_shot):
    """Unit assigned -> decode done covers the input transfer into the
    unit's memory and then the engine's three stages."""
    assert weak_shot.means["service"] == pytest.approx(
        weak_shot.means["input_link_per_window"] + weak_shot.means["fetch"]
        + weak_shot.means["algorithm"] + weak_shot.means["release"])
    assert weak_shot.means["algorithm"] == pytest.approx(0.028)


def test_configured_costs_appear_at_the_right_points(weak_config, weak_shot):
    engine = weak_config.decoder.engine
    tick_us = 1 / engine.frequency_mhz
    assert weak_shot.means["release"] == pytest.approx(
        engine.release_cycles_per_job * tick_us)
    assert weak_shot.means["frame_commit"] == pytest.approx(
        weak_config.pauli_frame_commit_us)
    links = weak_config.links
    assert weak_shot.means["c2b_per_round"] >= links["c2b"].latency_us
    assert weak_shot.means["input_link_per_window"] == pytest.approx(
        links["cwd"].latency_us)
    assert weak_shot.means["output_link_per_window"] == pytest.approx(
        links["wdo"].latency_us)


def test_reaction_time_orders_the_points(weak_shot):
    assert (weak_shot.means["buffer0_ready_to_frame"]
            <= weak_shot.means["buffer0_first_round_to_frame"])
    assert weak_shot.means["buffer0_ready_to_frame"] >= (
        weak_shot.means["service"] + weak_shot.means["output_link_per_window"]
        + weak_shot.means["frame_commit"])


def test_sweep_summary_and_report(weak_config, tmp_path):
    rows = summarize(run_sweep(weak_config))
    assert len(rows) == 1
    row = rows[0]
    assert row["shots"] == 2 and 0 <= row["logical_error_rate"] <= 1
    assert row["throughput_rounds_per_us"] > 0
    assert 0 < row["decoder_utilization"] < 1
    write_report(rows, tmp_path)
    assert (tmp_path / "sweep.csv").exists()
    text = (tmp_path / "sweep.md").read_text()
    assert "algorithm" in text and "frame_commit" in text
    from experiments.plots import plots
    plots(weak_config, rows, tmp_path)
    assert (tmp_path / "timeline.png").exists()
    # one p swept here, so no LER figure
    assert not (tmp_path / "ler.png").exists()


def test_strong_only_measures_the_strong_wires(strong_config, strong_shot):
    """The same points, on the strong path: csd carries the input, do the
    result, and the service identity holds unchanged."""
    links = strong_config.links
    assert strong_shot.means["input_link_per_window"] == pytest.approx(
        links["csd"].latency_us)
    assert strong_shot.means["output_link_per_window"] == pytest.approx(
        links["do"].latency_us)
    # dispatch -> done covers the transfer and the engine stages; with the
    # per-unit input slots the next window is dispatched under the current
    # compute, so slot wait can add on top of the components
    components = (strong_shot.means["input_link_per_window"]
                  + strong_shot.means["fetch"] + strong_shot.means["algorithm"]
                  + strong_shot.means["release"])
    assert strong_shot.means["service"] >= components - 1e-6
    assert strong_shot.means["algorithm"] == pytest.approx(5.0)
    assert not strong_shot.direct_mismatch


def test_strong_only_commits_every_window_on_the_strong_tier(strong_config):
    from experiments.build_run import build_run
    spec, _ = build_run(strong_config, physical_error_probability=0.001,
                        round_period_us=1.0, algorithm_latency_us=5.0, seed=1)
    completed = spec.build()
    tiers = {record.tier for record in completed.pauli_frame.snapshot().records}
    assert tiers == {"strong"}
    paths = {t["path"] for t in completed.result.link_traffic["transfers"]}
    assert "csd" in paths and "do" in paths and "wsd" not in paths


def test_trace_writes_one_log_file_per_shot(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    folder.joinpath("weak_baseline.yaml").write_text(
        (CONFIGS / "weak_baseline.yaml").read_text())
    folder.joinpath("traced.yaml").write_text(
        SMALL_SWEEP.format(base="weak_baseline", algorithm=0.028)
        + "trace: true\n")
    config = load_experiment(folder / "traced.yaml")
    monkeypatch.chdir(tmp_path)   # results_dir is cwd-relative
    measure_shot(config, physical_error_probability=0.001,
                 round_period_us=1.0, algorithm_latency_us=0.028, seed=0)
    trace_file = (tmp_path / "experiments/results/traced/trace"
                  / "p0.001_algo0.028_round1us_seed0.log")
    assert trace_file.exists()
    lines = trace_file.read_text().splitlines()
    assert any("START DECODE" in line for line in lines)
    assert any("QPU" in line for line in lines)


def test_loader_refuses_unknown_names(tmp_path, weak_config):
    bad = tmp_path / "bad.yaml"
    bad.write_text((CONFIGS / "weak_baseline.yaml").read_text()
                   .replace("mode: weak_baseline", "mode: strongest"))
    with pytest.raises(ValueError, match="mode"):
        load_experiment(bad)
