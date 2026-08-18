"""The baseline closed loop measures every point on real Stim data."""

from pathlib import Path

import pytest

from experiments.baseline_closed_loop import (
    DEFAULT_CONFIG, POINTS, load_config, measure_shot, run_sweep, summarize,
    write_report,
)


@pytest.fixture(scope="module")
def small_config():
    config = load_config(DEFAULT_CONFIG)
    config.update(rounds_per_shot=15, seeds=[0, 1], round_period_us=[1.0],
                  algorithm_latency_us=[0.028])
    return config


@pytest.fixture(scope="module")
def shot(small_config):
    return measure_shot(small_config, round_period_us=1.0,
                        algorithm_latency_us=0.028, seed=0)


def test_every_point_is_measured_and_positive_where_it_must_be(shot):
    assert shot.windows >= 3
    for point in POINTS:
        assert point in shot.means and point in shot.maxes
    for charged in ("c2b_per_round", "fetch", "algorithm", "release",
                    "cwd_per_window", "wdo_per_window", "frame_commit"):
        assert shot.means[charged] > 0, charged


def test_decoder_service_is_exactly_the_engine_stages(shot):
    assert shot.means["service"] == pytest.approx(
        shot.means["fetch"] + shot.means["algorithm"] + shot.means["release"])
    assert shot.means["algorithm"] == pytest.approx(0.028)


def test_configured_costs_appear_at_the_right_points(small_config, shot):
    engine = small_config["decoder"]["engine"]
    tick_us = 1 / engine["frequency_mhz"]
    assert shot.means["release"] == pytest.approx(engine["release_cycles_per_job"] * tick_us)
    assert shot.means["frame_commit"] == pytest.approx(small_config["pauli_frame"]["commit_us"])
    c2b = small_config["controller_to_buffer"]
    assert shot.means["c2b_per_round"] >= c2b["latency_us"]


def test_reaction_time_orders_the_points(shot):
    assert shot.means["last_round_to_frame"] <= shot.means["reaction_first_round"]
    assert shot.means["last_round_to_frame"] >= (
        shot.means["cwd_per_window"] + shot.means["service"]
        + shot.means["wdo_per_window"] + shot.means["frame_commit"])


def test_sweep_summary_and_report(small_config, tmp_path):
    rows = summarize(run_sweep(small_config))
    assert len(rows) == 1
    row = rows[0]
    assert row["shots"] == 2 and 0 <= row["logical_error_rate"] <= 1
    assert row["throughput_rounds_per_us"] > 0 and 0 < row["decoder_utilization"] < 1
    write_report(small_config, rows, tmp_path)
    assert (tmp_path / "sweep.csv").exists()
    text = (tmp_path / "sweep.md").read_text()
    assert "algorithm" in text and "frame_commit" in text


def test_anchor_published_numbers_load_and_host_method_runs():
    from experiments.baseline_anchor import (
        decsim_us_per_shot, host_us_per_shot, published_us_per_shot)
    published = published_us_per_shot()
    assert published[17] == pytest.approx(16.0062)
    assert 0 < host_us_per_shot(5, 0.001, num_shots=500) < 100
    assert decsim_us_per_shot(5, 0.001, shots=2) > 0
