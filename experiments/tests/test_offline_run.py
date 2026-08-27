"""The offline LER lane: same decode as the closed-loop runner, no engine.

The equivalence tests are the lane's whole warrant: shot for shot, the
offline predictions must match measure_shot at the same seed, on both
tiers, including seeds that fail. The pipeline test walks plan -> shard
-> merge end to end the way a slurm array does.
"""

import csv
import json

import pytest

from experiments.experiment_config import load_experiment
from experiments.measure_shot import measure_shot
from experiments.offline_run import (SweepPointDecoder, decode_shard, merge,
                                     plan)

from test_decoder_units import MINIMAL_CONFIG, write_config, strong_unit


def closed_loop_failures(config, probability, distance, seeds) -> list:
    return [measure_shot(config, physical_error_probability=probability,
                         distance=distance, round_period_us=1.0,
                         seed=seed).logical_failure
            for seed in range(seeds)]


def test_offline_matches_closed_loop_on_the_weak_tier(tmp_path):
    config_path = write_config(tmp_path, {
        "decoder": {"weak": {**MINIMAL_CONFIG["decoder"]["weak"],
                             "algorithm": "pymatching"}}})
    config = load_experiment(config_path)
    probability, distance, seeds = 0.005, 3, 10
    point = SweepPointDecoder(config, physical_error_probability=probability,
                              distance=distance)
    offline = [point.decode_shot(seed).logical_failure
               for seed in range(seeds)]
    assert offline == closed_loop_failures(config, probability, distance,
                                           seeds)


def test_offline_matches_closed_loop_on_the_strong_tier(tmp_path):
    config_path = write_config(tmp_path, {
        "mode": "strong_only",
        "decoder": strong_unit("belief_matching")})
    config = load_experiment(config_path)
    probability, distance, seeds = 0.005, 3, 6
    point = SweepPointDecoder(config, physical_error_probability=probability,
                              distance=distance)
    offline = [point.decode_shot(seed).logical_failure
               for seed in range(seeds)]
    assert offline == closed_loop_failures(config, probability, distance,
                                           seeds)


def test_plan_shard_merge_pipeline(tmp_path, monkeypatch):
    """The array pipeline end to end: plan writes the run dir and shard
    table, each shard decodes its seed range, merge folds the csvs and
    stamps the manifest finished."""
    config_path = write_config(tmp_path, {
        "sweep": [{"physical_error_probability": [0.003, 0.005],
                   "distance": [3], "round_period_us": [1.0], "shots": 5}]})
    monkeypatch.chdir(tmp_path)
    run_dir = plan(config_path, seeds_per_shard=3)

    assert (run_dir / "slurm_offline.sh").exists()
    shard_lines = (run_dir / "shards.tsv").read_text().splitlines()
    assert shard_lines == ["3\t0.003\t0\t3", "3\t0.003\t3\t2",
                           "3\t0.005\t0\t3", "3\t0.005\t3\t2"]
    for shard_number in range(1, len(shard_lines) + 1):
        decode_shard(run_dir, shard_number)
    ler_rows = merge(run_dir)

    with open(run_dir / "shots.csv") as handle:
        shots = list(csv.DictReader(handle))
    assert len(shots) == 10
    assert sorted(int(row["seed"]) for row in shots
                  if row["physical_error_probability"] == "0.003") == list(
                      range(5))
    assert [(row["distance"], row["physical_error_probability"],
             row["shots"]) for row in ler_rows] == [(3, 0.003, 5),
                                                    (3, 0.005, 5)]
    assert (run_dir / "ler.png").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["finished_utc"]


def test_offline_refuses_a_timing_sweep(tmp_path):
    config_path = write_config(tmp_path, {
        "sweep": [{"physical_error_probability": [0.001], "distance": [3],
                   "round_period_us": [1.0, 0.5], "shots": 1}]})
    with pytest.raises(ValueError, match="one round period"):
        plan(config_path)


def test_offline_refuses_non_sliding_windows(tmp_path):
    config_path = write_config(tmp_path, {
        "windowing": {"scheme": "sandwich", "commit_rounds": None,
                      "buffer_rounds": None}})
    config = load_experiment(config_path)
    with pytest.raises(ValueError, match="sliding"):
        SweepPointDecoder(config, physical_error_probability=0.001,
                          distance=3)
