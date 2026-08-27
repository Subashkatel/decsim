"""The decoder card: two tiers, two distinct units, the mode picks one.

The shape under test is the owner's 2026-08-26 ruling: the algorithm is
structure on a per-tier unit card (decoder.weak / decoder.strong), never a
sweep axis, mirroring the tiered architecture itself (Toshio arXiv
2510.25222: lightweight decoders decode constantly, a separate accurate
decoder is invoked on demand) and gem5's config split (structure on the
component, parameters swept around it). These tests also keep
reference.yaml and the loader from drifting apart.
"""

from pathlib import Path

import pytest
import yaml

from experiments.experiment_config import load_experiment
from experiments.measure_shot import measure_shot

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# A complete runnable config, small enough for a functional test. Tests
# override keys through the `overrides` dict (top-level replacement, the
# same rule as `extends`).
MINIMAL_CONFIG = {
    "mode": "weak_baseline",
    "code_task": "surface_code:rotated_memory_z",
    "rounds_per_shot": 15,
    "windowing": {"scheme": "sliding", "commit_rounds": None,
                  "buffer_rounds": None},
    "sweep": [{"physical_error_probability": [0.001], "distance": [3],
               "round_period_us": [1.0], "shots": 1}],
    "controller": {"clock": "fridge", "t_binary_availability_cycles": 0,
                   "t_pack_cycles": 0},
    "clocks": {"fridge": 250.0, "room": 250.0},
    "links": {"qc": {"latency_cycles": 1, "clock": "fridge",
                     "bits_per_cycle": None}},
    "buffers": {"buffer_0_size": None, "buffer_1_size": None,
                "packing_workspace_size": None},
    "decoder": {"weak": {"algorithm": 0.028, "units": 1,
                         "unit_buffer_size": None,
                         "engine": {"clock": "fridge",
                                    "fetch_cycles_per_round": 1,
                                    "release_cycles_per_job": 1}}},
    "pauli_frame": {"clock": "fridge", "commit_cycles": 1},
}


def write_config(tmp_path, overrides: dict) -> Path:
    raw = dict(MINIMAL_CONFIG)
    raw.update(overrides)
    config_path = tmp_path / "unit_test_config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return config_path


def strong_unit(algorithm) -> dict:
    return {"strong": {"algorithm": algorithm, "units": 1,
                       "unit_buffer_size": None,
                       "engine": {"clock": "room",
                                  "fetch_cycles_per_round": 1,
                                  "release_cycles_per_job": 1}}}


def test_reference_config_defines_both_tiers_and_the_mode_picks_weak():
    config = load_experiment(CONFIGS_DIR / "reference.yaml")
    assert config.decoder.weak.algorithm == "pymatching"
    assert config.decoder.strong.algorithm == "belief_matching"
    assert config.active_decoder is config.decoder.weak
    # engine cycles price on a named domain, resolved once like the links
    assert config.decoder.weak.engine.clock == "fridge"
    assert config.decoder.weak.engine.frequency_mhz == config.clocks["fridge"]
    assert config.decoder.strong.engine.clock == "room"


def test_every_shipped_config_loads():
    for config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        config = load_experiment(config_path)
        assert config.active_decoder is not None, config_path.name


def test_a_mode_without_its_tier_is_refused(tmp_path):
    config_path = write_config(tmp_path, {
        "mode": "strong_only"})   # decoder defines only weak
    with pytest.raises(ValueError, match="decoder.strong"):
        load_experiment(config_path)


def test_unknown_algorithms_and_stale_keys_fail_loudly(tmp_path):
    unknown_algorithm = write_config(tmp_path, {
        "decoder": {"weak": {**MINIMAL_CONFIG["decoder"]["weak"],
                             "algorithm": "union_find"}}})
    with pytest.raises(ValueError, match="algorithm"):
        load_experiment(unknown_algorithm)

    old_flat_decoder = write_config(tmp_path, {
        "decoder": {"units": 1, "unit_buffer_size": None,
                    "engine": {"clock": "fridge", "fetch_cycles_per_round": 1,
                               "release_cycles_per_job": 1}}})
    with pytest.raises(ValueError, match="tiers"):
        load_experiment(old_flat_decoder)

    old_sweep_axis = write_config(tmp_path, {
        "sweep": [{"physical_error_probability": [0.001], "distance": [3],
                   "round_period_us": [1.0],
                   "algorithm_latency_us": [0.028], "shots": 1}]})
    with pytest.raises(ValueError, match="decoder card"):
        load_experiment(old_sweep_axis)

    fixed_distance_key = write_config(tmp_path, {
        "sweep": [{"physical_error_probability": [0.001],
                   "round_period_us": [1.0], "shots": 1}]})
    with pytest.raises(KeyError):
        load_experiment(fixed_distance_key)


def test_engine_clock_must_name_a_clock_domain(tmp_path):
    config_path = write_config(tmp_path, {
        "decoder": {"weak": {**MINIMAL_CONFIG["decoder"]["weak"],
                             "engine": {"clock": "sfq",
                                        "fetch_cycles_per_round": 1,
                                        "release_cycles_per_job": 1}}}})
    with pytest.raises(KeyError):
        load_experiment(config_path)


def test_weak_unit_loop_matches_direct_pymatching(tmp_path):
    """The functional gate: the loop with the weak unit's real MWPM reaches
    the same prediction as whole-circuit PyMatching on the same events."""
    config_path = write_config(tmp_path, {
        "decoder": {"weak": {**MINIMAL_CONFIG["decoder"]["weak"],
                             "algorithm": "pymatching"}}})
    config = load_experiment(config_path)
    for seed in range(3):
        measurement = measure_shot(config, physical_error_probability=0.005,
                                   distance=3, round_period_us=1.0, seed=seed)
        assert measurement.algorithm == "pymatching"
        assert measurement.windows > 0
        assert not measurement.direct_mismatch


def test_strong_unit_runs_belief_matching(tmp_path):
    config_path = write_config(tmp_path, {
        "mode": "strong_only",
        "decoder": strong_unit("belief_matching")})
    config = load_experiment(config_path)
    measurement = measure_shot(config, physical_error_probability=0.001,
                               distance=3, round_period_us=1.0, seed=0)
    assert measurement.algorithm == "belief_matching"
    assert measurement.windows > 0
    assert not measurement.logical_failure


def test_report_rows_carry_the_algorithm_column(tmp_path):
    from experiments.sweep_report import link_rows, summarize
    config_path = write_config(tmp_path, {})
    config = load_experiment(config_path)
    measurements = [measure_shot(config, physical_error_probability=0.001,
                                 distance=3, round_period_us=1.0, seed=seed)
                    for seed in range(2)]
    rows = summarize(measurements)
    assert len(rows) == 1
    assert rows[0]["algorithm"] == 0.028
    per_link = link_rows(measurements)
    assert per_link and all(row["algorithm"] == 0.028 for row in per_link)


def test_rounds_per_shot_scales_with_the_swept_distance(tmp_path):
    """"10d" is Toshio 2510.25222's memory-experiment convention: the shot
    length follows the swept code distance."""
    config_path = write_config(tmp_path, {
        "rounds_per_shot": "10d",
        "sweep": [{"physical_error_probability": [0.001], "distance": [3, 5],
                   "round_period_us": [1.0], "shots": 1}]})
    config = load_experiment(config_path)
    assert config.rounds_per_shot.rounds_for(3) == 30
    assert config.rounds_per_shot.rounds_for(5) == 50
    assert str(config.rounds_per_shot) == "10d"
    measurement = measure_shot(config, physical_error_probability=0.001,
                               distance=5, round_period_us=1.0, seed=0)
    assert measurement.distance == 5
    assert measurement.windows > 5


def test_a_run_writes_its_manifest_and_per_shot_records(tmp_path, monkeypatch):
    """One tiny run end to end: a timestamped run dir with manifest.json,
    the config copy, shots.csv, sweep.csv and links.csv."""
    import csv
    import json
    from experiments.run import run_experiment

    config_path = write_config(tmp_path, {})
    monkeypatch.chdir(tmp_path)
    run_dir, rows = run_experiment(config_path)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["versions"]["stim"]
    assert manifest["resolved_config"]["mode"] == "weak_baseline"
    assert manifest["started_utc"] and manifest["finished_utc"]

    with open(run_dir / "shots.csv") as handle:
        shots = list(csv.DictReader(handle))
    assert len(shots) == 1 and shots[0]["seed"] == "0"

    assert run_dir.name.endswith("-unit_test_config")
    assert (run_dir / "config" / "unit_test_config.yaml").exists()
    assert (run_dir / "sweep.csv").exists() and (run_dir / "links.csv").exists()
