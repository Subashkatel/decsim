"""decsim.results: run folders read back as one table.

The folders come from `decsim run` itself, so the reader is pinned
against what the writer wrote.
"""

import pytest
import scipy.stats

import decsim.experiments.command as command
import decsim.results as results
import tests.experiments.yaml_configs as yaml_configs

TWO_POINT_SWEEP = [
    {
        "axes": {
            "workload.arguments.physical_error_probability": [0.003, 0.01],
            "qpu.distance": [3],
            "qpu.round_period_microseconds": [1.0],
        },
        "collection": {"max_shots": 20},
    }
]

# Settings columns a loaded row holds.
ROUNDS_COLUMN = "settings.workload.row_settings.arguments.rounds_per_shot"
PROBABILITY_COLUMN = (
    "settings.workload.row_settings.arguments.physical_error_probability"
)


def _collected(folder, overrides: dict):
    """One `decsim run` of the minimal config into its own folder."""
    folder.mkdir()
    sweep = {"sweep": TWO_POINT_SWEEP}
    sweep.update(overrides)
    config_path = yaml_configs.write_config(folder, sweep)
    run_dir = folder / "run"
    command.main(["run", str(config_path), "--out", str(run_dir)])
    return run_dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A run of one yaml, and one at another shot length."""
    root = tmp_path_factory.mktemp("runs")
    first_folder = root / "first"
    shorter_folder = root / "shorter"
    shorter_workload = yaml_configs.memory_workload(12)
    shorter_overrides = {"workload": shorter_workload}
    first = _collected(first_folder, {})
    shorter = _collected(shorter_folder, shorter_overrides)
    return {"first": first, "shorter": shorter}


def test_load_gives_a_row_per_point_with_its_results_and_settings(runs):
    rows = results.load(runs["first"], runs["shorter"])

    rounds = [row[ROUNDS_COLUMN] for row in rows]
    assert len(rows) == 4
    assert rounds == [15, 15, 12, 12]
    assert rows[0]["run_dir"] == str(runs["first"])
    assert rows[0]["shots"] == 20


def test_a_loaded_row_holds_what_an_error_rate_figure_is_drawn_from(runs):
    """The estimate, its exact limits and counts, and the point's values.

    decsim draws no error rate figure; a reader draws one from these
    rows, the numbers sinter's plot_error_rate reads off its csv. The
    point stopped at its shot cap, so its limits are Clopper and
    Pearson's, which scipy's exact binomial interval is.
    """
    rows = results.load(runs["first"])

    noisier = rows[1]
    failures = noisier["prefix_failures"]
    scored_shots = noisier["prefix_scored_shots"]
    test = scipy.stats.binomtest(failures, scored_shots)
    interval = test.proportion_ci(method="exact")
    assert noisier["state"] == "cap"
    assert failures > 0
    assert noisier["logical_error_rate_estimate"] == failures / scored_shots
    assert noisier["logical_error_rate_low"] == pytest.approx(
        interval.low, rel=1e-12
    )
    assert noisier["logical_error_rate_high"] == pytest.approx(
        interval.high, rel=1e-12
    )
    assert noisier[PROBABILITY_COLUMN] == 0.01
