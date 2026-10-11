"""The switching experiment in three files: two weak tiers escalating.

Union-find's cluster gap with Toshio's strong decoder, then PyMatching's
complementary gap with belief matching, at d = 3 and 5. The tasks are
experiments/switching/run.py's (tests/test_three_files_example.py).
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants


def stop(cell: dict) -> decsim.CollectionSettings:
    """A configuration's shots."""
    key = f"{cell['configuration']}_shots"
    shots = constants.stop[key]
    return decsim.CollectionSettings(max_shots=shots)


def metadata(cell: dict) -> dict:
    """The cells under the names the collected results carry, so ids hold."""
    probability = constants.physical_error_probability
    period = constants.round_period_microseconds
    return {
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": period,
    }


sweep = decsim.Sweep(
    "switching",
    config.machine,
    grid={"configuration": ["cluster_gap", "redo_window"], "distance": [3, 5]},
    task_name="{configuration}_d{distance}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
