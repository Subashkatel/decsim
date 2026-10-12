"""Switching per window in three files: each window's answer and its label.

Four configurations on the same shots, d = 5 to 11 at four rates. Each
setting runs the shots at which union-find alone makes about 100
failures, 100 / its rate in Experiment 1, the same for all four. The
tasks are experiments/switching_per_window/run.py's
(tests/test_three_files_example.py).
"""

import math

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
# Experiment 1's union-find alone, (shots, failures) at each (distance,
# physical error rate), every row at its 100-failure stop: results, not
# constants
UNION_FIND_EXPERIMENT_ONE = {
    (5, 0.002): (4445, 110),
    (5, 0.003): (2000, 133),
    (5, 0.004): (2000, 284),
    (5, 0.005): (2000, 477),
    (7, 0.002): (18727, 105),
    (7, 0.003): (4000, 106),
    (7, 0.004): (2000, 157),
    (7, 0.005): (2000, 325),
    (9, 0.002): (200000, 226),
    (9, 0.003): (11765, 114),
    (9, 0.004): (4000, 159),
    (9, 0.005): (2000, 201),
    (11, 0.002): (392157, 100),
    (11, 0.003): (35844, 103),
    (11, 0.004): (6445, 123),
    (11, 0.005): (2000, 131),
}


def stop(cell: dict) -> decsim.CollectionSettings:
    """The shots at which union-find alone makes about 100 failures."""
    setting = (cell["distance"], cell["physical_error_probability"])
    shots, failures = UNION_FIND_EXPERIMENT_ONE[setting]
    expected_shots = constants.union_find_failures * shots / failures
    max_shots = math.ceil(expected_shots)
    return decsim.CollectionSettings(max_shots=max_shots)


def metadata(cell: dict) -> dict:
    """The cells under the names the collected results carry, so ids hold."""
    probability = cell["physical_error_probability"]
    return {
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
    }


sweep = decsim.Sweep(
    "switching_per_window",
    config.machine,
    grid={
        "physical_error_probability": [0.002, 0.003, 0.004, 0.005],
        "distance": [5, 7, 9, 11],
        "configuration": list(config.CONFIGURATIONS),
    },
    task_name="{configuration}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
