"""Experiment 1 in three files: switching and its weak-alone pair.

d = 5 to 13 at six physical error rates, 100 rounds a shot. The tasks
are experiments/switching_baseline/run.py's, name for name and id for
id (tests/test_three_files_example.py): constants.yaml holds the
numbers, config.py the machine, and this file the sweep.

Usage
-----

From this folder:

```
decsim run run.py --only switching_d5_p0.003
decsim run run.py --set constants.escalation.threshold.threshold=25dB
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants


def stop(cell: dict) -> decsim.CollectionSettings:
    """A cell's stop; its piece is its decoding's."""
    settings = constants.stop
    piece_key = f"{cell['decoding']}_piece_rounds"
    return decsim.CollectionSettings(
        max_failures=settings.max_failures,
        max_shots=settings.max_shots,
        min_shots=settings.min_shots,
        piece_rounds=settings[piece_key],
    )


def metadata(cell: dict, settings: decsim.MachineSettings) -> dict:
    """The cells under the names the collected results carry, so ids hold."""
    probability = cell["physical_error_probability"]
    period = settings.qpu.round_period_microseconds
    return {
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": period,
    }


sweep = decsim.Sweep(
    "switching_baseline",
    config.machine,
    grid={
        "decoding": ["switching", "weak_alone"],
        "physical_error_probability": [
            0.0005,
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
        ],
        "distance": [5, 7, 9, 11, 13],
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
