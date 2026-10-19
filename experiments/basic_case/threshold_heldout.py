r"""The calibrated thresholds on held-out shots.

threshold_sweep.py ran seeds 0 to 19,999 at 0 to 50 dB, and
threshold_plots.py's calibration.csv gave, per point, the threshold that
escalates 5, 10 and 20% of windows on those seeds. This runs each on
seeds 20,000 to 39,999 of the same datasets, which calibration never
saw (CollectionSettings.first_seed): a threshold holds when it escalates
about its target there too.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run threshold_heldout.py --slurm \
  --out $PWD/../../results/basic_case_heldout
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
FIRST_HELDOUT_SEED = 20_000
HELDOUT_SHOTS = 20_000
THRESHOLD = "constants.escalation.threshold.threshold"
# calibration.csv of the sweep folded on 2026-10-19, in dB, for targets
# of 5, 10 and 20% of windows escalated
CALIBRATED = {
    (5, 0.002): {"0.05": 15.71, "0.1": 20.92, "0.2": 30.9},
    (5, 0.003): {"0.05": 8.66, "0.1": 14.98, "0.2": 18.49},
    (5, 0.005): {"0.05": 2.39, "0.1": 4.78, "0.2": 9.89},
    (7, 0.002): {"0.05": 15.55, "0.1": 21.1, "0.2": 30.62},
    (7, 0.003): {"0.05": 8.66, "0.1": 14.61, "0.2": 17.87},
    (7, 0.005): {"0.05": 1.99, "0.1": 3.97, "0.2": 8.16},
}


def point_experiment(point: tuple, targets: dict) -> decsim.Experiment:
    """One point's targets, each a task at its calibrated threshold."""
    distance, probability = point
    thresholds = [f"{decibels}dB" for decibels in targets.values()]
    target_of = {
        f"{decibels}dB": target for target, decibels in targets.items()
    }

    def machine():
        return config.machine("switching", distance, probability)

    def metadata(cell: dict, task: decsim.Task) -> dict:
        threshold = cell[THRESHOLD]
        qpu = task.machine.held("qpu")
        return {
            "decoding": "switching",
            "target": target_of[threshold],
            "threshold": threshold,
            "workload.arguments.physical_error_probability": probability,
            "qpu.distance": distance,
            "qpu.round_period_microseconds": qpu.round_period_microseconds,
        }

    stop = decsim.CollectionSettings(
        max_shots=HELDOUT_SHOTS, first_seed=FIRST_HELDOUT_SEED
    )
    name = f"switching_d{distance}_p{probability}_" + "{" + THRESHOLD + "}"
    sweep = decsim.Sweep(
        "basic_case_heldout",
        machine,
        grid={THRESHOLD: thresholds},
        task_name=name,
        stop=stop,
        metadata=metadata,
        constants=constants,
    )
    return sweep.experiment()


parts = [
    point_experiment(point, targets) for point, targets in CALIBRATED.items()
]
tasks = [task for part in parts for task in part.tasks]
experiment = decsim.Experiment(
    "basic_case_heldout", tasks, constants=parts[0].constants
)
