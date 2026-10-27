r"""The basic case's threshold sweep: switching at 0 to 50 dB.

Toshio et al. 2510.25222 Fig. 17 sweeps the gap threshold from 0 to
50 dB. This sweeps it in 5 dB steps at d = 5 and 7 and every p, on the
calibration shots, seeds 0 to 19,999 of each saved dataset, the same
shots as pilot.py. A threshold picked here is evaluated on the held-out
seeds from 20,000 (CollectionSettings.first_seed), which never meet
these.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run threshold_sweep.py --slurm \
  --out $PWD/../../results/basic_case_thresholds
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
CALIBRATION_SHOTS = 20_000
DISTANCES = (5, 7)
THRESHOLD = "constants.escalation.threshold.threshold"
THRESHOLDS = tuple(f"{decibels}dB" for decibels in range(0, 55, 5))


def machine(distance: int, physical_error_probability: float):
    """Switching at one point; the threshold is the cell's constant."""
    return config.machine("switching", distance, physical_error_probability)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    probability = cell["physical_error_probability"]
    qpu = task.machine.held("qpu")
    period = qpu.round_period_microseconds
    return {
        "decoding": "switching",
        "threshold": cell[THRESHOLD],
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": period,
    }


stop = decsim.CollectionSettings(max_shots=CALIBRATION_SHOTS)
sweep = decsim.Sweep(
    "basic_case_thresholds",
    machine,
    grid={
        "distance": DISTANCES,
        "physical_error_probability": config.PHYSICAL_ERROR_PROBABILITIES,
        THRESHOLD: THRESHOLDS,
    },
    task_name=(
        "switching_d{distance}_p{physical_error_probability}"
        "_{constants.escalation.threshold.threshold}"
    ),
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
