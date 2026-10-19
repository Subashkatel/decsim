r"""The rate controller on held-out shots, against the static threshold.

Each point's switching run with RateThreshold (slice 05) holding 5% or
10% of windows escalated, on seeds 20,000 to 29,999, the first half of
threshold_heldout.py's static runs. It starts either at the threshold
calibration.py gives for its target, or cold at 20 dB. The controller
works when its share settles at the target and its error rate matches
the static threshold's on the same shots at the same share.

An adaptive task's shots run in order, so this runs without --slurm:
one job, a process a task.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
sbatch -N1 -c24 --mem=96G -t 12:00:00 --wrap "python -m decsim run \
  rate_controller.py --processes 24 \
  --out $PWD/../../results/basic_case_controller"
```
"""

import decsim
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines

config = sweeps.load_beside(__file__, "config.py")
calibration = sweeps.load_beside(__file__, "calibration.py")
FIRST_HELDOUT_SEED = 20_000
CONTROLLER_SHOTS = 10_000
TARGETS = (0.05, 0.1)
COLD_START_DECIBELS = 20.0


def task(point: tuple, target: float, start: str) -> decsim.Task:
    """One point's switching run holding target, from start."""
    distance, probability = point
    decibels = COLD_START_DECIBELS
    if start == "calibrated":
        decibels = calibration.CALIBRATED[point][target]
    rate = threshold_sources.RateThreshold.Settings(decibels, target)
    switching = config.machine("switching", distance, probability)
    machine = switching.with_settings(machines.SWITCHING, threshold=rate)
    qpu = machine.held("qpu")
    metadata = {
        "decoding": "switching",
        "target": target,
        "start": start,
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": distance,
        "qpu.round_period_microseconds": qpu.round_period_microseconds,
    }
    stop = decsim.CollectionSettings(
        max_shots=CONTROLLER_SHOTS, first_seed=FIRST_HELDOUT_SEED
    )
    name = f"rate_d{distance}_p{probability}_{target}_{start}"
    return decsim.Task(name, machine, metadata, stop)


tasks = [
    task(point, target, start)
    for point in calibration.CALIBRATED
    for target in TARGETS
    for start in ("calibrated", "cold")
]
experiment = decsim.Experiment(
    "basic_case_controller", tasks, constants=(config.constants,)
)
