"""Experiment 1, the switching baseline, and its paired weak-alone run.

A union-find weak tier on Helios's cycle law beside a Relay-BP-5 strong
tier at its measured A100 time, escalating windows whose cluster gap is
under 20 dB into double windows (Toshio et al. 2510.25222 Sec. III C),
at d = 5 to 13 and six physical error rates, 100 rounds a shot. The
machine and every sourced value are decsim/presets/experiment_one.py's.
The weak-alone tasks run the same machine with no escalation, every
window kept by the weak tier; a shot's samples depend on its seed alone,
so each pairs with the switching shot of its seed.

A task stops at 100 failures or five million shots. A switching piece
is 50 shots: the slowest task, d = 13 at p = 0.005, is estimated at up
to 600 s a shot (a measured 32 s at d = 9, p = 0.003, about three times
the region's detectors, twice the escalations), so a piece is at most
about 8 hours, a third of a 24-hour Slurm job. A weak-alone piece is 2,000
shots, since weak-alone shots took 0.17 to 0.41 s each.

Usage
-----

```
decsim run experiments/switching_baseline/run.py
```
"""

import decsim
import decsim.presets.experiment_one as experiment_one

NAME = "switching_baseline"
DISTANCES = (5, 7, 9, 11, 13)
PHYSICAL_ERROR_PROBABILITIES = (0.0005, 0.001, 0.002, 0.003, 0.004, 0.005)
SWITCHING_COLLECTION = decsim.CollectionSettings(
    max_failures=100, max_shots=5_000_000, min_shots=0, piece_rounds=5000
)
WEAK_ALONE_COLLECTION = decsim.CollectionSettings(
    max_failures=100, max_shots=5_000_000, min_shots=0, piece_rounds=200_000
)


def switching_baseline_tasks() -> list:
    """Every switching task, then every weak-alone task.

    Each set runs rate by distance, the distance fastest, as its earlier
    runs did.
    """
    grid = decsim.grid(
        physical_error_probability=PHYSICAL_ERROR_PROBABILITIES,
        distance=DISTANCES,
    )
    switching_tasks = []
    weak_alone_tasks = []
    for values in grid:
        probability = values["physical_error_probability"]
        distance = values["distance"]
        metadata = _metadata(distance, probability)
        cell_name = f"d{distance}_p{probability}"
        switching_machine = experiment_one.switching(distance, probability)
        switching_task = decsim.Task(
            f"switching_{cell_name}",
            switching_machine,
            metadata,
            SWITCHING_COLLECTION,
        )
        switching_tasks.append(switching_task)
        weak_alone_machine = experiment_one.weak_alone(distance, probability)
        weak_alone_task = decsim.Task(
            f"weak_alone_{cell_name}",
            weak_alone_machine,
            metadata,
            WEAK_ALONE_COLLECTION,
        )
        weak_alone_tasks.append(weak_alone_task)
    return switching_tasks + weak_alone_tasks


def _metadata(distance: int, physical_error_probability: float) -> dict:
    """The swept cells under the earlier runs' names, so the ids match."""
    return {
        "workload.arguments.physical_error_probability": (
            physical_error_probability
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": (
            experiment_one.ROUND_PERIOD_MICROSECONDS
        ),
    }


tasks = switching_baseline_tasks()
experiment = decsim.Experiment(NAME, tasks)
