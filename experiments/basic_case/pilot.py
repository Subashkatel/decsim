r"""The basic case's working set: 20,000 shots at d = 5 and 7.

The same three runs as run.py at d = 5 and 7 and every p, on the first
20,000 root seeds of each saved dataset, with every window's outcome
recorded. Steps 1 to 10 of the plan are built and checked on it while
run.py's million-shot runs go in the background.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run pilot.py --slurm \
  --out $PWD/../../results/basic_case_pilot \
  --set constants.observation.record_window_outcomes=true
python plots.py $PWD/../../results/basic_case_pilot
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
SHOTS = 20_000
DISTANCES = (5, 7)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    probability = cell["physical_error_probability"]
    qpu = task.machine.held("qpu")
    period = qpu.round_period_microseconds
    return {
        "decoding": cell["decoding"],
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": period,
    }


stop = decsim.CollectionSettings(max_shots=SHOTS)
sweep = decsim.Sweep(
    "basic_case_pilot",
    config.machine,
    grid={
        "decoding": config.DECODINGS,
        "distance": DISTANCES,
        "physical_error_probability": config.PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
