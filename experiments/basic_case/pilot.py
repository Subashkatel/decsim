"""The basic case's pilot: the first 2,000 shots of every point.

The same three runs at the same points as run.py, on the first 2,000
root seeds of each saved dataset, with every window's outcome recorded:
it checks the windows match across the runs, the failure counts look
sane, and how long each full run will take, before the million-shot runs.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run pilot.py \
  --set constants.observation.record_window_outcomes=true
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
SHOTS = 2_000


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
        "distance": config.DISTANCES,
        "physical_error_probability": config.PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
