"""The basic case: weak alone, strong alone and switching at every point.

d = 5, 7, 9, 11 at p = 0.002, 0.003, 0.005, exactly a million shots a
run, every run reading the point's saved dataset (make_datasets.py), so
the three runs of a point decode the same shots. constants.yaml holds
the numbers, config.py the machine, and this file the sweep.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run run.py --only switching_d5_p0.002
python -m decsim run run.py --slurm --out $PWD/../../results/basic_case
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants


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


stop = decsim.CollectionSettings(max_shots=config.SHOTS)
sweep = decsim.Sweep(
    "basic_case",
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
