r"""Speculation beside the sliding and the double window, on memory.

A switching machine handles a window it escalates in one of three ways
(escalation/strong_window_shapes.py):
  - sliding: the redo window under held boundaries, serial switching:
    the window after an escalated one waits for the strong answer;
  - double: the double window (Toshio et al. 2510.25222 Sec. III C):
    the weak chain goes on past a padded strong region, and the weak
    boundaries it hands on are never checked;
  - speculation: the redo window under checked boundaries: the window
    after decodes on the weak boundary, and is decoded again when the
    strong boundary that lands is different (windows/input_checks.py).
Union-find alone and belief matching alone are the two ends.

Every task reads the point's saved dataset (make_datasets.py), so every
scheme decodes the same shots, and shots.csv pairs them by seed: sliding
and speculation must predict alike, shot by shot (the correctness gate),
and the double window's cost is in the shots where it does not. Task
names: {scheme}_d{distance}_p{probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run speculation.py --only speculation_d5_p0.003 --shots 20
python -m decsim run speculation.py --slurm --out RESULTS
```
"""

from typing import Any

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
# the decoders alone, the two ends every scheme sits between
ENDS = ("weak_alone", "strong_alone")
DISTANCES = (5, 7)
PHYSICAL_ERROR_PROBABILITIES = (0.002, 0.003, 0.005)
# about 1,000 failures a scheme at p = 0.003 (static frontier: 4 to 5% a
# shot at d = 7); the cost is about 0.8 s a shot at d = 5, 2.7 at d = 7
SHOTS_BY_DISTANCE = {5: 20_000, 7: 10_000}


def machine(
    scheme: str, distance: int, physical_error_probability: float
) -> Any:
    """One scheme, or one decoder alone, at one point."""
    if scheme in ENDS:
        return config.machine(scheme, distance, physical_error_probability)
    switching = config.machine(
        "switching", distance, physical_error_probability
    )
    return config.with_scheme(switching, config.SCHEMES[scheme])


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    probability = cell["physical_error_probability"]
    qpu = task.machine.held("qpu")
    period = qpu.round_period_microseconds
    return {
        "scheme": cell["scheme"],
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": period,
    }


def stop(cell: dict) -> decsim.CollectionSettings:
    """The shots of the cell's distance, the same for every scheme."""
    shots = SHOTS_BY_DISTANCE[cell["distance"]]
    return decsim.CollectionSettings(max_shots=shots)


sweep = decsim.Sweep(
    "speculation",
    machine,
    grid={
        "scheme": (*config.SCHEMES, *ENDS),
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="{scheme}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
