r"""The switching grid's 30% share at d = 9, which the pilot did not run.

The pilot (switching_pilot.py) ran d = 9 at 5, 10 and 20%. This adds
the 30% threshold for the two designs that span its answers: sliding on
1 core, whose answers speculation repeats, and the double window on 2
cores, whose answers its 1-core form repeats. The machines and seeds are
switching_grid.py's.

Usage
-----

```
export PYTHONPATH=$PWD/../..
python -m decsim run switching_grid_d9.py --record --out RESULTS
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

designs = sweeps.load_beside(__file__, "switching_designs.py")

sweep = decsim.Sweep(
    "five_results_switching_grid_d9",
    designs.machine,
    grid={
        "design": ("sliding_1core", "double_2core"),
        "share": (0.3,),
        "distance": (9,),
        "physical_error_probability": (0.003,),
    },
    task_name=designs.TASK_NAME,
    stop=designs.stop,
    metadata=designs.metadata,
)
experiment = sweep.experiment()
