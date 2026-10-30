r"""R3 and R5 on shots at d = 5 and 7: four designs at four escalated shares.

The switching pilot (switching_pilot.py) at d = 9 found that a second
strong core changes nothing for sliding and speculation on one patch,
since their strong decodes run one after another. So each distance here
runs four designs: sliding, speculation and the double window on 1
core, and the double window on 2. Each runs at the thresholds that
escalated 5, 10, 20 and 30% of union-find's windows on the calibration
seeds (calibration.py), on the evaluation seeds from 10,000, as
r1_shots.py does, so shots.csv pairs every design with the two
decoders alone seed by seed. Task names, as the pilot's:
{scheme}_{strong_unit_count}core_{share}_d{distance}_p{probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run switching_grid.py --only \
  double_2core_0.1_d5_p0.003 --shots 5
python -m decsim run switching_grid.py --record --out RESULTS
```

then the jobs and the fold of docs/how-to/run_on_a_slurm_cluster.md.
"""

import decsim
import decsim.experiments.sweeps as sweeps

designs = sweeps.load_beside(__file__, "switching_designs.py")
DISTANCES = (5, 7)

sweep = decsim.Sweep(
    "five_results_switching_grid",
    designs.machine,
    grid={
        "design": tuple(designs.DESIGNS),
        "share": designs.SHARES,
        "distance": DISTANCES,
        "physical_error_probability": designs.PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name=designs.TASK_NAME,
    stop=designs.stop,
    metadata=designs.metadata,
)
experiment = sweep.experiment()
