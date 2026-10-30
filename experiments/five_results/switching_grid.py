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
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
calibration = sweeps.load_beside(__file__, "calibration.py")
# design -> (window rule, strong cores)
DESIGNS = {
    "sliding_1core": ("sliding", 1),
    "speculation_1core": ("speculation", 1),
    "double_1core": ("double", 1),
    "double_2core": ("double", 2),
}
SHARES = (0.05, 0.1, 0.2, 0.3)
DISTANCES = (5, 7)
PHYSICAL_ERROR_PROBABILITIES = (0.003,)
TASK_NAME = "{design}_{share}_d{distance}_p{physical_error_probability}"


def machine(
    design: str,
    share: float,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """One design at the threshold calibrated to the share."""
    scheme, strong_unit_count = DESIGNS[design]
    point = (distance, physical_error_probability)
    threshold_decibels = calibration.CALIBRATED[point][share]
    return config.switching(
        scheme,
        distance,
        physical_error_probability,
        threshold_decibels,
        strong_unit_count,
    )


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The point's cells as config.metadata names them, and the design's parts."""
    scheme, strong_unit_count = DESIGNS[cell["design"]]
    named = config.metadata(cell, task)
    return {**named, "scheme": scheme, "strong_unit_count": strong_unit_count}


stop = decsim.CollectionSettings(
    max_shots=config.EVALUATION_SHOTS, first_seed=config.EVALUATION_FIRST_SEED
)
sweep = decsim.Sweep(
    "five_results_switching_grid",
    machine,
    grid={
        "design": tuple(DESIGNS),
        "share": SHARES,
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name=TASK_NAME,
    stop=stop,
    metadata=metadata,
)
experiment = sweep.experiment()
