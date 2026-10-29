r"""R3 and R5 on shots: switching designs at three escalated shares.

Each design is a window rule and a count of strong cores: sliding,
speculation or the double window, on 1 or 2 cores, at the thresholds
that escalated 5, 10 and 20% of union-find's windows on the calibration
seeds (calibration.py). R3 asks which designs keep the error rate near
correlated matching alone's; R5 compares the three rules on the same
shots. On one patch, sliding and speculation decode one window after
another, so their second core should change nothing, and the double
window keeps several strong decodes in flight (experiments.md, R3).
Designs with 4 patches wait for saved datasets of 4 patches.

Every task decodes the evaluation seeds (config.EVALUATION_FIRST_SEED),
as r1_shots.py does, so shots.csv pairs each design with the two
decoders alone seed by seed. Task names:
{scheme}_{strong_unit_count}core_{share}_d{distance}_p{probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run switching_pilot.py --only \
  double_2core_0.1_d9_p0.003 --shots 5
python -m decsim run switching_pilot.py --record --out RESULTS
```

then the jobs and the fold of docs/how-to/run_on_a_slurm_cluster.md.
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
calibration = sweeps.load_beside(__file__, "calibration.py")
SCHEMES = ("sliding", "speculation", "double")
STRONG_UNIT_COUNTS = (1, 2)
SHARES = (0.05, 0.1, 0.2)
DISTANCES = (9,)
PHYSICAL_ERROR_PROBABILITIES = (0.003,)


def machine(
    scheme: str,
    strong_unit_count: int,
    share: float,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """One design at the threshold calibrated to the share."""
    point = (distance, physical_error_probability)
    threshold_decibels = calibration.CALIBRATED[point][share]
    return config.switching(
        scheme,
        distance,
        physical_error_probability,
        threshold_decibels,
        strong_unit_count,
    )


stop = decsim.CollectionSettings(
    max_shots=config.EVALUATION_SHOTS, first_seed=config.EVALUATION_FIRST_SEED
)
sweep = decsim.Sweep(
    "five_results_switching_pilot",
    machine,
    grid={
        "scheme": SCHEMES,
        "strong_unit_count": STRONG_UNIT_COUNTS,
        "share": SHARES,
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name=(
        "{scheme}_{strong_unit_count}core_{share}"
        "_d{distance}_p{physical_error_probability}"
    ),
    stop=stop,
    metadata=config.metadata,
)
experiment = sweep.experiment()
