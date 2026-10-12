"""A few of Fig. 15a's points, before the whole grid.

PyMatching's complementary gap at d = 3, 5 and 7 and three physical
error rates, 20,000 shots each; weak alone, belief matching alone and
switching run the same seeds, so the same shots.

Usage
-----

From this folder, on a cluster (docs/tutorials/first_sweep.md): the
launch reads seconds_per_shot.csv beside this file, which `decsim
estimates` writes from a folded run.

```
decsim run pilot.py --slurm --out $PWD/../../results/switching_paper_pilot
python compare.py ../../results/switching_paper_pilot
```
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
DISTANCES = [3, 5, 7]
# three of Fig. 15's 20 rates, which run.py lists
PHYSICAL_ERROR_PROBABILITIES = [0.00195, 0.00479, 0.00811]
SHOTS = 20_000


def metadata(cell: dict) -> dict:
    """The cells under the names the collected results carry."""
    probability = cell["physical_error_probability"]
    period = constants.round_period_microseconds
    return {
        "decoding": cell["decoding"],
        "signal": cell["signal"],
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": period,
    }


sweep = decsim.Sweep(
    "switching_paper_pilot",
    config.machine,
    grid={
        "decoding": ["weak_alone", "strong_alone", "switching"],
        "signal": ["complementary_gap"],
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="{decoding}_{signal}_d{distance}_p{physical_error_probability}",
    stop=decsim.CollectionSettings(max_shots=SHOTS),
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
