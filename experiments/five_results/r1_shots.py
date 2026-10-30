r"""R1 on shots: union-find alone and correlated matching alone.

Both machines decode the same shots of the point's saved dataset
(basic_case/make_datasets.py), so shots.csv pairs them seed by seed: the
evaluation seeds, from 10,000, which the gap calibration's seeds 0 to
9,999 never meet (gap_calibration.py). switching_pilot.py runs the same
seeds, so these two machines are its ends. At p = 0.003 belief matching
failed 4 to 8% of shots of 10 d rounds at d = 5 and 7 (the speculation
study), so 10,000 shots give each machine more than the pilot's 100
failures at d = 9 (experiments.md, R1). Task names:
{decoding}_d{distance}_p{physical_error_probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run r1_shots.py --only strong_alone_d9_p0.003 --shots 5
python -m decsim run r1_shots.py --record --out RESULTS
```

then the jobs and the fold of docs/how-to/run_on_a_slurm_cluster.md.
"""

import decsim
import decsim.experiments.sweeps as sweeps

config = sweeps.load_beside(__file__, "config.py")
DECODINGS = ("weak_alone", "strong_alone")
# d = 9 ran first (2026-10-29); d = 5 and 7 on 2026-10-30
DISTANCES = (5, 7)
PHYSICAL_ERROR_PROBABILITIES = (0.003,)

stop = decsim.CollectionSettings(
    max_shots=config.EVALUATION_SHOTS, first_seed=config.EVALUATION_FIRST_SEED
)
sweep = decsim.Sweep(
    "five_results_r1_shots",
    config.alone,
    grid={
        "decoding": DECODINGS,
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=config.metadata,
)
experiment = sweep.experiment()
