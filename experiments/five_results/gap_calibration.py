r"""The gap calibration: every window's cluster gap, on union-find alone.

The run decodes the calibration seeds, 0 to 9,999, of the point's saved
dataset, with the gap a switching run reads recorded on every window
(config.recording_gap); its confidence_histogram.csv counts them, and
calibrate.py reads from it the threshold that escalates each share of
windows (experiments.md: the threshold is set to an escalated share, as
Toshio et al. pick theirs from the gap's distribution, 2510.25222 Fig.
17). The evaluation runs start at seed 10,000, so no shot is both. A
switching run's windows after an escalated one read the strong
boundary, so the share it escalates is near this, not equal; the
switching runs measure their own. Task names:
{decoding}_d{distance}_p{physical_error_probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim run gap_calibration.py --record --out RESULTS
```

then the jobs and the fold of docs/how-to/run_on_a_slurm_cluster.md, and
`python calibrate.py RESULTS`.
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
DISTANCES = (9,)
PHYSICAL_ERROR_PROBABILITIES = (0.003,)


def machine(distance: int, physical_error_probability: float) -> graph.Machine:
    """Union-find alone at the point, its gap recorded."""
    weak = config.alone("weak_alone", distance, physical_error_probability)
    return config.recording_gap(weak)


stop = decsim.CollectionSettings(max_shots=config.CALIBRATION_SHOTS)
sweep = decsim.Sweep(
    "five_results_gap_calibration",
    machine,
    grid={
        "distance": DISTANCES,
        "physical_error_probability": PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name="weak_alone_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=config.metadata,
)
experiment = sweep.experiment()
