r"""The switching grid on streams at d = 5, 7 and 9: which designs keep up?

Each task streams one memory (decsim stream) on one of
switching_grid.py's designs at one escalated share. The stream's record
holds the reaction's growth rate, the most windows waiting and a series
row every K rounds. A design keeps up when its reaction does not grow.

Usage
-----

```
export PYTHONPATH=$PWD/../..
python -m decsim stream switching_grid_streams.py \
  --only double_1core_0.2_d7_p0.003 --rounds 10000 --warmup 1000 \
  --seed 1 --set observation.series_every_rounds=500 --out double.json
```
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
designs = sweeps.load_beside(__file__, "switching_designs.py")
# a distance streams once calibration.py holds its thresholds, so d = 9
# can run while d = 5 and 7 are still being calibrated
DISTANCES = tuple(
    distance
    for distance in (5, 7, 9)
    if (distance, 0.003) in designs.calibration.CALIBRATED
)


def machine(
    design: str,
    share: float,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """One grid design, streaming its memory."""
    design_machine = designs.machine(
        design, share, distance, physical_error_probability
    )
    return config.streaming(design_machine)


# decsim stream runs one shot; an experiment still names its collection
stop = decsim.CollectionSettings(max_shots=1)
sweep = decsim.Sweep(
    "five_results_switching_grid_streams",
    machine,
    grid={
        "design": tuple(designs.DESIGNS),
        "share": designs.SHARES,
        "distance": DISTANCES,
        "physical_error_probability": designs.PHYSICAL_ERROR_PROBABILITIES,
    },
    task_name=designs.TASK_NAME,
    stop=stop,
    metadata=designs.metadata,
)
experiment = sweep.experiment()
