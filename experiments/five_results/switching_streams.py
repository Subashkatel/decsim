r"""R3 and R5 on streams: does each switching design keep up?

The switching pilot (switching_pilot.py) decodes shots of 90 rounds,
too short to show whether a design falls behind. Each task here streams
one memory (decsim stream) on one of the pilot's 18 designs: sliding,
speculation or the double window, on 1 or 2 strong cores, at the
thresholds calibrated to escalate 5, 10 and 20% of windows
(calibration.py). The stream's record holds the reaction's growth rate,
the most windows waiting and a series row every K rounds.

A design keeps up only if the work on its slowest path is under one
window's 9.9 us. From the pilot's loads: the sliding window holds each
next window for union-find's 4.5 us (decode and cluster gap) and every
escalated window's 60.3 us; speculation keeps union-find off that path,
but its strong decodes still run one after another; the double window's
strong decodes, 90.45 us each, run side by side on its cores. So the
expected loads on the slowest path at 5, 10 and 20% are about 0.73,
1.1 and 1.5 for sliding; 0.27, 0.65 and 1.01 for speculation; and 0.38,
0.87 and 1.26 for the double window on 1 core, half that on 2.

Needs the double window on a stream (B4, the scratch branch). Task
names: {scheme}_{strong_unit_count}core_{share}_d{distance}_p{probability}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md), one
stream of 100,000 rounds on the evaluation seed 1:

```
export PYTHONPATH=$PWD/../..
python -m decsim stream switching_streams.py \
  --only double_1core_0.2_d9_p0.003 --rounds 100000 --warmup 1000 \
  --seed 1 --set observation.series_every_rounds=1000 --out double.json
```
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
    """One pilot design, streaming its memory."""
    point = (distance, physical_error_probability)
    threshold_decibels = calibration.CALIBRATED[point][share]
    design = config.switching(
        scheme,
        distance,
        physical_error_probability,
        threshold_decibels,
        strong_unit_count,
    )
    return config.streaming(design)


# decsim stream runs one shot; an experiment still names its collection
stop = decsim.CollectionSettings(max_shots=1)
sweep = decsim.Sweep(
    "five_results_switching_streams",
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
