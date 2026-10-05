"""My first sweep: three distances at one physical error rate.

The weak decoder baseline with real PyMatching charged its measured
wall clock, so every window is decoded for real and the logical error
rate means something, at d = 3, 5 and 7 and p = 0.003. A point stops at
400 shots.

Usage
-----

```
decsim run examples/my_first_sweep.py
```
"""

import dataclasses

import decsim
import decsim.settings as machine_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

NAME = "my_first_sweep"
DISTANCES = (3, 5, 7)
PHYSICAL_ERROR_PROBABILITY = 0.003
ROUND_PERIOD_MICROSECONDS = 1.0
COLLECTION = decsim.CollectionSettings(max_shots=400)

points = []
for distance in DISTANCES:
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        ROUND_PERIOD_MICROSECONDS,
    )
    # the baseline charges its decoder a number; this sweep decodes for real
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=matching)
    machine = dataclasses.replace(base, weak_decoder=weak_decoder)
    # the point's settings, named as the results' columns name them
    metadata = {
        "workload.arguments.physical_error_probability": (
            PHYSICAL_ERROR_PROBABILITY
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": ROUND_PERIOD_MICROSECONDS,
    }
    point = decsim.Task(f"d{distance}", machine, metadata)
    points.append(point)

experiment = decsim.Experiment(NAME, points, COLLECTION)
