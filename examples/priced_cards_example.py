"""Priced cards: the decoder costs a stated number, not a measured one.

The weak decoder baseline with its decoder priced at 1.0 us a window:
PyMatching still answers every window, but the time it is charged is
the card's, so the timing is the same on every host. At d = 3, 5 and 7
and p = 0.001, a point stops at 20 shots.

Usage
-----

```
decsim run examples/priced_cards_example.py
```
"""

import dataclasses

import decsim
import decsim.settings as machine_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

NAME = "priced_cards_example"
DISTANCES = (3, 5, 7)
PHYSICAL_ERROR_PROBABILITY = 0.001
ROUND_PERIOD_MICROSECONDS = 1.0
DECODE_MICROSECONDS = 1.0
COLLECTION = decsim.CollectionSettings(max_shots=20)

points = []
for distance in DISTANCES:
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        ROUND_PERIOD_MICROSECONDS,
    )
    card = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=DECODE_MICROSECONDS
    )
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=card)
    machine = dataclasses.replace(base, weak_decoder=weak_decoder)
    # the point's settings, named as the results' columns name them
    metadata = {
        "workload.arguments.physical_error_probability": (
            PHYSICAL_ERROR_PROBABILITY
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": ROUND_PERIOD_MICROSECONDS,
    }
    point = decsim.Point(f"d{distance}", machine, metadata)
    points.append(point)

experiment = decsim.Experiment(NAME, points, COLLECTION)
