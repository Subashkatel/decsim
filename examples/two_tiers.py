"""Two tiers on priced cards: the whole switching loop, the same on every host.

The weak decoder baseline switching to a strong tier: a weak result is
kept when its complementary gap is at least the paper's 20 dB (Toshio
et al. 2510.25222 Sec. IV), and any other window is redone on the redo
window. Both tiers are Toshio's linear decoder (lines 968-971): a decode
of r rounds costs tau_dec r on the unit, and a round reaches the decoder
T_comm after it is measured, with T_weak_comm = tau_gen and
T_strong_comm = tau_strong_dec = 10 tau_gen (lines 1109-1114).
tau_weak_dec is 0.4 tau_gen, one of the values the paper sweeps (lines
1125-1126), split over the complementary gap's two decodes of a window.
tau_gen is the 1.0 us round period. At d = 3 and 5 and p = 0.008, a
point stops at 50 shots.

Usage
-----

```
decsim run examples/two_tiers.py
```
"""

import dataclasses

import decsim
import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.links.link_profiles as link_profiles
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

NAME = "two_tiers"
DISTANCES = (3, 5)
PHYSICAL_ERROR_PROBABILITY = 0.008
# the round period, the paper's tau_gen
ROUND_PERIOD_MICROSECONDS = 1.0
WEAK_DECODE_MICROSECONDS_PER_ROUND = 0.4 * ROUND_PERIOD_MICROSECONDS
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10 * ROUND_PERIOD_MICROSECONDS
WEAK_COMMUNICATION_MICROSECONDS = ROUND_PERIOD_MICROSECONDS
STRONG_COMMUNICATION_MICROSECONDS = 10 * ROUND_PERIOD_MICROSECONDS
THRESHOLD_DECIBELS = 20.0
COLLECTION = decsim.CollectionSettings(max_shots=50)

# Keep a weak result whose complementary gap is at least 20 dB; redo any
# other window on the redo window, after the weak verdict.
complementary_gap = complementary.ComplementaryGap.Settings()
threshold = threshold_sources.FixedThreshold.Settings(
    threshold_decibels=THRESHOLD_DECIBELS
)
switching = escalation_settings.SwitchingSettings(
    confidence=complementary_gap, threshold=threshold
)
weak_decoder = decoder_settings.linear_decoder_pool(
    WEAK_DECODE_MICROSECONDS_PER_ROUND,
    machine_settings.FRIDGE_CLOCK,
    solves_per_window=2,
)
strong_decoder = decoder_settings.linear_decoder_pool(
    STRONG_DECODE_MICROSECONDS_PER_ROUND,
    machine_settings.ROOM_CLOCK,
    solves_per_window=1,
)

points = []
for distance in DISTANCES:
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        ROUND_PERIOD_MICROSECONDS,
    )
    # T_weak_comm on the hop into the weak store, the strong side's hops
    # one host cycle each, and T_strong_comm on the escalation's hop
    weak_side = link_profiles.with_path_latency(
        base.links, "controller_to_weak_buffer", WEAK_COMMUNICATION_MICROSECONDS
    )
    strong_side = machine_settings.one_cycle_strong_side(weak_side)
    links = link_profiles.with_path_latency(
        strong_side,
        "weak_decoder_to_strong_decoder",
        STRONG_COMMUNICATION_MICROSECONDS,
    )
    windows = window_settings.switching_windows(
        base.windows, switching.strong_window
    )
    machine = dataclasses.replace(
        base,
        links=links,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
    )
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
