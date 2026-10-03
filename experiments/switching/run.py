"""Two switching points on the weak decoder baseline, at p = 0.008.

Near threshold, escalations and logical errors are frequent enough to
measure. Each keeps a weak result whose confidence is at least the
paper's 20 dB (Toshio 2510.25222 Sec. IV) and redoes any other window
on the redo window, its near face pinned on its neighbour's committed
correction (Bombin 2303.04846 lines 775-788).

- Cluster gap, at d = 3 and 5: a union-find weak tier whose own growth
  is walked for the confidence signal (Meister et al. 2405.07433
  Algorithm 2), the walk priced as a 12.0 us card on the weak unit that
  produced the evidence (decision D8), so the confidence step is a
  number a reader can point at. The strong tier is Toshio's linear
  decoder (2510.25222 lines 968-971): tau_strong_dec = 10 tau_gen a
  round, its rounds arriving T_strong_comm = 10 tau_gen after the
  escalation sends them (lines 1109-1114), tau_gen the 1.0 us round
  period.
- Redo window, at d = 3 and at d = 5, the paired accuracy study: weak
  PyMatching with the complementary gap, escalating to belief matching,
  both charged their measured wall clock.

Usage
-----

```
decsim run experiments/switching/run.py
```
"""

import dataclasses

import decsim
import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.links.link_profiles as link_profiles
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

NAME = "switching"
DISTANCES = (3, 5)
PHYSICAL_ERROR_PROBABILITY = 0.008
TAU_GEN_MICROSECONDS = 1.0
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10 * TAU_GEN_MICROSECONDS
STRONG_COMMUNICATION_MICROSECONDS = 10 * TAU_GEN_MICROSECONDS
# the keep threshold of 2510.25222 Sec. IV
THRESHOLD_DECIBELS = 20.0
CLUSTER_GAP_COLLECTION = decsim.CollectionSettings(max_shots=50)
REDO_WINDOW_COLLECTION = decsim.CollectionSettings(max_shots=2000)


def cluster_gap_switching(distance: int) -> machine_settings.MachineSettings:
    """Union-find's cluster gap decides; Toshio's strong decoder redoes.

    The union-find decode is charged the host's wall clock.
    """
    name = "cluster_gap_switching"
    base = machine_settings.weak_decoder_baseline(
        distance, PHYSICAL_ERROR_PROBABILITY, TAU_GEN_MICROSECONDS, name=name
    )
    union_find_decoder = union_find.UnionFindDecoder.Settings()
    weak_decoder = dataclasses.replace(
        base.weak_decoder, algorithm=union_find_decoder
    )
    strong_decoder = decoder_settings.toshio_decoder_pool(
        STRONG_DECODE_MICROSECONDS_PER_ROUND,
        machine_settings.ROOM_CLOCK,
        solves_per_window=1,
    )
    cluster_gap = cluster.ClusterGap.Settings(walk_microseconds=12.0)
    machine = _switching(base, cluster_gap, weak_decoder, strong_decoder, name)
    links = link_profiles.with_path_latency(
        machine.links,
        "weak_decoder_to_strong_decoder",
        STRONG_COMMUNICATION_MICROSECONDS,
    )
    return dataclasses.replace(machine, links=links)


def redo_window_switching(distance: int) -> machine_settings.MachineSettings:
    """PyMatching's complementary gap decides; belief matching redoes."""
    name = "redo_window_switching"
    base = machine_settings.weak_decoder_baseline(
        distance, PHYSICAL_ERROR_PROBABILITY, TAU_GEN_MICROSECONDS, name=name
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=matching)
    belief_matching_decoder = belief_matching.BeliefMatchingDecoder.Settings()
    # the baselines' engine card, counted on the host's clock
    strong_engine = dataclasses.replace(
        decoder_settings.ESTIMATED_ENGINE, clock=machine_settings.ROOM_CLOCK
    )
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=belief_matching_decoder, engine=strong_engine
    )
    complementary_gap = complementary.ComplementaryGap.Settings()
    return _switching(
        base, complementary_gap, weak_decoder, strong_decoder, name
    )


def switching_points() -> list:
    """The cluster-gap points, then the redo-window points, by distance."""
    points = []
    for distance in DISTANCES:
        machine = cluster_gap_switching(distance)
        metadata = _metadata(distance)
        point = decsim.Point(
            f"cluster_gap_d{distance}",
            machine,
            metadata,
            CLUSTER_GAP_COLLECTION,
        )
        points.append(point)
    for distance in DISTANCES:
        machine = redo_window_switching(distance)
        metadata = _metadata(distance)
        point = decsim.Point(
            f"redo_window_d{distance}",
            machine,
            metadata,
            REDO_WINDOW_COLLECTION,
        )
        points.append(point)
    return points


def _switching(
    base: machine_settings.MachineSettings,
    confidence: escalation_settings.ConfidenceSettings,
    weak_decoder: decoder_settings.DecoderPoolSettings,
    strong_decoder: decoder_settings.DecoderPoolSettings,
    name: str,
) -> machine_settings.MachineSettings:
    """The base switching on confidence to a strong decoder on the host.

    A weak result is kept at 20 dB or more; any other window is redone
    on the redo window, the strong decode starting after the weak
    verdict, which is free. The strong side's hops are one room cycle
    each. name labels the card, as the yaml reader labels it.
    """
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=THRESHOLD_DECIBELS
    )
    switching_slot = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    links = machine_settings.one_cycle_strong_side(base.links, name)
    windows = window_settings.switching_windows(
        base.windows, switching_slot.strong_window
    )
    return dataclasses.replace(
        base,
        links=links,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching_slot,
    )


def _metadata(distance: int) -> dict:
    """The point's swept cells as the yaml named them, so its id is theirs."""
    return {
        "workload.arguments.physical_error_probability": (
            PHYSICAL_ERROR_PROBABILITY
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": TAU_GEN_MICROSECONDS,
    }


points = switching_points()
experiment = decsim.Experiment(NAME, points)
