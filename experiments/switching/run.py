"""Two switching tasks on the weak decoder baseline, at p = 0.008.

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

import decsim
import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.ports as ports
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

NAME = "switching"
DISTANCES = (3, 5)
PHYSICAL_ERROR_PROBABILITY = 0.008
# the round period, the paper's tau_gen
ROUND_PERIOD_MICROSECONDS = 1.0
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10 * ROUND_PERIOD_MICROSECONDS
# T_strong_comm = 10 tau_gen, 10 us: 2500 cycles of the 250 MHz room clock
ESCALATION_LINK = fabrics.ClockedLink(
    clock=clocks.ROOM_CLOCK,
    latency_cycles=2500,
    source="T_strong_comm, 2510.25222 lines 1109-1114",
)
# the keep threshold of 2510.25222 Sec. IV
THRESHOLD_DECIBELS = 20.0
CLUSTER_GAP_COLLECTION = decsim.CollectionSettings(max_shots=50)
REDO_WINDOW_COLLECTION = decsim.CollectionSettings(max_shots=2000)


def cluster_gap_switching(distance: int) -> graph.Machine:
    """Union-find's cluster gap decides; Toshio's strong decoder redoes.

    The union-find decode is charged the host's measured time, and the
    escalation's rounds arrive T_strong_comm after it sends them.
    """
    base = machines.weak_baseline(
        distance, PHYSICAL_ERROR_PROBABILITY, ROUND_PERIOD_MICROSECONDS
    )
    host_time = cycle_count.HostMeasuredTime()
    union_find_decoder = union_find.UnionFindDecoder.Settings(timing=host_time)
    weak_base = base.with_settings(
        machines.WEAK_POOL, algorithm=union_find_decoder
    )
    strong_decoder = decoder_settings.linear_decoder_pool(
        STRONG_DECODE_MICROSECONDS_PER_ROUND,
        clocks.ROOM_CLOCK,
        solves_per_window=1,
    )
    cluster_gap = cluster.ClusterGap.Settings(walk_microseconds=12.0)
    return _switching(weak_base, cluster_gap, strong_decoder, ESCALATION_LINK)


def redo_window_switching(distance: int) -> graph.Machine:
    """PyMatching's complementary gap decides; belief matching redoes.

    The strong tier is the strong base's.
    """
    base = machines.weak_baseline(
        distance, PHYSICAL_ERROR_PROBABILITY, ROUND_PERIOD_MICROSECONDS
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_base = base.with_settings(machines.WEAK_POOL, algorithm=matching)
    strong_decoder = machines.STRONG_BASELINE_POOL
    complementary_gap = complementary.ComplementaryGap.Settings()
    return _switching(weak_base, complementary_gap, strong_decoder)


def switching_tasks() -> list:
    """The cluster-gap tasks, then the redo-window tasks, by distance."""
    tasks = []
    for distance in DISTANCES:
        machine = cluster_gap_switching(distance)
        metadata = _metadata(distance)
        task = decsim.Task(
            f"cluster_gap_d{distance}",
            machine,
            metadata,
            CLUSTER_GAP_COLLECTION,
        )
        tasks.append(task)
    for distance in DISTANCES:
        machine = redo_window_switching(distance)
        metadata = _metadata(distance)
        task = decsim.Task(
            f"redo_window_d{distance}",
            machine,
            metadata,
            REDO_WINDOW_COLLECTION,
        )
        tasks.append(task)
    return tasks


def _switching(
    base: graph.Machine,
    confidence: ports.ConfidenceSettings,
    strong_decoder: decoder_settings.DecoderPoolSettings,
    escalation: ports.LinkCard = link_cards.ONE_ROOM_CYCLE,
) -> graph.Machine:
    """The base switching on confidence to a strong decoder on the host.

    A weak result is kept at 20 dB or more; any other window is redone
    on the redo window, the strong decode starting after the weak
    verdict, which is free. The strong side's hops are one room cycle
    each, the escalation as given.
    """
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=THRESHOLD_DECIBELS
    )
    switching_slot = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    fabric = link_cards.strong_side(base.fabric, escalation)
    return machines.with_strong_tier(
        base, strong_decoder, switching_slot, fabric
    )


def _metadata(distance: int) -> dict:
    """The swept cells under the earlier runs' names, so the ids match."""
    return {
        "workload.arguments.physical_error_probability": (
            PHYSICAL_ERROR_PROBABILITY
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": ROUND_PERIOD_MICROSECONDS,
    }


tasks = switching_tasks()
experiment = decsim.Experiment(NAME, tasks)
