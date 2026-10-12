"""Recipes: ways to change the weak decoder baseline, one task each.

Every task is at d = 3 and p = 0.008 with 1.0 us rounds, and stops at
20 shots. Every decoder is priced by a card, so the ticks are the same
on any host.

- strong_only runs the strong decoder alone, on the host.
- table_threshold, online_threshold, double_window and run_both_at_once
  each change one field of a switching slot that keeps a weak result
  whose complementary gap is at least the paper's 20 dB (Toshio et al.
  2510.25222 Sec. IV) and redoes any other window.
- The grid sweeps the weak syndrome buffer's cycles a read and the
  latency of the path from that buffer to the weak decoder.

Usage
-----

```
decsim run examples/recipes.py
```
"""

import dataclasses
import pathlib

import decsim
import decsim.confidence.complementary as complementary
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph

NAME = "recipes"
DISTANCE = 3
PHYSICAL_ERROR_PROBABILITY = 0.008
# the round period, the paper's tau_gen
ROUND_PERIOD_MICROSECONDS = 1.0
WEAK_DECODE_MICROSECONDS_PER_ROUND = 0.4 * ROUND_PERIOD_MICROSECONDS
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10 * ROUND_PERIOD_MICROSECONDS
THRESHOLD_DECIBELS = 20.0
# A calibration table holds a threshold in decibels for each task
# (2510.25222 Sec. III B); this one holds the paper's 20 dB at this
# task, a stand-in, not a calibration.
THIS_FILE = pathlib.Path(__file__)
THRESHOLD_TABLE = THIS_FILE.with_name("threshold_table.csv")
# the cycles one word's read holds a port: the sky130 byte FIFO's one,
# the record's default, and AFS's four (2001.06598 lines 529-531)
CYCLES_PER_ACCESS = (1, 4)
# the path's latency: none, and half a microsecond, this example's choice
LINK_MICROSECONDS = (0.0, 0.5)
COLLECTION = decsim.CollectionSettings(max_shots=20)

# Toshio's linear decoders (2510.25222 lines 968-971), as
# examples/two_tiers.py prices them: tau_weak_dec is 0.4 tau_gen a round,
# split over the complementary gap's two solves of a window, and
# tau_strong_dec is 10 tau_gen (lines 1109-1114 and 1125-1126).
weak_decoder = decoder_settings.linear_decoder_pool(
    WEAK_DECODE_MICROSECONDS_PER_ROUND,
    clocks.FRIDGE_CLOCK,
    solves_per_window=2,
)
strong_decoder = decoder_settings.linear_decoder_pool(
    STRONG_DECODE_MICROSECONDS_PER_ROUND,
    clocks.ROOM_CLOCK,
    solves_per_window=1,
)
redo_window = strong_window_shapes.RedoWindow.Settings()


def strong_only() -> graph.Machine:
    """The strong decoder alone, in the weak decoder's place.

    Every window goes to the strong decoder, whose hops are one host
    cycle each. The controller still writes the weak store, which no
    decoder reads.
    """
    base = _weak_base()
    store = syndrome_buffer.SyndromeBufferSettings()
    strong_store = graph.Part(machines.STRONG_STORE, store, fabrics.HOST)
    strong_pool = graph.Part(machines.STRONG_POOL, strong_decoder, fabrics.HOST)
    without_weak_pool = base.without_part(machines.WEAK_POOL)
    with_store = without_weak_pool.with_part(strong_store)
    with_pool = with_store.with_part(strong_pool)
    wired = with_pool.with_wires(*machines.STRONG_TIER_WIRES)
    fabric = link_cards.strong_side(base.fabric)
    return dataclasses.replace(wired, fabric=fabric)


def switching(
    threshold: escalation_settings.ThresholdSettings,
    strong_window: escalation_settings.StrongWindowSettings = redo_window,
    run_both_at_once: bool = False,
) -> graph.Machine:
    """The base, switching on the complementary gap to the strong decoder.

    The strong side's hops are one host cycle each.
    """
    base = _weak_base()
    complementary_gap = complementary.ComplementaryGap.Settings()
    switching_slot = escalation_settings.SwitchingSettings(
        confidence=complementary_gap,
        threshold=threshold,
        run_both_at_once=run_both_at_once,
        strong_window=strong_window,
    )
    weak_pool = graph.Part(machines.WEAK_POOL, weak_decoder, fabrics.FRIDGE)
    weak_tier = base.with_part(weak_pool)
    fabric = link_cards.strong_side(base.fabric)
    return machines.with_strong_tier(
        weak_tier, strong_decoder, switching_slot, fabric
    )


def buffer_and_link(
    cycles_per_access: int, link_microseconds: float
) -> graph.Machine:
    """The base with a ported weak store and one path's latency set.

    The store keeps its record's other defaults; the path carries a
    window's rounds to the weak decoder, the wire alone, as the one-cycle
    hop it replaces.
    """
    base = _weak_base()
    weak_syndrome_buffer = ported_syndrome_buffer.PortedSyndromeBufferSettings(
        cycles_per_access=cycles_per_access
    )
    weak_store = graph.Part(
        machines.WEAK_STORE, weak_syndrome_buffer, fabrics.FRIDGE
    )
    link = fabrics.TimedLink(
        link_microseconds, excludes_receiver_processing=True
    )
    fabric = fabrics.PathLinks(
        weak_buffer_to_weak_decoder=link, base=base.fabric
    )
    with_store = base.with_part(weak_store)
    return dataclasses.replace(with_store, fabric=fabric)


def _weak_base() -> graph.Machine:
    return machines.weak_baseline(
        DISTANCE, PHYSICAL_ERROR_PROBABILITY, ROUND_PERIOD_MICROSECONDS
    )


fixed_threshold = threshold_sources.FixedThreshold.Settings(
    threshold_decibels=THRESHOLD_DECIBELS
)
table_threshold = threshold_sources.TableThreshold.Settings(
    table=THRESHOLD_TABLE
)
online_threshold = threshold_sources.OnlineThreshold.Settings(
    threshold_decibels=THRESHOLD_DECIBELS
)
double_window = strong_window_shapes.DoubleWindow.Settings()
machines_by_recipe = {
    "strong_only": strong_only(),
    "table_threshold": switching(table_threshold),
    "online_threshold": switching(online_threshold),
    "double_window": switching(fixed_threshold, strong_window=double_window),
    "run_both_at_once": switching(fixed_threshold, run_both_at_once=True),
}
tasks = []
for name, machine in machines_by_recipe.items():
    task = decsim.Task(name, machine, {"recipe": name})
    tasks.append(task)
cells = decsim.grid(
    cycles_per_access=CYCLES_PER_ACCESS, link_microseconds=LINK_MICROSECONDS
)
for cell in cells:
    machine = buffer_and_link(**cell)
    cycles_per_access = cell["cycles_per_access"]
    link_microseconds = cell["link_microseconds"]
    name = f"read_cycles_{cycles_per_access}_link_{link_microseconds}us"
    task = decsim.Task(name, machine, cell)
    tasks.append(task)

experiment = decsim.Experiment(NAME, tasks, COLLECTION)
