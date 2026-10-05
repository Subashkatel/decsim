"""Recipes: ways to change the weak decoder baseline, one task each.

Every task is at d = 3 and p = 0.008 with 1.0 us rounds, and stops at
20 shots. Every decoder is priced by a card, so the ticks are the same
on any host.

- strong_only fills the strong decoder slot alone.
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
import decsim.links.link_profiles as link_profiles
import decsim.settings as machine_settings
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.windows.settings as window_settings

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
    machine_settings.FRIDGE_CLOCK,
    solves_per_window=2,
)
strong_decoder = decoder_settings.linear_decoder_pool(
    STRONG_DECODE_MICROSECONDS_PER_ROUND,
    machine_settings.ROOM_CLOCK,
    solves_per_window=1,
)
redo_window = strong_window_shapes.RedoWindow.Settings()


def strong_only() -> machine_settings.MachineSettings:
    """The strong slot filled; the weak slot and switching left empty.

    Every window goes to the strong decoder, whose hops are one host
    cycle each.
    """
    base = _weak_base()
    links = machine_settings.one_cycle_strong_side(base.links)
    return dataclasses.replace(
        base, links=links, weak_decoder=None, strong_decoder=strong_decoder
    )


def switching(
    threshold: escalation_settings.ThresholdSettings,
    strong_window: escalation_settings.StrongWindowSettings = redo_window,
    run_both_at_once: bool = False,
) -> machine_settings.MachineSettings:
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
    links = machine_settings.one_cycle_strong_side(base.links)
    windows = window_settings.switching_windows(base.windows, strong_window)
    return dataclasses.replace(
        base,
        links=links,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching_slot,
    )


def buffer_and_link(
    cycles_per_access: int, link_microseconds: float
) -> machine_settings.MachineSettings:
    """The base with a ported weak store and one path's latency set.

    The store keeps its record's other defaults; the path carries a
    window's rounds to the weak decoder.
    """
    base = _weak_base()
    weak_syndrome_buffer = ported_syndrome_buffer.PortedSyndromeBufferSettings(
        cycles_per_access=cycles_per_access
    )
    links = link_profiles.with_path_latency(
        base.links, "weak_buffer_to_weak_decoder", link_microseconds
    )
    return dataclasses.replace(
        base, weak_syndrome_buffer=weak_syndrome_buffer, links=links
    )


def _weak_base() -> machine_settings.MachineSettings:
    return machine_settings.weak_decoder_baseline(
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
machines = {
    "strong_only": strong_only(),
    "table_threshold": switching(table_threshold),
    "online_threshold": switching(online_threshold),
    "double_window": switching(fixed_threshold, strong_window=double_window),
    "run_both_at_once": switching(fixed_threshold, run_both_at_once=True),
}
tasks = []
for name, machine in machines.items():
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
