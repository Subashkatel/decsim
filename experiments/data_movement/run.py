"""The data-movement study: the copies, references and moves of the data path.

Four blocks on the weak decoder baseline, each apart from the first in
what it studies alone, at d = 3, 5 and 7 and p = 0.003:

1. Every hop copies, the research note's defaults: the weak unit is
   given a copy of the rounds it decodes (AFS 2001.06598 lines 520-535,
   Collision Clustering 2309.05558 lines 268-272) and XORs the window's
   boundary into a duplicate of that copy (cuda-q QEC's
   sliding_window.cpp:287-293).
2. The weak input read in place, where the weak syndrome buffer keeps
   it: the software decoder on the controller host, which reads the
   detections as they stream into shared memory (2408.13687 lines
   476-477, 487-489).
3. The boundary folded in place, into the unit's own copy: the hardware
   decoder, which writes its on-chip storage directly (2001.06598 lines
   528-531, Helios 2301.08419 lines 632-640).
4. The strong tier and its links: switching at the paper's 20 dB, which
   opens the escalation hop and the strong syndrome buffer's write and
   read (Toshio 2510.25222 lines 605-622 and 1248-1250).

Both tiers are Toshio et al.'s linear decoder on priced cards, so the
counts are the same on any host: a decode of r rounds costs tau_dec r on
the unit (2510.25222 lines 968-971), tau_weak_dec is 0.4 tau_gen (lines
1125-1126), which block 4's complementary gap splits over its two
solves, and tau_strong_dec is 10 tau_gen. A round reaches the weak
decoder T_weak_comm = tau_gen after it is measured and the strong one
T_strong_comm = 10 tau_gen after the escalation sends it (lines
1109-1114). tau_gen is the 1.0 us round period.

Usage
-----

```
decsim run experiments/data_movement/run.py
```
"""

import dataclasses

import decsim
import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

NAME = "data_movement"
DISTANCES = (3, 5, 7)
PHYSICAL_ERROR_PROBABILITY = 0.003
# the round period, the paper's tau_gen
ROUND_PERIOD_MICROSECONDS = 1.0
WEAK_DECODE_MICROSECONDS_PER_ROUND = 0.4 * ROUND_PERIOD_MICROSECONDS
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10 * ROUND_PERIOD_MICROSECONDS
WEAK_COMMUNICATION_MICROSECONDS = ROUND_PERIOD_MICROSECONDS
STRONG_COMMUNICATION_MICROSECONDS = 10 * ROUND_PERIOD_MICROSECONDS
# the keep threshold of 2510.25222 Sec. IV
THRESHOLD_DECIBELS = 20.0
COLLECTION = decsim.CollectionSettings(max_shots=100)
# The word a results column shows for whether a pool copies its input,
# and its boundary fold.
INPUT_WORDS = {True: "copy", False: "in_place"}
BOUNDARY_FOLD_WORDS = {True: "copy", False: "in_place"}
# Blocks 1 to 3 run no strong tier. The weak base they extend escalates
# nothing, names three strong-side hops null, which keeps the reference
# card, and writes no fourth hop and no strong decoder; the results show
# those cells as written.
NO_STRONG_SIDE_CELLS = {
    "escalation": {"kind": "weak_baseline"},
    "strong_decoder": "",
    "links.controller_to_strong_buffer": None,
    "links.weak_decoder_to_strong_decoder": None,
    "links.strong_buffer_to_strong_decoder": None,
    "links.strong_decoder_to_frame": "",
}


def every_hop_copies(distance: int) -> machine_settings.MachineSettings:
    """Block 1: the weak tier on copies, every copy counted and traced.

    One traced shot a point records where every round sat.
    """
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        ROUND_PERIOD_MICROSECONDS,
    )
    links = link_profiles.with_path_latency(
        base.links, "controller_to_weak_buffer", WEAK_COMMUNICATION_MICROSECONDS
    )
    weak_decoder = decoder_settings.linear_decoder_pool(
        WEAK_DECODE_MICROSECONDS_PER_ROUND,
        machine_settings.FRIDGE_CLOCK,
        solves_per_window=1,
    )
    observation = dataclasses.replace(
        base.observation, trace="chrome", data_movement=True
    )
    return dataclasses.replace(
        base, links=links, weak_decoder=weak_decoder, observation=observation
    )


def weak_input_in_place(distance: int) -> machine_settings.MachineSettings:
    """Block 2: the weak unit reads its rounds where the store keeps them."""
    machine = every_hop_copies(distance)
    weak_decoder = dataclasses.replace(machine.weak_decoder, copies_input=False)
    return dataclasses.replace(machine, weak_decoder=weak_decoder)


def boundary_folded_in_place(
    distance: int,
) -> machine_settings.MachineSettings:
    """Block 3: the boundary is XORed into the unit's own copy."""
    machine = every_hop_copies(distance)
    weak_decoder = dataclasses.replace(
        machine.weak_decoder, copies_boundary_fold=False
    )
    return dataclasses.replace(machine, weak_decoder=weak_decoder)


def switching(distance: int) -> machine_settings.MachineSettings:
    """Block 4: weak first, escalating below 20 dB to the strong tier.

    The complementary gap decodes each weak window twice.
    """
    machine = every_hop_copies(distance)
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
    strong_side = machine_settings.one_cycle_strong_side(machine.links)
    links = link_profiles.with_path_latency(
        strong_side,
        "weak_decoder_to_strong_decoder",
        STRONG_COMMUNICATION_MICROSECONDS,
    )
    switching_slot = _complementary_gap_switching()
    windows = window_settings.switching_windows(
        machine.windows, switching_slot.strong_window
    )
    return dataclasses.replace(
        machine,
        links=links,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching_slot,
    )


# Each block: its name and its machine.
BLOCKS = (
    ("every_hop_copies", every_hop_copies),
    ("weak_input_in_place", weak_input_in_place),
    ("boundary_folded_in_place", boundary_folded_in_place),
    ("switching", switching),
)


def data_movement_points() -> list:
    """Every block at every distance, blocks in order."""
    points = []
    for block_name, machine_at in BLOCKS:
        for distance in DISTANCES:
            machine = machine_at(distance)
            metadata = reported_cells(machine, distance)
            point_name = f"{block_name}_d{distance}"
            point = decsim.Point(point_name, machine, metadata)
            points.append(point)
    return points


def reported_cells(
    machine: machine_settings.MachineSettings, distance: int
) -> dict:
    """Every cell the study's results show for a point, in column order.

    Each column is a setting some block changes, so every point reports
    its value there, read off the machine it runs.
    """
    weak_decoder = machine.weak_decoder
    weak_link = _link_cell(
        machine.links.controller_to_weak_buffer,
        "fridge",
        machine_settings.FRIDGE_CLOCK,
    )
    input_word = INPUT_WORDS[weak_decoder.copies_input]
    fold_word = BOUNDARY_FOLD_WORDS[weak_decoder.copies_boundary_fold]
    weak_fetch_cycles = weak_decoder.engine.fetch_cycles_per_round
    strong_side = NO_STRONG_SIDE_CELLS
    if machine.switching is not None:
        strong_side = _strong_side_cells(machine)
    return {
        "links.controller_to_weak_buffer": weak_link,
        "workload.arguments.physical_error_probability": (
            PHYSICAL_ERROR_PROBABILITY
        ),
        "qpu.distance": distance,
        "qpu.round_period_microseconds": ROUND_PERIOD_MICROSECONDS,
        "weak_decoder.input": input_word,
        "weak_decoder.boundary_fold": fold_word,
        "weak_decoder.engine.fetch_cycles_per_round": weak_fetch_cycles,
        **strong_side,
    }


def _complementary_gap_switching() -> escalation_settings.SwitchingSettings:
    """Keep a weak result whose complementary gap is at least 20 dB.

    The verdict is free, and an escalated window is redone on the redo
    window, the strong decode starting after the weak verdict.
    """
    complementary_gap = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=THRESHOLD_DECIBELS
    )
    return escalation_settings.SwitchingSettings(
        confidence=complementary_gap, threshold=threshold
    )


def _strong_side_cells(machine: machine_settings.MachineSettings) -> dict:
    """Block 4's escalation, strong pool and strong-side hops, on the host."""
    links = machine.links
    room = machine_settings.ROOM_CLOCK
    escalation = {
        "kind": "switching",
        "gap_threshold_db": THRESHOLD_DECIBELS,
        "strong_window": "redo_window",
    }
    strong_decoder = _pool_cell(machine.strong_decoder, "room")
    write = _link_cell(links.controller_to_strong_buffer, "room", room)
    escalated = _link_cell(links.weak_decoder_to_strong_decoder, "room", room)
    read = _link_cell(links.strong_buffer_to_strong_decoder, "room", room)
    result = _link_cell(links.strong_decoder_to_frame, "room", room)
    return {
        "escalation": escalation,
        "strong_decoder": strong_decoder,
        "links.controller_to_strong_buffer": write,
        "links.weak_decoder_to_strong_decoder": escalated,
        "links.strong_buffer_to_strong_decoder": read,
        "links.strong_decoder_to_frame": result,
    }


def _link_cell(
    path: link_settings.PathSettings, clock_name: str, clock: config.Clock
) -> dict:
    """A path's card in the cycles of its clock, on an unbounded wire."""
    latency_ticks = path.channel.propagation_latency_ticks
    latency_cycles = latency_ticks // clock.period_ticks
    return {
        "latency_cycles": latency_cycles,
        "clock": clock_name,
        "bits_per_cycle": None,
    }


def _pool_cell(
    pool: decoder_settings.DecoderPoolSettings, clock_name: str
) -> dict:
    """A priced matching pool: its card, units, input, memory and engine."""
    engine = pool.engine
    unit_memory = {"bits": pool.unit_memory.bits}
    input_word = INPUT_WORDS[pool.copies_input]
    engine_cell = {
        "clock": clock_name,
        "fetch_cycles_per_round": engine.fetch_cycles_per_round,
        "fetch_cycles_per_job": engine.fetch_cycles_per_job,
        "release_cycles_per_job": engine.release_cycles_per_job,
        "release_cycles_per_round": engine.release_cycles_per_round,
    }
    # a priced matching card's kind is its latency in microseconds
    return {
        "kind": pool.algorithm.preset_latency_microseconds,
        "units": pool.unit_count,
        "input": input_word,
        "unit_memory": unit_memory,
        "engine": engine_cell,
    }


points = data_movement_points()
experiment = decsim.Experiment(NAME, points, COLLECTION)
