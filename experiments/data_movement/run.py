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
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.links.link_profiles as link_profiles
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

NAME = "data_movement"
DISTANCES = (3, 5, 7)
PHYSICAL_ERROR_PROBABILITY = 0.003
TAU_GEN_MICROSECONDS = 1.0
WEAK_DECODE_MICROSECONDS_PER_ROUND = 0.4 * TAU_GEN_MICROSECONDS
STRONG_DECODE_MICROSECONDS_PER_ROUND = 10 * TAU_GEN_MICROSECONDS
WEAK_COMMUNICATION_MICROSECONDS = TAU_GEN_MICROSECONDS
STRONG_COMMUNICATION_MICROSECONDS = 10 * TAU_GEN_MICROSECONDS
# the keep threshold of 2510.25222 Sec. IV
THRESHOLD_DECIBELS = 20.0
COLLECTION = decsim.CollectionSettings(max_shots=100)
# Every block's swept cells as data_movement.yaml wrote them, which a
# point's metadata holds so that its id is the yaml point's.
EVERY_BLOCK_CELLS = {
    "links.controller_to_weak_buffer": {
        "latency_cycles": 250,
        "clock": "fridge",
        "bits_per_cycle": None,
    },
    "workload.arguments.physical_error_probability": PHYSICAL_ERROR_PROBABILITY,
    "qpu.round_period_microseconds": TAU_GEN_MICROSECONDS,
}
ONE_ROOM_CYCLE_CARD = {
    "latency_cycles": 1,
    "clock": "room",
    "bits_per_cycle": None,
}
SWITCHING_CELLS = {
    "weak_decoder.engine.fetch_cycles_per_round": 50,
    "escalation": {
        "kind": "switching",
        "gap_threshold_db": THRESHOLD_DECIBELS,
        "strong_window": "redo_window",
    },
    "strong_decoder": {
        "kind": 0.0,
        "units": 1,
        "input": "copy",
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "room",
            "fetch_cycles_per_round": 2500,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 0,
            "release_cycles_per_round": 0,
        },
    },
    "links.controller_to_strong_buffer": ONE_ROOM_CYCLE_CARD,
    "links.weak_decoder_to_strong_decoder": {
        "latency_cycles": 2500,
        "clock": "room",
        "bits_per_cycle": None,
    },
    "links.strong_buffer_to_strong_decoder": ONE_ROOM_CYCLE_CARD,
    "links.strong_decoder_to_frame": ONE_ROOM_CYCLE_CARD,
}


def every_hop_copies(distance: int) -> machine_settings.MachineSettings:
    """Block 1: the weak tier on copies, every copy counted and traced.

    One traced shot a point feeds residence.csv.
    """
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        TAU_GEN_MICROSECONDS,
        name=NAME,
    )
    links = link_profiles.with_path_latency(
        base.links, "controller_to_weak_buffer", WEAK_COMMUNICATION_MICROSECONDS
    )
    weak_decoder = decoder_settings.toshio_decoder_pool(
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
    weak_decoder = decoder_settings.toshio_decoder_pool(
        WEAK_DECODE_MICROSECONDS_PER_ROUND,
        machine_settings.FRIDGE_CLOCK,
        solves_per_window=2,
    )
    strong_decoder = decoder_settings.toshio_decoder_pool(
        STRONG_DECODE_MICROSECONDS_PER_ROUND,
        machine_settings.ROOM_CLOCK,
        solves_per_window=1,
    )
    strong_side = machine_settings.one_cycle_strong_side(machine.links, NAME)
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


# Each block: its name, its machine, and the cells it sweeps.
BLOCKS = (
    ("every_hop_copies", every_hop_copies, {}),
    (
        "weak_input_in_place",
        weak_input_in_place,
        {"weak_decoder.input": "in_place"},
    ),
    (
        "boundary_folded_in_place",
        boundary_folded_in_place,
        {"weak_decoder.boundary_fold": "in_place"},
    ),
    ("switching", switching, SWITCHING_CELLS),
)


def data_movement_points() -> list:
    """Every block at every distance, blocks in order."""
    points = []
    for block_name, machine_at, cells in BLOCKS:
        for distance in DISTANCES:
            machine = machine_at(distance)
            metadata = {
                **EVERY_BLOCK_CELLS,
                **cells,
                "qpu.distance": distance,
            }
            point_name = f"{block_name}_d{distance}"
            point = decsim.Point(point_name, machine, metadata)
            points.append(point)
    return points


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


points = data_movement_points()
experiment = decsim.Experiment(NAME, points, COLLECTION)
