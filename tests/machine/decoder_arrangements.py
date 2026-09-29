"""Sixteen decoder arrangements on one machine, each a runnable yaml.

Each of pymatching, union find, Relay-BP and BP-OSD as the weak tier
alone and as the strong tier alone (belief matching too), and switching
from a weak tier that reports a confidence to each strong tier: the
arrangements of the decoder experiments that ran in September 2026,
at their distance-3 point, p = 0.001 and one microsecond rounds. The
arrangements differ only in their escalation section and decoder rows,
so a check that holds on one machine holds on every decoder the tree
ships.
"""

import pathlib

import yaml


def link(clock: str) -> dict:
    """One hop's card: one cycle of latency on its clock, width unpriced."""
    return {"latency_cycles": 1, "clock": clock, "bits_per_cycle": None}


# Every section but the escalation and the decoder rows. One window plan
# for all sixteen: switching needs the lookahead tail (Toshio 2510.25222
# Sec. III C, a strong recovery reads past the commit) and the single
# tier arrangements take the same plan so the comparison is fair.
MACHINE = {
    "qpu": {"kind": "stim_device"},
    "workload": {
        "kind": "producer",
        "function": "decsim.producers:memory_circuit",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": "10d",
            "distance": "${qpu.distance}",
        },
    },
    "windows": {
        "kind": "sliding",
        "commit_rounds": None,
        "buffer_rounds": None,
        "terminal_policy": "lookahead",
    },
    "sweep": [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.001],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": 8},  # the eight shots a test runs
        }
    ],
    "controller": {
        "clock": "fridge",
        "readout_to_bits_cycles": 0,
        "decision_to_pulse_cycles": 0,
        "packing_cycles_per_round": 0,
        "packing_rounds_in_flight": None,
    },
    "clocks": {"fridge": 250.0, "room": 250.0},
    "links": {
        "qpu_to_controller": link("fridge"),
        "controller_to_weak_buffer": link("fridge"),
        "controller_to_strong_buffer": link("room"),
        "weak_buffer_to_weak_decoder": link("fridge"),
        "weak_decoder_to_strong_decoder": link("room"),
        "strong_buffer_to_strong_decoder": link("room"),
        "decoder_to_decoder": link("fridge"),
        "weak_decoder_to_frame": link("fridge"),
        "strong_decoder_to_frame": link("room"),
        "frame_to_controller": link("fridge"),
        "controller_to_qpu": link("fridge"),
    },
    "weak_syndrome_buffer": {"bits": None},
    "strong_syndrome_buffer": {"bits": None},
    "pauli_frame": {"clock": "fridge", "write_cycles": 1},
    "idle_policy": {"kind": "separate_decode_jobs"},
    "observation": {
        "check_windows_with": "none",
        "log_component_io": False,
        "log": "off",
        "trace": "off",
    },
}
WEAK_ONLY = {"kind": "weak_baseline"}
STRONG_ONLY = {"kind": "strong_only"}
# the paper's 20 dB threshold (Toshio 2510.25222 Sec. IV), the escalated
# window's near face pinned on its neighbour's committed correction
COMPLEMENTARY_GAP_SWITCHING = {
    "kind": "switching",
    "confidence": "complementary_gap",
    "gap_threshold_db": 20.0,
    "strong_window": "redo_window",
}
CLUSTER_GAP_SWITCHING = {
    "kind": "switching",
    "confidence": "cluster_gap",
    "gap_threshold_db": 20.0,
    "strong_window": "redo_window",
}


def decoder_unit(kind: str, clock: str) -> dict:
    """One tier's decoder row: one unit, its engine on the tier's clock."""
    return {
        "kind": kind,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": clock,
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 10,
            "release_cycles_per_round": 0,
        },
    }


def weak_only(kind: str) -> dict:
    """The weak tier alone, decoding every window in the fridge."""
    weak_decoder = decoder_unit(kind, "fridge")
    return {"escalation": WEAK_ONLY, "weak_decoder": weak_decoder}


def strong_only(kind: str) -> dict:
    """The strong tier alone, decoding every window at room temperature."""
    strong_decoder = decoder_unit(kind, "room")
    return {"escalation": STRONG_ONLY, "strong_decoder": strong_decoder}


def switching(escalation: dict, weak_kind: str, strong_kind: str) -> dict:
    """A weak tier escalating to a strong tier below the threshold."""
    weak_decoder = decoder_unit(weak_kind, "fridge")
    strong_decoder = decoder_unit(strong_kind, "room")
    return {
        "escalation": escalation,
        "weak_decoder": weak_decoder,
        "strong_decoder": strong_decoder,
    }


ARRANGEMENTS = {
    "pymatching_weak": weak_only("pymatching"),
    "union_find_weak": weak_only("union_find"),
    "relay_bp_weak": weak_only("relay_bp"),
    "bposd_weak": weak_only("bposd"),
    "pymatching_strong": strong_only("pymatching"),
    "union_find_strong": strong_only("union_find"),
    "belief_matching_strong": strong_only("belief_matching"),
    "relay_bp_strong": strong_only("relay_bp"),
    "bposd_strong": strong_only("bposd"),
    "pymatching_belief_matching_switching": switching(
        COMPLEMENTARY_GAP_SWITCHING, "pymatching", "belief_matching"
    ),
    "pymatching_relay_bp_switching": switching(
        COMPLEMENTARY_GAP_SWITCHING, "pymatching", "relay_bp"
    ),
    "pymatching_bposd_switching": switching(
        COMPLEMENTARY_GAP_SWITCHING, "pymatching", "bposd"
    ),
    "union_find_pymatching_switching": switching(
        CLUSTER_GAP_SWITCHING, "union_find", "pymatching"
    ),
    "union_find_belief_matching_switching": switching(
        CLUSTER_GAP_SWITCHING, "union_find", "belief_matching"
    ),
    "union_find_relay_bp_switching": switching(
        CLUSTER_GAP_SWITCHING, "union_find", "relay_bp"
    ),
    "union_find_bposd_switching": switching(
        CLUSTER_GAP_SWITCHING, "union_find", "bposd"
    ),
}


def write_arrangement(folder: pathlib.Path, name: str) -> pathlib.Path:
    """One arrangement as a yaml in folder, named for the arrangement."""
    raw = dict(MACHINE)
    raw.update(ARRANGEMENTS[name])
    config_path = folder / f"{name}.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    return config_path
