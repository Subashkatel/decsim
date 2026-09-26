"""A complete runnable yaml, small enough for a functional test.

The shape is a design rule: the algorithm is structure
on a per-tier unit card (weak_decoder / strong_decoder), never a sweep
axis, mirroring the tiered architecture itself (Toshio arXiv 2510.25222:
lightweight decoders decode constantly, a separate accurate decoder is
invoked on demand) and gem5's config split (structure on the component,
parameters swept around it).
"""

import pathlib

import yaml

import decsim.collect as collect
import decsim.experiments.measure as measure

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _TEST_FILE.parents[2]
CONFIGS_DIR = _REPOSITORY_ROOT / "configs"
# The yaml experiments this repository ships. The tests that check every
# shipped config walk this tuple and not the folder, so a config a user
# writes into configs/ of their own checkout fails none of them.
SHIPPED_CONFIGS = (
    "cluster_gap_switching.yaml",
    "data_movement.yaml",
    "data_movement_fold_in_place.yaml",
    "data_movement_input_in_place.yaml",
    "data_movement_switching.yaml",
    "my_first_sweep.yaml",
    "priced_cards_example.yaml",
    "reference.yaml",
    "seam_pinned_switching.yaml",
    "strong_decoder_baseline.yaml",
    "strong_latency.yaml",
    "strong_latency_preview.yaml",
    "strong_ler.yaml",
    "two_tiers.yaml",
    "weak_decoder_baseline.yaml",
    "weak_latency.yaml",
    "weak_ler.yaml",
)


def measure_point_shot(
    config,
    *,
    physical_error_probability,
    distance,
    round_period_microseconds,
    seed,
):
    """One seeded shot at one sweep point, collected and measured."""
    shots = seed + 1
    task = config.point_task(
        physical_error_probability=physical_error_probability,
        distance=distance,
        round_period_microseconds=round_period_microseconds,
        shots=shots,
    )
    shot = collect.run_shot(task, seed)
    return measure.measure_shot(shot)


# A complete runnable config, small enough for a functional test. Tests
# override keys through the `overrides` dict (top-level replacement, the
# same rule as `extends`).
MINIMAL_CONFIG = {
    "qpu": {"kind": "stim_device"},
    "escalation": {"kind": "weak_baseline"},
    "workload": {
        "kind": "producer",
        "function": "decsim.producers:memory_circuit",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": 15,
        },
    },
    "windows": {
        "kind": "sliding",
        "commit_rounds": None,
        "buffer_rounds": None,
    },
    "sweep": [
        {
            "physical_error_probability": [0.001],
            "distance": [3],
            "round_period_microseconds": [1.0],
            "shots": 1,
        }
    ],
    "controller": {
        "clock": "fridge",
        "readout_to_bits_cycles": 0,
        "packing_cycles_per_round": 0,
        "decision_to_pulse_cycles": 0,
        "packing_rounds_in_flight": None,
    },
    "clocks": {"fridge": 250.0, "room": 250.0},
    "links": {
        "qpu_to_controller": {
            "latency_cycles": 1,
            "clock": "fridge",
            "bits_per_cycle": None,
        }
    },
    "weak_syndrome_buffer": {"bits": None},
    "strong_syndrome_buffer": {"bits": None},
    "weak_decoder": {
        "kind": 0.028,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    },
    "pauli_frame": {"clock": "fridge", "write_cycles": 1},
}


def memory_workload(rounds_per_shot) -> dict:
    """The minimal config's workload section at another shot length."""
    return {
        "kind": "producer",
        "function": "decsim.producers:memory_circuit",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": rounds_per_shot,
        },
    }


def write_config(tmp_path, overrides: dict) -> pathlib.Path:
    raw = dict(MINIMAL_CONFIG)
    raw.update(overrides)
    config_path = tmp_path / "unit_test_config.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    return config_path


def strong_unit(algorithm) -> dict:
    return {
        "strong_decoder": {
            "kind": algorithm,
            "units": 1,
            "unit_memory": {"bits": None},
            "engine": {
                "clock": "room",
                "fetch_cycles_per_round": 1,
                "fetch_cycles_per_job": 0,
                "release_cycles_per_job": 1,
                "release_cycles_per_round": 0,
            },
        }
    }


def online_threshold() -> dict:
    """The overrides of a switching machine whose threshold learns online."""
    weak_decoder = dict(MINIMAL_CONFIG["weak_decoder"], kind="pymatching")
    escalation = {
        "kind": "switching",
        "gap_threshold_db": 15.0,
        "threshold_source": "online",
    }
    overrides = {"escalation": escalation, "weak_decoder": weak_decoder}
    strong_decoder = strong_unit("pymatching")
    overrides.update(strong_decoder)
    return overrides


def example_tool_config(
    tmp_path, qpu_kind: str, workload: dict, feedback_microseconds=4.0
) -> pathlib.Path:
    """The machine tools/deltakit_example.py and live_memory_example.py build.

    Their Python settings as a yaml: the logical reference links with the
    feedback path's latency on the fridge clock, a 0.1 us weak decoder on
    a 1 GHz engine, and the chrome trace with data movement. The frame
    section is required in a yaml, so it is written as the free frame.
    """
    fridge_megahertz = 250.0
    feedback_cycle_count = feedback_microseconds * fridge_megahertz
    feedback_cycles = round(feedback_cycle_count)
    raw = {
        "clocks": {"engine": 1000.0, "fridge": fridge_megahertz},
        "controller": {
            "clock": "fridge",
            "readout_to_bits_cycles": 0,
            "decision_to_pulse_cycles": 0,
            "packing_cycles_per_round": 0,
        },
        "links": {
            "kind": "logical_reference",
            "frame_to_controller": {
                "latency_cycles": feedback_cycles,
                "clock": "fridge",
                "bits_per_cycle": 32,
            },
        },
        "weak_syndrome_buffer": {"kind": "syndrome_buffer"},
        "strong_syndrome_buffer": {"kind": "syndrome_buffer"},
        "windows": {
            "kind": "sliding",
            "commit_rounds": None,
            "buffer_rounds": None,
        },
        "pauli_frame": {
            "kind": "logical_register",
            "clock": "fridge",
            "write_cycles": 0,
        },
        "weak_decoder": {
            "kind": 0.1,
            "units": 1,
            "unit_memory": {"bits": None},
            "engine": {
                "clock": "engine",
                "fetch_cycles_per_round": 1,
                "fetch_cycles_per_job": 0,
                "release_cycles_per_job": 1,
                "release_cycles_per_round": 0,
            },
        },
        "observation": {"trace": "chrome", "data_movement": True},
        "qpu": {"kind": qpu_kind},
        "workload": workload,
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_microseconds": [1.1],
                "shots": 1,
            }
        ],
    }
    config_path = tmp_path / "example_tool.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    return config_path
