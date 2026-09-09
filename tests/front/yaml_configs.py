"""A complete runnable yaml, small enough for a functional test.

The shape is the owner's 2026-08-26 ruling: the algorithm is structure
on a per-tier unit card (weak_decoder / strong_decoder), never a sweep
axis, mirroring the tiered architecture itself (Toshio arXiv 2510.25222:
lightweight decoders decode constantly, a separate accurate decoder is
invoked on demand) and gem5's config split (structure on the component,
parameters swept around it).
"""

from pathlib import Path

import yaml

import decsim.collect as collect
import decsim.front.measure as measure

_THIS_FILE = Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _TEST_FILE.parents[2]
CONFIGS_DIR = _REPOSITORY_ROOT / "configs"
# The yaml experiments this repository ships. The tests that check every
# shipped config walk this tuple and not the folder, so a config a user
# writes into configs/ of their own checkout fails none of them.
SHIPPED_CONFIGS = (
    "reference.yaml",
    "seam_pinned_switching.yaml",
    "strong_decoder_baseline.yaml",
    "strong_latency.yaml",
    "strong_latency_preview.yaml",
    "strong_ler.yaml",
    "weak_decoder_baseline.yaml",
    "weak_latency.yaml",
    "weak_ler.yaml",
)


def measure_point_shot(
    config, *, physical_error_probability, distance, round_period_us, seed
):
    """One seeded shot at one sweep point, collected and measured."""
    shots = seed + 1
    task = config.point_task(
        physical_error_probability=physical_error_probability,
        distance=distance,
        round_period_us=round_period_us,
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
        "kind": "memory_circuit",
        "code_task": "surface_code:rotated_memory_z",
        "rounds_per_shot": 15,
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
            "round_period_us": [1.0],
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
    "round_store": {"rounds": None},
    "strong_round_store": {"rounds": None},
    "weak_decoder": {
        "kind": 0.028,
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
        },
    },
    "pauli_frame": {"clock": "fridge", "write_cycles": 1},
}


def write_config(tmp_path, overrides: dict) -> Path:
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
            "unit_memory_rounds": None,
            "engine": {
                "clock": "room",
                "fetch_cycles_per_round": 1,
                "release_cycles_per_job": 1,
            },
        }
    }
