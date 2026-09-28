"""The runnable yamls this repository ships, and a small one for tests.

SHIPPED_CONFIGS names the runnable files under configs/; the files
under configs/bases/ are starting points other files extend, so they
are left out.
MINIMAL_CONFIG is a complete machine small enough for a functional test.
"""

import pathlib

import numpy
import pymatching
import yaml

import decsim.collect as collect
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _TEST_FILE.parents[2]
CONFIGS_DIR = _REPOSITORY_ROOT / "configs"
# The yaml experiments this repository ships. The tests that check every
# shipped config walk this tuple and not the folder, so a config a user
# writes into configs/ of their own checkout fails none of them.
SHIPPED_CONFIGS = (
    "reference.yaml",
    "examples/my_first_sweep.yaml",
    "examples/priced_cards_example.yaml",
    "examples/two_tiers.yaml",
    "experiments/burst_detection/burst_detection.yaml",
    "experiments/data_movement/data_movement.yaml",
    "experiments/decoder_baseline/decoder_baseline.yaml",
    "experiments/switching/cluster_gap_switching.yaml",
    "experiments/switching/seam_pinned_switching.yaml",
)
# Where a memory maker's physical error rate sits in a point's sections.
ERROR_RATE_PATH = "workload.arguments.physical_error_probability"
# The minimal config's sweep without the error rate, for a files row or a
# maker that takes none.
QPU_ONLY_SWEEP = [
    {
        "axes": {"qpu.distance": [3], "qpu.round_period_microseconds": [1.0]},
        "collection": {"max_shots": 1},
    }
]


def point_shot(
    config: experiment.ExperimentConfig,
    *,
    physical_error_probability: float,
    distance: int,
    round_period_microseconds: float,
    seed: int,
) -> collect.Shot:
    """One seeded shot at one sweep point, run."""
    task = config.point_task(
        {
            ERROR_RATE_PATH: physical_error_probability,
            "qpu.distance": distance,
            "qpu.round_period_microseconds": round_period_microseconds,
        },
    )
    return collect.run_shot(task, seed)


def measure_point_shot(
    config: experiment.ExperimentConfig, **point
) -> measure.ShotMeasurement:
    """One seeded shot at one sweep point, collected and measured."""
    shot = point_shot(config, **point)
    return measure.measure_shot(shot)


def run_sweep(tasks: list, shots: int) -> list:
    """Seeds 0 to shots - 1 of every task of a sweep, measured, unsaved."""
    return collect.collect(tasks, shots, measure.measure_shot)


def whole_circuit_predictions(shot: collect.Shot) -> list:
    """Each operation's observables, PyMatching on its whole circuit.

    The reference a windowed loop is checked against: the decomposed
    detector error model of the circuit the source sampled, decoded in
    one piece on the events it drew, operation by operation in the
    result's order.
    """
    sampled = shot.machine.observation.sampled_shots.shots_by_operation
    predictions = []
    for operation_result in shot.result.operation_results:
        sampled_shot = sampled[operation_result.operation_id]
        model = sampled_shot.circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        events = numpy.asarray(sampled_shot.detection_events, dtype=bool)
        predicted = matching.decode(events)
        prediction = tuple(int(bit) for bit in predicted)
        predictions.append(prediction)
    return predictions


def loop_predictions(shot: collect.Shot) -> list:
    """Each operation's observables as the machine's loop decoded them."""
    return [
        tuple(operation_result.logical_observables)
        for operation_result in shot.result.operation_results
    ]


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
            "distance": "${qpu.distance}",
        },
    },
    "windows": {
        "kind": "sliding",
        "commit_rounds": None,
        "buffer_rounds": None,
    },
    "sweep": [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.001],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": 1},
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
            "distance": "${qpu.distance}",
        },
    }


def write_config(tmp_path, overrides: dict) -> pathlib.Path:
    raw = dict(MINIMAL_CONFIG)
    raw.update(overrides)
    config_path = tmp_path / "unit_test_config.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    return config_path


def run_folder_of(experiment_dir) -> pathlib.Path:
    """The run folder a collect of one yaml wrote: combined/<name>-<id8>/."""
    combined = pathlib.Path(experiment_dir) / "combined"
    (folder,) = combined.iterdir()
    return folder


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


def fixed_threshold_switching() -> dict:
    """The overrides of a switching machine that escalates below 20 dB."""
    overrides = online_threshold()
    overrides["escalation"] = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "near_seam_pinned",
    }
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
                "axes": {
                    "qpu.distance": [3],
                    "qpu.round_period_microseconds": [1.1],
                },
                "collection": {"max_shots": 1},
            }
        ],
    }
    config_path = tmp_path / "example_tool.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    return config_path
