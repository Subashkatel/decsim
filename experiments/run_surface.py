"""Run one offline sliding-window surface-code accuracy point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from decsim.mwpm_decoder import matching_window_decoder

from .decoding import OfflineBatchDecoder, run_offline_parallel
from .harness import Experiment, SamplePlan, exact_batches


def _circuit(configuration):
    import stim

    probability = configuration["physical_error_rate"]
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=configuration["distance"],
        rounds=configuration["rounds"],
        after_clifford_depolarization=probability,
        after_reset_flip_probability=probability,
        before_measure_flip_probability=probability,
        before_round_data_depolarization=probability,
    )


def _windows(configuration):
    rounds = configuration["rounds"]
    commit = configuration["commit_rounds"]
    buffer = configuration["buffer_rounds"]
    windows = []
    for commit_lo in range(1, rounds + 1, commit):
        remaining = rounds - commit_lo + 1
        if remaining < commit and windows:
            previous_lo, _, _ = windows[-1]
            windows[-1] = (previous_lo, rounds, rounds)
            break
        commit_hi = min(commit_lo + commit - 1, rounds)
        windows.append((commit_lo, commit_hi, min(commit_hi + buffer, rounds)))
    return tuple(windows)


def _detector_rounds(circuit, rounds):
    return {
        detector: min(int(coordinates[-1]) + 1, rounds)
        for detector, coordinates in circuit.get_detector_coordinates().items()
    }


@dataclass(frozen=True)
class _SurfaceMwpmFactory:
    configuration: dict

    def __call__(self):
        circuit = _circuit(self.configuration)
        return OfflineBatchDecoder.prepare(
            circuit,
            _windows(self.configuration),
            matching_window_decoder(),
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
            fault_representation=FaultRepresentation.GRAPHLIKE,
            detector_rounds=_detector_rounds(
                circuit, self.configuration["rounds"]
            ),
        )


def run_surface_configuration(configuration, output_directory):
    """Run or resume one fixed-shot weak-MWPM accuracy configuration."""
    workers = configuration.get("workers", 1)
    scientific_configuration = dict(configuration)
    scientific_configuration.pop("workers", None)
    scientific_configuration.setdefault(
        "experiment_id", "surface-weak-mwpm"
    )
    circuit = _circuit(scientific_configuration)
    experiment_id = scientific_configuration["experiment_id"]
    experiment = Experiment(
        experiment_id=experiment_id,
        experiment_seed=scientific_configuration["seed"],
        configurations=(scientific_configuration,),
        sampling={
            "batch_shots": scientific_configuration["batch_shots"],
            "max_shots": scientific_configuration["shots"],
        },
        stopping={"method": "fixed"},
    )
    sample_plan = SamplePlan.create(
        experiment_id,
        scientific_configuration["seed"],
        {
            "circuit_sha256": hashlib.sha256(
                str(circuit).encode("utf-8")
            ).hexdigest(),
            "shots": scientific_configuration["shots"],
            "batch_shots": scientific_configuration["batch_shots"],
        },
    )
    return run_offline_parallel(
        _SurfaceMwpmFactory(scientific_configuration),
        experiment,
        sample_plan,
        scientific_configuration,
        exact_batches(
            scientific_configuration["shots"],
            scientific_configuration["batch_shots"],
        ),
        output_directory,
        workers=workers,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="experiment-results")
    arguments = parser.parse_args(argv)
    configuration = json.loads(Path(arguments.config).read_text())
    result = run_surface_configuration(configuration, arguments.output)
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
