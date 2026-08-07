"""Run one offline BB72 Z-memory BP-OSD experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from decsim.bposd_decoder import bposd_window_decoder
from decsim.detector_error_model import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
)
from decsim.schemes import SlidingWindowScheme

from .bb import build_bb72_memory_z
from .decoding import OfflineBatchDecoder, run_offline_parallel
from .harness import Experiment, SamplePlan, exact_batches


_FIXED_CONFIGURATION = {
    "code": "bb72-12-6",
    "basis": "Z",
    "noise_model": "standard",
    "decoder": "bposd",
}
_VARIABLE_FIELDS = {
    "physical_error_rate", "syndrome_rounds", "commit_rounds",
    "buffer_rounds", "shots", "batch_shots", "seed", "workers",
    "experiment_id",
}


def _sliding_window_entries(configuration, detector_layer_count):
    plan = SlidingWindowScheme().plan_operation(
        0,
        detector_layer_count,
        commit_round_count=configuration["commit_rounds"],
        buffer_round_count=configuration["buffer_rounds"],
    )
    return tuple(
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    )


def _scientific_configuration(configuration):
    unknown = set(configuration) - _VARIABLE_FIELDS - set(_FIXED_CONFIGURATION)
    if unknown:
        raise ValueError(f"unsupported BB configuration field: {min(unknown)}")
    resolved = dict(configuration)
    resolved.pop("workers", None)
    for name, value in _FIXED_CONFIGURATION.items():
        if name in resolved and resolved[name] != value:
            raise ValueError(f"{name} must be {value}")
        resolved[name] = value
    resolved.setdefault("experiment_id", "bb72-z-standard-bposd")
    for name in ("commit_rounds", "buffer_rounds", "shots", "batch_shots"):
        value = resolved[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(resolved["seed"]) is not int or resolved["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    return resolved


@dataclass(frozen=True)
class _BbBposdFactory:
    configuration: dict

    def __call__(self):
        circuit, detector_rounds, layer_count = build_bb72_memory_z(
            physical_error_rate=self.configuration["physical_error_rate"],
            syndrome_round_count=self.configuration["syndrome_rounds"],
        )
        return OfflineBatchDecoder.prepare(
            circuit,
            _sliding_window_entries(self.configuration, layer_count),
            bposd_window_decoder(),
            round_count=layer_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
            fault_representation=FaultRepresentation.PHYSICAL,
        )


def _run_parts(configuration):
    resolved = _scientific_configuration(configuration)
    circuit, _, _ = build_bb72_memory_z(
        physical_error_rate=resolved["physical_error_rate"],
        syndrome_round_count=resolved["syndrome_rounds"],
    )
    experiment = Experiment(
        experiment_id=resolved["experiment_id"],
        experiment_seed=resolved["seed"],
        configurations=(resolved,),
        sampling={"batch_shots": resolved["batch_shots"],
                  "max_shots": resolved["shots"]},
        stopping={"method": "fixed"},
    )
    sample_plan = SamplePlan.create(
        resolved["experiment_id"],
        resolved["seed"],
        {
            "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
            "shots": resolved["shots"],
            "batch_shots": resolved["batch_shots"],
        },
    )
    return resolved, experiment, sample_plan, exact_batches(
        resolved["shots"], resolved["batch_shots"]
    )


def run_bb_configuration(configuration, output_directory):
    """Run or resume one fixed-shot BB72 physical BP-OSD configuration."""
    workers = configuration.get("workers", 1)
    resolved, experiment, sample_plan, batches = _run_parts(configuration)
    return run_offline_parallel(
        _BbBposdFactory(resolved),
        experiment,
        sample_plan,
        resolved,
        batches,
        output_directory,
        workers=workers,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="experiment-results")
    arguments = parser.parse_args(argv)
    configuration = json.loads(Path(arguments.config).read_text())
    result = run_bb_configuration(configuration, arguments.output)
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
