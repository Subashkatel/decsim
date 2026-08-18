"""Run one offline QUITS-1.1 BB-memory experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path

from decsim.decoders.bposd import bposd_window_decoder
from decsim.detector_error_model import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
)
from decsim.windows.windowing_schemes import SlidingWindowScheme
from decsim.decoders.relay_bp import RelayBpWindowDecoder
from decsim.decoders.tesseract import TesseractDecoderConfig, TesseractWindowDecoder

from .bb import build_quits11_bb_memory
from .decoding import (
    OfflineBackendBatchDecoder, OfflineBatchDecoder, run_offline_parallel,
)
from .harness import Experiment, SamplePlan, exact_batches


_FIXED_CONFIGURATION = {"circuit_model": "quits-1.1-custom"}
_DECODERS = frozenset({"bposd", "relay_bp", "tesseract"})
_RELAY_PROFILE = {
    "alpha": None, "alpha_iteration_scaling_factor": 1.0, "gamma0": 0.1,
    "pre_iterations": 80, "relay_set_count": 300,
    "iterations_per_set": 60, "gamma_interval": (-0.24, 0.66),
    "converged_solution_count": 1,
}
_TESSERACT_PROFILE = {
    "detector_beam": 15, "beam_climbing": True,
    "no_revisit_detectors": True, "priority_queue_limit": 200_000,
    "detector_order_method": "index", "detector_order_count": 16,
}
_BACKEND_PACKAGES = {
    "relay_bp": ("relay-bp", "relay_bp", "0.2.2"),
    "tesseract": ("tesseract-decoder", "tesseract_decoder",
                  "0.1.1.dev20260802231159"),
}
_VARIABLE_FIELDS = {
    "definition_id", "basis", "noise_profile", "physical_error_rate",
    "syndrome_rounds", "commit_rounds", "buffer_rounds", "shots",
    "batch_shots", "seed", "workers", "experiment_id", "decoder",
    "decoder_seed",
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
    resolved.setdefault("experiment_id", "bb-offline")
    decoder = resolved.setdefault("decoder", "bposd")
    if decoder not in _DECODERS:
        raise ValueError("decoder must be bposd, relay_bp, or tesseract")
    decoder_seed = resolved.get("decoder_seed")
    if decoder == "bposd" and "decoder_seed" in resolved:
        raise ValueError("decoder_seed is not valid for bposd")
    if decoder != "bposd":
        if type(decoder_seed) is not int or not 0 <= decoder_seed < 2**64:
            raise ValueError("decoder_seed must be an unsigned 64-bit integer")
        profile = _RELAY_PROFILE if decoder == "relay_bp" else _TESSERACT_PROFILE
        resolved_profile = {
            "name": decoder, **profile, "decoder_seed": decoder_seed,
        }
        if decoder == "tesseract":
            resolved_profile.update(verbose=False, create_visualization=False)
        resolved["decoder_profile"] = resolved_profile
    for name in ("commit_rounds", "buffer_rounds", "shots", "batch_shots"):
        value = resolved[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(resolved["seed"]) is not int or resolved["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    return resolved


def _build_circuit(configuration):
    return build_quits11_bb_memory(
        definition_id=configuration["definition_id"],
        basis=configuration["basis"],
        noise_profile=configuration["noise_profile"],
        physical_error_rate=configuration["physical_error_rate"],
        syndrome_round_count=configuration["syndrome_rounds"],
    )


@dataclass(frozen=True)
class _BbDecoderFactory:
    configuration: dict

    def __call__(self):
        circuit, detector_rounds, layer_count = _build_circuit(self.configuration)
        decoder = self.configuration["decoder"]
        if decoder == "relay_bp":
            batch_decoder = OfflineBackendBatchDecoder
            decode_window = RelayBpWindowDecoder(
                **_RELAY_PROFILE, gamma_table_seed=self.configuration["decoder_seed"]
            ).decode
        elif decoder == "tesseract":
            batch_decoder = OfflineBackendBatchDecoder
            decode_window = TesseractWindowDecoder(TesseractDecoderConfig(
                **_TESSERACT_PROFILE,
                detector_order_seed=self.configuration["decoder_seed"],
            )).decode
        else:
            batch_decoder = OfflineBatchDecoder
            decode_window = bposd_window_decoder()
        return batch_decoder.prepare(
            circuit,
            _sliding_window_entries(self.configuration, layer_count),
            decode_window,
            round_count=layer_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
            fault_representation=FaultRepresentation.PHYSICAL,
        )


def _run_parts(configuration):
    resolved = _scientific_configuration(configuration)
    circuit, _, _ = _build_circuit(resolved)
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
            "definition_id": resolved["definition_id"],
            "circuit_model": resolved["circuit_model"],
            "basis": resolved["basis"],
            "noise_profile": resolved["noise_profile"],
            "physical_error_rate": resolved["physical_error_rate"],
            "syndrome_rounds": resolved["syndrome_rounds"],
            "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
            "shots": resolved["shots"],
            "batch_shots": resolved["batch_shots"],
        },
    )
    return resolved, experiment, sample_plan, exact_batches(
        resolved["shots"], resolved["batch_shots"]
    )


def _check_backend_dependency(decoder):
    if decoder == "bposd":
        return
    distribution, module, expected = _BACKEND_PACKAGES[decoder]
    try:
        installed = package_version(distribution)
        __import__(module)
    except (PackageNotFoundError, ImportError) as error:
        raise ImportError(f"{decoder} requires {distribution}=={expected}") from error
    if installed != expected:
        raise RuntimeError(
            f"{decoder} requires {distribution}=={expected}; found {installed}"
        )


def run_bb_configuration(configuration, output_directory):
    """Run or resume one fixed-shot QUITS-1.1 BB physical configuration."""
    workers = configuration.get("workers", 1)
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    resolved, experiment, sample_plan, batches = _run_parts(configuration)
    _check_backend_dependency(resolved["decoder"])
    return run_offline_parallel(
        _BbDecoderFactory(resolved),
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
