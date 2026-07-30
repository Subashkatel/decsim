"""Run one offline sliding-window surface-code accuracy point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import re
import shlex
import subprocess
import sys

from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from decsim.mwpm_decoder import matching_window_decoder

from .decoding import OfflineBatchDecoder, read_stored_batch_result, run_offline_parallel
from .harness import (
    Experiment,
    SamplePlan,
    canonical_json,
    exact_batches,
    offline_batch_seed,
)
from .plotting import plot_logical_error_rate, write_logical_error_rate_card
from .results import canonical_surface_plot_csv, surface_plot_row


_ARTIFACTS = {
    "configuration", "invocation", "runner_source", "circuit", "dem",
    "detector_coordinates", "detector_records", "results", "plot", "plot_card",
    "environment",
}
_RUN_NAME = re.compile(r"[A-Za-z0-9._-]+")


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


def _scientific_configuration(configuration):
    scientific_configuration = dict(configuration)
    for name in ("workers", "run_name", "artifacts", "detector_record_shots"):
        scientific_configuration.pop(name, None)
    scientific_configuration.setdefault("experiment_id", "surface-weak-mwpm")
    return scientific_configuration


def _surface_run_parts(configuration):
    scientific_configuration = _scientific_configuration(configuration)
    circuit = _circuit(scientific_configuration)
    experiment_id = scientific_configuration["experiment_id"]
    experiment = Experiment(
        experiment_id=experiment_id,
        experiment_seed=scientific_configuration["seed"],
        configurations=(scientific_configuration,),
        sampling={"batch_shots": scientific_configuration["batch_shots"],
                  "max_shots": scientific_configuration["shots"]},
        stopping={"method": "fixed"},
    )
    sample_plan = SamplePlan.create(
        experiment_id,
        scientific_configuration["seed"],
        {
            "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
            "shots": scientific_configuration["shots"],
            "batch_shots": scientific_configuration["batch_shots"],
        },
    )
    batches = exact_batches(scientific_configuration["shots"],
                            scientific_configuration["batch_shots"])
    return scientific_configuration, circuit, experiment, sample_plan, batches


def run_surface_configuration(configuration, output_directory):
    """Run or resume one fixed-shot weak-MWPM accuracy configuration."""
    workers = configuration.get("workers", 1)
    scientific_configuration, _, experiment, sample_plan, batches = _surface_run_parts(
        configuration
    )
    return run_offline_parallel(
        _SurfaceMwpmFactory(scientific_configuration),
        experiment,
        sample_plan,
        scientific_configuration,
        batches,
        output_directory,
        workers=workers,
    )


def _selected_artifacts(configuration):
    if "artifacts" not in configuration:
        return None
    artifacts = configuration["artifacts"]
    if type(artifacts) is not list or not artifacts:
        raise ValueError("artifacts must be a nonempty list")
    if any(type(name) is not str or name not in _ARTIFACTS for name in artifacts):
        raise ValueError("artifacts contains an unknown name")
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("artifacts contains a duplicate name")
    run_name = configuration.get("run_name")
    if type(run_name) is not str or _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run_name must be one nonempty path-safe component")
    if "detector_records" in artifacts:
        shots = configuration.get("detector_record_shots")
        if type(shots) is not list or not shots:
            raise ValueError("detector_record_shots must be a nonempty list")
        if any(type(shot) is not int for shot in shots):
            raise ValueError("detector_record_shots must contain integers")
        if shots != sorted(set(shots)):
            raise ValueError("detector_record_shots must be distinct and increasing")
        if shots[0] < 0 or shots[-1] >= configuration["shots"]:
            raise ValueError("detector_record_shots is outside the shot range")
    return tuple(artifacts)


def _new_output_directory(output_root, run_name, config_id):
    current = datetime.now(timezone.utc)
    dated_root = Path(output_root) / current.strftime("%Y-%m-%d")
    directory = dated_root / f"{current.strftime('%H%M%S%f')}_{run_name}_{config_id}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _invocation_observation(config_path, output_root, process_argv):
    replay_argv = [sys.executable, "-m", "experiments.run_surface", "--config",
                   str(Path(config_path).resolve()), "--output",
                   str(Path(output_root).resolve())]
    return {
        "process_argv_observed": list(process_argv),
        "cwd_observed": str(Path.cwd()),
        "replay_argv": replay_argv,
        "replay_command": shlex.join(replay_argv),
    }


def _environment_observation():
    repository = Path(__file__).parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    return {
        "repository_revision": revision,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "direct_package_versions": {name: version(name) for name in
                                    ("stim", "pymatching", "numpy", "scipy", "matplotlib")},
        "scope": "observed versions only; not a dependency closure",
    }


def _detector_record_bytes(
    configuration, output_root, result, batches, selected_shots
):
    decoder = _SurfaceMwpmFactory(configuration)()
    records = []
    for batch in batches:
        batch_shots = [shot for shot in selected_shots
                       if batch.first_shot <= shot < batch.first_shot + batch.shots]
        if not batch_shots:
            continue
        stored = read_stored_batch_result(output_root, result, batch.index)
        sample_seed = offline_batch_seed(configuration["seed"],
                                         result.sample_set_id, batch.index)
        records.extend(decoder.capture_detector_records(
            batch, sample_seed, batch_shots, stored.sample_batch_sha256))
    return b"".join(canonical_json(record) + b"\n" for record in records)


def write_surface_snapshot(requested_configuration, output_root, result,
                           artifacts, invocation):
    configuration, circuit, experiment, sample_plan, batches = _surface_run_parts(
        requested_configuration
    )
    row = surface_plot_row(configuration, result)
    directory = _new_output_directory(output_root,
                                      requested_configuration["run_name"],
                                      result.config_id)
    written = []

    def write_bytes(name, content):
        path = directory / name
        path.write_bytes(content)
        written.append(path)

    if "configuration" in artifacts:
        write_bytes("configuration.json", canonical_json({
            "requested": requested_configuration,
            "resolved": configuration,
            "experiment_sha256": experiment.sha256(),
            "config_id": result.config_id,
            "sample_set_id": sample_plan.sample_set_id,
            "batches": [asdict(batch) for batch in batches],
            "windows": _windows(configuration),
        }) + b"\n")
    if "invocation" in artifacts:
        write_bytes("invocation.json", canonical_json(invocation) + b"\n")
    if "runner_source" in artifacts:
        write_bytes("runner.py", Path(__file__).read_bytes())
    if "circuit" in artifacts:
        write_bytes("circuit.stim", str(circuit).encode("utf-8"))
    if "dem" in artifacts:
        model = circuit.detector_error_model(decompose_errors=True)
        write_bytes("detector_error_model.dem", str(model).encode("utf-8"))
    if "detector_coordinates" in artifacts:
        coordinates = [{"detector": detector, "coordinates": list(values)}
                       for detector, values in
                       sorted(circuit.get_detector_coordinates().items())]
        write_bytes("detector_coordinates.json", canonical_json(coordinates) + b"\n")
    if "detector_records" in artifacts:
        records = _detector_record_bytes(
            configuration,
            output_root,
            result,
            batches,
            requested_configuration["detector_record_shots"],
        )
        write_bytes("detector_records.jsonl", records)
    if "results" in artifacts:
        write_bytes("reduced_counts.json", canonical_json(asdict(result)) + b"\n")
        write_bytes("plot_data.csv", canonical_surface_plot_csv(row))
    if "plot" in artifacts:
        path = directory / "logical_error_rate.png"
        plot_logical_error_rate([row], path)
        written.append(path)
    if "plot_card" in artifacts:
        path = directory / "logical_error_rate.md"
        write_logical_error_rate_card([row], path)
        written.append(path)
    if "environment" in artifacts:
        write_bytes("environment.json", canonical_json(_environment_observation()) + b"\n")

    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(written, key=lambda path: path.name.encode("utf-8"))
    ]
    (directory / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="experiment-results")
    arguments = parser.parse_args(argv)
    configuration = json.loads(Path(arguments.config).read_text())
    artifacts = _selected_artifacts(configuration)
    result = run_surface_configuration(configuration, arguments.output)
    if artifacts is not None:
        invocation = _invocation_observation(arguments.config, arguments.output,
                                             sys.argv)
        write_surface_snapshot(configuration, arguments.output, result, artifacts,
                               invocation)
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
