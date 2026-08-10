"""Fast offline window decoding without the event engine."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

from decsim.detector_error_model import (
    build_window_error_models,
    decode_windowed_backend_outcomes,
    decode_windowed,
    resolve_detector_rounds,
)
from decsim.adapters.window_decode_results import (
    BackendDecodeStatus,
    OfflineShotRecord,
    summarize_offline_shots,
)

from .harness import Batch, offline_batch_seed, sample_batch_sha256
from .results import (
    ChunkResult,
    canonical_chunk_csv,
    publish_immutable,
    read_chunk_csv,
    reduce_chunks,
)


@dataclass(frozen=True)
class DecodedBatch:
    batch: Batch
    sample_batch_sha256: str
    attempted_shots: int
    primary_failures: int
    accepted_shots: int
    accepted_logical_failures: int
    window_attempts: int
    backend_low_confidence: int = 0
    backend_nonconverged: int = 0
    backend_invalid_correction: int = 0
    backend_empty_model_unsatisfiable: int = 0
    backend_error: int = 0


def load_layered_stim_input(circuit_path, expected_sha256,
                            detector_counts_by_round):
    """Load immutable Stim bytes and expand per-round detector counts."""
    import stim

    source = Path(circuit_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Stim input SHA-256 does not match the declaration")
    if type(detector_counts_by_round) is not tuple or not detector_counts_by_round:
        raise TypeError("detector_counts_by_round must be a nonempty tuple")
    if any(type(count) is not int or count < 1
           for count in detector_counts_by_round):
        raise ValueError("detector counts must be positive built-in ints")

    circuit = stim.Circuit(source.decode("utf-8"))
    if sum(detector_counts_by_round) != circuit.num_detectors:
        raise ValueError("declared detector counts do not match the circuit")
    detector_rounds = {}
    detector_id = 0
    for round_index, count in enumerate(detector_counts_by_round, start=1):
        for _ in range(count):
            detector_rounds[detector_id] = round_index
            detector_id += 1
    round_count = len(detector_counts_by_round)
    return circuit, resolve_detector_rounds(
        circuit, detector_rounds, round_count
    ), round_count


class OfflineBatchDecoder:
    """Reuse window models and a cached decoder while sampling fresh batches."""

    def __init__(self, circuit, window_models, decode_window, fault_representation):
        self.circuit = circuit
        self.window_models = tuple(window_models)
        self.decode_window = decode_window
        self.fault_representation = fault_representation

    @classmethod
    def prepare(
        cls,
        circuit,
        windows,
        decode_window,
        *,
        round_count,
        fault_model_requirement,
        fault_representation,
        detector_rounds=None,
        num_observables=None,
    ):
        models = build_window_error_models(
            circuit,
            windows,
            num_observables,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=(),
        )
        return cls(circuit, models, decode_window, fault_representation)

    def run(self, batch: Batch, sample_seed: int) -> DecodedBatch:
        sampler = self.circuit.compile_detector_sampler(seed=sample_seed)
        detectors, truth = sampler.sample(
            shots=batch.shots,
            separate_observables=True,
        )
        failures = 0
        for index in range(batch.shots):
            prediction = decode_windowed(
                self.window_models,
                detectors[index],
                self.decode_window,
                selected_fault_representation=self.fault_representation,
            )
            failures += tuple(int(bit) for bit in prediction) != tuple(
                int(bit) for bit in truth[index]
            )
        return DecodedBatch(
            batch=batch,
            sample_batch_sha256=sample_batch_sha256(detectors, truth),
            attempted_shots=batch.shots,
            primary_failures=failures,
            accepted_shots=batch.shots,
            accepted_logical_failures=failures,
            window_attempts=batch.shots * len(self.window_models),
        )

    def capture_detector_records(
        self,
        batch: Batch,
        sample_seed: int,
        global_shot_indices,
        expected_sample_sha256: str,
    ) -> tuple[dict, ...]:
        sampler = self.circuit.compile_detector_sampler(seed=sample_seed)
        detectors, truth = sampler.sample(
            shots=batch.shots,
            separate_observables=True,
        )
        actual_sample_sha256 = sample_batch_sha256(detectors, truth)
        if actual_sample_sha256 != expected_sample_sha256:
            raise ValueError("stored sample digest does not match reconstructed batch")

        records = []
        for global_shot_index in global_shot_indices:
            local_shot_index = global_shot_index - batch.first_shot
            detector_row = detectors[local_shot_index]
            truth_bits = tuple(int(bit) for bit in truth[local_shot_index])
            prediction_bits = tuple(int(bit) for bit in decode_windowed(
                self.window_models,
                detector_row,
                self.decode_window,
                selected_fault_representation=self.fault_representation,
            ))
            records.append({
                "shot_index": global_shot_index,
                "batch_index": batch.index,
                "sample_batch_sha256": actual_sample_sha256,
                "fired_detector_indices": [
                    detector_index
                    for detector_index, fired in enumerate(detector_row)
                    if fired
                ],
                "observable_truth": list(truth_bits),
                "prediction": list(prediction_bits),
                "logical_failure": prediction_bits != truth_bits,
            })
        return tuple(records)


_BACKEND_FAILURE_FIELDS = {
    BackendDecodeStatus.LOW_CONFIDENCE: "backend_low_confidence",
    BackendDecodeStatus.NONCONVERGED: "backend_nonconverged",
    BackendDecodeStatus.INVALID_CORRECTION: "backend_invalid_correction",
    BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE:
        "backend_empty_model_unsatisfiable",
    BackendDecodeStatus.BACKEND_ERROR: "backend_error",
}


class OfflineBackendBatchDecoder(OfflineBatchDecoder):
    """Preserve typed physical-backend failures in offline batch evidence."""

    def run(self, batch: Batch, sample_seed: int) -> DecodedBatch:
        sampler = self.circuit.compile_detector_sampler(seed=sample_seed)
        detectors, truth = sampler.sample(
            shots=batch.shots, separate_observables=True
        )
        records = []
        counts = {status: 0 for status in _BACKEND_FAILURE_FIELDS}
        window_attempts = 0
        for index in range(batch.shots):
            decoded = decode_windowed_backend_outcomes(
                self.window_models, detectors[index], self.decode_window
            )
            window_attempts += len(decoded.window_outcomes)
            terminal_status = decoded.window_outcomes[-1].status
            if terminal_status is not BackendDecodeStatus.SUCCEEDED:
                counts[terminal_status] += 1
            records.append(OfflineShotRecord(
                batch.first_shot + index,
                tuple(int(bit) for bit in truth[index]),
                decoded.window_outcomes,
                decoded.logical_prediction,
            ))
        summary = summarize_offline_shots(records)
        return DecodedBatch(
            batch=batch,
            sample_batch_sha256=sample_batch_sha256(detectors, truth),
            attempted_shots=summary.attempted_shots,
            primary_failures=summary.primary_failures,
            accepted_shots=summary.accepted_shots,
            accepted_logical_failures=summary.accepted_logical_failures,
            window_attempts=window_attempts,
            **{field: counts[status]
               for status, field in _BACKEND_FAILURE_FIELDS.items()},
        )


def _chunk_row(decoded, experiment, sample_plan, experiment_sha256, config_id):
    batch = decoded.batch
    return ChunkResult(
        schema_version=1,
        experiment_id=experiment.experiment_id,
        experiment_sha256=experiment_sha256,
        config_id=config_id,
        sample_set_id=sample_plan.sample_set_id,
        sample_batch_sha256=decoded.sample_batch_sha256,
        batch_index=batch.index,
        first_shot_index=batch.first_shot,
        requested_shots=batch.shots,
        attempted_shots=decoded.attempted_shots,
        primary_failures=decoded.primary_failures,
        accepted_shots=decoded.accepted_shots,
        accepted_logical_failures=decoded.accepted_logical_failures,
        backend_low_confidence=decoded.backend_low_confidence,
        backend_nonconverged=decoded.backend_nonconverged,
        backend_invalid_correction=decoded.backend_invalid_correction,
        backend_empty_model_unsatisfiable=
            decoded.backend_empty_model_unsatisfiable,
        backend_error=decoded.backend_error,
        window_attempts=decoded.window_attempts,
    )


def _result_directory(output_directory, experiment_sha256, config_id):
    return (
        Path(output_directory)
        / experiment_sha256
        / "scientific"
        / "chunks"
        / config_id
    )


def read_stored_batch_result(
    output_directory, reduced_result, batch_index
) -> ChunkResult:
    directory = _result_directory(
        output_directory,
        reduced_result.experiment_sha256,
        reduced_result.config_id,
    )
    return read_chunk_csv(directory / f"{batch_index}.csv")


def _publish_row(directory, row):
    path = directory / f"{row.batch_index}.csv"
    publish_immutable(path, canonical_chunk_csv([row]))
    return row


def _validate_stored_row(
    row, batch, experiment, sample_plan, experiment_sha256, config_id
):
    expected = {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "experiment_sha256": experiment_sha256,
        "config_id": config_id,
        "sample_set_id": sample_plan.sample_set_id,
        "batch_index": batch.index,
        "first_shot_index": batch.first_shot,
        "requested_shots": batch.shots,
    }
    for field, value in expected.items():
        if getattr(row, field) != value:
            raise ValueError(
                f"stored batch {batch.index} has stale {field}; "
                "refusing to mix scientific samples"
            )
    return row


def run_offline_experiment(
    decoder,
    experiment,
    sample_plan,
    configuration,
    batches,
    output_directory,
):
    """Run missing offline batches, publish them once, and reduce exact counts."""
    experiment_sha256 = experiment.sha256()
    config_id = experiment.config_sha256(configuration)
    directory = _result_directory(output_directory, experiment_sha256, config_id)
    rows = []
    for batch in batches:
        path = directory / f"{batch.index}.csv"
        if path.exists():
            row = _validate_stored_row(
                read_chunk_csv(path), batch, experiment, sample_plan,
                experiment_sha256, config_id,
            )
        else:
            decoded = decoder.run(
                batch,
                offline_batch_seed(
                    experiment.experiment_seed,
                    sample_plan.sample_set_id,
                    batch.index,
                ),
            )
            row = _publish_row(
                directory,
                _chunk_row(
                    decoded,
                    experiment,
                    sample_plan,
                    experiment_sha256,
                    config_id,
                ),
            )
        rows.append(row)
    return reduce_chunks(rows)


_worker_decoder = None


def _start_worker(decoder_factory):
    global _worker_decoder
    _worker_decoder = decoder_factory()


def _run_worker(task):
    batch, seed = task
    return _worker_decoder.run(batch, seed)


def run_offline_parallel(
    decoder_factory,
    experiment,
    sample_plan,
    configuration,
    batches,
    output_directory,
    *,
    workers=4,
):
    """Decode missing batches in bounded processes; publish in the parent."""
    batches = tuple(sorted(batches, key=lambda batch: batch.index))
    experiment_sha256 = experiment.sha256()
    config_id = experiment.config_sha256(configuration)
    directory = _result_directory(output_directory, experiment_sha256, config_id)
    missing = []
    for batch in batches:
        path = directory / f"{batch.index}.csv"
        if path.exists():
            _validate_stored_row(
                read_chunk_csv(path), batch, experiment, sample_plan,
                experiment_sha256, config_id,
            )
        else:
            missing.append(batch)
    if missing:
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", workers))
        worker_count = min(workers, allocated, len(missing))
        tasks = [
            (
                batch,
                offline_batch_seed(
                    experiment.experiment_seed,
                    sample_plan.sample_set_id,
                    batch.index,
                ),
            )
            for batch in missing
        ]
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_start_worker,
            initargs=(decoder_factory,),
        ) as pool:
            decoded_batches = pool.map(_run_worker, tasks)
            for decoded in decoded_batches:
                _publish_row(
                    directory,
                    _chunk_row(
                        decoded,
                        experiment,
                        sample_plan,
                        experiment_sha256,
                        config_id,
                    ),
                )
    return reduce_chunks(
        _validate_stored_row(
            read_chunk_csv(directory / f"{batch.index}.csv"),
            batch, experiment, sample_plan, experiment_sha256, config_id,
        )
        for batch in batches
    )
