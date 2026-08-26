"""Fast offline window decoding without the event engine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional
import hashlib
import os
from pathlib import Path

from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
from decsim.detector_error_model.fault_model_contracts import FaultRepresentation
from decsim.detector_error_model.window_model_builders import build_window_error_models
from decsim.decoders.window_decode_results import BackendDecodeStatus

from .harness import Batch, offline_batch_seed, sample_batch_sha256
from .results import (
    ChunkResult,
    canonical_chunk_csv,
    publish_immutable,
    read_chunk_csv,
    reduce_chunks,
)


@dataclass(frozen=True)
class OfflineShotRecord:
    """One attempted shot: sampled truth, every window outcome, and the
    accepted prediction (None when any window outcome failed)."""

    shot_index: int
    sampled_truth: tuple
    window_outcomes: tuple
    logical_prediction: tuple | None

    @property
    def accepted(self) -> bool:
        return self.logical_prediction is not None

    @property
    def logical_failure(self) -> bool:
        return not self.accepted or self.logical_prediction != self.sampled_truth


@dataclass(frozen=True)
class OfflineAccuracySummary:
    """Unconditional primary LER plus explicitly conditional acceptance metrics."""

    attempted_shots: int
    primary_failures: int
    accepted_shots: int
    accepted_logical_failures: int


def summarize_offline_shots(records) -> OfflineAccuracySummary:
    """Summarize attempted shots without filtering failed backend outcomes."""
    records = tuple(records)
    accepted_records = [record for record in records if record.accepted]
    return OfflineAccuracySummary(
        attempted_shots=len(records),
        primary_failures=sum(record.logical_failure for record in records),
        accepted_shots=len(accepted_records),
        accepted_logical_failures=sum(
            record.logical_prediction != record.sampled_truth
            for record in accepted_records),
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

    def __init__(self, circuit, window_models, decode_window, fault_representation,
                 forward_handoffs=()):
        self.circuit = circuit
        self.window_models = tuple(window_models)
        self.decode_window = decode_window
        self.fault_representation = fault_representation
        self.forward_handoffs = tuple(forward_handoffs)

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
    ):
        _require_forward_sliding(windows)
        models = build_window_error_models(
            circuit,
            windows,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=(),
        )
        round_of = resolve_detector_rounds(circuit, detector_rounds, round_count)
        handoffs = forward_handoffs_for(
            models, windows, round_of, fault_representation)
        return cls(circuit, models, decode_window, fault_representation, handoffs)

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
                forward_handoffs=self.forward_handoffs,
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
                forward_handoffs=self.forward_handoffs,
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
                self.window_models, detectors[index], self.decode_window,
                forward_handoffs=self.forward_handoffs,
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


# ---- the windowed reference decode -----------------------------------------
# Moved here from decsim.detector_error_model when the core trimmed it: the
# simulator decodes through the window manager, so the offline harness is the
# one owner of this list-ordered reference loop.

def forward_handoffs_for(window_models, window_plan, round_of,
                         representation: FaultRepresentation) -> tuple:
    """Per window, {column: detectors handed to a later window}: an owned
    column's detectors in rounds beyond that window's commit_hi. The last
    window hands nothing on. This reconstructs the forward-only handoff the
    core once stored as future_flips, from boundary_flips (each owned
    column's complete global detector effect) and the detector round map."""
    handoffs = []
    last_index = len(window_models) - 1
    for window_index, (model, entry) in enumerate(zip(window_models, window_plan)):
        commit_hi = entry[2] if len(entry) == 4 else entry[1]
        columns = {}
        if window_index != last_index:
            faults = model.require_faults(representation)
            for column, detector_ids in faults.boundary_flips.items():
                beyond_commit = tuple(detector_id for detector_id in detector_ids
                                      if round_of[detector_id] > commit_hi)
                if beyond_commit:
                    columns[int(column)] = beyond_commit
        handoffs.append(columns)
    return tuple(handoffs)


def _require_forward_sliding(plan) -> None:
    """The list-ordered decode loop supports forward sliding windows only:
    a 4-value entry with buffer_lo < commit_lo is a parallel A/B window and
    needs dependency-aware seam reconciliation."""
    for entry in plan:
        if len(entry) == 4 and entry[0] < entry[1]:
            raise ValueError(
                "list-ordered decode_windowed supports only forward sliding "
                "windows; parallel A/B windows require dependency-aware seam "
                "reconciliation")


@dataclass(frozen=True)
class WindowedBackendDecode:
    """Same-shot backend outcomes and a prediction only after full success."""

    window_outcomes: tuple
    logical_prediction: Optional[tuple]


def decode_windowed(
    window_models: list,
    detection_events,
    decode_window,
    *,
    selected_fault_representation: FaultRepresentation,
    forward_handoffs: tuple,
) -> "object":
    """Decode one shot through a forward-only sliding-window chain.

    Parallel block A/B decoding needs a dependency-aware seam stage and
    residual-syndrome handoff. This list-ordered helper intentionally rejects
    leading-buffer models instead of approximating that different algorithm.
    """
    logical_prediction, _ = _walk_windowed(
        window_models,
        detection_events,
        decode_window,
        selected_fault_representation,
        forward_handoffs,
        typed_backend_outcomes=False,
    )
    return logical_prediction


def decode_windowed_backend_outcomes(
    window_models: list,
    detection_events,
    decode_window,
    *,
    forward_handoffs: tuple,
) -> WindowedBackendDecode:
    """Walk physical windows once and preserve each exact backend outcome."""
    logical_prediction, outcomes = _walk_windowed(
        window_models,
        detection_events,
        decode_window,
        FaultRepresentation.PHYSICAL,
        forward_handoffs,
        typed_backend_outcomes=True,
    )
    return WindowedBackendDecode(
        window_outcomes=outcomes,
        logical_prediction=(
            None
            if logical_prediction is None
            else tuple(int(bit) for bit in logical_prediction)
        ),
    )


def _walk_windowed(
    window_models: list,
    detection_events,
    decode_window,
    selected_fault_representation: FaultRepresentation,
    forward_handoffs: tuple,
    *,
    typed_backend_outcomes: bool,
) -> tuple:
    """Single owner of detector selection, commitment, and boundary forwarding."""
    import numpy as np

    if not window_models:
        raise ValueError("windowed decode requires at least one window model")
    # Leading-buffer (parallel A/B) plans are refused at plan time by
    # _require_forward_sliding; the sliced models no longer carry bounds.
    pending: set = set()
    last_window_for_detector = {
        detector_id: window_index
        for window_index, model in enumerate(window_models)
        for detector_id in model.detector_ids
    }
    first_faults = window_models[0].require_faults(
        selected_fault_representation
    )
    total = np.zeros(first_faults.observables.shape[0], dtype=np.uint8)
    outcomes = []
    for window_index, model in enumerate(window_models):
        faults = model.require_faults(selected_fault_representation)
        syndrome = detection_events[list(model.detector_ids)].astype(np.uint8).copy()
        for detector_index, detector_id in enumerate(model.detector_ids):
            if detector_id in pending:
                syndrome[detector_index] ^= 1
                if last_window_for_detector[detector_id] == window_index:
                    pending.discard(detector_id)

        decoded = decode_window(model, syndrome)
        if typed_backend_outcomes:
            from decsim.decoders.window_decode_results import (
                validate_backend_outcome,
            )

            validate_backend_outcome(decoded, model, faults, syndrome)
            outcomes.append(decoded)
            if not decoded.succeeded:
                return None, tuple(outcomes)
            selected = np.asarray(
                decoded.physical_correction,
                dtype=np.uint8,
            )
        else:
            selected = np.asarray(decoded, dtype=np.uint8)
        if selected.shape != (faults.check.shape[1],):
            raise ValueError(
                "selected correction arity does not match the placed fault model"
            )
        committed = selected.astype(bool) & faults.owned
        total ^= (faults.observables @ committed.astype(np.uint8)) % 2
        for column_index in np.nonzero(committed)[0]:
            handed_on = forward_handoffs[window_index].get(int(column_index), ())
            for detector_id in handed_on:
                pending.symmetric_difference_update({detector_id})
    if pending:
        raise RuntimeError(f"artificial defects were never consumed: {sorted(pending)}"
                           ". The plan does not cover the full detector stream.")
    return total, tuple(outcomes)
