"""Fast offline window decoding without the event engine."""

from dataclasses import dataclass
from pathlib import Path

from decsim.detector_error_model import (
    build_window_error_models,
    decode_windowed,
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
        fault_model_requirement,
        fault_representation,
        detector_rounds=None,
        num_observables=None,
    ):
        models = build_window_error_models(
            circuit,
            windows,
            num_observables,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
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
    directory = (
        Path(output_directory)
        / experiment_sha256
        / "scientific"
        / "chunks"
        / config_id
    )
    rows = []
    for batch in batches:
        path = directory / f"{batch.index}.csv"
        if path.exists():
            row = read_chunk_csv(path)
        else:
            decoded = decoder.run(
                batch,
                offline_batch_seed(
                    experiment.experiment_seed,
                    sample_plan.sample_set_id,
                    batch.index,
                ),
            )
            row = ChunkResult(
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
                backend_low_confidence=0,
                backend_nonconverged=0,
                backend_invalid_correction=0,
                backend_empty_model_unsatisfiable=0,
                backend_error=0,
                window_attempts=decoded.window_attempts,
            )
            publish_immutable(path, canonical_chunk_csv([row]))
        rows.append(row)
    return reduce_chunks(rows)
