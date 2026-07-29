"""Fast offline window decoding without the event engine."""

from dataclasses import dataclass

from decsim.detector_error_model import (
    build_window_error_models,
    decode_windowed,
)

from .harness import Batch, sample_batch_sha256


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
