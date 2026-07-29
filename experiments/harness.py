"""Small deterministic building blocks shared by experiment runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from decsim.message import RunSeedPathSegment
from decsim.seeding import derive_component_seed


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Batch:
    index: int
    first_shot: int
    shots: int


def exact_batches(max_shots: int, batch_shots: int) -> tuple[Batch, ...]:
    """Cover `[0, max_shots)` exactly, including a short final batch."""
    return tuple(
        Batch(index, first, min(batch_shots, max_shots - first))
        for index, first in enumerate(range(0, max_shots, batch_shots))
    )


@dataclass(frozen=True)
class SamplePlan:
    experiment_id: str
    experiment_seed: int
    input_sha256: str
    sample_set_id: str

    @classmethod
    def create(
        cls,
        experiment_id: str,
        experiment_seed: int,
        trajectory,
    ) -> "SamplePlan":
        input_sha256 = _sha256(trajectory)
        sample_set_id = _sha256({
            "schema_version": 1,
            "experiment_id": experiment_id,
            "experiment_seed": experiment_seed,
            "input_sha256": input_sha256,
        })
        return cls(
            experiment_id,
            experiment_seed,
            input_sha256,
            sample_set_id,
        )


def offline_batch_seed(
    experiment_seed: int, sample_set_id: str, batch_index: int
) -> int:
    """Derive one fresh Stim sampler seed for a deterministic batch."""
    path = (
        RunSeedPathSegment("field", "experiments"),
        RunSeedPathSegment("string_key", sample_set_id),
        RunSeedPathSegment("field", "offline_batch_sampler"),
        RunSeedPathSegment("integer_key", batch_index),
    )
    return derive_component_seed(experiment_seed, path)


def sample_batch_sha256(detectors, observable_truth) -> str:
    """Hash raw sample arrays with dtype and shape framing."""
    digest = hashlib.sha256()
    for name, array in (("detectors", detectors), ("observable_truth", observable_truth)):
        metadata = _canonical_json({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        })
        payload = array.tobytes(order="C")
        for part in (metadata, payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    experiment_seed: int
    configurations: tuple
    sampling: dict
    stopping: dict
    dependencies: dict
    repository_revision: str
    schema_version: int = 1

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self)) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def config_sha256(self, configuration) -> str:
        return _sha256(configuration)
