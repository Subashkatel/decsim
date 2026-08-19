"""Deterministic scientific chunk storage and exact reduction."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import io
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class ChunkResult:
    schema_version: int
    experiment_id: str
    experiment_sha256: str
    config_id: str
    sample_set_id: str
    sample_batch_sha256: str
    batch_index: int
    first_shot_index: int
    requested_shots: int
    attempted_shots: int
    primary_failures: int
    accepted_shots: int
    accepted_logical_failures: int
    backend_low_confidence: int
    backend_nonconverged: int
    backend_invalid_correction: int
    backend_empty_model_unsatisfiable: int
    backend_error: int
    window_attempts: int


@dataclass(frozen=True)
class ReducedResult:
    schema_version: int
    experiment_id: str
    experiment_sha256: str
    config_id: str
    sample_set_id: str
    attempted_shots: int
    primary_failures: int
    accepted_shots: int
    accepted_logical_failures: int
    backend_low_confidence: int
    backend_nonconverged: int
    backend_invalid_correction: int
    backend_empty_model_unsatisfiable: int
    backend_error: int
    window_attempts: int

    @property
    def primary_ler(self) -> float:
        return self.primary_failures / self.attempted_shots


class ShardConflictError(RuntimeError):
    """An immutable scientific path already contains different bytes."""


_IDENTITY_FIELDS = (
    "schema_version",
    "experiment_id",
    "experiment_sha256",
    "config_id",
    "sample_set_id",
)
_COUNT_FIELDS = (
    "attempted_shots",
    "primary_failures",
    "accepted_shots",
    "accepted_logical_failures",
    "backend_low_confidence",
    "backend_nonconverged",
    "backend_invalid_correction",
    "backend_empty_model_unsatisfiable",
    "backend_error",
    "window_attempts",
)


def canonical_chunk_csv(rows) -> bytes:
    """Serialize chunk rows with a fixed header, ordering, and newline."""
    ordered = sorted(tuple(rows), key=lambda row: row.batch_index)
    stream = io.StringIO(newline="")
    names = [field.name for field in fields(ChunkResult)]
    writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in ordered)
    return stream.getvalue().encode("utf-8")


def read_chunk_csv(path) -> ChunkResult:
    with Path(path).open(newline="", encoding="utf-8") as source:
        row, = csv.DictReader(source)
    integer_fields = {
        "schema_version",
        "batch_index",
        "first_shot_index",
        "requested_shots",
        *_COUNT_FIELDS,
    }
    return ChunkResult(**{
        name: int(value) if name in integer_fields else value
        for name, value in row.items()
    })


def publish_immutable(destination, content: bytes) -> bool:
    """Publish once; return false only for an existing byte-identical shard."""
    if type(content) is not bytes:
        raise TypeError("scientific content must be bytes")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() == content:
                return False
            raise ShardConflictError(f"scientific shard conflict: {destination}")
        _fsync_directory(destination.parent)
        return True
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reduce_chunks(rows) -> ReducedResult:
    """Validate one configuration and sum integer evidence exactly."""
    ordered = sorted(tuple(rows), key=lambda row: row.batch_index)
    if not ordered:
        raise ValueError("at least one chunk is required")
    if len({row.batch_index for row in ordered}) != len(ordered):
        raise ValueError("duplicate batch index")
    first = ordered[0]
    for name in _IDENTITY_FIELDS:
        if any(getattr(row, name) != getattr(first, name) for row in ordered[1:]):
            raise ValueError(f"chunks disagree on {name}")
    next_shot = 0
    for expected_batch, row in enumerate(ordered):
        if row.batch_index != expected_batch:
            raise ValueError("batch indexes must be contiguous from zero")
        if row.first_shot_index != next_shot:
            raise ValueError("shot ranges must be contiguous from zero")
        next_shot += row.requested_shots
    totals = {
        name: sum(getattr(row, name) for row in ordered)
        for name in _COUNT_FIELDS
    }
    return ReducedResult(
        **{name: getattr(first, name) for name in _IDENTITY_FIELDS},
        **totals,
    )


def surface_plot_row(configuration, result: ReducedResult) -> dict:
    """Project exact surface-run counts into the plotting schema."""
    return {
        "schema_version": result.schema_version,
        "experiment_id": result.experiment_id,
        "experiment_sha256": result.experiment_sha256,
        "config_id": result.config_id,
        "sample_set_id": result.sample_set_id,
        "code_family": "rotated_surface",
        "distance": configuration["distance"],
        "rounds": configuration["rounds"],
        "physical_error_rate": configuration["physical_error_rate"],
        "basis": "z",
        "decoder": "mwpm",
        "shots": result.attempted_shots,
        "failures": result.primary_failures,
    }


def canonical_surface_plot_csv(row) -> bytes:
    stream = io.StringIO(newline="")
    names = tuple(row)
    writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode("utf-8")
