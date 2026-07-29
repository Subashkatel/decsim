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
    campaign_id: str
    manifest_sha256: str
    config_id: str
    sample_group_id: str
    sampling_domain: str
    seed_protocol_version: str
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
    campaign_id: str
    manifest_sha256: str
    config_id: str
    sample_group_id: str
    sampling_domain: str
    seed_protocol_version: str
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
    "campaign_id",
    "manifest_sha256",
    "config_id",
    "sample_group_id",
    "sampling_domain",
    "seed_protocol_version",
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


def _validate_chunk(row: ChunkResult) -> None:
    if not isinstance(row, ChunkResult):
        raise TypeError("rows must contain ChunkResult values")
    for name in _IDENTITY_FIELDS[1:] + ("sample_batch_sha256",):
        if type(getattr(row, name)) is not str or not getattr(row, name):
            raise ValueError(f"{name} must be a nonempty string")
    integer_fields = (
        "schema_version",
        "batch_index",
        "first_shot_index",
        "requested_shots",
    ) + _COUNT_FIELDS
    for name in integer_fields:
        value = getattr(row, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if row.schema_version != 1:
        raise ValueError("schema_version must be 1")
    if row.requested_shots == 0 or row.attempted_shots != row.requested_shots:
        raise ValueError("attempted_shots must equal positive requested_shots")
    if row.primary_failures > row.attempted_shots:
        raise ValueError("primary_failures cannot exceed attempted_shots")
    if row.accepted_shots > row.attempted_shots:
        raise ValueError("accepted_shots cannot exceed attempted_shots")
    if row.accepted_logical_failures > row.accepted_shots:
        raise ValueError(
            "accepted_logical_failures cannot exceed accepted_shots"
        )


def canonical_chunk_csv(rows) -> bytes:
    """Serialize chunk rows with a fixed header, ordering, and newline."""
    ordered = sorted(tuple(rows), key=lambda row: row.batch_index)
    for row in ordered:
        _validate_chunk(row)
    stream = io.StringIO(newline="")
    names = [field.name for field in fields(ChunkResult)]
    writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in ordered)
    return stream.getvalue().encode("utf-8")


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
    for row in ordered:
        _validate_chunk(row)
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


def validate_paired_chunks(left_rows, right_rows) -> None:
    """Require two offline arms to contain the same raw sampled shots."""
    left = sorted(tuple(left_rows), key=lambda row: row.batch_index)
    right = sorted(tuple(right_rows), key=lambda row: row.batch_index)
    left_summary = reduce_chunks(left)
    right_summary = reduce_chunks(right)
    if (
        left_summary.sampling_domain != "offline_stim_batch_v1"
        or right_summary.sampling_domain != "offline_stim_batch_v1"
    ):
        raise ValueError("paired chunks require the offline sampling domain")
    for name in (
        "schema_version",
        "campaign_id",
        "manifest_sha256",
        "sample_group_id",
        "seed_protocol_version",
    ):
        if getattr(left_summary, name) != getattr(right_summary, name):
            raise ValueError(f"paired chunks disagree on {name}")
    if len(left) != len(right):
        raise ValueError("paired chunks have different batch counts")
    for left_row, right_row in zip(left, right):
        left_range = (
            left_row.batch_index,
            left_row.first_shot_index,
            left_row.requested_shots,
        )
        right_range = (
            right_row.batch_index,
            right_row.first_shot_index,
            right_row.requested_shots,
        )
        if left_range != right_range:
            raise ValueError("paired chunks have different shot ranges")
        if left_row.sample_batch_sha256 != right_row.sample_batch_sha256:
            raise ValueError("paired chunks have different raw sample digest")
