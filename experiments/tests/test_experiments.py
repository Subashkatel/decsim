from dataclasses import replace

import pytest

from experiments.results import (
    ChunkResult,
    ShardConflictError,
    canonical_chunk_csv,
    publish_immutable,
    reduce_chunks,
    validate_paired_chunks,
)


def _chunk(batch_index=0, first_shot_index=0, requested_shots=3, **changes):
    values = dict(
        schema_version=1,
        campaign_id="campaign",
        manifest_sha256="a" * 64,
        config_id="config",
        sample_group_id="samples",
        sampling_domain="offline_stim_batch_v1",
        seed_protocol_version="decsim_derive_component_seed_v1",
        sample_batch_sha256=f"{batch_index + 1:064x}",
        batch_index=batch_index,
        first_shot_index=first_shot_index,
        requested_shots=requested_shots,
        attempted_shots=requested_shots,
        primary_failures=1,
        accepted_shots=requested_shots - 1,
        accepted_logical_failures=0,
        backend_low_confidence=1,
        backend_nonconverged=0,
        backend_invalid_correction=0,
        backend_empty_model_unsatisfiable=0,
        backend_error=0,
        window_attempts=requested_shots * 2,
    )
    values.update(changes)
    return ChunkResult(**values)


def test_chunk_csv_is_canonical_and_contains_no_operational_fields():
    first = _chunk()
    encoded = canonical_chunk_csv([first])

    assert encoded == canonical_chunk_csv([first])
    assert encoded.endswith(b"\n")
    assert b"wall" not in encoded
    assert b"hostname" not in encoded
    assert b"rss" not in encoded


def test_immutable_publication_is_idempotent_but_never_clobbers(tmp_path):
    destination = tmp_path / "chunk.csv"
    original = canonical_chunk_csv([_chunk()])
    conflicting = canonical_chunk_csv([replace(_chunk(), primary_failures=2)])

    assert publish_immutable(destination, original) is True
    assert publish_immutable(destination, original) is False
    with pytest.raises(ShardConflictError):
        publish_immutable(destination, conflicting)
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_reduction_sums_integer_counts_and_requires_exact_ranges():
    rows = [
        _chunk(),
        _chunk(
            batch_index=1,
            first_shot_index=3,
            requested_shots=2,
            attempted_shots=2,
            primary_failures=2,
            accepted_shots=1,
            accepted_logical_failures=1,
            backend_low_confidence=0,
            backend_nonconverged=1,
            window_attempts=5,
        ),
    ]

    summary = reduce_chunks(reversed(rows))

    assert summary.attempted_shots == 5
    assert summary.primary_failures == 3
    assert summary.accepted_shots == 3
    assert summary.accepted_logical_failures == 1
    assert summary.window_attempts == 11
    assert summary.primary_ler == 3 / 5

    with pytest.raises(ValueError, match="duplicate batch"):
        reduce_chunks([rows[0], rows[0]])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_chunks([rows[0], replace(rows[1], first_shot_index=4)])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_chunks([rows[0], replace(rows[1], first_shot_index=2)])


def test_reduction_rejects_impossible_counts():
    with pytest.raises(ValueError, match="primary_failures"):
        reduce_chunks([_chunk(primary_failures=4)])
    with pytest.raises(ValueError, match="accepted_logical_failures"):
        reduce_chunks([_chunk(accepted_shots=1, accepted_logical_failures=2)])


def test_paired_chunks_require_the_same_raw_detector_and_truth_digest():
    left = [_chunk()]
    right = [replace(_chunk(), config_id="other-decoder")]
    validate_paired_chunks(left, right)

    with pytest.raises(ValueError, match="raw sample digest"):
        validate_paired_chunks(
            left,
            [replace(right[0], sample_batch_sha256="f" * 64)],
        )
    with pytest.raises(ValueError, match="sampling domain"):
        validate_paired_chunks(
            left,
            [replace(right[0], sampling_domain="event_runspec_device_v1")],
        )
