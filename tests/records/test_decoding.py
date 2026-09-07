"""The decode records of decsim/records/decoding.py.

A job is priced and admitted for the distinct rounds it carries, which is
how a sliding-window decoder's work scales (Skoric et al. 2209.08552, tau_W
over n_W); a result leaves its optional fields unset when the decoder is
timing-only.
"""

import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records


def make_payload(round_index, size_bits=100):
    return round_records.RetainedSyndromeFragment(
        operation_id=3,
        patch_id="patch-a",
        round_index=round_index,
        bits=(0, 1),
        size_bits=size_bits,
        fragment_index=0,
    )


def test_distinct_round_count_counts_one_round_once():
    """Two payloads of one round are one round of decoder work."""
    payloads = (make_payload(1), make_payload(1), make_payload(2))
    assert decoding_records.distinct_round_count(payloads) == 2


def test_distinct_round_count_separates_the_rounds_of_two_operations():
    """A round index is only distinct within its own operation."""
    other_operation = round_records.RetainedSyndromeFragment(
        operation_id=4,
        patch_id="patch-a",
        round_index=1,
        bits=(0, 1),
        size_bits=100,
        fragment_index=0,
    )
    payloads = (make_payload(1), other_operation)
    assert decoding_records.distinct_round_count(payloads) == 2


def test_a_jobs_payload_bits_add_up():
    """The job's bits are the sum of its payloads' sizes."""
    payloads = [make_payload(1), make_payload(2, size_bits=200)]
    job = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        n_rounds=2,
        payloads=payloads,
    )
    assert job.payload_bits() == 300


def test_one_payload_of_unknown_size_leaves_the_job_size_unknown():
    """A single unsized payload makes the whole job's size unknown."""
    payloads = [make_payload(1), make_payload(2, size_bits=None)]
    job = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        n_rounds=2,
        payloads=payloads,
    )
    assert job.payload_bits() is None


def test_a_job_with_no_payloads_carries_no_bits():
    """An empty job is zero bits, not an unknown size."""
    job = decoding_records.DecodeJob(operation_id=1, window_id=0, n_rounds=0)
    assert job.payload_bits() == 0


def test_decode_result_supports_timing_only_and_richer_results():
    """A result leaves the correction unset until a decoder fills it."""
    timing_only = decoding_records.DecodeResult(operation_id=4, window_id=2)
    source = object()
    soft_output = decoding_records.SoftOutput(gap=1, source=source)
    boundary_data = object()
    rich = decoding_records.DecodeResult(
        operation_id=4,
        window_id=2,
        correction="X",
        logical_observables=(1, 0),
        soft_output=soft_output,
        boundary_defects={3: 1},
        boundary_data=boundary_data,
    )
    assert timing_only.correction is None
    assert timing_only.logical_observables is None
    assert rich.logical_observables == (1, 0)
    assert rich.soft_output is soft_output
    assert rich.boundary_defects == {3: 1}
    assert rich.boundary_data is boundary_data


def test_a_service_key_is_the_run_wide_ordinal_of_one_batch():
    """One service is one batch of requests, identified by its ordinal."""
    service_key = decoding_records.DecoderServiceKey(8)
    assert service_key.run_sequence == 8
