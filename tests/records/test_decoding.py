"""The decode records of decsim/records/decoding.py.

A job is priced and admitted for the distinct rounds it carries, which is
how a sliding-window decoder's work scales (Skoric et al. 2209.08552, tau_W
over n_W), and a job with one unsized payload has no known size.
"""

import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records


def make_payload(round_index, size_bits=100):
    return round_records.RetainedSyndromeFragment(
        operation_id=3,
        patch_ids=("patch-a",),
        round_index=round_index,
        bits=(0, 1),
        size_bits=size_bits,
        fragment_index=0,
    )


def test_distinct_round_count_separates_the_rounds_of_two_operations():
    """A round index is only distinct within its own operation."""
    other_operation = round_records.RetainedSyndromeFragment(
        operation_id=4,
        patch_ids=("patch-a",),
        round_index=1,
        bits=(0, 1),
        size_bits=100,
        fragment_index=0,
    )
    payloads = (make_payload(1), other_operation)
    assert decoding_records.distinct_round_count(payloads) == 2


def test_one_payload_of_unknown_size_leaves_the_job_size_unknown():
    """A single unsized payload makes the whole job's size unknown."""
    payloads = [make_payload(1), make_payload(2, size_bits=None)]
    job = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=2,
        payloads=payloads,
    )
    assert job.payload_bits() is None
