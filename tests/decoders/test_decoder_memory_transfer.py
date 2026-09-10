"""Staging a job's input, and folding a window's boundary into it.

The staging is the decoder side's writer of what a unit reads: the
rounds it deposits at a landing, and the masked input a window's gate
hands it once the window's boundary is known. Both writes land in
storage this side owns, so this side makes them and books its own copy:
a copy is booked where it lands, by the name of the structure it landed
in (docs/explanation/data_path.md), and the destination takes what it is
handed before it acts (OMNeT++
tmp/resources/omnetpp/src/sim/csimplemodule.cc:782-783). One input has
one writer, which is the memory's own rule (Helios 2301.08419 lines
632-640, decoder_memory.py rewrite).
"""

import dataclasses

import pytest

import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoder_memory_transfer as decoder_memory_transfer
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records


class _CopyRecorder:
    """Every copy the staging books, as the ledger would hear it."""

    def __init__(self) -> None:
        self.made = []

    def copy_made(self, job, bits, source_name, target_name) -> None:
        del job
        self.made.append((bits, source_name, target_name))


def _fragment(round_index: int) -> round_records.RetainedSyndromeFragment:
    return round_records.RetainedSyndromeFragment(
        operation_id=41,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0, 1),
        size_bits=3,
        fragment_index=0,
    )


def _job_with_rounds() -> decoding_records.DecodeJob:
    payloads = [_fragment(0)]
    job = decoding_records.DecodeJob(
        operation_id=41,
        window_id=7,
        round_count=1,
        payloads=payloads,
        label="job(41, 7)",
    )
    job.input_source_name = "Buffer 0"
    return job


def _landed_in(memory) -> decoding_records.DecodeJob:
    """One job whose rounds sit in that unit's memory."""
    job = _job_with_rounds()
    job.decoder_input = memory.deposit(job)
    job.memory = memory
    return job


def _staging(engine) -> decoder_memory_transfer.DecoderInputStaging:
    """The staging alone: the fold sends nothing, so it needs no transport."""
    return decoder_memory_transfer.DecoderInputStaging(None, engine)


def test_a_folded_copy_is_booked_by_the_side_that_makes_it():
    engine = engine_module.Engine()
    staging = _staging(engine)
    copies = _CopyRecorder()
    staging.trace.copy_made.connect(copies.copy_made)
    memory = decoder_memory.DecoderMemory("default", 0, None)
    job = _landed_in(memory)
    landed = job.decoder_input
    masked = dataclasses.replace(landed)

    staging.fold_into_a_copy(job, masked)

    assert job.decoder_input is masked
    assert memory.input_of(job) is landed
    assert copies.made == [(3, "unit default#0 memory", "masked view")]


def test_a_fold_in_place_is_written_by_the_memory_that_holds_the_input():
    engine = engine_module.Engine()
    staging = _staging(engine)
    memory = decoder_memory.DecoderMemory("default", 0, None)
    job = _landed_in(memory)
    masked = dataclasses.replace(job.decoder_input)

    staging.fold_in_place(job, masked)

    assert memory.input_of(job) is masked
    assert job.decoder_input is masked


def test_a_fold_in_place_with_no_unit_memory_is_refused():
    """A tier that reads its input in place has nothing to write into."""
    engine = engine_module.Engine()
    staging = _staging(engine)
    job = _job_with_rounds()
    landed = decoder_memory.materialize_decoder_input(job)

    with pytest.raises(RuntimeError, match="in_place needs the unit's own"):
        staging.fold_in_place(job, landed)
