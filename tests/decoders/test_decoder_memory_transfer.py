"""Staging a job's input, and folding a window's boundary into it.

The staging is the decoder side's writer of what a unit reads: the
rounds it deposits at a landing, and the masked input a window's gate
hands it once the window's boundary is known. Both writes land in
storage this side owns, so this side makes them and books its own copy:
a copy is booked where it lands, by the name of the structure it landed
in (docs/explanation/data_path.md), and the destination takes what it is
handed before it acts (OMNeT++
src/sim/csimplemodule.cc:782-783). One input has
one writer, which is the memory's own rule (Helios 2301.08419 lines
632-640, decoder_memory.py rewrite).
"""

import dataclasses

import pytest

import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoder_memory_transfer as decoder_memory_transfer
import decsim.decoders.detection_events as detection_events
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import tests.declared_run as declared_run


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
        patch_ids=(0,),
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
    job.input_source_name = "weak syndrome buffer"
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


def test_a_staged_job_claims_its_rounds_before_it_reads_them():
    """A round retires once its last reader reads it, so the claim is first.

    A job priced after its rounds left the store would find a claim
    that went at retirement, and pay for a round another job paid for.
    """
    engine = engine_module.Engine()
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",)
    )
    placement = formation.SeatedFormation(None, settings)
    tier = detection_events.TierFormation(placement, "weak_decoder")
    staging = decoder_memory_transfer.DecoderInputStaging(
        None, engine, formation=tier
    )
    memory = decoder_memory.DecoderMemory("default", 0, None)
    resident = _job_with_rounds()
    resident.input_key = "window 7's request"
    memory.deposit(resident)
    reader = _job_with_rounds()
    reader.input_key = "window 7's request"

    staging.stage(reader, memory, lambda _: None)

    assert reader.detection_event_rounds == ((41, 0),)


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


def test_an_in_place_fold_on_a_tier_that_reads_in_place_still_stops():
    """The weak tier folds in place; the strong tier keeps no copy to fold."""
    reading_in_place = declared_run.switching_run(
        rounds=9, escalates=True, strong_copies_input=False
    )
    settings = reading_in_place.settings
    weak = dataclasses.replace(
        settings.weak_decoder, copies_boundary_fold=False
    )
    folding_in_place = dataclasses.replace(settings, weak_decoder=weak)
    machine = machine_module.Machine.build(folding_in_place, 0)

    with pytest.raises(AttributeError):
        machine.run()


def test_a_companion_that_joins_after_the_fold_reads_the_written_rounds():
    """The second forced-class solve of a window lands after the first wrote.

    It reads the written rounds and writes nothing, the bits the copy fold
    gives it too (cudaq-x sliding_window.cpp:287-293 masks each window
    once); a second write would fold the one boundary twice.
    """
    engine = engine_module.Engine()
    staging = _staging(engine)
    memory = decoder_memory.DecoderMemory("default", 0, None)
    first = _job_with_rounds()
    first.input_key = "window 7's request"
    first.decoder_input = memory.deposit(first)
    first.memory = memory
    masked = dataclasses.replace(first.decoder_input)
    staging.fold_in_place(first, masked)
    companion = _job_with_rounds()
    companion.input_key = "window 7's request"
    companion.decoder_input = memory.add_reader(companion)
    companion.memory = memory
    masked_again = dataclasses.replace(companion.decoder_input)

    staging.fold_in_place(companion, masked_again)

    assert companion.decoder_input is masked
    assert memory.input_of(companion) is masked
