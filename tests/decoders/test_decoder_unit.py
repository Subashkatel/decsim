"""One unit's occupancy: two input slots, one compute, in-order flights.

Smith 1982 decoupled access-execute (rowD2): the second slot holds the
next window's input while the first computes. Hennessy and Patterson
App. C (rowD4): a pipelined unit retires its flights in issue order and
a full pipeline stalls the intake until a flight retires.
"""

import pytest

import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoder_unit as decoder_unit
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records


def _unit(capacity_rounds=None):
    memory = decoder_memory.DecoderMemory("default", 0, capacity_rounds)
    return decoder_unit.DecoderUnit("default", 0, memory)


def _job(label, rounds=1):
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=rounds, label=label
    )


def _demand(job):
    return job.round_count


def _payload_demand(job):
    return decoding_records.distinct_round_count(job.payloads)


def _carrying_job(label, rounds):
    payloads = []
    for round_index in range(rounds):
        fragment = round_records.RetainedSyndromeFragment(
            operation_id=1,
            patch_id="patch-0",
            round_index=round_index,
            bits=(0, 1),
            size_bits=2,
            fragment_index=0,
        )
        payloads.append(fragment)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=rounds,
        payloads=payloads,
        label=label,
    )


def test_a_unit_holds_two_inputs_and_one_compute():
    unit = _unit()
    first = _job("first")
    second = _job("second")
    unit.admit(first)
    unit.claim_compute(first)
    assert unit.has_room(second, 2, _demand) is True
    unit.admit(second)
    third = _job("third")
    assert unit.has_room(third, 2, _demand) is False
    assert unit.holder is first
    assert first.unit is unit


def test_a_second_resident_joins_only_when_both_inputs_fit_the_memory():
    unit = _unit(capacity_rounds=5)
    first = _job("first", rounds=3)
    unit.admit(first)
    small = _job("small", rounds=2)
    assert unit.has_room(small, 2, _demand) is True
    large = _job("large", rounds=3)
    assert unit.has_room(large, 2, _demand) is False


def test_a_second_window_waits_while_a_landed_one_fills_the_memory():
    """A full memory refuses the next window instead of raising later.

    gem5 src/mem/cache/base.hh lines 121, 266 and 271 at commit cbc94c1:
    with no miss-status register free the cache sets Blocked_NoMSHRs and
    the response port refuses the request until one frees; nothing is
    raised.
    """
    memory = decoder_memory.DecoderMemory("default", 0, capacity_rounds=3)
    unit = decoder_unit.DecoderUnit("default", 0, memory)
    landed = _carrying_job("landed", 3)
    unit.admit(landed)
    landed.decoder_input = memory.deposit(landed)
    landed.payloads = []

    second = _carrying_job("second", 3)

    assert unit.has_room(second, 2, _payload_demand) is False


def test_the_compute_goes_to_the_oldest_landed_resident_not_parked():
    unit = _unit()
    parked = _job("parked")
    parked.input_landed = True
    parked.is_parked = True
    ready = _job("ready")
    ready.input_landed = True
    unit.admit(parked)
    unit.admit(ready)
    assert unit.oldest_landed_resident_ready_to_start() is ready


def test_a_full_pipeline_stalls_until_a_flight_retires():
    unit = _unit()
    first = _job("first")
    second = _job("second")
    unit.add_flight(first, 40)
    unit.pipeline.intake_job = None
    unit.add_flight(second, 40)
    unit.claim_compute(second)
    unit.stall(second, 2)
    assert unit.lift_stall() is None
    assert unit.take_flight(first) is True
    assert unit.lift_stall() is second
    assert unit.flight_labels() == ["second"]


def test_the_residents_describe_their_phase():
    unit = _unit()
    computing = _job("w0", rounds=3)
    computing.input_landed = True
    computing.service_started = True
    landing = _job("w1", rounds=2)
    unit.admit(computing)
    unit.claim_compute(computing)
    unit.admit(landing)
    assert unit.describe_residents() == (
        "w0 computing, 3 rounds; w1 capturing, 2 rounds"
    )


def test_a_memory_config_with_a_zero_capacity_is_refused_at_construction():
    # A memory config from the experiments layer is a boundary: a unit
    # that holds zero rounds can serve nothing, so the mistake is caught
    # before a run
    # rather than at the first deposit.

    with pytest.raises(
        ValueError, match="pool 'default' needs a positive round capacity"
    ):
        decoder_memory.DecoderMemoryConfig({"default": 0})


def test_the_output_slot_holds_one_finished_result_until_it_is_taken():
    """AFS keeps the finished log in the unit until it is read (833-840)."""
    unit = _unit()
    completion = object()
    assert unit.output_windows() == []
    unit.hold_output((1, 0), completion)
    assert unit.output_windows() == [(1, 0)]
    assert unit.take_output((1, 0)) is completion
    assert unit.output_windows() == []
    assert unit.take_output((1, 0)) is None


def test_a_second_result_for_one_destination_is_refused():
    """A destination window has at most one unconsumed strong result."""
    unit = _unit()
    first = object()
    second = object()
    unit.hold_output((1, 0), first)
    with pytest.raises(RuntimeError, match="already holds a finished result"):
        unit.hold_output((1, 0), second)
