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


def _unit(capacity_rounds=None):
    memory = decoder_memory.DecoderMemory("default", 0, capacity_rounds)
    return decoder_unit.DecoderUnit("default", 0, memory)


def _job(label, rounds=1):
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, n_rounds=rounds, label=label
    )


def _demand(job):
    return job.n_rounds


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
    # A front-built memory config is a boundary: a unit that holds zero
    # rounds can serve nothing, so the mistake is caught before a run
    # rather than at the first deposit.

    with pytest.raises(
        ValueError, match="pool 'default' needs a positive round capacity"
    ):
        decoder_memory.DecoderMemoryConfig({"default": 0})
