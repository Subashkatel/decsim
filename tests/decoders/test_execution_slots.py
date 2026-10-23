"""A device's execution slots (decoders/execution_slots.py).

The referent is gem5's FUPool: a fixed count of units shared by every
thread, a request with none free waiting its turn, oldest first
(src/cpu/o3/fu_pool.cc:165-190). Here the count is a device's slots and
the requests are the decodes of every pool on it.
"""

import pytest

import decsim.decoders.execution_slots as execution_slots
import decsim.engine as engine_module


class Starts:
    """The decodes a device started late, in order, and when."""

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.started: list = []

    def start(self, job) -> None:
        self.started.append((job, self.engine.now))


def _device(slot_count: int) -> tuple:
    engine = engine_module.Engine()
    settings = execution_slots.ExecutionSlots.Settings(slot_count)
    slots = settings.build(engine)
    return engine, slots, Starts(engine)


def test_a_decode_takes_a_free_slot_at_once():
    _engine, slots, starts = _device(2)

    first = slots.take("first", starts.start)
    second = slots.take("second", starts.start)

    assert (first, second) == (True, True)
    assert slots.running == ["first", "second"]
    assert starts.started == []


def test_a_decode_with_no_free_slot_starts_when_one_is_given_back():
    engine, slots, starts = _device(1)
    slots.take("first", starts.start)
    taken = slots.take("second", starts.start)
    engine.schedule(7, lambda: slots.give_back("first"))

    engine.run()

    assert taken is False
    assert starts.started == [("second", 7)]
    assert slots.running == ["second"]


def test_waiting_decodes_start_oldest_first_whichever_pool_asked():
    engine, slots, starts = _device(1)
    slots.take("weak_first", starts.start)
    slots.take("fpga_b_second", starts.start)
    slots.take("weak_third", starts.start)
    engine.schedule(3, lambda: slots.give_back("weak_first"))
    engine.schedule(5, lambda: slots.give_back("fpga_b_second"))

    engine.run()

    assert starts.started == [("fpga_b_second", 3), ("weak_third", 5)]


def test_a_decode_that_started_second_gives_back_its_own_slot():
    """Units free in any order, so a slot is found by its decode."""
    engine, slots, starts = _device(2)
    slots.take("first", starts.start)
    slots.take("second", starts.start)
    slots.take("waiting", starts.start)
    engine.schedule(4, lambda: slots.give_back("second"))

    engine.run()

    assert starts.started == [("waiting", 4)]
    assert slots.running == ["first", "waiting"]


def test_a_withdrawn_decode_never_starts():
    engine, slots, starts = _device(1)
    slots.take("first", starts.start)
    slots.take("withdrawn", starts.start)
    slots.take("kept", starts.start)
    slots.withdraw("withdrawn")
    engine.schedule(2, lambda: slots.give_back("first"))

    engine.run()

    assert starts.started == [("kept", 2)]


def test_a_run_ending_with_a_decode_waiting_for_a_slot_is_a_contract_break():
    _engine, slots, starts = _device(1)
    slots.take("first", starts.start)
    slots.give_back("first")
    slots.take("second", starts.start)
    slots.take("stranded", starts.start)

    with pytest.raises(RuntimeError, match="1 decodes in device slots"):
        slots.check_settled()


def test_a_device_with_no_slots_is_refused():
    with pytest.raises(ValueError, match="slot_count is 0"):
        execution_slots.ExecutionSlots.Settings(0)


@pytest.mark.parametrize("slot_count", [1.5, float("nan"), True])
def test_a_slot_count_that_is_no_whole_number_is_refused(slot_count):
    """1.5 would grant two slots, NaN none, and True stands for no count."""
    with pytest.raises(ValueError, match="a whole number of decodes"):
        execution_slots.ExecutionSlots.Settings(slot_count)
