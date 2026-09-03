"""Syndrome buffer 1: the csb crossing, the arrival gate, and the
refcounted room-side lifetime (ports of the prototype's verified suite;
the shared slot/hold semantics are SyndromeBuffer's and are pinned by the
recorded scenarios)."""

import pytest

from decsim.config import microseconds_to_ticks
from decsim.engine import Engine
from decsim.links.fabric import LinkFabric
from decsim.links.link_profiles import (logical_reference_profile,
                                        with_controller_to_strong_buffer_path)
from decsim.message import (RetainedSyndromeFragment, SyndromeRoundPacket,
                            TransferAttribution)
from decsim.observe.link_traffic import TrafficLedger
from decsim.syndrome_buffer.syndrome_buffer_1 import SyndromeBuffer1


def _packet(round_index, *, operation_id=1, bits=(1, 0, 1)):
    fragment = RetainedSyndromeFragment(
        operation_id=operation_id, patch_id=0, round_index=round_index,
        bits=tuple(bits), size_bits=len(bits), fragment_index=0)
    return SyndromeRoundPacket(operation_id, round_index, (fragment,))


def _write(sb1, packet):
    sb1.write(packet, packet_bits=packet.fragments[0].size_bits,
              attribution=TransferAttribution(
                  operation_id=packet.operation_id, patch_ids=(0,),
                  window_id=None, first_round=packet.round_index,
                  last_round=packet.round_index))


def _free_fabric(engine):
    return LinkFabric(logical_reference_profile(), engine)


def _priced_fabric(engine, latency_us=0.5):
    """The fabric with a priced csb hop, and the ledger that counts it."""
    settings = with_controller_to_strong_buffer_path(
        logical_reference_profile(), latency_microseconds=latency_us,
        aggregate_bits_per_microsecond=None, source="test csb")
    ledger = TrafficLedger(settings)
    return LinkFabric(settings, engine, ledger), ledger


def _csb_transfer_count(ledger):
    return sum(1 for record in ledger.snapshot().transfers
               if record.path.value == "controller_to_strong_buffer")


def _free_buffer(**arguments):
    engine = Engine()
    return SyndromeBuffer1(engine, _free_fabric(engine), **arguments)


def test_every_round_crosses_once_and_bits_are_counted():
    sb1 = _free_buffer()
    sb1.register_hold("reader", [(1, 1), (1, 2)])
    _write(sb1, _packet(1))
    _write(sb1, _packet(2))
    assert sb1.copied_bits_total == 6
    with pytest.raises(ValueError, match="already written"):
        _write(sb1, _packet(1))
    assert sb1.copied_bits_total == 6


def test_priced_csb_gates_arrival():
    engine = Engine()
    links, ledger = _priced_fabric(engine, 0.5)
    sb1 = SyndromeBuffer1(engine, links)
    sb1.register_hold("reader", [(1, 1)])
    _write(sb1, _packet(1))
    # still in flight over the csb: reads refuse loudly
    assert sb1.retained_fragments((1, 1)) is None
    with pytest.raises(RuntimeError, match="not stored in syndrome buffer 1"):
        sb1.check_rounds_stored([(1, 1)])
    engine.run()
    assert sb1.publication_tick((1, 1)) == microseconds_to_ticks(0.5)
    sb1.check_rounds_stored([(1, 1)])
    assert _csb_transfer_count(ledger) == 1


def test_refused_write_leaves_no_trace():
    engine = Engine()
    links, ledger = _priced_fabric(engine, 0.5)
    sb1 = SyndromeBuffer1(engine, links, capacity_rounds=1)
    sb1.register_hold("reader", [(1, 1), (1, 2)])
    _write(sb1, _packet(1))                  # in flight, counts against capacity
    with pytest.raises(RuntimeError, match="over capacity"):
        _write(sb1, _packet(2))
    # the refusal happened before the send and before any store
    assert sb1.copied_bits_total == 3
    engine.run()
    assert _csb_transfer_count(ledger) == 1
    assert sb1.retained_fragments((1, 1)) is not None


def test_reader_resolved_while_round_in_flight_drops_it_on_landing():
    engine = Engine()
    links, _ledger = _priced_fabric(engine, 0.5)
    sb1 = SyndromeBuffer1(engine, links)
    sb1.register_hold("reader", [(1, 1)])
    _write(sb1, _packet(1))
    sb1.release_hold("reader")               # nobody can need it any more
    engine.run()
    assert sb1.retained_fragments((1, 1)) is None
    sb1.check_settled()


def test_unheld_round_is_dropped_on_arrival_but_still_priced():
    sb1 = _free_buffer()
    _write(sb1, _packet(1))                  # no registered consumer
    assert sb1.retained_fragments((1, 1)) is None
    assert sb1.copied_bits_total == 3
    sb1.check_settled()


def test_late_write_of_a_released_round_raises():
    sb1 = _free_buffer()
    sb1.register_hold("reader", [(1, 1)])
    _write(sb1, _packet(1))
    sb1.release_hold("reader")
    with pytest.raises(ValueError, match="already"):
        _write(sb1, _packet(1))


def test_refcounted_release_order():
    sb1 = _free_buffer()
    sb1.register_hold("first", [(1, 1)])
    sb1.register_hold("second", [(1, 1)])
    _write(sb1, _packet(1))
    sb1.release_hold("first")
    assert sb1.retained_fragments((1, 1)) is not None
    sb1.release_hold("second")
    assert sb1.retained_fragments((1, 1)) is None
    sb1.check_settled()


def test_settled_reports_rounds_expected_but_never_written():
    sb1 = _free_buffer()
    sb1.register_hold("reader", [(1, 5)])
    with pytest.raises(RuntimeError, match="unresolved holds"):
        sb1.check_settled()


def test_arrival_counter_and_stored_signal():
    stored = []
    sb1 = _free_buffer(on_round_stored=stored.append)
    sb1.register_hold("reader", [(1, 1), (1, 2)])
    _write(sb1, _packet(1))
    _write(sb1, _packet(2))
    assert sb1.rounds_arrived == {1: 2}
    assert stored == [1, 1]
