"""The run flight recorder (cycle/event ledger).

`event_ledger(completed)` assembles one causal record per
hardware-significant transition from the owners' own records: the
packing stage's round events, syndrome buffer 1's stored log, the
window stamps, the frame records, and the release times. Every event
carries its causal predecessor; `check()` proves the accounting: every
emitted window-input round reaches exactly one terminal state and no
event precedes its cause. Ticks below are the declared fabric's exact
arithmetic (round r emitted at r us; qc 2 + binary 3 + cwb 4 publishes
at r+9; csb 7 stores at r+12).
"""

import pytest

from decsim.config import us
from decsim.observe.run_views import LedgerEvent, RunLedgerView, event_ledger


def _kinds_and_ticks(chain):
    return [(event.kind, event.tick) for event in chain]


def test_weak_round_chain_is_exact(fabric):
    """Round 3 of a weak run: emitted 3, controller binary 8, packed and
    CWB-sent 8, published in Buffer 0 at 12, terminal."""
    ledger = event_ledger(fabric["weak_only_run"](rounds=6))
    chain = ledger.chain(op=1, round=3)

    assert _kinds_and_ticks(chain) == [
        ("EMITTED", us(3)), ("BINARY_AVAILABLE", us(8)), ("PACKED", us(8)),
        ("CWB_SENT", us(8)), ("PUBLISHED", us(12))]
    for earlier, later in zip(chain, chain[1:]):
        assert later.prev_event_id == earlier.event_id
    assert chain[-1].status == "terminal"


def test_weak_window_chain_is_exact(fabric):
    """The window chain: data complete 15, queued and assigned 15, done
    30, frame accepted 32, committed 33; its cause is round 6's
    publication event."""
    ledger = event_ledger(fabric["weak_only_run"](rounds=6))
    chain = ledger.chain(op=1, window=0)

    assert _kinds_and_ticks(chain) == [
        ("WINDOW_DATA_COMPLETE", us(15)), ("DECODE_QUEUED", us(15)),
        ("UNIT_ASSIGNED", us(15)), ("DECODE_DONE", us(30)),
        ("FRAME_ACCEPTED", us(32)), ("FRAME_COMMITTED", us(33))]
    events_by_id = {event.event_id: event for event in ledger.events}
    cause = events_by_id[chain[0].prev_event_id]
    assert (cause.kind, cause.round, cause.tick) == ("PUBLISHED", 6, us(15))


def test_every_emitted_round_reaches_exactly_one_terminal(fabric):
    """Conservation over the whole run: six emitted rounds, six
    publications, and the check passes."""
    ledger = event_ledger(fabric["weak_only_run"](rounds=6))
    ledger.check()

    emitted = [event for event in ledger.events if event.kind == "EMITTED"]
    published = [event for event in ledger.events if event.kind == "PUBLISHED"]
    assert len(emitted) == 6
    assert len(published) == 6


def test_strong_run_records_the_room_store_landing(fabric):
    """Strong-primary: the dual write lands round r in syndrome buffer 1
    at r+12 (csb 7 after binary availability), and the window chain runs
    18 / 54 / 58 / 59 on the strong tier."""
    ledger = event_ledger(fabric["strong_only_run"](rounds=6))
    ledger.check()

    assert _kinds_and_ticks(ledger.chain(op=1, round=3)) == [
        ("EMITTED", us(3)), ("BINARY_AVAILABLE", us(8)), ("PACKED", us(8)),
        ("CWB_SENT", us(8)), ("PUBLISHED", us(12)), ("STORED_SB1", us(15))]
    window_chain = ledger.chain(op=1, window=0)
    assert _kinds_and_ticks(window_chain) == [
        ("WINDOW_DATA_COMPLETE", us(18)), ("DECODE_QUEUED", us(18)),
        ("UNIT_ASSIGNED", us(18)), ("DECODE_DONE", us(54)),
        ("FRAME_ACCEPTED", us(58)), ("FRAME_COMMITTED", us(59))]
    assert window_chain[-1].route == "strong"


def test_switching_run_ledger_checks(fabric):
    """A full escalation run assembles and passes the accounting check."""
    ledger = event_ledger(
        fabric["switching_run"](rounds=6, escalation_probability=1.0))
    ledger.check()


def test_release_links_to_the_blocking_operations_commit(fabric):
    """A blocked operation's release decision is caused by the BLOCKING
    operation's final commit and costs exactly oc + cq."""
    completed = fabric["weak_only_run"](
        rounds=6, ops=[fabric["memory_op"](1),
                       fabric["memory_op"](2, blocked_by=1)])
    ledger = event_ledger(completed)
    ledger.check()

    (release,) = [event for event in ledger.events
                  if event.kind == "DECODE_RELEASED"]
    events_by_id = {event.event_id: event for event in ledger.events}
    cause = events_by_id[release.prev_event_id]
    oc_cq = us(fabric["DECLARED_US"]["oc"] + fabric["DECLARED_US"]["cq"])
    assert release.op == 2
    assert (cause.kind, cause.op) == ("FRAME_COMMITTED", 1)
    assert release.tick == cause.tick + oc_cq


def test_check_detects_an_effect_before_its_cause():
    """The checker has teeth: an event stamped before its cause fails."""
    cause = LedgerEvent(event_id=0, kind="EMITTED", tick=us(5), op=1, round=1)
    effect = LedgerEvent(event_id=1, kind="PUBLISHED", tick=us(4), op=1,
                         round=1, prev_event_id=0, status="terminal")
    with pytest.raises(RuntimeError, match="precedes its cause"):
        RunLedgerView(events=(cause, effect)).check()


def test_check_detects_a_disappeared_round():
    """A round that was emitted but never reached a terminal state fails
    the conservation check (the packet-disappearance question)."""
    orphan = LedgerEvent(event_id=0, kind="EMITTED", tick=us(1), op=1, round=1)
    with pytest.raises(RuntimeError, match="terminal states"):
        RunLedgerView(events=(orphan,)).check()
