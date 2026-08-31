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


def test_dropped_round_is_an_accounted_terminal_state():
    """LINK-004 whole-run accounting including the drop path: with
    DROP_ROUND policy and a one-round Buffer 0 that nobody drains, the
    second round is dropped; the ledger records DROPPED as its terminal
    state and the conservation check passes because the loss is
    accounted, not silent."""
    from types import SimpleNamespace

    from decsim.controller.syndrome_packing import (PackingOverflowPolicy,
                                                    SyndromePacking,
                                                    SyndromePackingPolicy)
    from decsim.engine import Engine
    from decsim.message import QPUReadout, WINDOW_INPUT_ROUTE
    from decsim.syndrome_buffer.syndrome_buffer import SyndromeBuffer

    engine = Engine(verbose=False)
    packing = SyndromePacking(
        engine, t_pack=0, packing_context_capacity=None,
        window_input_receiver=SimpleNamespace(
            accept_window_input=lambda packet: True),
        feedback_memory_receiver=None,
        syndrome_buffer=SyndromeBuffer(capacity=1),
        policy=SyndromePackingPolicy(
            overflow=PackingOverflowPolicy.DROP_ROUND))
    for round_index in (1, 2):
        packing.relay_qpu_readout(
            QPUReadout(1, 0, round_index, size_bits=24),
            WINDOW_INPUT_ROUTE, processing_ticks=0)
    engine.run()

    completed = SimpleNamespace(
        syndrome_packing=packing, syndrome_buffer_1=None,
        window_manager=SimpleNamespace(windows={}), pauli_frame=None,
        execution_runtime=SimpleNamespace(decode_release_time={},
                                          operations={}))
    ledger = event_ledger(completed)
    ledger.check()

    terminals = {(event.round, event.kind) for event in ledger.events
                 if event.status == "terminal"}
    assert terminals == {(1, "PUBLISHED"), (2, "DROPPED")}
    assert packing.packing_drops == 1


def test_reassembly_context_drop_is_an_accounted_terminal_state():
    """LINK-004 applies before packing as well as at Buffer 0 admission.

    Hold one fragmented round open in the controller's sole reassembly slot,
    then deliver another round.  DROP_ROUND must leave a terminal ledger row
    for the rejected round instead of changing only an internal counter.
    """
    from types import SimpleNamespace

    from decsim.controller.syndrome_packing import (PackingOverflowPolicy,
                                                    SyndromePacking,
                                                    SyndromePackingPolicy)
    from decsim.engine import Engine
    from decsim.message import QPUReadout, WINDOW_INPUT_ROUTE
    from decsim.syndrome_buffer.syndrome_buffer import SyndromeBuffer

    engine = Engine(verbose=False)
    packing = SyndromePacking(
        engine, t_pack=0, packing_context_capacity=1,
        window_input_receiver=SimpleNamespace(
            accept_window_input=lambda packet: True),
        feedback_memory_receiver=None,
        syndrome_buffer=SyndromeBuffer(capacity=None),
        policy=SyndromePackingPolicy(
            overflow=PackingOverflowPolicy.DROP_ROUND))
    # Event insertion order makes round 1 occupy the context, round 2 lose
    # admission, and the final fragment then complete round 1.
    for payload in (
        QPUReadout(1, 0, 1, n_fragments=2, fragment_index=0, size_bits=12),
        QPUReadout(1, 0, 2, size_bits=24),
        QPUReadout(1, 0, 1, n_fragments=2, fragment_index=1, size_bits=12),
    ):
        packing.relay_qpu_readout(payload, WINDOW_INPUT_ROUTE,
                                  processing_ticks=0)
    engine.run()

    completed = SimpleNamespace(
        syndrome_packing=packing, syndrome_buffer_1=None,
        window_manager=SimpleNamespace(windows={}), pauli_frame=None,
        execution_runtime=SimpleNamespace(decode_release_time={},
                                          operations={}))
    ledger = event_ledger(completed)
    ledger.check()

    terminals = {(event.round, event.kind) for event in ledger.events
                 if event.status == "terminal"}
    assert terminals == {(1, "PUBLISHED"), (2, "DROPPED")}
    assert packing.packing_drops == 1


_SWEEP_MODES = ("weak", "weak_blocked", "weak_pipelined", "strong",
                "switching_keep", "switching_escalate",
                "switching_parallel", "switching_double")


@pytest.mark.parametrize("seed", range(24))
def test_ledger_holds_over_randomized_configurations(fabric, seed):
    """E2E-001 as a property: over seeded random configurations of every
    run mode, the assembled ledger passes its causal and conservation
    checks, every decoded window's input rounds are accounted for in a
    store, and the window stamps are monotone."""
    import random

    from decsim.decoders.decoders import PipelinedDecoder, PresetLatencyDecoder
    from decsim.qpu.round_policies import FixedRounds
    from decsim.pauli_frame.pauli_frame import PauliFrameConfig
    from decsim.run_spec import RunSpec

    rng = random.Random(seed)
    mode = _SWEEP_MODES[seed % len(_SWEEP_MODES)]
    rounds = rng.randint(6, 9)
    if mode == "weak":
        ops = [fabric["memory_op"](op_id)
               for op_id in range(1, rng.randint(1, 3) + 1)]
        completed = fabric["weak_only_run"](rounds=rounds, ops=ops)
    elif mode == "weak_blocked":
        completed = fabric["weak_only_run"](
            rounds=rounds, ops=[fabric["memory_op"](1),
                                fabric["memory_op"](2, blocked_by=1)])
    elif mode == "weak_pipelined":
        spec = RunSpec(
            ops=[fabric["memory_op"](op_id)
                 for op_id in range(1, rng.randint(2, 4) + 1)],
            d=3, rounds_policy=FixedRounds(rng.randint(3, 6)),
            decoder=PipelinedDecoder(
                PresetLatencyDecoder(100.0), 1.0,
                pipeline_depth=rng.choice([None, 2, 3])),
            links=fabric["declared_profile"](cwb=True, csb=False),
            timing=fabric["declared_timing"](),
            pauli_frame=PauliFrameConfig(
                commit_us=fabric["DECLARED_US"]["frame"]),
            seed=seed)
        completed = spec.build()
    elif mode == "strong":
        completed = fabric["strong_only_run"](rounds=rounds)
    elif mode == "switching_keep":
        completed = fabric["switching_run"](rounds=rounds,
                                            escalation_probability=0.0)
    elif mode == "switching_escalate":
        completed = fabric["switching_run"](rounds=rounds,
                                            escalation_probability=1.0)
    elif mode == "switching_parallel":
        completed = fabric["switching_run"](
            rounds=rounds, escalation_probability=1.0,
            run_both_at_once=True, csb_us=2.0)
    else:
        completed = fabric["switching_run"](
            rounds=rounds, escalation_probability=1.0, double_window=True)

    ledger = event_ledger(completed)
    ledger.check()

    accounted_rounds = {(event.op, event.round) for event in ledger.events
                        if event.kind in ("PUBLISHED", "STORED_SB1")}
    emitted_hi = {}
    for event in ledger.events:
        if event.kind == "EMITTED":
            emitted_hi[event.op] = max(emitted_hi.get(event.op, 0), event.round)
    for window in completed.window_manager.windows.values():
        if window.t_done is None:
            continue
        # a sliding window's lookahead range is clamped to the operation's
        # actual rounds, exactly as the window manager reads it
        # (window_manager.py _read_keys_for_bounds)
        input_hi = min(window.buffer_hi, emitted_hi[window.op_id])
        for round_index in range(window.start_round, input_hi + 1):
            assert (window.op_id, round_index) in accounted_rounds, \
                (f"{mode} seed {seed}: window ({window.op_id}) decoded round "
                 f"{round_index} that no store ever accounted for")
        stamps = [tick for tick in (window.t_first_round,
                                    window.t_data_complete, window.t_queued,
                                    window.t_dispatch, window.t_done)
                  if tick is not None]
        assert stamps == sorted(stamps), \
            f"{mode} seed {seed}: window stamps not monotone: {stamps}"
