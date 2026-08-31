"""Switching and feedback under declared ticks: serial, parallel,
double-window, the strong-result ledger, and the release chain.

Contract: ../docs/architecture/STRONG_REQUEST_LIFECYCLE.md (sandbox root). The sampled
confidence decoder with probability 0.0 or 1.0 keeps every scenario
deterministic; all times are exact arithmetic over the declared ticks.
"""

import pytest

from decsim.config import us
from decsim.decoders.strong_escalation import (HeldStrongCompletion,
                                               StrongRequestLedger)
from decsim.message import (DecodeJob, DecodeResult, DecoderRequestKey,
                            DecoderTier, StrongDecodeCompletion)


# ---------------------------------------------------------------- serial

def test_confident_serial_weak_never_escalates(fabric):
    completed = fabric["switching_run"](escalation_probability=0.0, rounds=9)
    records = completed.pauli_frame.snapshot().records
    assert [record.tier for record in records] == ["weak"] * 3
    assert completed.decoder_manager.strong_needed == 0
    assert completed.decoder_manager.strong_cancelled == 0


def test_unconfident_serial_escalates_every_window(fabric):
    completed = fabric["switching_run"](escalation_probability=1.0, rounds=9)
    records = completed.pauli_frame.snapshot().records
    assert [record.tier for record in records] == ["strong"] * 3
    assert completed.decoder_manager.strong_needed == 3
    assert completed.window_manager.escalation.pending_escalations == {}


def test_serial_escalation_timeline_is_exact(fabric):
    """W0 weak done at 30 (data 15 + wbd 5 + weak 10); the strong start
    clamps to max(weak done + sbd 6, weak done + wsd 3) = 36; strong 30
    ends at 66; do 4 + frame 1 commit at 71; the Held boundary releases
    W1's parked decode at 71 + dd 0.5."""
    completed = fabric["switching_run"](escalation_probability=1.0, rounds=9,
                                        io_trace=True)
    lines = completed.engine.log_lines
    assert fabric["log_tick"](lines, "START DECODE strong(mem1 W0)") == us(36)
    records = completed.pauli_frame.snapshot().records
    first = next(r for r in records if r.window_key == (1, 0))
    assert first.accepted_ticks == us(70)      # 66 + do 4
    assert first.committed_ticks == us(71)     # + frame write 1
    assert fabric["log_tick"](lines, "START DECODE mem1 W1") == us(71.5)


def test_provisional_weak_result_never_reaches_the_frame(fabric):
    """An escalated window's weak result is withheld; exactly one
    authoritative correction per window reaches the frame."""
    completed = fabric["switching_run"](escalation_probability=1.0, rounds=9)
    frame = completed.pauli_frame.snapshot()
    keys = [record.window_key for record in frame.records]
    assert sorted(keys) == [(1, 0), (1, 1), (1, 2)]
    assert all(record.tier == "strong" for record in frame.records)
    assert frame.duplicate_drop_count == 0


# -------------------------------------------------------------- parallel

def test_parallel_requires_the_csb_margin(fabric):
    """With csb 7 slower than cwb 4, the two-sided context is not yet in
    syndrome buffer 1 when the weak window becomes ready; the immediate
    strong build refuses loudly instead of waiting (the documented
    fail-loud contract; a waiting design would need its own note)."""
    with pytest.raises(RuntimeError,
                       match="csb lag beyond the escalation margin"):
        fabric["switching_run"](escalation_probability=0.0, rounds=9,
                                run_both_at_once=True)


def test_parallel_confident_weak_cancels_every_strong(fabric):
    completed = fabric["switching_run"](escalation_probability=0.0, rounds=9,
                                        run_both_at_once=True, csb_us=2.0)
    records = completed.pauli_frame.snapshot().records
    assert [record.tier for record in records] == ["weak"] * 3
    assert completed.decoder_manager.strong_cancelled == 3
    assert completed.decoder_manager.strong_needed == 0
    completed.syndrome_buffer_1.check_settled()   # cancels released the holds


def test_parallel_unconfident_weak_takes_every_strong(fabric):
    completed = fabric["switching_run"](escalation_probability=1.0, rounds=9,
                                        run_both_at_once=True, csb_us=2.0)
    records = completed.pauli_frame.snapshot().records
    assert [record.tier for record in records] == ["strong"] * 3
    assert completed.decoder_manager.strong_needed == 3
    assert completed.decoder_manager.strong_cancelled == 0


# ---------------------------------------------------------- double window

def test_double_window_terminal_submits_exactly_once(fabric):
    """Escalating the last window defers until its clamped tail rounds
    are stored; the pending slab submits once and the registry drains."""
    completed = fabric["switching_run"](
        rounds=9, double_window=True, escalation_probability=0.0,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0,
        io_trace=True)
    lines = completed.engine.log_lines
    assert sum("terminal data complete -> strong window submitted" in line
               for line in lines) == 1
    records = completed.pauli_frame.snapshot().records
    assert [(r.window_key, r.tier) for r in records] == [
        ((1, 0), "weak"), ((1, 1), "weak"), ((1, 2), "strong")]
    assert completed.window_manager.escalation.pending_escalations == {}


def test_double_window_absorbs_covered_windows(fabric):
    """A mid-chain escalation whose region reaches the stream end absorbs
    the covered windows: they never produce their own frame records."""
    completed = fabric["switching_run"](
        rounds=12, double_window=True, escalation_probability=0.0,
        probability_for=lambda job: 1.0 if job.window_id == 1 else 0.0,
        io_trace=True)
    assert any("weak chain skips 2 window(s)" in line
               for line in completed.engine.log_lines)
    records = completed.pauli_frame.snapshot().records
    assert [(r.window_key, r.tier) for r in records] == [
        ((1, 0), "weak"), ((1, 1), "strong")]


def test_double_window_far_boundary_waits_for_the_restart_commit(fabric):
    """In the decode-ahead regime the strong start is triggered by the
    far-side weak boundary, not by a Buffer 1 arrival."""
    completed = fabric["switching_run"](
        rounds=15, double_window=True, escalation_probability=0.0,
        probability_for=lambda job: 1.0 if job.window_id == 1 else 0.0,
        round_us=4.0, io_trace=True)
    lines = completed.engine.log_lines
    deferred = fabric["log_index"](
        lines, "strong start deferred until the far-side weak boundary")
    submitted = fabric["log_index"](
        lines, "far-side weak boundary determined -> strong window submitted")
    assert deferred < submitted
    records = completed.pauli_frame.snapshot().records
    assert [(r.window_key, r.tier) for r in records] == [
        ((1, 0), "weak"), ((1, 4), "weak"), ((1, 1), "strong")]
    assert completed.window_manager.escalation.pending_escalations == {}


# ------------------------------------------------------------- the ledger

def _request_key(sequence):
    return DecoderRequestKey(1, 0, DecoderTier.STRONG, sequence)


def _strong_job(request_key):
    return DecodeJob(op_id=1, window_id=0, n_rounds=9,
                     strong_decode_for=(1, 0), request_key=request_key)


def test_strong_completion_before_selection_is_held_then_consumed():
    """A result finishing before its WSD selection waits in the ledger and
    is consumed the moment the selection lands."""
    ledger = StrongRequestLedger()
    key = _request_key(7)
    job = _strong_job(key)
    ledger.admit_strong(job, now=0)
    ledger.begin_selection((1, 0), key)
    ledger.finish_service(job)
    held = HeldStrongCompletion(
        job, StrongDecodeCompletion(key, DecodeResult(1, 0)), 50)

    assert ledger.complete(held) is False       # demand not selected yet
    assert ledger.select((1, 0), key).completion.request_key == key


def test_selection_before_completion_consumes_immediately():
    ledger = StrongRequestLedger()
    key = _request_key(7)
    job = _strong_job(key)
    ledger.admit_strong(job, now=0)
    ledger.begin_selection((1, 0), key)
    assert ledger.select((1, 0), key) is None    # nothing finished yet
    ledger.finish_service(job)
    held = HeldStrongCompletion(
        job, StrongDecodeCompletion(key, DecodeResult(1, 0)), 50)
    assert ledger.complete(held) is True         # consumed now


def test_stale_strong_result_raises_when_a_newer_request_owns_the_destination():
    ledger = StrongRequestLedger()
    old_key, new_key = _request_key(7), _request_key(9)
    old_job = _strong_job(old_key)
    ledger.admit_strong(old_job, now=0)
    ledger.take_live((1, 0))                     # the old request was cancelled
    ledger.admit_strong(_strong_job(new_key), now=1)
    stale = HeldStrongCompletion(
        old_job, StrongDecodeCompletion(old_key, DecodeResult(1, 0)), 50)
    with pytest.raises(RuntimeError, match="newer strong request"):
        ledger.complete(stale)


def test_unconsumable_strong_result_raises():
    ledger = StrongRequestLedger()
    key = _request_key(7)
    job = _strong_job(key)
    ledger.admit_strong(job, now=0)
    ledger.finish_service(job)                   # no selection, no open weak
    orphan = HeldStrongCompletion(
        job, StrongDecodeCompletion(key, DecodeResult(1, 0)), 50)
    with pytest.raises(RuntimeError, match="no destination waiting"):
        ledger.complete(orphan)


def test_duplicate_strong_admission_for_one_destination_raises():
    ledger = StrongRequestLedger()
    ledger.admit_strong(_strong_job(_request_key(7)), now=0)
    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        ledger.admit_strong(_strong_job(_request_key(8)), now=1)


# -------------------------------------------------------------- feedback

def test_no_feedback_when_none_is_required(fabric):
    """No blocked successor, no result return: no release and no OC or
    CQ traffic."""
    completed = fabric["weak_only_run"](rounds=6)
    assert completed.execution_runtime.decode_release_time == {}
    transfers = completed.result.link_traffic.get("transfers", [])
    assert not any(row.get("path") in ("oc", "cq") for row in transfers)


def test_release_travels_oc_then_cq_with_exact_cost(fabric):
    """The blocked successor's release lands frame-commit + oc 2 + cq 2
    later, and the successor starts on that boundary."""
    ops = [fabric["memory_op"](1), fabric["memory_op"](2, blocked_by=1)]
    completed = fabric["weak_only_run"](rounds=6, ops=ops)
    runtime = completed.execution_runtime
    frame_records = completed.pauli_frame.snapshot().records
    blocker_commit = next(r.committed_ticks for r in frame_records
                          if r.window_key == (1, 0))

    assert runtime.decode_release_time[2] == blocker_commit + us(2 + 2)
    assert runtime.op_start_time[2] == runtime.decode_release_time[2]


def test_successor_cannot_start_before_its_release(fabric):
    ops = [fabric["memory_op"](1), fabric["memory_op"](2, blocked_by=1)]
    completed = fabric["weak_only_run"](rounds=6, ops=ops)
    runtime = completed.execution_runtime
    assert runtime.op_start_time[2] >= runtime.decode_release_time[2]
    assert runtime.op_start_time[1] == 0


def test_decoder_manager_cannot_bypass_the_controller(fabric):
    """Releases travel ConditionalRelease -> Controller (OC) -> CQ; the
    decoder side holds no path to the QPU or the runtime."""
    completed = fabric["weak_only_run"](rounds=6)
    assert completed.conditional_release.controller is completed.controller
    assert not hasattr(completed.decoder_manager, "qpu")
    assert not hasattr(completed.decoder_manager, "conditional_release")
    assert not hasattr(completed.decoder_manager, "execution_runtime")
