"""The two strong-window shapes plan the region the paper gives.

Toshio et al. 2510.25222: the context window is the commit region with
one buffer of context on each side, r_strong = r_com + 2 r_buf, clipped
at the operation's edge (Sec. III A; text lines 1250-1252 of
tmp/papers/txt); the forward window starts at the escalated commit,
absorbs the windows it covers, and is decoded once both of its
boundaries are weak-determined: the restart window's commit, or the
terminal data (Sec. III C, Fig. 12). A d=3 sliding window commits 3
rounds and buffers 3, so r_strong is 9 rounds.
"""

import pytest

import decsim.message as message
import decsim.observe.run_views as run_views
import tests.escalation.declared_fabric as fabric


def _strong_request_record(machine, window_id: int):
    view = run_views.switching_records_view(
        machine.window_manager, machine.decode_records
    )
    for record in view.requests:
        is_window = record.request_key.window_id == window_id
        is_strong = record.request_key.tier is message.DecoderTier.STRONG
        if is_window and is_strong:
            return record
    raise AssertionError(f"no strong request for window {window_id}")


def test_the_context_window_reads_commit_plus_one_buffer_each_side():
    machine = fabric.switching_machine(
        rounds=12, escalated_windows={1}, record=True
    )
    machine.run()
    strong = _strong_request_record(machine, 1)
    # W1 commits rounds 4-6; one 3-round buffer each side reads 1-9
    assert (strong.input_round_lo, strong.input_round_hi) == (1, 9)
    assert strong.input_round_count == 9
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "strong"),
        ((1, 2), "weak"),
        ((1, 3), "weak"),
    ]


def test_the_context_window_is_clipped_at_the_operations_first_round():
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={0}, record=True
    )
    machine.run()
    strong = _strong_request_record(machine, 0)
    # W0 commits rounds 1-3; the left buffer has no rounds before 1
    assert (strong.input_round_lo, strong.input_round_hi) == (1, 6)
    assert strong.input_round_count == 6


def test_the_forward_window_absorbs_the_windows_it_covers():
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        double_window=True,
        round_microseconds=4.0,
    )
    machine.run()
    assigned = fabric.log_lines_containing(machine, "assigned")
    assert len(assigned) == 1
    # W1 commits 4-6; the strong window is 9 rounds, 4-12, so W2 and W3
    # (commits 7-9 and 10-12) are absorbed and W4 restarts the chain
    assert (
        "strong window rounds 4-12 assigned; weak chain skips 2 window(s); "
        "strong start deferred until the far-side weak boundary"
    ) in assigned[0]
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 4), "weak"),
        ((1, 1), "strong"),
    ]


def test_the_forward_window_is_submitted_once_at_the_far_boundary_commit():
    machine = fabric.switching_machine(
        rounds=15,
        escalated_windows={1},
        double_window=True,
        round_microseconds=4.0,
    )
    machine.run()
    lines = machine.engine.log_lines
    deferred_lines = fabric.log_lines_containing(
        machine, "deferred until the far-side weak boundary"
    )
    submitted_lines = fabric.log_lines_containing(
        machine, "far-side weak boundary determined -> strong window submitted"
    )
    restart_lines = fabric.log_lines_containing(machine, "DECODE DONE mem1 W4")
    assert len(submitted_lines) == 1
    deferred = lines.index(deferred_lines[0])
    restart_committed = lines.index(restart_lines[0])
    submitted = lines.index(submitted_lines[0])
    assert deferred < restart_committed < submitted
    assert not machine.window_manager.strong_redecode.has_pending()


def test_the_forward_window_at_the_operations_end_waits_for_terminal_data():
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={2}, double_window=True
    )
    machine.run()
    submitted = fabric.log_lines_containing(
        machine, "terminal data complete -> strong window submitted"
    )
    assert len(submitted) == 1
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "strong"),
    ]
    assert not machine.window_manager.strong_redecode.has_pending()


def test_a_second_escalation_of_one_window_is_refused():
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={2}, double_window=True
    )
    machine.run()
    shape = machine.window_manager.strong_redecode.shape
    again = message.DecodeJob(
        op_id=1, window_id=2, n_rounds=3, strong_label="strong(mem1 W2)"
    )
    with pytest.raises(RuntimeError, match="duplicate strong escalation"):
        shape.plan(again)
