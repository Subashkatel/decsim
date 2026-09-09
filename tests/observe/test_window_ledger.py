"""Every window's record, what owns its rounds, and what absorbed it.

A listener on the committer's window_committed and the forward strong
window's window_absorbed. It holds the Window records the plan laid out
at build, so a window's stamps are read here and never from the planner,
and the switching study's final rows come from what it heard.
"""

import pytest

import decsim.observe.window_ledger as window_ledger_module
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


def _window(operation_id: int, window_index: int):
    """One planned window over rounds 1 to 6, reading three more."""
    return window_records.Window(
        operation_id=operation_id,
        window_index=window_index,
        commit_lo=1,
        commit_hi=6,
        buffer_hi=9,
        round_count=9,
        buffer_lo=1,
    )


def _contribution(owner_key: tuple):
    return decoding_records.LogicalContribution(
        owner_key=owner_key,
        commit_lo=1,
        commit_hi=6,
        ownership_kind="weak",
        logical_observables=(0,),
    )


def test_the_planned_windows_are_loaded_before_anyone_could_listen():
    ledger = window_ledger_module.WindowLedger()
    planned = _window(1, 0)

    ledger.load_planned({planned.key: planned})

    assert ledger.windows[planned.key] is planned


def test_a_grown_stream_window_arrives_through_its_own_call():
    ledger = window_ledger_module.WindowLedger()
    grown = _window(1, 1)

    ledger.window_planned(grown)

    assert ledger.windows[grown.key] is grown


def test_a_committed_window_carries_the_contribution_that_owns_its_rounds():
    ledger = window_ledger_module.WindowLedger()
    window = _window(1, 0)
    contribution = _contribution(window.key)

    ledger.window_committed(window, contribution)
    rows = ledger.final_rows()

    assert rows[0].destination_key == window.key
    assert rows[0].final_commit_lo == 1
    assert rows[0].final_commit_hi == 6
    assert rows[0].window_disposition == "weak"
    assert rows[0].absorbed_into is None


def test_an_absorbed_window_names_its_owner_and_commits_nothing_itself():
    ledger = window_ledger_module.WindowLedger()
    absorbed = _window(1, 1)
    owner_key = (1, 0)

    ledger.window_planned(absorbed)
    ledger.window_absorbed(absorbed.key, owner_key)
    rows = ledger.final_rows()

    assert rows[0].window_disposition == "absorbed"
    assert rows[0].absorbed_into == owner_key
    assert rows[0].final_commit_lo is None
    assert rows[0].final_commit_hi is None


def test_a_window_that_neither_committed_nor_was_absorbed_is_refused():
    """A final row with no owner would be a silent hole in the ledger."""
    ledger = window_ledger_module.WindowLedger()
    never_finished = _window(1, 0)

    ledger.window_planned(never_finished)

    with pytest.raises(RuntimeError) as refusal:
        ledger.final_rows()

    assert "has no logical contribution" in str(refusal.value)


def test_the_rows_come_back_in_stable_key_order():
    ledger = window_ledger_module.WindowLedger()
    second = _window(1, 1)
    first = _window(1, 0)

    second_contribution = _contribution(second.key)
    first_contribution = _contribution(first.key)

    ledger.window_committed(second, second_contribution)
    ledger.window_committed(first, first_contribution)
    rows = ledger.final_rows()

    assert rows[0].destination_key == first.key
    assert rows[1].destination_key == second.key


def test_the_row_carries_the_weak_extents_off_the_window_record():
    ledger = window_ledger_module.WindowLedger()
    window = _window(1, 0)

    contribution = _contribution(window.key)

    ledger.window_committed(window, contribution)
    rows = ledger.final_rows()
    row = rows[0]

    assert row.weak_buffer_lo == window.start_round
    assert row.weak_commit_lo == 1
    assert row.weak_commit_hi == 6
    assert row.weak_buffer_hi == 9
