"""The logical ledger against the per-round oracle's law.

rowW5 (validation/component_matrix/rowW5_ledger/compare_round_oracle.py):
every round of a queried interval has exactly one owner, the owners are
collected as a set and their observables XORed once each (the Pauli
frame rule: PECOS pauli_frame.rs folds one mask per accepted
correction); any timing-only owner makes the interval timing-only; an
owner crossing the interval's edge is refused under strict, and under
stream_segment only when it carries observables. The installs and
queries here are fixed; the harness runs the random ones.
"""

import pytest

import decsim.records.decoding as decoding_records
import decsim.windows.committed_rounds as committed_rounds


def _contribution(index, commit_lo, commit_hi, observables):
    return decoding_records.LogicalContribution(
        ("s", index), commit_lo, commit_hi, "ordinary_window", observables
    )


def _ledger():
    ledger = committed_rounds.LogicalLedger()
    contribution = _contribution(0, 1, 3, (1, 0))
    ledger.install(contribution)
    contribution = _contribution(1, 4, 4, (1, 1))
    ledger.install(contribution)
    contribution = _contribution(2, 5, 8, (0, 1))
    ledger.install(contribution)
    return ledger


def test_an_interval_folds_each_owner_once_like_the_per_round_oracle():
    ledger = _ledger()
    # rounds 1..8: owners 0, 1, 2 once each: (1,0) ^ (1,1) ^ (0,1) = (0,0)
    assert ledger.observables_for_interval(
        "s", 1, 8, boundary_policy="strict"
    ) == (0, 0)
    # rounds 4..8: owners 1 and 2: (1,1) ^ (0,1) = (1,0)
    assert ledger.observables_for_interval(
        "s", 4, 8, boundary_policy="strict"
    ) == (1, 0)


def test_an_overlapping_install_is_refused():
    ledger = _ledger()
    with pytest.raises(RuntimeError):
        contribution = _contribution(3, 3, 5, (0, 0))
        ledger.install(contribution)


def test_a_round_without_an_owner_is_refused():
    ledger = _ledger()
    contribution = _contribution(3, 10, 11, (0, 0))
    ledger.install(contribution)
    with pytest.raises(RuntimeError):
        ledger.observables_for_interval("s", 8, 11, boundary_policy="strict")


def test_a_crossing_owner_is_refused_under_strict_only():
    ledger = _ledger()
    with pytest.raises(RuntimeError, match="crosses"):
        ledger.observables_for_interval("s", 2, 4, boundary_policy="strict")
    with pytest.raises(RuntimeError, match="crosses"):
        ledger.observables_for_interval(
            "s", 2, 4, boundary_policy="stream_segment"
        )


def test_a_timing_only_owner_may_cross_under_stream_segment():
    ledger = committed_rounds.LogicalLedger()
    contribution = _contribution(0, 1, 3, None)
    ledger.install(contribution)
    contribution = _contribution(1, 4, 6, (1,))
    ledger.install(contribution)
    assert (
        ledger.observables_for_interval(
            "s", 2, 6, boundary_policy="stream_segment"
        )
        is None
    )
    with pytest.raises(RuntimeError, match="crosses"):
        ledger.observables_for_interval("s", 2, 6, boundary_policy="strict")


def test_a_timing_only_owner_makes_the_interval_timing_only():
    ledger = committed_rounds.LogicalLedger()
    contribution = _contribution(0, 1, 2, (1,))
    ledger.install(contribution)
    contribution = _contribution(1, 3, 4, None)
    ledger.install(contribution)
    assert (
        ledger.observables_for_interval("s", 1, 4, boundary_policy="strict")
        is None
    )
