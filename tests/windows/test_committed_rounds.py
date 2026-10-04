"""The logical ledger against the per-round oracle's law.

The law:
every round of a queried interval has exactly one owner, the owners are
collected as a set and their observables XORed once each (the Pauli
frame rule: PECOS pauli_frame.rs folds one mask per accepted
correction); any timing-only owner makes the interval timing-only; an
owner crossing the interval's edge is refused under strict, and under
stream_segment only when it carries observables. The installs and
queries here are fixed; the harness runs the random ones.
"""

import dataclasses

import pytest

import decsim.decoders.decoder as decoder_module
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.windows.committed_rounds as committed_rounds
import tests.declared_run as declared_run


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


def test_an_overlapping_owner_is_refused_when_its_interval_is_read():
    ledger = _ledger()
    contribution = _contribution(3, 3, 5, (0, 0))
    ledger.install(contribution)
    with pytest.raises(RuntimeError, match="overlap"):
        ledger.observables_for_interval("s", 1, 8, boundary_policy="strict")


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


class _ReusedObservables(decoder_module.DecoderBase):
    """A user row that returns a list for window 1, then changes it."""

    def __init__(self) -> None:
        self.middle = [1]

    def latency(self, job) -> int:
        del job
        return 1_000_000

    def decode(self, job) -> decoding_records.DecodeResult:
        prediction = (0,)
        if job.window_id == 1:
            prediction = self.middle
        if job.window_id == 2:
            self.middle.pop()
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=prediction
        )


def test_a_contribution_keeps_the_observables_its_decoder_returned(
    monkeypatch,
):
    """The ledger holds the values of the install, not the decoder's list.

    Windows 0, 1 and 2 return (0,), [1] and (0,), and window 2's decode
    empties the list window 1 returned; the operation reads the XOR of
    what was returned, (1,).
    """
    row = _ReusedObservables()
    results = []

    def run_with_the_row(settings, seed=0, probes=()):
        del probes
        algorithm = declared_run.OneDecoder(row)
        weak = dataclasses.replace(settings.weak_decoder, algorithm=algorithm)
        with_the_row = dataclasses.replace(settings, weak_decoder=weak)
        machine = machine_module.Machine.build(with_the_row, seed)
        result = machine.run()
        results.append(result)
        return machine

    monkeypatch.setattr(declared_run, "run_machine", run_with_the_row)
    declared_run.weak_only_run(rounds=12)

    (operation_result,) = results[0].operation_results
    assert operation_result.logical_observables == (1,)
