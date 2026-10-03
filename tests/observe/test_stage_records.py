"""One run's decoder stage records, kept per operation and window.

A listener on every routed decoder's stage_recorded source. The decoder
fires one record as each stage's end becomes known and keeps nothing
itself, so a run with no ledger connected holds no stage history at all.
The stage vocabulary is the decoder row's own, so an ASIC or GPU model
fires its own stage names through the same source and the ledger asks
nothing about them.
"""

import decsim.observe.stage_records as stage_records


class _Record:
    """The two fields the ledger keys a record by."""

    def __init__(self, operation_id: int, window_id: int, stage: str) -> None:
        self.operation_id = operation_id
        self.window_id = window_id
        self.stage = stage


def test_every_record_is_kept_in_the_order_the_stages_ended():
    ledger = stage_records.StageLedger()
    fetch = _Record(1, 0, "fetch")
    compute = _Record(1, 0, "compute")

    ledger.stage_recorded(fetch)
    ledger.stage_recorded(compute)

    assert ledger.records == [fetch, compute]


def test_one_windows_records_come_back_in_stage_order():
    ledger = stage_records.StageLedger()
    first = _Record(1, 0, "fetch")
    second = _Record(1, 0, "compute")
    other_window = _Record(1, 1, "fetch")

    ledger.stage_recorded(first)
    ledger.stage_recorded(other_window)
    ledger.stage_recorded(second)

    assert ledger.records_for(1, 0) == (first, second)
