"""One run's decoder stage records, kept per operation and window.

A listener on every routed decoder's stage_recorded source. The decoder
fires one DecoderStageRecord as each stage's end becomes known and keeps
nothing itself, so a run with no ledger connected holds no stage history
at all; the front's per-window measurement and the latency plot read the
ledger. The stage vocabulary is the decoder row's own
(decoders/staged_decoder.py), so an ASIC or GPU model fires its own
stage names through the same source.
"""


class StageLedger:
    """The stage records of one run, in the order the stages ended."""

    def __init__(self) -> None:
        self.records: list = []
        self._by_window: dict = {}

    def stage_recorded(self, record) -> None:
        """One stage of one job ended on some unit."""
        self.records.append(record)
        key = (record.op_id, record.window_id)
        window_records = self._by_window.setdefault(key, [])
        window_records.append(record)

    def records_for(self, op_id, window_id) -> tuple:
        """One window's stage records, in stage order."""
        key = (op_id, window_id)
        window_records = self._by_window.get(key, ())
        return tuple(window_records)

    def windows(self) -> tuple:
        """Every (operation, window) that recorded a stage, in key order."""
        keys = self._by_window.keys()
        return tuple(sorted(keys))
