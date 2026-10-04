"""One run's decoder stage records, kept per operation and window.

The decoder keeps no stage history itself, so a run with no ledger holds
none. The stage names are the decoder row's own
(decoders/staged_decoder.py).
"""

from typing import Any

import decsim.decoders.staged_decoder as staged_decoder


class StageLedger:
    """The stage records of one run, in the order the stages ended."""

    def __init__(self) -> None:
        self.records: list = []
        self._by_window: dict = {}

    def stage_recorded(self, record: staged_decoder.DecoderStageRecord) -> None:
        """One stage of one job ended on some unit."""
        self.records.append(record)
        key = (record.operation_id, record.window_id)
        window_records = self._by_window.setdefault(key, [])
        window_records.append(record)

    def records_for(
        self,
        operation_id: Any,  # an opaque identity
        window_id: int,
    ) -> tuple:
        """One window's stage records, in stage order."""
        key = (operation_id, window_id)
        window_records = self._by_window.get(key, ())
        return tuple(window_records)
