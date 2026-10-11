"""One run's decoder stage records, kept per operation and window.

The decoder keeps no stage history itself, so a run with no ledger holds
none. The stage names are the decoder row's own
(decoders/staged_decoder.py).
"""

from typing import Any

import decsim.records.decoding as decoding_records


class StageLedger:
    """The stage records of one run, in the order the stages ended."""

    def __init__(self) -> None:
        self.records: list = []
        self._by_window: dict = {}

    def stage_recorded(
        self, record: decoding_records.DecoderStageRecord
    ) -> None:
        """One stage of one job ended on some unit."""
        self.records.append(record)
        for key in _windows_of(record):
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


def _windows_of(record: decoding_records.DecoderStageRecord) -> list:
    """The record's own window, then each other window its decode served.

    A merged batch's own key is (-1, 0), so its stages are filed under
    each member's window too, where a reader of that window looks.
    """
    own = (record.operation_id, record.window_id)
    windows = [own]
    for window in record.served_windows:
        if window not in windows:
            windows.append(window)
    return windows
