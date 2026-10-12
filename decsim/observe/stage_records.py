"""One run's decoder stage records, kept per operation and window.

The decoder keeps no stage history itself, so a run with no ledger holds
none. The stage names are the decoder row's own
(decoders/staged_decoder.py).
"""

from typing import Any

import decsim.records.decoding as decoding_records


class StageLedger:
    """The stage records of one run, in the order the stages ended.

    A stream keeps no run-long list (observation.folds_windows), only
    each window's records until the window is measured and let go.
    """

    def __init__(self, keeps_records: bool = True) -> None:
        self.keeps_records = keeps_records
        self.records: list = []
        self._by_window: dict = {}

    def stage_recorded(
        self, record: decoding_records.DecoderStageRecord
    ) -> None:
        """One stage of one job ended on some unit."""
        if self.keeps_records:
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

    def forget_window(self, window_key: tuple) -> None:
        """Let the records of a window a stream has measured go."""
        self._by_window.pop(window_key, None)


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
