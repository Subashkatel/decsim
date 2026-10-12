"""The frame's corrections: every window's write, as it landed."""

import decsim.records.decoding as decoding_records


class FrameCorrections:
    """The frame's corrections, in the order their writes landed.

    A listener on PauliFrame.correction_committed. The experiments layer
    reads the landed writes; the frame keeps no reader of its own. A
    stream keeps none (observation.folds_windows): it measures each
    window as it goes.
    """

    def __init__(self, keeps_records: bool = True) -> None:
        self.keeps_records = keeps_records
        self.committed: list = []

    def correction_committed(
        self, record: decoding_records.PauliFrameCommitRecord
    ) -> None:
        """One window's write has landed in the frame."""
        if self.keeps_records:
            self.committed.append(record)
