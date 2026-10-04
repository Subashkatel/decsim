"""The frame's corrections: every window's write, as it landed."""

import decsim.pauli_frame.pauli_frame as pauli_frame_module


class FrameCorrections:
    """The frame's corrections, in the order their writes landed.

    A listener on PauliFrame.correction_committed. The experiments layer
    reads the landed writes; the frame keeps no reader of its own.
    """

    def __init__(self) -> None:
        self.committed: list = []

    def correction_committed(
        self, record: pauli_frame_module.PauliFrameCommitRecord
    ) -> None:
        """One window's write has landed in the frame."""
        self.committed.append(record)
