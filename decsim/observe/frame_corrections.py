"""The frame's corrections: every window's write, as it landed."""


class FrameCorrections:
    """The frame's corrections, in the order their writes landed.

    A listener on PauliFrame.correction_committed. The experiments layer
    reads the landed writes; the frame keeps no reader of its own.
    """

    def __init__(self) -> None:
        self.committed: list = []

    def correction_committed(self, record) -> None:
        """One window's write has landed in the frame."""
        self.committed.append(record)
