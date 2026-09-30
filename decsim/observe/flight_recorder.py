"""The frame's corrections: every window's write, as accepted and as landed."""


class FrameCorrections:
    """The frame's corrections, as accepted and as landed.

    A listener on PauliFrame.correction_accepted and
    correction_committed. The experiments layer reads the landed
    writes; the run ledger the tests check (tests/observe/run_ledger.py)
    pairs the two for each window's write. The frame keeps no reader of
    its own.
    """

    def __init__(self) -> None:
        self.accepted: list = []
        self.committed: list = []

    def correction_accepted(self, record) -> None:
        """One window's correction was taken and its write charged."""
        self.accepted.append(record)

    def correction_committed(self, record) -> None:
        """One window's write has landed in the frame."""
        self.committed.append(record)
