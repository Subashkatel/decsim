"""The window referee's audit: what it re-decoded and where it disagreed.

A listener on the Decoder port's window_checked source. The referee row
(decoders/verify_windows.py) re-decodes every window with the official
Tesseract backend and compares the owned observable contribution; the
count of checks and the windows that disagreed are the run's accuracy
audit, and they belong to whoever watches the run, not to the decoder.
"""


class RefereeAudit:
    """Every referee check of one run, counted, disagreements by window."""

    def __init__(self) -> None:
        self.windows_checked = 0
        self.disagreeing_windows: list = []

    @property
    def window_disagreements(self) -> int:
        """How many checked windows the referee decoded differently."""
        return len(self.disagreeing_windows)

    def window_checked(self, window_key, is_agreement: bool) -> None:
        """One window re-decoded, and whether the referee reached the same."""
        self.windows_checked += 1
        if is_agreement:
            return
        self.disagreeing_windows.append(window_key)
