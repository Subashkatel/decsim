"""The rounds the burst detector fired on, for a shot's catch columns.

A listener on the BurstDetector port's round_flagged source. The burst
study scores a detector by its first alarm at or after a burst's onset
and whether it comes within a deadline (burst study PROTOCOL.md section
3, "Delay" and "Caught within k"), so the record keeps every round the
detector fired on and the measurement reads that alarm off it.
"""


class BurstFlags:
    """Every round the detector fired on, in the order it fired."""

    def __init__(self) -> None:
        self.flagged_rounds: list = []

    def round_flagged(self, operation_id, round_index: int) -> None:
        """The detector fired on one more round."""
        del operation_id
        self.flagged_rounds.append(round_index)

    def first_flag_from(self, first_round: int) -> int:
        """The first round at or after first_round fired on, 0 for none.

        Rounds count from 1, so 0 names no round.
        """
        later = []
        for round_index in self.flagged_rounds:
            if round_index >= first_round:
                later.append(round_index)
        return min(later, default=0)
