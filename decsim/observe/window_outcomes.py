"""Each window's committed answer beside its true label, per operation.

A window's label is the parity, on each logical observable, of the
fired errors it owns: Zhang et al.'s local ground truth
y_i = |E ∩ E_i ∩ L| mod 2 (2509.03815 Eq. (1), lines 497-508), whose
labels add up to the operation's truth (Eq. (2), lines 576-580). An
error is owned by the contribution whose commit rounds hold its first
round, since the logical ledger gives every round one owner
(windows/committed_rounds.py; Toshio et al. 2510.25222 Theorem 1). A
window can be wrong on its own label while the operation is right, the
seam trap 2509.03815 describes (lines 1130-1138), so a label is read
beside the operation's result, not instead of it. Only a source that
keeps its errors gives labels; with any other the answers stand alone.
"""

import dataclasses
from typing import Any, Optional

import decsim.records.decoding as decoding_records
import decsim.records.sampled_errors as sampled_error_records


@dataclasses.dataclass(frozen=True)
class WindowOutcome:
    """One contribution of one operation: its extent, answer and label.

    answer is the committed prediction, None for a timing-only window;
    label is None when the source kept no errors.
    """

    owner_key: tuple
    ownership_kind: str
    commit_lo: int
    commit_hi: int
    answer: Optional[tuple[int, ...]]
    label: Optional[tuple[int, ...]]


class WindowOutcomes(sampled_error_records.FiredErrors):
    """Each operation's windows' outcomes, labelled by its fired errors."""

    def __init__(self) -> None:
        sampled_error_records.FiredErrors.__init__(self)
        self.outcomes_by_operation: dict = {}

    def contributions_delivered(
        self,
        operation_id: Any,  # an opaque identity
        contributions: tuple,
    ) -> None:
        """The operation's result was delivered from these contributions."""
        outcomes = []
        for contribution in contributions:
            outcome = self._outcome_of(operation_id, contribution)
            outcomes.append(outcome)
        self.outcomes_by_operation[operation_id] = tuple(outcomes)

    def _outcome_of(
        self,
        operation_id: Any,  # an opaque identity
        contribution: decoding_records.LogicalContribution,
    ) -> WindowOutcome:
        answer = contribution.logical_observables
        label = None
        if answer is not None:
            label = self.label_of(
                operation_id,
                contribution.commit_lo,
                contribution.commit_hi,
                len(answer),
            )
        return WindowOutcome(
            owner_key=contribution.owner_key,
            ownership_kind=contribution.ownership_kind,
            commit_lo=contribution.commit_lo,
            commit_hi=contribution.commit_hi,
            answer=answer,
            label=label,
        )
