"""The errors one shot fired, as the source that sampled them reports them.

A fired error is one mechanism of Stim's detector error model, the
"edges in the decoding graph which have been flipped" that Zhang et al.
label each window by (2509.03815 lines 497-508). It is reduced to what
a window's label reads: the round it belongs to and the observables it
flips.
"""

import dataclasses
from typing import Any, Optional

import decsim.records.program as program_records


@dataclasses.dataclass(frozen=True)
class FiredError:
    """One error mechanism a shot fired.

    first_round is the earliest round of the detectors it flips: the
    window whose commit rounds hold that round owns it, as each window
    owns the faults touching its commit rounds and the earliest such
    window keeps them (detector_error_model/window_placement.py).
    detectors are the detectors it flips, sorted: with the observables,
    the identity of its physical fault column (decoders/truth.py).
    """

    first_round: int
    logical_observables: tuple[int, ...]
    detectors: tuple[int, ...] = ()


class FiredErrors:
    """The errors each operation's shot fired, by the operation's id.

    The source that keeps its errors fires errors_sampled once a shot
    (qpu/stim_device.py ErrorModelStimDevice). Window outcomes and the
    truth verdict keep them here and read the same labels from them.
    """

    def __init__(self) -> None:
        self.errors_by_operation: dict = {}

    def errors_sampled(
        self, operation: program_records.Operation, fired_errors: tuple
    ) -> None:
        """The source drew an operation's shot; these errors fired."""
        self.errors_by_operation[operation.id] = fired_errors

    def label_of(
        self,
        operation_id: Any,  # an opaque identity
        commit_lo: int,
        commit_hi: int,
        arity: int,
    ) -> Optional[tuple[int, ...]]:
        """The label of any extent: the parity of the errors it owns.

        None when the source kept no errors for the operation.
        """
        fired_errors = self.errors_by_operation.get(operation_id)
        if fired_errors is None:
            return None
        return label_of_errors(fired_errors, commit_lo, commit_hi, arity)


def label_of_errors(
    fired_errors: tuple, commit_lo: int, commit_hi: int, arity: int
) -> tuple[int, ...]:
    """The label of an extent: the parity of the fired errors it owns.

    An error is owned by the extent whose commit rounds hold its first
    round. Window outcomes (observe/window_outcomes.py) and the truth
    verdict (escalation/verdicts.py) read the same label, so the verdict
    switches exactly the windows window_outcomes.csv marks wrong.
    """
    label = [0] * arity
    for error in fired_errors:
        if not commit_lo <= error.first_round <= commit_hi:
            continue
        _fold_bits(label, error.logical_observables)
    return tuple(label)


def _fold_bits(bits: list[int], flipped: tuple[int, ...]) -> None:
    """XOR flipped into bits, observable by observable."""
    for observable_index, bit in enumerate(flipped):
        bits[observable_index] ^= bit
