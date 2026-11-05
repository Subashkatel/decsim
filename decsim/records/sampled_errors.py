"""The errors one shot fired, as the source that sampled them reports them.

A fired error is one mechanism of Stim's detector error model, the
"edges in the decoding graph which have been flipped" that Zhang et al.
label each window by (2509.03815 lines 497-508). It is reduced to what
a window's label reads: the round it belongs to and the observables it
flips.
"""

import dataclasses


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
