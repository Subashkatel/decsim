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
    """

    first_round: int
    logical_observables: tuple[int, ...]
