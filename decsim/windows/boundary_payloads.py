"""How a boundary message is represented on decoder_to_decoder.

A window's hand-off updates the detectors of its neighbour's oldest
round layer, and nothing else: Tan et al. 2209.09219 lines 936-946,
quits `syn_update` over one check layer (sliding_window.py:164-174),
cuda-q QEC's `syndrome_mods` bounded to the next window's first round
(sliding_window.cpp:325-344), and Bombin et al. 2303.04846 lines 784-786
("only a small number of check generators ... will need to have their
syndrome updated"). Two rows say how that update is written on the wire:
a dense mask over the layer, which is what both compiled
implementations carry, and a sparse list of the flipped detectors, which
is Skoric's artificial-defect list (2209.08552 lines 265-269, 1038-1040)
and Bombin's small set. The row is named by windows.boundary_payload.
"""

import math

import decsim.records.windows as window_records


class DenseSeamMask:
    """One bit per detector of the seam layer, whatever the noise.

    The cost is the layer's size, d*d-1 on a bulk layer of a rotated
    surface code, and so is independent of how many detectors the
    correction flips. quits allocates `syn_update` as one check layer
    and cuda-q QEC carries `syndrome_mods` over the window's rows, so a
    compiled decoder pays this whether or not a bit is set.
    """

    def bits(self, seam: window_records.BoundarySeam) -> int:
        """The seam layer's detectors, one bit each."""
        return seam.detector_count


class SparseSeamList:
    """One index per flipped detector of the seam layer.

    Skoric's blocks exchange the artificial defects themselves
    (2209.08552 lines 265-269, sent block to block at 1038-1040), and
    Bombin bounds the update to "a small number of check generators"
    (2303.04846 lines 784-786). An index addresses one detector of the
    layer, so it is ceil(log2(seam detectors)) wide, and a layer of one
    detector still needs one bit to name it.
    """

    def bits(self, seam: window_records.BoundarySeam) -> int:
        """One index per flip, each wide enough to name a seam detector."""
        index_width = _index_width(seam.detector_count)
        return seam.flip_count * index_width


def _index_width(detector_count: int) -> int:
    """The bits an index into the seam layer takes."""
    if detector_count <= 1:
        return 1
    exact_width = math.log2(detector_count)
    return math.ceil(exact_width)
