"""Which block of a coupling reads the other's region, and which waits.

A joint measurement couples code blocks, as the program states them
(Operation.couplings). Across the shared face one window reads the
other's region as a spatial buffer and the other waits for it; which
goes first is the scheme's choice (Bombín et al. 2303.04846 lines
720-723), so it is a row. A block is decsim's opaque id, so a row that
reads more than the id says what it reads.
"""

import dataclasses
from typing import Optional


class CoordinateOrder:
    """The block whose coordinates are each at least the other's reads.

    SWIPER's rule (2412.05115; swiper/window_manager.py
    SlidingWindowManager.process_round), for blocks named by grid
    coordinates, as the lattice-surgery frontend names its patches. A
    merge face joins neighbouring patches, so exactly one reads; blocks
    neither of which is at least the other read nothing.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The coordinate order, which takes no setting."""

        name = "coordinate_order"

        def build(self) -> "CoordinateOrder":
            """The order."""
            return CoordinateOrder()

    def orient(self, blocks: tuple) -> Optional[tuple]:
        """The coupled blocks as (reader, read); None when neither reads."""
        first, second = blocks
        if _is_at_least(first, second):
            return (first, second)
        if _is_at_least(second, first):
            return (second, first)
        return None


def _is_at_least(block: tuple, other: tuple) -> bool:
    """Whether each coordinate of the block is at least the other's."""
    return all(
        mine >= theirs for mine, theirs in zip(block, other, strict=True)
    )
