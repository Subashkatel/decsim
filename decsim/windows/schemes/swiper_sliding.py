"""The SWIPER sliding row: region windows across patches, one after another.

SWIPER 2412.05115 Sec. 3 and its code (github.com/jviszlai/swiper,
swiper/window_builder.py and swiper/window_manager.py
SlidingWindowManager): a window commits one region of a patch, at most
the commit rounds long and cut at each merge's edges, and reads the
next region of the patch as its buffer; across a merge face the patch
whose coordinates are both at least the other's reads the other's region
too, and the other's window waits for it. The regions and their reads
are the region layout's (windows/region_layout.py); this row says how
long a region is, how a coupling is oriented and when a window has its
data.
"""

import dataclasses
from typing import Optional

import decsim.records.windows as window_records
import decsim.windows.region_layout as region_layout
import decsim.windows.schemes.window_data as window_data


class SwiperSlidingScheme:
    """Region windows, each reading the next region and its merge partners."""

    has_trailing_tail_context = False
    # a merge's windows wait across patches, not only along one chain
    commits_in_one_serial_chain = False
    supports_dynamic_streams = True
    lays_regions = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The region length; None is the code distance.

        A window's buffer is the whole next region (SWIPER's
        SlidingWindowManager), so no buffer length is set.
        """

        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None

        def __post_init__(self) -> None:
            window_data.check_window_sizes(
                self.commit_rounds, self.buffer_rounds
            )
            if self.buffer_rounds is None:
                return
            raise ValueError(
                "the SWIPER sliding window reads the whole next region as "
                f"its buffer, so buffer_rounds is None (got "
                f"{self.buffer_rounds!r})"
            )

        # the word the reports name this row by
        name = "swiper_sliding"

        def build(self, terminal_policy: str) -> "SwiperSlidingScheme":
            """The scheme; its last region drains as SWIPER's does."""
            del terminal_policy
            return SwiperSlidingScheme()

    def plan_operation(
        self,
        stream_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """One stream's regions, each window reading the next region.

        The last region holds what is left and reads nothing past it,
        SWIPER's region before a discard.
        """
        del buffer_round_count
        layout = region_layout.RegionLayout()
        layout.open_stream(stream_id, commit_round_count)
        layout.seal(stream_id, round_count)
        regions = layout.extend(stream_id, round_count)
        windows = []
        for region in regions:
            buffer_hi = layout.buffer_last_round(region)
            geometry = window_records.WindowGeometry(
                buffer_lo=region.first_round,
                commit_lo=region.first_round,
                commit_hi=region.last_round,
                buffer_hi=buffer_hi,
            )
            windows.append(geometry)
        window_count = len(windows)
        last_index = window_count - 1
        dependencies = []
        for window_index in range(last_index):
            next_index = window_index + 1
            dependencies.append((window_index, next_index))
        return window_records.OperationWindowPlan(
            stream_id=stream_id,
            windows=tuple(windows),
            internal_dependencies=tuple(dependencies),
            entry_window_indices=(0,),
            exit_window_indices=(last_index,),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether its rounds and every region it reads across are in."""
        if not readiness.spatial_reads_arrived:
            return False
        return window_data.sliding_data_complete(window, readiness)

    def orient_coupling(self, blocks: tuple) -> Optional[tuple]:
        """The coupled patches as (reader, read); None when neither reads.

        SWIPER's rule (SlidingWindowManager.process_round): a patch reads
        its partner when both its coordinates are at least the partner's.
        A merge face joins neighbouring patches, so exactly one reads.
        """
        first, second = blocks
        if _is_at_least(first, second):
            return (first, second)
        if _is_at_least(second, first):
            return (second, first)
        return None


def _is_at_least(patch: tuple, other: tuple) -> bool:
    """Whether each coordinate of the patch is at least the other's."""
    return all(
        mine >= theirs for mine, theirs in zip(patch, other, strict=True)
    )
