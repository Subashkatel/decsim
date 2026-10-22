"""The region sliding row: each window commits a region and reads the next.

A stream is cut into commit regions of at most the code's commit rounds
(windows/region_streams.py); a window commits one and reads the next
region of its stream as its buffer, Bombín et al. 2303.04846's
neighbouring block as the buffer (lines 1262-1265), and a coupled
region also reads the region it is coupled to, whose window waits for
it, in the coupling order the settings name. Regions line up across a
coupling, so a stream is cut where a coupling begins and ends, and only
there. With the coordinate order on a surface code's patches this row
lays SWIPER's sliding windows (2412.05115 Sec. 3; swiper/window_builder.py
and swiper/window_manager.py SlidingWindowManager) on every schedule
the tests replay. One difference: SWIPER cuts where merge rounds meet
other rounds, so two merges back to back on a patch share a region
there, and here each is cut on its own (windows/region_layout.py).
"""

import dataclasses
from typing import Optional

import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.coupling_orders as coupling_orders
import decsim.windows.region_streams as region_streams
import decsim.windows.schemes.window_data as window_data


class RegionSlidingScheme:
    """Region windows, each reading the next region and its coupled ones."""

    has_trailing_tail_context = False
    # a coupled region's windows wait across streams, not along one chain
    commits_in_one_serial_chain = False
    supports_dynamic_streams = True
    reads_across_couplings = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The region length, None being the code distance, and the order.

        A window's buffer is the whole next region, so no buffer length
        is set. coupling_order is a row of windows/coupling_orders.py.
        """

        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None
        coupling_order: ports.CouplingOrderSettings = (
            coupling_orders.CoordinateOrder.Settings()
        )

        def __post_init__(self) -> None:
            window_data.check_window_sizes(
                self.commit_rounds, self.buffer_rounds
            )
            if self.buffer_rounds is None:
                return
            raise ValueError(
                "a region sliding window reads the whole next region as its "
                f"buffer, so buffer_rounds is None (got {self.buffer_rounds!r})"
            )

        # the word the reports name this row by
        name = "region_sliding"

        def build(self, terminal_policy: str) -> "RegionSlidingScheme":
            """The scheme; its last region reads nothing past the end."""
            del terminal_policy
            coupling_order = self.coupling_order.build()
            return RegionSlidingScheme(coupling_order)

    def __init__(self, coupling_order: ports.CouplingOrder) -> None:
        self.coupling_order = coupling_order

    def plan_operation(
        self,
        stream_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """One stream's regions, each window reading the next region.

        The last region holds what is left and reads nothing past it.
        """
        layouts = self.stream_layouts()
        stream = layouts.open_stream(
            stream_id,
            round_count,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )
        windows = stream.begun(round_count, None)
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

    def stream_layouts(self) -> region_streams.RegionStreamLayouts:
        """Next-region buffers, cut at couplings, read in the coupling order."""
        buffer_rule = region_streams.NextRegionBuffer()
        return region_streams.RegionStreamLayouts(
            buffer_rule,
            is_cut_at_every_segment=False,
            coupling_order=self.coupling_order,
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
