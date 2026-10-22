"""Growing streams laid in commit regions, one window on each.

Every growing stream is cut by one law (windows/region_layout.py): a
region lasts at most the code's commit rounds and ends early at a cut
or at the end of the stream, and the commit regions partition the
rounds (Bombín et al. 2303.04846 Eq. 2). A window commits one region.
Three choices sit on top of that law, and each scheme makes them:

- How far a window reads past its commit. FixedBuffer reads a fixed
  count of rounds (Skoric et al. 2209.08552 Sec. I.B); NextRegionBuffer
  reads through the next region, Bombín's neighbouring block as the
  buffer (lines 1262-1265), as SWIPER does (2412.05115 Sec. 3).
- Which segment edges cut a region. A run that reads each segment's
  result from whole windows (windows/operation_results.py) cuts at every
  segment edge; a run that needs regions only to line up across its
  couplings cuts where a coupling begins and ends, so a region is
  coupled throughout or not at all.
- Which block of a coupling reads the other's region (a CouplingOrder,
  windows/coupling_orders.py); the read region's window waits for it.

Nothing here reads a code's geometry: streams and blocks are opaque.
"""

import dataclasses
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.region_layout as region_layout


class FixedBuffer:
    """A window reads a fixed count of rounds past its commit.

    At a closed tail the window holding it is clipped: its buffer still
    counts past the tail, but no round arrives there, so it reads none.
    """

    def last_round(
        self,
        regions: region_layout.RegionLayout,
        region: window_records.RoundSpan,
        buffer_round_count: int,
    ) -> int:
        """The last round the region's window reads: B past its commit."""
        del regions
        return region.last_round + buffer_round_count

    def tail_change(
        self, holding: Optional[window_records.Window], moved: list
    ) -> window_records.TailChange:
        """The window holding the tail reads nothing past its commit."""
        del moved
        if holding is None:
            return window_records.TailChange()
        return window_records.TailChange(clipped=(holding,))


class NextRegionBuffer:
    """A window reads through the next region of its stream.

    The next region need not have begun: it starts the round after this
    one and its end is known as far as this one's is. At the end of the
    stream a window reads only its own region, so a closed tail lays the
    open windows again and each reads its whole new extent.
    """

    def last_round(
        self,
        regions: region_layout.RegionLayout,
        region: window_records.RoundSpan,
        buffer_round_count: int,
    ) -> int:
        """The last round the region's window reads: the next region's."""
        del buffer_round_count
        return regions.buffer_last_round(region)

    def tail_change(
        self, holding: Optional[window_records.Window], moved: list
    ) -> window_records.TailChange:
        """Every window the tail moved reads its new extent."""
        del holding
        return window_records.TailChange(relaid=tuple(moved))


class RegionStreamLayouts:
    """A run's growing streams, cut by one region law.

    buffer_rule is FixedBuffer or NextRegionBuffer; is_cut_at_every_segment
    says whether every segment edge cuts a region or only a coupling's;
    coupling_order is the CouplingOrder of a run that couples blocks.
    """

    def __init__(
        self,
        buffer_rule: "FixedBuffer | NextRegionBuffer",
        is_cut_at_every_segment: bool,
        coupling_order: Optional[ports.CouplingOrder] = None,
    ) -> None:
        self.regions = region_layout.RegionLayout()
        self.buffer_rule = buffer_rule
        self.is_cut_at_every_segment = is_cut_at_every_segment
        self.coupling_order = coupling_order
        self.stream_by_id: dict = {}

    def open_stream(
        self,
        stream_id: Any,  # an opaque identity
        round_limit: Optional[int],
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> "RegionStream":
        """The stream's layout, ending at round_limit when the source knows."""
        self.regions.open_stream(stream_id, commit_round_count)
        if round_limit is not None:
            self.regions.seal(stream_id, round_limit)
        stream = RegionStream(self, stream_id, buffer_round_count)
        self.stream_by_id[stream_id] = stream
        return stream

    def couple_segments(self, spans_by_block: dict, couplings: tuple) -> None:
        """Each segment begins and ends regions; each coupling is read across.

        A coupling the order orients neither way is read by neither side.
        """
        for span in spans_by_block.values():
            stream = self.stream_by_id[span.stream_id]
            stream.cut_segment(span)
        for blocks in couplings:
            oriented = self.coupling_order.orient(blocks)
            if oriented is None:
                continue
            reader, read = oriented
            reader_span = spans_by_block[reader]
            read_span = spans_by_block[read]
            self.regions.couple(reader_span, read_span)

    def window_keys(self, regions: tuple) -> list:
        """The key of the window on each region; None for one not begun."""
        keys = []
        for region in regions:
            window_index = self.regions.index_of(region)
            key = None
            if window_index is not None:
                key = (region.stream_id, window_index)
            keys.append(key)
        return keys


class RegionStream:
    """One growing stream's windows, one on each region as it begins."""

    def __init__(
        self,
        layouts: RegionStreamLayouts,
        stream_id: Any,  # an opaque identity
        buffer_round_count: int,
    ) -> None:
        self.layouts = layouts
        self.stream_id = stream_id
        self.buffer_round_count = buffer_round_count
        # rounds a segment ends on, in the order they were cut
        self.segment_cuts: list = []

    def begun(self, highest_known_round: int, round_cap: Optional[int]) -> list:
        """A geometry on each region that began; round_cap ends the stream."""
        regions = self.layouts.regions
        if round_cap is not None:
            regions.seal(self.stream_id, round_cap)
        begun = regions.extend(self.stream_id, highest_known_round)
        return [self._geometry(region) for region in begun]

    def cut_after(self, windows: list, last_round: int) -> list:
        """A segment ends on the round; the open windows that moved."""
        if not self.layouts.is_cut_at_every_segment:
            return []
        self.layouts.regions.cut_after(self.stream_id, last_round)
        self.segment_cuts.append(last_round)
        return self.relay(windows)

    def cut_segment(self, segment: window_records.RoundSpan) -> None:
        """A coupled segment's rounds begin and end regions."""
        round_before = segment.first_round - 1
        self.layouts.regions.cut_after(self.stream_id, round_before)
        self.layouts.regions.cut_after(self.stream_id, segment.last_round)
        self.segment_cuts.append(round_before)
        self.segment_cuts.append(segment.last_round)

    def close_tail(
        self, windows: list, last_round: int
    ) -> window_records.TailChange:
        """The stream's rounds end on the round, for now or for good."""
        self.layouts.regions.cut_after(self.stream_id, last_round)
        moved = self.relay(windows)
        holding = _holding(windows, last_round)
        return self.layouts.buffer_rule.tail_change(holding, moved)

    def relay(self, windows: list) -> list:
        """Lay each open window on its region again; return those that moved.

        A cut or the end of the stream ends the last region begun earlier
        than its stride, and with a next-region buffer the window before.
        """
        moved = []
        for window in windows:
            if window.queued or window.committed:
                continue
            extent_before = (window.commit_hi, window.buffer_hi)
            self._lay(window)
            extent_after = (window.commit_hi, window.buffer_hi)
            if extent_after != extent_before:
                moved.append(window)
        return moved

    def cut_rounds(self) -> tuple:
        """The rounds a segment of the stream ends on, in the order cut."""
        return tuple(self.segment_cuts)

    def reads_of(self, window: window_records.Window) -> tuple:
        """The key of each begun window this one reads across a coupling."""
        region = _region_of(window)
        reads = self.layouts.regions.reads_of(region)
        keys = self.layouts.window_keys(reads)
        return tuple(key for key in keys if key is not None)

    def readers_of(self, window: window_records.Window) -> tuple:
        """The key of each window reading this one; None for one not begun."""
        region = _region_of(window)
        readers = self.layouts.regions.readers_of(region)
        keys = self.layouts.window_keys(readers)
        return tuple(keys)

    def readers_at(self, round_index: int) -> tuple:
        """The keys of the begun windows reading the region of the round."""
        regions = self.layouts.regions
        region = regions.region_holding(self.stream_id, round_index)
        readers = regions.readers_of(region)
        keys = self.layouts.window_keys(readers)
        return tuple(key for key in keys if key is not None)

    def let_go_before(self, window_index: int) -> None:
        """The regions before the index go once their partners have gone.

        The segment cuts stay: a segment's result may still be read from
        its first round over folded windows (committed_rounds.py
        fold_before), and they number the segments, not the rounds.
        """
        self.layouts.regions.let_go_before(self.stream_id, window_index)

    def _geometry(
        self, region: window_records.RoundSpan
    ) -> window_records.WindowGeometry:
        """The window on the region: it commits it and reads past it."""
        buffer_hi = self._buffer_last_round(region)
        spatial_reads = self.layouts.regions.reads_of(region)
        return window_records.WindowGeometry(
            buffer_lo=region.first_round,
            commit_lo=region.first_round,
            commit_hi=region.last_round,
            buffer_hi=buffer_hi,
            spatial_reads=spatial_reads,
        )

    def _lay(self, window: window_records.Window) -> None:
        """Its commit, buffer and spatial reads, as its region now ends.

        Its read start stays where it is: a strong restart may have
        moved it before the commit (WindowPlanner.reslice_window).
        """
        regions = self.layouts.regions
        region = regions.region_from(self.stream_id, window.commit_lo)
        laid = self._geometry(region)
        geometry = dataclasses.replace(laid, buffer_lo=window.start_round)
        window.commit_hi = geometry.commit_hi
        window.buffer_hi = geometry.buffer_hi
        window.spatial_reads = geometry.spatial_reads
        window.round_count = geometry.round_count

    def _buffer_last_round(self, region: window_records.RoundSpan) -> int:
        rule = self.layouts.buffer_rule
        regions = self.layouts.regions
        return rule.last_round(regions, region, self.buffer_round_count)


def _holding(windows: list, last_round: int) -> Optional[window_records.Window]:
    """The window whose commit holds the round, if one is laid."""
    for window in windows:
        if window.commit_lo <= last_round <= window.commit_hi:
            return window
    return None


def _region_of(window: window_records.Window) -> window_records.RoundSpan:
    """The region a window commits."""
    return window_records.RoundSpan(
        window.stream_id, window.commit_lo, window.commit_hi
    )
