"""The region layout: decode streams cut into regions, a window on each.

A stream is cut into regions of at most its code's commit rounds; a
coupled segment, the rounds of a joint measurement, begins and ends
regions, and the seal ends the last (SWIPER 2412.05115,
swiper/window_builder.py build_windows; the commit regions partition the
rounds, Bombín et al. 2303.04846 Eq. 2). A window commits one region and
reads the next region of its stream as its buffer; a coupled region also
reads the region it is coupled to on another stream, and that region's
window waits for it (swiper/window_manager.py
SlidingWindowManager.process_round). Streams are opaque and the caller
orients each coupling, so nothing here reads a code's geometry.

SWIPER cuts where merge rounds meet other rounds, so a merge that runs
straight into another merge on the same patch shares a region with it;
here each coupled segment is cut on its own. The two agree whenever such
a merge lasts a whole number of regions, as on every schedule the tests
replay (tests/windows/test_region_layout.py).
"""

import bisect
from typing import Any, Optional

import decsim.records.windows as window_records


class RegionLayout:
    """Every stream's regions, and the regions each window reads.

    A region begins once its first round is known. The last region begun
    ends where its stream is next cut, so it ends earlier when a later
    cut or the seal falls inside it; a cut never falls before a known
    round, so every earlier region is final.
    """

    def __init__(self) -> None:
        self.regions_by_stream: dict = {}
        # (stream id, first round) of a region -> the (stream id, first
        # round) of each region it reads across a coupling
        self.reads_by_region: dict = {}
        # the same pairs the other way: a region -> each region reading it
        self.readers_by_region: dict = {}

    def open_stream(
        self,
        stream_id: Any,  # an opaque identity
        commit_round_count: int,
    ) -> None:
        """A stream whose regions last at most commit_round_count rounds."""
        regions = _StreamRegions(stream_id, commit_round_count)
        self.regions_by_stream[stream_id] = regions

    def cut_segment(self, segment: window_records.RoundSpan) -> None:
        """A coupled segment's rounds: regions begin and end with it."""
        regions = self.regions_by_stream[segment.stream_id]
        round_before = segment.first_round - 1
        regions.cut_after(round_before)
        regions.cut_after(segment.last_round)

    def couple(
        self,
        reader: window_records.RoundSpan,
        read: window_records.RoundSpan,
    ) -> None:
        """Each region of reader reads the region of read at its offset.

        The two are one joint measurement's segments, of one length, on
        two streams of one region length, each cut by cut_segment, so
        both are cut into regions at the same offsets.
        """
        regions = self.regions_by_stream[reader.stream_id]
        stride = regions.commit_round_count
        offsets = range(0, reader.round_count, stride)
        for offset in offsets:
            reading_first = reader.first_round + offset
            read_first = read.first_round + offset
            reading = (reader.stream_id, reading_first)
            read_region = (read.stream_id, read_first)
            reads = self.reads_by_region.setdefault(reading, [])
            reads.append(read_region)
            readers = self.readers_by_region.setdefault(read_region, [])
            readers.append(reading)

    def extend(
        self,
        stream_id: Any,  # an opaque identity
        round_index: int,
    ) -> list:
        """The stream's rounds through round_index are known.

        Returns the regions that began, in order.
        """
        regions = self.regions_by_stream[stream_id]
        return regions.extend(round_index)

    def seal(
        self,
        stream_id: Any,  # an opaque identity
        round_count: int,
    ) -> None:
        """The stream ends after round_count rounds."""
        regions = self.regions_by_stream[stream_id]
        regions.sealed_round_count = round_count

    def region_from(
        self,
        stream_id: Any,  # an opaque identity
        first_round: int,
    ) -> window_records.RoundSpan:
        """The stream's region from that round, as far as it is known."""
        regions = self.regions_by_stream[stream_id]
        return regions.span_from(first_round)

    def buffer_last_round(self, region: window_records.RoundSpan) -> int:
        """The last round of the next region, or the region's own at the seal.

        The next region need not have begun: it starts the round after
        this one, and its end is known as far as this one's is.
        """
        regions = self.regions_by_stream[region.stream_id]
        if region.last_round == regions.sealed_round_count:
            return region.last_round
        next_first = region.last_round + 1
        next_region = regions.span_from(next_first)
        return next_region.last_round

    def reads_of(self, region: window_records.RoundSpan) -> tuple:
        """The regions this one reads across its couplings."""
        key = (region.stream_id, region.first_round)
        reads = self.reads_by_region.get(key, ())
        return self._spans(reads)

    def readers_of(self, region: window_records.RoundSpan) -> tuple:
        """The regions reading this one across their couplings."""
        key = (region.stream_id, region.first_round)
        readers = self.readers_by_region.get(key, ())
        return self._spans(readers)

    def index_of(self, region: window_records.RoundSpan) -> Optional[int]:
        """The region's place in its stream, from 0; None until it begins."""
        regions = self.regions_by_stream[region.stream_id]
        return regions.index_from(region.first_round)

    def region_holding(
        self,
        stream_id: Any,  # an opaque identity
        round_index: int,
    ) -> window_records.RoundSpan:
        """The begun region the round falls in."""
        regions = self.regions_by_stream[stream_id]
        position = bisect.bisect_right(regions.first_rounds, round_index)
        first_round = regions.first_rounds[position - 1]
        return regions.span_from(first_round)

    def _spans(self, keys) -> tuple:
        spans = []
        for stream_id, first_round in keys:
            span = self.region_from(stream_id, first_round)
            spans.append(span)
        return tuple(spans)


class _StreamRegions:
    """One stream's region starts, its cuts and its seal."""

    def __init__(
        self,
        stream_id: Any,  # an opaque identity
        commit_round_count: int,
    ) -> None:
        self.stream_id = stream_id
        self.commit_round_count = commit_round_count
        # each begun region's first round, ascending
        self.first_rounds: list = []
        # rounds a region ends on, ascending
        self.cut_rounds: list = []
        self.sealed_round_count: Optional[int] = None
        self.known_round = 0

    def cut_after(self, round_index: int) -> None:
        assert round_index >= self.known_round, (
            f"a cut after round {round_index} of {self.stream_id} falls "
            f"before its known round {self.known_round}"
        )
        bisect.insort(self.cut_rounds, round_index)

    def extend(self, round_index: int) -> list:
        self.known_round = max(self.known_round, round_index)
        begun = []
        next_first = self._next_first_round()
        while next_first <= self.known_round:
            self.first_rounds.append(next_first)
            region = self.span_from(next_first)
            begun.append(region)
            next_first = region.last_round + 1
        return begun

    def index_from(self, first_round: int) -> Optional[int]:
        position = bisect.bisect_left(self.first_rounds, first_round)
        if position == len(self.first_rounds):
            return None
        assert self.first_rounds[position] == first_round, (
            f"round {first_round} of {self.stream_id} begins no region"
        )
        return position

    def span_from(self, first_round: int) -> window_records.RoundSpan:
        last_round = self._last_round_of(first_round)
        return window_records.RoundSpan(self.stream_id, first_round, last_round)

    def _next_first_round(self) -> int:
        if not self.first_rounds:
            return 1
        last_first = self.first_rounds[-1]
        last_round = self._last_round_of(last_first)
        return last_round + 1

    def _last_round_of(self, first_round: int) -> int:
        """A region ends a stride on, at the first cut, or at the seal."""
        stride_end = first_round + self.commit_round_count - 1
        ends = [stride_end]
        cut_index = bisect.bisect_left(self.cut_rounds, first_round)
        if cut_index < len(self.cut_rounds):
            ends.append(self.cut_rounds[cut_index])
        if self.sealed_round_count is not None:
            ends.append(self.sealed_round_count)
        return min(ends)
