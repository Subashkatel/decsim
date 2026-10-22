"""The region layout: decode streams cut into commit regions.

A stream is cut into regions of at most its code's commit rounds, each
ending early at a cut, and the seal ends the last; the commit regions
partition the rounds (Bombín et al. 2303.04846 Eq. 2). The caller says
where the cuts fall (windows/region_streams.py). A region may read the
region it is coupled to on another stream, at the same offset of one
joint measurement, and that region's window waits for it; the caller
orients each coupling. Streams are opaque, so nothing here reads a
code's geometry.

A stream lets its final windows go (window_release.py), and a region
goes with its window once every region it is coupled to has gone too:
a later decode sees a past region only as the correction committed on
the face it shares with it (Bombín et al. lines 775-788), so nothing is
kept for a region no live window reads. A region keeps its index, its
window's, after the regions before it go.

SWIPER (2412.05115, swiper/window_builder.py build_windows) cuts where
merge rounds meet other rounds, so a merge that runs straight into
another merge on the same patch shares a region with it; cut at each
coupled segment's edges, as region_streams.py does, each is cut on its
own. The two agree whenever such a merge lasts a whole number of
regions, as on every schedule the tests replay
(tests/windows/test_region_layout.py).
"""

import bisect
from typing import Any, Optional

import decsim.records.windows as window_records
import decsim.windows.window_positions as window_positions


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
        # stream id -> the index of its first window still kept
        self.let_go_index_by_stream: dict = {}

    def open_stream(
        self,
        stream_id: Any,  # an opaque identity
        commit_round_count: int,
    ) -> None:
        """A stream whose regions last at most commit_round_count rounds."""
        regions = _StreamRegions(stream_id, commit_round_count)
        self.regions_by_stream[stream_id] = regions

    def cut_after(
        self,
        stream_id: Any,  # an opaque identity
        last_round: int,
    ) -> None:
        """A region of the stream ends on the round; the next starts after."""
        regions = self.regions_by_stream[stream_id]
        regions.cut_after(last_round)

    def couple(
        self,
        reader: window_records.RoundSpan,
        read: window_records.RoundSpan,
    ) -> None:
        """Each region of reader reads the region of read at its offset.

        The two are one joint measurement's segments, of one length, on
        two streams of one region length, each cut at its edges, so both
        are cut into regions at the same offsets. Streams of different
        region lengths, blocks of two codes, are refused: their regions
        would not line up.
        """
        regions = self.regions_by_stream[reader.stream_id]
        stride = regions.commit_round_count
        read_stride = self.regions_by_stream[read.stream_id].commit_round_count
        if read_stride != stride:
            raise ValueError(
                f"streams {reader.stream_id} and {read.stream_id} are coupled "
                f"but their regions last {stride} and {read_stride} rounds; "
                "coupling blocks of different codes is not supported"
            )
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
        starts = regions.starts.kept
        position = bisect.bisect_right(starts, round_index)
        assert position > 0, (
            f"round {round_index} of {stream_id} lies in a region let go"
        )
        first_round = starts[position - 1]
        return regions.span_from(first_round)

    def let_go_before(
        self,
        stream_id: Any,  # an opaque identity
        region_index: int,
    ) -> None:
        """The stream's windows before the index are gone; so are its regions.

        A region read across a coupling goes once its partner has gone
        too, so this lets go what each partner kept for the stream.
        """
        self.let_go_index_by_stream[stream_id] = region_index
        partner_stream_ids = self._drop_let_go(stream_id)
        for partner_stream_id in partner_stream_ids:
            self._drop_let_go(partner_stream_id)

    def _drop_let_go(self, stream_id) -> set:
        """Drop the stream's let-go regions from the front, while it may.

        Returns the streams coupled to any let-go region, gone or kept.
        """
        regions = self.regions_by_stream[stream_id]
        let_go_index = self.let_go_index_by_stream.get(stream_id, 0)
        partner_stream_ids = set()
        while regions.has_let_go_front(let_go_index):
            key = (stream_id, regions.starts.kept[0])
            partners = self._partners_of(key)
            partner_stream_ids.update(partner[0] for partner in partners)
            if not all(self._is_let_go(partner) for partner in partners):
                break
            self._forget_couplings(key)
            regions.drop_front()
        return partner_stream_ids

    def _partners_of(self, key: tuple) -> list:
        """The regions the region reads, then the regions reading it."""
        reads = self.reads_by_region.get(key, [])
        readers = self.readers_by_region.get(key, [])
        return [*reads, *readers]

    def _is_let_go(self, key: tuple) -> bool:
        """Whether the region's window is gone from its stream."""
        stream_id, first_round = key
        regions = self.regions_by_stream[stream_id]
        let_go_index = self.let_go_index_by_stream.get(stream_id, 0)
        return regions.is_let_go(first_round, let_go_index)

    def _forget_couplings(self, key: tuple) -> None:
        """The region's couplings, gone from both of its tables."""
        for read in self.reads_by_region.pop(key, []):
            _remove_entry(self.readers_by_region, read, key)
        for reader in self.readers_by_region.pop(key, []):
            _remove_entry(self.reads_by_region, reader, key)

    def _spans(self, keys) -> tuple:
        spans = []
        for stream_id, first_round in keys:
            span = self.region_from(stream_id, first_round)
            spans.append(span)
        return tuple(spans)


class _StreamRegions:
    """One stream's region starts, its cuts and its seal.

    The starts are kept by window position, a region's index being its
    window's, so the front goes as the stream lets its windows go.
    """

    def __init__(
        self,
        stream_id: Any,  # an opaque identity
        commit_round_count: int,
    ) -> None:
        self.stream_id = stream_id
        self.commit_round_count = commit_round_count
        # each kept region's first round, ascending
        self.starts = window_positions.WindowPositions()
        # rounds a kept region may end on, ascending
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
            self.starts.append(next_first)
            region = self.span_from(next_first)
            begun.append(region)
            next_first = region.last_round + 1
        return begun

    def index_from(self, first_round: int) -> Optional[int]:
        starts = self.starts.kept
        position = bisect.bisect_left(starts, first_round)
        if position == len(starts):
            return None
        assert starts[position] == first_round, (
            f"round {first_round} of {self.stream_id} begins no kept region"
        )
        return self.starts.first_kept + position

    def is_let_go(self, first_round: int, let_go_index: int) -> bool:
        """Whether the region from the round has its window gone."""
        if first_round < self.starts.kept[0]:
            return True
        index = self.index_from(first_round)
        return index < let_go_index

    def has_let_go_front(self, let_go_index: int) -> bool:
        """Whether the first kept region's window is gone.

        The last region begun stays, since the next one starts after it.
        """
        if len(self.starts.kept) < 2:
            return False
        return self.starts.first_kept < let_go_index

    def drop_front(self) -> None:
        """The first kept region goes, and the cuts no kept region reads."""
        next_index = self.starts.first_kept + 1
        self.starts.let_go_before(next_index)
        first_kept = self.starts.kept[0]
        cut_count = bisect.bisect_left(self.cut_rounds, first_kept)
        del self.cut_rounds[:cut_count]

    def span_from(self, first_round: int) -> window_records.RoundSpan:
        last_round = self._last_round_of(first_round)
        return window_records.RoundSpan(self.stream_id, first_round, last_round)

    def _next_first_round(self) -> int:
        starts = self.starts.kept
        if not starts:
            return 1
        last_first = starts[-1]
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


def _remove_entry(table: dict, key: tuple, entry: tuple) -> None:
    """The entry, gone from the key's list; an empty list goes too."""
    entries = table[key]
    entries.remove(entry)
    if not entries:
        del table[key]
