"""The region layout (windows/region_layout.py) beside SWIPER's windows.

The referent is SWIPER's own window builder and sliding window manager
(github.com/jviszlai/swiper, swiper/window_builder.py and
swiper/window_manager.py SlidingWindowManager) run on its own device at
seed 0: tools/swiper_referent.py wrote each patch life's rounds, as runs
of one instruction or of idling, and every window SWIPER constructed
from them, with its dependency edges, into the json beside each
schedule. The replay here feeds the same rounds to the layout a cycle at
a time, as the machine does: a merge's segments when it starts, each
round as it is measured, a life's seal after its last round.
"""

import collections
import json
import pathlib

import pytest

import decsim.records.windows as window_records
import decsim.windows.region_layout as region_layout
import decsim.windows.schemes.swiper_sliding as swiper_sliding

HERE = pathlib.Path(__file__)
DATA = HERE.parents[1] / "data"
DISTANCE = 7
IDLE = -1


def span(stream_id, first_round: int, last_round: int):
    return window_records.RoundSpan(stream_id, first_round, last_round)


def stream_layout(round_count: int, *segments):
    """One stream of 7-round regions, its coupled segments cut, sealed."""
    layout = region_layout.RegionLayout()
    layout.open_stream("a", 7)
    for segment in segments:
        layout.cut_segment(segment)
    layout.extend("a", round_count)
    layout.seal("a", round_count)
    return layout


def regions_of(layout, stream_id, round_count: int) -> list:
    """The stream's regions in order, each as far as it is known."""
    regions = []
    first_round = 1
    while first_round <= round_count:
        region = layout.region_from(stream_id, first_round)
        regions.append(region)
        first_round = region.last_round + 1
    return regions


def test_a_stream_is_cut_into_regions_of_its_commit_rounds():
    layout = stream_layout(20)

    regions = regions_of(layout, "a", 20)

    assert regions == [span("a", 1, 7), span("a", 8, 14), span("a", 15, 20)]


def test_a_window_reads_the_next_region_and_the_last_reads_nothing_more():
    layout = stream_layout(20)

    first, second, last = regions_of(layout, "a", 20)

    assert layout.buffer_last_round(first) == 14
    assert layout.buffer_last_round(second) == 20
    assert layout.buffer_last_round(last) == 20


def test_a_coupled_segment_begins_and_ends_its_own_regions():
    merge = span("a", 4, 10)
    layout = stream_layout(12, merge)

    regions = regions_of(layout, "a", 12)

    assert regions == [span("a", 1, 3), span("a", 4, 10), span("a", 11, 12)]


def test_the_last_region_begun_ends_early_at_a_later_cut():
    layout = region_layout.RegionLayout()
    layout.open_stream("a", 7)
    (begun,) = layout.extend("a", 2)
    merge = span("a", 6, 12)

    layout.cut_segment(merge)
    region = layout.region_from("a", 1)

    assert begun == span("a", 1, 7)
    assert region == span("a", 1, 5)


def test_a_reader_reads_the_region_it_is_coupled_to_and_is_read_by_it():
    layout = region_layout.RegionLayout()
    layout.open_stream("a", 7)
    layout.open_stream("b", 7)
    reader = span("a", 1, 7)
    read = span("b", 3, 9)
    layout.cut_segment(reader)
    layout.cut_segment(read)

    layout.couple(reader, read)
    layout.extend("a", 7)
    layout.extend("b", 9)

    assert layout.reads_of(reader) == (read,)
    assert layout.readers_of(read) == (reader,)
    assert layout.index_of(read) == 1
    assert layout.region_holding("b", 5) == read


def test_a_region_not_begun_has_no_place_yet():
    layout = region_layout.RegionLayout()
    layout.open_stream("a", 7)
    layout.extend("a", 3)
    next_region = span("a", 8, 14)

    place = layout.index_of(next_region)

    assert place is None


@pytest.mark.parametrize("name", ("toffoli", "msd_15to1"))
def test_the_layout_of_swipers_rounds_is_swipers_sliding_windows(name):
    reading = _referent(name)
    replay = _Replay(reading)

    windows, edges = replay.windows_and_edges()

    swiper_windows = _swiper_windows(reading)
    swiper_edges = _swiper_edges(reading)
    assert sorted(windows) == sorted(swiper_windows)
    assert sorted(edges) == sorted(swiper_edges)


def _referent(name: str) -> dict:
    path = DATA / f"swiper_{name}_d{DISTANCE}.json"
    text = path.read_text()
    return json.loads(text)


def _swiper_windows(reading: dict) -> list:
    """Each window as its commit region and its sorted buffer regions."""
    windows = []
    for window in reading["sliding_windows"]:
        (commit,) = window["commit"]
        buffers = [_region_key(region) for region in window["buffers"]]
        buffers.sort()
        commit_key = _region_key(commit)
        windows.append((commit_key, tuple(buffers)))
    return windows


def _swiper_edges(reading: dict) -> list:
    edges = []
    for earlier, later in reading["sliding_window_edges"]:
        earlier_key = _region_key(earlier)
        later_key = _region_key(later)
        edges.append((earlier_key, later_key))
    return edges


def _region_key(region: list) -> tuple:
    patch, first_round, round_count = region
    return (tuple(patch), first_round, round_count)


class _Replay:
    """SWIPER's lives replayed on the layout, each life a stream."""

    def __init__(self, reading: dict) -> None:
        self.layout = region_layout.RegionLayout()
        self.scheme = swiper_sliding.SwiperSlidingScheme()
        # stream id -> the cycle of its first decoded round
        self.first_cycle_by_stream: dict = {}
        # cycle -> what happens in it: merges starting, rounds, seals
        self.cycles = collections.defaultdict(_Cycle)
        self.lengths_by_stream: dict = {}
        self.patch_by_stream: dict = {}
        self._read(reading)

    def windows_and_edges(self) -> tuple:
        for cycle in sorted(self.cycles):
            self._run_cycle(cycle)
        windows = []
        edges = []
        for stream_id, round_count in self.lengths_by_stream.items():
            regions = regions_of(self.layout, stream_id, round_count)
            self._collect(regions, windows, edges)
        return windows, edges

    def _read(self, reading: dict) -> None:
        decoded = reading["decoded"]
        segments_by_merge = collections.defaultdict(dict)
        for stream_id, life in enumerate(reading["patch_lives"]):
            runs = [run for run in life["runs"] if _is_decoded(run, decoded)]
            if runs:
                self._read_life(stream_id, life, runs, segments_by_merge)
        faces = reading["merge_faces"]
        for merge, segments in segments_by_merge.items():
            self._read_merge(segments, faces[merge])

    def _read_life(self, stream_id, life, runs, segments_by_merge) -> None:
        patch = tuple(life["patch"])
        self.patch_by_stream[stream_id] = patch
        first_cycle = runs[0][0]
        self.first_cycle_by_stream[stream_id] = first_cycle
        self.layout.open_stream(stream_id, DISTANCE)
        for cycle, round_count, instruction in runs:
            first_round = cycle - first_cycle + 1
            last_round = first_round + round_count - 1
            for offset in range(round_count):
                self.cycles[cycle + offset].rounds.append(stream_id)
            if instruction != IDLE:
                segment = span(stream_id, first_round, last_round)
                segments_by_merge[instruction][patch] = segment
        last_cycle = cycle + round_count - 1
        self.cycles[last_cycle].seals.append((stream_id, last_round))
        self.lengths_by_stream[stream_id] = last_round

    def _read_merge(self, segments: dict, faces: list) -> None:
        """A merge's segments, when it has faces, coupled as SWIPER orients."""
        if not faces:
            return
        couplings = []
        for first, second in faces:
            blocks = (tuple(first), tuple(second))
            couplings.append(blocks)
        segment, *_ = segments.values()
        first_cycle = self.first_cycle_by_stream[segment.stream_id]
        cycle = first_cycle + segment.first_round - 1
        self.cycles[cycle].merges.append((segments, couplings))

    def _run_cycle(self, cycle: int) -> None:
        happening = self.cycles[cycle]
        for spans_by_patch, couplings in happening.merges:
            self._couple(spans_by_patch, couplings)
        for stream_id in happening.rounds:
            first_cycle = self.first_cycle_by_stream[stream_id]
            round_index = cycle - first_cycle + 1
            self.layout.extend(stream_id, round_index)
        for stream_id, round_count in happening.seals:
            self.layout.seal(stream_id, round_count)

    def _couple(self, spans_by_patch: dict, couplings: list) -> None:
        for segment in spans_by_patch.values():
            self.layout.cut_segment(segment)
        for blocks in couplings:
            reader, read = self.scheme.orient_coupling(blocks)
            reader_span = spans_by_patch[reader]
            read_span = spans_by_patch[read]
            self.layout.couple(reader_span, read_span)

    def _collect(self, regions: list, windows: list, edges: list) -> None:
        """Each region's window as SWIPER names it, with its edges."""
        earlier = None
        for region in regions:
            key = self._key(region)
            reads = self.layout.reads_of(region)
            buffer_keys = [self._key(read) for read in reads]
            for read_key in buffer_keys:
                edges.append((key, read_key))
            buffer_last_round = self.layout.buffer_last_round(region)
            if buffer_last_round > region.last_round:
                next_first = region.last_round + 1
                next_region = self.layout.region_from(
                    region.stream_id, next_first
                )
                next_key = self._key(next_region)
                buffer_keys.append(next_key)
            if earlier is not None:
                edges.append((earlier, key))
            buffer_keys.sort()
            windows.append((key, tuple(buffer_keys)))
            earlier = key

    def _key(self, region) -> tuple:
        """The region as SWIPER names it: patch, first cycle, rounds."""
        stream_id = region.stream_id
        first_cycle = self.first_cycle_by_stream[stream_id]
        cycle = first_cycle + region.first_round - 1
        patch = self.patch_by_stream[stream_id]
        return (patch, cycle, region.round_count)


class _Cycle:
    def __init__(self) -> None:
        self.merges: list = []
        self.rounds: list = []
        self.seals: list = []


def _is_decoded(run: list, decoded: list) -> bool:
    """An idle run, or one of an instruction whose rounds are decoded."""
    instruction = run[2]
    if instruction == IDLE:
        return True
    return decoded[instruction]
