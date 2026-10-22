"""A lattice-surgery run decoded on patch streams (controller/patch_streams.py).

Timing only, as SWIPER-SIM is: each patch life is one decode stream,
every round its patch measures that stream's next round (SWIPER
2412.05115, swiper/device_manager.py _generate_syndrome_round), and an
operation on several patches releases its waiters once its streams and
those of its decoded ancestors committed (windows/release_rules.py
DecodedAncestors, a lattice-surgery run's rule). The
windows are SWIPER's sliding ones (swiper/window_manager.py
SlidingWindowManager), laid by the region layout as the run's merges
start and its rounds arrive; the layout itself is checked against
SWIPER's own windows in tests/windows/test_region_layout.py, so here a
run's windows are checked against the layout of the run's own rounds.
"""

import collections
import pathlib

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.lattice_surgery as lattice_surgery
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.settings as qpu_settings
import decsim.records.windows as window_records
import decsim.windows.coupling_orders as coupling_orders
import decsim.windows.region_layout as region_layout
import decsim.windows.schemes.region_sliding as region_sliding
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.frontends.test_patch_lives as patch_lives_test

HERE = pathlib.Path(__file__)
DATA = HERE.parents[1] / "data"
DISTANCE = 7
# a decode the one unit keeps up with on toffoli's ten or so live
# patches, each sending a round a microsecond; a slower one falls ever
# further behind (Terhal 1302.3428's backlog), and the run never drains
TOFFOLI_DECODE_MICROSECONDS = 0.1
T_GATE_DECODE_MICROSECONDS = 5.0
T_MERGE = 1
T_MERGE_STREAMS = {("patch", (0, 1), 0), ("patch", (0, 0), 0)}
CONDITIONAL_S_MERGE = 3
CONTROL_STREAM = ("patch", (0, 0), 0)
T_STREAM = ("patch", (0, 1), 0)
S_ANCILLA_STREAM = ("patch", (1, 0), 0)


def run(workload, decode_microseconds: float):
    settings = workload_settings.WorkloadSettings.running(workload)
    algorithm = decoders.PresetLatencyDecoder.Settings(decode_microseconds)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme)
    qpu = qpu_settings.QpuSettings(distance=DISTANCE)
    base = machines.base(settings, windows, qpu)
    graph = machines.with_pool(base, pool)
    machine = machine_module.Machine.build(graph, 0)
    result = machine.run()
    return settings, machine, result


def rounds_by_stream(machine) -> dict:
    """Each stream's round indices, in the order they left the QPU."""
    rounds = collections.defaultdict(list)
    for event in machine.observation.round_events.events:
        if event.kind == "EMITTED":
            rounds[event.stream_id].append(event.round_index)
    return rounds


def first_rounds(rounds: dict) -> dict:
    """Each stream's rounds as one run from round 1, as they should be."""
    expected = {}
    for key, each in rounds.items():
        stop = len(each) + 1
        indices = range(1, stop)
        expected[key] = list(indices)
    return expected


@pytest.fixture(scope="module")
def toffoli():
    workload = patch_lives_test.seed_zero_workload("toffoli")
    return run(workload, TOFFOLI_DECODE_MICROSECONDS)


@pytest.fixture(scope="module")
def t_gate(tmp_path_factory):
    folder = tmp_path_factory.mktemp("t_gate")
    path = folder / "t_gate.lss"
    path.write_text(patch_lives_test.T_GATE)
    workload = lattice_surgery.read_schedule(path, DISTANCE)
    return run(workload, T_GATE_DECODE_MICROSECONDS)


def test_every_patch_stream_sends_each_round_once_in_order(toffoli):
    settings, machine, result = toffoli

    rounds = rounds_by_stream(machine)

    assert result.terminal_status == "complete"
    lives = {life.stream_id for life in settings.patch_lives}
    decoded = {key: each for key, each in rounds.items() if key in lives}
    assert set(decoded) == lives
    assert decoded == first_rounds(decoded)


def test_every_patch_stream_is_decoded_in_windows(toffoli):
    settings, machine, _result = toffoli

    windows = machine.observation.windows.windows

    streams = {key[0] for key in windows}
    lives = {life.stream_id for life in settings.patch_lives}
    assert streams == lives


def test_a_two_patch_merge_is_a_segment_of_both_patch_streams(t_gate):
    _settings, machine, _result = t_gate

    bindings = machine.windows.results.bindings
    segments = bindings.segments_by_operation[T_MERGE]

    streams = {segment.stream_id for segment in segments}
    assert streams == T_MERGE_STREAMS


def test_the_conditional_s_waits_for_both_streams_of_the_t_merge(t_gate):
    _settings, machine, result = t_gate

    stalls = {stall.operation_id: stall for stall in result.gate_stalls}
    bindings = machine.windows.results.bindings
    segments = bindings.segments_by_operation[T_MERGE]
    windows = machine.observation.windows.windows.values()

    committing = [_last_committing(windows, segment) for segment in segments]
    released = stalls[CONDITIONAL_S_MERGE].released_ticks
    assert released > max(window.t_done for window in committing)


def _last_committing(windows, segment):
    """The window that commits the segment's last round, a merge's d-th."""
    end = segment.stream_offset + DISTANCE
    committing = []
    for window in windows:
        is_stream = window.stream_id == segment.stream_id
        if is_stream and window.commit_lo <= end <= window.commit_hi:
            committing.append(window)
    (window,) = committing
    return window


def test_a_patch_stream_is_cut_into_regions_at_its_merges(t_gate):
    _settings, machine, _result = t_gate

    windows = machine.observation.windows.windows
    control = [windows[(CONTROL_STREAM, index)] for index in range(5)]

    commits = [(window.commit_lo, window.commit_hi) for window in control]
    buffer_ends = [window.buffer_hi for window in control]
    assert commits == [(1, 7), (8, 14), (15, 19), (20, 26), (27, 32)]
    assert buffer_ends == [14, 19, 26, 32, 32]


def test_a_merge_window_reads_its_partners_region_and_is_waited_for(t_gate):
    _settings, machine, _result = t_gate

    windows = machine.observation.windows.windows
    t_merge = windows[(T_STREAM, 0)]
    s_merge = windows[(S_ANCILLA_STREAM, 0)]
    control_t_merge = windows[(CONTROL_STREAM, 0)]
    control_s_merge = windows[(CONTROL_STREAM, 3)]

    t_region = window_records.RoundSpan(CONTROL_STREAM, 1, 7)
    s_region = window_records.RoundSpan(CONTROL_STREAM, 20, 26)
    assert t_merge.spatial_reads == (t_region,)
    assert s_merge.spatial_reads == (s_region,)
    assert t_merge.key in control_t_merge.deps
    assert s_merge.key in control_s_merge.deps


def test_a_runs_windows_are_the_region_layout_of_its_rounds(toffoli):
    settings, machine, _result = toffoli

    layout = _layout_of_the_run(settings, machine)

    windows = machine.observation.windows.windows.values()
    laid = [_laid(window) for window in windows]
    expected = [_expected(layout, window) for window in windows]
    assert laid == expected


def _layout_of_the_run(settings, machine):
    """The layout of each stream's rounds, cut and coupled at its merges."""
    layout = region_layout.RegionLayout()
    lengths = rounds_by_stream(machine)
    for life in settings.patch_lives:
        layout.open_stream(life.stream_id, DISTANCE)
    bindings = machine.windows.results.bindings
    order = coupling_orders.CoordinateOrder()
    for operation in settings.operations:
        segments = bindings.segments_by_operation.get(operation.id, ())
        if operation.couplings and segments:
            _couple(layout, order, operation, segments)
    for life in settings.patch_lives:
        round_count = len(lengths[life.stream_id])
        layout.extend(life.stream_id, round_count)
        layout.seal(life.stream_id, round_count)
    return layout


def _couple(layout, order, operation, segments) -> None:
    spans_by_patch = {}
    for segment in segments:
        first_round = segment.stream_offset + 1
        last_round = segment.stream_offset + DISTANCE
        span = window_records.RoundSpan(
            segment.stream_id, first_round, last_round
        )
        round_before = first_round - 1
        layout.cut_after(span.stream_id, round_before)
        layout.cut_after(span.stream_id, last_round)
        patch = segment.stream_id[1]
        spans_by_patch[patch] = span
    for blocks in operation.couplings:
        reader, read = order.orient(blocks)
        layout.couple(spans_by_patch[reader], spans_by_patch[read])


def _laid(window) -> tuple:
    """The window as the run laid it: its reads and what it waits for."""
    deps = sorted(window.deps, key=str)
    return (
        window.key,
        window.commit_hi,
        window.buffer_hi,
        window.spatial_reads,
        deps,
    )


def _expected(layout, window) -> tuple:
    """The same, as the layout of the run's rounds lays its region."""
    region = layout.region_from(window.stream_id, window.commit_lo)
    buffer_hi = layout.buffer_last_round(region)
    reads = layout.reads_of(region)
    deps = []
    if window.window_index:
        previous = window.window_index - 1
        deps.append((window.stream_id, previous))
    for reader in layout.readers_of(region):
        reader_index = layout.index_of(reader)
        deps.append((reader.stream_id, reader_index))
    deps.sort(key=str)
    return (window.key, region.last_round, buffer_hi, reads, deps)
