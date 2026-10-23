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

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.lattice_surgery as lattice_surgery
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.records.workload as workload_records
import decsim.windows.coupling_orders as coupling_orders
import decsim.windows.region_layout as region_layout
import decsim.windows.release_rules as release_rules
import decsim.windows.routers as routers
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
OWN_STREAMS = release_rules.OwnStreams.Settings()
DECODED_ANCESTORS = release_rules.DecodedAncestors.Settings()
RELEASE_RULES = (OWN_STREAMS, DECODED_ANCESTORS)


def run(workload, decode_microseconds: float):
    settings, graph = patch_stream_graph(workload, decode_microseconds)
    machine = machine_module.Machine.build(graph, 0)
    result = machine.run()
    return settings, machine, result


def patch_stream_graph(workload, decode_microseconds: float) -> tuple:
    """The workload's settings and its one-pool machine on patch streams."""
    settings = workload_settings.WorkloadSettings.running(workload)
    pool = _preset_pool(decode_microseconds)
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme)
    qpu = qpu_settings.QpuSettings(distance=DISTANCE)
    base = machines.base(settings, windows, qpu)
    graph = machines.with_pool(base, pool)
    return settings, graph


def _preset_pool(decode_microseconds: float):
    algorithm = decoders.PresetLatencyDecoder.Settings(decode_microseconds)
    return decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )


# a second chip pool on the weak pool's row, and a slower row of its own
FPGA_B = ("fpga_b", _preset_pool(T_GATE_DECODE_MICROSECONDS))
SLOW = ("slow", _preset_pool(2 * T_GATE_DECODE_MICROSECONDS))


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
    assert commits == [(1, 7), (8, 14), (15, 19), (20, 26), (27, 31)]
    assert buffer_ends == [14, 19, 26, 31, 31]


def test_a_released_correction_runs_on_without_a_gap(t_gate):
    """The S merge's successors start on the boundary it ends on.

    Their releases landed with the merge's, long before, so their queued
    commands fire on schedule (eQASM 1808.02449 Sec. 3.5): the branch
    cost is paid once per decision, not once per instruction (Skoric et
    al. 2209.08552 Sec. D, a response time per T gate).
    """
    _settings, machine, _result = t_gate

    stamps = machine.observation.runtime_stamps
    merge_done = stamps.body_done[CONDITIONAL_S_MERGE]
    y_measurement, s_idle = CONDITIONAL_S_MERGE + 1, CONDITIONAL_S_MERGE + 2
    assert stamps.decode_release[y_measurement] < merge_done
    assert stamps.op_start[y_measurement] == merge_done
    assert stamps.op_start[s_idle] == merge_done


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


@pytest.mark.parametrize("patches", [((0, 0), (0, 1)), ((0, 1), (0, 0))])
def test_a_reader_on_a_host_pool_wakes_when_its_partner_lands(patches):
    """A host pool's rounds land in the store it reads (accept_room_round).

    The reader's own region may land before the region it reads, in
    either patch order, and it must still be decoded once both are in.
    """
    merge = program_records.Operation(
        0,
        "merge",
        patches,
        patches=patches,
        couplings=(patches,),
        kind=program_records.OpKind.MERGE,
    )
    workload = workload_records.Workload((merge,), {0: 1}, patch_streams=True)
    settings = workload_settings.WorkloadSettings.running(workload)
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme)
    qpu = qpu_settings.QpuSettings(distance=3)
    free_hops = dict.fromkeys(declared_run.DECLARED_HOP_MICROSECONDS, 0.0)
    fabric = declared_run.declared_fabric(**free_hops)
    base = machines.base(settings, windows, qpu, fabric=fabric)
    algorithm = decoders.PresetLatencyDecoder.Settings(0.1)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    graph = machines.with_pool(base, pool, name="strong", site="host")
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    assert result.terminal_status == "complete"


@pytest.mark.parametrize("rule", RELEASE_RULES)
def test_a_measurement_nothing_waits_on_returns_its_result(rule):
    """The QPU needs the outcome back, so it comes once it is decoded.

    No operation is blocked on the measurement, yet it asks for its
    result (requires_result_return_to_qpu), and either release rule
    delivers it as it would a blocker's (release_rules.py).
    """
    measurement = program_records.Operation(
        0,
        "measure",
        ((0, 0),),
        patches=((0, 0),),
        kind=program_records.OpKind.MEASURE,
        requires_result_return_to_qpu=True,
    )
    workload = workload_records.Workload(
        (measurement,), {0: 1}, patch_streams=True
    )
    settings = workload_settings.WorkloadSettings.running(workload)
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme, release_rule=rule)
    qpu = qpu_settings.QpuSettings(distance=3)
    base = machines.base(settings, windows, qpu)
    algorithm = decoders.PresetLatencyDecoder.Settings(0.1)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    graph = machines.with_pool(base, pool)
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    assert result.terminal_status == "complete"
    returned = machine.observation.runtime_stamps.result_return
    assert list(returned) == [0]


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


def _t_gate_by_patch(tmp_path, table: tuple, pools: tuple):
    """The T gate on the weak pool and pools, each patch's stream to its pool.

    VQD's decoder per logical qubit (2406.17995v2 Sec. 4.1); DecoNet
    statically assigns each decoder instance one logical qubit's blocks
    (2504.11805 Sec. VI-C).
    """
    path = tmp_path / "t_gate.lss"
    path.write_text(patch_lives_test.T_GATE)
    workload = lattice_surgery.read_schedule(path, DISTANCE)
    _settings, graph = patch_stream_graph(workload, T_GATE_DECODE_MICROSECONDS)
    rule = routers.ByPatch.Settings(pool_by_patch=table)
    return machines.with_router(graph, rule, pools)


def test_each_patch_stream_is_decoded_on_the_pool_its_router_names(tmp_path):
    table = (((0, 0), "weak"), ((0, 1), "fpga_b"), ((1, 0), "fpga_b"))
    graph = _t_gate_by_patch(tmp_path, table, (FPGA_B,))
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    units = collections.defaultdict(set)
    for record in machine.observation.stages.records:
        units[record.stream_id].add(record.unit_name)
    assert result.terminal_status == "complete"
    assert dict(units) == {
        CONTROL_STREAM: {"weak#0"},
        T_STREAM: {"fpga_b#0"},
        S_ANCILLA_STREAM: {"fpga_b#0"},
    }


def test_a_patch_stream_whose_patch_no_pool_routes_is_refused_at_build(
    tmp_path,
):
    """Patch (1,0) is missing: refused before the run, not mid-run."""
    table = (((0, 0), "weak"), ((0, 1), "fpga_b"))
    graph = _t_gate_by_patch(tmp_path, table, (FPGA_B,))

    with pytest.raises(ValueError, match=r"routes no pool .* \(\(1, 0\),\)"):
        machine_module.Machine.build(graph, 0)


def test_a_group_of_patches_shares_a_pool_and_a_region_has_its_own_row(
    tmp_path,
):
    """Patches (0,0) and (0,1) share weak; (1,0) decodes on slow's row.

    A fixed group of patches sharing one decoder is AFS's (2001.06598);
    a different decoder for one part of the machine, Tour de gross's
    (2506.03094).
    """
    table = (((0, 0), "weak"), ((0, 1), "weak"), ((1, 0), "slow"))
    graph = _t_gate_by_patch(tmp_path, table, (SLOW,))
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    units = collections.defaultdict(set)
    decode_ticks = collections.defaultdict(set)
    for record in machine.observation.stages.records:
        units[record.stream_id].add(record.unit_name)
        duration = record.end_ticks - record.start_ticks
        decode_ticks[(record.unit_name, record.stage)].add(duration)
    weak_ticks = config.microseconds_to_ticks(T_GATE_DECODE_MICROSECONDS)
    assert result.terminal_status == "complete"
    assert dict(units) == {
        CONTROL_STREAM: {"weak#0"},
        T_STREAM: {"weak#0"},
        S_ANCILLA_STREAM: {"slow#0"},
    }
    assert decode_ticks[("weak#0", "algorithm")] == {weak_ticks}
    assert decode_ticks[("slow#0", "algorithm")] == {2 * weak_ticks}
