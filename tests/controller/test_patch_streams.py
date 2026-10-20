"""A lattice-surgery run decoded on patch streams (controller/patch_streams.py).

Timing only, as SWIPER-SIM is: each patch life is one decode stream,
every round its patch measures that stream's next round (SWIPER
2412.05115, swiper/device_manager.py _generate_syndrome_round), and an
operation on several patches releases its waiters once each of its
streams committed its rounds (Bombín et al. 2303.04846 Eq. 4).
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


def run(workload, decode_microseconds: float):
    settings = workload_settings.WorkloadSettings.running(workload)
    algorithm = decoders.PresetLatencyDecoder.Settings(decode_microseconds)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    base = machines.base(settings)
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
