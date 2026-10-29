"""A stream lets its final windows go and decides as before (window_release.py).

The referent is the same stream run keeping every window: the result
each operation delivers, its scoring and the frame's correction must be
the same, while the machine keeps a bounded number of windows however
long the stream runs.
"""

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.presets.machines as machines
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.records.workload as workload_records
import decsim.windows.routers as routers
import decsim.windows.schemes.region_sliding as region_sliding
import decsim.windows.settings as window_settings
import decsim.windows.window_release as window_release
import tests.declared_run as declared_run
import tests.stream_machines as stream_machines

SEED = 17
# about 300 windows: several lettings-go, with escalations among them
LONG_RUN = stream_machines.StreamRun(900, 0.5, 1, True)


def _built_and_run(folds: bool):
    graph = stream_machines.stream_graph(LONG_RUN, folds)
    machine = machine_module.Machine.build(graph, SEED)
    result = machine.run()
    return machine, result


def test_a_stream_that_lets_windows_go_delivers_the_same_results():
    _whole_machine, whole = _built_and_run(folds=False)

    _machine, lean = _built_and_run(folds=True)

    assert lean.operation_results == whole.operation_results
    assert lean.fully_done_ticks == whole.fully_done_ticks


def test_the_frame_folds_what_it_lets_go_into_the_same_correction():
    whole_machine, _whole = _built_and_run(folds=False)
    whole_frame = whole_machine.control.pauli_frame

    machine, _result = _built_and_run(folds=True)

    frame = machine.control.pauli_frame
    stream_id = stream_machines.STREAM_ID
    expected = whole_frame.frame_for_stream(stream_id)
    assert frame.frame_for_stream(stream_id) == expected


def test_a_long_stream_keeps_a_bounded_number_of_windows():
    machine, _result = _built_and_run(folds=True)

    planner = machine.windows.planner
    stream_id = stream_machines.STREAM_ID
    kept_bound = window_release.KEPT_FINAL_WINDOWS + window_release.LET_GO_EVERY
    assert planner.window_count_of(stream_id) > 4 * kept_bound
    assert len(planner.windows_by_key) <= kept_bound
    assert len(planner.models.model_by_window) <= kept_bound
    assert len(machine.windows.ledger.contributions) <= kept_bound


# about 300 parallel blocks, decoded alone: switching refuses the row
PARALLEL_RUN = stream_machines.StreamRun(900, 0.5, 1, False)


def _parallel_built_and_run(folds: bool):
    graph = stream_machines.stream_graph(PARALLEL_RUN, folds)
    parallel_graph = stream_machines.on_parallel_windows(graph)
    machine = machine_module.Machine.build(parallel_graph, SEED)
    result = machine.run()
    return machine, result


def test_a_parallel_stream_that_lets_blocks_go_decides_as_one_that_keeps():
    """A B block finishes after the A after it, so the final prefix lags."""
    whole_machine, whole = _parallel_built_and_run(folds=False)

    machine, lean = _parallel_built_and_run(folds=True)

    stream_id = stream_machines.STREAM_ID
    whole_frame = whole_machine.control.pauli_frame
    frame = machine.control.pauli_frame
    planner = machine.windows.planner
    kept_bound = window_release.KEPT_FINAL_WINDOWS + window_release.LET_GO_EVERY
    assert lean.operation_results == whole.operation_results
    assert lean.fully_done_ticks == whole.fully_done_ticks
    assert frame.frame_for_stream(stream_id) == (
        whole_frame.frame_for_stream(stream_id)
    )
    assert planner.window_count_of(stream_id) > 4 * kept_bound
    assert len(planner.windows_by_key) <= kept_bound


def test_reading_a_window_let_go_says_it_was_let_go():
    machine, _result = _built_and_run(folds=True)

    planner = machine.windows.planner
    first_key = (stream_machines.STREAM_ID, 0)
    with pytest.raises(KeyError, match="let it go"):
        planner.window_at(first_key)


LOW_PATCH = (0, 0)
HIGH_PATCH = (0, 1)
# far more than the windows a patch's memory lays after the merge
SLOW_HOP_MICROSECONDS = 1000.0


def _coupled_run(folds: bool):
    """A merge of two patches decoded on two pools a slow hop apart.

    Each patch then idles 300 rounds, so its stream lets dozens of
    windows go while the merge's boundary to the other patch's window
    is still on the hop.
    """
    coupling = (LOW_PATCH, HIGH_PATCH)
    merge = program_records.Operation(
        0,
        "merge",
        coupling,
        couplings=(coupling,),
        kind=program_records.OpKind.MERGE,
    )
    low = program_records.Operation(
        1, "low memory", (LOW_PATCH,), kind=program_records.OpKind.MEMORY
    )
    high = program_records.Operation(
        2, "high memory", (HIGH_PATCH,), kind=program_records.OpKind.MEMORY
    )
    rounds = {0: 3, 1: 300, 2: 300}
    program = workload_records.Workload(
        (merge, low, high), rounds, patch_streams=True
    )
    settings = workload_settings.WorkloadSettings.running(program)
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme)
    observation = observe_settings.ObservationSettings(folds_windows=folds)
    hops = dict.fromkeys(declared_run.DECLARED_HOP_MICROSECONDS, 0.0)
    hops["forward"] = SLOW_HOP_MICROSECONDS
    hops["reverse"] = SLOW_HOP_MICROSECONDS
    fabric = declared_run.declared_fabric(**hops)
    qpu = qpu_settings.QpuSettings(distance=3)
    base = machines.base(
        settings, windows, qpu, observation=observation, fabric=fabric
    )
    algorithm = decoders.PresetLatencyDecoder.Settings(0.01)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    pooled = machines.with_pool(base, pool)
    framed = machines.with_frame(pooled)
    by_patch = routers.ByPatch.Settings(
        ((LOW_PATCH, "weak"), (HIGH_PATCH, "b"))
    )
    graph = machines.with_router(framed, by_patch, (("b", pool),))
    machine = machine_module.Machine.build(graph, 0)
    return machine, machine.run()


def test_a_boundary_still_on_its_hop_to_another_stream_is_not_let_go():
    _kept_machine, kept = _coupled_run(folds=False)
    _folded_machine, folded = _coupled_run(folds=True)

    assert folded.terminal_status == "complete"
    assert folded.operation_results == kept.operation_results
    assert folded.fully_done_ticks == kept.fully_done_ticks


def test_a_folded_patch_stream_keeps_a_region_only_while_it_keeps_a_window():
    """Each stream's region starts go with its windows (region_layout.py).

    The merge's regions go once both coupled windows have gone, so at
    the end each stream keeps the regions of the windows it keeps.
    """
    machine, _result = _coupled_run(folds=True)

    planner = machine.windows.planner
    regions = planner.layouts.regions
    for stream_id, stream_regions in regions.regions_by_stream.items():
        indices = planner.window_indices_of(stream_id)
        starts = stream_regions.starts
        assert indices.first_kept > 0
        assert starts.first_kept == indices.first_kept
        assert len(starts.kept) == len(indices.kept)
    assert regions.reads_by_region == {}
