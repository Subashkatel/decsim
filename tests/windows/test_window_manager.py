"""The windows facade's laws through its public surface.

A modelled window never reads the next operation's rounds, a stream's
later window waits on the one before it, a Tan seam waits in its input
slot rather than on the only unit, and a round past the plan stops the
run.
"""

import types

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph_layout as graph_layout
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.built_window_models as built_window_models
import decsim.windows.routers as routers
import decsim.windows.schemes.sandwich as sandwich_scheme
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner
import tests.declared_run as declared_run
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


def _weak_routes():
    """The weak base's routes: one pool, the planner's windows on it."""
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    at_task = weak_base.at_task()
    layout = graph_layout.layout_of(at_task)
    return layout.routes()


_ROUTES = _weak_routes()


def _weak_run():
    memory = program_records.Operation(0, "M", (0,), clifford=True)
    six_rounds = round_policies.FixedRounds(6)
    workload = workload_settings.WorkloadSettings(
        operations=[memory], rounds_policy=six_rounds
    )
    code = code_geometry.SurfaceCodeModel(distance=3)
    card = declared_run.GivenCard(code)
    qpu = qpu_settings.QpuSettings(
        code_card=card, round_period_microseconds=1.0
    )
    decoder = decoders.PresetLatencyDecoder.Settings(2.0)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=decoder, unit_count=1, engine=declared_run.DECLARED_ENGINE
    )
    base = machines.base(workload, qpu)
    graph = machines.with_pool(base, weak_decoder)
    machine = machine_module.Machine.build(graph, 0)
    machine.run()
    return machine


def _chained_stim_run(terminal_policy: str) -> machine_module.Machine:
    """Two nine-round d=3 memories, the second after the first."""
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 9, 3, 0.001
    )
    first = program_records.Operation(
        1, "first", (0,), patches=(0,), circuit=circuit
    )
    second = program_records.Operation(
        2,
        "second",
        (0,),
        patches=(0,),
        circuit=circuit,
        predecessors=(1,),
        decoder_boundary_predecessors=(1,),
    )
    nine_rounds = round_policies.FixedRounds(9)
    workload = workload_settings.WorkloadSettings(
        operations=(first, second),
        rounds_policy=nine_rounds,
    )
    device = stim_device.StimDevice()
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source, round_period_microseconds=1.0
    )
    decoder = decoders.PresetLatencyDecoder.Settings(2.0)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=decoder, unit_count=1, engine=declared_run.DECLARED_ENGINE
    )
    windows = window_settings.WindowSettings(terminal_policy=terminal_policy)
    base = machines.base(workload, qpu, windows)
    graph = machines.with_pool(base, weak_decoder)
    return machine_module.Machine.build(graph, 0)


def test_a_modelled_window_never_reads_the_next_streams_rounds():
    """Its model ends at its own last round, so its decode refuses it."""
    lookahead = _chained_stim_run("lookahead")
    with pytest.raises(RuntimeError, match="does not match job stream"):
        lookahead.run()
    machine = _chained_stim_run("flush")
    machine.run()

    assert machine.windows.window_manager.planner.total_windows == 4


def test_a_streams_later_window_waits_on_the_previous_ones_boundary():
    """Each later overlapping stream window depends on its predecessor."""
    manager = object.__new__(window_manager_module.WindowManager)
    boundary_payload = boundary_payloads.DenseSeamMask()
    manager.window_interaction = window_interactions.DefaultWindowInteraction(
        0, boundary_payload
    )
    manager.planner = _stream_planner()
    manager.tracker = types.SimpleNamespace(is_sealed=lambda _stream_id: False)
    courier = window_boundaries.BoundaryCourier(_ROUTES)
    courier.planner = manager.planner
    courier.interaction = manager.window_interaction
    manager.courier = courier
    manager.retention = types.SimpleNamespace(
        register_window=lambda _key, _window: None
    )
    stream = program_records.Operation("stream", "stream", (0,))
    manager.planner.register_stream(stream, None)

    manager._grow_stream("stream", 4, None)

    first, second = manager.planner.windows_of("stream")
    assert (first.commit_lo, first.commit_hi, first.buffer_hi) == (1, 3, 5)
    assert (second.commit_lo, second.commit_hi, second.buffer_hi) == (4, 6, 8)
    assert second.deps == [("stream", 0)]
    assert second.deps_remaining == 1
    assert first.dependents == [("stream", 1)]


def _stream_planner() -> window_planner.WindowPlanner:
    plan = window_records.WindowPlan(
        windows={},
        window_count_by_stream={},
        windows_by_stream={},
        successors_by_stream={},
        rounds_by_stream={},
        total_windows=0,
        windowed_by_stream={},
        batch_preceding_idle_rounds_by_stream={},
    )
    geometry = types.SimpleNamespace(commit_round_count=3, buffer_round_count=2)
    resolved = types.SimpleNamespace(
        operation_id="stream",
        code_geometry=geometry,
        round_count=9,
    )
    cache = built_window_models.BuiltWindowModels()
    built = window_planner.WindowModels(cache)
    built.decoder = _Decoder()
    scheme = sliding_scheme.SlidingWindowScheme()
    planner = window_planner.WindowPlanner([resolved], plan, ())
    planner.scheme = scheme
    planner.layouts = scheme.stream_layouts()
    planner.models = built
    planner.start()
    return planner


class _Decoder:
    """The decoder, as the models ask it what a model must offer."""

    # this decoder asks a window model for nothing
    fault_model_requirement = None


def _tan_memory_circuit():
    """A 27-round d=3 rotated memory at the reference noise strength."""
    stim = pytest.importorskip("stim")
    probability = 0.003
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=27,
        distance=3,
        after_clifford_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
        before_round_data_depolarization=probability,
    )


def _tan_sandwich_run(unit_count):
    """A 27-round d=3 Stim memory under Tan's sandwich schedule.

    A type-2 seam window fills before the two cores beside it and reads
    both of their boundaries (Tan et al. 2209.09219, supplement S8), so
    it becomes ready first and must wait in its input slot rather than on
    the unit: on one unit a seam that holds the compute while it waits
    for its neighbours would stop the run.
    """
    circuit = _tan_memory_circuit()
    memory = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(27)
    workload = workload_settings.WorkloadSettings(
        operations=[memory], rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(
        distance=3, round_period_microseconds=1.0, source=source
    )
    decoder = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=5.0
    )
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=decoder,
        unit_count=unit_count,
        engine=declared_run.DECLARED_ENGINE,
    )
    scheme = sandwich_scheme.TanSandwichScheme.Settings()
    windows = window_settings.WindowSettings(scheme=scheme)
    base = machines.base(workload, qpu, windows)
    graph = machines.with_pool(base, weak_decoder)
    machine = machine_module.Machine.build(graph, 3)
    result = machine.run()
    return machine, result


def test_a_tan_seam_waits_in_its_slot_and_never_holds_the_only_unit():
    machine, result = _tan_sandwich_run(unit_count=1)
    windows = machine.observation.windows.windows.values()
    decode_ends = [window.t_done for window in windows]
    assert None not in decode_ends
    assert result.operation_results[0].logical_failure is False


def test_a_seam_starting_before_the_core_ahead_of_it_is_stamped_first():
    """The request walk reaches a window that starts out of index order.

    Tan's seam (1, 2) reads from round 4 and the core (1, 1) before it
    from round 6, so the seam's first round lands two rounds earlier.
    """
    machine, _result = _tan_sandwich_run(unit_count=1)
    windows = machine.observation.windows.windows
    seam = windows[(1, 2)]
    core = windows[(1, 1)]
    assert seam.start_round < core.start_round
    assert seam.t_first_round < core.t_first_round


def test_a_round_arriving_after_the_last_window_committed_is_refused():
    """The device is a plug-in, so a round past the plan is its mistake.

    An operation's syndrome RAM is freed when its last window commits,
    so a later round has nothing to land in and no window left to feed;
    the facade stops loudly rather than accounting a round nobody reads
    (window_manager.py, _refuse_unplanned_round).
    """
    machine = _weak_run()
    fragment = round_records.RetainedSyndromeFragment(
        stream_id=0,
        patch_ids=(0,),
        round_index=2,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(0, 2, (fragment,))
    window_manager = machine.windows.window_manager
    with pytest.raises(RuntimeError, match="round 2 of M arrived after"):
        window_manager.accept_window_input(packet)


def test_a_lookahead_window_reads_its_successors_rounds_copied_to_its_pool():
    """Operation 1 decodes on fpga_b, whose store is on the host.

    Its last window looks ahead into operation 2's rounds 1-3, which
    enter at weak; the controller copies them into fpga_b's store as it
    writes them (controller/round_destinations.py), and each copy's
    landing wakes the windows that read across the end it lands past.
    """
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2, decoder_boundary_predecessors=(1,)),
    ]
    windows = window_settings.WindowSettings(terminal_policy="lookahead")
    graph = declared_run.weak_only_graph(
        rounds=6, operations=operations, windows=windows
    )
    algorithm = decoders.PresetLatencyDecoder.Settings(
        declared_run.DECLARED_MICROSECONDS["weak"]
    )
    fpga_b = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE
    )
    rule = routers.ByPatch.Settings(pool_by_patch=((1, "fpga_b"), (2, "weak")))
    stored = machines.with_store_per_pool(
        graph, rule, (("fpga_b", fpga_b),), site=fabrics.HOST
    )
    machine = machine_module.Machine.build(stored, 0)

    result = machine.run()

    transfers = result.link_traffic["transfers"]
    copy_into = {"copy_into": "fpga_b_buffer"}
    copies = [
        row for row in transfers if row["attribution"]["relation"] == copy_into
    ]
    copied = [row["attribution"]["rounds_by_stream"] for row in copies]
    reads = [record.last_rounds_read for record in _fetches_of(machine, 1)]
    assert result.terminal_status == "complete"
    assert [spans[0]["round_lo"] for spans in copied] == [1, 2, 3]
    assert reads == [((1, 6),), ((1, 6), (2, 3))]


def _fetches_of(machine, stream_id) -> list:
    """The fetch stage of each decode of one stream's windows, in order."""
    fetches = []
    for record in machine.observation.stages.records:
        if (record.stage, record.stream_id) == ("fetch", stream_id):
            fetches.append(record)
    return fetches
