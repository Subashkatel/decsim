"""Several pools escalate to one shared target pool.

A weak decoder per patch with one strong decoder they all share is
Delfosse's lazy decoders over shared decoding units (2001.11427 Sec. V)
and Toshio's N_window weak decoders over a single strong one
(2510.25222 Sec. III C). Here patch 1 enters at weak and patch 2 at
fpga_b, both on the chip, and both escalate every window to strong on
the host. Each answer that goes back through the weak chip goes to the
chip that asked.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.execution_slots as execution_slots
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.windows as window_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as topology_graph
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.escalation.test_switching_mode as switching_mode


def _shared_strong(
    answer_route: str,
    factory_pool="weak",
    strong_site=fabrics.HOST,
    fabric=None,
):
    """Two chip pools, by patch, each escalating every window to strong.

    The magic-state factory's decodes go to factory_pool; strong sits at
    strong_site, and fabric, when given, prices the hops.
    """
    weak, strong = declared_run.switching_decoders(True)
    engine = declared_run.DECLARED_ENGINE
    weak_pool = decoder_settings.DecoderPoolSettings(
        algorithm=weak, engine=engine
    )
    strong_pool = decoder_settings.DecoderPoolSettings(
        algorithm=strong, engine=engine
    )
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2),
    ]
    workload = declared_run.declared_workload(operations, 6)
    plain_windows = window_settings.WindowSettings()
    windows = window_settings.switching_windows(
        plain_windows, declared_run.REDO_WINDOW
    )
    switching = declared_run.declared_switching(
        strong_window=declared_run.REDO_WINDOW
    )
    switching_clock = config.Clock(4)
    switching = dataclasses.replace(
        switching, strong_answer_route=answer_route, clock=switching_clock
    )
    qpu = declared_run.declared_qpu()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    declared_fabric = declared_run.declared_fabric()
    machine = machines.base(
        workload, qpu, windows, controller, frame, fabric=declared_fabric
    )
    machine = machines.with_pool(machine, weak_pool)
    machine = machines.with_escalation(
        machine,
        machines.WEAK_POOL,
        machines.STRONG_POOL,
        strong_pool,
        switching,
        site=strong_site,
        switching_name=machines.SWITCHING,
    )
    if fabric is not None:
        machine = dataclasses.replace(machine, fabric=fabric)
    by_patch = routers.ByPatch.Settings(
        pool_by_patch=((1, "weak"), (2, "fpga_b"))
    )
    machine = machines.with_router(
        machine, by_patch, (("fpga_b", weak_pool),), factory_pool=factory_pool
    )
    return machines.with_escalation_to(machine, "fpga_b", "strong", switching)


def _run(answer_route: str):
    graph = _shared_strong(answer_route)
    machine = machine_module.Machine.build(graph, 0)
    result = machine.run()
    return machine, result


def test_both_pools_escalate_every_window_to_the_one_strong_pool():
    machine, result = _run("direct")
    frame_tiers = declared_run.frame_tiers(machine)
    tiers = dict(frame_tiers)
    transfers = switching_mode.transfers_by_path(result)
    assert tiers == {
        (1, 0): "strong",
        (1, 1): "strong",
        (2, 0): "strong",
        (2, 1): "strong",
    }
    assert transfers["strong_to_frame"] == 4
    assert transfers["weak_to_strong"] >= 2
    assert transfers["fpga_b_to_strong"] >= 2


def test_through_the_weak_chip_each_answer_goes_down_to_the_chip_that_asked():
    _machine, result = _run("through_weak_chip")
    transfers = switching_mode.transfers_by_path(result)
    answers = {
        path: transfers.get(path, 0)
        for path in (
            "strong_to_weak",
            "strong_to_fpga_b",
            "weak_to_frame",
            "fpga_b_to_frame",
            "strong_to_frame",
        )
    }
    assert answers == {
        "strong_to_weak": 2,
        "strong_to_fpga_b": 2,
        "weak_to_frame": 2,
        "fpga_b_to_frame": 2,
        "strong_to_frame": 0,
    }


def test_one_learned_threshold_over_two_entry_edges_is_refused():
    """The task's calibrator would decide on weak's edge and not fpga_b's."""
    # the refusal reads only that the task learns a threshold
    calibrator = object()

    graph = _shared_strong("direct")

    with pytest.raises(ValueError, match="2 edges leave entry pools"):
        machine_module.Machine.build(graph, 0, online_threshold=calibrator)


def test_the_factorys_decodes_go_to_the_pool_the_router_names():
    graph = _shared_strong("direct", factory_pool="fpga_b")
    machine = machine_module.Machine.build(graph, 0)

    fpga_b_id = window_records.PoolId("fpga_b")
    fpga_b = machine.decoders.manager_of(fpga_b_id)
    assert machine.qpu.factory.decode_queue is fpga_b


def test_two_entry_pools_with_no_factory_pool_are_refused():
    """No paper sets a default; the run names which pool serves them."""
    graph = _shared_strong("direct")
    router_part = graph.part(machines.ROUTER)
    unnamed = dataclasses.replace(router_part.settings, factory_pool=None)
    unnamed_graph = graph.with_record(machines.ROUTER, unnamed)

    with pytest.raises(ValueError, match="router.factory_pool names no pool"):
        machine_module.Machine.build(unnamed_graph, 0)


def test_a_strong_pool_on_its_own_site_shares_its_link_with_both_chips():
    """A GPU at site gpu0 behind one link: both chips' escalations queue on it.

    Any site name works; the shared site link puts every hop from the
    fridge into gpu0 on one channel, and every hop back on another.
    """
    cable = fabrics.TimedLink(1.0, 100_000.0)
    shared = fabrics.SiteLink((fabrics.FRIDGE, "gpu0"), cable, shared=True)
    declared_fabric = declared_run.declared_fabric()
    fabric = fabrics.SiteLinks((shared,), declared_fabric)
    graph = _shared_strong("direct", strong_site="gpu0", fabric=fabric)
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    channel_by_path = {}
    for path in machine.layout.card.paths:
        channel_by_path[path.name] = path.settings.channel.name
    frame_tiers = declared_run.frame_tiers(machine)
    tiers = dict(frame_tiers)
    tier_names = tiers.values()
    assert result.terminal_status == "complete"
    assert set(tier_names) == {"strong"}
    assert channel_by_path["weak_to_strong"] == "fridge_to_gpu0"
    assert channel_by_path["fpga_b_to_strong"] == "fridge_to_gpu0"
    assert channel_by_path["strong_to_weak"] == "gpu0_to_fridge"


def test_the_shared_strong_pool_escalates_on_to_a_deep_pool_for_both():
    """Three levels with a fan-in: weak and fpga_b, then strong, then deep.

    Strong doubts every window, so each window of both patches climbs
    the chain its own entry pool starts and is answered by deep.
    """
    graph = _shared_strong("direct")
    doubting, _strong = declared_run.switching_decoders(True)
    graph = graph.with_settings("strong", algorithm=doubting)
    deep_algorithm = decoders.PresetLatencyDecoder.Settings(50.0)
    deep_pool = decoder_settings.DecoderPoolSettings(
        algorithm=deep_algorithm, engine=declared_run.DECLARED_ENGINE
    )
    deep_switching = declared_run.declared_switching()
    graph = machines.with_escalation(
        graph, "strong", "deep", deep_pool, deep_switching
    )
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    frame_tiers = declared_run.frame_tiers(machine)
    tiers = dict(frame_tiers)
    transfers = switching_mode.transfers_by_path(result)
    assert result.terminal_status == "complete"
    assert tiers == {
        (1, 0): "deep",
        (1, 1): "deep",
        (2, 0): "deep",
        (2, 1): "deep",
    }
    assert transfers["deep_to_frame"] == 4
    assert "strong_to_frame" not in transfers


def test_a_predecoder_per_patch_feeds_one_gpu_over_its_link_and_slots():
    """Chip predecoders in the fridge, one GPU at gpu0 behind one link.

    design.md's mix of a cryo predecoder feeding a room-temperature
    GPU: both patches' escalations queue on the shared link, and the
    GPU's two strong units share its execution slots, so one slot
    decodes one window at a time and two slots decode as with none
    (decoders/execution_slots.py; CUDA-Q's streams share one GPU).
    """
    two_units = _predecoders_and_gpu()
    on_one_slot = _on_gpu_slots(two_units, 1)
    on_two_slots = _on_gpu_slots(two_units, 2)
    alone = _strong_decodes(two_units)
    one_slot = _strong_decodes(on_one_slot)
    two_slots = _strong_decodes(on_two_slots)

    assert len(alone) == 4
    assert _decoded_at_once(alone) > 0
    assert _decoded_at_once(one_slot) == 0
    assert two_slots == alone


def _predecoders_and_gpu():
    cable = fabrics.TimedLink(1.0, 100_000.0)
    shared = fabrics.SiteLink((fabrics.FRIDGE, "gpu0"), cable, shared=True)
    declared_fabric = declared_run.declared_fabric()
    fabric = fabrics.SiteLinks((shared,), declared_fabric)
    graph = _shared_strong("direct", strong_site="gpu0", fabric=fabric)
    return graph.with_settings("strong", unit_count=2)


def _on_gpu_slots(graph, slot_count: int):
    slots = execution_slots.ExecutionSlots.Settings(slot_count)
    device = topology_graph.Part("gpu", slots, "gpu0")
    return graph.with_part(device)


def _strong_decodes(graph) -> list:
    """(start, end) of each strong decode's algorithm stage, in order."""
    machine = machine_module.Machine.build(graph, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    spans = []
    for record in machine.observation.stages.records:
        if record.stage != "algorithm":
            continue
        if not record.unit_name.startswith("strong"):
            continue
        spans.append((record.start_ticks, record.end_ticks))
    return sorted(spans)


def _decoded_at_once(spans: list) -> int:
    """How many decodes started before the one started before them ended."""
    # each decode beside the one that started next
    pairs = zip(spans, spans[1:], strict=False)
    return sum(later[0] < earlier[1] for earlier, later in pairs)


def test_both_pools_windows_start_deep_beside_the_shared_strong_decode():
    """Every edge into a shared target is wired on to the edge out of it.

    Deep runs at once with strong (Toshio 2510.25222 Step 1, lines
    598-601), so each window strong decodes, from weak's patch and from
    fpga_b's, starts deep before strong's decode ends; an edge into
    strong left unwired would start deep only after strong answered.
    Deep has a unit per window, so no decode waits for a unit.
    """
    graph = _shared_strong("direct")
    doubting, _strong = declared_run.switching_decoders(True)
    graph = graph.with_settings("strong", algorithm=doubting)
    deep_algorithm = decoders.PresetLatencyDecoder.Settings(50.0)
    deep_pool = decoder_settings.DecoderPoolSettings(
        algorithm=deep_algorithm,
        engine=declared_run.DECLARED_ENGINE,
        unit_count=4,
    )
    at_once = declared_run.declared_switching(run_both_at_once=True)
    graph = machines.with_escalation(
        graph, "strong", "deep", deep_pool, at_once
    )
    machine = machine_module.Machine.build(graph, 0)

    result = machine.run()

    started_during_strong = _deep_started_during_strong(machine)
    assert result.terminal_status == "complete"
    assert started_during_strong == {
        (1, 0): True,
        (1, 1): True,
        (2, 0): True,
        (2, 1): True,
    }


def _deep_started_during_strong(machine) -> dict:
    """Each window's: did deep's decode start before strong's ended."""
    starts = {}
    ends = {}
    for record in machine.observation.stages.records:
        if record.stage != "algorithm":
            continue
        pool, _unit = record.unit_name.split("#")
        key = (pool, record.stream_id, record.window_id)
        starts[key] = record.start_ticks
        ends[key] = record.end_ticks
    started = {}
    for pool, stream_id, window_id in ends:
        if pool != "strong":
            continue
        deep_start = starts[("deep", stream_id, window_id)]
        strong_end = ends[("strong", stream_id, window_id)]
        started[(stream_id, window_id)] = deep_start < strong_end
    return started
