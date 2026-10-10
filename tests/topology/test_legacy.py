"""A graph as today's MachineSettings, and back (topology/legacy.py).

The referents are today's records: every shipped task and every
regression machine, written as a graph, resolves to the record it was
written from, with the same task id; a graph written by hand resolves
to the record its parts hold; a graph today's builder cannot build is
refused as not built yet, never as not allowed.
"""

import dataclasses
import pathlib

import pytest

import decsim.experiments.collect as collect
import decsim.experiments.experiment as experiment
import decsim.links.link_profiles as link_profiles
import decsim.settings as machine_settings
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.legacy as legacy
import tests.experiments.run_files as run_files
import tests.regression.test_shot_results as regression

THIS_FILE = pathlib.Path(__file__)
EXPERIMENTS = THIS_FILE.parents[2] / "experiments"
SHIPPED = ("switching_baseline", "switching", "data_movement")
FRIDGE_CLOCK = machine_settings.FRIDGE_CLOCK
# Experiment 1's cable to the host and the host's poll of its memory
ROCE_V2_CPU_LEG = fabrics.RoundTripLeg(2.305, 128, 100_000)
HOST_POLL = fabrics.TimedLink(0.0)
ROCE_V2_CPU = fabrics.FridgeHostFabric(
    cable=ROCE_V2_CPU_LEG, host_local=HOST_POLL
)
STAGE_FOUR = "not implemented yet"


def _shipped_tasks() -> list:
    tasks = []
    for name in (*SHIPPED, "switching_per_window"):
        run_file = EXPERIMENTS / name / "run.py"
        loaded = experiment.load(run_file)
        tasks.extend(loaded.tasks)
    return tasks


def _weak_graph() -> graph.Machine:
    machine = run_files.minimal_machine()
    return legacy.graph_of(machine)


def _with(machine: graph.Machine, parts=(), wires=()) -> graph.Machine:
    """The graph with parts and wires added."""
    return dataclasses.replace(
        machine,
        parts=(*machine.parts, *parts),
        wires=(*machine.wires, *wires),
    )


def test_every_shipped_task_resolves_to_its_record_and_its_id():
    tasks = _shipped_tasks()

    for task in tasks:
        written = legacy.graph_of(task.machine)
        machine = legacy.resolved(written)
        again = collect.Task(task.name, machine, task.metadata)
        assert machine == task.machine, task.name
        assert again.strong_id() == task.strong_id(), task.name
    assert len(tasks) == 140


@pytest.mark.parametrize("name", list(regression.SHAPES))
def test_every_regression_machine_resolves_to_its_record_and_its_id(name):
    build = regression.SHAPES[name]
    settings = build()
    task = collect.Task(name, settings)

    written = legacy.graph_of(task.machine)
    machine = legacy.resolved(written)

    again = collect.Task(name, machine)
    assert machine == task.machine
    assert again.strong_id() == task.strong_id()


def test_a_hand_written_graph_holds_its_parts_records():
    settings = run_files.minimal_machine()
    written = legacy.graph_of(settings)
    reference = fabrics.ReferenceFabric()

    referenced = dataclasses.replace(written, fabric=reference)
    machine = legacy.resolved(referenced)

    links = link_profiles.logical_reference_profile()
    assert machine == dataclasses.replace(settings, links=links)


def test_a_lone_pool_on_the_host_is_the_strong_decoder():
    settings = run_files.minimal_machine()
    written = legacy.graph_of(settings)
    pool = written.part("weak")
    hosted = dataclasses.replace(pool, placed=fabrics.HOST)
    others = [part for part in written.parts if part is not pool]
    moved = dataclasses.replace(written, parts=(*others, hosted))

    machine = legacy.resolved(moved)

    assert machine.weak_decoder is None
    assert machine.strong_decoder == settings.weak_decoder
    assert machine.weak_syndrome_buffer == settings.weak_syndrome_buffer


def _linked(wire: graph.Wire, links_by_target: dict) -> graph.Wire:
    link = links_by_target.get(wire.target)
    return dataclasses.replace(wire, link=link)


def test_experiment_one_is_its_parts_its_fiber_wires_and_a_cable():
    settings = regression.experiment_one_weak_alone()
    uplink = fabrics.ClockedLink(
        clock=FRIDGE_CLOCK,
        latency_cycles=40,
        bits_per_cycle=38.79,
        lane_count=4,
    )
    frame_read = fabrics.ClockedLink(
        clock=FRIDGE_CLOCK, latency_cycles=1, excludes_receiver_processing=False
    )
    downlink = fabrics.ClockedLink(
        clock=FRIDGE_CLOCK,
        latency_cycles=41,
        bits_per_cycle=38.79,
        lane_count=4,
        excludes_receiver_processing=False,
    )
    links_by_target = {
        "weak_buffer.rounds": uplink,
        "weak.rounds": frame_read,
        "frame.corrections": downlink,
    }
    written = legacy.graph_of(settings)
    wires = [_linked(wire, links_by_target) for wire in written.wires]
    linked = dataclasses.replace(
        written, wires=tuple(wires), fabric=ROCE_V2_CPU
    )

    machine = legacy.resolved(linked)

    assert machine == settings


def _extra_tier(written: graph.Machine, name: str) -> tuple:
    """A store and a pool beside the weak ones, written and read."""
    pool = written.part("weak")
    store = written.part("weak_buffer")
    store_name = f"{name}_buffer"
    parts = (
        graph.Part(name, pool.settings, fabrics.HOST),
        graph.Part(store_name, store.settings, fabrics.HOST),
    )
    wires = (
        graph.Wire("controller.rounds", f"{store_name}.rounds"),
        graph.Wire(f"{store_name}.reads", f"{name}.rounds"),
    )
    return parts, wires


def test_three_pools_wait_for_stage_four():
    written = _weak_graph()
    second_parts, second_wires = _extra_tier(written, "second")
    third_parts, third_wires = _extra_tier(written, "third")
    parts = (*second_parts, *third_parts)
    wires = (*second_wires, *third_wires)

    grown = _with(written, parts, wires)

    with pytest.raises(legacy.ShapeError, match="3 parts of kind decoder"):
        legacy.resolved(grown)


def test_two_pools_with_no_escalation_wait_for_stage_four():
    written = _weak_graph()
    parts, wires = _extra_tier(written, "second")

    grown = _with(written, parts, wires)

    with pytest.raises(legacy.ShapeError, match="escalating"):
        legacy.resolved(grown)


def test_a_store_feeding_a_store_waits_for_stage_four():
    written = _weak_graph()
    store = written.part("weak_buffer")
    without_read = [
        wire for wire in written.wires if wire.target != "weak.rounds"
    ]
    rewired = dataclasses.replace(written, wires=tuple(without_read))
    extra = (graph.Part("second_buffer", store.settings),)
    wires = (
        graph.Wire("weak_buffer.reads", "second_buffer.rounds"),
        graph.Wire("second_buffer.reads", "weak.rounds"),
    )

    grown = _with(rewired, extra, wires)

    with pytest.raises(legacy.ShapeError, match="a store feeds a store"):
        legacy.resolved(grown)


def test_a_controller_feeding_a_pool_waits_for_stage_four():
    written = _weak_graph()
    without_read = [
        wire for wire in written.wires if wire.target != "weak.rounds"
    ]
    direct = graph.Wire("controller.rounds", "weak.rounds")
    rewired = dataclasses.replace(written, wires=(*without_read, direct))

    with pytest.raises(legacy.ShapeError, match="no path of today's"):
        legacy.resolved(rewired)


def test_a_part_of_no_kind_waits_for_stage_four():
    written = _weak_graph()
    spare = graph.Part("spare_clock", written.clock)

    grown = _with(written, (spare,), ())

    with pytest.raises(legacy.ShapeError, match=STAGE_FOUR):
        legacy.resolved(grown)


def test_two_window_schemes_wait_for_stage_four():
    written = _weak_graph()
    windows = written.part("windows")
    second = graph.Part("second_windows", windows.settings)

    grown = _with(written, (second,), ())

    with pytest.raises(legacy.ShapeError, match="2 parts of kind windows"):
        legacy.resolved(grown)


def test_a_required_port_left_unwired_is_refused_by_name():
    written = _weak_graph()
    wires = [wire for wire in written.wires if wire.target != "weak.rounds"]

    unwired = dataclasses.replace(written, wires=tuple(wires))

    with pytest.raises(graph.GraphError, match=r"\['weak.rounds'\]"):
        legacy.resolved(unwired)
