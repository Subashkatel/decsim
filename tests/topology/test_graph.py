"""Parts by name and the wires between them (topology/graph.py).

gem5's rules for a wire: a port pairs only with a compatible role, and
a port bound once refuses a second peer (port_params.py:106-135); SST's
for the graph: a required port left unwired is found before the run
(configGraph.cc:238-262).
"""

import dataclasses

import pytest

import decsim.topology.graph as graph
import tests.experiments.run_files as run_files


def _weak_graph() -> graph.Machine:
    return run_files.minimal_machine()


def _with_wires(machine: graph.Machine, *ends) -> graph.Machine:
    """The machine with a wire added from each (source, target) pair."""
    wires = [graph.Wire(source, target) for source, target in ends]
    return dataclasses.replace(machine, wires=(*machine.wires, *wires))


def test_a_weak_machine_wires_every_required_port():
    machine = _weak_graph()

    assert machine.unbound_ports() == []


def test_a_switching_machine_wires_every_required_port():
    machine = run_files.switching_machine()

    assert machine.unbound_ports() == []


def test_a_required_port_left_unwired_is_named():
    machine = _weak_graph()
    wires = [wire for wire in machine.wires if wire.target != "weak.rounds"]

    unwired = dataclasses.replace(machine, wires=tuple(wires))

    assert unwired.unbound_ports() == ["weak.rounds"]


def test_a_part_is_found_by_name():
    machine = _weak_graph()

    part = machine.part("weak")

    assert part.kind.name == "decoder_pool"
    assert part.placed == "fridge"


def test_an_unknown_part_is_refused_with_the_parts_named():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="no part 'relay'.*'weak'"):
        _with_wires(machine, ("weak.escalate", "relay.escalated"))


def test_an_unknown_port_is_refused_with_the_kinds_ports_named():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="no port 'answers'.*escalate"):
        _with_wires(machine, ("weak.answers", "frame.corrections"))


def test_a_wire_end_without_a_port_is_refused():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="part.port"):
        _with_wires(machine, ("weak", "frame.corrections"))


def test_a_wire_from_an_input_is_refused():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="do not pair"):
        _with_wires(machine, ("frame.corrections", "weak.rounds"))


def test_a_wire_between_two_signals_is_refused():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="'escalations output'"):
        _with_wires(machine, ("weak.escalate", "frame.corrections"))


def test_a_pools_escalations_reach_a_switching_part_never_a_pool():
    """The edge's verdict sits between the pools; a direct wire is refused."""
    machine = run_files.switching_machine()
    wires = [wire for wire in machine.wires if "switching." not in wire.target]
    without_edge = dataclasses.replace(machine, wires=tuple(wires))

    with pytest.raises(graph.GraphError, match="do not pair"):
        _with_wires(without_edge, ("weak.escalate", "strong.escalated"))


def test_a_port_taking_one_wire_refuses_a_second():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="weak.corrections takes one"):
        _with_wires(machine, ("weak.corrections", "frame.corrections"))


def test_a_port_taking_many_wires_takes_a_second():
    machine = _weak_graph()
    store = machine.part("weak_buffer")
    second_store = graph.Part("second_buffer", store.settings)
    parts = (*machine.parts, second_store)
    wire = graph.Wire("controller.rounds", "second_buffer.rounds")

    rewired = dataclasses.replace(
        machine, parts=parts, wires=(*machine.wires, wire)
    )

    assert rewired.unbound_ports() == []


def test_two_parts_of_one_name_are_refused():
    machine = _weak_graph()
    parts = (*machine.parts, machine.part("weak"))

    with pytest.raises(graph.GraphError, match="two parts are named 'weak'"):
        dataclasses.replace(machine, parts=parts)


def test_a_part_of_no_kind_is_held_and_its_ports_wait_for_stage_five():
    machine = _weak_graph()
    clock_part = graph.Part("spare_clock", machine.clock)
    parts = (*machine.parts, clock_part)
    held = dataclasses.replace(machine, parts=parts)

    with pytest.raises(graph.GraphError, match="not implemented yet"):
        _with_wires(held, ("spare_clock.ticks", "weak.rounds"))


def test_a_lowered_workload_fixes_a_qpus_distance_and_a_pools_nothing():
    machine = _weak_graph()
    workload = machine.part("workload")
    qpu = machine.part("qpu")
    pool = machine.part("weak")

    assert qpu.run_file_fields(workload.settings) == ("distance",)
    assert pool.run_file_fields(workload.settings) == ()
    paths = machine.set_in_the_run_file()
    assert paths[0] == "qpu.distance"


def test_a_part_of_a_known_name_takes_its_namesakes_place():
    machine = _weak_graph()
    pool = machine.part("weak")
    two_units = dataclasses.replace(pool.settings, unit_count=2)
    replacement = graph.Part("weak", two_units, "fridge")

    changed = machine.with_part(replacement)

    names = [part.name for part in changed.parts]
    assert names == [part.name for part in machine.parts]
    changed_pool = changed.part("weak")
    assert changed_pool.settings.unit_count == 2


def test_a_part_of_a_new_name_is_added_last():
    machine = _weak_graph()
    store = machine.part("weak_buffer")
    second_store = graph.Part("spare_buffer", store.settings, "host")

    grown = machine.with_part(second_store)

    assert grown.parts[-1] == second_store
    assert len(grown.parts) == len(machine.parts) + 1


def test_a_parts_fields_change_and_its_placement_stays():
    machine = _weak_graph()

    changed = machine.with_settings("weak", unit_count=3)

    pool = changed.part("weak")
    assert pool.settings.unit_count == 3
    assert pool.placed == "fridge"


def test_a_record_swapped_in_keeps_its_parts_site():
    machine = run_files.switching_machine()
    strong = machine.part("strong")
    two_units = dataclasses.replace(strong.settings, unit_count=2)

    changed = machine.with_record("strong", two_units)

    changed_strong = changed.part("strong")
    assert changed_strong.settings == two_units
    assert changed_strong.placed == "host"


def test_added_wires_follow_the_machines_own():
    machine = _weak_graph()
    without = machine.without_part("weak")
    pool = machine.part("weak")
    readded = without.with_part(pool)
    weak_wires = [wire for wire in machine.wires if wire not in without.wires]

    rewired = readded.with_wires(*weak_wires)

    assert rewired.unbound_ports() == []
    assert rewired.wires[-len(weak_wires) :] == tuple(weak_wires)


def test_a_part_removed_takes_every_wire_that_reaches_it():
    machine = _weak_graph()

    without = machine.without_part("weak")

    ends = []
    for wire in without.wires:
        ends.extend((wire.source, wire.target))
    reaching = [end for end in ends if end.startswith("weak.")]
    assert reaching == []
    assert "weak" not in [part.name for part in without.parts]


def test_removing_an_unknown_part_is_refused():
    machine = _weak_graph()

    with pytest.raises(graph.GraphError, match="no part 'relay'"):
        machine.without_part("relay")
