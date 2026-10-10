"""Parts by name and the wires between them (topology/graph.py).

gem5's rules for a wire: a port pairs only with a compatible role, and
a port bound once refuses a second peer (port_params.py:106-135); SST's
for the graph: a required port left unwired is found before the run
(configGraph.cc:238-262).
"""

import dataclasses

import pytest

import decsim.topology.graph as graph
import decsim.topology.legacy as legacy
import tests.experiments.run_files as run_files


def _weak_graph() -> graph.Machine:
    machine = run_files.minimal_machine()
    return legacy.graph_of(machine)


def _with_wires(machine: graph.Machine, *ends) -> graph.Machine:
    """The machine with a wire added from each (source, target) pair."""
    wires = [graph.Wire(source, target) for source, target in ends]
    return dataclasses.replace(machine, wires=(*machine.wires, *wires))


def test_a_weak_machine_wires_every_required_port():
    machine = _weak_graph()

    assert machine.unbound_ports() == []


def test_a_switching_machine_wires_every_required_port():
    settings = run_files.switching_machine()

    machine = legacy.graph_of(settings)

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

    with pytest.raises(graph.GraphError, match="'windows output'"):
        _with_wires(machine, ("weak.escalate", "frame.corrections"))


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


def test_a_part_of_no_kind_is_held_and_its_ports_wait_for_stage_four():
    machine = _weak_graph()
    clock_part = graph.Part("spare_clock", machine.clock)
    parts = (*machine.parts, clock_part)
    held = dataclasses.replace(machine, parts=parts)

    with pytest.raises(graph.GraphError, match="not implemented yet"):
        _with_wires(held, ("spare_clock.ticks", "weak.rounds"))


def test_a_qpus_distance_is_set_in_the_run_file_and_a_pools_nothing():
    machine = _weak_graph()
    qpu = machine.part("qpu")
    pool = machine.part("weak")

    assert qpu.run_file_fields() == ("distance",)
    assert pool.run_file_fields() == ()
