"""Paths over a machine's settings, assignments along them (tree.py).

The referent is the record built in Python: an assignment by path gives
the record dataclasses.replace gives with the same value, and every
path's value written back as text gives the machine it was read from.
Patterns are OMNeT++'s ini patterns (simulation manual §10.3): `*` one
name, `**` any number of them; the later assignment wins, as ns-3's
command line wins over its defaults.
"""

import dataclasses

import pytest

import decsim.confidence.no_signal as no_signal
import decsim.decoders.settings as decoder_settings
import decsim.parameters.assignments as assignments_module
import decsim.parameters.fields as fields
import decsim.parameters.tree as tree
import decsim.presets.machines as machines
import decsim.topology.slotted as slotted
import tests.experiments.run_files as run_files
from decsim.decoders.union_find import decoder as union_find

ORIGIN = "a test"


def _assignment(pattern: str, written) -> tree.Assignment:
    return tree.Assignment(pattern, written, ORIGIN)


def _applied(machine, *pairs):
    assignments = [_assignment(pattern, written) for pattern, written in pairs]
    applied = tree.apply(machine, assignments)
    return applied.settings


def _settable(machine) -> list:
    """Every path's value as written and read back, as --set reads it.

    The paths set in the run file and the values set in Python only
    (shown as <type>) are left out.
    """
    settings = tree.settings_of(machine)
    assignments = []
    for setting in settings:
        if tree.is_set_in_the_run_file(machine, setting.path):
            continue
        if setting.written.startswith("<"):
            continue
        text = f"{setting.path}={setting.written}"
        parsed = assignments_module.parse(text, ORIGIN)
        assignments.extend(parsed)
    return assignments


def _weak_graph():
    machine = run_files.minimal_machine()
    return slotted.graph_of(machine)


def _switching_graph():
    machine = run_files.switching_machine()
    return slotted.graph_of(machine)


def _with_part(machine, name, **changes):
    """The graph with one part's record replaced, the referent of a path."""
    parts = []
    for part in machine.parts:
        new_part = part
        if part.name == name:
            settings = dataclasses.replace(part.settings, **changes)
            new_part = dataclasses.replace(part, settings=settings)
        parts.append(new_part)
    return dataclasses.replace(machine, parts=tuple(parts))


@pytest.mark.parametrize(
    "build",
    [
        run_files.minimal_machine,
        run_files.switching_machine,
        _weak_graph,
        _switching_graph,
    ],
)
def test_every_value_written_back_gives_the_machine_it_was_read_from(build):
    machine = build()
    assignments = _settable(machine)

    applied = tree.apply(machine, assignments)

    assert applied.settings == machine
    assert len(applied.matched) == len(assignments)


def test_an_assignment_gives_the_record_replace_gives():
    machine = run_files.minimal_machine()
    pool = dataclasses.replace(machine.weak_decoder, unit_count=3)
    replaced = dataclasses.replace(machine, weak_decoder=pool)

    assigned = _applied(machine, ("weak_decoder.unit_count", 3))

    assert assigned == replaced


def test_the_later_of_two_assignments_to_one_path_wins():
    machine = run_files.minimal_machine()

    assigned = _applied(
        machine,
        ("weak_decoder.unit_count", 3),
        ("weak_decoder.unit_count", 5),
    )

    assert assigned.weak_decoder.unit_count == 5


def test_a_star_stands_for_one_name_and_two_stars_for_any_number():
    machine = run_files.switching_machine()

    one_level = _applied(machine, ("*_decoder.unit_count", 2))
    any_level = _applied(machine, ("**.fetch_per_round", "2 cycles"))

    assert one_level.weak_decoder.unit_count == 2
    assert one_level.strong_decoder.unit_count == 2
    assert any_level.weak_decoder.engine.fetch_cycles_per_round == 2
    assert any_level.strong_decoder.engine.fetch_cycles_per_round == 2


def test_a_time_for_cycles_counts_on_the_parts_clock_or_the_machines():
    machine = run_files.minimal_machine()
    clock = machine.weak_decoder.engine.clock

    assigned = _applied(
        machine, ("weak_decoder.engine.release_per_job", "20ns")
    )

    engine = assigned.weak_decoder.engine
    assert engine.release_cycles_per_job == clock.cycles_for(20_000)


def test_a_record_built_with_its_clock_counts_a_time_on_that_clock():
    """25 ns on a 100 MHz part is 3 cycles, with no machine clock to fall on."""
    clock = tree.Assignment("clock", "100MHz", ORIGIN)
    release = tree.Assignment("release_per_job", "25ns", ORIGIN)

    built = tree.built(decoder_settings.EngineSettings, [release, clock])

    engine = built.settings
    assert engine.release_cycles_per_job == 3


def test_a_type_assignment_swaps_the_part_and_its_parameters_follow():
    machine = run_files.minimal_machine()

    assigned = _applied(
        machine,
        ("weak_decoder.algorithm.type", "union_find"),
        ("weak_decoder.algorithm.timing.type", "HostMeasuredTime"),
        ("weak_decoder.algorithm.weight_step", 0.5),
    )

    algorithm = assigned.weak_decoder.algorithm
    expected = union_find.UnionFindDecoder.Settings(
        weight_step=0.5, timing=run_files.HOST_TIME
    )
    assert algorithm == expected


def test_a_part_is_added_to_an_empty_slot_and_emptied_again():
    machine = run_files.minimal_machine()
    capacity = "links.qpu_to_controller.channel.capacity"

    added = _applied(
        machine,
        (f"{capacity}.type", "Capacity"),
        (f"{capacity}.input", "100Gbps"),
        (f"{capacity}.source", "a test's wire"),
    )
    emptied = _applied(added, (f"{capacity}.type", "none"))

    channel = added.links.qpu_to_controller.channel
    assert channel.capacity.input_bits_per_microsecond == 100_000
    assert emptied == machine


def test_none_in_a_slot_that_must_hold_a_record_is_the_row_named_none():
    machine = run_files.switching_machine()

    applied = _applied(machine, ("switching.confidence.type", "none"))

    assert applied.switching.confidence == no_signal.NoSignal.Settings()
    with pytest.raises(fields.ParameterError, match="named 'none'"):
        _applied(machine, ("weak_decoder.algorithm.type", "none"))


def test_a_new_part_with_a_required_field_unset_is_refused_by_its_path():
    machine = run_files.minimal_machine()
    capacity = "links.qpu_to_controller.channel.capacity"

    with pytest.raises(
        fields.ParameterError, match=f"{capacity}.input is required"
    ):
        _applied(machine, (f"{capacity}.type", "Capacity"))


def test_a_value_decsim_refuses_is_refused_by_its_records_path():
    machine = run_files.minimal_machine()

    with pytest.raises(
        fields.ParameterError, match="^weak_decoder: decsim refused"
    ):
        _applied(machine, ("weak_decoder.unit_count", 0))


def test_a_run_file_path_is_refused_by_name_and_passed_by_a_wildcard():
    machine = run_files.minimal_machine()

    wildcard = _assignment("**.distance", 5)
    passed = tree.apply(machine, [wildcard])

    assert passed.settings == machine
    assert not passed.matched
    with pytest.raises(fields.ParameterError, match="set in the run file"):
        _applied(machine, ("qpu.distance", 5))


def test_the_listing_names_a_slots_type_before_its_parameters():
    machine = run_files.minimal_machine()

    settings = tree.settings_of(machine)

    paths = [setting.path for setting in settings]
    type_position = paths.index("weak_decoder.engine.type")
    fetch_position = paths.index("weak_decoder.engine.fetch_per_round")
    assert type_position < fetch_position
    assert "workload.operations[0].circuit" in paths


def test_a_parts_parameter_sits_at_the_parts_name():
    machine = _weak_graph()
    replaced = _with_part(machine, "weak", unit_count=3)

    assigned = _applied(machine, ("weak.unit_count", 3))

    assert assigned == replaced


def test_a_parts_site_sits_beside_its_records_parameters():
    machine = _weak_graph()

    assigned = _applied(machine, ("weak.placed", "host"))

    pool = assigned.part("weak")
    old_pool = machine.part("weak")
    assert pool.placed == "host"
    assert pool.settings == old_pool.settings


def test_a_parts_record_is_swapped_within_its_kind():
    machine = _weak_graph()

    assigned = _applied(
        machine, ("weak_buffer.type", "PortedSyndromeBufferSettings")
    )

    store = assigned.part("weak_buffer")
    store_class = type(store.settings)
    assert store_class.__name__ == "PortedSyndromeBufferSettings"
    with pytest.raises(
        fields.ParameterError, match="no record of this slot is named"
    ):
        _applied(machine, ("weak_buffer.type", "UnionFindDecoder"))


def test_a_parts_cycles_count_on_the_machines_clock():
    machine = _weak_graph()
    replaced = _with_part(machine, "controller", packing_cycles_per_round=2)

    assigned = _applied(machine, ("controller.packing_per_round", "8ns"))

    assert assigned == replaced


def test_a_graphs_run_file_path_is_named_by_its_part():
    machine = _weak_graph()

    with pytest.raises(fields.ParameterError, match="set in the run file"):
        _applied(machine, ("qpu.distance", 5))


def test_a_distance_is_set_by_path_where_the_memory_is_made_at_the_task():
    machine = machines.weak_baseline(3, 0.002, 1.0)

    assigned = _applied(machine, ("qpu.distance", 5))

    qpu = assigned.part("qpu")
    assert qpu.settings.distance == 5


def test_a_memory_rows_own_fields_are_set_by_path():
    machine = machines.weak_baseline(3, 0.002, 1.0)

    assigned = _applied(
        machine,
        ("workload.rounds_per_distance", "none"),
        ("workload.rounds", 40),
        ("workload.physical_error_probability", 0.004),
    )

    workload = assigned.part("workload")
    assert workload.settings.rounds == 40
    assert workload.settings.physical_error_probability == 0.004


def test_a_rewired_graph_is_checked_again():
    machine = _weak_graph()

    with pytest.raises(fields.ParameterError, match="do not pair"):
        _applied(machine, ("wires[5].source", "frame.decisions"))
