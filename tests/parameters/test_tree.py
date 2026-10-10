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

import decsim.parameters.assignments as assignments_module
import decsim.parameters.fields as fields
import decsim.parameters.tree as tree
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


@pytest.mark.parametrize(
    "build", [run_files.minimal_machine, run_files.switching_machine]
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
