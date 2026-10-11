"""The kinds of part and their ports (topology/part_kinds.py).

The referent for a kind is MachineSettings itself: a record is of the
kind whose slot's type takes it (parameters/kinds.py), and a slot
taking it is what the run does with it today.
"""

import collections
import dataclasses
import typing

import decsim.frontends.settings as workload_settings
import decsim.parameters.kinds as kinds
import decsim.producers as producers
import decsim.settings as machine_settings
import decsim.topology.part_kinds as part_kinds
from decsim.decoders.union_find import decoder as union_find
from decsim.syndrome_buffer import ported_syndrome_buffer


def _slot_records() -> list:
    """(kind, record class) for each record each kind's slots take."""
    hints = typing.get_type_hints(machine_settings.MachineSettings)
    pairs = []
    for kind in part_kinds.KINDS:
        for slot in kind.slots:
            taken = kinds.records_for(hints[slot])
            pairs.extend((kind, record_class) for record_class in taken)
    return pairs


def test_a_record_is_of_the_kind_whose_slot_takes_it():
    for kind, record_class in _slot_records():
        assert part_kinds.kind_of(record_class) is kind


def test_every_slot_holding_a_part_belongs_to_one_kind():
    machine_level = {"clock", "links", "observation"}
    fields = dataclasses.fields(machine_settings.MachineSettings)
    slots = [field.name for field in fields if field.name not in machine_level]

    kind_slots = [slot for kind in part_kinds.KINDS for slot in kind.slots]

    assert sorted(kind_slots) == sorted(slots)


def test_a_ported_store_is_a_syndrome_buffer():
    record_class = ported_syndrome_buffer.PortedSyndromeBufferSettings

    kind = part_kinds.kind_of(record_class)

    assert kind.name == "syndrome_buffer"


def test_a_decoders_algorithm_is_no_part_of_its_own():
    algorithm = union_find.UnionFindDecoder.Settings

    kind = part_kinds.kind_of(algorithm)

    assert kind is None


def test_every_signal_has_an_output_and_an_input():
    directions = collections.defaultdict(set)

    for kind in part_kinds.KINDS:
        for port in kind.ports:
            directions[port.signal].add(port.direction)

    for signal, found in directions.items():
        assert found == {part_kinds.OUTPUT, part_kinds.INPUT}, signal


def test_a_workload_lowered_in_the_run_file_fixes_a_qpus_distance():
    qpu = part_kinds.KINDS[0]
    lowered = workload_settings.WorkloadSettings()

    fields = part_kinds.run_file_fields(qpu, lowered)

    assert fields == ("distance",)


def test_a_memory_made_at_the_task_leaves_a_qpus_distance_free():
    qpu = part_kinds.KINDS[0]
    memory = producers.MemorySettings(0.001, rounds=5)

    fields = part_kinds.run_file_fields(qpu, memory)

    assert fields == ()


def test_parts_with_no_workload_part_run_the_machines_default():
    defaults = machine_settings.MachineSettings()

    workload = part_kinds.workload_of(())

    assert workload == defaults.workload
