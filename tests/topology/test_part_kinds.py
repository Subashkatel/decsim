"""The kinds of part and their ports (topology/part_kinds.py).

A record is of the kind whose type takes it (parameters/kinds.py), and
a kind a machine holds once runs on its default record when the machine
holds no part of it.
"""

import collections

import decsim.frontends.settings as workload_settings
import decsim.parameters.kinds as kinds
import decsim.producers as producers
import decsim.topology.graph as graph
import decsim.topology.part_kinds as part_kinds
from decsim.decoders.union_find import decoder as union_find
from decsim.syndrome_buffer import ported_syndrome_buffer


def test_a_record_is_of_the_kind_whose_type_takes_it():
    for kind in part_kinds.KINDS:
        for record_class in kinds.records_for(kind.hint):
            assert part_kinds.kind_of(record_class) is kind, record_class


def test_a_kind_is_found_by_its_name():
    for kind in part_kinds.KINDS:
        assert part_kinds.kind_named(kind.name) is kind


def test_each_kind_held_once_defaults_to_a_record_of_its_own():
    held_once = [kind for kind in part_kinds.KINDS if not kind.is_many]
    defaulted = [kind for kind in held_once if kind.default is not None]

    for kind in defaulted:
        default_class = type(kind.default)
        assert part_kinds.kind_of(default_class) is kind, kind.name


def test_only_the_frame_and_the_router_are_absent_when_not_held():
    held_once = [kind for kind in part_kinds.KINDS if not kind.is_many]

    absent = {kind.name for kind in held_once if kind.default is None}

    assert absent == {"pauli_frame", "router"}


def test_a_kind_held_many_times_has_no_default():
    held_many = [kind for kind in part_kinds.KINDS if kind.is_many]

    defaults = [kind.default for kind in held_many]

    assert defaults == [None] * len(held_many)


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


def test_a_machine_with_no_workload_part_runs_the_kinds_default():
    machine = graph.Machine()

    workload = machine.held(part_kinds.WORKLOAD)

    assert workload == workload_settings.WorkloadSettings()
