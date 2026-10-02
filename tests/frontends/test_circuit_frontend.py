"""A maker's workload lowered into the program the machine runs.

The wiring fills each operation's predecessors from program order on
its patches, so two operations that share a patch always carry a
dependency edge between them. The live stream's owner, its protected
region and its rounds are derived as tools/live_memory_example.py
writes them by hand; the region follows the papers' rule that a
waiting patch keeps measuring (Terhal 1302.3428 lines 2697-2698,
Holmes 2004.04794 line 451).
"""

import pytest
import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.workload as workload_records


def _by_id(operations) -> dict:
    indexed = {}
    for operation in operations:
        indexed[operation.id] = operation
    return indexed


def _lowered_operations(operations) -> dict:
    workload = workload_records.Workload(tuple(operations))
    program = circuit_frontend.lowered(workload)
    return _by_id(program.operations)


def _live_workload() -> workload_records.Workload:
    """Decode after three rounds, then wait for the answer."""
    prefix = program_records.Operation(
        1, "prefix", ("p",), patches=("p",), stream_id=100, stream_offset=0
    )
    protect = program_records.Operation(
        2, "protect", ("p",), patches=("p",), emits_detector_data=False
    )
    resume = program_records.Operation(
        3,
        "resume",
        ("p",),
        patches=("p",),
        blocked_by=1,
        emits_detector_data=False,
    )
    readout = program_records.Operation(
        4, "readout", ("p",), patches=("p",), emits_detector_data=False
    )
    fragment = stim.Circuit("M 0")
    program = circuit_records.RepeatedStimCircuit(
        fragment, fragment, fragment, fragment
    )
    rounds = {1: 3, 2: 0, 3: 1, 4: 0}
    operations = (prefix, protect, resume, readout)
    return workload_records.Workload(operations, rounds, program)


def test_two_operations_that_share_a_patch_carry_an_edge_between_them():
    first = program_records.Operation(0, "Op0", (0, 1))
    second = program_records.Operation(1, "Op1", (1, 2))

    indexed = _lowered_operations([first, second])

    assert indexed[1].predecessors == (0,)


def test_two_operations_that_share_no_patch_carry_no_edge():
    first = program_records.Operation(0, "Op0", (0, 1))
    second = program_records.Operation(1, "Op1", (2, 3))

    indexed = _lowered_operations([first, second])

    assert indexed[0].predecessors == ()
    assert indexed[1].predecessors == ()


def test_a_declared_predecessor_is_kept_beside_the_patch_order_ones():
    first = program_records.Operation(0, "Op0", (0,))
    second = program_records.Operation(1, "Op1", (1,))
    third = program_records.Operation(2, "Op2", (1,), predecessors=(0,))

    indexed = _lowered_operations([first, second, third])

    assert indexed[2].predecessors == (0, 1)


def test_the_decoder_boundary_edges_are_the_program_order_edges():
    first = program_records.Operation(0, "Op0", (0, 1))
    second = program_records.Operation(1, "Op1", (2, 3))
    third = program_records.Operation(2, "Op2", (1, 3))

    indexed = _lowered_operations([first, second, third])

    operations = indexed.values()
    boundaries = [each.decoder_boundary_predecessors for each in operations]
    predecessors = [each.predecessors for each in operations]
    assert boundaries == predecessors
    assert indexed[2].predecessors == (0, 1)


def test_an_operation_with_no_detector_data_has_no_decoder_boundary():
    """Only an operation the plan decodes as windows joins another's."""
    first = program_records.Operation(0, "Op0", (0,))
    wait = program_records.Operation(1, "wait", (0,), emits_detector_data=False)
    later = program_records.Operation(2, "Op2", (0,), predecessors=(0,))

    indexed = _lowered_operations([first, wait, later])

    assert indexed[1].decoder_boundary_predecessors == ()
    assert indexed[2].predecessors == (0, 1)
    assert indexed[2].decoder_boundary_predecessors == (0,)


def test_an_operation_that_lists_one_qubit_twice_is_refused():
    twice = program_records.Operation(0, "Op0", (1, 1))

    with pytest.raises(ValueError) as refusal:
        _lowered_operations([twice])

    assert "lists the same qubit more than once" in str(refusal.value)


def test_a_live_stream_gets_its_owner_region_and_rounds_derived():
    """What tools/live_memory_example.py protection_workload writes."""
    workload = _live_workload()
    program = circuit_frontend.lowered(workload)
    owner = program.dynamic_streams[0]
    region = program.protected_regions[0]
    policy = program.rounds_policy

    assert owner.id == 100
    assert owner.patches == ("p",)
    assert region == program_records.ProtectedRegion(100, 2, 4)
    assert policy.rounds_by_operation == (
        (1, 3),
        (2, 0),
        (3, 1),
        (4, 0),
        (100, 0),
    )
    assert program.physical_circuits == ((100, workload.physical),)


def test_one_circuit_under_two_operations_without_ranges_is_refused():
    """No merged circuit is built, so each operation needs its range."""
    first = program_records.Operation(1, "memory", (0,))
    second = program_records.Operation(2, "merge", (0, 1))
    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]")
    physical = workload_records.FiniteCircuit(circuit, {0: 1})
    workload = workload_records.Workload((first, second), {}, physical)

    with pytest.raises(ValueError, match="none names its round range"):
        circuit_frontend.lowered(workload)


def test_the_one_operation_running_a_circuit_takes_the_circuits_rounds():
    memory = program_records.Operation(1, "memory", (0,))
    circuit = stim.Circuit("M 0\nM 0\nDETECTOR rec[-1] rec[-2]")
    physical = workload_records.FiniteCircuit(circuit, {0: 1, 1: 2})
    workload = workload_records.Workload((memory,), {}, physical)

    program = circuit_frontend.lowered(workload)
    operation = program.operations[0]

    assert operation.circuit == circuit
    assert program.rounds_policy.rounds_by_operation == ((1, 2),)
    assert program.physical_circuits == ((1, physical),)
