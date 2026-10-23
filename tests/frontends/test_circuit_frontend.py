"""A maker's workload lowered into the program the machine runs.

The wiring fills each operation's predecessors from program order on
its patches, so two operations that share a patch always carry a
dependency edge between them. The live stream's owner, its protected
region and its rounds are derived as examples/live_memory_example.py
writes them by hand; the region follows the papers' rule that a
waiting patch keeps measuring (Terhal 1302.3428 lines 2697-2698,
Holmes 2004.04794 line 451).

A lattice-surgery circuit run as one operation over its whole history
is decoded as PyMatching decodes the whole circuit's error model
(stim.Circuit.detector_error_model, decompose_errors), shot for shot,
when one window commits every round.
"""

import pathlib

import numpy
import pymatching
import pytest
import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.workload as workload_records
import tests.declared_run as declared_run

THIS_FILE = pathlib.Path(__file__)
DATA_DIRECTORY = THIS_FILE.parents[1] / "data"
# tqec's lattice-surgery CNOT at k = 1: d = 3, twelve rounds, a merge
# readout and two observables, its detectors timed by coordinate
TQEC_CNOT_PATH = DATA_DIRECTORY / "tqec_cnot_k1.stim"
TQEC_CNOT = stim.Circuit.from_file(TQEC_CNOT_PATH)


def _by_id(operations) -> dict:
    indexed = {}
    for operation in operations:
        indexed[operation.id] = operation
    return indexed


def _lowered_operations(operations) -> dict:
    workload = workload_records.Workload(tuple(operations))
    program = circuit_frontend.lowered(workload)
    return _by_id(program.operations)


def _one_window_machine(circuit: stim.Circuit, seed: int):
    """The weak baseline running the circuit as one operation, one window."""
    physical = workload_records.FiniteCircuit(circuit)
    program = program_records.Operation(1, "program", (0,), patches=(0,))
    round_count = physical.round_count()
    workload = workload_records.Workload((program,), {}, physical)
    running = workload_settings.WorkloadSettings.running(workload)
    graph = machines.weak_baseline(3, 0.002, 1.0)
    graph = graph.with_held("workload", running)
    base_windows = graph.held("windows")
    windows = declared_run.windows_on(
        base_windows, commit_rounds=round_count, buffer_rounds=0
    )
    graph = graph.with_held("windows", windows)
    return machine_module.Machine.build(graph, seed)


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

    with pytest.raises(ValueError, match="Op0 lists the same qubit more"):
        _lowered_operations([twice])


def test_a_live_stream_gets_its_owner_region_and_rounds_derived():
    """What examples/live_memory_example.py protection_workload writes."""
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

    with pytest.raises(ValueError, match="the workload's one circuit"):
        circuit_frontend.lowered(workload)


@pytest.mark.parametrize("seed", range(24))
def test_a_lattice_surgery_circuit_decodes_as_whole_circuit_matching(seed):
    """No schedule is declared: the rounds come off the detectors."""
    machine = _one_window_machine(TQEC_CNOT, seed)
    model = TQEC_CNOT.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    result = machine.run()
    (row,) = result.operation_results
    events = machine.qpu.syndrome_source.sampled_detection_events(1)
    event_bits = numpy.array(events, dtype=numpy.uint8)
    whole_circuit = matching.decode(event_bits)

    assert tuple(whole_circuit) == tuple(row.logical_observables)
