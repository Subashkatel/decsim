"""The router: each patch's windows on the pool its rule names.

VQD (Maurya and Tannu 2406.17995v2) gives each logical qubit its own
decoder as its baseline (Sec. 4.1, Fig. 5) and routes each qubit's
syndromes to it by a schedule fixed before the run (Sec. 5). On the
declared fabric (tests/declared_run.py) two memories, mem1 on patch 1
and mem2 on patch 2, each have two windows: W0's input lands at 20 us
and W1's waits for W0's boundary. A decode takes 10 us, the answer
rides weak_to_frame (2 us) and commits 1 us later. A pool past the
first reads the weak store over its own hop, priced as a written
store's read (5 us), and hands a boundary to itself as the weak pool
does, as a same-pool hop in the fridge (0.5 us).
"""

import pytest

import decsim.config as config
import decsim.experiments.collect as collect
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import decsim.topology.graph as graph
import decsim.windows.routers as routers
import tests.declared_run as declared_run
import tests.topology.test_graph_layout as layout_tests

SECOND_POOL = "fpga_b"
BY_PATCH = routers.ByPatch.Settings(
    pool_by_patch=((1, "weak"), (2, SECOND_POOL))
)
# the real memories sit on patches 0 and 1
BY_PATCH_FROM_ZERO = routers.ByPatch.Settings(
    pool_by_patch=((0, "weak"), (1, SECOND_POOL))
)


def test_by_patch_sends_an_operation_to_its_first_patchs_pool():
    """A merge over patches 2 and 1 goes where patch 2's windows go."""
    merge = program_records.Operation(
        id=3, name="merge", qubits=(2, 1), patches=(2, 1)
    )

    pool = BY_PATCH.pool_for(merge)

    assert pool == SECOND_POOL


def test_by_patch_refuses_a_patch_named_twice():
    with pytest.raises(ValueError, match="names a patch twice"):
        routers.ByPatch.Settings(pool_by_patch=((1, "weak"), (1, "other")))


def test_by_patch_refuses_an_operation_on_a_patch_it_does_not_route():
    stray = declared_run.memory_operation(3)
    pool_names = ("weak", SECOND_POOL)

    with pytest.raises(ValueError, match="routes no pool for operation"):
        BY_PATCH.check_operations((stray,), pool_names)


def test_by_patch_refuses_a_pool_the_router_does_not_reach():
    first = declared_run.memory_operation(1)

    with pytest.raises(ValueError, match="does not reach"):
        BY_PATCH.check_operations((first,), ("weak",))


def test_two_memories_on_one_unit_decode_one_after_the_other():
    """The one unit stages each next input while it decodes.

    The four windows decode back to back: mem1 W0 20-30 us, mem2 W0
    30-40, mem1 W1 40-50, mem2 W1 50-60. mem2 W1's answer lands at 62 us
    and commits at 63 us.
    """
    one_unit = _two_memories()
    machine = _run(one_unit)

    records = _records_by_window(machine)

    assert records[(2, 0)] == ("weak", _ticks(42.0), _ticks(43.0))
    assert records[(2, 1)] == ("weak", _ticks(62.0), _ticks(63.0))


def test_by_patch_decodes_each_memory_on_its_own_pool_side_by_side():
    """Both W0s decode 20-30 us and commit at 33 us, one on each pool.

    Each W0's boundary leaves its pool at 30 us for its W1 on the same
    pool (0.5 us), whose input then crosses (5 us): both W1s decode
    35.5-45.5 us, land at 47.5 us and commit at 48.5 us. Had mem2's
    boundary gone to the weak pool, it would have crossed fpga_b_to_weak
    (4 us) and W1 would start at 39 us.
    """
    routed = _routed_two_memories()
    machine = _run(routed)

    records = _records_by_window(machine)

    assert records[(1, 0)] == ("weak", _ticks(32.0), _ticks(33.0))
    assert records[(2, 0)] == (SECOND_POOL, _ticks(32.0), _ticks(33.0))
    assert records[(1, 1)] == ("weak", _ticks(47.5), _ticks(48.5))
    assert records[(2, 1)] == (SECOND_POOL, _ticks(47.5), _ticks(48.5))


def test_a_pool_past_the_first_reads_the_entry_store_over_its_own_hop():
    routed = _routed_two_memories()
    layout = layout_tests.laid_out(routed)

    entry_store = layout.stores[0]
    (reader,) = entry_store.other_readers

    assert layout.entry_count == 2
    assert reader.read_hop == f"weak_buffer_to_{SECOND_POOL}"
    assert len(layout.stores) == 1


def test_routing_by_patch_changes_no_correction_on_one_decoder():
    """The same decoder on both pools: every window's answer is unchanged.

    A distance-3 memory on each of two patches, decoded by the weak
    base's PyMatching, on one pool or one pool per patch.
    """
    base = _real_two_memories()
    weak_part = base.part(machines.WEAK_POOL)
    pools = ((SECOND_POOL, weak_part.settings),)
    routed = machines.with_router(base, BY_PATCH_FROM_ZERO, pools)

    for seed in range(2):
        one_pool = _answers(base, seed)
        by_patch = _answers(routed, seed)
        assert by_patch == one_pool


def _two_memories() -> graph.Machine:
    """The declared weak-only machine running mem1 and mem2, 9 rounds."""
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2),
    ]
    return declared_run.weak_only_graph(rounds=9, operations=operations)


def _routed_two_memories() -> graph.Machine:
    """The two memories, patch 2's windows on a second weak pool."""
    one_unit = _two_memories()
    pools = ((SECOND_POOL, _weak_pool()),)
    return machines.with_router(one_unit, BY_PATCH, pools)


def _weak_pool():
    weak_only = declared_run.weak_only_graph()
    weak = weak_only.part(machines.WEAK_POOL)
    return weak.settings


def _run(declared: graph.Machine) -> machine_module.Machine:
    machine = machine_module.Machine.build(declared, 0)
    machine.run()
    return machine


def _records_by_window(machine: machine_module.Machine) -> dict:
    """Each window's answering pool, landing and commit ticks."""
    snapshot = machine.control.pauli_frame.snapshot()
    records = {}
    for record in snapshot.records:
        records[record.window_key] = (
            record.tier,
            record.accepted_ticks,
            record.committed_ticks,
        )
    return records


def _real_two_memories() -> graph.Machine:
    """The weak base running a distance-3 memory on patches 0 and 1."""
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 6, 3, 0.01
    )
    operations = []
    for index in range(2):
        stream_id = index + 1
        operation = program_records.Operation(
            id=stream_id,
            name=f"mem{index}",
            qubits=(index,),
            patches=(index,),
            circuit=circuit,
        )
        operations.append(operation)
    rounds_policy = round_policies.FixedRounds(6)
    workload = workload_settings.WorkloadSettings(
        operations=tuple(operations), rounds_policy=rounds_policy
    )
    base = machines.weak_baseline(3, 0.01, 1.0)
    workload_part = graph.Part("workload", workload)
    return base.with_part(workload_part)


def _answers(machine: graph.Machine, seed: int) -> list:
    """Each window's logical observables, by window."""
    task = collect.Task("routing", machine)
    shot = collect.run_shot(task, seed)
    snapshot = shot.machine.control.pauli_frame.snapshot()
    answers = []
    for record in snapshot.records:
        answers.append((record.window_key, record.logical_observables))
    return sorted(answers)


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)
