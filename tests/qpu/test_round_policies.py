"""A round policy yields the documented round count for an operation.

Sources: the lattice-surgery unit of d rounds per step that
decsim/qpu/round_policies.py takes from Horsman et al. 1111.4022 and
Litinski 1808.02892; neither paper is on disk in the sandbox, so the
section citations live in that module and are not verified here. The
policies receive the OperationPlanningView the planner builds
(decsim/frontends/planner.py). The yaml configs' "10d"
rounds-per-shot rule is resolved by the workload settings
(decsim/frontends/settings.py, RoundsPerShot.rounds_for: ten times the
distance) and handed to the run as FixedRounds(rounds)
(decsim/frontends/settings.py:147); CodeRounds(scale=10) gives the same
count from the surface card and is not what the yaml configs use.
"""

import pytest

import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records


def operation(operation_id, qubits=(0,), kind=program_records.OpKind.GENERIC):
    workload_operation = program_records.Operation(
        id=operation_id, name="op", qubits=qubits, kind=kind
    )
    return program_records.OperationPlanningView.from_operation(
        workload_operation
    )


def test_a_fixed_policy_gives_every_operation_the_same_count():
    policy = round_policies.FixedRounds(7)
    memory = operation(1)
    distance_three = code_geometry.SurfaceCodeModel(distance=3)
    distance_nine = code_geometry.SurfaceCodeModel(distance=9)
    assert policy.rounds_for(memory, distance_three) == 7
    assert policy.rounds_for(memory, distance_nine) == 7


def test_the_code_rounds_policy_scaled_by_ten_gives_ten_d_rounds():
    policy = round_policies.CodeRounds(scale=10)
    memory = operation(1)
    distance_three = code_geometry.SurfaceCodeModel(distance=3)
    distance_five = code_geometry.SurfaceCodeModel(distance=5)
    assert policy.rounds_for(memory, distance_three) == 30
    assert policy.rounds_for(memory, distance_five) == 50


def test_code_rounds_never_fall_below_one():
    policy = round_policies.CodeRounds(scale=0.1)
    memory = operation(1)
    distance_three = code_geometry.SurfaceCodeModel(distance=3)
    assert policy.rounds_for(memory, distance_three) == 1


def test_a_merge_costs_two_steps_of_d_rounds():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    merge = operation(1, kind=program_records.OpKind.MERGE)
    assert policy.rounds_for(merge, code) == 10


def test_a_measurement_costs_one_round():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    measure = operation(2, kind=program_records.OpKind.MEASURE)
    assert policy.rounds_for(measure, code) == 1


def test_an_injection_costs_one_round():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    inject = operation(3, kind=program_records.OpKind.INJECT)
    assert policy.rounds_for(inject, code) == 1


def test_memory_costs_d_rounds():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    memory = operation(4, kind=program_records.OpKind.MEMORY)
    assert policy.rounds_for(memory, code) == 5


def test_memory_on_two_qubits_still_costs_d_rounds():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    memory = operation(4, qubits=(0, 1), kind=program_records.OpKind.MEMORY)
    assert policy.rounds_for(memory, code) == 5


def test_idle_costs_d_rounds():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    idle = operation(4, qubits=(0, 1), kind=program_records.OpKind.IDLE)
    assert policy.rounds_for(idle, code) == 5


def test_a_generic_two_qubit_operation_costs_a_merge():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    two_qubit = operation(5, qubits=(0, 1))
    assert policy.rounds_for(two_qubit, code) == 10


def test_a_generic_one_qubit_operation_costs_memory():
    policy = round_policies.GateRounds()
    code = code_geometry.SurfaceCodeModel(distance=5)
    one_qubit = operation(6, qubits=(0,))
    assert policy.rounds_for(one_qubit, code) == 5


def test_a_temporal_distance_replaces_d_for_surgery_only():
    policy = round_policies.TemporalRounds(4)
    code = code_geometry.SurfaceCodeModel(distance=7)
    merge = operation(1, kind=program_records.OpKind.MERGE)
    two_qubit = operation(2, qubits=(0, 1))
    memory = operation(3, kind=program_records.OpKind.MEMORY)
    assert policy.rounds_for(merge, code) == 4
    assert policy.rounds_for(two_qubit, code) == 4
    assert policy.rounds_for(memory, code) == 7


def test_per_operation_counts_win_over_the_fallback_and_may_be_zero():
    fallback = round_policies.FixedRounds(9)
    policy = round_policies.PerOperationRounds({1: 0, 2: 4}, fallback=fallback)
    code = code_geometry.SurfaceCodeModel(distance=3)
    first = operation(1)
    second = operation(2)
    third = operation(3)
    assert policy.rounds_for(first, code) == 0
    assert policy.rounds_for(second, code) == 4
    assert policy.rounds_for(third, code) == 9


def test_a_fixed_policy_without_a_round_is_refused():
    with pytest.raises(ValueError, match=">= 1 round"):
        round_policies.FixedRounds(0)


def test_a_negative_per_operation_count_is_refused():
    with pytest.raises(ValueError, match=">= 0 rounds"):
        round_policies.PerOperationRounds({1: -1})
