"""A round policy yields the documented round count for an operation.

Sources: Horsman et al. 1111.4022v3 (Sec. 3.1, 3.2 and 6: one lattice
surgery step is d rounds of error correction) and Litinski 1808.02892v3
("Translation to surface codes": a multi-patch measurement is one time
step of d code cycles); the experiment configs' "10d" rounds-per-shot rule
is CodeRounds with a scale of ten.
"""

import pytest

from decsim.message import Operation, OpKind
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.qpu.round_policies import (
    CodeRounds,
    FixedRounds,
    GateRounds,
    PerOperationRounds,
    TemporalRounds,
)


def operation(operation_id, qubits=(0,), kind=OpKind.GENERIC):
    return Operation(id=operation_id, name="op", qubits=qubits, kind=kind)


def test_a_fixed_policy_gives_every_operation_the_same_count():
    policy = FixedRounds(7)
    memory = operation(1)
    distance_three = SurfaceCodeModel(distance=3)
    distance_nine = SurfaceCodeModel(distance=9)
    assert policy.rounds_for(memory, distance_three) == 7
    assert policy.rounds_for(memory, distance_nine) == 7


def test_ten_d_rounds_is_the_code_rounds_policy_scaled_by_ten():
    policy = CodeRounds(scale=10)
    memory = operation(1)
    distance_three = SurfaceCodeModel(distance=3)
    distance_five = SurfaceCodeModel(distance=5)
    assert policy.rounds_for(memory, distance_three) == 30
    assert policy.rounds_for(memory, distance_five) == 50


def test_code_rounds_never_fall_below_one():
    policy = CodeRounds(scale=0.1)
    memory = operation(1)
    distance_three = SurfaceCodeModel(distance=3)
    assert policy.rounds_for(memory, distance_three) == 1


def test_a_merge_costs_two_steps_of_d_rounds_and_a_measurement_one_round():
    policy = GateRounds()
    code = SurfaceCodeModel(distance=5)
    merge = operation(1, kind=OpKind.MERGE)
    measure = operation(2, kind=OpKind.MEASURE)
    inject = operation(3, kind=OpKind.INJECT)
    memory = operation(4, kind=OpKind.MEMORY)
    two_qubit = operation(5, qubits=(0, 1))
    one_qubit = operation(6, qubits=(0,))
    assert policy.rounds_for(merge, code) == 10
    assert policy.rounds_for(measure, code) == 1
    assert policy.rounds_for(inject, code) == 1
    assert policy.rounds_for(memory, code) == 5
    assert policy.rounds_for(two_qubit, code) == 10
    assert policy.rounds_for(one_qubit, code) == 5


def test_a_temporal_distance_replaces_d_for_surgery_only():
    policy = TemporalRounds(4)
    code = SurfaceCodeModel(distance=7)
    merge = operation(1, kind=OpKind.MERGE)
    two_qubit = operation(2, qubits=(0, 1))
    memory = operation(3, kind=OpKind.MEMORY)
    assert policy.rounds_for(merge, code) == 4
    assert policy.rounds_for(two_qubit, code) == 4
    assert policy.rounds_for(memory, code) == 7


def test_per_operation_counts_win_over_the_fallback_and_may_be_zero():
    fallback = FixedRounds(9)
    policy = PerOperationRounds({1: 0, 2: 4}, fallback=fallback)
    code = SurfaceCodeModel(distance=3)
    first = operation(1)
    second = operation(2)
    third = operation(3)
    assert policy.rounds_for(first, code) == 0
    assert policy.rounds_for(second, code) == 4
    assert policy.rounds_for(third, code) == 9


def test_a_policy_that_would_give_no_rounds_is_refused():
    with pytest.raises(ValueError, match=">= 1 round"):
        FixedRounds(0)
    with pytest.raises(ValueError, match=">= 0 rounds"):
        PerOperationRounds({1: -1})
