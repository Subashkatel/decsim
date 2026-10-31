"""The final-result gate: a decision starts once earlier ones are final.

A decision released on a provisional answer may be repaired by a Clifford
correction, which must be in before the next decision that descends from
it starts, because that decision's basis is fixed at its start
(Chamberland, Iyer and Poulin 1704.06662 Sec. 2; Litinski 1808.02892
p. 5). The final notice crosses frame_to_controller like a decision
(tests/pauli_frame/test_decision_dispatch.py); here it lands at once.
"""

import decsim.controller.final_result_gate as final_result_gate
import decsim.engine as engine_module
import decsim.records.program as program_records

# a T injection, its conditional S, then a second T on the same patch
# after the S, with its own conditional S
T_FIRST = program_records.Operation(1, "t first", (0,), patches=(0,))
S_FIRST = program_records.Operation(
    2, "s first", (0,), patches=(0,), predecessors=(1,), blocked_by=1
)
T_SECOND = program_records.Operation(
    3, "t second", (0,), patches=(0,), predecessors=(2,)
)
S_SECOND = program_records.Operation(
    4, "s second", (0,), patches=(0,), predecessors=(3,), blocked_by=3
)
TWO_T_GATES = (T_FIRST, S_FIRST, T_SECOND, S_SECOND)


class LandingDispatch:
    """A frame path with no fabric: the notice lands at once."""

    def __init__(self):
        self.sent = []

    def dispatch_final_result(self, operation_id, landed):
        self.sent.append(operation_id)
        landed()


class CountingRuntime:
    def __init__(self):
        self.retries = 0

    def retry_ready_operations(self):
        self.retries += 1


def gate_over(operations):
    engine = engine_module.Engine()
    gate = final_result_gate.FinalResultGate(engine, operations)
    gate.dispatch = LandingDispatch()
    gate.runtime = CountingRuntime()
    return gate


def test_a_decision_waits_for_the_final_result_it_descends_from():
    gate = gate_over(TWO_T_GATES)

    held = gate.is_open(T_SECOND)
    gate.note_result_final(T_FIRST)

    assert not held
    assert gate.is_open(T_SECOND)
    assert gate.dispatch.sent == [1]
    assert gate.runtime.retries == 1


def test_an_operation_that_is_not_a_decision_is_never_held():
    """The conditional S goes ahead on the provisional answer."""
    gate = gate_over(TWO_T_GATES)

    assert gate.is_open(S_FIRST)
    assert gate.is_open(S_SECOND)


def test_a_decision_with_no_decision_before_it_starts_at_once():
    gate = gate_over(TWO_T_GATES)

    assert gate.is_open(T_FIRST)


def test_the_walk_stops_at_the_nearest_decision():
    """The third T needs the second final, and the second needs the first."""
    t_third = program_records.Operation(
        5, "t third", (0,), patches=(0,), predecessors=(4,)
    )
    s_third = program_records.Operation(
        6, "s third", (0,), patches=(0,), predecessors=(5,), blocked_by=5
    )
    operations = TWO_T_GATES + (t_third, s_third)

    earlier = final_result_gate.earlier_decisions(operations)

    assert earlier == {1: (), 3: (1,), 5: (3,)}


def test_a_result_the_qpu_needs_back_is_a_decision():
    readout = program_records.Operation(
        5,
        "readout",
        (0,),
        patches=(0,),
        predecessors=(4,),
        requires_result_return_to_qpu=True,
    )
    operations = TWO_T_GATES + (readout,)
    gate = gate_over(operations)

    gate.note_result_final(T_FIRST)

    assert not gate.is_open(readout)


def test_a_decision_after_a_merge_waits_for_both_patches():
    """Patch order reaches each patch's last decision before the merge."""
    t_on_zero = program_records.Operation(1, "t a", (0,), patches=(0,))
    s_on_zero = program_records.Operation(
        2, "s a", (0,), patches=(0,), predecessors=(1,), blocked_by=1
    )
    t_on_one = program_records.Operation(3, "t b", (1,), patches=(1,))
    s_on_one = program_records.Operation(
        4, "s b", (1,), patches=(1,), predecessors=(3,), blocked_by=3
    )
    merge = program_records.Operation(
        5, "merge", (0, 1), patches=(0, 1), predecessors=(2, 4)
    )
    t_after = program_records.Operation(
        6, "t after", (0,), patches=(0,), predecessors=(5,)
    )
    s_after = program_records.Operation(
        7, "s after", (0,), patches=(0,), predecessors=(6,), blocked_by=6
    )
    before = (t_on_zero, s_on_zero, t_on_one, s_on_one, merge)
    operations = before + (t_after, s_after)
    gate = gate_over(operations)

    gate.note_result_final(t_on_zero)
    one_final = gate.is_open(t_after)
    gate.note_result_final(t_on_one)

    assert not one_final
    assert gate.is_open(t_after)
