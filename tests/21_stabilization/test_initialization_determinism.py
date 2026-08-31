"""Setup determinism: one workload, two views, registration before data.

The initialization contract these tests pin is
../validation/responsibility_audit_2026_08_30/initialization_contract.md.
"""

import pytest

from decsim.config import us
from decsim.message import Operation, RetainedSyndromeFragment, SyndromeRoundPacket


def test_execution_and_decoding_views_agree(fabric):
    """The sequencer and the window manager see the same resolved workload."""
    completed = fabric["weak_only_run"](rounds=6)
    runtime = completed.execution_runtime
    window_manager = completed.window_manager

    assert set(runtime.operations) == {1}
    assert set(window_manager._ops) == {1}
    # the decoding view's resolved round count matches the executed body
    assert window_manager.rounds_for(runtime.operations[1]) == 6
    assert window_manager.rounds_arrived[1] == 6


def test_registration_is_idempotent(fabric):
    """Re-registering a known operation never resets its accounts."""
    completed = fabric["weak_only_run"](rounds=6)
    window_manager = completed.window_manager
    operation = window_manager._ops[1]
    arrived_before = window_manager.rounds_arrived[1]
    memory_before = window_manager.memory_rounds[1]

    window_manager.register_op(operation)

    assert window_manager.rounds_arrived[1] == arrived_before
    assert window_manager.memory_rounds[1] == memory_before
    assert window_manager._ops[1] is operation


def test_window_plan_installs_exactly_once(fabric):
    """A second plan load is a no-op: the compile-time windows are fixed."""
    completed = fabric["weak_only_run"](rounds=6)
    window_manager = completed.window_manager
    windows_before = dict(window_manager.windows)

    window_manager.load_execution_plan(None, None)

    assert window_manager._windows_built
    assert window_manager.windows == windows_before


def test_unknown_operation_round_fails(fabric):
    """A syndrome round for an unregistered operation refuses loudly."""
    completed = fabric["weak_only_run"](rounds=6)
    fragment = RetainedSyndromeFragment(
        operation_id=99, patch_id=0, round_index=1, bits=None,
        size_bits=None, fragment_index=0)
    packet = SyndromeRoundPacket(99, 1, (fragment,))

    with pytest.raises(ValueError, match="unknown syndrome operation"):
        completed.window_manager.on_syndrome_arrival(packet)


def test_round_after_operation_close_fails(fabric):
    """A round arriving after the operation's syndrome RAM was freed refuses."""
    completed = fabric["weak_only_run"](rounds=6)
    fragment = RetainedSyndromeFragment(
        operation_id=1, patch_id=1, round_index=2, bits=None,
        size_bits=None, fragment_index=0)
    packet = SyndromeRoundPacket(1, 2, (fragment,))

    with pytest.raises(RuntimeError,
                       match="arrived after the op's last window committed"):
        completed.window_manager.on_syndrome_arrival(packet)


def test_duplicate_operation_ids_are_rejected(fabric):
    """Two workload entries with one id cannot enter a run."""
    ops = [fabric["memory_op"](1), fabric["memory_op"](1)]
    with pytest.raises(ValueError, match="more than once"):
        fabric["weak_only_run"](rounds=6, ops=ops)


def test_scheduled_start_round_delays_the_root(fabric):
    """A scheduled start releases the operation on its exact boundary."""
    op = Operation(id=1, name="late", qubits=(1,), patches=(1,),
                   scheduled_start_round=4)
    completed = fabric["weak_only_run"](rounds=6, ops=[op])
    assert completed.execution_runtime.op_start_time[1] == us(4 * fabric["ROUND_US"])


def test_component_boundaries_are_structural(fabric):
    """The window manager cannot command the QPU; the sequencer cannot
    register windows; the decoder manager cannot reach the QPU."""
    completed = fabric["weak_only_run"](rounds=6)
    assert not hasattr(completed.window_manager, "issue_operation")
    assert not hasattr(completed.execution_runtime, "register_op")
    assert not hasattr(completed.execution_runtime, "create_dynamic_window")
    assert not hasattr(completed.decoder_manager, "qpu")


def test_every_program_operation_is_registered(fabric):
    """The execution-view registration covers non-emitting operations,
    which the decode-plan view never sees; dropping it would leave them
    without readiness accounts."""
    quiet = Operation(id=1, name="quiet", qubits=(1,), patches=(1,),
                      emits_detector_data=False)
    completed = fabric["weak_only_run"](rounds=6,
                                        ops=[quiet, fabric["memory_op"](2)])
    window_manager = completed.window_manager

    assert {1, 2} <= set(window_manager._ops)
    assert 1 in window_manager.memory_rounds
    assert set(completed.execution_runtime.body_done_time) == {1, 2}
