"""Setup determinism: one workload, two views, registration before data.

The initialization contract these tests pin is
../validation/responsibility_audit_2026_08_30/initialization_contract.md.
"""

import pytest

import decsim.records.program as program_records
import decsim.records.rounds as round_records
from decsim.config import microseconds_to_ticks


def test_execution_and_decoding_views_agree(fabric):
    """The sequencer and the window manager see the same resolved workload."""
    completed = fabric["weak_only_run"](rounds=6)
    runtime = completed.execution_runtime
    window_manager = completed.window_manager

    assert set(runtime.operations) == {1}
    assert set(window_manager.tracker.operation_by_id) == {1}
    # the decoding view's resolved round count matches the executed body
    assert window_manager.planner.round_count_of(1) == 6
    assert window_manager.tracker.rounds_arrived(1) == 6


def test_registration_is_idempotent(fabric):
    """Re-registering a known operation never resets its accounts."""
    completed = fabric["weak_only_run"](rounds=6)
    window_manager = completed.window_manager
    operation = window_manager.tracker.operation_by_id[1]
    arrived_before = window_manager.tracker.rounds_arrived(1)
    memory_before = window_manager.tracker.memory_rounds(1)

    window_manager.register_operation(operation)

    assert window_manager.tracker.rounds_arrived(1) == arrived_before
    assert window_manager.tracker.memory_rounds(1) == memory_before
    assert window_manager.tracker.operation_by_id[1] is operation


def test_round_after_operation_close_fails(fabric):
    """A round arriving after the operation's syndrome RAM was freed refuses."""
    completed = fabric["weak_only_run"](rounds=6)
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=1,
        round_index=2,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    packet = round_records.SyndromeRoundPacket(1, 2, (fragment,))

    with pytest.raises(
        RuntimeError, match="arrived after the op's last window committed"
    ):
        completed.window_manager.accept_window_input(packet)


def test_duplicate_operation_ids_are_rejected(fabric):
    """Two workload entries with one id cannot enter a run."""
    ops = [fabric["memory_op"](1), fabric["memory_op"](1)]
    with pytest.raises(ValueError, match="more than once"):
        fabric["weak_only_run"](rounds=6, ops=ops)


def test_scheduled_start_round_delays_the_root(fabric):
    """A scheduled start releases the operation on its exact boundary."""
    op = program_records.Operation(
        id=1, name="late", qubits=(1,), patches=(1,), scheduled_start_round=4
    )
    completed = fabric["weak_only_run"](rounds=6, ops=[op])
    assert completed.observation.runtime_stamps.op_start[
        1
    ] == microseconds_to_ticks(4 * fabric["ROUND_US"])


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
    quiet = program_records.Operation(
        id=1, name="quiet", qubits=(1,), patches=(1,), emits_detector_data=False
    )
    completed = fabric["weak_only_run"](
        rounds=6, ops=[quiet, fabric["memory_op"](2)]
    )
    window_manager = completed.window_manager

    assert {1, 2} <= set(window_manager.tracker.operation_by_id)
    assert 1 in window_manager.tracker.arrivals_by_operation
    assert set(completed.observation.runtime_stamps.body_done) == {1, 2}
