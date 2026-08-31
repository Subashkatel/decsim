"""Far-boundary escalation in a backlog regime refuses with contract
messages, never a bare KeyError (stabilization finding R3).

The guard in defer_strong_escalation reads the restart window's Buffer 0
hold. Once the restart window's weak decode is built, that hold is
transferred to the job (DecoderInputHold), so the window-key lookup can
miss while every round is still retained. In the shipped fabric the
sibling retained-payload check fires first (the absorbed windows'
withdrawal releases the seam rounds moments earlier); the narrower
regime where that check passes is reached here by letting it record
instead of raise, which leaves the run's real store state untouched.
"""

import pytest

from decsim.windows.window_manager import WindowManager

def escalate_only(window_id):
    def probability(job):
        return 1.0 if job.window_id == window_id else 0.0
    return probability


def backlog_far_boundary_run(fabric):
    """rounds=15 on the declared fabric escalating window 1: by the
    escalation tick the weak chain has consumed the strong region."""
    return fabric["switching_run"](
        rounds=15, escalation_probability=0.0,
        probability_for=escalate_only(1),
        double_window=True, seed=0)


def test_backlog_far_boundary_refuses_with_retained_payload_message(fabric):
    with pytest.raises(RuntimeError, match="requires retained payload "
                                           "rounds that are no longer "
                                           "available"):
        backlog_far_boundary_run(fabric)


def test_missing_restart_hold_refuses_with_contract_message(fabric, monkeypatch):
    """The narrower regime: every restart read still retained, but the
    restart window's own hold already transferred to its weak job."""
    original = WindowManager._require_retained_payloads

    def record_instead_of_raise_for_restart(self, round_keys, purpose,
                                            store=None):
        restart_check_on_buffer0 = (
            purpose.startswith("strong-region plan") and store is None)
        if restart_check_on_buffer0:
            return None
        return original(self, round_keys, purpose, store=store)

    monkeypatch.setattr(WindowManager, "_require_retained_payloads",
                        record_instead_of_raise_for_restart)
    with pytest.raises(RuntimeError, match="restart window .* weak reads "
                                           "under its window hold"):
        backlog_far_boundary_run(fabric)
