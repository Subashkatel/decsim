"""The referee's audit: what it re-decoded and where it disagreed.

A listener on the Decoder port's window_checked source. The audit is the
run's accuracy check and belongs to whoever watches the run, so the
decoder keeps none of it.
"""

import decsim.observe.referee_audit as referee_audit


def test_a_run_with_no_referee_checked_nothing():
    audit = referee_audit.RefereeAudit()

    assert audit.windows_checked == 0
    assert audit.window_disagreements == 0
    assert audit.disagreeing_windows == []


def test_every_check_is_counted_and_an_agreement_names_no_window():
    audit = referee_audit.RefereeAudit()

    audit.window_checked((1, 0), True)
    audit.window_checked((1, 1), True)

    assert audit.windows_checked == 2
    assert audit.disagreeing_windows == []


def test_a_disagreement_names_the_window_it_was_found_on():
    audit = referee_audit.RefereeAudit()

    audit.window_checked((1, 0), True)
    audit.window_checked((1, 1), False)
    audit.window_checked((2, 0), False)

    assert audit.windows_checked == 3
    assert audit.window_disagreements == 2
    assert audit.disagreeing_windows == [(1, 1), (2, 0)]
