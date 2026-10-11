"""The rules an edge applies to a window's revisions (revision_rules.py).

The referents: latest is today's rule, each revision replacing the one
before; best_status takes the revision whose backend reports the best
status, the latest among equals; cancel_downstream is today's keep,
which cancels the speculative strong decode (Toshio et al. 2510.25222
Sec. III A, Step 3); no_cancel lets one already running end, and drops
its result. On the declared switching machine, a run whose revisions all
succeed answers the same under latest and best_status.
"""

import dataclasses

import decsim.records.decoding as decoding_records
import decsim.windows.revision_rules as revision_rules
import decsim.windows.window_commits as window_commits
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric

STATUS = decoding_records.BackendDecodeStatus


def _revision(status, is_first=False, corrected=True):
    observables = None
    if corrected:
        observables = (0,)
    result = decoding_records.DecodeResult(
        operation_id=1,
        window_id=0,
        logical_observables=observables,
        decode_status=status,
    )
    return window_commits.Revision(None, None, result, None, is_first)


class _Edge:
    """An edge that records what a keep asked of it."""

    def __init__(self) -> None:
        self.asked = []

    def cancel_strong_request(self, window_key: tuple) -> None:
        """Note the cancel."""
        self.asked.append(("cancel", window_key))

    def abandon_strong_request(self, window_key: tuple) -> None:
        """Note the run-out."""
        self.asked.append(("abandon", window_key))


def test_latest_takes_the_latest_revision():
    rule = revision_rules.LatestRevision()
    first = _revision(STATUS.SUCCEEDED, is_first=True)
    latest = _revision(STATUS.NONCONVERGED)

    chosen = rule.final_revision((first,), latest)

    assert chosen is latest


def test_best_status_takes_the_best_and_the_latest_among_equals():
    rule = revision_rules.BestStatus()
    first = _revision(STATUS.SUCCEEDED, is_first=True)
    second = _revision(STATUS.NONCONVERGED)
    latest = _revision(STATUS.SUCCEEDED)
    worse = _revision(STATUS.INVALID_CORRECTION)

    best_of_three = rule.final_revision((first, second), latest)
    best_of_two = rule.final_revision((first,), worse)

    assert best_of_three is latest
    assert best_of_two is first


def test_a_revision_with_no_correction_ranks_below_every_status():
    rule = revision_rules.BestStatus()
    failed = _revision(STATUS.BACKEND_ERROR, is_first=True)
    uncorrected = _revision(STATUS.SUCCEEDED, corrected=False)

    chosen = rule.final_revision((failed,), uncorrected)

    assert chosen is failed


def test_a_keep_cancels_or_lets_run_out_by_the_cancel_rule():
    edge = _Edge()
    cancelling = revision_rules.CancelDownstream()
    running_out = revision_rules.NoCancel()

    cancelling.on_keep(edge, (1, 0))
    running_out.on_keep(edge, (1, 1))

    assert edge.asked == [("cancel", (1, 0)), ("abandon", (1, 1))]


def test_a_best_status_run_whose_revisions_succeed_answers_as_latest():
    best_status = revision_rules.BestStatus.Settings()
    switching = declared_run.declared_switching(replace_rule=best_status)
    latest_run = fabric.switching_machine(rounds=9, escalated_windows={1})
    best_run = fabric.switching_machine(
        rounds=9, escalated_windows={1}, switching=switching
    )

    latest_result = latest_run.run()
    best_result = best_run.run()

    assert fabric.frame_tiers(best_run) == fabric.frame_tiers(latest_run)
    assert best_result.operation_results == latest_result.operation_results


def test_a_no_cancel_run_drops_each_speculative_result_it_let_run_out():
    """Every window is kept; the strong decodes started beside them end.

    The escalation hop is short, so each speculative decode is running
    when its weak result is kept; none is cancelled, each ends and is
    dropped, and the frame carries the weak tier alone.
    """
    no_cancel = revision_rules.NoCancel.Settings()
    switching = declared_run.declared_switching(
        run_both_at_once=True, cancel_rule=no_cancel
    )
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        switching=switching,
        escalation_microseconds=0.001,
    )

    machine.run()

    counts = machine.decoders.decoder_manager.strong_requests.counts
    tiers = fabric.frame_tiers(machine)
    assert [tier for _, tier in tiers] == ["weak", "weak", "weak"]
    assert counts.cancelled == 0
    assert counts.abandoned >= 1


def test_the_default_rules_are_todays():
    rules = revision_rules.DEFAULT_RULES

    replace = type(rules.replace)
    cancel = type(rules.cancel)

    assert replace is revision_rules.LatestRevision
    assert cancel is revision_rules.CancelDownstream
    assert dataclasses.is_dataclass(rules)
