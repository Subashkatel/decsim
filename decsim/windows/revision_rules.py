"""The rules an escalate edge applies to a window's revisions.

Every decode of a window is a revision (window_commits.Revision). The
replace rule says which revision the frame takes when the window is
final, and whether a later revision replaces the prediction before
then; the cancel rule says what a keep does to the edge's target work
for the window. An edge's switching settings name one of each
(SwitchingSettings.replace_rule, cancel_rule).

    latest            the frame takes the latest revision, which
                      replaces each earlier one as it lands; the default
    best_status       the frame takes the revision whose backend status
                      is best, the latest among equals
    cancel_downstream a keep cancels the target's decode of the window,
                      wherever it is; the default (Toshio et al.
                      2510.25222 Sec. III A, Step 3)
    no_cancel         a keep lets a target decode already asked for run
                      to its end, and its result is dropped, as a
                      kernel already launched on an accelerator runs out
"""

import dataclasses

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.windows.window_commits as window_commits

# the backend's statuses, best first, and a revision with no correction
# below them all
_STATUS_ORDER = tuple(decoding_records.BackendDecodeStatus)
_NO_CORRECTION_RANK = len(_STATUS_ORDER)


class LatestRevision:
    """The frame takes the latest revision; each replaces the one before."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The latest row, which takes no setting."""

        name = "latest"
        replaces_provisionally = True

        def build(self) -> "LatestRevision":
            """A rule that keeps the latest revision."""
            return LatestRevision()

    def final_revision(
        self, earlier: tuple, latest: window_commits.Revision
    ) -> window_commits.Revision:
        """The latest revision."""
        del earlier
        return latest

    def on_provisional(
        self,
        committer: window_commits.WindowCommitter,
        revision: window_commits.Revision,
    ) -> None:
        """The revision replaces the prediction until the next lands."""
        committer.replace_provisionally(revision.window, revision.result)


class BestStatus:
    """The frame takes the revision whose backend status is best.

    A succeeded result ranks first, then the statuses in
    BackendDecodeStatus order, then a result with no correction; among
    equals, the latest. A later revision replaces nothing until the
    window is final.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The best-status row, which takes no setting."""

        name = "best_status"
        replaces_provisionally = False

        def build(self) -> "BestStatus":
            """A rule that keeps the best-status revision."""
            return BestStatus()

    def final_revision(
        self, earlier: tuple, latest: window_commits.Revision
    ) -> window_commits.Revision:
        """The best revision of the window, the latest among equals."""
        best = latest
        best_rank = status_rank(latest.result)
        for revision in reversed(earlier):
            rank = status_rank(revision.result)
            if rank < best_rank:
                best = revision
                best_rank = rank
        return best

    def on_provisional(
        self,
        committer: window_commits.WindowCommitter,
        revision: window_commits.Revision,
    ) -> None:
        """The prediction stays until the window is final."""
        del committer, revision


class CancelDownstream:
    """A keep cancels the target's decode of the window, wherever it is."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The cancelling row, which takes no setting."""

        name = "cancel_downstream"
        cancels_on_keep = True

        def build(self) -> "CancelDownstream":
            """A rule that cancels on a keep."""
            return CancelDownstream()

    def on_keep(self, edge: ports.EscalateEdge, window_key: tuple) -> None:
        """Cancel the edge's target decode of the window."""
        edge.cancel_strong_request(window_key)


class NoCancel:
    """A keep lets the target's decode run out, then drops its result.

    A decode the target has not been asked for yet, one held for its
    input, is dropped: it was never sent.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row that never cancels, which takes no setting."""

        name = "no_cancel"
        cancels_on_keep = False

        def build(self) -> "NoCancel":
            """A rule that lets a target decode run out on a keep."""
            return NoCancel()

    def on_keep(self, edge: ports.EscalateEdge, window_key: tuple) -> None:
        """Let the edge's target decode of the window run to its end."""
        edge.abandon_strong_request(window_key)


@dataclasses.dataclass(frozen=True)
class EdgeRules:
    """An edge's replace rule and cancel rule, built."""

    replace: ports.ReplaceRule
    cancel: ports.CancelRule


# the rules of an edge whose settings name none: today's
LATEST = LatestRevision()
CANCEL = CancelDownstream()
DEFAULT_RULES = EdgeRules(LATEST, CANCEL)


def status_rank(result: decoding_records.DecodeResult) -> int:
    """A result's place in BestStatus's order, 0 the best."""
    if result.logical_observables is None:
        return _NO_CORRECTION_RANK
    status = result.decode_status
    if status is None:
        return 0
    return _STATUS_ORDER.index(status)
