"""A stream lets its final windows go, a few windows behind.

A stream of 10^6 rounds lays some 3 x 10^5 windows, and the machine
keeps something per window: the planner the window and its model, the
logical ledger its contribution, the result cursors their place, the
courier its boundary, the frame its correction, the stores their spent
holds, a strong edge its delivered selections. Once a window is final
only its neighbours read it again: a strong redo pins its face on the
window before it (escalation/strong_regions.py redo_pin_source), and a
window hands its boundary on to the next. So in a stream that folds its
windows (observation.folds_windows) the release lets every window more
than KEPT_FINAL_WINDOWS behind the stream's final prefix go, and a read
of one let go raises (WindowPlanner.window_at). A window is never let
go during its own window_final, so a listener that reads it there, as
the stream's measures do (experiments/stream_measures.py), reads it
whole. The logical ledger and the frame fold what they let go, so the
result and the frame read the same; a stream that lets go measures what
one that keeps everything does (tests/windows/test_window_release.py).

This is decsim's own bookkeeping: no paper describes a decoder system
simulator's memory, and none of a real system's timing depends on it.
"""

from typing import Any

import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.operation_results as operation_results
import decsim.windows.window_planner as window_planner

# the final windows kept behind the final prefix; a strong redo reads
# one window back (redo_pin_source), so eight leave a margin
KEPT_FINAL_WINDOWS = 8
# the windows that become final between two lettings-go, so each walk
# over a keeper's kept records serves that many
LET_GO_EVERY = 64


class WindowRelease:
    """Lets each stream's final windows go, a few behind its final prefix.

    keepers are told in order, each through let_go_before: the result
    cursors first, which read the planner's windows, the planner last.
    """

    def __init__(
        self,
        planner: window_planner.WindowPlanner,
        results: operation_results.OperationResults,
        keepers: tuple[ports.WindowKeeper, ...],
    ) -> None:
        self.planner = planner
        self.results = results
        self.keepers = keepers
        # stream id -> the index of its first window still kept
        self.first_kept_by_stream: dict = {}

    def window_final(self, window: window_records.Window) -> None:
        """A window is final: let its stream's old windows go, if enough."""
        stream_id = window.stream_id
        if not self.planner.has_stream(stream_id):
            return
        final_count = self.results.final_window_count(stream_id)
        before_this = min(final_count, window.window_index)
        first_kept = before_this - KEPT_FINAL_WINDOWS
        kept_from = self.first_kept_by_stream.get(stream_id, 0)
        if first_kept - kept_from < LET_GO_EVERY:
            return
        self.let_go_before(stream_id, first_kept)

    def let_go_before(
        self,
        stream_id: Any,  # an opaque identity
        window_index: int,
    ) -> None:
        """Tell every keeper the stream's windows before the index go."""
        self.results.let_go_before(stream_id, window_index)
        for keeper in self.keepers:
            keeper.let_go_before(stream_id, window_index)
        self.planner.let_go_before(stream_id, window_index)
        self.first_kept_by_stream[stream_id] = window_index
