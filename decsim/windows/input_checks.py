"""The input checks: a decode on a provisional boundary waits for the final one.

Under the checked boundary row (boundary_policies.Checked) a window
ships its boundary at every commit, and the final one again once its
result is final. A decode that read a boundary not final yet decodes on
a guess, as SWIPER's windows decode on a predicted boundary (Vittal et
al. 2412.05115 lines 625-644), with the weak decoder's own boundary as
the prediction. The window commits provisionally and waits. When the
final boundary lands, the state it leaves in the window is compared with
the state the decode read. The same: the result stands, and the step the
window waits with runs, its final commit or its escalation. Different:
the window is decoded again on the state it holds now; its new boundary
is checked in turn by the window after it, so only a window whose input
changed is decoded again, SWIPER's optimistic restart (lines 643-644). A
decode still running when its input changes runs to its end, and its
result is dropped then.

A decode that read only final boundaries is recorded nowhere, so under
the held and eager rows, which never ship a boundary they check again,
every input is final and nothing here acts.
"""

import dataclasses
import enum
from collections.abc import Callable

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry
import decsim.windows.window_interactions as window_interactions


class InputState(enum.Enum):
    """What a finished decode's input was when its result came."""

    # every boundary it read was final
    FINAL = "final"
    # a boundary it read is not final yet, and has not changed
    PROVISIONAL = "provisional"
    # a boundary it read changed since: the result is dropped
    STALE = "stale"


class InputChecks:
    """Which window decodes read a provisional boundary, checked as each lands.

    The courier tells it every current landing (boundary_landed), the
    gate every decode's start (note_read), and the verdict asks it of
    every result of a window's entry pool (take_result), then gives it
    the steps a provisional one waits with (wait_for_final_input).
    """

    # whether a landing changed what a decode read
    interaction = ports.Port(window_interactions.WindowInteraction)

    STATISTICS = (
        statistics_registry.Statistic(
            "provisional_reads",
            statistic_kinds.Count,
            "window decodes that read a boundary not final yet",
            "decodes",
        ),
        statistics_registry.Statistic(
            "confirmed",
            statistic_kinds.Count,
            "provisional reads whose final boundary was the same",
            "decodes",
        ),
        statistics_registry.Statistic(
            "decoded_again",
            statistic_kinds.Count,
            "provisional reads whose boundary changed: decoded again",
            "decodes",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    def __init__(self) -> None:
        # window key -> the sources whose latest landed boundary is not final
        self.provisional_sources_by_window: dict = {}
        # window key -> the latest decode's read, while it waits
        self.read_by_window: dict = {}
        # the windows whose running decode read a boundary that changed
        self.stale_windows: set = set()
        # window key -> the steps its committed result waits with
        self.waiting_by_window: dict = {}

    # ---- what the courier and the gate tell

    def boundary_landed(
        self,
        window: window_records.Window,
        source_key: tuple,
        is_final: bool,
    ) -> None:
        """A current boundary landed: note it, and check a decode that read.

        A boundary that changed what the decode read makes it stale; a
        final one that did not confirms its source.
        """
        self._note_landing(window.key, source_key, is_final)
        read = self.read_by_window.get(window.key)
        if read is None:
            return
        if not self.interaction.same_boundary(read.state, window.boundary_in):
            self._make_stale(window.key)
            return
        if not is_final or source_key not in read.unconfirmed:
            return
        read.unconfirmed.discard(source_key)
        if read.unconfirmed:
            return
        self._confirm(window.key)

    def note_read(self, job: decoding_records.DecodeJob) -> None:
        """A decode of a window's entry pool starts: keep what it reads.

        Only a read with a source not final is kept; a strong job reads
        a strong window, whose faces are pinned on final boundaries. The
        two forced-class solves of one attempt are one read: had a
        boundary changed between their starts, the first one's read
        would be stale and gone already.
        """
        if job.kind is not decoding_records.DecodeJobKind.WINDOW:
            return
        window = job.window
        provisional = self.provisional_sources_by_window.get(window.key)
        if not provisional:
            return
        if window.key in self.read_by_window:
            return
        self.statistics.provisional_reads.add()
        read = _Read(window.boundary_in, set(provisional))
        self.read_by_window[window.key] = read

    # ---- what the verdict asks

    def take_result(self, job: decoding_records.DecodeJob) -> InputState:
        """The input of a finished decode of a window's entry pool.

        A result on a final input ends the window's read.
        """
        key = (job.stream_id, job.window_id)
        if key in self.stale_windows:
            self.stale_windows.remove(key)
            # a read the attempt's other solve made after the change
            self.read_by_window.pop(key, None)
            return InputState.STALE
        read = self.read_by_window.get(key)
        if read is None:
            return InputState.FINAL
        if read.unconfirmed:
            return InputState.PROVISIONAL
        del self.read_by_window[key]
        return InputState.FINAL

    def wait_for_final_input(
        self,
        window_key: tuple,
        on_final: Callable[[], None],
        on_stale: Callable[[], None],
    ) -> None:
        """A result on a provisional input waits: on_final, or on_stale."""
        self.waiting_by_window[window_key] = _Waiting(on_final, on_stale)

    # ---- private

    def _note_landing(
        self, window_key: tuple, source_key: tuple, is_final: bool
    ) -> None:
        """The window's sources whose latest boundary is not final."""
        provisional = self.provisional_sources_by_window.pop(window_key, set())
        provisional.discard(source_key)
        if not is_final:
            provisional.add(source_key)
        if provisional:
            self.provisional_sources_by_window[window_key] = provisional

    def _make_stale(self, window_key: tuple) -> None:
        """The read is wrong: decode again now, or when the result comes."""
        del self.read_by_window[window_key]
        self.statistics.decoded_again.add()
        waiting = self.waiting_by_window.pop(window_key, None)
        if waiting is None:
            self.stale_windows.add(window_key)
            return
        waiting.on_stale()

    def _confirm(self, window_key: tuple) -> None:
        """Every source the read was not sure of is final and the same.

        A decode still running keeps its read, emptied, so its result
        finds the input final (take_result).
        """
        self.statistics.confirmed.add()
        waiting = self.waiting_by_window.pop(window_key, None)
        if waiting is None:
            return
        del self.read_by_window[window_key]
        waiting.on_final()


@dataclasses.dataclass
class _Read:
    """A decode's input: the window's boundary state, the sources not final.

    The state is the window's own record, which a landing replaces and
    never changes in place: the courier merges into a copy
    (window_boundaries.py _merged_update).
    """

    state: object
    unconfirmed: set


@dataclasses.dataclass(frozen=True)
class _Waiting:
    """The steps a committed result on a provisional input waits with."""

    on_final: Callable[[], None]
    on_stale: Callable[[], None]
