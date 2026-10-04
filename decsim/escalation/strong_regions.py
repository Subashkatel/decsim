"""Where a strong window sits: its rounds, its faults, its neighbours.

One owner for the region arithmetic of the strong window shapes. A
region is asked for by window key and comes back resolved against the
live window graph: the rounds the strong decoder commits and reads, the
windows it absorbs, the window that restarts the weak chain after it,
and the error model of each. A later window that would commit across the
region's edge stops the run, since nothing would own its faults.

Toshio et al. 2510.25222 Sec. III C and Fig. 12 give the forward region,
r_strong = r_com + 2 r_buf with the weak chain resuming after it (lines
1232-1235, 1248-1251); the redo region is the escalated window's commit
region with its past face pinned (Bombin et al. 2303.04846 lines
775-788, 1456-1458).
"""

import copy
import dataclasses
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.program as program_records
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class RedoRegion:
    """One strong redo of one window: the window, and the rounds it reads."""

    window: window_records.Window
    context_read_keys: tuple


@dataclasses.dataclass(frozen=True)
class DoubleWindowRegion:
    """One double-window region, resolved against the live window graph."""

    plan: window_records.StrongRegionPlan
    absorbed_window_keys: tuple
    restart_window_key: Optional[tuple]
    restart_read_keys: tuple
    context_round_keys: tuple
    strong_window: window_records.Window
    strong_model: object
    proposed_restart_window: Optional[window_records.Window]
    restart_model: object


class StrongRegions:
    """The strong region of a window, planned, checked and modelled."""

    planner = ports.Port(ports.WindowPlan)
    tracker = ports.Port(ports.WindowRounds)
    retention = ports.Port(ports.WindowRetention)
    # the one call this package makes on the window interaction; the rest
    # of that class is the windows package's own seam
    interaction = ports.Port(ports.RegionProposer)

    def redo_region(self, key: tuple) -> RedoRegion:
        """The escalated window's commit rounds, its past face pinned.

        The rounds before the commit region arrive as the neighbour's
        committed boundary instead (Bombin et al. 2303.04846 lines
        1456-1458); one buffer region of raw context stays on the open
        future face. A strong side that forms the events also reads the
        raw rounds its recipes need before the commit
        (strong_rounds_before).
        """
        weak_window = self.planner.window_at(key)
        strong_window = _near_pinned_window_of(weak_window)
        read_keys = self.retention.strong_rounds_before(
            key[0], strong_window.buffer_lo, strong_window.buffer_hi
        )
        read_keys += self.retention.read_keys_for_bounds(
            key[0],
            strong_window.buffer_lo,
            strong_window.buffer_hi,
            strong_window,
        )
        return RedoRegion(strong_window, tuple(read_keys))

    def redo_pin_source(self, key: tuple) -> Optional[tuple]:
        """The window whose commit region ends where this one's begins.

        The source is the escalated window's own dependency that commits
        the round before it (Bombin et al. 2303.04846 lines 703-704).
        None for a window with no earlier neighbour, whose past face the
        initialisation closes.
        """
        weak_window = self.planner.window_at(key)
        seam_round = weak_window.commit_lo - 1
        for dependency_key in weak_window.deps:
            neighbour = self.planner.window_at(dependency_key)
            if neighbour.commit_hi == seam_round:
                return dependency_key
        return None

    def double_window_near_face(self, key: tuple) -> Optional[tuple]:
        """The commit a double-window region's near face pins on.

        The window before the region, when one committed the round
        before it. Otherwise, past the operation's first round, the
        escalated window restarted the weak chain after an earlier
        region, and its own weak commit holds the faults crossing that
        seam (Toshio et al. 2510.25222 lines 1248-1250). None at round
        one, whose face the initialisation closes.
        """
        neighbour = self.redo_pin_source(key)
        if neighbour is not None:
            return neighbour
        weak_window = self.planner.window_at(key)
        if weak_window.commit_lo == 1:
            return None
        return key

    def redecode_model(
        self,
        key: tuple,
        window: window_records.Window,
        pinned_source_keys: tuple,
    ) -> object:
        """The error model of one strong redo of a window.

        The redo owns the faults of its own rounds and none before its
        commit region. A face read raw keeps those faults as columns, so
        the decoder can still explain a defect with one; a pinned face
        does not, because the neighbour's answer is already in the input
        (Bombin et al. 2303.04846 lines 775-788).
        """
        round_count = self.round_count_for(key[0], window)
        exclusions = _left_fault_exclusions(window.commit_lo)
        operation = self.tracker.operation(key[0])
        prior_faults = self._faults_the_pins_carry(pinned_source_keys)
        return self.planner.strong_window_model(
            operation, window, round_count, exclusions, prior_faults
        )

    def _faults_the_pins_carry(self, pinned_source_keys: tuple):
        """What the windows behind the pinned faces have committed, or None."""
        owned_sets = []
        for source_key in pinned_source_keys:
            owned = self.planner.owned_faults_of(source_key)
            if owned is not None:
                owned_sets.append(owned)
        return _union_of_owned_faults(owned_sets)

    def double_window_region(
        self, key: tuple, *, near_source_key: Optional[tuple]
    ) -> DoubleWindowRegion:
        """The double-window region, read with no context past its commit.

        Fig. 12 assigns the strong decoder r_strong rounds "after the
        boundary conditions at both ends have been determined by the
        weak decoder" (Toshio et al. 2510.25222 lines 1248-1250), and a
        determined boundary needs no buffer (Bombin et al. 2303.04846
        lines 1456-1458). The near face is open only at the operation's
        first round, with one buffer region of raw context (Bombin lines
        850-852). The restart window owns the faults crossing the far
        face at every re-read width, so the far pin carries only those
        (Fig. 12 step 5).
        """
        proposal = self._proposed_double_window_region(key)
        plan = proposal.plan
        context_lo = plan.context_lo
        if near_source_key is not None:
            context_lo = plan.commit_lo
        pinned_plan = dataclasses.replace(
            plan, context_lo=context_lo, context_hi=plan.commit_hi
        )
        pinned = dataclasses.replace(proposal, plan=pinned_plan)
        return self._resolved_double_window_region(key, pinned, near_source_key)

    def _proposed_double_window_region(
        self, key: tuple
    ) -> "_DoubleWindowProposal":
        """The extent the interaction proposes, with what it was read on."""
        operation_id, escalated_index = key
        weak_window = self.planner.window_at(key)
        round_count = self.round_count_for(operation_id, weak_window)
        later_windows = self.planner.later_windows(
            operation_id, escalated_index
        )
        plan = self._plan_region(weak_window, later_windows, round_count)
        return _DoubleWindowProposal(
            round_count=round_count,
            later_windows=later_windows,
            plan=plan,
        )

    def _resolved_double_window_region(
        self,
        key: tuple,
        proposal: "_DoubleWindowProposal",
        near_source_key: Optional[tuple],
    ) -> DoubleWindowRegion:
        """The proposed extent checked against the windows that exist."""
        operation_id = key[0]
        round_count = proposal.round_count
        later_windows = proposal.later_windows
        plan = proposal.plan
        absorbed = _absorbed_window_keys(later_windows, plan)
        _refuse_crossing_window(later_windows, plan)
        self.planner.check_absorbable(absorbed)
        restart_key = _restart_window_key(later_windows, plan)
        restart_reads = self._restart_reads(restart_key, plan)
        context_keys = self.retention.strong_rounds_before(
            operation_id, plan.context_lo, plan.context_hi
        )
        context_keys += _context_round_keys(operation_id, plan)
        self._require_reads_retained(key, context_keys, restart_reads)
        return self._modelled_region(
            key,
            round_count,
            plan,
            absorbed,
            restart_key,
            restart_reads,
            context_keys,
            near_source_key,
        )

    def operation(
        self,
        operation_id: Any,  # an opaque identity
    ) -> program_records.Operation:
        """The operation record a strong window's transfers are named by."""
        return self.tracker.operation(operation_id)

    def round_count_for(
        self,
        operation_id: Any,  # an opaque identity
        window: window_records.Window,
    ) -> int:
        """The rounds the operation runs, as this window is planned."""
        return self.tracker.round_count_for_window(operation_id, window)

    def strong_rounds_stored(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """How far the strong syndrome buffer is filled for the operation."""
        return self.tracker.strong_rounds_arrived(operation_id)

    def _plan_region(
        self,
        weak_window: window_records.Window,
        later_windows: list,
        round_count: int,
    ) -> window_records.StrongRegionPlan:
        weak_info = window_records.WindowInfo.from_window(weak_window)
        later_infos = []
        for window in later_windows:
            window_info = window_records.WindowInfo.from_window(window)
            later_infos.append(window_info)
        return self.interaction.plan_strong_region(
            weak_info, later_infos, round_count
        )

    def _restart_reads(
        self,
        restart_key: Optional[tuple],
        plan: window_records.StrongRegionPlan,
    ) -> tuple:
        """The restart window's reads from its re-sliced buffer start."""
        if restart_key is None:
            return ()
        restart = self.planner.window_at(restart_key)
        reads = self.retention.read_keys_for_bounds(
            restart.operation_id,
            plan.restart_buffer_lo,
            restart.buffer_hi,
            restart,
        )
        return tuple(reads)

    def _require_reads_retained(
        self, key: tuple, context_keys: list, restart_reads: tuple
    ) -> None:
        """Both sides' rounds must still be stored before the plan lands.

        The strong window's context is in the strong syndrome buffer;
        the restart window's reads are in the weak syndrome buffer under
        its potential restart hold.
        """
        purpose = f"strong-region plan for {key}"
        self.retention.require_strong_retained(context_keys, purpose)
        self.retention.require_retained(restart_reads, purpose)

    def _modelled_region(
        self,
        key: tuple,
        round_count: int,
        plan: window_records.StrongRegionPlan,
        absorbed: tuple,
        restart_key: Optional[tuple],
        restart_reads: tuple,
        context_keys: list,
        near_source_key: Optional[tuple],
    ) -> DoubleWindowRegion:
        """The resolved region with the two windows' error models.

        The restart window is modelled first: a row that pins its far
        face on that window's commit takes the faults it owns as prior
        faults.
        """
        operation = self.tracker.operation(key[0])
        strong_exclusions, restart_exclusions = _fault_exclusions(
            plan, round_count, restart_key
        )
        strong_window = _strong_window_of(key, plan)
        proposed_restart = None
        restart_model = None
        if restart_key is not None:
            proposed_restart = self._proposed_restart_window(restart_key, plan)
            restart_model = self.planner.strong_window_model(
                operation,
                proposed_restart,
                round_count,
                restart_exclusions,
                None,
            )
        prior_faults = self._double_window_prior_faults(
            key, near_source_key, restart_model
        )
        strong_model = self.planner.strong_window_model(
            operation,
            strong_window,
            round_count,
            strong_exclusions,
            prior_faults,
        )
        return DoubleWindowRegion(
            plan=plan,
            absorbed_window_keys=absorbed,
            restart_window_key=restart_key,
            restart_read_keys=restart_reads,
            context_round_keys=tuple(context_keys),
            strong_window=strong_window,
            strong_model=strong_model,
            proposed_restart_window=proposed_restart,
            restart_model=restart_model,
        )

    def _double_window_prior_faults(
        self,
        key: tuple,
        near_source_key: Optional[tuple],
        restart_model,
    ):
        """What the faces of a double-window region's pins carry, or None."""
        owned_sets = []
        if near_source_key is not None:
            near_owned = self._near_face_faults(key, near_source_key)
            if near_owned is not None:
                owned_sets.append(near_owned)
        if restart_model is not None:
            restart_owned = restart_model.owned_fault_ids()
            owned_sets.append(restart_owned)
        return _union_of_owned_faults(owned_sets)

    def _near_face_faults(self, key: tuple, near_source_key: tuple):
        """The faults the near face's commit takes out of the columns.

        A region pinned on the escalated window's own weak commit takes
        only the faults crossing behind that window's first round, since
        the rest of that commit is what the region decodes again.
        """
        if near_source_key != key:
            return self.planner.owned_faults_of(near_source_key)
        return self.planner.crossing_faults_of(key)

    def _proposed_restart_window(
        self, restart_key: tuple, plan: window_records.StrongRegionPlan
    ) -> window_records.Window:
        """The restart window as it will read after the re-slice."""
        restart = self.planner.window_at(restart_key)
        proposed = copy.deepcopy(restart)
        proposed.buffer_lo = plan.restart_buffer_lo
        return proposed


@dataclasses.dataclass(frozen=True)
class _DoubleWindowProposal:
    """The extent an interaction proposes, before the row narrows it."""

    round_count: int
    later_windows: list
    plan: window_records.StrongRegionPlan


def _union_of_owned_faults(owned_sets: list):
    """One prior-fault map from several windows' owned sets, or None."""
    if not owned_sets:
        return None
    union: dict = {}
    empty: frozenset = frozenset()
    for owned in owned_sets:
        for representation, fault_ids in owned.items():
            already = union.get(representation, empty)
            union[representation] = already | fault_ids
    return union


def _near_pinned_window_of(
    weak_window: window_records.Window,
) -> window_records.Window:
    """The weak window's commit rounds plus one trailing buffer region."""
    bounds = window_records.strong_context_bounds(weak_window)
    return _window_over(
        weak_window.operation_id, weak_window.window_index, bounds
    )


def _left_fault_exclusions(commit_lo: int) -> tuple:
    """The rounds before the strong window, whose faults it never owns."""
    if commit_lo > 1:
        return ((1, commit_lo - 1),)
    return ()


def _context_round_keys(
    operation_id, plan: window_records.StrongRegionPlan
) -> list:
    """The (operation, round) keys of the strong window's whole context."""
    stop_round = plan.context_hi + 1
    round_keys = []
    for round_index in range(plan.context_lo, stop_round):
        round_keys.append((operation_id, round_index))
    return round_keys


def _strong_window_of(
    key: tuple, plan: window_records.StrongRegionPlan
) -> window_records.Window:
    bounds = (plan.context_lo, plan.commit_lo, plan.commit_hi, plan.context_hi)
    return _window_over(key[0], key[1], bounds)


def _window_over(
    operation_id, window_index: int, bounds: tuple
) -> window_records.Window:
    """The strong window over (context_lo, commit_lo, commit_hi, context_hi)."""
    context_lo, commit_lo, commit_hi, context_hi = bounds
    round_count = context_hi - context_lo + 1
    return window_records.Window(
        operation_id=operation_id,
        window_index=window_index,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=context_hi,
        buffer_lo=context_lo,
        round_count=round_count,
    )


def _absorbed_window_keys(
    later_windows: list, plan: window_records.StrongRegionPlan
) -> tuple:
    """The later windows whose whole commit region the strong window covers."""
    absorbed = []
    for window in later_windows:
        if window.commit_hi <= plan.commit_hi:
            absorbed.append(window.key)
    return tuple(absorbed)


def _refuse_crossing_window(
    later_windows: list, plan: window_records.StrongRegionPlan
) -> None:
    """A window committing across the strong window's edge has no owner."""
    for window in later_windows:
        is_begun = window.commit_lo <= plan.commit_hi
        is_unfinished = plan.commit_hi < window.commit_hi
        if is_begun and is_unfinished:
            raise RuntimeError(
                f"window {window.key} commits {window.commit_lo}-"
                f"{window.commit_hi} across the strong-region edge "
                f"{plan.commit_hi}"
            )


def _restart_window_key(
    later_windows: list, plan: window_records.StrongRegionPlan
) -> Optional[tuple]:
    """The first later window past the strong window, or None at the end."""
    for window in later_windows:
        if window.commit_lo > plan.commit_hi:
            return window.key
    return None


def _fault_exclusions(
    plan: window_records.StrongRegionPlan,
    round_count: int,
    restart_key: Optional[tuple],
) -> tuple:
    """(strong window's, restart window's) fault exclusion ranges.

    The restart window owns the crossing faults at every re-read width,
    so the strong window excludes the rounds past its edge and the
    restart window only the rounds before the strong window. The window
    decoded first commits the edges leaving its commit region, and the
    window filled in between has closed faces and owns nothing across
    them (Skoric et al. 2209.08552 lines 258-261 and 380-388).
    """
    left_exclusions = _left_fault_exclusions(plan.commit_lo)
    if restart_key is None:
        return left_exclusions, None
    right_exclusion = (plan.commit_hi + 1, round_count)
    strong_exclusions = left_exclusions + (right_exclusion,)
    return strong_exclusions, left_exclusions
