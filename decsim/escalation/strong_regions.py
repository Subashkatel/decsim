"""Where a strong window sits: its rounds, its faults, its neighbours.

One owner for the region arithmetic of both strong window shapes
(strong_window_shapes.py). A region is asked for by window key and comes
back resolved against the live window graph: the rounds the strong
decoder commits and reads, the windows its extent absorbs, the window
that restarts the weak chain after it, the faults each side owns, and
the error model of each. The interaction that proposes the region is a
plug-in (windows/window_interactions.py, plan_strong_region), so its
plan is checked here like an input: bounds nest, no later window commits
across the region's edge, committed regions tile without a gap.

Toshio et al. 2510.25222 Sec. III C and Fig. 12 give the forward
region's rule, r_strong = r_com + 2 r_buf with the weak chain resuming
after it (lines 1232-1235, 1248-1251); the two-sided context region is
decsim's own (see ContextWindow).
"""

import copy
import dataclasses
from typing import Optional

import decsim.ports as ports
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class ContextRegion:
    """The two-sided context region: the window to decode and its model."""

    window: window_records.Window
    context_read_keys: tuple


@dataclasses.dataclass(frozen=True)
class ForwardRegion:
    """One forward strong region, resolved against the live window graph."""

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

    def __init__(
        self,
        planner: ports.WindowPlan,
        tracker,
        retention: ports.WindowRetention,
        interaction,
    ) -> None:
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.interaction = interaction

    def context_region(self, key: tuple) -> ContextRegion:
        """The escalated window with one buffer of raw context per side."""
        weak_window = self.planner.window_at(key)
        strong_window = _context_window_of(weak_window)
        read_keys = self.retention.read_keys_for_bounds(
            key[0],
            strong_window.buffer_lo,
            strong_window.buffer_hi,
            strong_window,
        )
        return ContextRegion(strong_window, tuple(read_keys))

    def context_model(self, key: tuple, window: window_records.Window):
        """The error model of a two-sided context region."""
        round_count = self.round_count_for(key[0], window)
        exclusions = _left_fault_exclusions(window.commit_lo)
        operation = self.tracker.operation(key[0])
        return self.planner.strong_window_model(
            operation, window, round_count, exclusions
        )

    def forward_region(self, key: tuple) -> ForwardRegion:
        """The forward strong region of an escalated window, resolved.

        The interaction proposes the extent, this checks it against the
        windows that exist, and the rounds each side reads are required
        to be retained before anything moves.
        """
        operation_id, escalated_index = key
        weak_window = self.planner.window_at(key)
        round_count = self.round_count_for(operation_id, weak_window)
        later_windows = self.planner.later_windows(
            operation_id, escalated_index
        )
        plan = self._plan_region(weak_window, later_windows, round_count)
        _check_region_bounds(key, weak_window, round_count, plan)
        absorbed = _absorbed_window_keys(later_windows, plan)
        _refuse_crossing_window(later_windows, plan)
        self.planner.check_absorbable(absorbed)
        restart_key = _restart_window_key(later_windows, plan)
        restart_reads = self._restart_reads(key, restart_key, plan)
        context_keys = _context_round_keys(operation_id, plan)
        self._require_reads_retained(key, context_keys, restart_reads)
        return self._modelled_region(
            key,
            round_count,
            plan,
            absorbed,
            restart_key,
            restart_reads,
            context_keys,
        )

    def round_count_for(
        self, operation_id, window: window_records.Window
    ) -> int:
        """The rounds the operation runs, as this window is planned."""
        return self.tracker.round_count_for_window(operation_id, window)

    def strong_rounds_stored(self, operation_id) -> int:
        """How far the room-side store has been filled for the operation."""
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
        key: tuple,
        restart_key: Optional[tuple],
        plan: window_records.StrongRegionPlan,
    ) -> tuple:
        """The restart window's reads from its re-sliced buffer start."""
        if restart_key is None:
            _refuse_terminal_restart_data(key, plan)
            return ()
        restart = self.planner.window_at(restart_key)
        _check_restart_tiling(key, restart_key, restart, plan)
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

        The strong window's context lives in syndrome buffer 1; the
        restart window's weak reads, the re-read range among them, sit
        in Buffer 0 under its potential restart hold (planned with the
        window, live until the weak chain restarts).
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
    ) -> ForwardRegion:
        """The resolved region with the two windows' error models."""
        operation = self.tracker.operation(key[0])
        strong_exclusions, restart_exclusions = _fault_exclusions(
            plan, round_count, restart_key
        )
        strong_window = _strong_window_of(key, plan)
        strong_model = self.planner.strong_window_model(
            operation, strong_window, round_count, strong_exclusions
        )
        proposed_restart = None
        restart_model = None
        if restart_key is not None:
            proposed_restart = self._proposed_restart_window(restart_key, plan)
            restart_model = self.planner.strong_window_model(
                operation, proposed_restart, round_count, restart_exclusions
            )
        return ForwardRegion(
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

    def _proposed_restart_window(
        self, restart_key: tuple, plan: window_records.StrongRegionPlan
    ) -> window_records.Window:
        """The restart window as it will read after the re-slice."""
        restart = self.planner.window_at(restart_key)
        proposed = copy.deepcopy(restart)
        proposed.buffer_lo = plan.restart_buffer_lo
        return proposed


def _context_window_of(
    weak_window: window_records.Window,
) -> window_records.Window:
    """The weak window with one buffer of raw context on each side.

    The strong window starts with the empty boundary state a Window is
    given, and not the escalated weak window's. This row folds no
    neighbour's boundary into its input (FOLDS_NO_BOUNDARY): both faces
    are read raw, so a mask on the commit_lo layer would flip a seam the
    rounds before it already carry as raw defects, which is the double
    count Bombin et al. 2303.04846 lines 775-788 rule out and the row's
    own module docstring forbids.
    """
    bounds = window_records.strong_context_bounds(weak_window)
    context_lo, commit_lo, commit_hi, context_hi = bounds
    round_count = context_hi - context_lo + 1
    strong_window = window_records.Window(
        operation_id=weak_window.operation_id,
        window_index=weak_window.window_index,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=context_hi,
        buffer_lo=context_lo,
        round_count=round_count,
    )
    return strong_window


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
    round_count = plan.context_hi - plan.context_lo + 1
    return window_records.Window(
        operation_id=key[0],
        window_index=key[1],
        commit_lo=plan.commit_lo,
        commit_hi=plan.commit_hi,
        buffer_hi=plan.context_hi,
        buffer_lo=plan.context_lo,
        round_count=round_count,
    )


def _check_region_bounds(
    key: tuple,
    weak_window: window_records.Window,
    round_count: int,
    plan: window_records.StrongRegionPlan,
) -> None:
    """Context holds commit holds the weak window, inside the operation."""
    nested_bounds = (
        1,
        plan.context_lo,
        plan.commit_lo,
        weak_window.commit_lo,
        weak_window.commit_hi,
        plan.commit_hi,
        plan.context_hi,
        round_count,
    )
    if not _is_nondecreasing(nested_bounds):
        raise RuntimeError(
            f"invalid strong-region bounds for {key}: context "
            f"{plan.context_lo}-{plan.context_hi}, commit "
            f"{plan.commit_lo}-{plan.commit_hi}, operation 1-{round_count}"
        )
    if plan.commit_lo != weak_window.commit_lo:
        raise RuntimeError(
            f"strong-region commit for {key} must start at the "
            f"escalated window's commit start {weak_window.commit_lo}"
        )


def _is_nondecreasing(values: tuple) -> bool:
    for left, right in zip(values, values[1:]):
        if right < left:
            return False
    return True


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
    """A window committing across the strong window's edge has no owner.

    The settings refuse the shape at build for the shipped interaction
    (policies.py, _refuse_crossing_strong_region); this is the contract
    behind it, since the interaction that planned the region is a
    plug-in and its plan is checked like an input.
    """
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


def _refuse_terminal_restart_data(
    key: tuple, plan: window_records.StrongRegionPlan
) -> None:
    has_restart_data = plan.restart_buffer_lo is not None
    has_seam_owner = plan.restart_seam_fault_owner is not None
    if has_restart_data or has_seam_owner:
        raise RuntimeError(
            f"terminal strong-region plan for {key} cannot define "
            f"restart seam data"
        )


def _check_restart_tiling(
    key: tuple,
    restart_key: tuple,
    restart: window_records.Window,
    plan: window_records.StrongRegionPlan,
) -> None:
    """The restart window commits right after the strong window."""
    expected_start = plan.commit_hi + 1
    if restart.commit_lo != expected_start:
        raise RuntimeError(
            f"strong-region plan for {key} ends at "
            f"{plan.commit_hi}, but restart {restart_key} "
            f"starts at {restart.commit_lo}; committed regions must "
            "tile without a gap"
        )
    if plan.restart_buffer_lo is None:
        raise RuntimeError(
            f"strong-region restart {restart_key} needs a "
            f"buffer start in 1-{restart.commit_lo}"
        )
    is_inside = 1 <= plan.restart_buffer_lo <= restart.commit_lo
    if not is_inside:
        raise RuntimeError(
            f"strong-region restart {restart_key} needs a "
            f"buffer start in 1-{restart.commit_lo}"
        )


def _fault_exclusions(
    plan: window_records.StrongRegionPlan,
    round_count: int,
    restart_key: Optional[tuple],
) -> tuple:
    """(strong window's, restart window's) fault exclusion ranges.

    The seam's owner keeps the crossing faults: when the strong region
    owns them the restart window excludes everything up to the strong
    window's edge; otherwise the strong window excludes the rounds past
    its edge and the restart window excludes only the rounds before it.
    """
    left_exclusions = _left_fault_exclusions(plan.commit_lo)
    if restart_key is None:
        return left_exclusions, None
    if (
        plan.restart_seam_fault_owner
        is window_records.SeamFaultOwner.STRONG_REGION
    ):
        restart_exclusions = ((1, plan.commit_hi),)
        return left_exclusions, restart_exclusions
    right_exclusion = (plan.commit_hi + 1, round_count)
    strong_exclusions = left_exclusions + (right_exclusion,)
    return strong_exclusions, left_exclusions
