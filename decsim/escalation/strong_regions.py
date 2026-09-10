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
import decsim.records.program as program_records
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class RedoRegion:
    """One strong redo of one window: the window, and the rounds it reads."""

    window: window_records.Window
    context_read_keys: tuple


@dataclasses.dataclass(frozen=True)
class _ForwardProposal:
    """The extent an interaction proposes, before it is checked.

    A row that reads the extent with no context narrows the plan here,
    so the checks, the holds and the models that follow all read the
    rounds the row will really read.
    """

    weak_window: window_records.Window
    round_count: int
    later_windows: list
    plan: window_records.StrongRegionPlan


@dataclasses.dataclass(frozen=True)
class _PinnedFaces:
    """The faces a forward row pins its input on.

    A row that pins nothing passes None instead of this record and
    reads every face raw. near_source_key is the window before the
    region, or None when no committed correction closes that face; the
    far face pins on the restart window whenever the region has one.
    """

    near_source_key: Optional[tuple]


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

    def context_region(self, key: tuple) -> RedoRegion:
        """The escalated window with one buffer of raw context per side."""
        weak_window = self.planner.window_at(key)
        strong_window = _context_window_of(weak_window)
        read_keys = self.retention.read_keys_for_bounds(
            key[0],
            strong_window.buffer_lo,
            strong_window.buffer_hi,
            strong_window,
        )
        return RedoRegion(strong_window, tuple(read_keys))

    def near_seam_region(self, key: tuple) -> RedoRegion:
        """The escalated window's commit rounds, its past face pinned.

        The rounds before the commit region are not read: their defects
        arrive as the neighbour's committed boundary instead (Bombin et
        al. 2303.04846 lines 1456-1458). One buffer region of raw
        context stays on the open future face.
        """
        weak_window = self.planner.window_at(key)
        strong_window = _near_pinned_window_of(weak_window)
        read_keys = self.retention.read_keys_for_bounds(
            key[0],
            strong_window.buffer_lo,
            strong_window.buffer_hi,
            strong_window,
        )
        return RedoRegion(strong_window, tuple(read_keys))

    def near_seam_source(self, key: tuple) -> Optional[tuple]:
        """The window whose commit region ends where this one's begins.

        A pinned face is the seam between two commit regions, so the
        source is the escalated window's own dependency that commits the
        round before it (Bombin et al. 2303.04846 lines 703-704: what a
        task commits is the restriction to its commit region). None for
        a window with no earlier neighbour, whose past face is the
        operation's first round layer, closed by the initialisation.
        """
        weak_window = self.planner.window_at(key)
        seam_round = weak_window.commit_lo - 1
        for dependency_key in weak_window.deps:
            neighbour = self.planner.window_at(dependency_key)
            if neighbour.commit_hi == seam_round:
                return dependency_key
        return None

    def redecode_model(
        self,
        key: tuple,
        window: window_records.Window,
        pinned_source_keys: tuple,
    ):
        """The error model of one strong redo of a window.

        The redo owns the faults of its own rounds and none of the
        rounds before its commit region: those belong to the window
        that committed them, whether this row reads them as raw context
        or pins them. A face read raw keeps those faults as columns, so
        the decoder can still explain a defect with one; a pinned face
        does not, because the neighbour has decided them and its answer
        is already in the input (Bombin et al. 2303.04846 lines
        775-788, task j decodes over its own error generators).
        """
        round_count = self.round_count_for(key[0], window)
        exclusions = _left_fault_exclusions(window.commit_lo)
        operation = self.tracker.operation(key[0])
        prior_faults = self._faults_the_pins_carry(pinned_source_keys)
        return self.planner.strong_window_model(
            operation, window, round_count, exclusions, prior_faults
        )

    def _faults_the_pins_carry(self, pinned_source_keys: tuple):
        """What the windows behind the pinned faces have committed.

        The union over the pinned faces, per fault representation. None
        when the row pins nothing, or when the run builds no error
        models and no window owns anything.
        """
        owned_sets = []
        for source_key in pinned_source_keys:
            owned = self.planner.owned_faults_of(source_key)
            if owned is not None:
                owned_sets.append(owned)
        return _union_of_owned_faults(owned_sets)

    def forward_region(self, key: tuple) -> ForwardRegion:
        """The forward strong region of an escalated window, resolved.

        The interaction proposes the extent, this checks it against the
        windows that exist, and the rounds each side reads are required
        to be retained before anything moves.
        """
        proposal = self._proposed_forward_region(key)
        return self._resolved_forward_region(key, proposal, pinned_faces=None)

    def forward_seam_region(
        self, key: tuple, *, near_source_key: Optional[tuple]
    ) -> ForwardRegion:
        """The same extent, read with no context on its pinned faces.

        Toshio et al. 2510.25222 Fig. 12 assigns the strong decoder
        r_strong rounds and no more, "after the boundary conditions at
        both ends have been determined by the weak decoder" (lines
        1248-1250). A determined boundary condition needs no buffer
        behind it (Bombin et al. 2303.04846 lines 1456-1458), and the
        future face is determined either by the restart window's commit
        or, at the operation's end, by the readout (Tan et al.
        2209.09219 lines 953-955), so the region never reads past its
        commit region. A near face that no committed correction closes
        is open, and an open face keeps one buffer region of raw context
        (Bombin lines 850-852).
        """
        proposal = self._proposed_forward_region(key)
        plan = proposal.plan
        context_lo = plan.context_lo
        if near_source_key is not None:
            context_lo = plan.commit_lo
        pinned_plan = dataclasses.replace(
            plan, context_lo=context_lo, context_hi=plan.commit_hi
        )
        pinned = dataclasses.replace(proposal, plan=pinned_plan)
        pinned_faces = _PinnedFaces(near_source_key)
        return self._resolved_forward_region(
            key, pinned, pinned_faces=pinned_faces
        )

    def _proposed_forward_region(self, key: tuple) -> "_ForwardProposal":
        """The extent the interaction proposes, with what it was read on."""
        operation_id, escalated_index = key
        weak_window = self.planner.window_at(key)
        round_count = self.round_count_for(operation_id, weak_window)
        later_windows = self.planner.later_windows(
            operation_id, escalated_index
        )
        plan = self._plan_region(weak_window, later_windows, round_count)
        return _ForwardProposal(
            weak_window=weak_window,
            round_count=round_count,
            later_windows=later_windows,
            plan=plan,
        )

    def _resolved_forward_region(
        self,
        key: tuple,
        proposal: "_ForwardProposal",
        *,
        pinned_faces: Optional["_PinnedFaces"],
    ) -> ForwardRegion:
        """The proposed extent checked against the windows that exist."""
        operation_id = key[0]
        weak_window = proposal.weak_window
        round_count = proposal.round_count
        later_windows = proposal.later_windows
        plan = proposal.plan
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
            pinned_faces,
        )

    def operation(self, operation_id) -> program_records.Operation:
        """The operation record a strong window's transfers are named by."""
        return self.tracker.operation(operation_id)

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
        pinned_faces: Optional["_PinnedFaces"],
    ) -> ForwardRegion:
        """The resolved region with the two windows' error models.

        The restart window is modelled first, because a row that pins
        its far face on that window's commit takes the faults the
        restart window owns as prior faults of its own model.
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
        prior_faults = self._forward_prior_faults(pinned_faces, restart_model)
        strong_model = self.planner.strong_window_model(
            operation,
            strong_window,
            round_count,
            strong_exclusions,
            prior_faults,
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

    def _forward_prior_faults(
        self, pinned_faces: Optional["_PinnedFaces"], restart_model
    ):
        """What the faces of a forward region's pins carry.

        The near face pins on the window before the region, whose
        committed faults the planner holds; the far face pins on the
        restart window, whose model is the one built just above. A row
        that pins neither face gets None and keeps every candidate fault
        as a column.
        """
        if pinned_faces is None:
            return None
        owned_sets = []
        near_source_key = pinned_faces.near_source_key
        if near_source_key is not None:
            near_owned = self.planner.owned_faults_of(near_source_key)
            if near_owned is not None:
                owned_sets.append(near_owned)
        if restart_model is not None:
            restart_owned = restart_model.owned_fault_ids()
            owned_sets.append(restart_owned)
        return _union_of_owned_faults(owned_sets)

    def _proposed_restart_window(
        self, restart_key: tuple, plan: window_records.StrongRegionPlan
    ) -> window_records.Window:
        """The restart window as it will read after the re-slice."""
        restart = self.planner.window_at(restart_key)
        proposed = copy.deepcopy(restart)
        proposed.buffer_lo = plan.restart_buffer_lo
        return proposed


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


def _near_pinned_window_of(
    weak_window: window_records.Window,
) -> window_records.Window:
    """The weak window's commit rounds plus one trailing buffer region."""
    bounds = window_records.near_pinned_bounds(weak_window)
    context_lo, commit_lo, commit_hi, context_hi = bounds
    round_count = context_hi - context_lo + 1
    return window_records.Window(
        operation_id=weak_window.operation_id,
        window_index=weak_window.window_index,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=context_hi,
        buffer_lo=context_lo,
        round_count=round_count,
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
