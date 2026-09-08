"""The round retention: which rounds each window and request holds.

A window holds [start_round, buffer_hi] plus the successor overflow in
the store its tier reads; when the strong tier may re-decode it, the
same rounds plus one buffer of context on each side are held in the
room-side store as a potential strong read (Skoric et al. 2209.08552:
the buffer region is re-read by the next window). At admission the
window's hold becomes the request's and is released once the input
lands in the unit's memory. Under the forward strong window a window
that an earlier window bounds also keeps the rounds its restart would
read as a potential restart read (PotentialRestart, planned in
frontends/planner.py), past its own request and landing: an earlier
escalation may re-slice it as the restart window, which re-reads
escalation.restart_reread_buffer_regions buffer regions of the strong
region (Toshio et al. 2510.25222 Sec. III C).
The read ends when the window before it commits, or when the window is
re-sliced or absorbed. The stores hold slots and holders; which
rounds a window needs is decided here, gem5's split between the cache
that allocates a miss buffer and the queue that holds the entry
(src/mem/cache/base.hh allocateMissBuffer).
"""

import functools
from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


class RoundRetention:
    """Which rounds each window and request keeps alive, and where."""

    def __init__(
        self,
        weak_store,
        strong_store,
        planner,
        tracker,
        *,
        is_strong_context_retained: bool,
        primary_tier: window_records.DecoderTier,
    ) -> None:
        self.weak_store = weak_store
        self.strong_store = strong_store
        self.planner = planner
        self.tracker = tracker
        self.is_strong_context_retained = is_strong_context_retained
        self.primary_tier = primary_tier

    # ---- the stores

    @property
    def primary_store(self):
        """The store the primary tier reads.

        Buffer 0 for the weak lane, syndrome buffer 1 for a strong-primary
        plan.
        """
        if self.primary_tier is window_records.DecoderTier.STRONG:
            return self.strong_store
        return self.weak_store

    def store_for(self, store):
        """The given store, or Buffer 0 when none is named."""
        if store is None:
            return self.weak_store
        return store

    def install_planned_holds(self, buffering_plan) -> None:
        """Refuse a store the yaml sized below the plan; place its holds."""
        self._check_store_capacities(buffering_plan)
        for owner, identities in buffering_plan.weak_holds:
            self.primary_store.register_hold(owner, identities)
        for owner, identities in buffering_plan.potential_holds:
            self.strong_store.register_hold(owner, identities)

    def release_round_if_unheld(self, round_key: tuple) -> None:
        """Free a Buffer 0 round whose every consumer already resolved."""
        self.weak_store.release_round_if_unheld(round_key)

    # ---- a window's reads

    def register_window(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Register the weak and possible-strong holds of a new window."""
        weak = self.read_keys_for_bounds(
            window.operation_id, window.start_round, window.buffer_hi, window
        )
        strong = self.strong_context_read_keys(window, weak)
        self.weak_store.register_hold(key, weak)
        if self.is_strong_context_retained:
            potential = decoding_records.PotentialStrong(key)
            held = weak + strong
            self.strong_store.register_hold(potential, held)

    def replace_window_reads(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Re-point the window's live holds at its reads.

        The potential strong read moves first, so a shrinking weak read
        stays in strong retention before its release. A re-sliced
        restart window whose own hold already moved to its request
        keeps its potential restart hold, which now names the rounds
        the fresh request reads, the re-read range among them.
        """
        weak = self.read_keys_for_bounds(
            window.operation_id, window.start_round, window.buffer_hi, window
        )
        strong = self.strong_context_read_keys(window, weak)
        potential = decoding_records.PotentialStrong(key)
        if self.strong_store is not None and self.strong_store.has_hold(
            potential
        ):
            held = weak + strong
            self.strong_store.replace_hold(potential, held)
        if self.weak_store.has_hold(key):
            self.weak_store.replace_hold(key, weak)
        restart = decoding_records.PotentialRestart(key)
        if self.weak_store.has_hold(restart):
            self.weak_store.replace_hold(restart, weak)

    def release_restart_reads(self, key: tuple) -> None:
        """No earlier escalation can re-slice the window: its claim ends."""
        restart = decoding_records.PotentialRestart(key)
        self.release_hold_if_live(restart)

    def reset_clipped_window_reads(self, window: window_records.Window) -> None:
        """After clipping a live tail, retain only the weak commit range."""
        stop_round = window.commit_hi + 1
        new_reads = []
        for round_index in range(window.start_round, stop_round):
            new_reads.append((window.operation_id, round_index))
        new_reads.sort()
        self.weak_store.replace_hold(window.key, new_reads)

    def read_keys_for_bounds(
        self,
        operation_id,
        start_round: int,
        buffer_hi: int,
        window: Optional[window_records.Window] = None,
    ) -> list:
        """Retained payload round keys for a possibly cross-operation range."""
        operation_rounds = self.tracker.effective_round_count_for_window(
            operation_id, window
        )
        last_local_round = min(buffer_hi, operation_rounds)
        stop_round = last_local_round + 1
        reads = []
        for round_index in range(start_round, stop_round):
            reads.append((operation_id, round_index))
        overflow = buffer_hi - operation_rounds
        if overflow <= 0:
            return reads
        successor_ids = self.planner.successors_by_operation.get(
            operation_id, []
        )
        overflow_stop = overflow + 1
        for successor_id in successor_ids:
            for round_index in range(1, overflow_stop):
                reads.append((successor_id, round_index))
        return reads

    def strong_context_read_keys(
        self, window: window_records.Window, weak_reads: list
    ) -> list:
        """Rounds kept until the strong decoder is known to need them."""
        if not self.is_strong_context_retained:
            return []
        context_lo, _commit_lo, _commit_hi, context_hi = strong_context_bounds(
            window
        )
        weak = set(weak_reads)
        strong = self.read_keys_for_bounds(
            window.operation_id, context_lo, context_hi, window
        )
        return [round_key for round_key in strong if round_key not in weak]

    # ---- holds moving between owners

    def transfer_hold(self, previous, replacement, store=None) -> tuple:
        """Move a live hold to a new owner; returns the rounds it keeps."""
        store = self.store_for(store)
        keys = store.hold_round_identities(previous)
        store.transfer_hold(previous, replacement)
        return keys

    def release_hold_if_live(self, owner, store=None) -> None:
        """Drop a hold that is still live; nothing for one already gone."""
        store = self.store_for(store)
        if store.has_hold(owner):
            store.release_hold(owner)

    def transfer_potential_to_pending(self, window_key, request_key) -> tuple:
        """A window's possible strong read becomes an admitted request's."""
        potential = decoding_records.PotentialStrong(window_key)
        pending = decoding_records.PendingStrong(request_key)
        return self.transfer_hold(potential, pending, self.strong_store)

    def holds_input(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the job's input hold is already live in Buffer 0."""
        if job.request_key is None:
            return False
        owner = decoding_records.DecoderInputHold(job.request_key)
        return self.weak_store.has_hold(owner)

    def bind_input_hold(
        self, job: decoding_records.DecodeJob, previous_owner, store=None
    ) -> None:
        """Transfer upstream retention to an admitted input request.

        The release runs only after decoder memory materialization, so
        overlapping rounds remain upstream until their last consumer
        transfer.
        """
        store = self.store_for(store)
        owner = decoding_records.DecoderInputHold(job.request_key)
        if previous_owner != owner:
            _move_hold_to_input(store, previous_owner, owner, job)
        job.input_hold = functools.partial(store.release_hold, owner)

    def hold_strong_input(self, job: decoding_records.DecodeJob) -> None:
        """The strong job's context becomes its input hold in syndrome buffer 1.

        The window's potential strong read or the request's pending
        hold, whichever is live, moves to the input in flight; a job
        with neither takes a fresh hold on its payload rounds. The
        rounds stay stored until the input lands in the unit's memory.
        """
        strong_store = self.strong_store
        in_flight = decoding_records.StrongInputInFlight(job.request_key)
        potential = decoding_records.PotentialStrong(job.strong_decode_for)
        pending = decoding_records.PendingStrong(job.request_key)
        if strong_store.has_hold(potential):
            self.transfer_hold(potential, in_flight, strong_store)
        elif strong_store.has_hold(pending):
            self.transfer_hold(pending, in_flight, strong_store)
        else:
            round_identities = round_identities_of(job.payloads)
            strong_store.register_hold(in_flight, round_identities)
        self.bind_input_hold(job, in_flight, strong_store)

    # ---- what a strong window holds

    def open_operation_store(self, operation_id) -> None:
        """Open the operation's rounds in the store its windows read."""
        self.weak_store.open_operation(operation_id)

    def has_operation_store(self, operation_id) -> bool:
        """Whether the operation's syndrome RAM is still open."""
        return self.weak_store.has_operation(operation_id)

    def strong_window_input(self, builder, window) -> list:
        """The room-side payloads of a strong window, its first round stamped.

        Which store a strong window reads is the retention's to say, so
        the shape asks for the input rather than for the store.
        """
        builder.stamp_first_round(window, self.strong_store)
        return builder.assemble_payloads(window, self.strong_store)

    def require_context_stored(self, key: tuple, read_keys) -> None:
        """Every context round that already arrived sits in buffer 1.

        A round the QPU has not produced yet is not late; a round that
        reached Buffer 0 and is not in the room-side store was either
        released early or is still crossing, and both are the run's
        mistake to report loudly.
        """
        missing = []
        for round_key in read_keys:
            arrived = self.tracker.rounds_arrived(round_key[0])
            if round_key[1] > arrived:
                continue
            fragments = self.strong_store.retained_fragments(round_key)
            if fragments is None:
                missing.append(round_key)
        if not missing:
            return
        raise RuntimeError(
            f"strong context for {key} arrived at Buffer 0 but is not "
            f"stored in syndrome buffer 1: {missing} "
            f"(controller_to_strong_buffer lag beyond the escalation "
            f"margin, or an early release)"
        )

    def hold_strong_context(
        self, key: tuple, strong_request_key, context_keys
    ) -> None:
        """The window's potential strong read becomes the request's hold."""
        self.transfer_potential_to_pending(key, strong_request_key)
        pending_hold = decoding_records.PendingStrong(strong_request_key)
        self.strong_store.replace_hold(pending_hold, list(context_keys))

    def guard_restart_reads(
        self,
        key: tuple,
        restart_key: Optional[tuple],
        proposed_restart,
        strong_request_key,
        context_keys,
        restart_read_keys,
    ) -> Optional[decoding_records.RephaseGuard]:
        """Hold the restart window's strong context while a plan lands.

        Syndrome buffer 1 loses the absorbed windows' potential strong
        holds as the plan lands; the guard keeps the restart window's
        context until its re-sliced potential strong hold names it. Its
        Buffer 0 reads need no guard: its potential restart hold is live
        until the weak chain restarts.
        """
        if restart_key is None:
            return None
        self._require_restart_claim(key, restart_key)
        guard = decoding_records.RephaseGuard(strong_request_key)
        guarded = self._guarded_strong_reads(
            key, restart_key, proposed_restart, context_keys, restart_read_keys
        )
        self.strong_store.register_hold(guard, guarded)
        return guard

    def release_strong_hold_if_live(self, owner) -> None:
        """Drop a room-side hold when it is still registered."""
        self.release_hold_if_live(owner, self.strong_store)

    def release_absorbed_strong_hold(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """Drop the absorbed window's potential read; the request holds it."""
        absorbed = decoding_records.PotentialStrong(key)
        needed_identities = self.strong_store.hold_round_identities(absorbed)
        needed = set(needed_identities)
        replacement_identities = self.strong_store.hold_round_identities(
            replacement
        )
        replacements = set(replacement_identities)
        if restart_key is not None:
            restart_potential = decoding_records.PotentialStrong(restart_key)
            restart_identities = self.strong_store.hold_round_identities(
                restart_potential
            )
            replacements.update(restart_identities)
        if not needed <= replacements:
            raise RuntimeError("absorption replacement does not cover packets")
        self.strong_store.release_hold(absorbed)

    def require_rounds_retained(
        self, label: str, payloads: list, first_round: int, last_round: int
    ) -> None:
        """A strong window starts only once every round it reads is held."""
        covered = set()
        for payload in payloads:
            covered.add(payload.round_index)
        stop_round = last_round + 1
        needed = set(range(first_round, stop_round))
        if covered == needed:
            return
        listed = sorted(covered)
        raise RuntimeError(
            f"{label}: strong window submitted with rounds "
            f"{listed} but it needs "
            f"{first_round}-{last_round}; a strong window may "
            "only start once every required round is retained"
        )

    def _require_restart_claim(self, key: tuple, restart_key: tuple) -> None:
        """The restart window still claims its reads and the re-read range.

        The claim ends only when the window before the restart window
        commits, at absorption, or at the end of the plan, and the
        absorbed windows are checked uncommitted first, so no runtime
        path reaches a released claim here: an invariant, not a check.
        """
        claim = decoding_records.PotentialRestart(restart_key)
        assert self.weak_store.has_hold(claim), (
            f"strong-region plan for {key}: restart window {restart_key}'s "
            f"potential restart hold is no longer live"
        )

    def _guarded_strong_reads(
        self,
        key: tuple,
        restart_key: tuple,
        proposed_restart,
        context_keys,
        restart_read_keys,
    ) -> list:
        """Every room-side round the plan must keep while it lands."""
        escalated_potential = decoding_records.PotentialStrong(key)
        restart_potential = decoding_records.PotentialStrong(restart_key)
        escalated_identities = self.strong_store.hold_round_identities(
            escalated_potential
        )
        restart_identities = self.strong_store.hold_round_identities(
            restart_potential
        )
        guarded = list(escalated_identities)
        guarded += list(restart_identities)
        guarded += list(context_keys)
        restart_reads = list(restart_read_keys)
        guarded += self.strong_context_read_keys(
            proposed_restart, restart_reads
        )
        return guarded

    def require_strong_retained(self, round_keys, purpose: str) -> None:
        """Every listed round must still sit in the room-side store."""
        self.require_retained(round_keys, purpose, self.strong_store)

    def require_retained(
        self, round_keys: list, purpose: str, store=None
    ) -> None:
        """Reject a new consumer if any already-arrived input was released."""
        store = self.store_for(store)
        arrived_for = self.tracker.rounds_arrived
        if store is not self.weak_store:
            arrived_for = self.tracker.strong_rounds_arrived
        missing = []
        for round_key in round_keys:
            if _is_released(store, arrived_for, round_key):
                missing.append(round_key)
        if missing:
            raise RuntimeError(
                f"{purpose} requires retained payload rounds that are no "
                f"longer available: {missing}"
            )

    # ---- private

    def _check_store_capacities(self, buffering_plan) -> None:
        strong_is_primary = (
            self.primary_tier is window_records.DecoderTier.STRONG
        )
        capacity = self.weak_store.settings.rounds
        minimum = buffering_plan.minimum_live_rounds
        if strong_is_primary:
            minimum = ()
        if capacity is not None and capacity < len(minimum):
            raise ValueError(
                f"upstream syndrome buffer needs {len(minimum)} packet slots, "
                f"got {capacity}"
            )
        if self.strong_store is None:
            return
        strong_capacity = self.strong_store.settings.rounds
        strong_minimum = buffering_plan.sb1_minimum_live_rounds
        if strong_is_primary:
            # the plan's window reads live on the room-side store
            strong_minimum = buffering_plan.minimum_live_rounds
        if strong_capacity is not None and strong_capacity < len(
            strong_minimum
        ):
            raise ValueError(
                f"syndrome buffer 1 needs {len(strong_minimum)} packet "
                f"slots, got {strong_capacity}"
            )


def strong_context_bounds(window: window_records.Window) -> tuple:
    """(context_lo, commit_lo, commit_hi, context_hi) of a strong redo.

    One buffer of context on each side of the commit region, clipped at
    round 1.
    """
    buffer_span = window.buffer_hi - window.commit_hi
    buffer_rounds = max(0, buffer_span)
    context_start = window.commit_lo - buffer_rounds
    context_lo = max(1, context_start)
    context_hi = window.commit_hi + buffer_rounds
    return context_lo, window.commit_lo, window.commit_hi, context_hi


def _is_released(store, arrived_for, round_key: tuple) -> bool:
    """True when a round that already arrived is no longer in the store."""
    operation_id, round_index = round_key
    if not store.has_operation(operation_id):
        return True
    arrived = arrived_for(operation_id)
    if round_index > arrived:
        return False
    fragments = store.retained_fragments(round_key)
    return fragments is None


def _move_hold_to_input(
    store, previous_owner, owner, job: decoding_records.DecodeJob
) -> None:
    """The window's hold becomes the request's, or a fresh one is made."""
    if store.has_hold(previous_owner):
        store.transfer_hold(previous_owner, owner)
        return
    identities = round_identities_of(job.payloads)
    store.register_hold(owner, identities)


def round_identities_of(payloads) -> tuple:
    """The distinct (operation, round) keys of the payloads, in order."""
    identities = {}
    for fragment in payloads:
        identities[(fragment.operation_id, fragment.round_index)] = None
    return tuple(identities)
