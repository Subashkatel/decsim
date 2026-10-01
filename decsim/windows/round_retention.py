"""The round retention: which rounds each window and request holds.

A window holds [start_round, buffer_hi] plus the successor overflow in
the store its tier reads; when the strong tier may re-decode it, the
same rounds and the ones a strong redo of it reads (its commit and one
buffer past it, records/windows.py strong_context_bounds) are held as a
potential strong read in both stores: in the weak syndrome
buffer, because the chip keeps them until the verdict and carries them
up with the escalation (Toshio 2510.25222 lines 1247 to 1250), and in
the strong syndrome buffer, where they are expected. At admission the
window's hold becomes the request's and is released once the input
lands in the unit's memory. Under the double window a window
that an earlier window bounds also keeps the rounds its restart would
read as a potential restart read (PotentialRestart, planned in
frontends/planner.py), past its own request and landing: an earlier
escalation may re-slice it as the restart window, which re-reads
escalation.restart_reread_buffer_regions buffer regions of the strong
region (Toshio et al. 2510.25222 Sec. III C). A read whose reader forms
the detection events and never read the rounds before its first (a
strong read, a restart read past a strong region) also holds the raw
rounds before its first that its rounds' recipes read
(strong_rounds_before, primary_rounds_before), under the read's own
hold, as an HEVC decoder keeps each picture the current reference set
names (FFmpeg hevc/refs.c:486-517). A window's own weak read holds none:
the earlier window's read holds those rounds until it lands in the same
decoder, which forms them and keeps each packet while a round it has
not formed reads it (detector_formation.StreamingDetectorFormer), even
when it forms a later block first, so the hold ends
when the reader no longer needs the round in the store, as on a read of
no formation. The read ends when the window before it commits, or when
the window is re-sliced or absorbed. The stores hold slots and
holders; which rounds a window needs is decided here, gem5's split
between the cache that allocates a miss buffer and the queue that holds
the entry
(src/mem/cache/base.hh allocateMissBuffer).
"""

import functools
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.windows.round_tracker as round_tracker
import decsim.windows.window_planner as window_planner


class RoundRetention:
    """Which rounds each window and request keeps alive, and where."""

    weak_store = ports.Port(ports.RetainedRounds)
    # a run that reads no rounds from the room side has no store there
    strong_store = ports.Port(ports.RetainedRounds, optional=True)
    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker.RoundTracker)
    # the run's placement, asked how many rounds before a read's first
    # round its recipes read when a seat on that read forms the events
    detection_events = ports.Port(ports.DetectionEventPlacement)

    def __init__(
        self,
        *,
        is_strong_context_retained: bool,
        primary_tier: window_records.DecoderTier,
        strong_side_forms: bool = False,
        primary_reader_forms: bool = False,
    ) -> None:
        self.is_strong_context_retained = is_strong_context_retained
        self.primary_tier = primary_tier
        # a seat past the weak syndrome buffer on the escalation path
        # forms the detection events (detection_events.formed_at)
        self.strong_side_forms = strong_side_forms
        # the decoder the primary store feeds forms the detection events
        self.primary_reader_forms = primary_reader_forms

    # ---- the stores

    @property
    def primary_store(self) -> ports.RetainedRounds:
        """The store the primary tier reads.

        The weak syndrome buffer for the weak lane, strong syndrome buffer for
        a strong-primary plan.
        """
        if self.primary_tier is window_records.DecoderTier.STRONG:
            return self.strong_store
        return self.weak_store

    def store_for(
        self, store: Optional[ports.RetainedRounds]
    ) -> ports.RetainedRounds:
        """The given store, or the store the primary tier reads."""
        if store is None:
            return self.primary_store
        return store

    def install_planned_holds(self, buffering_plan) -> None:
        """Place the holds the plan's windows keep on the stores."""
        for owner, identities in buffering_plan.weak_holds:
            self.primary_store.register_hold(owner, identities)
        for owner, identities in buffering_plan.potential_holds:
            for store in self._strong_context_stores():
                store.register_hold(owner, identities)

    def release_round_if_unheld(self, round_key: tuple) -> None:
        """Free a weak syndrome buffer round whose every consumer resolved."""
        self.weak_store.release_round_if_unheld(round_key)

    # ---- a window's reads

    def register_window(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Register the primary read and any possible escalation context.

        A committed window before it may still keep its potential strong
        read for this one (release_committed_strong_read); that hold
        ends once this window has claimed its own rounds.
        """
        primary_reads = self.primary_read_keys(window)
        strong_context = self.strong_context_read_keys(window, primary_reads)
        reads = decoding_records.WindowReads(key)
        self.primary_store.register_hold(reads, primary_reads)
        potential = decoding_records.PotentialStrong(key)
        held = primary_reads + strong_context
        for store in self._strong_context_stores():
            store.register_hold(potential, held)
        self._release_the_committed_window_before(window)

    def release_committed_strong_read(
        self, window: window_records.Window
    ) -> None:
        """A final window's potential strong read ends, or waits for the next.

        A strong side that forms reads the rounds before a strong
        region's commit, which are the tail of the window before it. On
        a live stream whose next window is not admitted yet, this hold
        is what keeps them, so it ends when the next window registers
        (register_window), the order the stream path keeps for every
        hold: a read claims its rounds before the release that would
        free them.
        """
        if self._keeps_for_the_next_window(window):
            return
        potential = decoding_records.PotentialStrong(window.key)
        self.release_strong_hold_if_live(potential)

    def replace_window_reads(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Re-point the window's live holds at its reads.

        The potential strong read moves first, so a shrinking primary read
        stays in strong retention before its release. A re-sliced
        restart window whose own hold already moved to its request
        keeps its potential restart hold, which now names the rounds
        the fresh request reads, the re-read range among them.
        """
        primary_reads = self.primary_read_keys(window)
        self._repoint_reads(key, window, primary_reads)

    def replace_restart_reads(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Re-point a re-sliced restart window's live holds at its reads.

        The restart window starts inside a strong region whose absorbed
        windows no weak decode read, so its decoder never formed the
        rounds before its start: when it forms the events, its read
        also holds the raw rounds before its start that its rounds read
        (primary_rounds_before).
        """
        primary_reads = self.primary_rounds_before(
            window.operation_id, window.start_round, window.buffer_hi
        )
        primary_reads += self.primary_read_keys(window)
        self._repoint_reads(key, window, primary_reads)

    def release_restart_reads(self, key: tuple) -> None:
        """No earlier escalation can re-slice the window: its claim ends."""
        restart = decoding_records.PotentialRestart(key)
        self.release_hold_if_live(restart)

    def reset_clipped_window_reads(self, window: window_records.Window) -> None:
        """After clipping a live tail, retain only its primary commit range.

        A window already queued has handed its reads to its decode
        request (decode_requests.py, _bind_input_hold), so it has none
        left to shrink. A measurement_closed boundary queues the last
        window before the seal reaches it.
        """
        reads = decoding_records.WindowReads(window.key)
        if not self.primary_store.has_hold(reads):
            return
        stop_round = window.commit_hi + 1
        new_reads = []
        for round_index in range(window.start_round, stop_round):
            new_reads.append((window.operation_id, round_index))
        self.primary_store.replace_hold(reads, new_reads)

    def primary_read_keys(self, window: window_records.Window) -> list:
        """The rounds the window's primary decode reads, from its start.

        None of the rounds before its start: an earlier window's read
        holds them until it lands in the same decoder, which forms them
        there and keeps each packet while a round it has not formed
        reads it.
        """
        return self.read_keys_for_bounds(
            window.operation_id, window.start_round, window.buffer_hi, window
        )

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
        bounds = window_records.strong_context_bounds(window)
        context_lo, _commit_lo, _commit_hi, context_hi = bounds
        weak = set(weak_reads)
        strong = self.strong_rounds_before(
            window.operation_id, context_lo, context_hi
        )
        strong += self.read_keys_for_bounds(
            window.operation_id, context_lo, context_hi, window
        )
        return [round_key for round_key in strong if round_key not in weak]

    def strong_rounds_before(
        self, operation_id: Any, first_round: int, last_round: int
    ) -> list:
        """The raw rounds a strong read of these rounds reads before them.

        When the strong side forms the events, a strong read starts at
        the earliest round any of its rounds' recipes read: the strong
        side's former has not seen the rounds the weak side decoded.
        None otherwise.
        """
        if not self.strong_side_forms:
            return []
        return self._rounds_read_before(operation_id, first_round, last_round)

    def primary_rounds_before(
        self, operation_id: Any, first_round: int, last_round: int
    ) -> list:
        """The raw rounds a primary read of these rounds reads before them.

        A decoder that forms the events and joins at first_round without
        having formed it reads the rounds its recipes name before it
        (detection_events, rounds_needed_before); which of them it still
        needs is known only at the read, so the hold keeps them all.
        """
        if not self.primary_reader_forms:
            return []
        return self._rounds_read_before(operation_id, first_round, last_round)

    # ---- holds moving between owners

    def release_hold_if_live(self, owner, store=None) -> None:
        """Drop a hold that is still live; nothing for one already gone."""
        store = self.store_for(store)
        if store.has_hold(owner):
            store.release_hold(owner)

    def holds_input(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the job's input is held in the primary tier's store."""
        if job.request_key is None:
            return False
        owner = decoding_records.DecoderInputHold(job.request_key)
        return self.primary_store.has_hold(owner)

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
        """A strong job's context is its hold, in both stores that keep it.

        The window's potential strong read or the request's pending
        hold, whichever is live, moves to the input in flight; a job
        with neither takes a fresh hold on its payload rounds. The
        rounds stay stored until the input lands in the unit's memory,
        on the chip too, so the region could be carried up again.
        """
        in_flight = decoding_records.StrongInputInFlight(job.request_key)
        owner = decoding_records.DecoderInputHold(job.request_key)
        for store in self._strong_context_stores():
            self._move_to_in_flight(store, job, in_flight)
            _move_hold_to_input(store, in_flight, owner, job)
        job.input_hold = functools.partial(self._release_strong_input, owner)

    # ---- what a strong window holds

    def open_operation_store(self, operation_id: Any) -> None:
        """Open the operation's rounds in the store its windows read."""
        # Operation identities are opaque to retention.
        self.primary_store.open_operation(operation_id)

    def has_operation_store(self, operation_id: Any) -> bool:
        """Whether the operation's syndrome RAM is still open."""
        return self.primary_store.has_operation(operation_id)

    def strong_window_input(self, builder, window) -> list:
        """The room-side payloads of a strong window, its first round stamped.

        Which store a strong window reads is the retention's to say, so
        the shape asks for the input rather than for the store.
        """
        builder.stamp_first_round(window, self.strong_store)
        return builder.assemble_payloads(window, self.strong_store)

    def context_rounds_in_flight(self, key: tuple, read_keys) -> tuple:
        """The rounds the strong syndrome buffer lacks that the weak one has.

        A round the QPU has not produced yet is not late. A round that
        reached the weak syndrome buffer and is not in the strong syndrome
        buffer is either still to be carried up by the escalation, or
        crossing on it, which its own live hold says, or was released
        while a reader still needs it, which is the run's mistake to
        report loudly. A caller carries or waits for the first and never
        for the second.
        """
        crossing = []
        released = []
        for round_key in read_keys:
            arrived = self.tracker.rounds_arrived(round_key[0])
            if round_key[1] > arrived:
                continue
            fragments = self.strong_store.retained_fragments(round_key)
            if fragments is not None:
                continue
            if self.strong_store.is_round_held(round_key):
                crossing.append(round_key)
                continue
            released.append(round_key)
        if released:
            raise RuntimeError(
                f"strong context for {key} arrived at the weak syndrome buffer "
                f"and is neither stored in the strong syndrome buffer nor "
                f"expected there: "
                f"{released} (the rounds were released while a strong "
                f"window still reads them)"
            )
        return tuple(crossing)

    def escalated_rounds(self, round_keys) -> tuple:
        """The weak syndrome buffer's packets of these rounds, to carry up.

        The rounds are there: the window's potential strong read keeps
        them on the chip until its verdict, and the request's hold after.
        """
        packets = []
        for round_key in round_keys:
            fragments = self.weak_store.retained_fragments(round_key)
            assert fragments is not None, (
                f"round {round_key} left the weak syndrome buffer before "
                f"its escalation carried it"
            )
            packet = round_records.SyndromeRoundPacket(
                round_key[0], round_key[1], fragments
            )
            packets.append(packet)
        return tuple(packets)

    def hold_strong_context(
        self,
        key: tuple,
        strong_request_key: window_records.DecoderRequestKey,
        context_keys: tuple,
    ) -> None:
        """The window's potential strong read becomes the request's hold."""
        potential_hold = decoding_records.PotentialStrong(key)
        pending_hold = decoding_records.PendingStrong(strong_request_key)
        stores = self._strong_context_stores()
        for store in stores:
            store.transfer_hold(potential_hold, pending_hold)
        for store in stores:
            store.replace_hold(pending_hold, context_keys)

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

        The strong syndrome buffer loses the absorbed windows' potential strong
        holds as the plan lands; the guard keeps the restart window's
        context until its re-sliced potential strong hold names it. Its
        the weak syndrome buffer reads need no guard: its potential restart
        hold is live until the weak chain restarts.
        """
        if restart_key is None:
            return None
        self._require_restart_claim(key, restart_key)
        guard = decoding_records.RephaseGuard(strong_request_key)
        guarded = self._guarded_strong_reads(
            key, restart_key, proposed_restart, context_keys, restart_read_keys
        )
        for store in self._strong_context_stores():
            store.register_hold(guard, guarded)
        return guard

    def release_strong_hold_if_live(self, owner) -> None:
        """Drop a strong context hold, in both stores, when it is registered."""
        for store in self._strong_context_stores():
            self.release_hold_if_live(owner, store)

    def release_absorbed_strong_hold(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """Drop the absorbed window's potential read; the request holds it.

        From the request's first round on, every round the absorbed
        window's own strong window would have read is the request's or
        the restart window's. A round behind that first round is behind
        the strong window too, and the absorbed window is never decoded,
        so no reader of it is left.
        """
        absorbed = decoding_records.PotentialStrong(key)
        absorbed_identities = self.strong_store.hold_round_identities(absorbed)
        replacement_identities = self.strong_store.hold_round_identities(
            replacement
        )
        needed = _from_the_first_round_on(
            absorbed_identities, replacement_identities
        )
        replacements = set(replacement_identities)
        if restart_key is not None:
            restart_potential = decoding_records.PotentialStrong(restart_key)
            restart_identities = self.strong_store.hold_round_identities(
                restart_potential
            )
            replacements.update(restart_identities)
        unheld = needed - replacements
        if unheld:
            listed = sorted(unheld)
            raise RuntimeError(
                f"absorbing window {key}: rounds {listed} are held by "
                f"neither the strong request nor the restart window"
            )
        for store in self._strong_context_stores():
            store.release_hold(absorbed)

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
        """Every listed round must still sit in the strong syndrome buffer."""
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

    def _keeps_for_the_next_window(self, window: window_records.Window) -> bool:
        """Whether a strong read of the next window may need this hold."""
        if not self.strong_side_forms:
            return False
        operation_id = window.operation_id
        if self.tracker.is_sealed(operation_id):
            return False
        next_key = (operation_id, window.window_index + 1)
        return next_key not in self.planner.windows_by_key

    def _release_the_committed_window_before(
        self, window: window_records.Window
    ) -> None:
        """End the potential strong read a committed window kept for this.

        Only a strong side that forms keeps one (_keeps_for_the_next_window).
        """
        if not self.strong_side_forms:
            return
        previous_key = (window.operation_id, window.window_index - 1)
        previous = self.planner.windows_by_key.get(previous_key)
        if previous is None or not previous.committed:
            return
        potential = decoding_records.PotentialStrong(previous_key)
        self.release_strong_hold_if_live(potential)

    def _repoint_reads(
        self, key: tuple, window: window_records.Window, primary_reads: list
    ) -> None:
        """Move the window's potential strong, read and restart holds."""
        strong_context = self.strong_context_read_keys(window, primary_reads)
        potential = decoding_records.PotentialStrong(key)
        held = primary_reads + strong_context
        for store in self._strong_context_stores():
            if store.has_hold(potential):
                store.replace_hold(potential, held)
        reads = decoding_records.WindowReads(key)
        if self.primary_store.has_hold(reads):
            self.primary_store.replace_hold(reads, primary_reads)
        restart = decoding_records.PotentialRestart(key)
        if self.weak_store.has_hold(restart):
            self.weak_store.replace_hold(restart, primary_reads)

    def _rounds_read_before(
        self, operation_id: Any, first_round: int, last_round: int
    ) -> list:
        """The round keys before first_round the read reaches, in order."""
        reach_count = self.detection_events.rounds_read_before(
            operation_id, first_round, last_round
        )
        earliest_round = first_round - reach_count
        round_keys = []
        for round_index in range(earliest_round, first_round):
            round_keys.append((operation_id, round_index))
        return round_keys

    def _strong_context_stores(self) -> tuple:
        """The stores that keep a window's strong context: both, or none."""
        if self.is_strong_context_retained:
            return (self.weak_store, self.strong_store)
        return ()

    def _move_to_in_flight(self, store, job, in_flight) -> None:
        """One store's potential or pending hold becomes the input in flight."""
        potential = decoding_records.PotentialStrong(job.strong_decode_for)
        pending = decoding_records.PendingStrong(job.request_key)
        if store.has_hold(potential):
            store.transfer_hold(potential, in_flight)
            return
        if store.has_hold(pending):
            store.transfer_hold(pending, in_flight)
            return
        round_identities = round_identities_of(job.payloads)
        store.register_hold(in_flight, round_identities)

    def _release_strong_input(self, owner) -> None:
        """The strong input landed in its unit: both stores let it go."""
        for store in self._strong_context_stores():
            store.release_hold(owner)


def round_identities_of(payloads) -> tuple:
    """The distinct (operation, round) keys of the payloads, in order."""
    identities = {}
    for fragment in payloads:
        identities[(fragment.operation_id, fragment.round_index)] = None
    return tuple(identities)


def _from_the_first_round_on(identities: tuple, first_of: tuple) -> set:
    """The identities at or after the earliest identity of first_of.

    An absorbed strong window's rounds behind the first round of the
    window that replaces it have no reader left; this keeps the rest,
    the rounds that still need a holder.
    """
    first_identity = min(first_of)
    kept = set()
    for identity in identities:
        if identity < first_identity:
            continue
        kept.add(identity)
    return kept


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
