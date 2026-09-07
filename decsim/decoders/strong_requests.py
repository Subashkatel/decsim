"""Which destination window waits for which strong result.

The decoder side of the strong tier: the strong request live for each
destination window and the physical job serving it (itself, or a merged
batch), the destinations waiting for a strong result or for its
selection to arrive over the weak-to-strong link, the strong results
held because nobody asked for them yet, and the weak decodes still
open. A run without strong escalation keeps this ledger too: only its
weak side (one open decode per destination) is ever touched, and every
strong query answers "nothing".

Invariants: a destination window has at most one unconsumed strong
result; a destination decodes weakly once at a time, so a strong result
reaches the attempt that asked; a strong result with no possible
consumer is an error, never dropped silently (Toshio et al. 2510.25222,
one strong re-decode per escalated window).
"""

import dataclasses
from typing import Optional

import decsim.message as message
import decsim.records.identity as identity_records

ACCURACY_FIELDS = (
    "correction",
    "logical_observables",
    "soft_output",
    "boundary_defects",
    "boundary_data",
)


@dataclasses.dataclass(frozen=True)
class LiveStrongRequest:
    """A strong request admitted for one destination, and its service job.

    The service job is the request itself, or the merged batch serving
    it.
    """

    request_job: message.DecodeJob
    service_job: message.DecodeJob


@dataclasses.dataclass(frozen=True)
class HeldStrongCompletion:
    """A finished strong result, per request, with the tick its decode ended.

    The request job carries the request key the result answers.
    """

    request_job: message.DecodeJob
    result: message.DecodeResult
    decode_output_ticks: int


@dataclasses.dataclass
class StrongCounts:
    """How many strong results were asked for, and how many cancelled."""

    needed: int = 0
    cancelled: int = 0


class StrongRequests:
    """The strong requests by destination window and their states."""

    def __init__(self) -> None:
        self.running_by_window: dict[tuple, LiveStrongRequest] = {}
        self.waiting_selection_by_window: dict[
            tuple, message.DecoderRequestKey
        ] = {}
        self.waiting_result_by_window: dict[
            tuple, message.DecoderRequestKey
        ] = {}
        self.held_by_window: dict[tuple, HeldStrongCompletion] = {}
        self.unresolved_weak_windows: set = set()
        self.counts = StrongCounts()

    # ------------------------------------------------------ admission

    def admit(self, job: message.DecodeJob, now: int) -> None:
        """Open the request's place: a strong destination, or a weak attempt.

        A gap half is neither tier: it feeds the join, not the ledger.
        """
        if job.strong_decode_for is not None:
            self.admit_strong(job, now)
            return
        if job.on_done is not None:
            return
        if job.gap_sibling_for is not None:
            return
        self.admit_weak(job, now)

    def admit_strong(self, job: message.DecodeJob, now: int) -> None:
        """Give one destination's next strong result to this request."""
        key = job.strong_decode_for
        if key in self.running_by_window or key in self.held_by_window:
            raise RuntimeError(
                f"duplicate strong decode for window {key}: a destination "
                "window has at most one unconsumed strong result"
            )
        self.running_by_window[key] = LiveStrongRequest(job, job)
        job.request_admitted_ticks = now

    def admit_weak(self, job: message.DecodeJob, now: int) -> None:
        """Open one destination window's decode attempt."""
        key = (job.op_id, job.window_id)
        if key in self.unresolved_weak_windows:
            raise RuntimeError(
                f"second weak decode for window {key} while the first is "
                "unresolved: a destination window decodes once at a time, "
                "so that its strong result reaches the attempt that asked"
            )
        self.unresolved_weak_windows.add(key)
        job.request_admitted_ticks = now

    def resolve_weak(self, key: tuple) -> None:
        """The destination's weak decode has its directive.

        The destination may be decoded again and stops keeping a strong
        result.
        """
        self.unresolved_weak_windows.remove(key)

    def is_live_request(self, job: message.DecodeJob) -> bool:
        """Whether the strong job still carries its destination's request.

        A request cancelled while its input crossed the link is not.
        """
        live = self.running_by_window.get(job.strong_decode_for)
        if live is None:
            return False
        return live.request_job is job

    # -------------------------------------------------------- queries

    def live(self, key: tuple) -> Optional[LiveStrongRequest]:
        """The request live for the destination, or None."""
        return self.running_by_window.get(key)

    def members_of(self, service_job: message.DecodeJob) -> list:
        """The request jobs one physical decode serves, in admission order."""
        members = []
        for live in self.running_by_window.values():
            if live.service_job is service_job:
                members.append(live.request_job)
        return members

    def carriers_for(self, key: tuple) -> tuple:
        """The live request or held completion that carries the result."""
        carriers = []
        live = self.running_by_window.get(key)
        if live is not None:
            carriers.append(live)
        held = self.held_by_window.get(key)
        if held is not None:
            carriers.append(held)
        return tuple(carriers)

    def destination_may_consume(self, key: tuple) -> bool:
        """Whether the destination waits for a strong result now or may yet.

        Its weak decode is open and its directive may still ask for one.
        """
        if key in self.waiting_result_by_window:
            return True
        if key in self.waiting_selection_by_window:
            return True
        return key in self.unresolved_weak_windows

    # --------------------------------------------- batching and service

    def register_batch(
        self, window_keys: list, request_jobs: list, batch: message.DecodeJob
    ) -> None:
        """One merged batch now serves every member request."""
        for window_key, request_job in zip(window_keys, request_jobs):
            self.running_by_window[window_key] = LiveStrongRequest(
                request_job, batch
            )

    def finish_service(self, service_job: message.DecodeJob) -> None:
        """The physical decode ended; its requests are no longer running."""
        entries = self.running_by_window.items()
        entries = tuple(entries)
        for key, live in entries:
            if live.service_job is service_job:
                self.running_by_window.pop(key)

    def take_live(self, key: tuple) -> Optional[LiveStrongRequest]:
        """Drop and return the destination's live request, if any."""
        return self.running_by_window.pop(key, None)

    def take_held(self, key: tuple) -> Optional[HeldStrongCompletion]:
        """Drop and return the destination's held completion, if any."""
        return self.held_by_window.pop(key, None)

    def has_survivors(self, service_job: message.DecodeJob) -> bool:
        """Whether any request the decode serves is still live."""
        for live in self.running_by_window.values():
            if live.service_job is service_job:
                return True
        return False

    # --------------------------------------- selection and completion

    def begin_selection(
        self, key: tuple, request_key: message.DecoderRequestKey
    ) -> None:
        """The destination asked for this request's result.

        The selection is on its way over the weak-to-strong link.
        """
        self.counts.needed += 1
        self.waiting_selection_by_window[key] = request_key

    def select(
        self, key: tuple, request_key: message.DecoderRequestKey
    ) -> Optional[HeldStrongCompletion]:
        """The selection arrived: the destination now waits for the result.

        Returns a completion held from before, or None.
        """
        if self.waiting_selection_by_window.get(key) != request_key:
            return None
        del self.waiting_selection_by_window[key]
        self.waiting_result_by_window[key] = request_key
        held = self.held_by_window.get(key)
        if held is None:
            return None
        if held.request_job.request_key != request_key:
            return None
        return self.held_by_window.pop(key)

    def complete(self, held: HeldStrongCompletion) -> bool:
        """A strong result is ready: True when a destination consumes it now.

        Otherwise it is held for the demand that is still coming.
        """
        request_key = held.request_job.request_key
        key = (request_key.operation_id, request_key.window_id)
        if self.waiting_result_by_window.get(key) == request_key:
            del self.waiting_result_by_window[key]
            return True
        if key in self.running_by_window:
            raise RuntimeError(
                f"strong result for window {key} arrived after a newer "
                "strong request took the destination's next result: "
                "nothing would consume this one"
            )
        if not self.destination_may_consume(key):
            raise RuntimeError(
                f"strong result for window {key} has no destination "
                "waiting for it: the destination registered no strong "
                "demand and its decode attempt has resolved"
            )
        self.held_by_window[key] = held
        return False

    def deliveries_for(
        self,
        service_job: message.DecodeJob,
        result: message.DecodeResult,
        now: int,
    ) -> tuple:
        """Per-request completions of one finished strong decode.

        A merged batch may only carry timing: no accuracy-bearing field.
        """
        requests = self.members_of(service_job)
        keys = []
        for request in requests:
            keys.append(request.strong_decode_for)
        result_identity = (result.op_id, result.window_id)
        is_merged = len(keys) > 1
        for key in keys:
            if key != result_identity:
                is_merged = True
        if not is_merged:
            request = requests[0]
            return (HeldStrongCompletion(request, result, now),)
        populated = _populated_accuracy_fields(result)
        if populated:
            listed = ", ".join(populated)
            raise RuntimeError(
                "merged strong decode returned accuracy-bearing fields "
                f"({listed}); disable bulk_strong for accuracy-coupled "
                "switching"
            )
        deliveries = []
        for key, request in zip(keys, requests):
            empty = message.DecodeResult(op_id=key[0], window_id=key[1])
            held = HeldStrongCompletion(request, empty, now)
            deliveries.append(held)
        return tuple(deliveries)

    # ---------------------------------------------------- observation

    def unsettled(self) -> dict:
        """Every state still holding a destination, by its name."""
        states = (
            ("waiting for a strong result", self.waiting_result_by_window),
            ("waiting for strong selection", self.waiting_selection_by_window),
            ("holding an unclaimed strong result", self.held_by_window),
            ("still holding a strong request", self.running_by_window),
            ("decoding with no outcome", self.unresolved_weak_windows),
        )
        unsettled = {}
        for state, keys in states:
            if keys:
                unsettled[state] = sorted(keys)
        return unsettled

    def snapshot(self, queue_memberships: dict) -> tuple:
        """Each physical strong job once in its authoritative phase.

        running, queued, or in_transit (admitted but not yet queued: in
        input transport). queue_memberships maps id(job) to the queued
        jobs with that identity.
        """
        jobs_by_identity = {}
        keys_by_identity = {}
        for destination_key, live in self.running_by_window.items():
            job = live.service_job
            identity = id(job)
            jobs_by_identity[identity] = job
            keys = keys_by_identity.setdefault(identity, [])
            keys.append(destination_key)
        records = []
        for identity, job in jobs_by_identity.items():
            queued_matches = queue_memberships.get(identity, ())
            phase = _phase_of(job, queued_matches)
            destination_keys = sorted(
                keys_by_identity[identity],
                key=identity_records.stable_identity_order_key,
            )
            records.append((tuple(destination_keys), phase, job.n_rounds))
        return tuple(sorted(records, key=_snapshot_order))


def is_merged_batch(job: message.DecodeJob) -> bool:
    """A batch serves several strong requests and has no request itself."""
    if job.request_key is not None:
        return False
    return job.strong_decode_for is not None


def _populated_accuracy_fields(result: message.DecodeResult) -> list:
    populated = []
    for field_name in ACCURACY_FIELDS:
        value = getattr(result, field_name)
        if value is not None:
            populated.append(field_name)
    return populated


def _phase_of(job: message.DecodeJob, queued_matches) -> str:
    for candidate in queued_matches:
        if candidate is not job:
            raise RuntimeError("strong-work identity collision in ready queues")
    if len(queued_matches) > 1:
        raise RuntimeError("one strong job appears in multiple ready queues")
    if job.pool is not None:
        if queued_matches:
            raise RuntimeError("running strong job also remains queued")
        return "running"
    if queued_matches:
        return "queued"
    return "in_transit"


def _snapshot_order(record: tuple) -> tuple:
    destination_keys, phase, round_count = record
    ordered_keys = []
    for key in destination_keys:
        order_key = identity_records.stable_identity_order_key(key)
        ordered_keys.append(order_key)
    return tuple(ordered_keys), phase, round_count
