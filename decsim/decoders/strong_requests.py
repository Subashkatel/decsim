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
import enum
from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.windows as window_records

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

    request_job: decoding_records.DecodeJob
    service_job: decoding_records.DecodeJob


@dataclasses.dataclass(frozen=True)
class HeldStrongCompletion:
    """A finished strong result, per request, with the tick its decode ended.

    The request job carries the request key the result answers.
    """

    request_job: decoding_records.DecodeJob
    result: decoding_records.DecodeResult
    decode_output_ticks: int


@dataclasses.dataclass
class StrongCounts:
    """How many strong results were asked for, and how many cancelled."""

    needed: int = 0
    cancelled: int = 0


class StrongPhase(enum.Enum):
    """Where one destination window stands in its strong request.

    NONE while it has asked for nothing; WAITING_SELECTION while its
    selection crosses the weak-to-strong link; WAITING_RESULT once the
    selection has landed and the result is owed.
    """

    NONE = "none"
    WAITING_SELECTION = "waiting for strong selection"
    WAITING_RESULT = "waiting for a strong result"


@dataclasses.dataclass
class WindowRequests:
    """One destination window's requests, in one record.

    live is the strong request admitted for it and the job serving that
    request; phase and selected_request_key are how far its demand for
    a strong result has travelled; held is a finished strong result
    nobody has asked for yet; open_weak_requests are the request keys of
    its weak attempt, one or two, still without a directive.
    """

    live: Optional[LiveStrongRequest] = None
    phase: StrongPhase = StrongPhase.NONE
    selected_request_key: Optional[window_records.DecoderRequestKey] = None
    held: Optional[HeldStrongCompletion] = None
    open_weak_requests: set = dataclasses.field(default_factory=set)

    def is_empty(self) -> bool:
        """Whether the window is waiting on nothing and holds nothing."""
        if self.live is not None or self.held is not None:
            return False
        if self.phase is not StrongPhase.NONE:
            return False
        return not self.open_weak_requests


class StrongRequests:
    """The strong requests by destination window and their states."""

    def __init__(self) -> None:
        self.by_window: dict[tuple, WindowRequests] = {}
        self.counts = StrongCounts()
        kinds = decoding_records.DecodeJobKind
        self.admit_by_job_kind = {
            kinds.WINDOW: self.admit_weak,
            kinds.STRONG_REDECODE: self.admit_strong,
        }

    # ------------------------------------------------------ admission

    def admit(self, job: decoding_records.DecodeJob, now: int) -> None:
        """Open the request's place: a strong destination, or a weak attempt.

        A job's kind says which; a self-contained decode has no window
        to be a request for, and a batch is registered by the queue that
        merged it.
        """
        admit = self.admit_by_job_kind.get(job.kind)
        if admit is None:
            return
        admit(job, now)

    def admit_strong(self, job: decoding_records.DecodeJob, now: int) -> None:
        """Give one destination's next strong result to this request."""
        key = job.strong_decode_for
        record = self._record(key)
        if record.live is not None or record.held is not None:
            raise RuntimeError(
                f"duplicate strong decode for window {key}: a destination "
                "window has at most one unconsumed strong result"
            )
        record.live = LiveStrongRequest(job, job)
        job.request_admitted_ticks = now

    def admit_weak(self, job: decoding_records.DecodeJob, now: int) -> None:
        """Open one request of a destination window's decode attempt.

        A window's attempt is the forced-class requests its window side
        submits together, so an attempt holds one or two request keys;
        each of them is admitted once.
        """
        key = (job.operation_id, job.window_id)
        record = self._record(key)
        if job.request_key in record.open_weak_requests:
            raise RuntimeError(
                f"weak request {job.request_key} for window {key} is already "
                "open: a DecodeJob is admitted once"
            )
        record.open_weak_requests.add(job.request_key)
        job.request_admitted_ticks = now

    def resolve_weak(self, key: tuple) -> None:
        """The destination's weak decode has its directive.

        The destination may be decoded again and stops keeping a strong
        result.
        """
        record = self.by_window[key]
        assert record.open_weak_requests, (
            f"window {key} has no open weak request to resolve"
        )
        record.open_weak_requests = set()
        self._drop_if_empty(key)

    def is_live_request(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the strong job still carries its destination's request.

        A request cancelled while its input crossed the link is not.
        """
        live = self.live(job.strong_decode_for)
        if live is None:
            return False
        return live.request_job is job

    # -------------------------------------------------------- queries

    def live(self, key: tuple) -> Optional[LiveStrongRequest]:
        """The request live for the destination, or None."""
        record = self.by_window.get(key)
        if record is None:
            return None
        return record.live

    def members_of(self, service_job: decoding_records.DecodeJob) -> list:
        """The request jobs one physical decode serves, in admission order."""
        members = []
        for live in self._live_requests():
            if live.service_job is service_job:
                members.append(live.request_job)
        return members

    def carriers_for(self, key: tuple) -> tuple:
        """The live request or held completion that carries the result."""
        record = self.by_window.get(key)
        if record is None:
            return ()
        carriers = []
        if record.live is not None:
            carriers.append(record.live)
        if record.held is not None:
            carriers.append(record.held)
        return tuple(carriers)

    def destination_may_consume(self, key: tuple) -> bool:
        """Whether the destination waits for a strong result now or may yet.

        Its weak decode is open and its directive may still ask for one.
        """
        record = self.by_window.get(key)
        if record is None:
            return False
        if record.phase is not StrongPhase.NONE:
            return True
        return bool(record.open_weak_requests)

    # --------------------------------------------- batching and service

    def register_batch(
        self,
        window_keys: list,
        request_jobs: list,
        batch: decoding_records.DecodeJob,
    ) -> None:
        """One merged batch now serves every member request."""
        for window_key, request_job in zip(window_keys, request_jobs):
            record = self._record(window_key)
            record.live = LiveStrongRequest(request_job, batch)

    def finish_service(self, service_job: decoding_records.DecodeJob) -> None:
        """The physical decode ended; its requests are no longer running."""
        keys = tuple(self.by_window)
        for key in keys:
            record = self.by_window[key]
            if record.live is None:
                continue
            if record.live.service_job is service_job:
                record.live = None
                self._drop_if_empty(key)

    def take_live(self, key: tuple) -> Optional[LiveStrongRequest]:
        """Drop and return the destination's live request, if any."""
        record = self.by_window.get(key)
        if record is None:
            return None
        live = record.live
        record.live = None
        self._drop_if_empty(key)
        return live

    def take_held(self, key: tuple) -> Optional[HeldStrongCompletion]:
        """Drop and return the destination's held completion, if any."""
        record = self.by_window.get(key)
        if record is None:
            return None
        held = record.held
        record.held = None
        self._drop_if_empty(key)
        return held

    def has_survivors(self, service_job: decoding_records.DecodeJob) -> bool:
        """Whether any request the decode serves is still live."""
        for live in self._live_requests():
            if live.service_job is service_job:
                return True
        return False

    # --------------------------------------- selection and completion

    def begin_selection(
        self, key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The destination asked for this request's result.

        The selection is on its way over the weak-to-strong link.
        """
        self.counts.needed += 1
        record = self._record(key)
        record.phase = StrongPhase.WAITING_SELECTION
        record.selected_request_key = request_key

    def select(
        self, key: tuple, request_key: window_records.DecoderRequestKey
    ) -> Optional[HeldStrongCompletion]:
        """The selection arrived: the destination now waits for the result.

        Returns a completion held from before, or None.
        """
        record = self.by_window.get(key)
        if record is None:
            return None
        if record.phase is not StrongPhase.WAITING_SELECTION:
            return None
        if record.selected_request_key != request_key:
            return None
        record.phase = StrongPhase.WAITING_RESULT
        held = record.held
        if held is None:
            return None
        if held.request_job.request_key != request_key:
            return None
        record.held = None
        self._drop_if_empty(key)
        return held

    def complete(self, held: HeldStrongCompletion) -> bool:
        """A strong result is ready: True when a destination consumes it now.

        Otherwise it is held for the demand that is still coming.
        """
        request_key = held.request_job.request_key
        key = (request_key.operation_id, request_key.window_id)
        if self._takes_the_awaited_result(key, request_key):
            return True
        if self.live(key) is not None:
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
        record = self._record(key)
        record.held = held
        return False

    def deliveries_for(
        self,
        service_job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        now: int,
    ) -> tuple:
        """Per-request completions of one finished strong decode.

        A merged batch may only carry timing: no accuracy-bearing field.
        """
        requests = self.members_of(service_job)
        keys = []
        for request in requests:
            keys.append(request.strong_decode_for)
        result_identity = (result.operation_id, result.window_id)
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
            empty = decoding_records.DecodeResult(
                operation_id=key[0], window_id=key[1]
            )
            held = HeldStrongCompletion(request, empty, now)
            deliveries.append(held)
        return tuple(deliveries)

    # ------------------------------------------------------- private

    def _takes_the_awaited_result(
        self, key: tuple, request_key: window_records.DecoderRequestKey
    ) -> bool:
        """Whether the destination was waiting for exactly this result."""
        record = self.by_window.get(key)
        if record is None:
            return False
        if record.phase is not StrongPhase.WAITING_RESULT:
            return False
        if record.selected_request_key != request_key:
            return False
        record.phase = StrongPhase.NONE
        record.selected_request_key = None
        self._drop_if_empty(key)
        return True

    def _record(self, key: tuple) -> WindowRequests:
        """The window's record, made on the first request that needs it."""
        record = self.by_window.get(key)
        if record is None:
            record = WindowRequests()
            self.by_window[key] = record
        return record

    def _drop_if_empty(self, key: tuple) -> None:
        """A window waiting on nothing leaves the ledger."""
        record = self.by_window.get(key)
        if record is None:
            return
        if record.is_empty():
            del self.by_window[key]

    def _live_requests(self) -> list:
        """Every live strong request, in the order its window was admitted."""
        live_requests = []
        for record in self.by_window.values():
            if record.live is not None:
                live_requests.append(record.live)
        return live_requests

    # ---------------------------------------------------- observation

    def unsettled(self) -> dict:
        """Every state still holding a destination, by its name."""
        unsettled = {}
        for key, record in self.by_window.items():
            for state in _states_of(record):
                keys = unsettled.setdefault(state, [])
                keys.append(key)
        for state in unsettled:
            unsettled[state] = sorted(unsettled[state])
        return unsettled

    def snapshot(self, queue_memberships: dict) -> tuple:
        """Each physical strong job once in its authoritative phase.

        running, queued, or in_transit (admitted but not yet queued: in
        input transport). queue_memberships maps id(job) to the queued
        jobs with that identity.
        """
        jobs_by_identity = {}
        keys_by_identity = {}
        for destination_key, record in self.by_window.items():
            if record.live is None:
                continue
            job = record.live.service_job
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
            records.append((tuple(destination_keys), phase, job.round_count))
        return tuple(sorted(records, key=_snapshot_order))


def _states_of(record: WindowRequests) -> list:
    """The names unsettled() reports one window's record under."""
    states = []
    if record.phase is not StrongPhase.NONE:
        states.append(record.phase.value)
    if record.held is not None:
        states.append("holding an unclaimed strong result")
    if record.live is not None:
        states.append("still holding a strong request")
    if record.open_weak_requests:
        states.append("decoding with no outcome")
    return states


def is_merged_batch(job: decoding_records.DecodeJob) -> bool:
    """A batch serves several strong requests and has no request itself."""
    return job.kind is decoding_records.DecodeJobKind.STRONG_BATCH


def _populated_accuracy_fields(result: decoding_records.DecodeResult) -> list:
    populated = []
    for field_name in ACCURACY_FIELDS:
        value = getattr(result, field_name)
        if value is not None:
            populated.append(field_name)
    return populated


def _phase_of(job: decoding_records.DecodeJob, queued_matches) -> str:
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
