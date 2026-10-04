"""Which destination window waits for which strong result.

The decoder side of the strong tier: the strong request live for each
destination window and the job serving it (itself, or a merged batch),
how far each destination's demand has travelled, and the weak decodes
still open. A run without escalation touches only the weak side, and
every strong query answers nothing.

The ledger holds no result: a finished result nobody has asked for waits
in the output slot of the unit that produced it (decoder_unit.py), as a
sender keeps a packet until the far side accepts it (gem5
port.hh:244-255).

Invariants: a destination window has at most one unconsumed strong
result; a destination decodes weakly once at a time, so a strong result
reaches the attempt that asked; a strong result with no possible
consumer is an error, never dropped (Toshio et al. 2510.25222, one
strong re-decode per escalated window).
"""

import dataclasses
import enum
from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records

# the fields a merged batch must leave empty: its members' results are
# timing alone, so anything here would be dropped without a word
ACCURACY_FIELDS = (
    "correction",
    "logical_observables",
    "soft_output",
    "boundary_defects",
    "boundary_data",
)


@dataclasses.dataclass(frozen=True)
class LiveStrongRequest:
    """A strong request admitted for one destination.

    The service job is the request itself, or the merged batch serving
    it.
    """

    request_job: decoding_records.DecodeJob
    service_job: decoding_records.DecodeJob


@dataclasses.dataclass(frozen=True)
class StrongCompletion:
    """A finished strong result, per request, with the tick its decode ended.

    The unit is the one that produced it; its output slot keeps the
    result until the destination asks for it.
    """

    request_job: decoding_records.DecodeJob
    result: decoding_records.DecodeResult
    decode_output_ticks: int
    unit: Optional[object] = None


@dataclasses.dataclass
class StrongCounts:
    """The run's tally of strong results.

    needed counts every strong result asked for; cancelled counts each
    one cancelled.
    """

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
    a strong result has travelled; open_weak_requests are the request
    keys of its weak attempt, one or two, still without a directive.
    """

    live: Optional[LiveStrongRequest] = None
    phase: StrongPhase = StrongPhase.NONE
    selected_request_key: Optional[window_records.DecoderRequestKey] = None
    open_weak_requests: set = dataclasses.field(default_factory=set)

    def is_empty(self) -> bool:
        """Whether the window is waiting on nothing and asks for nothing."""
        if self.live is not None:
            return False
        if self.phase is not StrongPhase.NONE:
            return False
        return not self.open_weak_requests


class StrongRequests:
    """The ledger of strong requests, by destination window."""

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
        if record.live is not None:
            raise RuntimeError(
                f"duplicate strong decode for window {key}: a destination "
                "window has at most one unconsumed strong result"
            )
        record.live = LiveStrongRequest(job, job)
        job.request_admitted_ticks = now

    def admit_weak(self, job: decoding_records.DecodeJob, now: int) -> None:
        """Open one request of a destination window's decode attempt.

        An attempt is the forced-class requests its window side submits
        together, one or two request keys.
        """
        key = (job.operation_id, job.window_id)
        record = self._record(key)
        record.open_weak_requests.add(job.request_key)
        job.request_admitted_ticks = now

    def resolve_weak(self, key: tuple) -> None:
        """The destination's weak decode has its directive.

        The destination may be decoded again and stops keeping a strong
        result.
        """
        record = self.by_window[key]
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
        for window_key, request_job in zip(
            window_keys, request_jobs, strict=True
        ):
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
    ) -> bool:
        """The selection arrived: the destination now waits for the result.

        True when this selection is the one the destination sent, so its
        result may leave the unit that produced it.
        """
        record = self.by_window.get(key)
        if record is None:
            return False
        if record.phase is not StrongPhase.WAITING_SELECTION:
            return False
        if record.selected_request_key != request_key:
            return False
        record.phase = StrongPhase.WAITING_RESULT
        return True

    def complete(self, completion: StrongCompletion) -> bool:
        """A strong result is ready: True when a destination consumes it now.

        False leaves it in the output slot of the unit that produced it,
        for the demand that is still coming.
        """
        request_key = completion.request_job.request_key
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
        return False

    def deliveries_for(
        self,
        service_job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        now: int,
    ) -> tuple:
        """Per-request completions of one finished strong decode.

        A merged batch carries timing only, so each member gets an empty
        result. The unit is read here, before the decode's end frees the
        job's slot, since its output slot holds the result.
        """
        unit = service_job.unit
        requests = self.members_of(service_job)
        if not _is_merged_service(requests, result):
            request = requests[0]
            return (StrongCompletion(request, result, now, unit),)
        populated = _populated_accuracy_fields(result)
        if populated:
            listed = ", ".join(populated)
            raise RuntimeError(
                "merged strong decode returned accuracy-bearing fields "
                f"({listed}); disable bulk_strong for accuracy-coupled "
                "switching"
            )
        deliveries = []
        for request in requests:
            key = request.strong_decode_for
            empty = decoding_records.DecodeResult(
                operation_id=key[0], window_id=key[1]
            )
            completion = StrongCompletion(request, empty, now, unit)
            deliveries.append(completion)
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


def is_merged_batch(job: decoding_records.DecodeJob) -> bool:
    """A batch serves several strong requests and has no request itself."""
    return job.kind is decoding_records.DecodeJobKind.STRONG_BATCH


def _states_of(record: WindowRequests) -> list:
    """The names unsettled() reports one window's record under."""
    states = []
    if record.phase is not StrongPhase.NONE:
        states.append(record.phase.value)
    if record.live is not None:
        states.append("still holding a strong request")
    if record.open_weak_requests:
        states.append("decoding with no outcome")
    return states


def _populated_accuracy_fields(result: decoding_records.DecodeResult) -> list:
    populated = []
    for field_name in ACCURACY_FIELDS:
        value = getattr(result, field_name)
        if value is not None:
            populated.append(field_name)
    return populated


def _is_merged_service(
    requests: list, result: decoding_records.DecodeResult
) -> bool:
    """Whether one decode served several requests, or another window's."""
    if len(requests) > 1:
        return True
    result_identity = (result.operation_id, result.window_id)
    for request in requests:
        if request.strong_decode_for != result_identity:
            return True
    return False
