"""Which destination window waits for which escalated result.

The decoder side of every escalate edge: the request live for each
destination window on the edge's target pool and the job serving it
(itself, or a merged batch), how far each destination's demand has
travelled, and the source pool's decodes of it still open. A record is
kept per destination window and target pool, so a window escalated
along a chain has one record per edge. A run without escalation touches
only the source side, and every query of a target answers nothing.

The ledger holds no result: a finished result nobody has asked for waits
in the output slot of the unit that produced it (decoder_unit.py), as a
sender keeps a packet until the far side accepts it (gem5
port.hh:244-255).

Invariants: a destination window has at most one unconsumed attempt per
target pool: one request, or one per forced logical class when the edge
after the pool reads a complementary gap, which takes the minimum-weight
correction in each class (Toshio et al. 2510.25222 Sec. II C; Smith et
al. 2405.03766 Sec. II); a destination decodes on an edge's source once
at a time, so a target's result reaches the attempt that asked; a result
with no possible consumer is an error, never dropped (Toshio et al.
2510.25222, one strong re-decode per escalated window).
"""

import dataclasses
import enum
from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records

# the target each source pool escalates to: a legacy machine's one edge
LEGACY_TARGET_BY_SOURCE = {
    window_records.WEAK_POOL_ID: window_records.STRONG_POOL_ID,
}

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
    one cancelled; abandoned each one let run out and dropped at its end
    (revision_rules.NoCancel).
    """

    needed: int = 0
    cancelled: int = 0
    abandoned: int = 0


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

    live are the strong requests of its attempt admitted and not yet
    finished, with the jobs serving them; phase and selected_request_key
    are how far its demand for a strong result has travelled, the key
    naming the attempt by its lead request; results_owed counts the
    selected attempt's results still to reach it; companions_by_lead
    counts the forced-class companions admitted beside each lead;
    open_weak_requests are the request keys of its weak attempt, one or
    two, still without a directive.
    """

    live: list = dataclasses.field(default_factory=list)
    phase: StrongPhase = StrongPhase.NONE
    selected_request_key: Optional[window_records.DecoderRequestKey] = None
    results_owed: int = 0
    companions_by_lead: dict = dataclasses.field(default_factory=dict)
    open_weak_requests: set = dataclasses.field(default_factory=set)

    def is_empty(self) -> bool:
        """Whether the window is waiting on nothing and asks for nothing."""
        if self.live:
            return False
        if self.phase is not StrongPhase.NONE:
            return False
        return not self.open_weak_requests


class StrongRequests:
    """The ledger of escalated requests, by destination window and target.

    target_by_source names the pool each source pool escalates to, so a
    source decode opens its attempt in the record its target's request
    is kept in.
    """

    def __init__(self, target_by_source: Optional[dict] = None) -> None:
        if target_by_source is None:
            target_by_source = LEGACY_TARGET_BY_SOURCE
        self.target_by_source = target_by_source
        self.by_window: dict[tuple, WindowRequests] = {}
        # the live requests let run out, whose results are dropped
        self.abandoned_request_keys: set = set()
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
        """Give one destination's next result on the job's pool to it.

        A pool that escalates further is also the source of its own
        edge, so the job opens its attempt there too.
        """
        window_key = job.strong_decode_for
        pool = job.request_key.pool
        record = self._record((window_key, pool))
        lead_key = attempt_lead_of(job)
        is_companion = lead_key != job.request_key
        if record.live and not is_companion:
            raise RuntimeError(
                f"duplicate strong decode for window {window_key}: a "
                "destination window has at most one unconsumed strong result"
            )
        if is_companion:
            _admit_companion(record, lead_key)
        live = LiveStrongRequest(job, job)
        record.live.append(live)
        job.request_admitted_ticks = now
        if pool in self.target_by_source:
            self._open_attempt(window_key, job)

    def admit_weak(self, job: decoding_records.DecodeJob, now: int) -> None:
        """Open one request of a destination window's decode attempt.

        An attempt is the forced-class requests its window side submits
        together, one or two request keys.
        """
        window_key = (job.operation_id, job.window_id)
        self._open_attempt(window_key, job)
        job.request_admitted_ticks = now

    def resolve_weak(
        self, window_key: tuple, pool: window_records.PoolId
    ) -> None:
        """The destination's decode on pool, a source, has its directive.

        The destination may be decoded again and stops keeping its
        target's result.
        """
        key = self._attempt_key(window_key, pool)
        record = self.by_window[key]
        record.open_weak_requests = set()
        self._drop_if_empty(key)

    def is_live_request(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the strong job still carries its destination's request.

        A request cancelled while its input crossed the link is not.
        """
        window_key = job.strong_decode_for
        pool = job.request_key.pool
        for live in self.live_requests(window_key, pool):
            if live.request_job is job:
                return True
        return False

    # -------------------------------------------------------- queries

    def live_requests(
        self, window_key: tuple, pool: window_records.PoolId
    ) -> list:
        """The requests live for the destination on a target pool."""
        record = self.by_window.get((window_key, pool))
        if record is None:
            return []
        return list(record.live)

    def members_of(self, service_job: decoding_records.DecodeJob) -> list:
        """The request jobs one physical decode serves, in admission order."""
        members = []
        for live in self._live_requests():
            if live.service_job is service_job:
                members.append(live.request_job)
        return members

    def destination_may_consume(
        self, window_key: tuple, pool: window_records.PoolId
    ) -> bool:
        """Whether the destination waits for pool's result now or may yet.

        Its decode on the edge's source is open and its directive may
        still ask for one.
        """
        record = self.by_window.get((window_key, pool))
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
            pool = request_job.request_key.pool
            record = self._record((window_key, pool))
            record.live = [LiveStrongRequest(request_job, batch)]

    def finish_service(self, service_job: decoding_records.DecodeJob) -> None:
        """The physical decode ended; its requests are no longer running."""
        keys = tuple(self.by_window)
        for key in keys:
            record = self.by_window[key]
            remaining = _not_served_by(record.live, service_job)
            if len(remaining) == len(record.live):
                continue
            record.live = remaining
            self._drop_if_empty(key)

    def take_live(self, window_key: tuple, pool: window_records.PoolId) -> list:
        """Drop and return the destination's live requests on pool.

        The requests will meet no verdict, so the attempt each opened on
        the edge its pool is the source of closes with it.
        """
        key = (window_key, pool)
        record = self.by_window.get(key)
        if record is None:
            return []
        taken = record.live
        record.live = []
        record.companions_by_lead = {}
        self._drop_if_empty(key)
        for live in taken:
            self._close_attempt(window_key, live.request_job)
        return taken

    def abandon(self, window_key: tuple, pool: window_records.PoolId) -> None:
        """The destination's live request on pool runs out, unwanted.

        Its decode goes on, and its result is dropped when it ends
        (drop_abandoned).
        """
        for live in self.live_requests(window_key, pool):
            request_key = live.request_job.request_key
            self.abandoned_request_keys.add(request_key)

    def drop_abandoned(self, completion: StrongCompletion) -> bool:
        """Whether a finished result is an abandoned request's, dropped now."""
        request_job = completion.request_job
        request_key = request_job.request_key
        if request_key not in self.abandoned_request_keys:
            return False
        self.abandoned_request_keys.discard(request_key)
        self.counts.abandoned += 1
        self._close_attempt(request_job.strong_decode_for, request_job)
        return True

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
        record = self._record((key, request_key.pool))
        record.phase = StrongPhase.WAITING_SELECTION
        record.selected_request_key = request_key
        companions = record.companions_by_lead.get(request_key, 0)
        record.results_owed = 1 + companions

    def select(
        self, key: tuple, request_key: window_records.DecoderRequestKey
    ) -> bool:
        """The selection arrived: the destination now waits for the result.

        True when this selection is the one the destination sent, so its
        result may leave the unit that produced it.
        """
        record = self.by_window.get((key, request_key.pool))
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
        request_job = completion.request_job
        request_key = request_job.request_key
        key = (request_key.operation_id, request_key.window_id)
        pool = request_key.pool
        if self._takes_the_awaited_result(key, request_job):
            return True
        if self._has_a_newer_request(key, request_job):
            raise RuntimeError(
                f"strong result for window {key} arrived after a newer "
                "strong request took the destination's next result: "
                "nothing would consume this one"
            )
        if not self.destination_may_consume(key, pool):
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
        self, key: tuple, request_job: decoding_records.DecodeJob
    ) -> bool:
        """Whether the destination was waiting for this attempt's result.

        The selection names the attempt; it is over once each of the
        attempt's results has arrived.
        """
        lead_key = attempt_lead_of(request_job)
        record_key = (key, lead_key.pool)
        record = self.by_window.get(record_key)
        if record is None:
            return False
        if record.phase is not StrongPhase.WAITING_RESULT:
            return False
        if record.selected_request_key != lead_key:
            return False
        record.results_owed -= 1
        if record.results_owed > 0:
            return True
        record.phase = StrongPhase.NONE
        record.selected_request_key = None
        record.companions_by_lead.pop(lead_key, None)
        self._drop_if_empty(record_key)
        return True

    def _has_a_newer_request(
        self, key: tuple, request_job: decoding_records.DecodeJob
    ) -> bool:
        """Whether a later attempt took the destination's next result."""
        lead_key = attempt_lead_of(request_job)
        for live in self.live_requests(key, lead_key.pool):
            if attempt_lead_of(live.request_job) != lead_key:
                return True
        return False

    def _open_attempt(
        self, window_key: tuple, job: decoding_records.DecodeJob
    ) -> None:
        """The job's decode is open on the edge its pool is the source of."""
        key = self._attempt_key(window_key, job.request_key.pool)
        record = self._record(key)
        record.open_weak_requests.add(job.request_key)

    def _close_attempt(
        self, window_key: tuple, job: decoding_records.DecodeJob
    ) -> None:
        """The job's decode will meet no verdict: its attempt is closed.

        Only a pool that is the source of an edge opened one.
        """
        pool = job.request_key.pool
        if pool not in self.target_by_source:
            return
        key = self._attempt_key(window_key, pool)
        record = self.by_window.get(key)
        if record is None:
            return
        record.open_weak_requests.discard(job.request_key)
        self._drop_if_empty(key)

    def _attempt_key(
        self, window_key: tuple, pool: window_records.PoolId
    ) -> tuple:
        """The record a source pool's decode of the window is kept in."""
        target = self.target_by_source.get(pool)
        return (window_key, target)

    def _record(self, key: tuple) -> WindowRequests:
        """The record of a window and target, made on its first request."""
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
            live_requests.extend(record.live)
        return live_requests

    # ---------------------------------------------------- observation

    def unsettled(self) -> dict:
        """Every state still holding a destination window, by its name."""
        unsettled = {}
        for key, record in self.by_window.items():
            window_key = key[0]
            for state in _states_of(record):
                keys = unsettled.setdefault(state, [])
                keys.append(window_key)
        for state in unsettled:
            unsettled[state] = sorted(unsettled[state])
        return unsettled


def attempt_lead_of(
    job: decoding_records.DecodeJob,
) -> window_records.DecoderRequestKey:
    """The request that names the job's attempt: its own, or its lead's.

    A forced-class companion reads its lead's input, so the input key
    it shares is the lead's request key.
    """
    if job.input_key is None:
        return job.request_key
    return job.input_key


def is_merged_batch(job: decoding_records.DecodeJob) -> bool:
    """A batch serves several strong requests and has no request itself."""
    return job.kind is decoding_records.DecodeJobKind.STRONG_BATCH


def _states_of(record: WindowRequests) -> list:
    """The names unsettled() reports one window's record under."""
    states = []
    if record.phase is not StrongPhase.NONE:
        states.append(record.phase.value)
    if record.live:
        states.append("still holding a strong request")
    if record.open_weak_requests:
        states.append("decoding with no outcome")
    return states


def _admit_companion(
    record: WindowRequests, lead_key: window_records.DecoderRequestKey
) -> None:
    """Count a companion beside its lead; a selected attempt owes it too."""
    count = record.companions_by_lead.get(lead_key, 0)
    record.companions_by_lead[lead_key] = count + 1
    if record.phase is StrongPhase.NONE:
        return
    if record.selected_request_key == lead_key:
        record.results_owed += 1


def _not_served_by(live_requests: list, service_job) -> list:
    """The live requests another decode than service_job serves."""
    remaining = []
    for live in live_requests:
        if live.service_job is not service_job:
            remaining.append(live)
    return remaining


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
