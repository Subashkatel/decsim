"""The strong decoder tier on the decoder side: which strong request is live
for which destination window, which destinations are waiting for a strong
result (or for its WSD selection to arrive), which strong results are held
because nobody asked for them yet, and which weak decodes are still open.
A run without strong escalation keeps this ledger too: only its weak side
(one open decode per destination) is ever touched, and every strong query
answers "nothing".

Invariants: a destination window has at most one unconsumed strong result;
a destination decodes weakly once at a time, so a strong result reaches the
attempt that asked; a strong result with no possible consumer is an error,
never dropped silently. The window side of the tier (escalation planning,
slabs, rephasing) lives with the window manager.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..message import DecodeJob, DecodeResult, DecoderRequestKey, StrongDecodeCompletion


@dataclass(frozen=True)
class LiveStrongRequest:
    """A strong request admitted for one destination, and the physical job
    serving it (itself, or a merged batch)."""

    request_job: DecodeJob
    service_job: DecodeJob


@dataclass(frozen=True)
class HeldStrongCompletion:
    """A finished strong result, per request, with the tick its decode ended."""

    request_job: DecodeJob
    completion: StrongDecodeCompletion
    decode_output_ticks: int


class StrongRequestLedger:
    def __init__(self):
        self.strong_needed = 0
        self.strong_cancelled = 0
        self._running: dict[tuple, LiveStrongRequest] = {}
        self._waiting_selection: dict[tuple, DecoderRequestKey] = {}
        self._waiting_result: dict[tuple, DecoderRequestKey] = {}
        self._completed: dict[tuple, HeldStrongCompletion] = {}
        self._unresolved_weak: set[tuple] = set()

    # ---- admission

    def admit_strong(self, job: DecodeJob, now: int) -> None:
        """Give one destination's next strong result to this request."""
        key = job.strong_decode_for
        if key in self._running or key in self._completed:
            raise RuntimeError(
                f"duplicate strong decode for window {key}: a destination "
                f"window has at most one unconsumed strong result")
        self._running[key] = LiveStrongRequest(job, job)
        job.request_admitted_ticks = now

    def admit_weak(self, job: DecodeJob, now: int) -> None:
        """Open one destination window's decode attempt."""
        key = (job.op_id, job.window_id)
        if key in self._unresolved_weak:
            raise RuntimeError(
                f"second weak decode for window {key} while the first is "
                f"unresolved: a destination window decodes once at a time, so "
                f"that its strong result reaches the attempt that asked")
        self._unresolved_weak.add(key)
        job.request_admitted_ticks = now

    def resolve_weak(self, key: tuple) -> None:
        """This destination's weak decode has produced its directive; the
        destination may be decoded again and stops keeping a strong result."""
        self._unresolved_weak.remove(key)

    # ---- queries

    def live(self, key: tuple):
        return self._running.get(key)

    def running_items(self):
        return self._running.items()

    def members_of(self, service_job: DecodeJob) -> list:
        """The request jobs one physical decode serves, in admission order."""
        return [live.request_job for live in self._running.values()
                if live.service_job is service_job]

    def carriers_for(self, key: tuple) -> tuple:
        """The live request or held completion that will carry the strong result for a destination."""
        return tuple(filter(None, (self._running.get(key), self._completed.get(key))))

    def destination_may_consume(self, key: tuple) -> bool:
        """The destination is waiting for a strong result now, or its weak
        decode is open and its directive may still ask for one."""
        return (key in self._waiting_result
                or key in self._waiting_selection
                or key in self._unresolved_weak)

    # ---- batching and service

    def register_batch(self, window_keys, request_jobs, batch: DecodeJob) -> None:
        for window_key, request_job in zip(window_keys, request_jobs):
            self._running[window_key] = LiveStrongRequest(request_job, batch)

    def finish_service(self, service_job: DecodeJob) -> None:
        for key, live in tuple(self._running.items()):
            if live.service_job is service_job:
                self._running.pop(key)

    def take_live(self, key: tuple):
        return self._running.pop(key, None)

    def take_held(self, key: tuple):
        return self._completed.pop(key, None)

    def has_survivors(self, service_job: DecodeJob) -> bool:
        return any(live.service_job is service_job for live in self._running.values())

    # ---- selection and completion

    def begin_selection(self, key: tuple, request_key: DecoderRequestKey) -> None:
        self.strong_needed += 1
        self._waiting_selection[key] = request_key

    def select(self, key: tuple, request_key: DecoderRequestKey):
        """WSD delivered the selection: the destination now waits for this
        request's result. Returns a completion held from before, or None."""
        if self._waiting_selection.get(key) != request_key:
            return None
        del self._waiting_selection[key]
        self._waiting_result[key] = request_key
        held = self._completed.get(key)
        if held is not None and held.completion.request_key == request_key:
            return self._completed.pop(key)
        return None

    def complete(self, held: HeldStrongCompletion) -> bool:
        """A strong result is ready. True when a destination consumes it now;
        otherwise it is held for the demand that is still coming."""
        completion = held.completion
        key = (completion.request_key.operation_id, completion.request_key.window_id)
        if self._waiting_result.get(key) == completion.request_key:
            del self._waiting_result[key]
            return True
        if key in self._running:
            raise RuntimeError(
                f"strong result for window {key} arrived after a newer strong "
                f"request took the destination's next result: nothing would "
                f"consume this one")
        if not self.destination_may_consume(key):
            raise RuntimeError(
                f"strong result for window {key} has no destination waiting "
                f"for it: the destination registered no strong demand and its "
                f"decode attempt has resolved")
        self._completed[key] = held
        return False

    def deliveries_for(self, service_job: DecodeJob, result: DecodeResult, now: int) -> tuple:
        """Per-request completions of one finished strong decode; a merged
        batch may only carry timing (no accuracy-bearing fields)."""
        requests = tuple(self.members_of(service_job))
        keys = tuple(request.strong_decode_for for request in requests)
        result_identity = (result.op_id, result.window_id)
        is_merged_delivery = len(keys) > 1 or any(key != result_identity for key in keys)
        if not is_merged_delivery:
            completion = StrongDecodeCompletion(requests[0].request_key, result)
            return (HeldStrongCompletion(requests[0], completion, now),)
        populated_field_names = [
            field_name for field_name in ("correction", "logical_observables",
                                          "soft_output", "boundary_defects",
                                          "boundary_data")
            if getattr(result, field_name) is not None]
        if populated_field_names:
            raise RuntimeError(
                "merged strong decode returned accuracy-bearing fields "
                f"({', '.join(populated_field_names)}); disable bulk_strong for "
                "accuracy-coupled switching")
        return tuple(HeldStrongCompletion(
            request,
            StrongDecodeCompletion(request.request_key,
                                   DecodeResult(op_id=key[0], window_id=key[1])),
            now,
        ) for key, request in zip(keys, requests))

    # ---- observation

    def unsettled(self) -> dict:
        return {
            state: sorted(keys) for state, keys in (
                ("waiting for a strong result", self._waiting_result),
                ("waiting for strong selection", self._waiting_selection),
                ("holding an unclaimed strong result", self._completed),
                ("still holding a strong request", self._running),
                ("decoding with no outcome", self._unresolved_weak),
            ) if keys
        }

    def snapshot(self, queue_memberships: dict) -> tuple:
        """Each physical strong job once in its authoritative phase: running,
        queued, or in_transit (admitted but not yet queued: in input transport
        or waiting for round credits). ``queue_memberships`` maps id(job) to
        the queued jobs with that identity."""
        from ..message import stable_identity_order_key
        jobs_by_identity = {}
        keys_by_identity = {}
        for destination_key, live in self._running.items():
            job = live.service_job
            identity = id(job)
            jobs_by_identity[identity] = job
            keys_by_identity.setdefault(identity, []).append(destination_key)
        records = []
        for identity, job in jobs_by_identity.items():
            queued_matches = queue_memberships.get(identity, ())
            if any(candidate is not job for candidate in queued_matches):
                raise RuntimeError("strong-work identity collision in ready queues")
            if len(queued_matches) > 1:
                raise RuntimeError("one strong job appears in multiple ready queues")
            if job.pool is not None:
                if queued_matches:
                    raise RuntimeError("running strong job also remains queued")
                phase = "running"
            elif queued_matches:
                phase = "queued"
            else:
                phase = "in_transit"
            destination_keys = tuple(sorted(
                keys_by_identity[identity], key=stable_identity_order_key))
            records.append((destination_keys, phase, job.n_rounds))
        return tuple(sorted(
            records,
            key=lambda record: (
                tuple(stable_identity_order_key(key) for key in record[0]),
                record[1], record[2])))
