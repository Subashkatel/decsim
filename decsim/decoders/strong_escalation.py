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
never dropped silently.

The window side (StrongEscalation, below) plans the escalation: the strong
context window or the forward slab, the windows it absorbs, the rephased
suffix, and when the deferred strong job is submitted. Both sides live in
this file so the strong tier is one deletable feature; a run without it uses
NoStrongTier on the window side and an empty ledger on the decoder side.
"""

from __future__ import annotations

from dataclasses import dataclass

from copy import deepcopy
from dataclasses import replace
from enum import Enum, auto
from types import MappingProxyType
from typing import Optional

from ..links.links import LinkPath
from ..message import (DecodeJob, DecodeResult, DecoderRequestKey, DecoderTier,
                       LogicalContribution, Operation, SeamFaultOwner, StrongDecodeCompletion,
                       StrongRegionPlan, Window, WindowInfo, stable_identity_order_key)
from ..syndrome_buffer.syndrome_buffer import (CsdInput, PendingStrong, PotentialStrong,
                                               RephaseGuard)


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


# ---- the window side ------------------------------------------------------

@dataclass(frozen=True)
class _ResolvedStrongRegion:
    """One policy-selected region resolved against the live window graph."""

    plan: StrongRegionPlan
    absorbed_window_keys: tuple
    restart_window_key: Optional[tuple]
    restart_read_keys: tuple
    strong_fault_exclusion_ranges: tuple
    restart_fault_exclusion_ranges: Optional[tuple]
class _EscalationPhase(Enum):
    """The one readiness condition that can transfer a pending strong job."""

    WAITING_FAR_BOUNDARY = auto()
    WAITING_TERMINAL_DATA = auto()
@dataclass(frozen=True)
class _PendingEscalation:
    """All immutable state retained until one strong-job transfer."""

    key: tuple
    weak_job: DecodeJob
    label: str
    resolved_region: _ResolvedStrongRegion
    strong_window: Window
    strong_model: object
    wsd_arrival_ticks: Optional[int]
    phase: _EscalationPhase
    strong_request_key: DecoderRequestKey
    strong_request_created_ticks: int
class _EscalationRegistry:
    """Own pending escalations and their one exact readiness index."""

    def __init__(self) -> None:
        self._by_key: dict[tuple, _PendingEscalation] = {}
        self._by_far_boundary: dict[tuple, tuple] = {}
        self._by_terminal_operation: dict[object, tuple] = {}

    def register_far(
        self,
        pending: _PendingEscalation,
        far_boundary_key: tuple,
    ) -> None:
        self._register(
            pending,
            expected_phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
            readiness_index=self._by_far_boundary,
            readiness_key=far_boundary_key,
        )

    def register_terminal(
        self,
        pending: _PendingEscalation,
        operation_id,
    ) -> None:
        self._register(
            pending,
            expected_phase=_EscalationPhase.WAITING_TERMINAL_DATA,
            readiness_index=self._by_terminal_operation,
            readiness_key=operation_id,
        )

    def _register(
        self,
        pending: _PendingEscalation,
        *,
        expected_phase: _EscalationPhase,
        readiness_index: dict,
        readiness_key,
    ) -> None:
        if pending.phase is not expected_phase:
            raise RuntimeError(
                f"pending escalation {pending.key} has phase "
                f"{pending.phase.name}, expected {expected_phase.name}")
        if pending.key in self._by_key:
            raise RuntimeError(
                f"duplicate strong escalation for window {pending.key}: one "
                "switching event creates exactly one strong job")
        if readiness_key in readiness_index:
            raise RuntimeError(
                f"readiness index collision for {readiness_key}")
        self._by_key[pending.key] = pending
        readiness_index[readiness_key] = pending.key

    def update_wsd_arrival(
        self,
        expected: _PendingEscalation,
        wsd_arrival_ticks: int,
    ) -> _PendingEscalation:
        """Record validated pre-submission WSD timing without moving ownership."""
        if self._by_key.get(expected.key) is not expected:
            raise RuntimeError(
                f"stale escalation timing update for {expected.key}")
        if expected.wsd_arrival_ticks is not None:
            raise RuntimeError(f"duplicate WSD reservation for {expected.key}")
        updated = replace(expected, wsd_arrival_ticks=wsd_arrival_ticks)
        self._by_key[expected.key] = updated
        return updated

    def peek_key(self, key: tuple) -> Optional[_PendingEscalation]:
        return self._by_key.get(key)

    def peek_far(self, far_boundary_key: tuple) -> Optional[_PendingEscalation]:
        key = self._by_far_boundary.get(far_boundary_key)
        return None if key is None else self._by_key[key]

    def peek_terminal(self, operation_id) -> Optional[_PendingEscalation]:
        key = self._by_terminal_operation.get(operation_id)
        return None if key is None else self._by_key[key]

    def take_far(
        self,
        far_boundary_key: tuple,
        expected: _PendingEscalation,
    ) -> _PendingEscalation:
        return self._take(
            expected,
            expected_phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
            readiness_index=self._by_far_boundary,
            readiness_key=far_boundary_key,
        )

    def take_terminal(
        self,
        operation_id,
        expected: _PendingEscalation,
    ) -> _PendingEscalation:
        return self._take(
            expected,
            expected_phase=_EscalationPhase.WAITING_TERMINAL_DATA,
            readiness_index=self._by_terminal_operation,
            readiness_key=operation_id,
        )

    def _take(
        self,
        expected: _PendingEscalation,
        *,
        expected_phase: _EscalationPhase,
        readiness_index: dict,
        readiness_key,
    ) -> _PendingEscalation:
        if expected.phase is not expected_phase:
            raise RuntimeError(
                f"wrong-phase take for escalation {expected.key}")
        primary = self._by_key.get(expected.key)
        indexed_key = readiness_index.get(readiness_key)
        if primary is not expected or indexed_key != expected.key:
            raise RuntimeError(
                f"stale escalation take for readiness key {readiness_key}")
        del readiness_index[readiness_key]
        del self._by_key[expected.key]
        return expected

    def snapshot_phases(self):
        return MappingProxyType({
            key: pending.phase for key, pending in self._by_key.items()
        })

    def snapshot_work(self) -> tuple:
        """Return pending strong assignments without exposing live windows."""
        phase_names = {
            _EscalationPhase.WAITING_FAR_BOUNDARY: "waiting_far_boundary",
            _EscalationPhase.WAITING_TERMINAL_DATA: "waiting_terminal_data",
        }
        records = (
            (key, phase_names[pending.phase], pending.strong_window.n_rounds)
            for key, pending in self._by_key.items()
        )
        return tuple(sorted(
            records,
            key=lambda record: stable_identity_order_key(record[0]),
        ))


class NoStrongTier:
    """The window side of a run that never escalates: nothing is pending and
    every hook is a no-op. Baseline never asks it to build a strong job."""

    pending_escalations: dict = {}

    def after_arrival(self, op_id) -> None:
        pass

    def after_weak_commit(self, key) -> None:
        pass

    def pending_strong_work_snapshot(self) -> tuple:
        return ()


class StrongEscalation:
    """The window side of the strong tier: when a weak window escalates, how
    its strong job is built (two-sided context, or a forward slab that
    absorbs the windows it covers and restarts the weak chain past it), when
    the deferred job is submitted (the far weak boundary, or the terminal
    data), and how the strong result's selection reaches the decoder side.
    It works on the window manager's own tables (windows, holds, models,
    links), so it is constructed with the manager and reads its state
    directly; it is the one object outside the manager allowed to.
    """

    def __init__(self, window_manager, check_strong_route):
        self.wm = window_manager
        self._check_strong_route = check_strong_route
        self._escalations = _EscalationRegistry()

    def submit_strong(self, strong_job) -> None:
        """An escalation policy submitted a strong job alongside the weak one (run both at once)."""
        self._submit_strong_with_csd(strong_job)

    def after_weak_commit(self, key) -> None:
        """A weak commit at a slab's far boundary releases the deferred strong job."""
        pending = self._escalations.peek_far(key)
        if pending is not None:
            self._submit_far_strong(key, pending)

    def _submit_strong_with_csd(
        self,
        strong_job: DecodeJob,
        *,
        wsd_arrival_ticks: Optional[int] = None,
    ) -> None:
        """Queue a strong job now; its CSD input transfer is reserved at dispatch.

        The unit is assigned first, then the input moves into that unit's memory
        (CSD link); a serial job also waits for its WSD selection to arrive.
        """
        request_key = strong_job.request_key
        window_key = strong_job.strong_decode_for
        if self.wm.syndrome_buffer.has_hold(PotentialStrong(window_key)):
            self.wm._transfer_retention_hold(
                PotentialStrong(window_key), CsdInput(request_key))
        elif self.wm.syndrome_buffer.has_hold(PendingStrong(request_key)):
            self.wm._transfer_retention_hold(
                PendingStrong(request_key), CsdInput(request_key))
        else:
            packet_ids = tuple(dict.fromkeys(
                (fragment.operation_id, fragment.round_index)
                for fragment in strong_job.payloads))
            self.wm.syndrome_buffer.register_hold(CsdInput(request_key), packet_ids)
        self.wm._bind_decoder_input_hold(strong_job, CsdInput(request_key))
        payload_bits = self.wm._job_payload_bits(strong_job)

        def reserve_transfer() -> int:
            arrival = self.wm._link_arrival(LinkPath.CSD, strong_job, payload_bits=payload_bits)
            if wsd_arrival_ticks is not None:
                arrival = max(arrival, wsd_arrival_ticks)
            return arrival - self.wm.engine.now

        self.wm.submit_fn(strong_job, reserve_transfer)
    def prepare_strong_selection(
        self,
        weak_job: DecodeJob,
        strong_request_key: DecoderRequestKey,
        serial_strong_job: Optional[DecodeJob],
        *,
        deferred: bool,
    ) -> int:
        """Reserve real input legs and return WSD selection-delivery delay."""
        key = (weak_job.op_id, weak_job.window_id)
        pending = self._escalations.peek_key(key)
        if deferred:
            wsd_arrival_ticks = self.wm._link_arrival(
                LinkPath.WSD,
                weak_job,
                payload_bits=None,
                request_key=strong_request_key,
            )
            pending = self._escalations.update_wsd_arrival(
                pending, wsd_arrival_ticks)
            selection_delay = max(
                0, pending.wsd_arrival_ticks - self.wm.engine.now)
            if (pending.phase is _EscalationPhase.WAITING_TERMINAL_DATA
                    and self.wm.rounds_arrived[pending.key[0]] >=
                    pending.resolved_region.plan.context_hi):
                self._submit_terminal_strong(pending.key[0], pending)
            return selection_delay
        if serial_strong_job is not None:
            wsd_arrival_ticks = self.wm._link_arrival(
                LinkPath.WSD,
                weak_job,
                payload_bits=None,
                request_key=strong_request_key,
            )
            self._submit_strong_with_csd(
                serial_strong_job,
                wsd_arrival_ticks=wsd_arrival_ticks,
            )
            return wsd_arrival_ticks - self.wm.engine.now
        if pending is not None:
            raise RuntimeError("deferred pending request needs an explicit key")
        wsd_arrival_ticks = self.wm._link_arrival(
            LinkPath.WSD,
            weak_job,
            payload_bits=None,
            request_key=strong_request_key,
        )
        return wsd_arrival_ticks - self.wm.engine.now
    def make_strong_job(self, weak_job: DecodeJob, n_rounds: int,
                        label: str) -> DecodeJob:
        """Build the strong job for a weak one; a route back to the weak decoder fails now."""
        strong = self.make_strong_decode_job(weak_job, n_rounds, label)
        self.check_strong_route(weak_job, strong)
        return strong
    def check_strong_route(self, weak_job: DecodeJob, strong_job: DecodeJob) -> None:
        self._check_strong_route(weak_job, strong_job)
    def make_strong_decode_job(self, weak_job: DecodeJob, round_count: int,
                               label: str) -> DecodeJob:
        """Build the two-sided strong re-decode job for an escalated window."""
        key = (weak_job.op_id, weak_job.window_id)
        weak_window = self.wm.windows[key]
        op = self.wm._ops[weak_job.op_id]
        strong_window = self._strong_context_window(weak_window)
        model_round_count = self.wm._round_count_for_window(op.id, strong_window)
        left_exclusions = (
            ((1, strong_window.commit_lo - 1),)
            if strong_window.commit_lo > 1
            else ()
        )
        dem = self._build_strong_window_model(
            op,
            strong_window,
            model_round_count,
            left_exclusions,
        )
        request_key = self.wm._new_request_key(
            weak_job.op_id, weak_job.window_id, DecoderTier.STRONG)
        self.wm._stamp_first_round_tick(strong_window)
        return DecodeJob(
            op_id=weak_job.op_id, window_id=weak_job.window_id,
            n_rounds=round_count, ready_time=self.wm.engine.now,
            label=label, hint="strong",
            spatial_nodes=weak_job.spatial_nodes, code=weak_job.code,
            dem=dem, payloads=self.wm._assemble_payloads(strong_window),
            attempt=1, window=strong_window, strong_decode_for=key,
            request_key=request_key, request_created_ticks=self.wm.engine.now)
    def _strong_context_window(self, weak_window: Window) -> Window:
        buffer_lo, commit_lo, commit_hi, buffer_hi = \
            self.wm._strong_context_bounds(weak_window)
        strong_window = Window(
            op_id=weak_window.op_id, k=weak_window.k, commit_lo=commit_lo,
            commit_hi=commit_hi, buffer_hi=buffer_hi, buffer_lo=buffer_lo,
            n_rounds=buffer_hi - buffer_lo + 1)
        strong_window.boundary_in = weak_window.boundary_in
        return strong_window
    def defer_strong_escalation(self, weak_job: DecodeJob) -> DecoderRequestKey:
        """Lay out the forward slab, absorb the windows it covers, and hold
        the strong job until the restart window's weak commit
        (waiting_far_boundary) or, terminally, until every clamped slab
        round is stored (waiting_terminal_data). One strong job per
        escalation; duplicates raise."""
        key = (weak_job.op_id, weak_job.window_id)
        existing_contribution = self.wm.ledger.contributions.get(key)
        if (
            self._escalations.peek_key(key) is not None
            or (
                existing_contribution is not None
                and existing_contribution.ownership_kind == "strong_slab"
            )
        ):
            raise RuntimeError(
                f"duplicate strong escalation for window {key}: one switching "
                f"event creates exactly one strong job")
        if weak_job.strong_label is None:
            raise RuntimeError(
                f"double-window escalation {key} needs a declared strong label")
        strong_request_key = self.wm._new_request_key(
            weak_job.op_id, weak_job.window_id, DecoderTier.STRONG)
        strong_request_created_ticks = self.wm.engine.now
        weak_window = self.wm.windows[key]
        op_id, escalated_index = key
        round_count = self.wm._round_count_for_window(op_id, weak_window)
        later_windows = [
            self.wm.windows[(op_id, window_index)]
            for window_index in self.wm.op_windows[op_id]
            if window_index > escalated_index
        ]
        plan = self.wm.window_interaction.plan_strong_region(
            WindowInfo.from_window(weak_window),
            [WindowInfo.from_window(window) for window in later_windows],
            round_count,
        )
        crossing_windows = [
            window for window in later_windows
            if window.commit_lo <= plan.commit_hi < window.commit_hi
        ]
        if crossing_windows:
            self._defer_crossing_strong_escalation(
                weak_job, weak_window, later_windows, round_count, plan,
                crossing_windows[0], strong_request_key,
                strong_request_created_ticks)
            return strong_request_key
        resolved_region = self._resolve_strong_region_plan(
            key, weak_window, later_windows, round_count, plan)
        restart_key = resolved_region.restart_window_key
        if restart_key is None:
            readiness_collision = self._escalations.peek_terminal(op_id)
            readiness_key = op_id
        else:
            readiness_collision = self._escalations.peek_far(restart_key)
            readiness_key = restart_key
        if readiness_collision is not None:
            raise RuntimeError(
                f"readiness index collision for {readiness_key}")

        restart_model = None
        if restart_key is not None:
            proposed_restart = deepcopy(self.wm.windows[restart_key])
            proposed_restart.buffer_lo = plan.restart_buffer_lo
            restart_model = self._build_strong_window_model(
                self.wm._ops[op_id],
                proposed_restart,
                round_count,
                resolved_region.restart_fault_exclusion_ranges,
            )
        slab = Window(
            op_id=key[0], k=key[1],
            commit_lo=plan.commit_lo,
            commit_hi=plan.commit_hi,
            buffer_hi=plan.context_hi,
            buffer_lo=plan.context_lo,
            n_rounds=plan.context_hi - plan.context_lo + 1,
        )
        strong_model = self._build_strong_window_model(
            self.wm._ops[op_id], slab, round_count,
            resolved_region.strong_fault_exclusion_ranges)
        logical_candidate = self._strong_slab_ownership_candidate(
            key, resolved_region)
        guard = None
        if restart_key is not None:
            guard = RephaseGuard(strong_request_key)
            guarded = list(self.wm.syndrome_buffer.hold_round_identities(restart_key))
            guarded += list(resolved_region.restart_read_keys)
            guarded += list(self.wm.syndrome_buffer.hold_round_identities(PotentialStrong(key)))
            guarded += list(self.wm.syndrome_buffer.hold_round_identities(PotentialStrong(restart_key)))
            guarded += [(op_id, round_index) for round_index in
                        range(plan.context_lo, plan.context_hi + 1)]
            guarded += self.wm._strong_context_read_keys(
                proposed_restart, list(resolved_region.restart_read_keys))
            self.wm.syndrome_buffer.register_hold(guard, guarded)
        try:
            self.wm.ledger.contributions = logical_candidate
            phase = (
                _EscalationPhase.WAITING_TERMINAL_DATA
                if restart_key is None
                else _EscalationPhase.WAITING_FAR_BOUNDARY
            )
            pending = _PendingEscalation(
                key=key,
                weak_job=weak_job,
                label=weak_job.strong_label,
                resolved_region=resolved_region,
                strong_window=slab,
                strong_model=strong_model,
                wsd_arrival_ticks=None,
                phase=phase,
                strong_request_key=strong_request_key,
                strong_request_created_ticks=strong_request_created_ticks,
            )
            if restart_key is None:
                self._escalations.register_terminal(pending, op_id)
            else:
                self._escalations.register_far(pending, restart_key)
            self.wm._transfer_potential_to_pending(key, strong_request_key)
            self.wm.syndrome_buffer.replace_hold(
                PendingStrong(strong_request_key),
                [(op_id, round_index) for round_index in
                 range(plan.context_lo, plan.context_hi + 1)])
            for absorbed_key in resolved_region.absorbed_window_keys:
                self._absorb_window(
                    absorbed_key, restart_key,
                    PendingStrong(strong_request_key))
            readiness_description = (
                "terminal data"
                if restart_key is None
                else "the far-side weak boundary"
            )
            self.wm.engine.log(
                "DecoderCluster",
                f"{pending.label}: slab rounds {plan.commit_lo}-"
                f"{plan.commit_hi} assigned; weak chain skips "
                f"{len(resolved_region.absorbed_window_keys)} window(s); "
                f"strong start deferred until {readiness_description}",
            )
            if restart_key is not None:
                self._reslice_restart_window(
                    restart_key,
                    plan.restart_buffer_lo,
                    restart_model,
                    plan.commit_hi,
                    plan.restart_seam_fault_owner,
                )
                self.wm.check_window(restart_key)  # its absorbed dep is gone
            return strong_request_key
        finally:
            if guard is not None:
                self.wm._release_hold_if_live(guard)
    def _defer_crossing_strong_escalation(
        self, weak_job: DecodeJob, weak_window: Window, later_windows: list,
        round_count: int, plan: StrongRegionPlan, crossing_window: Window,
        strong_request_key: DecoderRequestKey,
        strong_request_created_ticks: int,
    ) -> None:
        """Atomically replace a non-aligned post-slab suffix before deferral."""
        key = weak_window.key
        op_id = key[0]
        if not (
            1 <= plan.context_lo <= plan.commit_lo
            <= weak_window.commit_lo <= weak_window.commit_hi
            <= plan.commit_hi <= plan.context_hi <= round_count
        ):
            raise RuntimeError(
                f"invalid strong-region bounds for {key}: context "
                f"{plan.context_lo}-{plan.context_hi}, commit "
                f"{plan.commit_lo}-{plan.commit_hi}, operation 1-{round_count}")
        if plan.commit_lo != weak_window.commit_lo:
            raise RuntimeError(
                f"strong-region commit for {key} must start at the "
                f"escalated window's commit start {weak_window.commit_lo}")
        if (
            plan.restart_buffer_lo is None
            or not 1 <= plan.restart_buffer_lo <= plan.commit_hi + 1
            or not isinstance(plan.restart_seam_fault_owner, SeamFaultOwner)
        ):
            raise RuntimeError(
                f"strong-region restart after {key} needs an exact buffer "
                f"start and seam owner")

        absorbed = tuple(
            window.key for window in later_windows
            if window.commit_hi <= plan.commit_hi
        )
        crossing_index = later_windows.index(crossing_window)
        reusable_keys = tuple(
            window.key for window in later_windows[crossing_index:]
        )
        residual_round_count = round_count - plan.commit_hi
        suffix_plan = self.wm.scheme.plan_operation(
            op_id,
            residual_round_count,
            commit_round_count=self.wm._code_geometry.commit_round_count,
            buffer_round_count=self.wm._code_geometry.buffer_round_count,
        )
        if not suffix_plan.windows:
            raise RuntimeError(
                f"crossing strong-region plan for {key} produced no suffix")
        if len(suffix_plan.windows) > len(reusable_keys):
            raise RuntimeError(
                f"crossing strong-region plan for {key} needs "
                f"{len(suffix_plan.windows)} stable keys but only "
                f"{len(reusable_keys)} remain")

        retained_keys = reusable_keys[:len(suffix_plan.windows)]
        retired_keys = reusable_keys[len(suffix_plan.windows):]
        replacement_windows = []
        for index, (window_key, geometry) in enumerate(zip(
            retained_keys, suffix_plan.windows,
        )):
            buffer_lo = geometry.buffer_lo + plan.commit_hi
            if index == 0:
                buffer_lo = plan.restart_buffer_lo
            commit_lo = geometry.commit_lo + plan.commit_hi
            commit_hi = geometry.commit_hi + plan.commit_hi
            buffer_hi = min(
                geometry.buffer_hi + plan.commit_hi,
                round_count,
            )
            if not (
                1 <= buffer_lo <= commit_lo <= commit_hi
                <= buffer_hi <= round_count
            ):
                raise RuntimeError(
                    f"invalid rephased suffix geometry for {window_key}: "
                    f"buffer {buffer_lo}-{buffer_hi}, commit "
                    f"{commit_lo}-{commit_hi}, operation 1-{round_count}")
            replacement_windows.append(Window(
                op_id=op_id,
                k=window_key[1],
                buffer_lo=buffer_lo,
                commit_lo=commit_lo,
                commit_hi=commit_hi,
                buffer_hi=buffer_hi,
                n_rounds=buffer_hi - buffer_lo + 1,
            ))
        if (
            replacement_windows[0].commit_lo != plan.commit_hi + 1
            or replacement_windows[-1].commit_hi != round_count
            or any(
                left.commit_hi + 1 != right.commit_lo
                for left, right in zip(
                    replacement_windows, replacement_windows[1:]
                )
            )
        ):
            raise RuntimeError(
                f"rephased suffix for {key} must tile rounds "
                f"{plan.commit_hi + 1}-{round_count} exactly")

        affected_keys = absorbed + reusable_keys
        if op_id in self.wm._finished_ops or op_id in self.wm.op_results:
            raise RuntimeError(
                f"cannot rephase suffix for completed operation {op_id}")
        historical_sets = (
            self.wm.committed_windows,
            self.wm.absorbed_windows,
            self.wm._pending_strong_windows,
        )
        for affected_key in affected_keys:
            window = self.wm.windows[affected_key]
            if window.queued or window.committed:
                raise RuntimeError(
                    f"cannot rephase window {affected_key}: decode lifecycle "
                    "already started")
            if any(affected_key in values for values in historical_sets):
                raise RuntimeError(
                    f"cannot rephase historical window {affected_key}")
            if (self.wm.courier.is_published(affected_key)
                    or affected_key in self.wm.ledger.contributions):
                raise RuntimeError(
                    f"cannot rephase window {affected_key} with published state")
            if self._escalations.peek_key(affected_key) is not None:
                raise RuntimeError(
                    f"cannot rephase pending escalation {affected_key}")
            readiness_owner = self._escalations.peek_far(affected_key)
            if readiness_owner is not None:
                raise RuntimeError(
                    f"cannot rephase readiness key {affected_key}: owned by "
                    f"pending escalation {readiness_owner.key}")
        if self.wm.courier.touches(affected_keys):
            raise RuntimeError(
                f"cannot rephase suffix for {key} after boundary delivery")

        serial_windows = [weak_window, *later_windows]
        for source, destination in zip(serial_windows, serial_windows[1:]):
            if source.dependents != [destination.key]:
                raise RuntimeError(
                    f"cannot rephase suffix with external or non-serial edge "
                    f"from {source.key}: {source.dependents}")
            if destination.deps != [source.key] or destination.deps_remaining != 1:
                raise RuntimeError(
                    f"cannot rephase suffix with released or non-serial edge "
                    f"into {destination.key}")
        if later_windows[-1].dependents:
            raise RuntimeError(
                f"cannot rephase suffix with external edge from "
                f"{later_windows[-1].key}")

        for index, window in enumerate(replacement_windows):
            predecessor_key = key if index == 0 else retained_keys[index - 1]
            window.deps = [predecessor_key]
            window.deps_remaining = 1
            if index + 1 < len(replacement_windows):
                window.dependents = [retained_keys[index + 1]]
            window.boundary_in = self.wm.window_interaction.initial_boundary_state(
                WindowInfo.from_window(window))

        left_exclusions = (
            ((1, plan.commit_lo - 1),) if plan.commit_lo > 1 else ()
        )
        if plan.restart_seam_fault_owner is SeamFaultOwner.STRONG_REGION:
            strong_exclusions = left_exclusions
            suffix_exclusions = ((1, plan.commit_hi),)
        else:
            strong_exclusions = left_exclusions + (
                (plan.commit_hi + 1, round_count),
            )
            suffix_exclusions = left_exclusions

        operation = self.wm._ops[op_id]
        suffix_models = []
        if self.wm.error_model_provider is not None:
            suffix_models = self.wm.error_model_provider.window_models_for_operation(
                operation,
                replacement_windows,
                round_count,
                fault_model_requirement=self.wm._fault_model_requirement(operation),
                fault_exclusion_ranges=suffix_exclusions,
                window_protocol=self.wm._protocol_by_operation[operation.id],
            )
            if suffix_models and len(suffix_models) != len(replacement_windows):
                raise RuntimeError(
                    f"device returned {len(suffix_models)} models for "
                    f"{len(replacement_windows)} rephased windows")
        slab = Window(
            op_id=op_id,
            k=key[1],
            commit_lo=plan.commit_lo,
            commit_hi=plan.commit_hi,
            buffer_lo=plan.context_lo,
            buffer_hi=plan.context_hi,
            n_rounds=plan.context_hi - plan.context_lo + 1,
        )
        strong_model = self._build_strong_window_model(
            operation, slab, round_count, strong_exclusions)
        restart_window = replacement_windows[0]
        restart_reads = self.wm._read_keys_for_bounds(
            op_id, restart_window.start_round, restart_window.buffer_hi,
            restart_window)
        resolved_region = _ResolvedStrongRegion(
            plan=plan,
            absorbed_window_keys=absorbed,
            restart_window_key=restart_window.key,
            restart_read_keys=tuple(restart_reads),
            strong_fault_exclusion_ranges=strong_exclusions,
            restart_fault_exclusion_ranges=suffix_exclusions,
        )
        if self._escalations.peek_far(restart_window.key) is not None:
            raise RuntimeError(
                f"readiness index collision for {restart_window.key}")
        logical_candidate = self._strong_slab_ownership_candidate(
            key, resolved_region)
        pending = _PendingEscalation(
            key=key,
            weak_job=weak_job,
            label=weak_job.strong_label,
            resolved_region=resolved_region,
            strong_window=slab,
            strong_model=strong_model,
            wsd_arrival_ticks=None,
            phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
            strong_request_key=strong_request_key,
            strong_request_created_ticks=strong_request_created_ticks,
        )

        absent = object()
        owner_tokens = list(dict.fromkeys(
            token for window_key in affected_keys
            for token in (window_key, PotentialStrong(window_key))
        ))
        strong_token = PotentialStrong(key)
        owner_tokens.append(strong_token)
        owner_snapshot = {
            token: self.wm.syndrome_buffer.hold_round_identities(token)
            if self.wm.syndrome_buffer.has_hold(token) else absent
            for token in owner_tokens
        }
        if owner_snapshot[strong_token] is absent:
            raise RuntimeError(f"strong owner for {key} is not live")
        replacement_memberships = {}
        for window in replacement_windows:
            weak_reads = self.wm._read_keys_for_bounds(
                op_id, window.start_round, window.buffer_hi, window)
            replacement_memberships[window.key] = weak_reads
            replacement_memberships[PotentialStrong(window.key)] = (
                weak_reads + self.wm._strong_context_read_keys(window, weak_reads)
            )
        pending_reads = [
            (op_id, round_index)
            for round_index in range(plan.context_lo, plan.context_hi + 1)
        ]
        replacement_memberships[strong_token] = pending_reads
        guarded_reads = list(dict.fromkeys(
            identity
            for memberships in (owner_snapshot, replacement_memberships)
            for reads in memberships.values()
            if reads is not absent
            for identity in reads
        ))
        self.wm._require_retained_payloads(
            guarded_reads, f"suffix rephase guard for {key}")

        guard = RephaseGuard(strong_request_key)
        windows_snapshot = dict(self.wm.windows)
        op_window_indices = self.wm.op_windows[op_id]
        op_window_snapshot = list(op_window_indices)
        window_count_snapshot = self.wm.window_count[op_id]
        total_windows_snapshot = self.wm.total_windows
        window_models_snapshot = dict(self.wm.window_models)
        committed_count_snapshot = self.wm._committed_per_op.get(op_id, absent)
        logical_snapshot = self.wm.ledger.contributions
        escalated_dependents = list(weak_window.dependents)

        self.wm.syndrome_buffer.register_hold(guard, guarded_reads)
        registered = False
        try:
            weak_window.dependents[:] = [restart_window.key]
            for absorbed_key in absorbed:
                absorbed_window = deepcopy(self.wm.windows[absorbed_key])
                absorbed_window.queued = True
                absorbed_window.committed = True
                absorbed_window.deps = []
                absorbed_window.dependents = []
                absorbed_window.deps_remaining = 0
                self.wm.windows[absorbed_key] = absorbed_window
            for window in replacement_windows:
                self.wm.windows[window.key] = window
            for retired_key in retired_keys:
                self.wm.windows.pop(retired_key)
                self.wm.window_models.pop(retired_key, None)
            retired_indices = {retired_key[1] for retired_key in retired_keys}
            op_window_indices[:] = [
                index for index in op_window_indices
                if index not in retired_indices
            ]
            self.wm.window_count[op_id] -= len(retired_keys)
            self.wm.total_windows -= len(retired_keys)
            self.wm.committed_windows.update(absorbed)
            self.wm.absorbed_windows.update(absorbed)
            self.wm._committed_per_op[op_id] = (
                self.wm._committed_per_op.get(op_id, 0) + len(absorbed)
            )
            for affected_key in affected_keys:
                self.wm.window_models.pop(affected_key, None)
            for model_key, model in zip(retained_keys, suffix_models):
                self.wm.window_models[model_key] = model
            self.wm.ledger.contributions = logical_candidate
            self._escalations.register_far(pending, restart_window.key)
            registered = True
            for token, old_reads in owner_snapshot.items():
                if old_reads is absent:
                    continue
                self.wm.syndrome_buffer.replace_hold(
                    token, replacement_memberships.get(token, ()))
        except Exception:
            for token, old_reads in owner_snapshot.items():
                if old_reads is not absent:
                    self.wm.syndrome_buffer.replace_hold(token, old_reads)
            weak_window.dependents[:] = escalated_dependents
            self.wm.windows.clear()
            self.wm.windows.update(windows_snapshot)
            op_window_indices[:] = op_window_snapshot
            self.wm.window_count[op_id] = window_count_snapshot
            self.wm.total_windows = total_windows_snapshot
            self.wm.window_models.clear()
            self.wm.window_models.update(window_models_snapshot)
            self.wm.committed_windows.difference_update(absorbed)
            self.wm.absorbed_windows.difference_update(absorbed)
            if committed_count_snapshot is absent:
                self.wm._committed_per_op.pop(op_id, None)
            else:
                self.wm._committed_per_op[op_id] = committed_count_snapshot
            self.wm.ledger.contributions = logical_snapshot
            if registered:
                self._escalations.take_far(restart_window.key, pending)
            self.wm._release_hold_if_live(guard)
            raise

        self.wm._transfer_potential_to_pending(key, strong_request_key)
        for token, old_reads in owner_snapshot.items():
            if (
                old_reads is not absent
                and token not in replacement_memberships
                and self.wm.syndrome_buffer.has_hold(token)
            ):
                self.wm.syndrome_buffer.release_hold(token)
        self.wm._release_hold_if_live(guard)

        self.wm.engine.log(
            "DecoderCluster",
            f"{pending.label}: slab rounds {plan.commit_lo}-"
            f"{plan.commit_hi} assigned; suffix rephased to "
            f"{len(replacement_windows)} window(s); strong start deferred "
            "until the far-side weak boundary",
        )
        self.wm.check_window(restart_window.key)
    def _strong_slab_ownership_candidate(
        self,
        key: tuple,
        resolved_region: _ResolvedStrongRegion,
    ) -> dict:
        """Prepare the complete logical-owner map without changing live state."""
        plan = resolved_region.plan
        replaced_owner_keys = {
            key, *resolved_region.absorbed_window_keys,
        }
        candidate = {
            owner_key: contribution
            for owner_key, contribution in self.wm.ledger.contributions.items()
            if owner_key not in replaced_owner_keys
        }
        for other_key, contribution in candidate.items():
            if other_key[0] != key[0]:
                continue
            if (
                contribution.commit_lo <= plan.commit_hi
                and plan.commit_lo <= contribution.commit_hi
            ):
                raise RuntimeError(
                    f"strong slab {key} extent {plan.commit_lo}-"
                    f"{plan.commit_hi} overlaps unabsorbed logical "
                    f"contribution {other_key} extent "
                    f"{contribution.commit_lo}-{contribution.commit_hi}")
        candidate[key] = LogicalContribution(
            owner_key=key,
            commit_lo=plan.commit_lo,
            commit_hi=plan.commit_hi,
            ownership_kind="strong_slab",
            logical_observables=None,
        )
        return candidate
    def _resolve_strong_region_plan(
        self, key: tuple, weak_window: Window, later_windows: list,
        round_count: int, plan,
    ) -> _ResolvedStrongRegion:
        if not (
            1 <= plan.context_lo <= plan.commit_lo
            <= weak_window.commit_lo
            <= weak_window.commit_hi
            <= plan.commit_hi <= plan.context_hi <= round_count
        ):
            raise RuntimeError(
                f"invalid strong-region bounds for {key}: context "
                f"{plan.context_lo}-{plan.context_hi}, commit "
                f"{plan.commit_lo}-{plan.commit_hi}, operation 1-{round_count}")
        if plan.commit_lo != weak_window.commit_lo:
            raise RuntimeError(
                f"strong-region commit for {key} must start at the "
                f"escalated window's commit start {weak_window.commit_lo}")
        absorbed = tuple(
            window.key for window in later_windows
            if window.commit_hi <= plan.commit_hi
        )
        crossing = [
            window for window in later_windows
            if window.commit_lo <= plan.commit_hi < window.commit_hi
        ]
        if crossing:
            window = crossing[0]
            raise RuntimeError(
                f"window {window.key} commits {window.commit_lo}-"
                f"{window.commit_hi} across the strong-region edge "
                f"{plan.commit_hi}")
        for absorbed_key in absorbed:
            absorbed_window = self.wm.windows[absorbed_key]
            if absorbed_window.queued or absorbed_window.committed:
                raise RuntimeError(
                    f"cannot absorb window {absorbed_key}: already "
                    f"{'queued' if absorbed_window.queued else 'committed'}")
        expected_restart = next(
            (window.key for window in later_windows
             if window.commit_lo > plan.commit_hi),
            None,
        )
        if expected_restart is None:
            restart_reads = []
            if (plan.restart_buffer_lo is not None
                    or plan.restart_seam_fault_owner is not None):
                raise RuntimeError(
                    f"terminal strong-region plan for {key} cannot define "
                    f"restart seam data")
        else:
            restart = self.wm.windows[expected_restart]
            if restart.commit_lo != plan.commit_hi + 1:
                raise RuntimeError(
                    f"strong-region plan for {key} ends at "
                    f"{plan.commit_hi}, but restart {expected_restart} "
                    f"starts at {restart.commit_lo}; committed regions must "
                    "tile without a gap")
            if (plan.restart_buffer_lo is None
                    or not 1 <= plan.restart_buffer_lo <= restart.commit_lo):
                raise RuntimeError(
                    f"strong-region restart {expected_restart} needs a "
                    f"buffer start in 1-{restart.commit_lo}")
            restart_reads = self.wm._read_keys_for_bounds(
                restart.op_id, plan.restart_buffer_lo, restart.buffer_hi,
                restart)

        left_exclusions = (
            ((1, plan.commit_lo - 1),) if plan.commit_lo > 1 else ()
        )
        strong_exclusions = left_exclusions
        restart_exclusions = None
        if expected_restart is not None:
            if (plan.restart_seam_fault_owner
                    is SeamFaultOwner.STRONG_REGION):
                restart_exclusions = ((1, plan.commit_hi),)
            else:
                strong_exclusions = left_exclusions + (
                    (plan.commit_hi + 1, round_count),
                )
                restart_exclusions = left_exclusions

        required_reads = [
            (weak_window.op_id, round_index)
            for round_index in range(plan.context_lo, plan.context_hi + 1)
        ]
        required_reads.extend(restart_reads)
        self.wm._require_retained_payloads(
            required_reads, f"strong-region plan for {key}")
        return _ResolvedStrongRegion(
            plan=plan,
            absorbed_window_keys=absorbed,
            restart_window_key=expected_restart,
            restart_read_keys=tuple(restart_reads),
            strong_fault_exclusion_ranges=strong_exclusions,
            restart_fault_exclusion_ranges=restart_exclusions,
        )
    def _build_strong_window_model(
        self, operation: Operation, window: Window, round_count: int,
        fault_exclusions: tuple,
    ):
        """Build through the historical or explicit multi-range device port."""
        if self.wm.error_model_provider is None:
            return None
        if len(fault_exclusions) <= 1:
            exclusion = fault_exclusions[0] if fault_exclusions else None
            return self.wm.error_model_provider.strong_window_model_for_operation(
                operation, window, round_count,
                fault_model_requirement=self.wm._fault_model_requirement(operation),
                exclude_faults_touching=exclusion,
            )
        builder = (
            self.wm.error_model_provider
            .strong_window_model_for_operation_with_exclusions
        )
        return builder(
            operation, window, round_count,
            fault_model_requirement=self.wm._fault_model_requirement(operation),
            fault_exclusion_ranges=fault_exclusions,
        )
    def _reslice_restart_window(
        self, restart_key: tuple, buffer_lo: int, model,
        slab_hi: int, seam_owner: SeamFaultOwner,
    ) -> None:
        """Install a restart model prepared before plan mutation."""
        restart = self.wm.windows[restart_key]
        restart.buffer_lo = buffer_lo
        restart.n_rounds = restart.buffer_hi - restart.buffer_lo + 1
        self.wm._replace_window_read_refs(restart_key, restart)
        if model is not None:
            self.wm.window_models[restart_key] = model
        self.wm.engine.log("DecoderCluster",
                        f"restart window {restart_key} re-sliced across slab "
                        f"edge {slab_hi} (reads rounds {restart.buffer_lo}-"
                        f"{restart.buffer_hi}; crossing faults owned by "
                        f"{seam_owner.name.lower()})")
    def _absorb_window(
        self, key: tuple, restart_key: Optional[tuple], replacement,
    ) -> None:
        """A slab-covered window is never weak-decoded: count it committed
        with no logical contribution and unhook the restart window."""
        window = self.wm.windows[key]
        if window.queued or window.committed:
            raise RuntimeError(f"cannot absorb window {key}: already "
                               f"{'queued' if window.queued else 'committed'}")
        window.queued = True                  # keeps check_window() away
        window.committed = True
        self.wm.committed_windows.add(key)
        self.wm._committed_per_op[key[0]] = self.wm._committed_per_op.get(key[0], 0) + 1
        self.wm.absorbed_windows.add(key)
        if restart_key is not None:
            restart = self.wm.windows[restart_key]
            if key in restart.deps:
                restart.deps.remove(key)
                restart.deps_remaining -= 1
            if restart_key in window.dependents:
                window.dependents.remove(restart_key)
        self.wm._release_hold_if_live(key)
        absorbed = PotentialStrong(key)
        needed = set(self.wm.syndrome_buffer.hold_round_identities(absorbed))
        replacements = set(self.wm.syndrome_buffer.hold_round_identities(replacement))
        if restart_key is not None:
            replacements.update(self.wm.syndrome_buffer.hold_round_identities(
                PotentialStrong(restart_key)))
        if not needed <= replacements:
            raise RuntimeError("absorption replacement does not cover packets")
        self.wm.syndrome_buffer.release_hold(absorbed)
        self.wm.engine.log("DecoderCluster",
                        f"window {key} absorbed into the strong slab "
                        f"(weak chain skips it)")
    def _build_pending_strong_job(
        self,
        pending: _PendingEscalation,
    ) -> DecodeJob:
        """Build a slab job after both boundary conditions are satisfied.

        The slab commits all r_strong rounds and reads one buffer of raw
        context per face, owning nothing that touches pre-slab rounds (see the
        seam formalism on Switching).
        """
        key = pending.key
        weak_job = pending.weak_job
        slab = pending.strong_window
        dem = pending.strong_model
        payloads = self.wm._assemble_payloads(slab)
        covered = {payload.round_index for payload in payloads}
        plan = pending.resolved_region.plan
        needed = set(range(plan.context_lo, plan.context_hi + 1))
        if covered != needed:
            raise RuntimeError(
                f"{pending.label}: slab submitted with rounds "
                f"{sorted(covered)} but it needs "
                f"{plan.context_lo}-{plan.context_hi}; a slab may "
                "only start once every required round is retained")
        self.wm._stamp_first_round_tick(slab)
        return DecodeJob(
            op_id=key[0], window_id=key[1],
            n_rounds=slab.n_rounds,
            ready_time=self.wm.engine.now,
            label=pending.label, hint="strong",
            spatial_nodes=weak_job.spatial_nodes, code=weak_job.code,
            dem=dem, payloads=payloads,
            attempt=1, window=slab, strong_decode_for=key,
            request_key=pending.strong_request_key,
            request_created_ticks=pending.strong_request_created_ticks)
    def _submit_far_strong(
        self,
        far_boundary_key: tuple,
        pending: _PendingEscalation,
    ) -> None:
        if pending.wsd_arrival_ticks is None:
            raise RuntimeError("far strong submission requires WSD reservation")
        strong_job = self._build_pending_strong_job(pending)
        self.check_strong_route(pending.weak_job, strong_job)
        self._escalations.take_far(far_boundary_key, pending)
        self._submit_strong_with_csd(
            strong_job,
            wsd_arrival_ticks=pending.wsd_arrival_ticks,
        )
        self.wm.engine.log(
            "DecoderCluster",
            f"{pending.label}: far-side weak boundary determined -> strong "
            "slab submitted",
        )
    def _submit_terminal_strong(
        self,
        operation_id,
        pending: _PendingEscalation,
    ) -> None:
        if pending.wsd_arrival_ticks is None:
            raise RuntimeError("terminal strong submission requires WSD reservation")
        strong_job = self._build_pending_strong_job(pending)
        self.check_strong_route(pending.weak_job, strong_job)
        self._escalations.take_terminal(operation_id, pending)
        self._submit_strong_with_csd(
            strong_job,
            wsd_arrival_ticks=pending.wsd_arrival_ticks,
        )
        self.wm.engine.log(
            "DecoderCluster",
            f"{pending.label}: terminal data complete -> strong slab submitted",
        )
    def after_arrival(self, op_id) -> None:
        """A round arrived: a terminal slab waits for its clamped tail rounds to be stored."""
        pending = self._escalations.peek_terminal(op_id)
        if pending is None:
            return
        if (
            self.wm.rounds_arrived[op_id]
            >= pending.resolved_region.plan.context_hi
        ):
            self._submit_terminal_strong(op_id, pending)
    @property
    def pending_escalations(self) -> dict:
        """Typed deferral phase per escalated window, for tests and metrics."""
        return {
            key: phase.name.lower()
            for key, phase in self._escalations.snapshot_phases().items()
        }
    def pending_strong_work_snapshot(self) -> tuple:
        """Snapshot strong slabs assigned but not yet admitted for service."""
        return self._escalations.snapshot_work()
