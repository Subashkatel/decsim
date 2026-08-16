from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import decsim
from decsim.decoder_input_store import (
    DecoderInputStoreCapacityExhaustion,
    DecoderInputStoreConfig,
    DecoderInputStoreOverflowPolicy,
    DecoderInputStoreStager,
    DecoderInputStoreUnsatisfiableDemand,
    count_decoder_input_round_demand,
    materialize_decoder_input,
)
from decsim.decoder_input_transfer import FixedLatencyDecoderInputTransfer
from decsim.decoder_manager import DecoderManager
from decsim.decoders import (
    PerRoundDecoder,
    SAMPLED_CONFIDENCE_SOURCE,
    SampledConfidenceDecoder,
    SwitchingRouter,
)
from decsim.frontends.circuit import three_cnot_circuit
from decsim.message import (
    DecodeJob,
    DecodeResult,
    DecoderRequestKey,
    DecoderTier,
    Operation,
    RetainedSyndromeFragment,
)
from decsim.metrics import DecoderInputStoreOccupancy, DecoderUtilization
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec
from decsim.schemes import SlidingTerminalPolicy, SlidingWindowScheme
from decsim.syndrome_buffer import SyndromeBufferingConfig
from decsim.syndrome_ingress import (
    IngressOverflowPolicy,
    SyndromeIngressOverflow,
    SyndromeIngressPolicy,
)
from decsim.switching import Switching
from decsim.views import decoder_input_store_view


class ManualEngine:
    """Run scheduled callbacks at explicitly advanced integer ticks."""

    def __init__(self) -> None:
        self.now = 0
        self.events: list[tuple[int, object, str]] = []

    def schedule(self, delay: int, action, label: str = "") -> None:
        self.events.append((self.now + delay, action, label))

    def advance(self, tick: int) -> None:
        while True:
            due = sorted(
                (event for event in self.events if event[0] <= tick),
                key=lambda event: event[0],
            )
            if not due:
                break
            event = due[0]
            self.events.remove(event)
            self.now = event[0]
            event[1]()
        self.now = tick

    def log(self, *args) -> None:
        return None


class FifoScheduler:
    """Select the first queued job."""

    def pop(self, queue):
        return queue.pop(0)


class TimingDecoder:
    """Return a timing-only result after a fixed latency."""

    def __init__(self, latency_ticks: int = 1) -> None:
        self.latency_ticks = latency_ticks

    def latency(self, job) -> int:
        return self.latency_ticks

    def decode(self, job) -> DecodeResult:
        return DecodeResult(job.op_id, job.window_id)


class OneDecoderRouter:
    """Route every request to one decoder."""

    def __init__(self, decoder=None) -> None:
        self.decoder = decoder or TimingDecoder()

    def route(self, job):
        return self.decoder


class PureDelayTransfer:
    """Deliver unchanged jobs at their exact scheduled ticks."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.in_flight = {}
        self.deliveries = []

    @staticmethod
    def _key(job):
        return job.request_key if job.request_key is not None else id(job)

    def deliver(self, job, delay_ticks, receiver) -> None:
        key = self._key(job)
        self.in_flight[key] = True

        def complete() -> None:
            if not self.in_flight.pop(key, False):
                return
            self.deliveries.append((self.engine.now, job))
            receiver(job)

        if delay_ticks == 0:
            complete()
        else:
            self.engine.schedule(delay_ticks, complete, "pure delay")

    def cancel(self, job) -> None:
        self.in_flight.pop(self._key(job), None)


def make_fragment(
    operation_id: object,
    round_index: int,
    fragment_index: int = 0,
    *,
    bits: tuple[int, ...] | None = None,
) -> RetainedSyndromeFragment:
    return RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_id=f"patch-{fragment_index}",
        round_index=round_index,
        bits=bits,
        code="surface",
        size_bits=None if bits is None else len(bits),
        fragment_index=fragment_index,
    )


def make_job(
    name: str,
    round_indices: tuple[int, ...],
    *,
    operation_id: int = 1,
    n_rounds: int | None = None,
    hint: str | None = None,
    run_sequence: int = 0,
    strong_decode_for: tuple[int, int] | None = None,
    bits: tuple[int, ...] | None = None,
) -> DecodeJob:
    tier = DecoderTier.STRONG if strong_decode_for is not None else DecoderTier.WEAK
    return DecodeJob(
        op_id=operation_id,
        window_id=run_sequence,
        n_rounds=999 if n_rounds is None else n_rounds,
        payloads=[
            make_fragment(operation_id, round_index, fragment_index, bits=bits)
            for fragment_index, round_index in enumerate(round_indices)
        ],
        label=name,
        hint=hint,
        strong_decode_for=strong_decode_for,
        request_key=DecoderRequestKey(
            operation_id, run_sequence, tier, run_sequence),
    )


def make_manager(
    engine,
    *,
    config=None,
    unit_pools=None,
    transfer=None,
    bulk_strong=False,
) -> DecoderManager:
    return DecoderManager(
        engine,
        router=OneDecoderRouter(),
        scheduler=FifoScheduler(),
        unit_pools=unit_pools,
        decoder_input_transfer=transfer,
        decoder_input_store=config,
        bulk_strong=bulk_strong,
    )


@pytest.mark.parametrize(
    ("name", "round_indices", "n_rounds", "bits", "expected"),
    [
        ("weak", (3, 3, 4), 80, (0,), 2),
        ("strong", (0, 1, 2, 3), 2, None, 4),
        ("clamped", (7,), 50, (1, 0), 1),
        ("timing", (), 12, None, 0),
    ],
)
def test_round_demand_counts_distinct_materialized_rounds_not_service_extent(
    name: str,
    round_indices: tuple[int, ...],
    n_rounds: int,
    bits: tuple[int, ...] | None,
    expected: int,
) -> None:
    """Weak, strong, clamped, and timing jobs charge their materialized rounds."""
    job = make_job(name, round_indices, n_rounds=n_rounds, bits=bits)

    demand = count_decoder_input_round_demand(job)

    assert demand == expected
    assert demand != n_rounds or expected == n_rounds
    assert len({(item.operation_id, item.round_index) for item in job.payloads}) == demand
    assert len(materialize_decoder_input(job).rounds) == demand


def test_stager_materializes_once_after_credits_fit_and_unwinds_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission materializes once after reserve and returns credits on failure."""
    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=DecoderInputStoreConfig({"weak": 2}),
        pool_names=("weak",),
    )
    job = make_job("job", (0, 1))
    materializations = 0
    original = __import__(
        "decsim.decoder_input_store", fromlist=["materialize_decoder_input"]
    ).materialize_decoder_input

    def counted(candidate):
        nonlocal materializations
        materializations += 1
        return original(candidate)

    monkeypatch.setattr(
        "decsim.decoder_input_store.materialize_decoder_input", counted
    )

    with pytest.raises(RuntimeError, match="continuation failed"):
        stager.admit(
            job,
            pool="weak",
            on_admitted=lambda admitted: (_ for _ in ()).throw(
                RuntimeError("continuation failed")
            ),
        )

    assert materializations == 1
    assert job.decoder_input is None
    assert stager.snapshot().per_pool_store[0].occupied_rounds == 0


def test_stall_uses_strict_fifo_and_retries_at_the_release_tick() -> None:
    """A released pool admits fitting FIFO heads at the same tick without skipping."""
    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=DecoderInputStoreConfig({"strong": 3}),
        pool_names=("strong",),
    )
    first = make_job("first", (0, 1), run_sequence=1)
    large_head = make_job("large", (2, 3), run_sequence=2)
    small_tail = make_job("small", (4,), run_sequence=3)
    admitted = []
    callback = lambda job: admitted.append((engine.now, job.label))
    stager.admit(first, pool="strong", on_admitted=callback)
    stager.admit(large_head, pool="strong", on_admitted=callback)
    stager.admit(small_tail, pool="strong", on_admitted=callback)

    before = stager.snapshot()
    assert before.waiting_jobs_by_pool == (("strong", 2),)
    assert admitted == [(0, "first")]
    engine.now = 7
    stager.release(first)
    stager.drain_admissible_requests()

    assert admitted == [(0, "first"), (7, "large"), (7, "small")]
    after = stager.snapshot()
    assert after.waiting_jobs_by_pool == (("strong", 0),)
    assert tuple(record.arrival_tick for record in after.stall_records) == (0, 0)
    assert tuple(record.admitted_tick for record in after.stall_records) == (7, 7)
    assert tuple(record.round_demand for record in after.stall_records) == (2, 1)


def test_new_small_arrival_cannot_bypass_an_existing_fifo_head() -> None:
    """A later fitting request waits behind an older head that cannot fit yet."""
    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=DecoderInputStoreConfig({"strong": 3}),
        pool_names=("strong",),
    )
    occupying = make_job("occupying", (0, 1), run_sequence=1)
    head = make_job("head", (2, 3), run_sequence=2)
    tail = make_job("tail", (4,), run_sequence=3)
    admitted = []
    callback = lambda job: admitted.append(job.label)

    stager.admit(occupying, pool="strong", on_admitted=callback)
    stager.admit(head, pool="strong", on_admitted=callback)
    stager.admit(tail, pool="strong", on_admitted=callback)

    assert admitted == ["occupying"]
    assert stager.snapshot().waiting_jobs_by_pool == (("strong", 2),)


def test_cancel_stalled_request_has_no_callback_credit_or_stall_record() -> None:
    """A cancelled waiter keeps its arrival event but adds no record, ticks, callback, or credit."""
    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=DecoderInputStoreConfig({"weak": 1}),
        pool_names=("weak",),
    )
    occupying = make_job("occupying", (0,), run_sequence=1)
    waiting = make_job("waiting", (1,), run_sequence=2)
    callbacks = []
    stager.admit(occupying, pool="weak", on_admitted=callbacks.append)
    stager.admit(waiting, pool="weak", on_admitted=callbacks.append)
    occupied_before = stager.snapshot().per_pool_store[0].occupied_rounds

    stager.cancel(waiting)
    stager.cancel(waiting)

    snapshot = stager.snapshot()
    assert callbacks == [occupying]
    assert snapshot.per_pool_store[0].occupied_rounds == occupied_before
    assert snapshot.waiting_jobs_by_pool == (("weak", 0),)
    assert snapshot.stall_events_by_pool == (("weak", 1),)
    assert snapshot.stall_ticks_by_pool == (("weak", 0),)
    assert snapshot.stall_records == ()


def test_fail_stop_reports_typed_enriched_error_and_pre_failure_snapshot() -> None:
    """Fail-stop overflow exposes exact pool demand, capacity, and frozen state."""
    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=DecoderInputStoreConfig(
            {"weak": 2}, DecoderInputStoreOverflowPolicy.FAIL_STOP
        ),
        pool_names=("weak",),
    )
    stager.admit(
        make_job("first", (0,), run_sequence=1),
        pool="weak",
        on_admitted=lambda job: None,
    )

    with pytest.raises(DecoderInputStoreCapacityExhaustion) as caught:
        stager.admit(
            make_job("second", (1, 2), run_sequence=2),
            pool="weak",
            on_admitted=lambda job: None,
        )

    error = caught.value
    assert (error.pool, error.requested_rounds, error.capacity_rounds) == (
        "weak", 2, 2)
    assert error.snapshot.occupied_rounds == 1
    with pytest.raises(FrozenInstanceError):
        error.snapshot.occupied_rounds = 0


def test_unsatisfiable_stager_demand_never_enters_a_wait_queue() -> None:
    """A request exceeding its whole pool budget raises instead of stalling forever."""
    stager = DecoderInputStoreStager(
        ManualEngine(),
        config=DecoderInputStoreConfig({"weak": 1}),
        pool_names=("weak",),
    )

    with pytest.raises(DecoderInputStoreUnsatisfiableDemand):
        stager.admit(
            make_job("too-large", (0, 1)),
            pool="weak",
            on_admitted=lambda job: None,
        )

    assert stager.snapshot().waiting_jobs_by_pool == (("weak", 0),)


def test_saturated_strong_pool_never_blocks_weak_admission() -> None:
    """Strong saturation leaves an independently budgeted weak pool progressing."""
    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=DecoderInputStoreConfig({"weak": 1, "strong": 1}),
        pool_names=("weak", "strong"),
    )
    callbacks = []
    strong_live = make_job("strong-live", (0,), operation_id=1, hint="strong", run_sequence=1)
    strong_wait = make_job("strong-wait", (1,), operation_id=2, hint="strong", run_sequence=2)
    weak = make_job("weak", (2,), operation_id=3, run_sequence=3)

    stager.admit(strong_live, pool="strong", on_admitted=callbacks.append)
    stager.admit(strong_wait, pool="strong", on_admitted=callbacks.append)
    stager.admit(weak, pool="weak", on_admitted=callbacks.append)

    assert [job.label for job in callbacks] == ["strong-live", "weak"]
    snapshot = stager.snapshot()
    stores = {row.pool: row for row in snapshot.per_pool_store}
    assert stores["strong"].occupied_rounds == 1
    assert stores["weak"].occupied_rounds == 1
    assert dict(snapshot.waiting_jobs_by_pool) == {"strong": 1, "weak": 0}


def test_transport_cancel_before_delivery_suppresses_receiver_idempotently() -> None:
    """Cancelling in-flight transport prevents delivery and repeated cancel is harmless."""
    engine = ManualEngine()
    transfer = FixedLatencyDecoderInputTransfer(engine)
    job = make_job("cancelled", (0,))
    received = []

    transfer.deliver(job, 5, received.append)
    transfer.cancel(job)
    transfer.cancel(job)
    engine.advance(5)

    assert received == []
    transfer.cancel(job)


def test_custom_pure_delay_transport_always_ends_at_the_common_stager() -> None:
    """A custom timing transport cannot bypass finite manager-owned storage."""
    engine = ManualEngine()
    transfer = PureDelayTransfer(engine)
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 1}),
        transfer=transfer,
    )
    manager.pool_free["default"] = 0
    first = make_job("first", (0,), run_sequence=1)
    second = make_job("second", (1,), run_sequence=2)

    manager.enqueue(first, delay_ticks=3)
    manager.enqueue(second, delay_ticks=3)
    assert manager.decoder_input_stager.snapshot().per_pool_store[0].occupied_rounds == 0
    engine.advance(3)

    snapshot = manager.decoder_input_stager.snapshot()
    assert transfer.deliveries == [(3, first), (3, second)]
    assert snapshot.per_pool_store[0].occupied_rounds == 1
    assert snapshot.waiting_jobs_by_pool == (("default", 1),)


def test_old_custom_materializing_transport_fails_loudly() -> None:
    """A legacy transport that materializes before staging is rejected by design."""
    class OldMaterializingTransfer(PureDelayTransfer):
        def deliver(self, job, delay_ticks, receiver) -> None:
            job.decoder_input = object()
            receiver(job)

    engine = ManualEngine()
    manager = make_manager(engine, transfer=OldMaterializingTransfer(engine))

    with pytest.raises(RuntimeError, match="materialized.*before storage admission"):
        manager.enqueue(make_job("legacy", (0,)))


def test_unset_custom_transport_preserves_exact_arrival_and_late_pool_selection() -> None:
    """Unset storage adds no delay and keeps custom transport behavior and late routing."""
    class MutableLanePolicy:
        pool = "default"

        def pool_for(self, job):
            return self.pool

    engine = ManualEngine()
    transfer = PureDelayTransfer(engine)
    lane = MutableLanePolicy()
    manager = DecoderManager(
        engine,
        router=OneDecoderRouter(),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1, "strong": 1},
        lane_policy=lane,
        decoder_input_transfer=transfer,
        decoder_input_store=None,
    )
    manager.pool_free = {"default": 0, "strong": 0}
    job = make_job("late-route", (0,), run_sequence=1)
    manager.enqueue(job, delay_ticks=4)
    lane.pool = "strong"
    engine.advance(4)

    assert transfer.deliveries == [(4, job)]
    assert job.ready_time == 4
    assert job.pool is None
    assert manager.pool_ready["strong"] == [job]
    assert manager.decoder_input_stager.snapshot().enabled is False


def test_finite_pool_choice_is_closed_before_transport_without_mutating_job_pool() -> None:
    """Finite storage closes its input pool early but leaves dispatch state unset."""
    class MutableLanePolicy:
        pool = "default"

        def pool_for(self, job):
            return self.pool

    engine = ManualEngine()
    transfer = PureDelayTransfer(engine)
    lane = MutableLanePolicy()
    manager = DecoderManager(
        engine,
        router=OneDecoderRouter(),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1, "strong": 1},
        lane_policy=lane,
        decoder_input_transfer=transfer,
        decoder_input_store=DecoderInputStoreConfig({"default": 1, "strong": 1}),
    )
    manager.pool_free = {"default": 0, "strong": 0}
    job = make_job("closed-route", (0,), run_sequence=1)
    manager.enqueue(job, delay_ticks=4)
    lane.pool = "strong"
    assert job.pool is None
    engine.advance(4)

    assert job.pool is None
    assert manager.ready == [job]
    assert manager.pool_ready["strong"] == []
    stores = {
        row.pool: row
        for row in manager.decoder_input_stager.snapshot().per_pool_store
    }
    assert stores["default"].occupied_rounds == 1
    assert stores["strong"].occupied_rounds == 0


def test_normalized_store_pool_keys_must_match_unit_pools_exactly() -> None:
    """Finite pool keys must equal the manager's normalized decoder unit pools."""
    engine = ManualEngine()
    with pytest.raises(ValueError, match="must be exactly"):
        make_manager(
            engine,
            config=DecoderInputStoreConfig({"default": 1, "extra": 1}),
            unit_pools={"default": 1, "strong": 1},
        )
    with pytest.raises(ValueError, match="must be exactly"):
        make_manager(
            engine,
            config=DecoderInputStoreConfig({"strong": 1}),
            unit_pools={"default": 1, "strong": 1},
        )


def test_nonreentrant_dispatch_drains_every_same_tick_credit_return() -> None:
    """Nested credit returns are consumed by one stable outer dispatch transition."""
    engine = ManualEngine()
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 1}),
    )
    manager.pool_free["default"] = 0
    jobs = [make_job(name, (index,), run_sequence=index) for index, name in enumerate(
        ("first", "second", "third"), 1
    )]
    admitted = []

    def admit_and_release(job) -> None:
        admitted.append((engine.now, job.label))
        manager.decoder_input_stager.release(job)
        manager.try_dispatch()

    manager.decoder_input_stager.admit(
        jobs[0], pool="default", on_admitted=lambda job: None
    )
    manager.decoder_input_stager.admit(
        jobs[1], pool="default", on_admitted=admit_and_release
    )
    manager.decoder_input_stager.admit(
        jobs[2], pool="default", on_admitted=admit_and_release
    )
    manager.decoder_input_stager.release(jobs[0])
    manager.try_dispatch()

    assert admitted == [(0, "second"), (0, "third")]
    snapshot = manager.decoder_input_stager.snapshot()
    assert snapshot.waiting_jobs_by_pool == (("default", 0),)
    assert snapshot.per_pool_store[0].occupied_rounds == 0
    assert not manager.decoder_input_stager.has_drainable_work()
    assert engine.events == []


@pytest.mark.parametrize("mode", ["completion", "partial_cancel", "full_cancel"])
def test_bulk_strong_members_return_exactly_one_credit_each(mode: str) -> None:
    """Bulk completion and cancellation return every original member's credits."""
    engine = ManualEngine()
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 1, "strong": 3}),
        unit_pools={"default": 1, "strong": 1},
        bulk_strong=True,
    )
    jobs = [
        make_job(
            f"strong-{index}",
            (index,),
            operation_id=index,
            hint="strong",
            run_sequence=index,
            strong_decode_for=(index, 0),
        )
        for index in range(1, 4)
    ]
    for job in jobs:
        manager._admit_strong_request(job)
        manager.decoder_input_stager.admit(
            job, pool="strong", on_admitted=lambda admitted: None
        )
    service_job = manager._merge_strong_batch(list(jobs))
    service_job.pool = "strong"
    manager.pool_free["strong"] = 0
    assert {
        row.pool: row.occupied_rounds
        for row in manager.decoder_input_stager.snapshot().per_pool_store
    }["strong"] == 3

    if mode == "completion":
        manager._release_service_decoder_inputs(service_job)
    elif mode == "partial_cancel":
        manager.cancel_strong((1, 0))
        assert manager.decoder_input_stager.snapshot().per_pool_store[1].occupied_rounds == 2
        manager._release_service_decoder_inputs(service_job)
    else:
        for key in ((1, 0), (2, 0), (3, 0)):
            manager.cancel_strong(key)

    stores = {
        row.pool: row
        for row in manager.decoder_input_stager.snapshot().per_pool_store
    }
    assert stores["strong"].occupied_rounds == 0
    assert stores["strong"].slots_in_use == 0
    assert stores["strong"].admissions == 3


def test_settlement_rejects_storage_leaks_then_accepts_all_budgets_returned() -> None:
    """Terminal settlement rejects live storage and accepts exact budget return."""
    manager = make_manager(
        ManualEngine(),
        config=DecoderInputStoreConfig({"default": 1}),
    )
    job = make_job("live", (0,))
    manager.decoder_input_stager.admit(
        job, pool="default", on_admitted=lambda admitted: None
    )

    with pytest.raises(RuntimeError, match="still holding decoder input"):
        manager.check_decode_work_settled()

    manager.decoder_input_stager.release(job)
    with pytest.raises(RuntimeError, match="undrained"):
        manager.check_decode_work_settled()
    manager.try_dispatch()
    manager.check_decode_work_settled()
    assert manager.decoder_input_stager.snapshot().per_pool_store[0].occupied_rounds == 0


def test_frozen_view_reports_current_peak_waiting_and_stall_facts() -> None:
    """The immutable view exposes current pool occupancy and completed waits."""
    engine = ManualEngine()
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 2}),
    )
    live = make_job("live", (0, 1), run_sequence=1)
    waiting = make_job("waiting", (2,), run_sequence=2)
    manager.decoder_input_stager.admit(
        live, pool="default", on_admitted=lambda job: None
    )
    manager.decoder_input_stager.admit(
        waiting, pool="default", on_admitted=lambda job: None
    )
    engine.now = 5
    manager.decoder_input_stager.release(live)
    manager.try_dispatch()

    view = decoder_input_store_view(manager)
    assert view.enabled is True
    assert view.per_pool[0].occupied_rounds == 1
    assert view.per_pool[0].peak_occupied_rounds == 2
    assert view.per_pool[0].waiting_jobs == 0
    assert view.per_pool[0].stall_events == 1
    assert view.per_pool[0].stall_ticks == 5
    assert view.stall_records[0].round_demand == 1
    with pytest.raises(FrozenInstanceError):
        view.per_pool[0].occupied_rounds = 0


def test_opt_in_metric_reports_current_peak_time_average_wait_and_records() -> None:
    """The opt-in observer reports exact occupancy integrals and request stalls."""
    engine = ManualEngine()
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 2}),
    )
    metric = DecoderInputStoreOccupancy(manager)
    live = make_job("live", (0, 1), run_sequence=1)
    waiting = make_job("waiting", (2,), run_sequence=2)
    manager.decoder_input_stager.admit(
        live, pool="default", on_admitted=lambda job: None
    )
    metric.observe(engine)
    engine.now = 5
    manager.decoder_input_stager.admit(
        waiting, pool="default", on_admitted=lambda job: None
    )
    metric.observe(engine)
    engine.now = 10
    manager.decoder_input_stager.release(live)
    manager.try_dispatch()
    metric.observe(engine)
    engine.now = 20
    manager.decoder_input_stager.release(waiting)
    manager.try_dispatch()
    metric.observe(engine)

    result = metric.result()
    pool = result["per_pool"]["default"]
    assert result["enabled"] is True
    assert result["observation_span_ticks"] == 20
    assert pool["occupied_rounds"] == 0
    assert pool["peak_occupied_rounds"] == 2
    assert pool["time_avg_occupied_rounds"] == pytest.approx(1.5)
    assert pool["waiting_jobs"] == 0
    assert pool["waiting_rounds"] == 0
    assert pool["slots_in_use"] == 0
    assert pool["peak_waiting_jobs"] == 1
    assert pool["time_avg_waiting_jobs"] == pytest.approx(0.25)
    assert result["aggregate"]["occupied_rounds"] == 0
    assert result["aggregate"]["waiting_jobs"] == 0
    assert result["aggregate"]["waiting_rounds"] == 0
    assert result["aggregate"]["peak_occupied_rounds"] == 2
    assert result["stall_records"] == [{
        "request_key": {
            "operation_id": {"kind": "integer", "value": "1", "items": None},
            "window_id": 2,
            "tier": "weak",
            "run_sequence": 2,
        },
        "pool": "default",
        "arrival_ticks": 5,
        "admitted_ticks": 10,
        "stall_ticks": 5,
        "round_demand": 1,
    }]


def test_unset_metric_is_empty_and_existing_public_schema_is_unchanged() -> None:
    """Unset storage reports disabled without adding finite-store observations."""
    manager = make_manager(ManualEngine(), config=None)
    metric = DecoderInputStoreOccupancy(manager)

    assert decoder_input_store_view(manager).per_pool == ()
    assert metric.result() == {
        "enabled": False,
        "observation_span_ticks": 0,
        "per_pool": {},
        "aggregate": {},
        "stall_records": [],
    }


def test_public_package_exports_only_the_new_metric() -> None:
    """The package exports the metric but keeps store configuration module-scoped."""
    assert decsim.DecoderInputStoreOccupancy is DecoderInputStoreOccupancy
    assert not hasattr(decsim, "DecoderInputStoreConfig")
    assert not hasattr(decsim, "DecoderInputStoreOverflowPolicy")


def test_removed_decoder_input_slots_keyword_fails_loudly() -> None:
    """The deleted request-slot configuration has no alias or silent conversion."""
    with pytest.raises(TypeError, match="decoder_input_slots"):
        SyndromeBufferingConfig(decoder_input_slots=2)
    assert not hasattr(SyndromeBufferingConfig(), "decoder_input_slots")


@pytest.mark.parametrize("store_config", [None, DecoderInputStoreConfig({"default": 2})])
def test_checked_custom_and_timing_materialization_survives_both_store_modes(
    store_config: DecoderInputStoreConfig | None,
) -> None:
    """Custom-model and timing-only inputs retain their checked materialization behavior."""
    class CustomModel:
        """Represent a decoder model with no row-layout contract."""

    engine = ManualEngine()
    stager = DecoderInputStoreStager(
        engine,
        config=store_config,
        pool_names=("default",),
    )
    custom_job = make_job("custom", (0,), run_sequence=1, bits=(1, 0))
    custom_job.dem = CustomModel()
    timing_job = make_job("timing", (1,), run_sequence=2, bits=None)
    admitted = []

    stager.admit(custom_job, pool="default", on_admitted=admitted.append)
    stager.admit(timing_job, pool="default", on_admitted=admitted.append)

    assert admitted == [custom_job, timing_job]
    assert custom_job.decoder_input.rounds[0].fragments[0].bits == (1, 0)
    assert timing_job.decoder_input.rounds[0].fragments[0].bits is None
    stager.release(custom_job)
    stager.release(timing_job)
    stager.drain_admissible_requests()
    assert stager.unsettled_storage() == ()


@pytest.mark.parametrize("store_config", [None, DecoderInputStoreConfig({"default": 2})])
def test_two_operation_model_divergence_rolls_back_in_both_store_modes(
    store_config: DecoderInputStoreConfig | None,
) -> None:
    """A second operation cannot occupy a checked first-operation model in either mode."""
    model = SimpleNamespace(
        detector_ids=(10,),
        defect_positions={10: (0, 0)},
    )
    job = make_job("divergent", (0,), operation_id=2, run_sequence=1, bits=(1,))
    job.op_id = 1
    job.dem = model
    stager = DecoderInputStoreStager(
        ManualEngine(),
        config=store_config,
        pool_names=("default",),
    )

    with pytest.raises(ValueError, match="round operation 2.*job operation 1"):
        stager.admit(job, pool="default", on_admitted=lambda admitted: None)

    snapshot = stager.snapshot()
    assert all(store.occupied_rounds == 0 for store in snapshot.per_pool_store)
    assert job.decoder_input is None


def test_unbounded_upstream_isolates_local_stall_and_completes_finitely() -> None:
    """Unbounded upstream storage isolates local round-credit stalls in a finite run."""
    operations = [
        Operation(id=index, name=f"operation {index}", qubits=(index,), patches=(index,))
        for index in range(1, 4)
    ]

    completed = RunSpec(
        ops=operations,
        d=3,
        rounds_policy=FixedRounds(3),
        decoder=PerRoundDecoder(tau_us=100),
        decoder_input_store=DecoderInputStoreConfig({"default": 3}),
        syndrome_buffering=SyndromeBufferingConfig(upstream_packet_slots=None),
    ).build()

    snapshot = completed.decoder_manager.decoder_input_stager.snapshot()
    assert completed.result.terminal_status == "complete"
    assert snapshot.stall_events_by_pool == (("default", 2),)
    assert snapshot.per_pool_store[0].occupied_rounds == 0
    assert snapshot.waiting_jobs_by_pool == (("default", 0),)


def test_finite_upstream_fail_stop_is_attributed_to_ingress_not_local_store() -> None:
    """Finite upstream fail-stop remains an ingress outcome rather than a local stall."""
    operations = [
        Operation(id=index, name=f"operation {index}", qubits=(index,), patches=(index,))
        for index in range(1, 4)
    ]

    with pytest.raises(SyndromeIngressOverflow) as caught:
        RunSpec(
            ops=operations,
            d=3,
            rounds_policy=FixedRounds(3),
            decoder=PerRoundDecoder(tau_us=100),
            decoder_input_store=DecoderInputStoreConfig({"default": 3}),
            syndrome_buffering=SyndromeBufferingConfig(upstream_packet_slots=3),
            syndrome_ingress_policy=SyndromeIngressPolicy(
                overflow=IngressOverflowPolicy.FAIL_STOP
            ),
        ).build()

    assert caught.value.status == "controller_ingress_overflow"


def test_finite_upstream_drop_is_attributed_without_false_local_admissions() -> None:
    """Finite upstream drop completes as an ingress outcome without fake local stalls."""
    operations = [
        Operation(id=index, name=f"operation {index}", qubits=(index,), patches=(index,))
        for index in range(1, 4)
    ]

    completed = RunSpec(
        ops=operations,
        d=3,
        rounds_policy=FixedRounds(3),
        decoder=PerRoundDecoder(tau_us=100),
        decoder_input_store=DecoderInputStoreConfig({"default": 3}),
        syndrome_buffering=SyndromeBufferingConfig(upstream_packet_slots=3),
        syndrome_ingress_policy=SyndromeIngressPolicy(
            overflow=IngressOverflowPolicy.DROP_ROUND
        ),
    ).build()

    snapshot = completed.decoder_manager.decoder_input_stager.snapshot()
    assert completed.result.terminal_status == "complete"
    assert snapshot.per_pool_store[0].admissions == 0
    assert snapshot.stall_records == ()


def test_default_smoke_metric_remains_exactly_the_frozen_baseline() -> None:
    """Unset storage preserves the frozen smoke timing and metric identity exactly."""
    def make_metrics(engine, window_manager, decoder_manager, execution_runtime, factory):
        return [DecoderUtilization(decoder_manager)]

    completed = RunSpec(
        ops=three_cnot_circuit(),
        decoder=PerRoundDecoder(tau_us=0.1),
        make_metrics=make_metrics,
        decoder_input_store=None,
    ).build()

    assert completed.result.terminal_status == "complete"
    assert completed.result.event_queue_empty is True
    assert completed.result.execution_workload_complete is True
    assert completed.result.metric_values() == {
        "decoder_utilization": {
            "observation_span_ticks": 16950000,
            "aggregate_busy_fraction": 0.10619469026548672,
            "aggregate_total_units": 1,
            "per_pool_busy_fraction": {"default": 0.10619469026548672},
            "per_pool_total_units": {"default": 1},
        }
    }
    assert "decoder_input_store_occupancy" not in completed.result.metric_values()


def test_unset_custom_pure_delay_run_matches_default_timing_and_results() -> None:
    """A pure-delay custom transport preserves default run timing and scientific results."""
    operation = Operation(id=1, name="timing", qubits=(0,), patches=(0,))
    shared = dict(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(3),
        decoder=PerRoundDecoder(tau_us=0.1),
        decoder_input_store=None,
    )
    built_in = RunSpec(**shared).build()
    custom = RunSpec(
        **shared,
        make_decoder_input_transfer=lambda engine, links, buffering: PureDelayTransfer(engine),
    ).build()

    assert custom.result == built_in.result
    assert custom.engine.now == built_in.engine.now
    assert custom.result.metric_results == ()


@pytest.mark.parametrize("store_config", [None, DecoderInputStoreConfig({"default": 1})])
def test_manager_cancel_before_transport_delivery_never_reaches_storage(
    store_config: DecoderInputStoreConfig | None,
) -> None:
    """Cancelling a strong request in transport suppresses delivery in both modes."""
    engine = ManualEngine()
    transfer = PureDelayTransfer(engine)
    manager = make_manager(engine, config=store_config, transfer=transfer)
    job = make_job(
        "strong",
        (0,),
        run_sequence=1,
        strong_decode_for=(1, 0),
    )

    manager.enqueue(job, delay_ticks=5)
    manager.cancel_strong((1, 0))
    engine.advance(5)

    assert transfer.deliveries == []
    assert manager.ready == []
    snapshot = manager.decoder_input_stager.snapshot()
    assert all(store.occupied_rounds == 0 for store in snapshot.per_pool_store)
    assert all(count == 0 for _, count in snapshot.waiting_jobs_by_pool)


def test_admitted_release_and_cancel_are_idempotent_exact_credit_returns() -> None:
    """Repeated release or cancellation cannot return one request's credits twice."""
    stager = DecoderInputStoreStager(
        ManualEngine(),
        config=DecoderInputStoreConfig({"default": 2}),
        pool_names=("default",),
    )
    released = make_job("released", (0,), run_sequence=1)
    cancelled = make_job("cancelled", (1,), run_sequence=2)
    stager.admit(released, pool="default", on_admitted=lambda job: None)
    stager.admit(cancelled, pool="default", on_admitted=lambda job: None)
    assert stager.snapshot().per_pool_store[0].occupied_rounds == 2

    stager.release(released)
    stager.release(released)
    stager.cancel(released)
    assert stager.snapshot().per_pool_store[0].occupied_rounds == 1
    stager.cancel(cancelled)
    stager.cancel(cancelled)
    stager.release(cancelled)
    assert stager.snapshot().per_pool_store[0].occupied_rounds == 0


def test_settlement_rejects_waiters_as_well_as_live_and_drainable_credits() -> None:
    """Terminal settlement names a pending FIFO until capacity returns and drains."""
    manager = make_manager(
        ManualEngine(),
        config=DecoderInputStoreConfig({"default": 1}),
    )
    live = make_job("live", (0,), run_sequence=1)
    waiting = make_job("waiting", (1,), run_sequence=2)
    manager.decoder_input_stager.admit(
        live, pool="default", on_admitted=lambda job: None
    )
    manager.decoder_input_stager.admit(
        waiting, pool="default", on_admitted=lambda job: None
    )

    with pytest.raises(RuntimeError, match="waiting for decoder-input round credits"):
        manager.check_decode_work_settled()

    manager.decoder_input_stager.cancel(waiting)
    manager.decoder_input_stager.release(live)
    manager.try_dispatch()
    manager.check_decode_work_settled()


def test_same_size_within_round_fragment_permutation_remains_outside_layout_check() -> None:
    """Same-size fragment identity permutations remain accepted within one model row."""
    model = SimpleNamespace(
        detector_ids=(10, 11),
        defect_positions={10: (0, 0), 11: (0, 1)},
    )
    first = make_fragment(1, 0, 0, bits=(1,))
    second = make_fragment(1, 0, 1, bits=(0,))
    job = make_job("permuted", (), n_rounds=1)
    job.payloads = [second, first]
    job.dem = model

    stager = DecoderInputStoreStager(
        ManualEngine(), config=None, pool_names=("default",)
    )
    stager.admit(job, pool=None, on_admitted=lambda admitted: None)

    assert job.decoder_input.rounds[0].fragments == (second, first)


def test_public_enqueued_external_job_returns_storage_credits_on_completion() -> None:
    """Every payload-bearing manager job returns its stored rounds when service completes."""
    engine = ManualEngine()
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 1}),
    )
    completed = []
    job = make_job("external", (0,), run_sequence=1)
    job.on_done = lambda: completed.append(engine.now)

    manager.enqueue(job)
    assert manager.decoder_input_stager.snapshot().per_pool_store[0].occupied_rounds == 1
    engine.advance(1)

    assert completed == [1]
    manager.check_decode_work_settled()
    assert manager.decoder_input_stager.snapshot().per_pool_store[0].occupied_rounds == 0


def test_same_tick_metric_peak_uses_the_store_exact_high_water() -> None:
    """A same-tick admit and release remains visible in the metric's per-pool peak."""
    engine = ManualEngine()
    manager = make_manager(
        engine,
        config=DecoderInputStoreConfig({"default": 3}),
    )
    metric = DecoderInputStoreOccupancy(manager)
    job = make_job("transient", (0, 1, 2), run_sequence=1)

    manager.decoder_input_stager.admit(
        job, pool="default", on_admitted=lambda admitted: None
    )
    manager.decoder_input_stager.release(job)
    manager.try_dispatch()
    metric.observe(engine)

    store = manager.decoder_input_stager.snapshot().per_pool_store[0]
    assert store.occupied_rounds == 0
    assert store.peak_occupied_rounds == 3
    assert metric.result()["per_pool"]["default"]["peak_occupied_rounds"] == 3


def test_every_finite_run_boundary_has_no_free_credit_with_a_fitting_fifo_head() -> None:
    """Every stable engine boundary drains any FIFO head that fits its pool's free credits."""
    class CreditHeadInvariant:
        """Check round-credit conservation at every stable engine boundary."""

        name = "decoder_input_credit_head_invariant"
        result_schema_version = 1

        def __init__(self, decoder_manager) -> None:
            self.decoder_manager = decoder_manager
            self.observations = 0
            self.boundaries_with_waiters = 0

        def observe(self, engine) -> None:
            self.observations += 1
            stager = self.decoder_manager.decoder_input_stager
            stores = {
                row.pool: row for row in stager.snapshot().per_pool_store
            }
            for pool, waiting in stager._waiting_by_pool.items():
                if not waiting:
                    continue
                self.boundaries_with_waiters += 1
                store = stores[pool]
                free_rounds = store.capacity_rounds - store.occupied_rounds
                assert free_rounds < waiting[0].round_demand

        def result(self) -> dict:
            return {
                "observations": self.observations,
                "boundaries_with_waiters": self.boundaries_with_waiters,
            }

    operations = [
        Operation(id=index, name=f"operation {index}", qubits=(index,), patches=(index,))
        for index in range(1, 4)
    ]

    def make_metrics(engine, window_manager, decoder_manager, execution_runtime, factory):
        return [CreditHeadInvariant(decoder_manager)]

    completed = RunSpec(
        ops=operations,
        d=3,
        rounds_policy=FixedRounds(3),
        decoder=PerRoundDecoder(tau_us=100),
        decoder_input_store=DecoderInputStoreConfig({"default": 3}),
        make_metrics=make_metrics,
    ).build()

    result = completed.result.metric_values()["decoder_input_credit_head_invariant"]
    assert completed.result.terminal_status == "complete"
    assert result["observations"] > 0
    assert result["boundaries_with_waiters"] > 0


def test_real_multi_patch_switching_storm_isolates_weak_and_strong_budgets() -> None:
    """A real multi-patch switching storm stalls strong inputs while weak work progresses."""
    operations = [
        Operation(id=index, name=f"patch {index}", qubits=(index,), patches=(index,))
        for index in range(1, 4)
    ]
    weak_decoder = SampledConfidenceDecoder(
        PerRoundDecoder(tau_us=0.1), escalation_probability=1.0
    )
    strong_decoder = PerRoundDecoder(tau_us=100)

    completed = RunSpec(
        ops=operations,
        d=3,
        rounds_policy=FixedRounds(3),
        scheme=SlidingWindowScheme(
            SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
        ),
        router=SwitchingRouter(weak_decoder, strong_decoder),
        strategy=Switching(
            0.5,
            SAMPLED_CONFIDENCE_SOURCE,
            run_both_at_once=True,
        ),
        unit_pools={"default": 1, "strong": 1},
        decoder_input_store=DecoderInputStoreConfig(
            {"default": 9, "strong": 3}
        ),
        seed=7,
    ).build()

    snapshot = completed.decoder_manager.decoder_input_stager.snapshot()
    stores = {row.pool: row for row in snapshot.per_pool_store}
    records = snapshot.stall_records
    assert completed.result.terminal_status == "complete"
    assert completed.result.execution_done_ticks == 3_300_000
    assert completed.result.fully_done_ticks == 2_706_450_000
    assert stores["default"].capacity_rounds == 9
    assert stores["default"].peak_occupied_rounds == 9
    assert stores["default"].admissions == 3
    assert stores["strong"].capacity_rounds == 3
    assert stores["strong"].peak_occupied_rounds == 3
    assert stores["strong"].admissions == 3
    assert dict(snapshot.stall_events_by_pool) == {"default": 0, "strong": 2}
    assert dict(snapshot.stall_ticks_by_pool) == {
        "default": 0,
        "strong": 2_700_000_000,
    }
    assert tuple(record.request_key.operation_id for record in records) == (2, 3)
    assert tuple(record.request_key.tier for record in records) == (
        DecoderTier.STRONG,
        DecoderTier.STRONG,
    )
    assert tuple(record.arrival_tick for record in records) == (
        5_450_000,
        5_450_000,
    )
    assert tuple(record.admitted_tick for record in records) == (
        905_450_000,
        1_805_450_000,
    )
    assert tuple(record.round_demand for record in records) == (3, 3)
    assert all(store.occupied_rounds == 0 for store in stores.values())
    assert dict(snapshot.waiting_jobs_by_pool) == {"default": 0, "strong": 0}
    assert completed.decoder_manager.decoder_input_stager.unsettled_storage() == ()
