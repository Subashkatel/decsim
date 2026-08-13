"""Focused contracts for the admitted decoder-input delivery seam."""

import pytest

from decsim.decoder_input import FixedLatencyDecoderInputTransfer
from decsim.decoder_manager import DecoderManager
from decsim.decoders import CodeRouter, PerRoundDecoder, PresetLatencyDecoder
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
from decsim.engine import Engine
from decsim.message import (
    DecodeJob,
    DecodeResult,
    DecoderRequestKey,
    DecoderTier,
    Operation,
    RetainedSyndromeFragment,
    RunSeedReservation,
)
from decsim.planner import FixedRounds
from decsim.protocols import DecoderInputTransfer
from decsim.run_spec import RunSpec
from decsim.schedulers import FifoScheduler


class _HoldingTransfer:
    """Test double that retains only the public delivery arguments."""

    def __init__(self):
        self.deliveries = []
        self.submitted_at_delivery = []

    def deliver(self, job, delay_ticks, receiver, *, on_materialized=None):
        self.submitted_at_delivery.append(job.submitted)
        self.deliveries.append((job, delay_ticks, receiver, on_materialized))

    def release(self, index=0):
        job, _delay_ticks, receiver, on_materialized = self.deliveries[index]
        if on_materialized is not None:
            on_materialized(job)
        receiver(job)


class _ImmediateRecordingTransfer:
    def __init__(self):
        self.deliveries = []

    def deliver(self, job, delay_ticks, receiver, *, on_materialized=None):
        self.deliveries.append((job, delay_ticks))
        if on_materialized is not None:
            on_materialized(job)
        receiver(job)


def _manager(engine, transfer=None, decoder=None):
    return DecoderManager(
        engine,
        router=CodeRouter(decoder or PresetLatencyDecoder(1.0)),
        scheduler=FifoScheduler(),
        decoder_input_transfer=transfer,
    )


def _payloads(round_count):
    """Retained fragments the default transfer materializes decoder-locally."""
    return [
        RetainedSyndromeFragment(
            operation_id=3,
            patch_id=0,
            round_index=round_index,
            bits=(0,),
            code=None,
            size_bits=1,
            fragment_index=0,
        )
        for round_index in range(1, round_count + 1)
    ]


def _weak_job(sequence=0):
    return DecodeJob(
        op_id=3,
        window_id=4,
        n_rounds=5,
        label=f"weak-{sequence}",
        payloads=_payloads(5),
        request_key=DecoderRequestKey(3, 4, DecoderTier.WEAK, sequence),
        request_created_ticks=0,
    )


def _strong_job(sequence):
    return DecodeJob(
        op_id=3,
        window_id=4,
        n_rounds=9,
        label=f"strong-{sequence}",
        hint="strong",
        attempt=1,
        strong_decode_for=(3, 4),
        payloads=_payloads(9),
        request_key=DecoderRequestKey(3, 4, DecoderTier.STRONG, sequence),
        request_created_ticks=0,
    )


def test_default_zero_delay_delivery_is_synchronous():
    engine = Engine(verbose=False)
    manager = _manager(engine)
    job = _weak_job()
    submitted_payloads = list(job.payloads)

    manager.enqueue(job)

    assert isinstance(manager.decoder_input_transfer, DecoderInputTransfer)
    assert type(manager.decoder_input_transfer) is (
        FixedLatencyDecoderInputTransfer)
    assert job.submitted
    assert job.pool == "default"
    assert job.ready_time == engine.now == 0
    assert [entry.round_index for entry in job.decoder_input.rounds] == [
        1, 2, 3, 4, 5]
    # Phase A compatibility: the decode-facing payload view is repopulated
    # from the decoder-local materialization, never the upstream objects.
    assert job.payloads == submitted_payloads
    assert all(
        payload is not original
        for payload, original in zip(job.payloads, submitted_payloads)
    )


def test_default_positive_delay_delivers_at_tick_exactly_once():
    engine = Engine(verbose=False)
    manager = _manager(engine, decoder=PresetLatencyDecoder(100.0))
    job = _strong_job(1)

    manager.enqueue(job, delay_ticks=7)
    assert job.pool is None
    assert [event.label for event in engine._event_queue] == [
        f"fixed-latency decoder input {job.label}"
    ]

    engine.run(until=6)
    assert job.pool is None
    engine.run(until=7)
    assert job.pool == "default"
    assert job.ready_time == 7
    assert sum("READY -> enqueue" in line for line in engine.log_lines) == 1


def test_direct_transfer_rejects_malformed_delay_before_scheduling():
    engine = Engine(verbose=False)
    transfer = FixedLatencyDecoderInputTransfer(engine)
    job = _weak_job()

    for malformed in (True, 1.0, "1"):
        with pytest.raises(TypeError, match="delay_ticks"):
            transfer.deliver(job, malformed, lambda _job: None)
    with pytest.raises(ValueError, match="nonnegative"):
        transfer.deliver(job, -1, lambda _job: None)
    assert engine._event_queue == []


def test_decoder_manager_uses_custom_transfer_without_internal_context():
    engine = Engine(verbose=False)
    transfer = _ImmediateRecordingTransfer()
    manager = _manager(engine, transfer=transfer)
    job = _weak_job()

    manager.enqueue(job, delay_ticks=0)

    assert manager.decoder_input_transfer is transfer
    assert transfer.deliveries == [(job, 0)]
    assert job.pool == "default"


def test_admission_precedes_transfer_and_rejects_held_duplicates():
    weak_engine = Engine(verbose=False)
    weak_transfer = _HoldingTransfer()
    weak_manager = _manager(weak_engine, transfer=weak_transfer)
    weak = _weak_job(1)

    weak_manager.enqueue(weak, delay_ticks=0)
    assert weak.submitted
    assert weak_transfer.submitted_at_delivery == [True]
    assert weak.pool is None
    duplicate = _weak_job(2)
    with pytest.raises(RuntimeError, match="second weak decode"):
        weak_manager.enqueue(duplicate, delay_ticks=0)
    assert not duplicate.submitted
    assert duplicate.request_admitted_ticks is None

    strong_engine = Engine(verbose=False)
    strong_transfer = _HoldingTransfer()
    strong_manager = _manager(strong_engine, transfer=strong_transfer)
    strong = _strong_job(3)

    strong_manager.enqueue(strong, delay_ticks=0)
    assert strong.submitted
    assert strong_transfer.submitted_at_delivery == [True]
    assert strong_manager.admitted_strong_work_snapshot() == (
        (((3, 4),), "in_transit", 9),
    )
    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        strong_manager.enqueue(_strong_job(4), delay_ticks=0)


class _TrackingDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self):
        self.started_labels = []

    def latency(self, job):
        self.started_labels.append(job.label)
        return 100

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id)


def test_cancelled_in_transit_strong_never_queues_and_replacement_starts():
    engine = Engine(verbose=False)
    transfer = _HoldingTransfer()
    decoder = _TrackingDecoder()
    manager = _manager(engine, transfer=transfer, decoder=decoder)
    manager.submit_decode(1, lambda: None, label="blocker")
    original = _strong_job(10)

    manager.enqueue(original, delay_ticks=0)
    manager.cancel_strong((3, 4))
    replacement = _strong_job(11)
    manager.enqueue(replacement, delay_ticks=0)

    transfer.release(0)  # stale delivery must not disturb the replacement
    assert manager.queued_total() == 0
    assert decoder.started_labels == ["blocker"]
    assert manager.admitted_strong_work_snapshot() == (
        (((3, 4),), "in_transit", 9),
    )

    transfer.release(1)
    assert manager.queue_for("default") == [replacement]

    engine.run(until=100)
    assert decoder.started_labels == ["blocker", "strong-11"]
    assert "strong-10" not in decoder.started_labels


class _SeedAwareRecordingTransfer:
    def __init__(self, engine):
        self.direct = FixedLatencyDecoderInputTransfer(engine)
        self.delivery_count = 0
        self.committed_seed = None
        self._pending = None

    def deliver(self, job, delay_ticks, receiver, *, on_materialized=None):
        self.delivery_count += 1
        self.direct.deliver(
            job, delay_ticks, receiver, on_materialized=on_materialized)

    def reserve_run_seed(self, seed):
        self._pending = RunSeedReservation("derived", seed, seed)
        return self._pending

    def commit_run_seed(self, reservation):
        assert reservation is self._pending
        self.committed_seed = reservation.prepared_state
        self._pending = None

    def cancel_run_seed(self, reservation):
        if reservation is self._pending:
            self._pending = None


def test_runspec_factory_is_used_and_simple_baseline_completes():
    built = {}

    def make_transfer(engine, links, buffering):
        built["engine"] = engine
        built["links"] = links
        built["buffering"] = buffering
        built["transfer"] = _SeedAwareRecordingTransfer(engine)
        return built["transfer"]

    completed = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        d=3,
        rounds_policy=FixedRounds(9),
        decoder=PerRoundDecoder(0.5),
        make_decoder_input_transfer=make_transfer,
        seed=17,
    ).build(verbose=False)

    transfer = built["transfer"]
    assert built["engine"] is completed.engine
    assert built["links"] is completed.window_manager.links
    assert built["buffering"] is not None
    assert completed.decoder_manager.decoder_input_transfer is transfer
    assert completed.window_manager.submit_fn.__self__ is completed.decoder_manager
    assert completed.result.terminal_status == "complete"
    assert transfer.delivery_count > 0
    assert transfer.committed_seed is not None


def test_delayed_weak_ready_time_is_delivery_tick():
    engine = Engine(verbose=False)
    manager = _manager(engine, decoder=PresetLatencyDecoder(100.0))
    job = _weak_job()

    manager.enqueue(job, delay_ticks=7)
    engine.run(until=7)

    assert job.ready_time == 7


def test_transfer_validation_rejects_noncallable_deliver():
    class BadTransfer:
        deliver = 3

    with pytest.raises(TypeError, match="DecoderInputTransfer"):
        _manager(Engine(verbose=False), transfer=BadTransfer())


def test_runspec_calls_falsey_factory_and_rejects_none_result():
    class FalseyFactory:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            return False

        def __call__(self, engine, links, buffering):
            self.calls += 1
            return FixedLatencyDecoderInputTransfer(engine)

    factory = FalseyFactory()
    completed = RunSpec(
        ops=[Operation(0, "memory", (0,))], d=3,
        rounds_policy=FixedRounds(9), decoder=PerRoundDecoder(0.5),
        scheduler=FifoScheduler(), make_decoder_input_transfer=factory,
    ).build(verbose=False)
    assert factory.calls == 1
    assert type(completed.decoder_manager.decoder_input_transfer) is\
        FixedLatencyDecoderInputTransfer

    with pytest.raises(TypeError, match="must return"):
        RunSpec(
            ops=[Operation(0, "memory", (0,))], d=3,
            rounds_policy=FixedRounds(9), decoder=PerRoundDecoder(0.5),
            scheduler=FifoScheduler(),
            make_decoder_input_transfer=lambda _engine, _links, _buffering: None,
        ).build(verbose=False)


def test_transfer_with_declared_engine_must_use_run_engine():
    manager_engine = Engine(verbose=False)
    transfer = FixedLatencyDecoderInputTransfer(Engine(verbose=False))

    with pytest.raises(ValueError, match="different engine"):
        _manager(manager_engine, transfer=transfer)


def test_transfer_cannot_deliver_same_job_twice_or_substitute_identity():
    class TwiceTransfer:
        def deliver(self, job, delay_ticks, receiver, *, on_materialized=None):
            receiver(job)
            receiver(job)

    manager = _manager(Engine(verbose=False), transfer=TwiceTransfer())
    with pytest.raises(RuntimeError, match="delivered .* twice"):
        manager.enqueue(_weak_job())
    assert manager.queued_total() == 0  # first delivery already dispatched

    class WrongJobTransfer:
        def deliver(self, job, delay_ticks, receiver, *, on_materialized=None):
            receiver(_weak_job(99))

    wrong_manager = _manager(
        Engine(verbose=False), transfer=WrongJobTransfer())
    with pytest.raises(RuntimeError, match="different job"):
        wrong_manager.enqueue(_weak_job())
    assert wrong_manager.queued_total() == 0


def test_cancelled_in_transit_strong_releases_late_materialized_local_input():
    engine = Engine(verbose=False)
    transfer = FixedLatencyDecoderInputTransfer(engine)
    manager = _manager(engine, transfer=transfer, decoder=PresetLatencyDecoder(1.0))
    job = _strong_job(30)

    manager.enqueue(job, delay_ticks=7)
    manager.cancel_strong((3, 4))
    assert transfer.local_memory.slots_in_use == 0
    replacement = _strong_job(32)
    manager.enqueue(replacement, delay_ticks=7)
    engine.run(until=7)

    assert transfer.local_memory.slots_in_use == 1
    manager.cancel_strong((3, 4))
    assert transfer.local_memory.slots_in_use == 0
    assert job.decoder_input is None
    assert manager.queued_total() == 0


def test_cancelled_running_strong_releases_decoder_local_input_immediately():
    engine = Engine(verbose=False)
    transfer = FixedLatencyDecoderInputTransfer(engine)
    manager = _manager(engine, transfer=transfer, decoder=PresetLatencyDecoder(1.0))
    job = _strong_job(31)

    manager.enqueue(job)
    assert transfer.local_memory.slots_in_use == 1
    manager.cancel_strong((3, 4))

    assert transfer.local_memory.slots_in_use == 0
    assert job.decoder_input is None
    engine.run()
    assert transfer.local_memory.slots_in_use == 0
