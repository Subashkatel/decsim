"""bulk_strong merge invariants and the strong-demand contract they share.

The bulk_strong paths of DecoderManager: merge-and-deliver, refusal of
accuracy-coupled merges, running-rounds accounting, and cancellation of one
key of a running merged batch, which keeps the batch alive for its siblings
and drops only the cancelled key from delivery.

The same destination-keyed state carries the strong-demand contract, which is
exercised here at both bulk_strong settings. A destination window owns at most
one unconsumed strong result, a live request or a held completion, and
admission refuses a second. Whether a result will be consumed is decided when
it completes, not when it is requested: it is delivered to a registered
demand, held while the destination's weak decode is still open to raise one,
and otherwise refused.

Keying by destination is what requires the other two halves tested here: one
open weak decode per destination, so the key identifies an attempt; and no
window left unfinal once the run is quiescent.
"""
import sys
import pathlib
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.decoders import CodeRouter, PerRoundDecoder, SwitchingRouter
from decsim.decoder_manager import DecoderManager, StrategyServicesImpl
from decsim.message import DecodeJob, DecodeResult, Operation
from decsim.planner import FixedRounds
from decsim.protocols import Directive, OutcomeDirective, Submission
from decsim.run_spec import RunSpec, simulate
from decsim.schedulers import EarliestDeadlineScheduler, FifoScheduler
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Baseline


class _NullStrategy:
    def on_decode_outcome(self, outcome, services):
        return None


class _AdoptTheWeakResult:
    """Every weak outcome is confident enough to keep, which halts the strong
    computation (arXiv:2510.25222 Sec. III A Step 3)."""

    def on_decode_outcome(self, outcome, services):
        if outcome.job.strong_decode_for is not None:
            return OutcomeDirective(Directive.FINALIZE_STRONG)
        return OutcomeDirective(Directive.FINALIZE)


class _EscalateThenAdopt:
    """Every weak outcome asks for a strong result; every strong one is
    adopted. The shape that makes an early strong completion holdable."""

    def on_decode_outcome(self, outcome, services):
        if outcome.job.strong_decode_for is not None:
            return OutcomeDirective(Directive.FINALIZE_STRONG)
        return OutcomeDirective(Directive.AWAIT_STRONG)


class _ImmediateSelectionServices:
    def __init__(self, manager):
        self.manager = manager

    def prepare_strong_selection(self, weak_job, serial_submission):
        if serial_submission is not None:
            self.manager.enqueue(serial_submission.job)
        return 0


def build(bulk_strong=True, decoder=None):
    eng = Engine(verbose=False)
    if decoder is None:
        decoder = PerRoundDecoder(tau_us=1.0)
    manager = DecoderManager(
        eng, router=CodeRouter(default=decoder),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1, "strong": 1}, bulk_strong=bulk_strong)
    manager.strategy = _NullStrategy()
    manager.services = _ImmediateSelectionServices(manager)
    results = []
    manager.on_strong_window_decoded = \
        lambda key, res: results.append((eng.now, key))
    return eng, manager, results


def strong_job(op, rounds, label=None):
    return DecodeJob(op_id=op, window_id=0, n_rounds=rounds,
                     strong_decode_for=(op, 0), hint="strong",
                     label=label or f"s{op}")


def weak_job(op, rounds, label=None):
    return DecodeJob(op_id=op, window_id=0, n_rounds=rounds,
                     label=label or f"w{op}")


class _RecordingDecoder(PerRoundDecoder):
    """Names every job that actually reaches a decoder."""

    def __init__(self, tau_us=1.0):
        super().__init__(tau_us=tau_us)
        self.decoded = []

    def decode(self, job):
        self.decoded.append(job.label)
        return super().decode(job)


def escalating_world(
    strategy,
    name="escalate-every-window",
    configure=None,
):
    """A real RunSpec whose windows all escalate through the strategy seam."""
    weak = PerRoundDecoder(tau_us=0.05)
    strong = PerRoundDecoder(tau_us=2.0)
    return RunSpec(
        ops=[Operation(88, name, (6,), clifford=True)],
        d=3, rounds_policy=FixedRounds(30), round_us=1.0,
        scheme=SlidingWindowScheme(), strategy=strategy,
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
        make_metrics=(
            None
            if configure is None
            else lambda engine, window_manager, decoder_manager, chip, factory: (
                configure(
                    engine, window_manager, decoder_manager, chip, factory
                ) or []
            )
        ),
    ).build(verbose=False)


def escalating_attempt(eng, manager, op, weak_rounds):
    """Start destination (op, 0)'s decode attempt with a weak decode slow
    enough to still be outstanding when a strong result lands."""
    manager.strategy = _EscalateThenAdopt()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(op, weak_rounds, f"w{op}-slow"))


def occupy_then_merge(eng, manager):
    """Blocker holds the strong unit; two 5-round strongs queue+merge."""
    manager.enqueue(strong_job(1, 10, "s-block"))
    for op in (2, 3):
        manager.enqueue(strong_job(op, 5))
    for key in [(1, 0), (2, 0), (3, 0)]:
        manager._windows_waiting_for_strong_result.add(key)


def test_bulk_merge_delivers_every_key_and_frees_units():
    eng, manager, results = build()
    occupy_then_merge(eng, manager)
    eng.run()
    assert [(t / us(1), k) for t, k in results] == \
        [(10.0, (1, 0)), (20.0, (2, 0)), (20.0, (3, 0))]
    assert manager.pool_free == {"default": 1, "strong": 1}
    assert manager.admitted_strong_work_snapshot() == ()
    assert not manager._windows_waiting_for_strong_result


def test_strong_handoff_changes_readiness_without_renewing_deadline():
    eng, manager, _ = build()
    job = strong_job(2, 5)
    job.deadline = us(99)

    manager.enqueue(job, delay_ticks=us(2))
    eng.run(until=us(2))

    assert job.ready_time == us(2)
    assert job.deadline == us(99)


def test_in_transit_strong_snapshot_never_consults_prospective_lane_policy():
    class RaisingLanePolicy:
        def pool_for(self, job):
            raise AssertionError("measurement must not route in-transit work")

    eng, manager, _ = build()
    manager.lane_policy = RaisingLanePolicy()
    job = strong_job(2, 5)
    job.hint = None

    manager.enqueue(job, delay_ticks=us(2))

    assert manager.admitted_strong_work_snapshot() == (
        (((2, 0),), "in_transit", 5),
    )


@pytest.mark.parametrize("deadlines", [(us(20), us(70)), (us(70), us(20))])
def test_timing_only_strong_batch_keeps_earliest_member_deadline(deadlines):
    _, manager, _ = build()
    first = strong_job(2, 5)
    second = strong_job(3, 5)
    first.deadline, second.deadline = deadlines

    batch = manager._merge_strong_batch([first, second])

    assert batch.deadline == min(deadlines)


def test_preserved_strong_deadlines_drive_edf_ahead_of_admission_order():
    eng = Engine(verbose=False)
    decoder = _RecordingDecoder(tau_us=1.0)
    manager = DecoderManager(
        eng,
        router=CodeRouter(default=decoder),
        scheduler=EarliestDeadlineScheduler(),
        unit_pools={"default": 1, "strong": 1},
        bulk_strong=False,
    )
    manager.strategy = _NullStrategy()
    manager.on_strong_window_decoded = lambda _key, _result: None
    blocker = strong_job(1, 10, "blocker")
    admitted_first = strong_job(2, 1, "later-deadline")
    urgent = strong_job(3, 1, "earlier-deadline")
    blocker.deadline = us(1)
    admitted_first.deadline = us(70)
    urgent.deadline = us(20)
    manager._windows_waiting_for_strong_result.update(
        {(1, 0), (2, 0), (3, 0)}
    )

    manager.enqueue(blocker)
    manager.enqueue(admitted_first)
    manager.enqueue(urgent)
    eng.run()

    assert decoder.decoded == ["blocker", "earlier-deadline", "later-deadline"]


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("correction", ("aggregate-correction",)),
        ("logical_observables", (1,)),
        ("soft_output", 0.25),
        ("boundary_defects", {6: [1]}),
        ("boundary_data", {"confidence": "aggregate"}),
    ],
)
def test_merged_result_rejects_accuracy_fields_before_lifecycle_mutation(
    field_name, field_value,
):
    class AccuracyBearingBatchDecoder(PerRoundDecoder):
        def decode(self, job):
            fields = {}
            if job.op_id == -1:
                fields[field_name] = field_value
            return DecodeResult(job.op_id, job.window_id, **fields)

    class RecordingStrategy(Baseline):
        def __init__(self):
            self.outcomes = []

        def on_decode_outcome(self, outcome, services):
            self.outcomes.append(outcome)

    eng, manager, results = build(decoder=AccuracyBearingBatchDecoder())
    strategy = RecordingStrategy()
    manager.strategy = strategy
    occupy_then_merge(eng, manager)

    with pytest.raises(
        RuntimeError,
        match="merged strong decode.*accuracy-bearing",
    ):
        eng.run()

    assert [key for _, key in results] == [(1, 0)]
    assert [(outcome.job.op_id, outcome.job.window_id)
            for outcome in strategy.outcomes] == [(1, 0)]
    assert manager.pool_free == {"default": 1, "strong": 0}
    assert manager.admitted_strong_work_snapshot() == (
        (((2, 0), (3, 0)), "running", 10),
    )
    assert set(manager._running_strong_decodes) == {(2, 0), (3, 0)}
    assert manager._windows_waiting_for_strong_result == {(2, 0), (3, 0)}
    merged_job = manager._running_strong_decodes[(2, 0)]
    assert not merged_job.completed


def test_timing_only_merged_result_is_reidentified_for_each_window():
    eng, manager, _ = build()
    delivered = []
    manager.on_strong_window_decoded = (
        lambda key, result: delivered.append((key, result))
    )
    occupy_then_merge(eng, manager)

    eng.run()

    assert [(key, (result.op_id, result.window_id))
            for key, result in delivered] == [
                ((1, 0), (1, 0)),
                ((2, 0), (2, 0)),
                ((3, 0), (3, 0)),
            ]
    assert len({id(result) for _, result in delivered}) == len(delivered)


def test_cancel_one_merged_key_keeps_sibling_result():
    """Cancelling one key of a running merged batch drops only that key from
    delivery: the siblings still get their results and the rounds accounting
    settles."""
    eng, manager, _ = build()
    delivered = []
    manager.on_strong_window_decoded = (
        lambda key, result: delivered.append((eng.now, key, result))
    )
    occupy_then_merge(eng, manager)
    eng.schedule(us(12), lambda: manager.cancel_strong((2, 0)))
    eng.run()
    keys = [key for _, key, _ in delivered]
    assert (3, 0) in keys, "sibling result lost on merged-key cancel"
    assert (2, 0) not in keys, "cancelled key must not deliver"
    survivor_result = next(
        result for _, key, result in delivered if key == (3, 0))
    assert (survivor_result.op_id, survivor_result.window_id) == (3, 0)
    assert manager.strong_cancelled == 1
    assert manager.admitted_strong_work_snapshot() == ()
    assert manager.pool_free == {"default": 1, "strong": 1}
    assert manager._windows_waiting_for_strong_result == {(2, 0)}


def test_cancel_all_merged_keys_cancels_the_batch_once():
    eng, manager, results = build()
    occupy_then_merge(eng, manager)
    eng.schedule(us(12), lambda: (manager.cancel_strong((2, 0)),
                                  manager.cancel_strong((3, 0))))
    eng.run()
    assert [k for _, k in results] == [(1, 0)]
    assert manager.strong_cancelled == 2
    assert manager.admitted_strong_work_snapshot() == ()
    assert manager.pool_free == {"default": 1, "strong": 1}


def test_bulk_strong_refuses_accuracy_coupled_merges():
    eng, manager, _ = build()
    manager.enqueue(strong_job(1, 10, "s-block"))
    j2 = strong_job(2, 5)
    j2.dem = object()                       # accuracy-coupled marker
    manager.enqueue(j2)
    manager.enqueue(strong_job(3, 5))
    for key in [(1, 0), (2, 0), (3, 0)]:
        manager._windows_waiting_for_strong_result.add(key)
    with pytest.raises(RuntimeError, match="bulk_strong only merges"):
        eng.run()


def test_running_snapshot_tracks_merged_batch_lifecycle():
    eng, manager, _ = build()
    occupy_then_merge(eng, manager)
    seen = []

    def watch():
        running_rounds = sum(
            rounds for _keys, phase, rounds
            in manager.admitted_strong_work_snapshot()
            if phase == "running"
        )
        seen.append((eng.now / us(1), running_rounds))
        if eng.now < us(25):
            eng.schedule(us(1), watch)
    eng.schedule(0, watch)
    eng.run()
    by_time = dict(seen)
    assert by_time[11.0] == 10              # merged batch (5+5) running
    assert by_time[21.0] == 0               # settled after completion


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_duplicate_active_strong_destination_is_rejected_before_any_mutation(
    bulk_strong,
):
    """A destination window has at most one live strong request, and a
    refused duplicate leaves nothing a later wait could consume."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.enqueue(strong_job(1, 10, "s-block"))
    live = strong_job(2, 5, "s-first")
    manager.enqueue(live)
    before = (dict(manager.pool_free), manager.admitted_strong_work_snapshot(),
              set(manager._running_strong_decodes), manager.queued_total(),
              dict(manager._completed_strong_results),
              len(manager.queue_log), len(eng.log_lines))

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        manager.enqueue(strong_job(2, 7, "s-duplicate"))

    assert (dict(manager.pool_free), manager.admitted_strong_work_snapshot(),
            set(manager._running_strong_decodes), manager.queued_total(),
            dict(manager._completed_strong_results),
            len(manager.queue_log), len(eng.log_lines)) == before
    # the live request still owns the destination: same job object, so
    # cancel_strong((2, 0)) still reaches the decode that is really running
    assert manager._running_strong_decodes[(2, 0)] is live

    manager._windows_waiting_for_strong_result.update({(1, 0), (2, 0)})
    eng.run()
    assert [key for _, key in results] == [(1, 0), (2, 0)]
    assert manager._completed_strong_results == {}

    manager._windows_waiting_for_strong_result.add((2, 0))   # a replay of the
    manager._apply_held_strong_result((2, 0))                # same key
    assert manager._windows_waiting_for_strong_result == {(2, 0)}
    assert [key for _, key in results] == [(1, 0), (2, 0)]


def test_public_strategy_duplicate_strong_submission_is_rejected_end_to_end():
    """Destination uniqueness is owned by the manager, not the strategy: a
    real DecodingStrategy that escalates one weak window twice is refused."""

    class DuplicateStrongStrategy(Baseline):
        bulk_strong = True

        def on_window_ready(self, window, weak_job, services):
            first = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"first-{weak_job.label}")
            second = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"second-{weak_job.label}")
            return [Submission(weak_job), Submission(first), Submission(second)]

        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            return OutcomeDirective(Directive.AWAIT_STRONG)

        def metrics(self):
            return {}

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        escalating_world(DuplicateStrongStrategy(), "duplicate-request")


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_early_strong_result_is_held_until_its_destination_asks(bulk_strong):
    """The case the hold map exists for: the strong decode finishes while the
    destination's weak decode is still running, and the demand that weak
    outcome raises is what consumes it."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    escalating_attempt(eng, manager, 2, weak_rounds=50)
    manager.enqueue(strong_job(2, 5, "s-early"))
    while_the_weak_still_runs = []
    eng.schedule(us(10), lambda: while_the_weak_still_runs.append(
        (set(manager._completed_strong_results), list(results))))

    eng.run()

    assert while_the_weak_still_runs == [({(2, 0)}, [])], \
        "the early strong result was not held for the running weak attempt"
    assert [key for _, key in results] == [(2, 0)]
    assert manager._completed_strong_results == {}
    assert manager._unresolved_weak_decodes == set()


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_strong_result_for_a_destination_that_already_adopted_one_is_refused(
    bulk_strong,
):
    """The duplicate separated in time. Admission cannot see it, because the
    first request is gone by then, so the completion is where it is caught:
    the destination adopted a result and has no weak decode outstanding to
    ask for another, so nothing would consume this one."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager._windows_waiting_for_strong_result.add((2, 0))
    manager.enqueue(strong_job(2, 5, "s-first"))
    eng.run()
    assert [key for _, key in results] == [(2, 0)]

    manager.enqueue(strong_job(2, 7, "s-late"))    # the destination owns none
    with pytest.raises(RuntimeError, match="no destination waiting"):
        eng.run()

    assert [key for _, key in results] == [(2, 0)]
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_time_separated_duplicate_is_refused_however_it_is_timed(bulk_strong):
    """Re-timing an illegal pair does not make it legal: the same two
    requests are refused whether the second is submitted straight away or
    handed over the weak->strong link to land after the first completed."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager._windows_waiting_for_strong_result.add((2, 0))
    manager.enqueue(strong_job(2, 5, "s-first"))
    log_lines_before = len(eng.log_lines)

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        manager.enqueue(strong_job(2, 7, "s-second"), delay_ticks=us(20))

    assert len(eng.log_lines) == log_lines_before, \
        "a refused request announced a handoff that never happened"
    eng.run()
    assert [key for _, key in results] == [(2, 0)]
    assert manager._completed_strong_results == {}
    assert manager.queued_total() == 0


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_second_request_while_a_completion_is_held_is_refused(bulk_strong):
    """A held completion is unconsumed state: a second request would replace
    it, and the destination's demand would then be released by a decode the
    first request produced nothing for."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    escalating_attempt(eng, manager, 2, weak_rounds=50)
    manager.enqueue(strong_job(2, 5, "s-first"))
    refused_against_the_held_result = []

    def submit_a_second_request():
        held = manager._completed_strong_results[(2, 0)]
        with pytest.raises(RuntimeError, match="duplicate strong decode"):
            manager.enqueue(strong_job(2, 7, "s-second"))
        refused_against_the_held_result.append(
            manager._completed_strong_results[(2, 0)] is held)

    eng.schedule(us(10), submit_a_second_request)
    eng.run()

    assert refused_against_the_held_result == [True]
    assert [key for _, key in results] == [(2, 0)]
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_destination_that_finalizes_across_the_link_leaves_nothing_held(
    bulk_strong,
):
    """A destination that keeps its weak result while the strong request is
    still crossing the weak->strong link cancels that request: it is never
    dispatched, so no completion outlives the destination's finality."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.enqueue(strong_job(4, 5, "s-link"), delay_ticks=us(3))
    eng.schedule(us(1), lambda: manager.cancel_strong((4, 0)))

    eng.run()

    assert results == []
    assert manager._completed_strong_results == {}
    assert manager._running_strong_decodes == {}
    assert manager.queued_total() == 0
    assert manager.strong_cancelled == 1
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_strong_result_cancelled_inside_the_completion_hook_is_refused(
    bulk_strong,
):
    """The destination's demand can disappear while its own result is being
    delivered; the result is then refused rather than parked."""

    class CancelInHookStrategy(Baseline):
        def on_decode_outcome(self, outcome, services):
            services.cancel_strong(outcome.job.strong_decode_for)
            return OutcomeDirective(Directive.FINALIZE_STRONG)

    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = CancelInHookStrategy()
    manager.services = StrategyServicesImpl(eng, None, manager)
    manager.enqueue(strong_job(2, 5, "s-only"))

    with pytest.raises(RuntimeError, match="no destination waiting"):
        eng.run()

    assert results == []
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_strong_result_displaced_by_a_new_request_in_the_hook_is_refused(
    bulk_strong,
):
    """Once the destination's next result belongs to a newer request, the
    completing one has no consumer left and is refused instead of replacing
    it in the hold map."""

    class ResubmitInHookStrategy(Baseline):
        def __init__(self, manager):
            self.manager = manager
            self.resubmitted = False

        def on_decode_outcome(self, outcome, services):
            if not self.resubmitted:
                self.resubmitted = True
                self.manager.enqueue(strong_job(2, 7, "s-newer"))
            return OutcomeDirective(Directive.FINALIZE_STRONG)

    eng, manager, results = build(bulk_strong=bulk_strong)
    escalating_attempt(eng, manager, 2, weak_rounds=50)
    manager.strategy = ResubmitInHookStrategy(manager)
    manager.enqueue(strong_job(2, 5, "s-first"))

    with pytest.raises(RuntimeError, match="took the destination's next"):
        eng.run()

    assert results == []
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_destination_that_adopted_a_result_escalates_again_on_its_next_attempt(
    bulk_strong,
):
    """Adopting a strong result resolves one attempt, not the window: the next
    attempt for the same destination may escalate again, and its strong result
    may again arrive before that attempt's weak decode finishes."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _EscalateThenAdopt()
    manager.on_window_decoded = lambda job, result: None
    manager._windows_waiting_for_strong_result.add((2, 0))
    manager.enqueue(strong_job(2, 5, "s-first"))
    eng.run()
    assert [key for _, key in results] == [(2, 0)]

    replayed_weak = DecodeJob(op_id=2, window_id=0, n_rounds=50,
                              ready_time=eng.now, deadline=eng.now,
                              label="w2 replay", attempt=1)
    manager.enqueue(replayed_weak)                # slow: the strong lands first
    manager.enqueue(strong_job(2, 5, "s-replay"))
    eng.run()

    assert [key for _, key in results] == [(2, 0), (2, 0)]
    assert results[-1][0] > results[0][0], "the replayed attempt never delivered"
    assert manager._completed_strong_results == {}
    assert manager._windows_waiting_for_strong_result == set()


def test_public_strategy_run_finalizes_with_nothing_held_end_to_end():
    """Real RunSpec, real strategy seam: every window escalates once, every
    strong result is adopted by its own destination, the operation finalizes
    with nothing held, and a later wait finds nothing to consume."""

    class EscalateEveryWindowStrategy(Baseline):
        bulk_strong = True

        def on_window_ready(self, window, weak_job, services):
            strong = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"strong-{weak_job.label}")
            return [Submission(weak_job), Submission(strong)]

        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            return OutcomeDirective(Directive.AWAIT_STRONG)

        def metrics(self):
            return {}

    delivered = []

    def configure(_engine, _window_manager, pool, _chip, _factory):
        commit_strong_result = pool.on_strong_window_decoded

        def record(key, result):
            delivered.append(key)
            commit_strong_result(key, result)

        pool.on_strong_window_decoded = record

    completed_run = escalating_world(
        EscalateEveryWindowStrategy(),
        configure=configure,
    )

    assert completed_run.window_manager._finished_ops == {88}
    assert len(delivered) == len(set(delivered)) > 1
    assert completed_run.decoder_manager._completed_strong_results == {}
    assert completed_run.decoder_manager._windows_waiting_for_strong_result == set()

    finalized_key = delivered[-1]
    completed_run.decoder_manager._windows_waiting_for_strong_result.add(finalized_key)
    completed_run.decoder_manager._apply_held_strong_result(finalized_key)
    assert completed_run.decoder_manager._windows_waiting_for_strong_result == {finalized_key}
    assert delivered.count(finalized_key) == 1, \
        "a wait was released after finality without a decode of its own"


def test_public_strategy_cannot_supply_a_strong_transport_delay():
    """The fabric exclusively owns strong transport; strategy delays are not a
    compatibility path around WSD/CSD reservation."""

    class DelayedDuplicateStrongStrategy(Baseline):
        bulk_strong = True

        def on_window_ready(self, window, weak_job, services):
            first = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"first-{weak_job.label}")
            second = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"second-{weak_job.label}")
            return [Submission(weak_job), Submission(first),
                    Submission(second, delay_ticks=us(400))]

        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            return OutcomeDirective(Directive.AWAIT_STRONG)

        def metrics(self):
            return {}

    with pytest.raises(ValueError, match="owned by the link fabric"):
        escalating_world(
            DelayedDuplicateStrongStrategy(),
            "duplicate-request",
        )


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_request_cancelled_across_the_link_is_replaced_by_one_decode(
    bulk_strong,
):
    """A cancel while the request is crossing the weak->strong link ends that
    request alone. The replacement admitted before the cancelled job's handoff
    lands owns the destination, and the cancelled job must not enqueue
    alongside it: two live decodes for one destination is the defect this
    contract exists to prevent."""
    decoder = _RecordingDecoder()
    eng, manager, results = build(bulk_strong=bulk_strong, decoder=decoder)
    manager._windows_waiting_for_strong_result.add((2, 0))
    manager.enqueue(strong_job(2, 5, "s-cancelled"), delay_ticks=us(20))

    def cancel_then_replace():
        manager.cancel_strong((2, 0))
        manager.enqueue(strong_job(2, 7, "s-replacement"), delay_ticks=us(20))

    eng.schedule(us(5), cancel_then_replace)
    eng.run()

    assert decoder.decoded == ["s-replacement"], \
        "the cancelled request enqueued alongside its replacement"
    assert [key for _, key in results] == [(2, 0)]
    assert manager.strong_cancelled == 1
    assert manager._running_strong_decodes == {}
    assert manager._completed_strong_results == {}
    assert manager.queued_total() == 0
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_destination_that_keeps_its_weak_result_discards_its_held_strong(
    bulk_strong,
):
    """Sec. III A Step 3: a confident destination adopts the weak result and
    halts the strong computation. A strong result that already landed is
    discarded with the request rather than left for a later demand."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _AdoptTheWeakResult()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(2, 50, "w2-slow"))
    manager.enqueue(strong_job(2, 5, "s-early"))
    while_the_weak_still_runs = []
    eng.schedule(us(10), lambda: while_the_weak_still_runs.append(
        set(manager._completed_strong_results)))

    eng.run()

    assert while_the_weak_still_runs == [{(2, 0)}], \
        "the early strong result was never held, so the discard is untested"
    assert results == []
    assert manager._completed_strong_results == {}
    assert manager._windows_waiting_for_strong_result == set()
    assert manager._unresolved_weak_decodes == set()
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_strong_result_for_a_destination_that_committed_without_one_is_refused(
    bulk_strong,
):
    """The other half of a resolved attempt. The weak decode ran and kept its
    own result, so a strong request submitted afterwards has no consumer: its
    result is refused rather than parked for a demand that will not come."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _AdoptTheWeakResult()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(2, 5, "w2"))
    eng.run()
    assert manager._unresolved_weak_decodes == set()

    manager.enqueue(strong_job(2, 5, "s-after-the-weak-committed"))
    with pytest.raises(RuntimeError, match="no destination waiting"):
        eng.run()

    assert results == []
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_later_attempt_cannot_consume_an_earlier_requests_result(bulk_strong):
    """A completion outlives nothing. An earlier request whose destination has
    no weak decode outstanding to ask for it is refused when it completes, so
    a later attempt's demand has nothing stale to be released by."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _EscalateThenAdopt()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(strong_job(2, 5, "earlier-request"))

    with pytest.raises(RuntimeError, match="no destination waiting"):
        eng.run()
    assert manager._completed_strong_results == {}

    manager.enqueue(weak_job(2, 5, "later-attempt"))
    eng.run()

    assert results == [], \
        "a later attempt was released by an earlier request's result"
    assert manager._windows_waiting_for_strong_result == {(2, 0)}


def test_result_for_a_destination_with_no_decode_attempt_is_refused():
    """Nothing is decoding for this destination and nothing ever asked, so its
    result is refused when it completes rather than retained for a consumer
    that never arrives."""
    eng, manager, results = build()
    manager.enqueue(strong_job(999, 5, "orphan"))

    with pytest.raises(RuntimeError, match="no destination waiting"):
        eng.run()

    assert results == []
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_cancelled_job_cannot_be_submitted_again(bulk_strong):
    """A DecodeJob carries the unit it occupies and the cancellation handle
    its destination holds, so a second submission would hand out both twice
    and its completion callback would return without releasing the unit."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager._windows_waiting_for_strong_result.add((2, 0))
    job = strong_job(2, 5, "reused")
    manager.enqueue(job)
    manager.cancel_strong((2, 0))
    before = (dict(manager.pool_free), set(manager._running_strong_decodes),
              manager.queued_total(), len(manager.queue_log))

    with pytest.raises(RuntimeError, match="has already been cancelled"):
        manager.enqueue(job)

    assert (dict(manager.pool_free), set(manager._running_strong_decodes),
            manager.queued_total(), len(manager.queue_log)) == before
    eng.run()
    assert results == []
    assert manager.pool_free == {"default": 1, "strong": 1}


def test_a_completed_job_cannot_be_submitted_again():
    """The same rule from the other end of the lifecycle: a job that already
    produced its result would be decoded and delivered twice."""
    eng, manager, results = build()
    manager._windows_waiting_for_strong_result.add((2, 0))
    job = strong_job(2, 5, "already-run")
    manager.enqueue(job)
    eng.run()
    assert [key for _, key in results] == [(2, 0)]

    with pytest.raises(RuntimeError, match="has already been completed"):
        manager.enqueue(job)

    eng.run()
    assert [key for _, key in results] == [(2, 0)]
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_dispatched_job_cannot_be_submitted_again(bulk_strong):
    """The middle of the lifecycle, which cancelled and completed do not
    cover: a job already occupying a unit would take a second one, and its
    single completion releases only one."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _AdoptTheWeakResult()
    manager.on_window_decoded = lambda job, result: None
    job = weak_job(2, 5, "dispatched")
    manager.enqueue(job)
    assert job.pool == "default" and not job.completed
    before = (dict(manager.pool_free), manager.queued_total(),
              len(manager.queue_log))

    with pytest.raises(RuntimeError, match="has already been admitted"):
        manager.enqueue(job)

    assert (dict(manager.pool_free), manager.queued_total(),
            len(manager.queue_log)) == before
    eng.run()
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_queued_job_cannot_be_submitted_again(bulk_strong):
    """The same rule one step earlier, where a unit-occupancy test would not
    reach: a job still in the ready queue holds a queue slot, so a second
    submission is dispatched twice once a unit frees."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _AdoptTheWeakResult()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(1, 10, "holds-the-unit"))
    job = weak_job(2, 5, "queued")
    manager.enqueue(job)
    assert job.pool is None and job in manager.ready
    before = (dict(manager.pool_free), manager.queued_total(),
              len(manager.queue_log))

    with pytest.raises(RuntimeError, match="has already been admitted"):
        manager.enqueue(job)

    assert (dict(manager.pool_free), manager.queued_total(),
            len(manager.queue_log)) == before
    eng.run()
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_public_strategy_listing_one_weak_submission_twice_is_refused(
    bulk_strong,
):
    """The single-use rule at the seam that carries it. Submission order is
    free (Sec. III A Step 1), so the same list a strategy may reorder is also
    the list that can name one job twice; the refusal is the documented one at
    submission, and the window occupies one queue slot or unit, never two."""

    class DuplicateWeakStrategy(Baseline):
        def __init__(self, bulk_strong):
            self.bulk_strong = bulk_strong

        def on_window_ready(self, window, weak_job, services):
            return [Submission(weak_job), Submission(weak_job)]

        def on_decode_outcome(self, outcome, services):
            return OutcomeDirective(Directive.FINALIZE)

        def metrics(self):
            return {}

    with pytest.raises(RuntimeError, match="has already been admitted"):
        escalating_world(DuplicateWeakStrategy(bulk_strong))


@pytest.mark.parametrize("strong_first", [True, False])
def test_public_strategy_may_list_its_submissions_in_either_order(strong_first):
    """arXiv:2510.25222 Sec. III A Step 1 feeds the weak and strong decoders
    simultaneously, so a Submission list carries no order: the pool admits the
    pair either way, to the same run."""

    class OrderedEscalationStrategy(Baseline):
        def on_window_ready(self, window, weak_job, services):
            strong = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"strong-{weak_job.label}")
            return ([Submission(strong), Submission(weak_job)] if strong_first
                    else [Submission(weak_job), Submission(strong)])

        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            return OutcomeDirective(Directive.AWAIT_STRONG)

        def metrics(self):
            return {}

    completed_run = escalating_world(OrderedEscalationStrategy())

    assert completed_run.window_manager._finished_ops == {88}
    assert completed_run.decoder_manager.strong_needed == completed_run.window_manager.window_count[88]
    assert completed_run.decoder_manager._completed_strong_results == {}
    assert completed_run.decoder_manager._windows_waiting_for_strong_result == set()


def test_public_strategy_may_cancel_and_replace_inside_its_outcome_hook():
    """Sec. III A Step 3 halts the speculative strong computation. A policy
    that then asks for a different strong decode instead of keeping the weak
    result is a legitimate variation: the cancel ends one request, and the
    replacement the same directive carries is admitted against a destination
    that directive has already recorded as waiting."""

    class CancelAndReplaceStrategy(Baseline):
        def on_window_ready(self, window, weak_job, services):
            speculative = services.make_strong_job(
                weak_job, weak_job.n_rounds, f"speculative-{weak_job.label}")
            return [Submission(weak_job), Submission(speculative)]

        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            services.cancel_strong((outcome.job.op_id, outcome.job.window_id))
            replacement = services.make_strong_job(
                outcome.job, outcome.job.n_rounds,
                f"replacement-{outcome.job.label}")
            return OutcomeDirective(Directive.AWAIT_STRONG,
                                    extra=Submission(replacement))

        def metrics(self):
            return {}

    delivered = []

    def configure(_engine, _window_manager, pool, _chip, _factory):
        commit_strong_result = pool.on_strong_window_decoded

        def record(key, result):
            delivered.append(key)
            commit_strong_result(key, result)

        pool.on_strong_window_decoded = record

    completed_run = escalating_world(
        CancelAndReplaceStrategy(),
        configure=configure,
    )

    window_count = completed_run.window_manager.window_count[88]
    assert completed_run.window_manager._finished_ops == {88}
    assert sorted(delivered) == sorted(set(delivered))
    assert len(delivered) == window_count
    assert completed_run.decoder_manager.strong_cancelled == window_count
    assert completed_run.decoder_manager._completed_strong_results == {}
    assert completed_run.decoder_manager._windows_waiting_for_strong_result == set()


# ------------------------------- one open weak decode per destination window

def pool_state(manager):
    """Every structure a submission could touch."""
    return (dict(manager.pool_free), list(manager.ready),
            {p: list(q) for p, q in manager.pool_ready.items()},
            list(manager.queue_log), dict(manager._running_strong_decodes),
            dict(manager._completed_strong_results),
            set(manager._windows_waiting_for_strong_result),
            set(manager._unresolved_weak_decodes),
            manager.strong_needed, manager.strong_cancelled)


def test_a_second_weak_decode_for_one_destination_is_refused():
    """Every strong structure is keyed by destination, so two open decodes of
    one window make the two attempts indistinguishable: one's result releases
    the other's demand and the loser waits forever.  The refusal is at
    submission and leaves the pool exactly as it found it.

    Weak jobs route to the default pool, which bulk_strong never touches, so
    this and the two below run once rather than at both settings.
    """
    eng, manager, results = build()
    manager.strategy = _EscalateThenAdopt()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(2, 40, "w2-first"))

    second = weak_job(2, 10, "w2-second")
    before = pool_state(manager)
    with pytest.raises(RuntimeError, match="second weak decode for window"):
        manager.enqueue(second)

    assert pool_state(manager) == before
    assert not second.submitted
    assert len(eng._event_queue) == 1                 # only the first decode


def test_a_destination_may_be_decoded_again_once_its_attempt_resolved():
    """The guard bounds concurrency, not the number of attempts: Eager replay
    decodes one window repeatedly and every re-decode must be admitted."""
    eng, manager, results = build()
    manager.strategy = _AdoptTheWeakResult()
    manager.on_window_decoded = lambda job, result: None
    for attempt in range(3):
        manager.enqueue(weak_job(2, 5, f"w2-attempt{attempt}"))
        eng.run()
        assert manager._unresolved_weak_decodes == set()
    assert manager.pool_free == {"default": 1, "strong": 1}


def test_the_destination_stays_open_until_its_outcome_hook_returns():
    """A decode produces its directive by returning from on_decode_outcome, so
    the destination is still open for the whole call and free again by the time
    the directive is applied and the window commits."""

    class ObserveTheReservation:
        def __init__(self, manager):
            self.manager = manager
            self.inside_the_hook = []

        def on_decode_outcome(self, outcome, services):
            key = (outcome.job.op_id, outcome.job.window_id)
            self.inside_the_hook.append(
                key in self.manager._unresolved_weak_decodes)
            return OutcomeDirective(Directive.FINALIZE)

    eng, manager, _ = build()
    strategy = ObserveTheReservation(manager)
    manager.strategy = strategy
    at_commit = []
    manager.on_window_decoded = lambda job, result: at_commit.append(
        (job.op_id, job.window_id) in manager._unresolved_weak_decodes)
    manager.enqueue(weak_job(2, 5, "w2"))
    eng.run()

    assert strategy.inside_the_hook == [True]
    assert at_commit == [False]
    manager.check_decode_work_settled()


def test_a_second_weak_decode_from_inside_the_outcome_hook_is_refused():
    """A strategy holding the pool may submit from its own outcome hook, which
    is the one position where the destination is open and no queue slot or unit
    is held for it. The refusal is the same there as anywhere else in the
    attempt, and leaves the refused job and the pool untouched."""

    class SubmitFromTheHook:
        def __init__(self, manager):
            self.manager = manager
            self.second = weak_job(2, 5, "w2-from-hook")
            self.hook_calls = 0
            self.refusals = []
            self.left_the_pool_alone = []

        def on_decode_outcome(self, outcome, services):
            self.hook_calls += 1
            if self.hook_calls == 1:
                before = pool_state(self.manager)
                try:
                    self.manager.enqueue(self.second)
                except RuntimeError as refusal:
                    self.refusals.append(str(refusal))
                self.left_the_pool_alone.append(
                    pool_state(self.manager) == before)
            return OutcomeDirective(Directive.FINALIZE)

    eng, manager, _ = build()
    strategy = SubmitFromTheHook(manager)
    manager.strategy = strategy
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(2, 5, "w2"))
    eng.run()

    assert len(strategy.refusals) == 1
    assert "second weak decode for window (2, 0)" in strategy.refusals[0]
    assert strategy.left_the_pool_alone == [True]
    assert not strategy.second.submitted
    assert strategy.hook_calls == 1
    manager.check_decode_work_settled()


def test_two_open_weak_decodes_cannot_coalesce_onto_one_strong_result():
    """What the refusal costs if it stops one hook too early.  Both attempts of
    one destination commit AWAIT_STRONG, their demands are the same key, and
    the single strong result clears it once: the later attempt waits forever
    while every settlement structure is empty, so no check downstream reports
    the window as unfinal."""

    class EscalateAndSubmitFromTheHook:
        def __init__(self, manager):
            self.manager = manager
            self.refusals = []
            self.weak_outcomes = 0

        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            self.weak_outcomes += 1
            if self.weak_outcomes > 1:
                return OutcomeDirective(Directive.AWAIT_STRONG)
            try:
                self.manager.enqueue(weak_job(2, 1, "w2-second"))
            except RuntimeError as refusal:
                self.refusals.append(str(refusal))
            return OutcomeDirective(
                Directive.AWAIT_STRONG,
                extra=Submission(strong_job(2, 40, "s2")))

    eng, manager, deliveries = build()
    strategy = EscalateAndSubmitFromTheHook(manager)
    manager.strategy = strategy
    commits = []
    manager.on_window_decoded = lambda job, result: commits.append(
        (job.label, job.awaiting_strong_result))
    manager.enqueue(weak_job(2, 1, "w2-first"))
    eng.run()

    assert len(strategy.refusals) == 1
    assert strategy.weak_outcomes == 1
    assert commits == [("w2-first", True)]
    assert [key for _, key in deliveries] == [(2, 0)]
    manager.check_decode_work_settled()


def test_public_strategy_listing_two_weak_decodes_for_one_window_is_refused():
    """The precondition at the seam that can break it.  A strategy builds the
    second job itself, so the single-use guard does not apply and only the
    per-destination rule stands between it and a stranded window."""

    class TwoWeakDecodesStrategy(Baseline):
        def on_window_ready(self, window, weak_job, services):
            twin = replace(weak_job, label=f"twin-{weak_job.label}")
            return [Submission(weak_job), Submission(twin)]

        def on_decode_outcome(self, outcome, services):
            return OutcomeDirective(Directive.FINALIZE)

        def metrics(self):
            return {}

    with pytest.raises(RuntimeError, match="second weak decode for window"):
        escalating_world(TwoWeakDecodesStrategy())


# ---------------------------------------- no window is left unfinal at the end

def test_a_run_that_leaves_a_window_waiting_for_a_strong_result_fails():
    """Sec. III A Step 4: an unconfident window's final estimate *is* the
    strong result, so a destination still waiting when the simulation has gone
    quiescent never became final.  No metric or view reports the pool's demand
    set, so without this check the run returns a logical accounting with that
    window's value silently missing."""

    class AwaitWithoutRequestingStrategy(Baseline):
        """Registers the demand and never submits anything to satisfy it."""

        def on_window_ready(self, window, weak_job, services):
            return [Submission(weak_job)]

        def on_decode_outcome(self, outcome, services):
            return OutcomeDirective(Directive.AWAIT_STRONG)

        def metrics(self):
            return {}

    weak = PerRoundDecoder(tau_us=0.05)
    run = RunSpec(
        ops=[Operation(88, "await-and-strand", (6,), clifford=True)],
        d=3, rounds_policy=FixedRounds(30), round_us=1.0,
        scheme=SlidingWindowScheme(),
        strategy=AwaitWithoutRequestingStrategy(),
        router=SwitchingRouter(weak, PerRoundDecoder(tau_us=2.0)),
        unit_pools={"default": 1, "strong": 1},
    )
    with pytest.raises(RuntimeError, match="waiting for a strong result"):
        simulate(run)


def test_the_settled_check_names_every_kind_of_unsettled_decode_work():
    """One message per structure, so a failing run says which window and what
    it was still doing rather than only that something was left over."""
    eng, manager, _ = build()
    assert manager.check_decode_work_settled() is None

    manager._windows_waiting_for_strong_result.add((1, 0))
    manager._completed_strong_results[(2, 0)] = DecodeResult(op_id=2,
                                                            window_id=0)
    manager._running_strong_decodes[(3, 0)] = strong_job(3, 5, "s3")
    manager._unresolved_weak_decodes.add((4, 0))

    with pytest.raises(RuntimeError) as refusal:
        manager.check_decode_work_settled()
    message = str(refusal.value)
    for state, key in (("waiting for a strong result", "(1, 0)"),
                       ("holding an unclaimed strong result", "(2, 0)"),
                       ("still holding a strong request", "(3, 0)"),
                       ("decoding with no outcome", "(4, 0)")):
        assert state in message and key in message


# ------------------------- where the single-use flag is set, and that it stays

@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_strong_request_refused_as_a_duplicate_may_be_submitted_later(
    bulk_strong,
):
    """A duplicate is refused for what the destination owns, not for anything
    about the job, so the refused object is still fresh: it is marked submitted
    only after admission accepts it, and it decodes normally once the
    destination's result has been consumed."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _EscalateThenAdopt()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(4, 2, "w4"))
    manager.enqueue(strong_job(4, 3, "first"))

    second = strong_job(4, 3, "second")
    before = pool_state(manager)
    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        manager.enqueue(second)

    assert pool_state(manager) == before
    assert not second.submitted

    eng.run()                                        # first is consumed
    assert [key for _, key in results] == [(4, 0)]

    manager.enqueue(weak_job(4, 2, "w4-again"))      # reopen the attempt
    manager.enqueue(second)                          # the same object, now legal
    assert manager._running_strong_decodes[(4, 0)] is second
    eng.run()
    assert [key for _, key in results] == [(4, 0), (4, 0)]
    assert manager.pool_free == {"default": 1, "strong": 1}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_request_cancelled_across_the_link_is_refused_and_replaced(
    bulk_strong,
):
    """The other direction. This job was admitted, so it is spent for good even
    though it never reached a unit and carries no cancelled flag; only a fresh
    replacement may take the destination's next result."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = _EscalateThenAdopt()
    manager.on_window_decoded = lambda job, result: None
    manager.enqueue(weak_job(5, 2, "w5"))
    crossing = strong_job(5, 3, "crossing")
    manager.enqueue(crossing, delay_ticks=30)

    manager.cancel_strong((5, 0))
    assert not crossing.cancelled, \
        "a job still crossing the link is dropped by identity, not by flag"
    assert crossing.submitted

    with pytest.raises(RuntimeError, match="has already been admitted"):
        manager.enqueue(crossing, delay_ticks=30)

    manager.enqueue(strong_job(5, 3, "replacement"), delay_ticks=30)
    eng.run()
    assert [key for _, key in results] == [(5, 0)]
    assert manager.pool_free == {"default": 1, "strong": 1}
    assert manager._running_strong_decodes == {}
