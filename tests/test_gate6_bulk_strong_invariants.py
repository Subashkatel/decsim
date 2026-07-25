"""Gate-6 leftover: bulk_strong merge invariants (V9 caveat closure).

Covers the previously-uncovered bulk_strong paths of DecoderManager:
merge-and-deliver, refusal of accuracy-coupled merges, running-rounds
accounting, and the cancel-one-merged-key edge. The last one exposed
TWO real bugs (2026-07-04, probe-verified before the fix): cancelling
one key of a RUNNING merged batch cancelled the whole batch — the
sibling keys' results were silently lost (their windows hung in
_windows_waiting_for_strong_result forever) — and strong_running_rounds
leaked (never decremented on a cancelled batch). Fixed in
decoder_manager.cancel_strong: a running merged batch with live
siblings survives the cancel (only the cancelled key is dropped from
delivery); a batch with no survivors cancels AND settles the rounds
accounting.

The same destination-keyed state carries the entitlement contract, which
is exercised here at both bulk_strong settings: a destination window owns
at most one unconsumed strong result, admission hands that entitlement out
and refuses a request no destination would consume, and a completion is
held only while its destination can still ask for one.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.decoders import CodeRouter, PerRoundDecoder, SwitchingRouter
from decsim.decoder_manager import DecoderManager, StrategyServicesImpl
from decsim.message import DecodeJob, DecodeResult, Operation
from decsim.planner import FixedRounds
from decsim.protocols import Directive, OutcomeDirective, Submission
from decsim.run_spec import RunSpec
from decsim.schedulers import FifoScheduler
from decsim.schemes import SlidingWindowScheme


class _NullStrategy:
    def on_decode_outcome(self, outcome, services):
        return None


def build(bulk_strong=True, decoder=None):
    eng = Engine(verbose=False)
    if decoder is None:
        decoder = PerRoundDecoder(tau_us=1.0)
    manager = DecoderManager(
        eng, router=CodeRouter(default=decoder),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1, "strong": 1}, bulk_strong=bulk_strong)
    manager.strategy = _NullStrategy()
    results = []
    manager.on_strong_window_decoded = \
        lambda key, res: results.append((eng.now, key))
    return eng, manager, results


def strong_job(op, rounds, label=None):
    return DecodeJob(op_id=op, window_id=0, n_rounds=rounds,
                     strong_decode_for=(op, 0), hint="strong",
                     label=label or f"s{op}")


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
    assert manager.strong_running_rounds == 0
    assert not manager._windows_waiting_for_strong_result


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("correction", ("aggregate-correction",)),
        ("logical_value", 1),
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

    class RecordingStrategy:
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
    assert manager.strong_running_rounds == 10
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
    """THE bug: before the fix, cancelling (2,0) killed the whole
    running batch and (3,0) hung forever with rounds leaked."""
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
    assert manager.strong_running_rounds == 0, "rounds accounting leaked"
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
    assert manager.strong_running_rounds == 0
    assert manager.pool_free == {"default": 1, "strong": 1}


def test_bulk_strong_refuses_accuracy_coupled_merges():
    eng, manager, _ = build()
    manager.enqueue(strong_job(1, 10, "s-block"))
    j2 = strong_job(2, 5)
    j2.dem = object()                       # accuracy-coupled marker
    manager.enqueue(j2)
    manager.enqueue(strong_job(3, 5))
    with pytest.raises(RuntimeError, match="bulk_strong only merges"):
        eng.run()


def test_running_rounds_tracks_merged_batch_lifecycle():
    eng, manager, _ = build()
    occupy_then_merge(eng, manager)
    seen = []

    def watch():
        seen.append((eng.now / us(1), manager.strong_running_rounds))
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
    before = (dict(manager.pool_free), manager.strong_running_rounds,
              set(manager._running_strong_decodes), manager.queued_total(),
              dict(manager._completed_strong_results),
              len(manager.queue_log), len(eng.log_lines))

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        manager.enqueue(strong_job(2, 7, "s-duplicate"))

    assert (dict(manager.pool_free), manager.strong_running_rounds,
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

    manager._wait_for_strong_result((2, 0))          # a replay of the same key
    assert manager._windows_waiting_for_strong_result == {(2, 0)}
    assert [key for _, key in results] == [(1, 0), (2, 0)]


def test_public_strategy_duplicate_strong_submission_is_rejected_end_to_end():
    """Destination uniqueness is owned by the manager, not the strategy: a
    real DecodingStrategy that escalates one weak window twice is refused."""

    class DuplicateStrongStrategy:
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

    weak = PerRoundDecoder(tau_us=0.05)
    strong = PerRoundDecoder(tau_us=2.0)
    world = RunSpec(
        ops=[Operation(88, "duplicate-request", (6,), clifford=True)],
        d=3, rounds_policy=FixedRounds(30), round_us=1.0,
        scheme=SlidingWindowScheme(), strategy=DuplicateStrongStrategy(),
        decoder=weak, router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    ).build(verbose=False)
    world.gate.load(world.ops)

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        world.engine.run()

    assert world.window_manager._finished_ops == set()
    assert world.pool._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_early_strong_result_is_held_until_its_destination_asks(bulk_strong):
    """The case the hold map exists for: the strong decode finishes before
    the destination's weak commit registers the wait, and that wait is what
    consumes it."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.enqueue(strong_job(2, 5, "s-early"))
    eng.run()

    assert results == []
    assert set(manager._completed_strong_results) == {(2, 0)}

    manager._wait_for_strong_result((2, 0))
    assert [key for _, key in results] == [(2, 0)]
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_strong_request_for_a_destination_that_already_adopted_one_is_refused(
    bulk_strong,
):
    """The duplicate separated in time: the first request has completed and
    been adopted when the second is submitted, so the guard on live requests
    never sees it and its result would be parked for nobody."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    manager._windows_waiting_for_strong_result.add((2, 0))
    manager.enqueue(strong_job(2, 5, "s-first"))
    eng.run()
    assert [key for _, key in results] == [(2, 0)]

    with pytest.raises(RuntimeError, match="settled window"):
        manager.enqueue(strong_job(2, 7, "s-late"))

    eng.run()
    assert [key for _, key in results] == [(2, 0)]
    assert manager._completed_strong_results == {}

    manager._wait_for_strong_result((2, 0))      # a replay of the same key
    assert manager._windows_waiting_for_strong_result == {(2, 0)}
    assert [key for _, key in results] == [(2, 0)], \
        "a wait was released by a decode that was never assigned to it"


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
    it, and the destination's wait would then be released by a decode the
    first request produced nothing for."""
    eng, manager, results = build(bulk_strong=bulk_strong)
    first = strong_job(2, 5, "s-first")
    manager.enqueue(first)
    eng.run()
    held = manager._completed_strong_results[(2, 0)]

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        manager.enqueue(strong_job(2, 7, "s-second"))

    assert manager._completed_strong_results[(2, 0)] is held
    manager._wait_for_strong_result((2, 0))
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

    class CancelInHookStrategy:
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

    class ResubmitInHookStrategy:
        def __init__(self, manager):
            self.manager = manager
            self.resubmitted = False

        def on_decode_outcome(self, outcome, services):
            if not self.resubmitted:
                self.resubmitted = True
                self.manager.enqueue(strong_job(2, 7, "s-newer"))
            return OutcomeDirective(Directive.FINALIZE_STRONG)

    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = ResubmitInHookStrategy(manager)
    manager.enqueue(strong_job(2, 5, "s-first"))

    with pytest.raises(RuntimeError, match="no destination waiting"):
        eng.run()

    assert results == []
    assert manager._completed_strong_results == {}


@pytest.mark.parametrize("bulk_strong", [True, False])
def test_a_new_weak_attempt_reopens_a_settled_destination(bulk_strong):
    """A settled destination is settled for its attempt, not forever: a new
    weak attempt for the window may escalate again, and its strong result may
    again arrive before the new weak commit."""

    class AwaitStrongStrategy:
        def on_decode_outcome(self, outcome, services):
            if outcome.job.strong_decode_for is not None:
                return OutcomeDirective(Directive.FINALIZE_STRONG)
            return OutcomeDirective(Directive.AWAIT_STRONG)

    eng, manager, results = build(bulk_strong=bulk_strong)
    manager.strategy = AwaitStrongStrategy()
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

    class EscalateEveryWindowStrategy:
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

    weak = PerRoundDecoder(tau_us=0.05)
    strong = PerRoundDecoder(tau_us=2.0)
    world = RunSpec(
        ops=[Operation(88, "escalate-every-window", (6,), clifford=True)],
        d=3, rounds_policy=FixedRounds(30), round_us=1.0,
        scheme=SlidingWindowScheme(), strategy=EscalateEveryWindowStrategy(),
        decoder=weak, router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    ).build(verbose=False)
    world.gate.load(world.ops)
    commit_strong_result = world.pool.on_strong_window_decoded
    delivered = []

    def record(key, result):
        delivered.append(key)
        commit_strong_result(key, result)

    world.pool.on_strong_window_decoded = record
    world.engine.run()

    assert world.window_manager._finished_ops == {88}
    assert len(delivered) == len(set(delivered)) > 1
    assert world.pool._completed_strong_results == {}
    assert world.pool._windows_waiting_for_strong_result == set()

    finalized_key = delivered[-1]
    world.pool._wait_for_strong_result(finalized_key)
    assert world.pool._windows_waiting_for_strong_result == {finalized_key}
    assert delivered.count(finalized_key) == 1, \
        "a wait was released after finality without a decode of its own"


def test_public_strategy_delayed_duplicate_submission_is_refused_end_to_end():
    """The same public seam with the second submission handed over the
    weak->strong link: the refusal happens at submission, so no run reaches
    finality on a stale result."""

    class DelayedDuplicateStrongStrategy:
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

    weak = PerRoundDecoder(tau_us=0.05)
    strong = PerRoundDecoder(tau_us=2.0)
    world = RunSpec(
        ops=[Operation(88, "duplicate-request", (6,), clifford=True)],
        d=3, rounds_policy=FixedRounds(30), round_us=1.0,
        scheme=SlidingWindowScheme(),
        strategy=DelayedDuplicateStrongStrategy(),
        decoder=weak, router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    ).build(verbose=False)
    world.gate.load(world.ops)

    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        world.engine.run()

    assert world.window_manager._finished_ops == set()
    assert world.pool._completed_strong_results == {}
