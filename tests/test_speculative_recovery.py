"""Eager switching repairs windows decoded from an obsolete weak boundary.

The paper's faithful double-window protocol avoids speculation entirely.  These
tests cover the repository's separate Eager policy: when a later strong result
changes a boundary already consumed by the weak chain, every affected window
must be replayed from the same retained syndrome data before finality.
"""

from dataclasses import replace
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.codes import SurfaceCodeModel
from decsim.config import TimingConfig
from decsim.decoders import SwitchingRouter
from decsim.message import DecodeResult, Operation
from decsim.planner import FixedRounds, PerOpRounds
from decsim.policies import Eager, Held
from decsim.run_spec import RunSpec, simulate
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.window_interactions import DefaultWindowInteraction


class _WeakBoundaryDecoder:
    """Make W1 uncertain and make W2's logical bit depend on W1's boundary."""

    def __init__(self, uncertain=(1,), ticks=1, propagate=False):
        self.window_ids = []
        self.uncertain = set(uncertain)
        self.ticks = ticks
        self.propagate = propagate

    def latency(self, job):
        return self.ticks

    def decode(self, job):
        self.window_ids.append(job.window_id)
        has_boundary_defect = any(
            payload.bits is not None and any(int(bit) for bit in payload.bits)
            for payload in job.payloads
        )
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_value=int(job.window_id >= 2 and has_boundary_defect),
            soft_output=0.0 if job.window_id in self.uncertain else 1.0,
            boundary_defects={job.window.commit_hi + 1: [1]}
            if (job.window_id in self.uncertain
                or (self.propagate and has_boundary_defect)) else None,
        )


class _CorrectingStrongDecoder:
    """Retract W1's weak boundary after the downstream weak chain has run."""

    def __init__(self, ticks=10_000_000):
        self.ticks = ticks

    def latency(self, job):
        return self.ticks

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id, logical_value=0,
                            boundary_defects={job.window.commit_hi + 1: [0]})


def _deterministic_run(boundary_policy, *, uncertain=(1,), strong=None,
                       run_both_at_once=False, weak_ticks=1, operation=None,
                       propagate=False, rounds=15, timing=None,
                       window_interaction=None, strategy=None):
    weak = _WeakBoundaryDecoder(
        uncertain, ticks=weak_ticks, propagate=propagate)
    strong = strong if strong is not None else _CorrectingStrongDecoder()
    result = simulate(RunSpec(
        ops=[operation if operation is not None
             else Operation(0, "memory", (0,))],
        d=3,
        rounds_policy=FixedRounds(rounds),
        scheme=SlidingWindowScheme(),
        strategy=strategy if strategy is not None
        else Switching(confidence_threshold=0.5,
                       run_both_at_once=run_both_at_once),
        boundary_policy=boundary_policy,
        window_interaction=window_interaction,
        decoder=weak,
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
        timing=timing if timing is not None else TimingConfig(),
    ))
    return result, weak


def test_interaction_invalidation_roots_expand_to_the_causal_replay_scope():
    class ImmediateChildOnly(DefaultWindowInteraction):
        def __init__(self):
            self.calls = []

        def invalidated_windows(self, source_key, windows):
            self.calls.append(source_key)
            return list(windows[source_key].dependents)

    interaction = ImmediateChildOnly()
    recovered, weak = _deterministic_run(
        Eager(), window_interaction=interaction, propagate=True, rounds=12)
    held, _ = _deterministic_run(Held(), propagate=True, rounds=12)

    assert interaction.calls == [(0, 1)]
    assert recovered["cluster"].op_results == held["cluster"].op_results
    assert weak.window_ids.count(2) == 2
    assert weak.window_ids.count(3) == 2
    assert recovered["cluster"].window_manager.speculative_replays == 1


def test_interaction_cannot_invalidate_unrelated_finished_work():
    class UnrelatedFinishedTarget(DefaultWindowInteraction):
        def boundary_targets(self, source, windows):
            if source.key == (0, 0):
                return [(1, 0)]
            return []

        def merge_boundary(self, delivery, destination, current_state):
            update = super().merge_boundary(
                delivery, destination, current_state)
            return replace(update, release_dependency=False)

        def invalidated_windows(self, source_key, windows):
            return [(1, 0)] if source_key == (0, 0) else []

    class OrderedWeakDecoder:
        def __init__(self):
            self.payload_counts = []

        def latency(self, job):
            return 20 if job.op_id == 0 else 1

        def decode(self, job):
            self.payload_counts.append(
                (job.op_id, job.window_id, len(job.payloads)))
            return DecodeResult(
                job.op_id,
                job.window_id,
                logical_value=0,
                soft_output=0.0 if job.op_id == 0 else 1.0,
                boundary_defects={job.window.commit_hi + 1: [1]}
                if job.op_id == 0 else None,
            )

    weak = OrderedWeakDecoder()
    with pytest.raises(RuntimeError, match="downstream|finished"):
        simulate(RunSpec(
            ops=[
                Operation(1, "already-final", (1,)),
                Operation(0, "later-source", (0,)),
            ],
            d=3,
            rounds_policy=FixedRounds(3),
            strategy=Switching(confidence_threshold=0.5),
            boundary_policy=Eager(),
            window_interaction=UnrelatedFinishedTarget(),
            decoder=weak,
            router=SwitchingRouter(weak, _CorrectingStrongDecoder(ticks=10)),
            unit_pools={"default": 2, "strong": 1},
            timing=TimingConfig(t_dd_us=0.0, t_ws_us=0.0),
        ))

    assert weak.payload_counts == [(1, 0, 3), (0, 0, 3)]


def test_eager_boundary_disagreement_replays_the_transitive_weak_cone():
    recovered, weak = _deterministic_run(Eager())
    held, _ = _deterministic_run(Held())

    runtime = recovered["cluster"].window_manager
    assert runtime.op_results[0] == held["cluster"].op_results[0] == 0
    assert runtime._window_logical_values[(0, 2)] == 0
    assert weak.window_ids.count(0) == 1
    assert weak.window_ids.count(1) == 1
    assert weak.window_ids.count(2) == 2
    assert weak.window_ids.count(3) == 2
    assert weak.window_ids.count(4) == 2
    assert runtime.speculative_replays == 1
    assert runtime.payloads_held == 0


@pytest.mark.parametrize("strong_first", [True, False])
def test_a_replayed_window_re_escalates_whichever_order_it_is_submitted_in(
    strong_first,
):
    """A replayed window escalates again, and Sec. III A Step 1 feeds the weak
    and strong decoders simultaneously, so the pool must admit the replayed
    attempt's strong request whether the strategy lists it before or after the
    attempt's weak job."""

    class _OrderedSwitching(Switching):
        def on_window_ready(self, window, weak_job, services):
            submissions = super().on_window_ready(window, weak_job, services)
            if strong_first and len(submissions) == 2:
                submissions = list(reversed(submissions))
            return submissions

    ordered, _ = _deterministic_run(
        Eager(), uncertain=(1, 2),
        strategy=_OrderedSwitching(confidence_threshold=0.5,
                                   run_both_at_once=True))
    baseline, _ = _deterministic_run(Eager(), uncertain=(1, 2),
                                     run_both_at_once=True)

    runtime = ordered["cluster"].window_manager
    assert runtime.op_results[0] == baseline["cluster"].op_results[0]
    assert runtime.speculative_replays == \
        baseline["cluster"].window_manager.speculative_replays
    assert runtime._finished_ops == {0}
    assert ordered["cluster"].pool._completed_strong_results == {}


def test_overlapping_escalations_discard_stale_descendant_strong_result():
    recovered, weak = _deterministic_run(Eager(), uncertain=(1, 2))
    held, _ = _deterministic_run(Held(), uncertain=(1, 2))

    runtime = recovered["cluster"].window_manager
    assert runtime.op_results[0] == held["cluster"].op_results[0]
    assert not runtime._pending_strong_windows
    assert runtime.speculative_replays == 2
    assert weak.window_ids.count(2) == 2
    assert weak.window_ids.count(3) == 3
    assert weak.window_ids.count(4) == 3


def test_leaf_escalation_wakes_an_ancestor_waiting_to_replay():
    recovered, _ = _deterministic_run(Eager(), uncertain=(1, 4))
    held, _ = _deterministic_run(Held(), uncertain=(1, 4))

    runtime = recovered["cluster"].window_manager
    assert runtime.op_results[0] == held["cluster"].op_results[0] == 0
    assert runtime.speculative_replays == 1
    assert not runtime._pending_strong_windows
    assert runtime._finished_ops == {0}


def test_agreeing_descendant_wakes_an_ancestor_waiting_to_replay():
    class _MixedStrongDecoder(_CorrectingStrongDecoder):
        def decode(self, job):
            boundary_bit = 0 if job.window_id == 1 else 1
            return DecodeResult(
                job.op_id,
                job.window_id,
                logical_value=0,
                boundary_defects={job.window.commit_hi + 1: [boundary_bit]},
            )

    recovered, _ = _deterministic_run(
        Eager(), uncertain=(1, 2), strong=_MixedStrongDecoder())
    held, _ = _deterministic_run(
        Held(), uncertain=(1, 2), strong=_MixedStrongDecoder())

    runtime = recovered["cluster"].window_manager
    assert runtime.op_results[0] == held["cluster"].op_results[0]
    assert runtime.speculative_replays == 1
    assert not runtime._pending_strong_windows
    assert runtime._finished_ops == {0}


def test_strong_can_add_a_defect_to_an_empty_weak_boundary():
    class _EmptyBoundaryWeakDecoder(_WeakBoundaryDecoder):
        def decode(self, job):
            result = super().decode(job)
            result.boundary_defects = None
            return result

    class _AddingStrongDecoder(_CorrectingStrongDecoder):
        def decode(self, job):
            return DecodeResult(
                job.op_id,
                job.window_id,
                correction=[1],
                logical_value=0,
                boundary_defects={job.window.commit_hi + 1: [1]},
            )

    def run(policy):
        weak = _EmptyBoundaryWeakDecoder()
        strong = _AddingStrongDecoder()
        return simulate(RunSpec(
            ops=[Operation(0, "memory", (0,))],
            d=3,
            rounds_policy=FixedRounds(15),
            scheme=SlidingWindowScheme(),
            strategy=Switching(confidence_threshold=0.5),
            boundary_policy=policy,
            decoder=weak,
            router=SwitchingRouter(weak, strong),
            unit_pools={"default": 1, "strong": 1},
        ))

    recovered = run(Eager())["cluster"].window_manager
    held = run(Held())["cluster"].window_manager
    assert recovered.op_results[0] == held.op_results[0] == 1
    assert recovered._window_logical_values[(0, 2)] == 1
    assert recovered.speculative_replays == 1


def test_early_strong_correction_waits_for_inflight_weak_cone_then_replays():
    recovered, weak = _deterministic_run(
        Eager(), strong=_CorrectingStrongDecoder(ticks=1))

    runtime = recovered["cluster"].window_manager
    assert runtime.op_results[0] == 0
    assert runtime.speculative_replays == 1
    # The correction beats W2's original dispatch, so W2-W4 consume the
    # corrected seam on their first and only decode.
    assert weak.window_ids.count(2) == 1
    assert weak.window_ids.count(3) == 1
    assert weak.window_ids.count(4) == 1
    assert not runtime._pending_strong_windows


@pytest.mark.parametrize("weak_ticks,strong_ticks", [(10, 1), (1, 1)])
def test_parallel_early_and_same_tick_strong_results_recover_deterministically(
        weak_ticks, strong_ticks):
    recovered, _ = _deterministic_run(
        Eager(),
        run_both_at_once=True,
        weak_ticks=weak_ticks,
        strong=_CorrectingStrongDecoder(ticks=strong_ticks),
    )
    held, _ = _deterministic_run(
        Held(),
        run_both_at_once=True,
        weak_ticks=weak_ticks,
        strong=_CorrectingStrongDecoder(ticks=strong_ticks),
    )

    runtime = recovered["cluster"].window_manager
    assert runtime.op_results[0] == held["cluster"].op_results[0]
    assert runtime.speculative_replays == 1
    assert not runtime._pending_strong_windows
    assert runtime._finished_ops == {0}


def test_replay_invalidates_an_inflight_descendant_boundary():
    """A reset W2 cannot release W3 with its obsolete outbound boundary."""
    timing = TimingConfig(t_dd_us=0.5, t_ws_us=0.5)
    eager, weak = _deterministic_run(
        Eager(),
        strong=_CorrectingStrongDecoder(ticks=2_900_000),
        propagate=True,
        rounds=12,
        timing=timing,
    )
    held, _ = _deterministic_run(
        Held(),
        strong=_CorrectingStrongDecoder(ticks=2_900_000),
        propagate=True,
        rounds=12,
        timing=timing,
    )
    runtime = eager["cluster"].window_manager
    held_runtime = held["cluster"].window_manager

    assert runtime.op_results[0] == held_runtime.op_results[0]
    assert runtime._window_logical_values == held_runtime._window_logical_values
    assert all(window.deps_remaining == 0
               for window in runtime.windows.values())
    assert weak.window_ids.count(3) == 1


def test_replay_invalidates_inflight_boundaries_from_unaffected_parents():
    """A restored merge parent cannot also arrive through its old callback."""
    class _MergeWeakDecoder:
        def __init__(self):
            self.calls = []

        def latency(self, job):
            return 10

        def decode(self, job):
            self.calls.append((job.op_id, job.window_id))
            escalates = job.op_id == 0 and job.window_id == 0
            return DecodeResult(
                job.op_id,
                job.window_id,
                logical_value=0,
                soft_output=0.0 if escalates else 1.0,
                boundary_defects={job.window.commit_hi + 1: [1]},
            )

    class _MergeStrongDecoder:
        def latency(self, job):
            return 1_000

        def decode(self, job):
            return DecodeResult(
                job.op_id,
                job.window_id,
                correction=[0],
                logical_value=0,
                boundary_defects=None,
            )

    def run(policy):
        weak = _MergeWeakDecoder()
        result = simulate(RunSpec(
            ops=[
                Operation(0, "speculative-parent", (0,)),
                Operation(1, "ordinary-parent", (1,)),
                Operation(2, "merge-child", (0, 1), predecessors=(0, 1)),
            ],
            code=SurfaceCodeModel(d=3),
            rounds_policy=PerOpRounds({0: 3, 1: 3, 2: 3}),
            scheme=SlidingWindowScheme(),
            strategy=Switching(confidence_threshold=0.5),
            boundary_policy=policy,
            decoder=weak,
            router=SwitchingRouter(weak, _MergeStrongDecoder()),
            unit_pools={"default": 1, "strong": 1},
            timing=TimingConfig(t_dd_us=0.5, t_ws_us=0.0),
        ))
        return result, weak

    eager, eager_weak = run(Eager())
    held, _ = run(Held())
    runtime = eager["cluster"].window_manager
    held_runtime = held["cluster"].window_manager

    assert runtime.op_results == held_runtime.op_results
    assert all(window.deps_remaining == 0
               for window in runtime.windows.values())
    assert eager_weak.calls.count((2, 0)) == 1


def test_non_clifford_result_is_not_final_until_recovery_finishes():
    operation = Operation(
        0,
        "measurement",
        (0,),
        clifford=False,
        requires_result_return_to_chip=True,
    )
    recovered, _ = _deterministic_run(
        Eager(), operation=operation)
    runtime = recovered["cluster"].window_manager

    assert runtime.speculative_replays == 1
    assert not runtime._pending_strong_windows
    assert runtime._finished_ops == {0}
    orchestrator = recovered["orchestrator"]
    assert orchestrator.stats["result_returns"] == 1
    assert orchestrator.history[-1]["t"] >= runtime.op_strong_commit_time[0]


def test_equal_strong_boundary_preserves_eager_progress_without_replay():
    class _AgreeingStrongDecoder(_CorrectingStrongDecoder):
        def decode(self, job):
            return DecodeResult(
                job.op_id,
                job.window_id,
                logical_value=0,
                boundary_defects={job.window.commit_hi + 1: [1]},
            )

    recovered, weak = _deterministic_run(
        Eager(), strong=_AgreeingStrongDecoder())
    runtime = recovered["cluster"].window_manager

    assert runtime.speculative_replays == 0
    assert weak.window_ids == [0, 1, 2, 3, 4]
    assert runtime.windows[(0, 4)].t_done < runtime.op_strong_commit_time[0]
    assert runtime.payloads_held == 0


def test_dynamic_streams_reject_eager_speculative_recovery():
    strategy = Switching(confidence_threshold=0.5)
    stream = Operation(10, "stream", (0,))
    with pytest.raises(ValueError, match="dynamic streams"):
        RunSpec(
            ops=[Operation(0, "driver", (1,))],
            dynamic_streams=[stream],
            strategy=strategy,
            decoder=_WeakBoundaryDecoder(),
        ).validate()

    RunSpec(
        ops=[Operation(0, "driver", (1,))],
        dynamic_streams=[Operation(10, "stream", (0,))],
        strategy=Switching(confidence_threshold=0.5),
        boundary_policy=Held(),
        decoder=_WeakBoundaryDecoder(),
    ).validate()


def test_static_operation_seam_replays_without_a_data_dependent_crash():
    class _OperationWeakDecoder(_WeakBoundaryDecoder):
        def decode(self, job):
            result = super().decode(job)
            escalates = job.op_id == 0 and job.window_id == 0
            result.soft_output = 0.0 if escalates else 1.0
            result.boundary_defects = (
                {job.window.commit_hi + 1: [1]} if escalates else None)
            return result

    weak = _OperationWeakDecoder()
    strong = _CorrectingStrongDecoder(ticks=10)
    spec = RunSpec(
        ops=[
            Operation(0, "A", (0,), has_successor=True),
            Operation(1, "B", (0,), predecessors=(0,)),
        ],
        d=3,
        rounds_policy=FixedRounds(6),
        scheme=SlidingWindowScheme(),
        strategy=Switching(confidence_threshold=0.5),
        boundary_policy=Eager(),
        decoder=weak,
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    )

    spec.validate()
    runtime = simulate(spec)["cluster"].window_manager
    assert runtime._finished_ops == {0, 1}
    assert runtime.speculative_replays == 1
    assert not runtime._pending_strong_windows


def test_cross_operation_result_is_published_only_after_ancestor_recovery():
    """A provisional seam may be consumed, but its result must not escape."""
    class _CrossOperationWeakDecoder:
        def __init__(self):
            self.calls = []

        def latency(self, job):
            return 1

        def decode(self, job):
            self.calls.append((job.op_id, job.window_id,
                               tuple(sorted(job.window.boundary_in))))
            boundary_bit = any(
                payload.bits is not None
                and any(int(bit) for bit in payload.bits)
                for payload in job.payloads
            )
            escalates = job.op_id == 0 and job.window_id == 1
            return DecodeResult(
                job.op_id,
                job.window_id,
                logical_value=int(
                    job.op_id == 1 and job.window_id == 0 and boundary_bit),
                soft_output=0.0 if escalates else 1.0,
                boundary_defects={job.window.commit_hi + 1: [1]}
                if escalates else None,
            )

    class _LateEmptyBoundaryStrongDecoder:
        def latency(self, job):
            return 100_000_000

        def decode(self, job):
            return DecodeResult(
                job.op_id,
                job.window_id,
                logical_value=0,
                correction=[0],
                boundary_defects=None,
            )

    def run(policy):
        weak = _CrossOperationWeakDecoder()
        strong = _LateEmptyBoundaryStrongDecoder()
        result = simulate(RunSpec(
            ops=[
                Operation(0, "A", (0,), has_successor=True),
                Operation(1, "B", (0,), predecessors=(0,)),
            ],
            d=3,
            rounds_policy=FixedRounds(6),
            scheme=SlidingWindowScheme(),
            strategy=Switching(confidence_threshold=0.5),
            boundary_policy=policy,
            decoder=weak,
            router=SwitchingRouter(weak, strong),
            unit_pools={"default": 1, "strong": 1},
        ))
        return result, weak

    eager, weak = run(Eager())
    held, _ = run(Held())
    runtime = eager["cluster"].window_manager
    held_runtime = held["cluster"].window_manager

    assert runtime.speculative_replays == 1
    assert runtime.op_results[1] == held_runtime.op_results[1] == 0
    assert weak.calls.count((1, 0, (1,))) == 1
    assert sum(op_id == 1 and window_id == 0
               for op_id, window_id, _ in weak.calls) == 2

    publications = [record for record in eager["orchestrator"].history
                    if record["op_id"] == 1]
    assert len(publications) == 1
    assert publications[0]["outcome"] == 0
    assert publications[0]["t"] >= runtime.op_strong_commit_time[0]
    assert eager["orchestrator"].frame.snapshot() == \
        held["orchestrator"].frame.snapshot()


class _StreamWeakDecoder:
    def __init__(self, uncertain_windows):
        self.uncertain_windows = set(uncertain_windows)
        self.calls = []
        self.committed_prefixes_at_w2 = []
        self.window_manager = None

    def latency(self, job):
        return 10_000_000

    def decode(self, job):
        self.calls.append((job.op_id, job.window_id))
        if job.window_id == 2 and self.window_manager is not None:
            self.committed_prefixes_at_w2.append(
                self.window_manager.committed_stream_round_count(job.op_id))
        escalates = job.window_id in self.uncertain_windows
        has_boundary_defect = any(
            payload.bits is not None
            and any(int(bit) for bit in payload.bits)
            for payload in job.payloads
        )
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_value=int(job.window_id >= 2 and has_boundary_defect),
            soft_output=0.0 if escalates else 1.0,
            boundary_defects={job.window.commit_hi + 1: [1]}
            if escalates else None,
        )


class _StreamStrongDecoder:
    def __init__(self, *, agreeing_windows=(), latency_by_window=None):
        self.agreeing_windows = set(agreeing_windows)
        self.latency_by_window = dict(latency_by_window or {})

    def latency(self, job):
        return self.latency_by_window.get(job.window_id, 100_000_000)

    def decode(self, job):
        boundary = None
        if job.window_id in self.agreeing_windows:
            boundary = {job.window.commit_hi + 1: [1]}
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_value=0,
            correction=[0],
            boundary_defects=boundary,
        )


def _run_static_stream_recovery(policy, *, first_segment_rounds,
                                uncertain_windows, strong):
    stream_rounds = 15
    stream = Operation(0, "stream", (0,))
    first_segment = Operation(
        1, "segment-0", (0,), clifford=False,
        consumes_magic_state=False, stream_id=stream.id, stream_offset=0,
        has_successor=True,
    )
    second_segment = Operation(
        2, "segment-1", (0,), stream_id=stream.id,
        stream_offset=first_segment_rounds, predecessors=(first_segment.id,),
    )
    feedback_consumer = Operation(
        3, "feedback-consumer", (1,), blocked_by=first_segment.id)
    weak = _StreamWeakDecoder(uncertain_windows)
    world = RunSpec(
        ops=[first_segment, second_segment, feedback_consumer],
        decode_ops=[stream, feedback_consumer],
        code=SurfaceCodeModel(d=3),
        rounds_policy=PerOpRounds({
            stream.id: stream_rounds,
            first_segment.id: first_segment_rounds,
            second_segment.id: stream_rounds - first_segment_rounds,
            feedback_consumer.id: 1,
        }),
        scheme=SlidingWindowScheme(),
        strategy=Switching(confidence_threshold=0.5),
        boundary_policy=policy,
        decoder=weak,
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    ).build(verbose=False)
    weak.window_manager = world.window_manager
    publication_states = []
    integrate = world.orchestrator.integrate

    def record_segment_publication(operation, result):
        if operation.id == first_segment.id:
            publication_states.append({
                "t": world.engine.now,
                "committed": {
                    index: world.window_manager.windows[(0, index)].committed
                    for index in range(5)
                },
                "decode_counts": {
                    index: weak.calls.count((0, index))
                    for index in range(5)
                },
            })
        integrate(operation, result)

    world.orchestrator.integrate = record_segment_publication
    world.gate.load(world.ops)
    world.engine.run()
    return world, weak, publication_states


def test_static_stream_segment_waits_for_its_corrected_replay_cone():
    """A stream segment cannot release QPU work from a reset prefix."""
    eager, weak, eager_publication_states = _run_static_stream_recovery(
        Eager(), first_segment_rounds=9, uncertain_windows=(1,),
        strong=_StreamStrongDecoder())
    held, _, held_publication_states = _run_static_stream_recovery(
        Held(), first_segment_rounds=9, uncertain_windows=(1,),
        strong=_StreamStrongDecoder())

    runtime = eager.window_manager
    segment_publications = [
        record for record in eager.orchestrator.history
        if record["op_id"] == 1
    ]
    replay_done = max(runtime.windows[(0, index)].t_done
                      for index in (2, 3, 4))

    assert runtime.speculative_replays == 1
    assert weak.calls.count((0, 2)) == 2
    assert weak.committed_prefixes_at_w2 == [6, 6]
    assert len(segment_publications) == 1
    assert len(eager_publication_states) == 1
    assert eager_publication_states[0]["t"] == segment_publications[0]["t"]
    assert eager_publication_states[0]["committed"][2]
    assert eager_publication_states[0]["decode_counts"][2] == 2
    assert segment_publications[0]["t"] >= replay_done
    assert eager.gate.op_start_time[3] >= replay_done
    assert held_publication_states[0]["committed"][2]
    assert eager.orchestrator.frame.snapshot() == \
        held.orchestrator.frame.snapshot()


def test_overlapping_stream_roots_hold_segment_until_both_resolve():
    """One agreeing root cannot release a segment blocked by a later root."""
    strong = _StreamStrongDecoder(
        agreeing_windows=(1,),
        latency_by_window={1: 50_000_000, 2: 100_000_000},
    )
    eager, weak, publication_states = _run_static_stream_recovery(
        Eager(), first_segment_rounds=12, uncertain_windows=(1, 2),
        strong=strong)
    held, _, _ = _run_static_stream_recovery(
        Held(), first_segment_rounds=12, uncertain_windows=(1, 2),
        strong=_StreamStrongDecoder(
            agreeing_windows=(1,),
            latency_by_window={1: 50_000_000, 2: 100_000_000},
        ))
    runtime = eager.window_manager
    publications = [record for record in eager.orchestrator.history
                    if record["op_id"] == 1]
    replay_done = max(runtime.windows[(0, index)].t_done
                      for index in (3, 4))

    assert runtime.speculative_replays == 1
    assert weak.calls.count((0, 3)) == 2
    assert len(publications) == len(publication_states) == 1
    assert publications[0]["t"] >= replay_done
    assert publications[0]["t"] >= runtime.op_strong_commit_time[0]
    assert eager.gate.op_start_time[3] >= replay_done
    assert eager.orchestrator.frame.snapshot() == \
        held.orchestrator.frame.snapshot()
    assert not runtime.speculative_recovery.has_finality_blockers


def test_real_stim_recovery_uses_same_shot_truth_and_matches_held():
    stim = pytest.importorskip("stim")
    np = pytest.importorskip("numpy")
    pytest.importorskip("pymatching")

    from decsim.adapters.stim_device import StimDevice
    from decsim.decoders import SampledConfidenceDecoder
    from decsim.mwpm_decoder import PyMatchingDecoder, UnweightedPyMatchingDecoder

    class _Latency:
        def __init__(self, ticks):
            self.ticks = ticks

        def latency(self, job):
            return self.ticks

    class _Recording:
        def __init__(self, inner):
            self.inner = inner
            self.jobs = []
            self.results = []

        def latency(self, job):
            return self.inner.latency(job)

        def decode(self, job):
            result = self.inner.decode(job)
            self.jobs.append(job)
            self.results.append(result)
            return result

    rounds = 15
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.02,
        after_reset_flip_probability=0.02,
        before_measure_flip_probability=0.02,
        before_round_data_depolarization=0.02,
    )

    def run(policy, seed):
        operation = Operation(0, "memory", (0,), circuit=circuit)
        weak = _Recording(SampledConfidenceDecoder(
            UnweightedPyMatchingDecoder(_Latency(1)), 0.0,
            probability_for=lambda job: 1.0 if job.window_id == 1 else 0.0))
        strong = _Recording(PyMatchingDecoder(_Latency(10_000_000)))
        device = StimDevice(seed=seed)
        result = simulate(RunSpec(
            ops=[operation],
            d=3,
            rounds_policy=FixedRounds(rounds),
            scheme=SlidingWindowScheme(),
            strategy=Switching(confidence_threshold=0.5),
            boundary_policy=policy,
            device=device,
            decoder=weak,
            router=SwitchingRouter(weak, strong),
            unit_pools={"default": 1, "strong": 1},
        ))
        return result, device, weak, strong

    for seed in range(500):
        eager, eager_device, eager_weak, eager_strong = run(Eager(), seed)
        if eager["cluster"].window_manager.speculative_replays:
            break
    else:
        pytest.fail("no real strong-boundary disagreement in seeds 0..499")

    held, held_device, _, _ = run(Held(), seed)
    assert np.array_equal(eager_device._dets[0], held_device._dets[0])
    assert np.array_equal(eager_device._truth[0], held_device._truth[0])

    truth = int(eager_device._truth[0][0])
    eager_prediction = int(eager["cluster"].op_results[0])
    held_prediction = int(held["cluster"].op_results[0])
    assert eager_prediction == held_prediction
    # Truth is the observable sampled with these exact detector arrays, not a
    # strong-decoder result.  This compares the two policies' actual logical
    # error classification; it does not claim that either decoder is perfect.
    eager_logical_error = eager_prediction ^ truth
    held_logical_error = held_prediction ^ int(held_device._truth[0][0])
    assert eager_logical_error == held_logical_error
    strong_job = eager_strong.jobs[0]
    weak_result = next(
        result for job, result in zip(eager_weak.jobs, eager_weak.results)
        if (job.op_id, job.window_id) == strong_job.strong_decode_for)
    assert weak_result.boundary_defects != eager_strong.results[0].boundary_defects
    assert eager["cluster"].window_manager.speculative_replays == 1
