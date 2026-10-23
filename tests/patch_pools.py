"""Memory patches decoded for real on a pool per patch, for oracle tests.

Three surface-code patches of Stim's rotated memory, d = 3 for 12 rounds
at p = 0.03 so that windows fail and escalate, each patch's windows sent
by a router to a pool of its own: weak for patch 0, w1 and w2 for the
others, each Experiment 1's union-find row. A router decides where and
when a window decodes, never what it reads, so every machine here must
answer every window as one pool does. The strong pool is belief matching
on stated times, a decoder that draws nothing, so two pools of it answer
alike; Relay-BP draws from its pool's own seed and would not.
"""

import dataclasses

import decsim.confidence.cluster as cluster
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.presets.experiment_one as experiment_one
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.topology.graph as graph
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings

DISTANCE = 3
ERROR_RATE = 0.03
ROUNDS = 12
PATCHES = 3
# the pool each patch's windows go to
POOL_BY_PATCH = ((0, machines.WEAK_POOL), (1, "w1"), (2, "w2"))
# a seed at which every weak pool escalates a window
SEED = 1
_TEN_DECIBELS = threshold_sources.FixedThreshold.Settings(
    threshold_decibels=10.0
)
_STATED_TIMES = belief_matching.IterationTime(
    iteration_microseconds=0.5, matching_microseconds=10.0
)
_BELIEF_MATCHING = belief_matching.BeliefMatchingDecoder.Settings(
    max_iterations=30, timing=_STATED_TIMES
)
_QUICK_WALK = cluster.ClusterGap.Settings(walk_microseconds=0.2)
_REDO_WINDOW = strong_window_shapes.RedoWindow.Settings()


def one_pool() -> graph.Machine:
    """The three patches on Experiment 1's one weak pool."""
    weak = experiment_one.weak_alone(DISTANCE, ERROR_RATE)
    workload = producers.MemorySettings(
        ERROR_RATE, rounds=ROUNDS, patches=PATCHES
    )
    return weak.with_held("workload", workload)


def pool_per_patch(
    machine: graph.Machine, has_store_per_pool: bool = False
) -> graph.Machine:
    """The machine with a router sending each patch to its own pool.

    has_store_per_pool puts the router before the stores, one store a
    pool, rather than after the one store.
    """
    weak_pool = machine.part(machines.WEAK_POOL)
    added = (("w1", weak_pool.settings), ("w2", weak_pool.settings))
    rule = routers.ByPatch.Settings(pool_by_patch=POOL_BY_PATCH)
    if has_store_per_pool:
        return machines.with_store_per_pool(machine, rule, added)
    return machines.with_router(machine, rule, added)


def one_pair() -> graph.Machine:
    """The three patches on one weak pool escalating to one strong pool.

    The cluster gap under 10 dB escalates, into a redo window.
    """
    machine = one_pool()
    base = experiment_one.switching(DISTANCE, ERROR_RATE)
    slot = base.part(machines.SWITCHING)
    switching_settings = dataclasses.replace(
        slot.settings,
        threshold=_TEN_DECIBELS,
        confidence=_QUICK_WALK,
        strong_window=_REDO_WINDOW,
    )
    held_windows = machine.held(machines.WINDOWS)
    windows = window_settings.switching_windows(
        held_windows, switching_settings.strong_window
    )
    with_windows = machine.with_held(machines.WINDOWS, windows)
    strong_pool = experiment_one.strong_decoder_pool()
    strong_pool = dataclasses.replace(strong_pool, algorithm=_BELIEF_MATCHING)
    return machines.with_escalation(
        with_windows,
        machines.WEAK_POOL,
        machines.STRONG_POOL,
        strong_pool,
        switching_settings,
        switching_name=machines.SWITCHING,
    )


def shared_strong_per_patch(has_store_per_pool: bool = False) -> graph.Machine:
    """A weak pool per patch, each escalating to the one strong pool."""
    pair = one_pair()
    routed = pool_per_patch(pair, has_store_per_pool)
    slot = routed.part(machines.SWITCHING)
    for pool in ("w1", "w2"):
        routed = machines.with_escalation_to(
            routed, pool, machines.STRONG_POOL, slot.settings
        )
    return routed


def shot(machine: graph.Machine) -> collect.Shot:
    """The machine's shot at SEED."""
    task = collect.Task("patches", machine, {})
    return collect.run_shot(task, SEED)


def answers(patch_shot: collect.Shot) -> tuple:
    """Every window's committed observables, and each patch's outcome.

    The pool that committed a window is left out: it is where, not what.
    """
    observation = patch_shot.machine.observation
    by_window = {}
    for record in observation.frame_corrections.committed:
        by_window[record.window_key] = record.logical_observables
    outcomes = []
    for result in patch_shot.result.operation_results:
        outcome = (result.operation_id, result.logical_observables)
        outcomes.append(outcome)
    return by_window, tuple(outcomes)


def committing_pools(patch_shot: collect.Shot) -> set:
    """The pools that committed at least one window of the shot."""
    observation = patch_shot.machine.observation
    committed = observation.frame_corrections.committed
    return {record.tier for record in committed}
