"""Building the switching part: the signal, the policy, the strong side.

A run whose switching slot is filled gets the confidence its weak
decoder reports, the policy that decides on it with the slot's
threshold, and the strong window side; a run with none gets nothing,
and the escalation ports stay unbound.
"""

import dataclasses

import pytest

import decsim.build.escalation as escalation_build
import decsim.build.layout as layout_module
import decsim.confidence.complementary as complementary
import decsim.confidence.no_signal as no_signal
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.experiments.collect as collect
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.settings as machine_settings
import tests.declared_run as declared_run
import tests.topology.test_graph_layout as layout_tests
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


class _OwnPolicy:
    """A policy of a researcher's own; the test reads only its class."""

    def __init__(self, threshold) -> None:
        self.threshold = threshold


@dataclasses.dataclass(frozen=True)
class _OwnPolicySwitching(escalation_settings.SwitchingSettings):
    """A switching record that builds a policy of its own."""

    def build_policy(self, threshold) -> _OwnPolicy:
        """The researcher's policy on the slot's threshold."""
        return _OwnPolicy(threshold)


def _switching(
    record=escalation_settings.SwitchingSettings, **changes
) -> escalation_settings.SwitchingSettings:
    """A switching slot on a fixed threshold and the complementary gap."""
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return record(confidence=confidence, threshold=threshold, **changes)


def _weak() -> decoder_settings.DecoderPoolSettings:
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    return decoder_settings.DecoderPoolSettings(algorithm=matching)


def _built(
    settings: escalation_settings.SwitchingSettings,
    engine: engine_module.Engine,
) -> escalation_build.Switching:
    """The switching part of the one edge a weak and a strong pool make."""
    weak = _weak()
    machine = machine_settings.MachineSettings(
        weak_decoder=weak, strong_decoder=weak, switching=settings
    )
    layout = layout_module.legacy_layout(machine)
    switching_parts = escalation_build.build_switching(layout, engine)
    return switching_parts[0]


def test_every_machine_builds_its_own_policy_from_one_settings_record():
    """One settings record builds a machine per shot, each its own policy."""
    settings = _switching()
    first_engine = engine_module.Engine()
    second_engine = engine_module.Engine()

    first = _built(settings, first_engine)
    second = _built(settings, second_engine)

    assert first.policy is not second.policy
    assert first.strong_redecode is not second.strong_redecode


def test_a_switching_record_builds_the_policy_its_run_binds():
    """A record that subclasses the switching slot brings its own policy."""
    settings = _switching(_OwnPolicySwitching)
    engine = engine_module.Engine()

    switching = _built(settings, engine)

    assert isinstance(switching.policy, _OwnPolicy)


_DOUBLE_WINDOW = strong_window_shapes.DoubleWindow.Settings()
_TWO_SOLVES = complementary.ComplementaryGap.Settings()
_PAST_THE_ENTRY_CHANGES = [
    {"strong_window": _DOUBLE_WINDOW},
    {"confidence": _TWO_SOLVES},
]
_PAST_THE_ENTRY_IDS = ["double_window", "two_solves"]


@pytest.mark.parametrize(
    "change", _PAST_THE_ENTRY_CHANGES, ids=_PAST_THE_ENTRY_IDS
)
def test_an_edge_past_the_entry_refuses_what_only_the_entry_does(change):
    """An edge past the entry waits for stage 5 on what reads its source.

    An absorbing region and a gap of two solves each work on the entry
    pool's windows alone.
    """
    deep_switching = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, **change
    )
    chain = layout_tests.chain_machine(deep_switching)
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()

    with pytest.raises(NotImplementedError, match=r"stage 5"):
        escalation_build.build_switching(layout, engine)


def test_an_edge_past_the_entry_builds_on_one_solve():
    chain = layout_tests.chain_machine()
    layout = layout_tests.laid_out(chain)
    engine = engine_module.Engine()

    parts = escalation_build.build_switching(layout, engine)

    edges = [part.edge.name for part in parts]
    assert edges == ["switching", layout_tests.DEEP_SWITCHING]


def test_an_edge_past_a_double_window_may_start_beside_its_source():
    """Each edge checks the run with its own window row.

    The entry's double window ships its boundaries eagerly and starts
    late; the deep edge redoes the strong revision, may start with it,
    and ships no boundary of the entry pool's.
    """
    speculates = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, run_both_at_once=True
    )
    chain = layout_tests.chain_machine(speculates, _DOUBLE_WINDOW)
    task = collect.Task("chain", chain)

    machine = machine_module.Machine.build(
        task.machine, 0, None, task.online_threshold, layout=task.layout
    )

    deep_edge = machine.windows.window_manager.edges.edges[1]
    assert deep_edge.policy.decodes_both_at_once(None)


def test_an_edge_past_the_entry_reads_its_signal_off_its_source():
    """A cluster gap at 0 dB keeps every window: the deep pool answers none.

    The deep edge's join computes the gap from the strong pool's union-
    find solve; a deep verdict with no gap would escalate them all.
    """
    zero_decibels = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=0.0
    )
    keeps_all = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, threshold=zero_decibels
    )
    redo = strong_window_shapes.RedoWindow.Settings()
    chain = layout_tests.chain_machine(keeps_all, redo)
    always = verdicts.Always.Settings()
    escalates_all = chain.with_settings("switching", verdict=always)
    task = collect.Task("chain", escalates_all)

    shot = collect.run_shot(task, 0)

    snapshot = shot.machine.control.pauli_frame.snapshot()
    tiers = {record.tier for record in snapshot.records}
    assert tiers == {"strong"}


@pytest.mark.parametrize("seed", [1, 2])
def test_a_deep_redo_on_the_strong_decoder_gives_the_strong_answers(seed):
    """Union-find at both depths: each deep answer is the strong one.

    Both edges escalate every window. The strong pool decodes double
    windows and the deep pool redoes each strong revision on its region,
    faces and error model. A redo on a model of its own answered seam
    windows differently on these seeds at d=3, p=0.01.
    """
    chain = _double_then_deep_redo()
    task = collect.Task("chain", chain)
    machine = machine_module.Machine.build(
        task.machine, seed, None, task.online_threshold, layout=task.layout
    )
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    machine.run()

    strong = _answers_of(requests, "strong")
    deep = _answers_of(requests, "deep")

    assert len(strong) == 4
    assert deep == strong


def _double_then_deep_redo():
    """Union-find double windows, every one redone by union-find again."""
    always = verdicts.Always.Settings()
    no_gap = no_signal.NoSignal.Settings()
    deep = dataclasses.replace(
        layout_tests.DEEP_SWITCHING_SLOT, confidence=no_gap, verdict=always
    )
    entry = dataclasses.replace(deep, strong_window=declared_run.DOUBLE_WINDOW)
    base = machines.weak_baseline(3, 0.01, 1.0)
    middle = layout_tests.MIDDLE_POOL
    two_pools = machines.with_strong_tier(base, middle, entry)
    return machines.with_escalation(
        two_pools, machines.STRONG_POOL, "deep", middle, deep
    )


def _answers_of(requests, pool_name: str) -> dict:
    """Each window's logical observables, as one pool answered it."""
    answers = {}
    for ended in requests.ended:
        job = ended.job
        if job.request_key.pool.word == pool_name:
            observables = ended.result.logical_observables
            answers[job.strong_decode_for] = tuple(observables)
    return answers
