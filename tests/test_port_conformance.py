"""Port-conformance suite (plan Task 21, spec §8 test 8).

One case per v1.0 port: every shipped implementation satisfies its
runtime_checkable Protocol, checked on parts wired into a REAL built completed_run
wherever one exists (so the conformance holds for what spec.build actually
assembles, not just for hand-made instances).
"""
import pytest

from decsim import protocols
from decsim.policies import Eager, Held
from decsim.codes import SurfaceCodeModel
from decsim.decoders import (PerRoundDecoder, PresetLatencyDecoder,
                             SAMPLED_CONFIDENCE_SOURCE,
                             SampledConfidenceDecoder, SwitchingRouter)
from decsim.engine import Engine
from decsim.factories import (DistillationFactory, DistillLevel,
                              InfiniteFactory, MultiLevelDistillationFactory)
from decsim.frontends.circuit import CircuitFrontend
from decsim.policies import ExtendStream, Ignore, SeparateDecodeJobs
from decsim.layouts import UniformLayout
from decsim.run_spec import simulate
from decsim.message import Operation
from decsim.metrics import (DecodeBacklog, DecoderUtilization, ReadyQueueStats,
                            StrongDecoderBacklog)
from decsim.run_spec import RunSpec
from decsim.planner import (CodeRounds, FixedRounds,
                           GateRounds, PerOpRounds, TemporalRounds)
from decsim.schedulers import (EarliestDeadlineScheduler, EnqueueTimeDeadline,
                               FifoScheduler, ReactionPathDeadline)
from decsim.schemes import (NaiveOnlineScheme, ParallelWindowScheme,
                            SlidingWindowScheme)
from decsim.switching import Baseline, Switching
from decsim.soft_output import UnionFindDecoder
from decsim.window_interactions import DefaultWindowInteraction


class AlternateWindowInteraction(DefaultWindowInteraction):
    pass


def _ops():
    return CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()


def _base_kwargs(**overrides):
    # rounds_policy pinned here (was presets.today's job; module removed
    # at owner direction 2026-07-06) — SWAPS overrides still replace it
    kwargs = dict(ops=_ops(), d=3, decoder=PerRoundDecoder(0.5), num_units=2,
                  rounds_policy=FixedRounds(11))
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def completed_run():
    return RunSpec(**_base_kwargs()).build()


#==================================================================
# CONFORMANCE: shipped implementations satisfy their Protocols
#==================================================================

def test_wired_completed_run_parts_satisfy_their_ports(completed_run):
    checks = [
        (completed_run.window_manager.scheme, protocols.DecodingScheme),          # port 6
        (completed_run.window_manager.deadline_policy, protocols.DeadlinePolicy),  # port 13
        (completed_run.window_manager.boundary_policy, protocols.BoundaryPolicy),  # port 16
        (completed_run.window_manager.window_interaction,
         protocols.WindowInteraction),                         # port 21
        (completed_run.decoder_manager, protocols.ResourcePool),                    # port 12
        (completed_run.decoder_manager.scheduler, protocols.Scheduler),             # port 11
        (completed_run.chip.idle_policy, protocols.IdlePolicy),          # port 17
        (completed_run.chip.source, protocols.SyndromeSource),                   # port 2
        (completed_run.controller, protocols.Controller),                 # port 14
        (completed_run.orchestrator, protocols.Orchestrator),          # port 15
        (completed_run.factory, protocols.MagicStateFactory),                      # port 19
        (completed_run.window_manager.strategy,
         protocols.DecodingStrategy),                                   # port 10
    ]
    for part, port in checks:
        assert isinstance(part, port), \
            f"{type(part).__name__} does not satisfy {port.__name__}"


def test_every_shipped_part_family_satisfies_its_port():
    engine = Engine(verbose=False)
    code = SurfaceCodeModel(d=3)
    families = {
        protocols.InputFrontend: [CircuitFrontend([])],                       # port 1
        protocols.CodeModel: [code],                                          # port 3
        protocols.LayoutModel: [UniformLayout(code)],                         # port 4
        protocols.RoundsPolicy: [FixedRounds(11), GateRounds(),
                             PerOpRounds({}), CodeRounds(),
                             TemporalRounds(d_m=3)],                 # port 5
        protocols.DecodingScheme: [SlidingWindowScheme(), NaiveOnlineScheme(),
                             ParallelWindowScheme()],                # port 6
        protocols.Decoder: [PerRoundDecoder(0.5), PresetLatencyDecoder(1.0),
                        SampledConfidenceDecoder(PerRoundDecoder(1.0), 0.5),
                        UnionFindDecoder(PerRoundDecoder(0.5))],  # 8
        protocols.DecodingStrategy: [Baseline(),
                                 Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5)],  # 10
        protocols.Scheduler: [FifoScheduler(), EarliestDeadlineScheduler()],  # 11
        protocols.DeadlinePolicy: [EnqueueTimeDeadline(),
                               ReactionPathDeadline(slack_ticks=100)],  # 13
        protocols.BoundaryPolicy: [Eager(), Held()],                     # port 16
        protocols.WindowInteraction: [DefaultWindowInteraction()],        # port 21
        protocols.IdlePolicy: [Ignore(), ExtendStream(), SeparateDecodeJobs()],  # 17
        protocols.MagicStateFactory: [InfiniteFactory(engine)],                    # port 19
    }
    for port, impls in families.items():
        for impl in impls:
            assert isinstance(impl, port), \
                f"{type(impl).__name__} does not satisfy {port.__name__}"


def test_metrics_satisfy_the_metric_port(completed_run):
    for metric in (DecoderUtilization(completed_run.decoder_manager),
                   ReadyQueueStats(completed_run.decoder_manager),
                   DecodeBacklog(completed_run.window_manager,
                                 completed_run.decoder_manager),
                   StrongDecoderBacklog(completed_run.window_manager,
                                        completed_run.decoder_manager)):
        assert isinstance(metric, protocols.Metric)                      # port 20


def test_strategy_services_satisfy_their_seam(completed_run):
    assert isinstance(completed_run.decoder_manager.services, protocols.StrategyServices) \
        or isinstance(completed_run.window_manager.services, protocols.StrategyServices)


def test_strong_backlog_is_global_and_has_no_pool_compatibility_argument(
    completed_run,
):
    with pytest.raises(TypeError, match="unexpected keyword argument 'pool'"):
        StrongDecoderBacklog(
            completed_run.window_manager,
            completed_run.decoder_manager,
            pool="strong",
        )


def test_factories_conform_and_smoke(completed_run):
    engine = Engine(verbose=False)
    factories = [
        InfiniteFactory(engine),
        DistillationFactory(engine, num_units=1, cycle_ticks=10,
                            decode_service=completed_run.decoder_manager, corr_rounds=1, n_corr=1),
        MultiLevelDistillationFactory(engine, [DistillLevel(units=1, d=3)],
                                      W_ticks=10),
    ]
    for factory in factories:
        assert isinstance(factory, protocols.MagicStateFactory)
        ticket = factory.request(0, lambda: None)   # smoke: Ticket API
        assert hasattr(ticket, "cancel")
        factory.shutdown()


def test_memory_model_port_observes_a_real_run():
    """Port 18: a MemoryModel wired through RunSpec sees every retained
    fragment stored and every one of them evicted by the end of the run."""
    class RecordingModel:
        def __init__(self):
            self.stored, self.evicted = [], []

        def store(self, key, payload):
            self.stored.append(key)

        def evict(self, key):
            self.evicted.append(key)

    model = RecordingModel()
    assert isinstance(model, protocols.MemoryModel)
    simulate(RunSpec(**_base_kwargs(), memory_model=model))
    assert model.stored                                  # observed retention
    assert sorted(model.evicted) == sorted(model.stored)  # no leaks at end
