"""The decoders part: each tier's units and the manager that schedules them.

A unit is the tier's decoder between its fetch and release stages. The
units and the managers' pools are one record, gem5's
CacheConfig.config_cache shape (configs/common/CacheConfig.py), built
before the parts because the window models are compiled for the units.
"""

import dataclasses
import functools
from typing import Optional

import decsim.build.layout as layout_module
import decsim.config as config
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.detection_events as detection_events_module
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry

# The name the tier's event-detection stage carries in the trace, the
# stage ledger and the narrator log.
FORMATION_STAGE = "detection_event_formation"
# The path the decode side's own statistics are kept at: the managers'
# card's path, since the jobs waiting over every pool are theirs.
MANAGERS_PATH = "decoder_manager"
MANAGERS_STATISTICS = (
    statistics_registry.Statistic(
        "queued",
        statistic_kinds.History,
        "jobs waiting over every pool",
        "jobs",
    ),
)


@dataclasses.dataclass(frozen=True)
class PoolUnits:
    """What one pool's manager is built over: its unit and its pool.

    unit is None for a run with no decoder, whose one pool holds an idle
    unit.
    """

    plan: layout_module.PoolPlan
    unit: Optional[ports.Decoder]
    settings: decoder_pool_module.PoolSettings


@dataclasses.dataclass(frozen=True)
class Decoders:
    """The decode side: each pool's decoder manager over its units.

    The entry pool's manager decodes the planner's windows; a switching
    run has a manager over each edge's target too (LATTE 2509.03954
    lines 705-720). They share one ledger of escalated requests, since
    a source's side opens a request and its target's side serves it.
    """

    pools: tuple
    managers: tuple
    strong_requests: strong_requests_module.StrongRequests

    @classmethod
    def build(
        cls,
        manager_settings: decoder_settings.DecoderManagerSettings,
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        pools: tuple,
        target_by_source: dict,
    ) -> "Decoders":
        """Each manager over its pool, bound to its pool's unit.

        target_by_source names the pool each source escalates to.
        """
        clocked_manager = config.with_machine_clock(
            manager_settings, machine_clock
        )
        strong_requests = strong_requests_module.StrongRequests(
            target_by_source
        )
        target_pools = target_by_source.values()
        targets = set(target_pools)
        managers = []
        for pool in pools:
            is_target = pool.plan.pool in targets
            manager = _decoder_manager(clocked_manager, engine, pool, is_target)
            manager.decoder = pool.unit
            manager.strong_requests = strong_requests
            managers.append(manager)
        return cls(
            pools=pools,
            managers=tuple(managers),
            strong_requests=strong_requests,
        )

    @property
    def decoder_manager(self) -> decoder_manager_module.DecoderManager:
        """The entry pool's manager, which decodes the planner's windows."""
        return self.managers[0]

    @property
    def primary_decoder(self) -> Optional[ports.Decoder]:
        """The entry pool's unit."""
        entry = self.pools[0]
        return entry.unit

    @property
    def target_decoders(self) -> tuple:
        """The unit of each pool an edge leads to, in graph order."""
        units = []
        for pool in self.pools[1:]:
            if pool.unit is not None:
                units.append(pool.unit)
        return tuple(units)

    def manager_of(
        self, pool: window_records.PoolId
    ) -> decoder_manager_module.DecoderManager:
        """The manager of one pool."""
        for manager in self.managers:
            if manager.pool_id == pool:
                return manager
        raise KeyError(pool)

    def attach_statistics(
        self, run_statistics: statistics_registry.Registry
    ) -> None:
        """Each manager's statistics at its pool's path.

        The entry pool's are at its settings path (weak_decoder, or
        strong_decoder for a strong-only run), as are each target's.
        """
        queues = []
        for pool, manager in zip(self.pools, self.managers, strict=True):
            path = pool.plan.names.statistics_path
            manager.attach_statistics(run_statistics, path)
            queues.append(manager.queue)
        over_pools = run_statistics.group(MANAGERS_PATH, MANAGERS_STATISTICS)
        _count_the_jobs_waiting_over_pools(over_pools.queued, queues)

    def start(self) -> None:
        """Let each manager hear its units before any window model exists."""
        for manager in self.managers:
            manager.start()

    def check_settled(self) -> None:
        """Every decode each manager admitted has finished."""
        for manager in self.managers:
            manager.check_decode_work_settled()

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes.

        The entry manager's transport is the one input transport.
        """
        roots = []
        for pool, manager in zip(self.pools, self.managers, strict=True):
            names = pool.plan.names
            roots.append((names.unit_seed, pool.unit))
            roots.append((names.manager_seed, manager))
        input_transport = self.decoder_manager.input_transport()
        roots.append(("decoder_memory_transfer", input_transport))
        return tuple(roots)


def build_decoder_unit(
    tier_settings: Optional[decoder_settings.DecoderPoolSettings],
    seat: str,
    machine_clock: Optional[config.Clock],
    formation: Optional[detection_events_module.TierFormation],
    signal: Optional[ports.ConfidenceSignal],
) -> Optional[staged_decoder.StagedDecoder]:
    """The decoder unit of one pool, at its seat; None for an empty slot.

    A tier handed a confidence signal must produce the evidence it reads.
    """
    if tier_settings is None:
        return None
    algorithm_settings = tier_settings.algorithm
    algorithm = algorithm_settings.build()
    if signal is not None:
        # the record the run wrote: a priced row's results word is its
        # latency, which names no decoder
        record = type(algorithm_settings)
        _check_serves_the_confidence(
            algorithm, record.__qualname__, seat, signal
        )
    return _staged_unit(tier_settings, algorithm, formation, machine_clock)


def build_pool_units(
    layout: layout_module.Layout,
    machine_clock: Optional[config.Clock],
    detection_events: ports.DetectionEventPlacement,
    signal_by_pool: dict,
) -> tuple:
    """Each pool's unit and its manager's pool, entry first.

    Each pool seated by the detection event placement gets its own
    event-detection logic, so a round two pools read is formed and
    charged by each. A pool whose results meet an edge's verdict must
    produce the evidence that edge's confidence signal reads
    (signal_by_pool).
    """
    built = []
    for index, plan in enumerate(layout.pools):
        names = plan.names
        formation = _tier_formation(detection_events, names.seat)
        signal = signal_by_pool.get(plan.pool)
        unit = build_decoder_unit(
            plan.settings, names.seat, machine_clock, formation, signal
        )
        blocks_unit = False
        if index == 0:
            blocks_unit = _blocks_unit(plan.settings)
        settings = _pool(plan.settings, names.queue, formation, blocks_unit)
        pool = PoolUnits(plan, unit, settings)
        built.append(pool)
    return tuple(built)


def _pool(
    tier_settings: Optional[decoder_settings.DecoderPoolSettings],
    name: str,
    formation: Optional[detection_events_module.TierFormation],
    blocks_unit: bool,
) -> decoder_pool_module.PoolSettings:
    """One manager's pool; one idle unit for a run with no decoder."""
    if tier_settings is None:
        return decoder_pool_module.PoolSettings(
            name=name,
            unit_count=1,
            blocks_unit=blocks_unit,
            formation=formation,
        )
    return decoder_pool_module.PoolSettings(
        name=name,
        unit_count=tier_settings.unit_count,
        capacity_bits=tier_settings.unit_memory.bits,
        copies_input=tier_settings.copies_input,
        blocks_unit=blocks_unit,
        formation=formation,
    )


def _blocks_unit(
    tier_settings: Optional[decoder_settings.DecoderPoolSettings],
) -> bool:
    """The blocking rule of the pool that decodes the windows.

    An edge's target never blocks: its result waits in its unit's
    output slot (decoder_unit.py hold_output), so result_blocks_unit is
    read on the entry pool alone.
    """
    if tier_settings is None:
        return False
    return tier_settings.result_blocks_unit


def _tier_formation(
    detection_events: ports.DetectionEventPlacement, seat: str
) -> Optional[detection_events_module.TierFormation]:
    """This tier's event-detection logic; None when rounds arrive formed."""
    if not detection_events.forms_at(seat):
        return None
    return detection_events_module.TierFormation(detection_events, seat)


def _check_serves_the_confidence(
    algorithm, record_name: str, seat: str, signal: ports.ConfidenceSignal
) -> None:
    """Refuse a weak tier that cannot serve the run's confidence signal.

    A priced card is not refused: a forced pair is two decodes, each
    charged the card's time.
    """
    required = signal.decoder_evidence_requirement
    missing = required - algorithm.decoder_evidence
    if not missing:
        return
    reason = _missing_evidence_reason(algorithm, missing, signal)
    signal_name = signal.source.method
    raise ValueError(
        f"{seat} {record_name} cannot serve the confidence "
        f"{signal_name}: {reason}"
    )


def _missing_evidence_reason(algorithm, missing, signal) -> str:
    """The decoder's own reason it cannot serve, else the signal's."""
    reasons = algorithm.missing_evidence_reasons
    for member in sorted(missing, key=_evidence_order):
        reason = reasons.get(member)
        if reason is not None:
            return reason
    return signal.evidence_refusal


def _evidence_order(member) -> str:
    return member.value


def _staged_unit(
    tier_settings: decoder_settings.DecoderPoolSettings,
    algorithm,
    formation: Optional[detection_events_module.TierFormation],
    machine_clock: Optional[config.Clock],
) -> staged_decoder.StagedDecoder:
    """The algorithm between its stages, on the tier's engine clock.

    Event detection first when the rounds reach it raw, then fetch, the
    algorithm and release, each priced once a job and once a round.
    """
    clocked_engine = config.with_machine_clock(
        tier_settings.engine, machine_clock
    )
    before = []
    formation_stage = _formation_stage(formation)
    if formation_stage is not None:
        before.append(formation_stage)
    fetch = staged_decoder.MemoryFetchStage(
        "fetch",
        cycles_per_job=clocked_engine.fetch_cycles_per_job,
        cycles_per_round=clocked_engine.fetch_cycles_per_round,
        word_bits=tier_settings.unit_memory.word_bits,
    )
    before.append(fetch)
    release = staged_decoder.DecoderStage(
        "release",
        cycles_per_job=clocked_engine.release_cycles_per_job,
        cycles_per_round=clocked_engine.release_cycles_per_round,
    )
    timing = staged_decoder.UnitTiming(
        before=tuple(before),
        after=(release,),
        clock=clocked_engine.clock,
    )
    return staged_decoder.StagedDecoder(algorithm, timing)


def _formation_stage(
    formation: Optional[detection_events_module.TierFormation],
) -> Optional[detection_events_module.DetectionEventFormationStage]:
    """This tier's event-detection stage; None when the tier forms none.

    The stage is the tier's own hardware in front of its decoder core
    (LILLIPUT's Event Detection Logic block, 2108.06569 lines 499-510;
    Yang 2605.04892 lines 1273-1275, Table I lines 1049-1052), so it is
    a stage of the unit's timing, on the detection_events card's clock.
    """
    if formation is None:
        return None
    return detection_events_module.DetectionEventFormationStage(
        FORMATION_STAGE, formation=formation
    )


def _decoder_manager(
    manager_settings: decoder_settings.DecoderManagerSettings,
    engine: engine_module.Engine,
    pool: PoolUnits,
    is_target: bool,
) -> decoder_manager_module.DecoderManager:
    """One manager over its pool, on the run's one manager card.

    bulk_strong merges the queue of every pool an edge escalates to.
    """
    scheduler = manager_settings.scheduler.build()
    plan = pool.plan
    merges_strong = manager_settings.bulk_strong and is_target
    return decoder_manager_module.DecoderManager(
        engine,
        scheduler=scheduler,
        pool_settings=pool.settings,
        pool_id=plan.pool,
        merges_strong=merges_strong,
        clock=manager_settings.clock,
        dispatch_cycles=manager_settings.dispatch_cycles,
    )


def _count_the_jobs_waiting_over_pools(queued, queues: list) -> None:
    """The jobs waiting over the pools as each queue's depth changes.

    The total is the sum of each queue's last depth, sampled whenever
    either changes; nothing is kept when the run records no statistics.
    """
    if queued is statistic_kinds.DISABLED:
        return
    note = functools.partial(_note_jobs_waiting, queued, queues)
    for queue in queues:
        queue.trace.depth_changed.connect(note)


def _note_jobs_waiting(queued, queues: list, tick: int, depth: int) -> None:
    del tick, depth
    depths = [queue.statistics.queued.level for queue in queues]
    total = sum(depths)
    queued.set(total)
