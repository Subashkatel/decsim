"""The decoders part: each tier's units and the manager that schedules them.

A unit is the tier's decoder between its fetch and release stages. The
units and the managers' pools are one record, gem5's
CacheConfig.config_cache shape (configs/common/CacheConfig.py), built
before the parts because the window models are compiled for the units.
"""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.detection_events as detection_events_module
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.windows as window_records

# The name the tier's event-detection stage carries in the trace, the
# stage ledger and the narrator log.
FORMATION_STAGE = "detection_event_formation"


@dataclasses.dataclass(frozen=True)
class DecoderPool:
    """The tiers' units and the managers' pool knobs.

    active and chip are the unit and pool of the tier that decodes the
    plan's windows (active None with no decoder); strong and host are
    the strong tier's, on a switching run only.
    """

    active: Optional[ports.Decoder]
    strong: Optional[ports.Decoder]
    chip: decoder_pool_module.PoolSettings
    host: Optional[decoder_pool_module.PoolSettings]


@dataclasses.dataclass(frozen=True)
class Decoders:
    """The decode side: the units of both tiers and the managers over them.

    A switching run has the host's manager over the strong tier's units
    too (LATTE 2509.03954 lines 705-720); the two share one ledger of
    strong requests, since the chip side opens a strong request and the
    host side serves it.
    """

    primary_decoder: Optional[ports.Decoder]
    strong_decoder: Optional[ports.Decoder]
    strong_requests: strong_requests_module.StrongRequests
    decoder_manager: decoder_manager_module.DecoderManager
    strong_decoder_manager: Optional[decoder_manager_module.DecoderManager]

    @classmethod
    def build(
        cls,
        manager_settings: decoder_settings.DecoderManagerSettings,
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        pool: DecoderPool,
    ) -> "Decoders":
        """Each manager over its pool, bound to its tier's unit."""
        clocked_manager = config.with_machine_clock(
            manager_settings, machine_clock
        )
        strong_requests = strong_requests_module.StrongRequests()
        decoder_manager = _decoder_manager(clocked_manager, engine, pool.chip)
        decoder_manager.decoder = pool.active
        decoder_manager.strong_requests = strong_requests
        strong_decoder_manager = None
        if pool.host is not None:
            strong_decoder_manager = _decoder_manager(
                clocked_manager, engine, pool.host
            )
            strong_decoder_manager.decoder = pool.strong
            strong_decoder_manager.strong_requests = strong_requests
        return cls(
            primary_decoder=pool.active,
            strong_decoder=pool.strong,
            strong_requests=strong_requests,
            decoder_manager=decoder_manager,
            strong_decoder_manager=strong_decoder_manager,
        )

    def start(self) -> None:
        """Let each manager hear its units before any window model exists."""
        self.decoder_manager.start()
        if self.strong_decoder_manager is not None:
            self.strong_decoder_manager.start()

    def check_settled(self) -> None:
        """Every decode either manager admitted has finished."""
        self.decoder_manager.check_decode_work_settled()
        if self.strong_decoder_manager is not None:
            self.strong_decoder_manager.check_decode_work_settled()

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes."""
        input_transport = self.decoder_manager.input_transport()
        return (
            ("primary_decoder", self.primary_decoder),
            ("strong_decoder", self.strong_decoder),
            ("decoder_manager", self.decoder_manager),
            ("strong_decoder_manager", self.strong_decoder_manager),
            ("decoder_memory_transfer", input_transport),
        )


def build_decoder_unit(
    tier_settings: Optional[decoder_settings.DecoderPoolSettings],
    tier: str,
    machine_clock: Optional[config.Clock],
    formation: Optional[detection_events_module.TierFormation],
    signal: Optional[ports.ConfidenceSignal],
) -> Optional[staged_decoder.StagedDecoder]:
    """The decoder unit of one tier, weak or strong; None for an empty slot.

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
            algorithm, record.__qualname__, tier, signal
        )
    return _staged_unit(tier_settings, algorithm, formation, machine_clock)


def build_decoder_pool(
    window_decoder: Optional[decoder_settings.DecoderPoolSettings],
    strong_decoder: Optional[decoder_settings.DecoderPoolSettings],
    window_tier: window_records.DecoderTier,
    escalates: bool,
    machine_clock: Optional[config.Clock],
    detection_events: ports.DetectionEventPlacement,
    signal: Optional[ports.ConfidenceSignal],
) -> DecoderPool:
    """The tiers' units and the managers' pools.

    Each tier seated by the detection event placement gets its own
    event-detection logic, so a round both tiers read is formed and
    charged by each.
    """
    active_tier = window_tier.value
    active_seat = f"{active_tier}_decoder"
    active_formation = _tier_formation(detection_events, active_seat)
    active = build_decoder_unit(
        window_decoder, active_tier, machine_clock, active_formation, signal
    )
    blocks_unit = _blocks_unit(window_decoder)
    chip = _pool(
        window_decoder, decode_queue.DEFAULT_POOL, active_formation, blocks_unit
    )
    if not escalates:
        return DecoderPool(active=active, strong=None, chip=chip, host=None)
    strong_formation = _tier_formation(detection_events, "strong_decoder")
    strong = build_decoder_unit(
        strong_decoder, "strong", machine_clock, strong_formation, None
    )
    host = _pool(
        strong_decoder, decode_queue.STRONG_POOL, strong_formation, False
    )
    return DecoderPool(active=active, strong=strong, chip=chip, host=host)


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

    The strong pool never blocks: a strong result waits in its unit's
    output slot (decoder_unit.py hold_output), so result_blocks_unit is
    read on the primary tier alone.
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
    algorithm, record_name: str, tier: str, signal: ports.ConfidenceSignal
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
        f"{tier}_decoder {record_name} cannot serve the confidence "
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
    pool_settings: decoder_pool_module.PoolSettings,
) -> decoder_manager_module.DecoderManager:
    """One manager over its pool, on the run's one manager card."""
    scheduler = manager_settings.scheduler.build()
    return decoder_manager_module.DecoderManager(
        engine,
        scheduler=scheduler,
        pool_settings=pool_settings,
        bulk_strong=manager_settings.bulk_strong,
        clock=manager_settings.clock,
        dispatch_cycles=manager_settings.dispatch_cycles,
    )
