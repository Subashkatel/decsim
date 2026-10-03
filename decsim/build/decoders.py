"""The decoders part: each tier's units and the manager that schedules them.

A tier's algorithm is a decoder row's Settings record, and the unit is
the decoder it builds between its fetch and release stages. The two
tiers' units and the managers' pools are one record, gem5's
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

    active is the unit of the tier that decodes the plan's windows, None
    on a run with no decoder; strong is the strong tier's unit on a
    switching run, else None. chip is the chip's manager's pool, of the
    tier that decodes the plan's windows; host is the host's manager's,
    of the strong tier, on a switching run only.
    """

    active: Optional[ports.Decoder]
    strong: Optional[ports.Decoder]
    chip: decoder_pool_module.PoolSettings
    host: Optional[decoder_pool_module.PoolSettings]


@dataclasses.dataclass(frozen=True)
class Decoders:
    """The decode side: the units of both tiers and the managers over them.

    The chip's manager schedules the tier that decodes the plan's windows.
    A switching run has the host's manager too, over the strong
    tier's units (LATTE 2509.03954 lines 705-720), and the two share one
    ledger of strong requests, since the chip side opens a strong request
    and the host side serves it. Nothing here reaches outward: the other
    parts are handed the managers as their decode queues.
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
        """Each manager over its pool, bound to its tier's unit.

        A manager card that names no clock dispatches on machine_clock. A
        switching run's policy is bound by the switching part.
        """
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
        """Let each manager hear its rows before any window model exists."""
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
):
    """The decoder unit of one tier, weak or strong, as the root builds it.

    The tier's algorithm decodes inside a StagedDecoder whose fetch and
    release stages are cycles of the tier's clock, with this tier's
    event-detection logic in front of them when the tier's decoder is a
    seat of the run's detection event placement (formation). A tier
    handed a confidence signal must produce the evidence it reads. None
    when the tier's slot is empty.
    """
    if tier_settings is None:
        return None
    algorithm_settings = tier_settings.algorithm
    algorithm = algorithm_settings.build()
    if signal is not None:
        _check_serves_the_confidence(
            algorithm, algorithm_settings.name, tier, signal
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

    The chip's pool holds units of window_tier, whose settings are
    window_decoder and whose unit serves the run's confidence signal; a
    run that escalates gives the strong tier a pool of its own. Each tier
    seated by the detection event placement gets its own event-detection
    logic, so a round both tiers read is formed and charged by each.
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
    """One manager's pool, from the settings of the tier it holds units of.

    A run with no decoder holds one unit of a tier's defaults, which
    nothing is sent to.
    """
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

    The strong pool of a switching run never blocks: a strong re-decode
    frees its unit's compute at its end and its result waits in the
    unit's output slot until the window side takes it (decoder_unit.py
    hold_output), so <tier>.result_blocks_unit is read on the primary
    tier alone.
    """
    if tier_settings is None:
        return False
    return tier_settings.result_blocks_unit


def _tier_formation(
    detection_events: ports.DetectionEventPlacement, seat: str
) -> Optional[detection_events_module.TierFormation]:
    """This tier's event-detection logic; None when rounds arrive formed.

    The placement decides whether the tier's decoder is a seat. A source
    with no recipes leaves it nothing to convert and the same stage to
    pay.
    """
    if not detection_events.forms_at(seat):
        return None
    return detection_events_module.TierFormation(detection_events, seat)


def _check_serves_the_confidence(
    algorithm, row_name: str, tier: str, signal: ports.ConfidenceSignal
) -> None:
    """Refuse a weak tier that cannot serve the run's confidence signal.

    A confidence is the decoder's own, so the signal's evidence
    requirement is held against the row's own declaration. A priced card
    is not refused: it prices one decode of one window, and a forced pair
    is two decodes, so the card is charged once per forced solve.
    """
    required = signal.decoder_evidence_requirement
    missing = required - algorithm.decoder_evidence
    if not missing:
        return
    reason = _missing_evidence_reason(algorithm, missing, signal)
    signal_name = signal.source.method
    raise ValueError(
        f"{tier}_decoder {row_name!r} cannot serve the confidence "
        f"{signal_name}: {reason}"
    )


def _missing_evidence_reason(algorithm, missing, signal) -> str:
    """Why this row cannot serve this signal, cited.

    A row that a reader would expect to produce the evidence says why it
    does not; every other row gets the signal's own sentence about what
    a decoder must do to report it.
    """
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

    This tier's event-detection logic first when the rounds reach it raw,
    then the fetch stage, then the algorithm, then the release stage,
    each stage priced once a job and once a round. An engine card that
    names no clock counts on machine_clock.
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
    Yang's preprocessing stage inside the decoder subtotal, 2605.04892
    lines 1273-1275, Table I lines 1049-1052), so it is a stage of the
    unit's timing rather than a component the manager schedules: the
    decoder row behind it never learns that its rounds were raw. Its
    cycles are the detection_events card's, on that card's clock.
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
