"""Build the decoder units of both tiers and the pool the manager schedules.

A tier's kind names a row of decoders/settings.py's DECODERS, and the
unit is that algorithm between its fetch and release stages. The router
over the two tiers and the manager's pools are one record, gem5's
CacheConfig.config_cache shape (configs/common/CacheConfig.py).
"""

import dataclasses
from typing import Any, Optional

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoders as decoders
import decsim.decoders.detection_events as detection_events_module
import decsim.decoders.memory_rounds as memory_rounds_module
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.escalation.settings as escalation_settings
import decsim.observe.settings as observe_settings
import decsim.ports as ports
import decsim.settings as machine_settings
import decsim.tables as tables
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# The name the tier's event-detection stage carries in the trace, the
# stage ledger and the narrator log.
FORMATION_STAGE = "detection_event_formation"


@dataclasses.dataclass(frozen=True)
class DecoderPool:
    """The router over the tiers' units and the manager's pool knobs."""

    router: Any
    active: Any
    unit_pools: dict
    decoder_memory: Optional[decoder_memory_module.DecoderMemoryConfig]
    scheduler: Any
    # pool name -> whether that tier's unit is given a copy of the
    # rounds it decodes (weak_decoder.input, strong_decoder.input)
    copies_input_by_pool: dict
    # pool name -> that tier's event-detection logic, for a run that
    # seats the former at the tier's decoder (detection_events.formed_at)
    formation_by_pool: dict
    # pool name -> whether a finished decode holds its unit until the
    # window side reads the result (<tier>.result_blocks_unit)
    blocks_unit_by_pool: dict


def build_decoder_unit(
    settings: machine_settings.MachineSettings,
    tier: str,
    policy,
    formation: Optional[detection_events_module.TierFormation] = None,
):
    """The decoder unit of one tier, weak or strong, as the root builds it.

    A named row decodes for real inside a StagedDecoder whose fetch and
    release stages are cycles of the tier's clock, with this tier's
    event-detection logic in front of them when the tier's decoder is a
    seat of the run's detection event placement (formation); a number is a
    fixed core latency on the MWPM path. The tier that decodes the plan's
    windows carries the Tesseract referee when the observation asks for
    it, and under a switching escalation must produce the evidence the
    run's confidence signal reads. A Python-built decoder is returned as
    it is. None when the tier names no decoder.
    """
    tier_settings = settings.decoder_settings_for(tier)
    if tier_settings.decoder is not None:
        return tier_settings.decoder
    if tier_settings.kind is None:
        return None
    algorithm = _algorithm(tier_settings, tier)
    is_active = tier == policy.primary_tier.value
    decides_on_a_confidence = policy.decides_on_a_confidence
    if is_active and decides_on_a_confidence:
        _check_serves_the_confidence(
            algorithm, tier_settings.kind, tier, settings.escalation
        )
    check = tables.row(
        observe_settings.WINDOW_CHECKS,
        "observation.check_windows_with",
        settings.observation.check_windows_with,
    )
    if is_active and check is not None:
        algorithm = check(algorithm)
    return _staged_unit(tier_settings, algorithm, formation)


def build_decoder_pool(
    settings: machine_settings.MachineSettings,
    plan: plan_build.Plan,
    policy,
    detection_events: ports.DetectionEventPlacement,
) -> DecoderPool:
    """The router over the tiers and the manager's pools.

    Switching puts two units behind one router: the strong pool serves
    escalated jobs. Any other escalation routes every job to the tier
    that decodes the plan's windows. Each tier whose decoder is a seat
    of the run's detection event placement gets its own event-detection
    logic, so a round both tiers read is formed and charged by each.
    """
    manager = settings.decoder_manager
    scheduler = manager.scheduler
    if scheduler is None:
        scheduler = schedulers.FifoScheduler()
    weak_formation = _tier_formation(detection_events, "weak_decoder")
    strong_formation = _tier_formation(detection_events, "strong_decoder")
    weak = build_decoder_unit(settings, "weak", policy, weak_formation)
    strong = build_decoder_unit(settings, "strong", policy, strong_formation)
    active_tier = policy.primary_tier.value
    active = weak
    if active_tier == "strong":
        active = strong
    _check_the_active_tier_decodes(manager, active, active_tier, plan)
    router, unit_pools = _router_and_pools(
        settings, policy, weak, strong, active
    )
    decoder_memory = _decoder_memory(settings, policy, unit_pools)
    copies_input_by_pool = _copies_input_by_pool(settings, policy, unit_pools)
    blocks_unit_by_pool = _blocks_unit_by_pool(settings, policy, unit_pools)
    formation_by_pool = _formation_by_pool(
        policy, unit_pools, weak_formation, strong_formation
    )
    return DecoderPool(
        router=router,
        active=active,
        unit_pools=dict(unit_pools),
        decoder_memory=decoder_memory,
        scheduler=scheduler,
        copies_input_by_pool=copies_input_by_pool,
        blocks_unit_by_pool=blocks_unit_by_pool,
        formation_by_pool=formation_by_pool,
    )


def build_memory_round_arrivals(parts):
    """The decoders' end of the memory route."""
    return memory_rounds_module.MemoryRoundArrivals(parts.engine)


def _check_the_active_tier_decodes(
    manager: decoder_settings.DecoderManagerSettings,
    active,
    active_tier: str,
    plan: plan_build.Plan,
) -> None:
    """A plan with windows needs a decoder on the tier that decodes them."""
    has_no_decoder = active is None and manager.router is None
    if not has_no_decoder or not plan.planned_operations:
        return
    raise ValueError(
        f"the plan decodes windows on the {active_tier} tier, which "
        "names no decoder: give it a kind or a Python-built decoder"
    )


def _router_and_pools(
    settings: machine_settings.MachineSettings, policy, weak, strong, active
) -> tuple:
    """The router over the tiers and each pool's unit count.

    A router or pools given in Python are used as they are; switching
    routes escalated jobs to a strong pool of their own, and any other
    escalation routes every job to the active tier's one default pool.
    """
    manager = settings.decoder_manager
    router = manager.router
    unit_pools = manager.unit_pools
    if policy.requires_strong_context and router is None:
        router, unit_pools = _switching_pools(settings, weak, strong)
    if router is None:
        router = decoders.CodeRouter(default=active)
    if unit_pools is None:
        active_settings = _active_tier_settings(settings, policy)
        unit_pools = {"default": active_settings.units}
    return router, unit_pools


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


def _formation_by_pool(
    policy,
    unit_pools: dict,
    weak_formation: Optional[detection_events_module.TierFormation],
    strong_formation: Optional[detection_events_module.TierFormation],
) -> dict:
    """Each pool's event-detection logic, from the tier whose units it holds.

    The strong pool is the strong tier's; every other pool decodes the
    plan's windows on the active tier. The map is empty when no tier
    forms anything.
    """
    active_formation = weak_formation
    if policy.primary_tier.value == "strong":
        active_formation = strong_formation
    formation_by_pool = {}
    if active_formation is None:
        return formation_by_pool
    for pool in unit_pools:
        formation_by_pool[pool] = active_formation
    if decode_queue.STRONG_POOL in formation_by_pool:
        formation_by_pool[decode_queue.STRONG_POOL] = strong_formation
    return formation_by_pool


def _blocks_unit_by_pool(
    settings: machine_settings.MachineSettings, policy, unit_pools: dict
) -> dict:
    """Every pool's blocking rule, from the tier that decodes the windows.

    A strong re-decode holds its result in the unit that produced it
    until the window side takes it in any case (decoder_unit.py
    hold_output), so the row is read on the primary tier, and
    <tier>.result_blocks_unit names it there.
    """
    tier = policy.primary_tier.value
    tier_settings = settings.decoder_settings_for(tier)
    blocks_unit = tier_settings.result_blocks_unit
    blocks_by_pool = {}
    for pool in unit_pools:
        blocks_by_pool[pool] = blocks_unit
    return blocks_by_pool


def _copies_input_by_pool(
    settings: machine_settings.MachineSettings, policy, unit_pools: dict
) -> dict:
    """Each pool's input rule, from the tier whose units that pool holds.

    The strong pool is the strong tier's; every other pool decodes the
    plan's windows on the active tier. A value that is not a row of
    DECODER_INPUTS is refused here, where the tier is named.
    """
    active_settings = _active_tier_settings(settings, policy)
    active_copies = _copies_input(active_settings, policy.primary_tier.value)
    strong_copies = _copies_input(settings.strong_decoder, "strong")
    copies_by_pool = {}
    for pool in unit_pools:
        copies_by_pool[pool] = active_copies
    if decode_queue.STRONG_POOL in copies_by_pool:
        copies_by_pool[decode_queue.STRONG_POOL] = strong_copies
    return copies_by_pool


def _copies_input(
    tier_settings: decoder_settings.DecoderSettings, tier: str
) -> bool:
    """Whether this tier's unit is given a copy of the rounds it reads."""
    return tables.row(
        decoder_settings.DECODER_INPUTS,
        f"{tier}_decoder.input",
        tier_settings.input,
    )


def _switching_pools(
    settings: machine_settings.MachineSettings, weak, strong
) -> tuple:
    """The switching router and its pools: default and strong.

    A window's two forced-class solves are two ordinary jobs of the
    default pool, so weak_decoder.units alone decides whether they
    overlap.
    """
    if strong is None:
        raise ValueError(
            "escalation switching escalates to the strong_decoder, which "
            "this configuration does not define"
        )
    unit_pools = {
        "default": settings.weak_decoder.units,
        "strong": settings.strong_decoder.units,
    }
    router = decoders.SwitchingRouter(weak, strong)
    return router, unit_pools


def _active_tier_settings(
    settings: machine_settings.MachineSettings, policy
) -> decoder_settings.DecoderSettings:
    tier = policy.primary_tier.value
    return settings.decoder_settings_for(tier)


def _decoder_memory(
    settings: machine_settings.MachineSettings, policy, unit_pools: dict
) -> Optional[decoder_memory_module.DecoderMemoryConfig]:
    """Each pool's unit memory, from the tier whose units that pool holds.

    The strong pool is the strong tier's; every other pool decodes the
    plan's windows on the active tier. A Python-built memory is used as
    it is. A pool whose tier sets no capacity is left out, which is an
    unbounded memory.
    """
    given = settings.decoder_manager.decoder_memory
    if given is not None:
        return given
    active_settings = _active_tier_settings(settings, policy)
    bits_by_pool = {}
    for pool in unit_pools:
        bits_by_pool[pool] = active_settings.unit_memory.bits
    if decode_queue.STRONG_POOL in bits_by_pool:
        strong_bits = settings.strong_decoder.unit_memory.bits
        bits_by_pool[decode_queue.STRONG_POOL] = strong_bits
    bounded_pools = _without_unset(bits_by_pool)
    if not bounded_pools:
        return None
    return decoder_memory_module.DecoderMemoryConfig(bounded_pools)


def _without_unset(value_by_pool: dict) -> dict:
    """The pools whose value is set; a None value is a key left unset."""
    set_by_pool = {}
    for pool, value in value_by_pool.items():
        if value is None:
            continue
        set_by_pool[pool] = value
    return set_by_pool


def _algorithm(tier_settings: decoder_settings.DecoderSettings, tier: str):
    """A tier's algorithm: a table row, or a fixed latency on MWPM.

    A table row is built from its own settings alone: with keys of its
    own it takes the Settings record the section reader split off the
    tier's keys (decsim/tables.py), and with none it takes nothing, so a
    new row declares no parameter it does not read.
    """
    kind = tier_settings.kind
    if not isinstance(kind, str):
        latency_model = decoders.PresetLatencyDecoder(kind)
        return minimum_weight_perfect_matching.PyMatchingDecoder(latency_model)
    row = tables.row(decoder_settings.DECODERS, f"{tier}_decoder.kind", kind)
    row_settings = tier_settings.row_settings
    if row_settings is None:
        algorithm = row()
    else:
        algorithm = row(settings=row_settings)
    algorithm.compile_key = (row, row_settings)
    return algorithm


def _check_serves_the_confidence(
    algorithm,
    kind,
    tier: str,
    escalation: escalation_settings.EscalationSettings,
) -> None:
    """Refuse a weak tier that cannot serve the run's confidence signal.

    A confidence is the decoder's own, so the signal's evidence
    requirement is held against the row's own declaration. A priced card
    is not refused: it prices one decode of one window, and a forced pair
    is two decodes, so the card is charged once per forced solve.
    """
    signal = escalation_build.confidence_row(escalation)
    required = signal.decoder_evidence_requirement
    missing = required - algorithm.decoder_evidence
    if not missing:
        return
    reason = _missing_evidence_reason(algorithm, missing, signal)
    signal_name = escalation.confidence
    raise ValueError(
        f"{tier}_decoder.kind {kind!r} cannot serve the confidence "
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
    tier_settings: decoder_settings.DecoderSettings,
    algorithm,
    formation: Optional[detection_events_module.TierFormation],
) -> staged_decoder.StagedDecoder:
    """The algorithm between its stages, on the tier's engine clock.

    This tier's event-detection logic first when the rounds reach it raw,
    then the fetch stage, then the algorithm, then the release stage,
    each stage priced once a job and once a round.
    """
    before = []
    formation_stage = _formation_stage(formation)
    if formation_stage is not None:
        before.append(formation_stage)
    fetch = staged_decoder.MemoryFetchStage(
        "fetch",
        cycles_per_job=tier_settings.engine.fetch_cycles_per_job,
        cycles_per_round=tier_settings.engine.fetch_cycles_per_round,
        word_bits=tier_settings.unit_memory.word_bits,
    )
    before.append(fetch)
    release = staged_decoder.DecoderStage(
        "release",
        cycles_per_job=tier_settings.engine.release_cycles_per_job,
        cycles_per_round=tier_settings.engine.release_cycles_per_round,
    )
    timing = staged_decoder.UnitTiming(
        before=tuple(before),
        after=(release,),
        clock=tier_settings.engine.clock,
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
