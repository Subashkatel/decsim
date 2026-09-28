"""Build the decoder units of both tiers and the pool the manager schedules.

A tier's kind names a row of decoders/settings.py's DECODERS, and the
unit is that algorithm between its fetch and release stages. The two
tiers' units and the managers' pools are one record, gem5's
CacheConfig.config_cache shape (configs/common/CacheConfig.py).
"""

import dataclasses
from typing import Optional

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.decoders as decoders
import decsim.decoders.detection_events as detection_events_module
import decsim.decoders.memory_rounds as memory_rounds_module
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
    """The tiers' units and the managers' pool knobs.

    active is the unit of the tier that decodes the plan's windows, None
    when that tier names no decoder; strong is the strong tier's unit on
    a run that may escalate, else None. chip is the chip's manager's
    pool, of the tier that decodes the plan's windows; host is the host's
    manager's, of the strong tier, on a switching run only.
    """

    active: Optional[ports.Decoder]
    strong: Optional[ports.Decoder]
    chip: decoder_pool_module.PoolSettings
    host: Optional[decoder_pool_module.PoolSettings]


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
    """The tiers' units and the managers' pools.

    Switching gives the strong tier a pool of its own, which serves
    escalated jobs. Any other escalation has one pool, of the tier that
    decodes the plan's windows. Each tier whose decoder is a seat of the
    run's detection event placement gets its own event-detection logic,
    so a round both tiers read is formed and charged by each.
    """
    weak_formation = _tier_formation(detection_events, "weak_decoder")
    strong_formation = _tier_formation(detection_events, "strong_decoder")
    weak = build_decoder_unit(settings, "weak", policy, weak_formation)
    strong = build_decoder_unit(settings, "strong", policy, strong_formation)
    active_tier = policy.primary_tier.value
    active = weak
    active_formation = weak_formation
    if active_tier == "strong":
        active = strong
        active_formation = strong_formation
    _check_the_active_tier_decodes(active, active_tier, plan)
    blocks_unit = _blocks_unit(settings, active_tier)
    chip = _pool(
        settings,
        decode_queue.DEFAULT_POOL,
        active_tier,
        active_formation,
        blocks_unit,
    )
    if not policy.requires_strong_context:
        return DecoderPool(active=active, strong=None, chip=chip, host=None)
    _check_switching_has_a_strong(strong)
    host = _pool(
        settings, decode_queue.STRONG_POOL, "strong", strong_formation, False
    )
    return DecoderPool(active=active, strong=strong, chip=chip, host=host)


def build_memory_round_arrivals(parts):
    """The decoders' end of the memory route."""
    return memory_rounds_module.MemoryRoundArrivals(parts.engine)


def _check_the_active_tier_decodes(
    active, active_tier: str, plan: plan_build.Plan
) -> None:
    """A plan with windows needs a decoder on the tier that decodes them."""
    if active is not None or not plan.planned_operations:
        return
    raise ValueError(
        f"the plan decodes windows on the {active_tier} tier, which "
        "names no decoder: give it a kind or a Python-built decoder"
    )


def _pool(
    settings: machine_settings.MachineSettings,
    name: str,
    tier: str,
    formation: Optional[detection_events_module.TierFormation],
    blocks_unit: bool,
) -> decoder_pool_module.PoolSettings:
    """One manager's pool, from the settings of the tier whose units it holds.

    A value of <tier>.input that is not a row of DECODER_INPUTS is
    refused here, where the tier is named.
    """
    tier_settings = settings.decoder_settings_for(tier)
    copies_input = tables.row(
        decoder_settings.DECODER_INPUTS,
        f"{tier}_decoder.input",
        tier_settings.input,
    )
    return decoder_pool_module.PoolSettings(
        name=name,
        unit_count=tier_settings.units,
        capacity_bits=tier_settings.unit_memory.bits,
        copies_input=copies_input,
        blocks_unit=blocks_unit,
        formation=formation,
    )


def _blocks_unit(
    settings: machine_settings.MachineSettings, active_tier: str
) -> bool:
    """The blocking rule of the pool that decodes the windows.

    The strong pool of a switching run never blocks: a strong re-decode
    frees its unit's compute at its end and its result waits in the
    unit's output slot until the window side takes it (decoder_unit.py
    hold_output), so <tier>.result_blocks_unit is read on the primary
    tier alone.
    """
    tier_settings = settings.decoder_settings_for(active_tier)
    return tier_settings.result_blocks_unit


def _check_switching_has_a_strong(strong) -> None:
    """Switching escalates to the strong tier, so the run must name one."""
    if strong is not None:
        return
    raise ValueError(
        "escalation switching escalates to the strong_decoder, which "
        "this configuration does not define"
    )


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
