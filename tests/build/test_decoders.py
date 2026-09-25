"""Building the two tiers' units and the pool the manager schedules.

A tier's kind names a row of decoders/settings.py's DECODERS and the
unit is that algorithm between its fetch and release stages. Which pools
the manager gets is the escalation policy's declared fact, not its name:
a policy that may escalate puts two units behind one router with a
strong pool, and every other policy routes every job to the tier that
decodes the plan's windows.
"""

import dataclasses

import pytest

import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.config as config
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.decoder as union_find
import decsim.detector_error_model.detection_event_formation as event_formation
import decsim.detector_error_model.settings as event_settings
import decsim.escalation.settings as escalation_settings
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings
import tests.declared_run as declared_run


def _settings(*, escalation=None, weak=None, strong=None):
    if escalation is None:
        escalation = escalation_settings.EscalationSettings()
    if weak is None:
        weak = decoder_settings.DecoderSettings()
    if strong is None:
        strong = decoder_settings.DecoderSettings()
    workload = declared_run.declared_workload(None, 6)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        links=links,
        controller=controller,
        pauli_frame=frame,
        escalation=escalation,
        weak_decoder=weak,
        strong_decoder=strong,
    )


def _pool(settings, detection_events=None):
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    plan = plan_build.build_plan(settings, policy)
    if detection_events is None:
        at_the_controller = event_settings.DetectionEventSettings()
        detection_events = event_formation.SeatedFormation(
            None, at_the_controller
        )
    return decoder_build.build_decoder_pool(
        settings, plan, policy, detection_events
    )


def _preset(microseconds: float):
    preset = decoders.PresetLatencyDecoder(microseconds)
    return decoder_settings.DecoderSettings(decoder=preset)


def test_the_units_two_stages_carry_all_four_of_the_engines_cycle_keys():
    """Each stage is priced once a job and once a round, on its clock.

    A swap or a drop in the wiring moves a job of several rounds: the
    fetch and the release read different keys and the per-job and the
    per-round cycles are different numbers.
    """
    period_ticks = config.microseconds_to_ticks(0.01)
    clock = config.Clock(period_ticks)
    weak = decoder_settings.DecoderSettings(
        kind="pymatching",
        fetch_cycles_per_job=2,
        fetch_cycles_per_round=3,
        release_cycles_per_job=5,
        release_cycles_per_round=7,
        engine_clock=clock,
    )
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    unit = decoder_build.build_decoder_unit(settings, "weak", policy)
    job = decoding_records.DecodeJob(operation_id=1, window_id=0, round_count=4)

    ticks = unit.timing.stage_ticks(job)

    fetch_cycles = 2 + 3 * 4
    release_cycles = 5 + 7 * 4
    assert ticks["fetch"] == fetch_cycles * period_ticks
    assert ticks["release"] == release_cycles * period_ticks


def test_the_union_find_row_is_built_with_the_tiers_weight_step():
    """The growth resolution the yaml names reaches the row that grows."""
    period_ticks = config.microseconds_to_ticks(0.01)
    clock = config.Clock(period_ticks)
    union_find_settings = union_find.UnionFindDecoder.Settings(weight_step=0.25)
    weak = decoder_settings.DecoderSettings(
        kind="union_find",
        engine_clock=clock,
        row_settings=union_find_settings,
    )
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )

    unit = decoder_build.build_decoder_unit(settings, "weak", policy)

    assert unit.decoder.weight_step == 0.25


def test_a_python_built_decoder_is_returned_as_it_is():
    built = decoders.PresetLatencyDecoder(10.0)
    weak = decoder_settings.DecoderSettings(decoder=built)
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )

    unit = decoder_build.build_decoder_unit(settings, "weak", policy)

    assert unit is built


def test_a_tier_that_names_no_decoder_builds_none():
    settings = _settings()
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )

    unit = decoder_build.build_decoder_unit(settings, "strong", policy)

    assert unit is None


def test_a_weak_only_run_routes_every_job_to_the_one_tier():
    weak = _preset(10.0)
    settings = _settings(weak=weak)

    pool = _pool(settings)

    assert isinstance(pool.router, decoders.CodeRouter)
    assert sorted(pool.unit_pools) == ["default"]


def test_a_run_that_may_escalate_gets_a_strong_pool_behind_one_router():
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
    )
    weak = _preset(10.0)
    strong = _preset(30.0)
    settings = _settings(escalation=escalation, weak=weak, strong=strong)

    pool = _pool(settings)

    assert isinstance(pool.router, decoders.SwitchingRouter)
    assert decode_queue.STRONG_POOL in pool.unit_pools


def test_each_tiers_unit_memory_reaches_the_pool_of_its_own_units():
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
    )
    weak_memory = decoder_settings.UnitMemorySettings(bits=12)
    strong_memory = decoder_settings.UnitMemorySettings(bits=30)
    weak_preset = _preset(10.0)
    strong_preset = _preset(30.0)
    weak = dataclasses.replace(weak_preset, unit_memory=weak_memory)
    strong = dataclasses.replace(strong_preset, unit_memory=strong_memory)
    settings = _settings(escalation=escalation, weak=weak, strong=strong)

    pool = _pool(settings)

    memory = pool.decoder_memory
    assert memory.capacity_for("default") == 12
    assert memory.capacity_for(decode_queue.STRONG_POOL) == 30


def test_a_plan_whose_active_tier_names_no_decoder_is_refused():
    settings = _settings()

    with pytest.raises(ValueError) as refusal:
        _pool(settings)

    sentence = str(refusal.value)
    assert "names no decoder" in sentence
    assert "weak tier" in sentence


def test_every_pool_declares_whether_its_unit_takes_a_copy():
    """The copy-or-reference key of I5, answered per pool at build."""
    weak = _preset(10.0)
    settings = _settings(weak=weak)

    pool = _pool(settings)

    assert sorted(pool.copies_input_by_pool) == sorted(pool.unit_pools)
    assert sorted(pool.blocks_unit_by_pool) == sorted(pool.unit_pools)


def test_a_decoder_seat_gives_each_pool_its_stage_whatever_the_source():
    """A source with no recipes has nothing to convert and the stage to pay."""
    weak = _preset(10.0)
    settings = _settings(weak=weak)
    no_recipes = None
    both_tiers = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder", "strong_decoder")
    )
    at_the_decoder = event_formation.SeatedFormation(no_recipes, both_tiers)
    controller = event_settings.DetectionEventSettings()
    at_the_controller = event_formation.SeatedFormation(no_recipes, controller)

    formed_at_the_decoder = _pool(settings, at_the_decoder)
    formed_at_the_controller = _pool(settings, at_the_controller)

    pools_with_a_stage = set(formed_at_the_decoder.formation_by_pool)
    assert pools_with_a_stage == set(formed_at_the_decoder.unit_pools)
    assert formed_at_the_controller.formation_by_pool == {}


def test_a_decoder_kind_that_names_no_row_is_refused():
    weak = decoder_settings.DecoderSettings(kind="oracle")
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )

    with pytest.raises(ValueError) as refusal:
        decoder_build.build_decoder_unit(settings, "weak", policy)

    assert "oracle" in str(refusal.value)
