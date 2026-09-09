"""Building the two tiers' units and the pool the manager schedules.

A tier's kind names a row of decoders/settings.py's DECODERS and the
unit is that algorithm between its fetch and release stages. Which pools
the manager gets is the escalation policy's declared fact, not its name:
a policy that may escalate puts two units behind one router with a
strong pool, and every other policy routes every job to the tier that
decodes the plan's windows.
"""

import pytest

import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.detection_event_formation as event_formation
import decsim.escalation.settings as escalation_settings
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


def _pool(settings):
    policy = escalation_build.build_escalation_policy(settings.escalation)
    plan = plan_build.build_plan(settings, policy)
    formed_at_the_controller = event_formation.ControllerSideFormation(None, 0)
    return decoder_build.build_decoder_pool(
        settings, plan, policy, formed_at_the_controller
    )


def _preset(microseconds: float):
    preset = decoders.PresetLatencyDecoder(microseconds)
    return decoder_settings.DecoderSettings(decoder=preset)


def test_a_python_built_decoder_is_returned_as_it_is():
    built = decoders.PresetLatencyDecoder(10.0)
    weak = decoder_settings.DecoderSettings(decoder=built)
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(settings.escalation)

    unit = decoder_build.build_decoder_unit(settings, "weak", policy)

    assert unit is built


def test_a_tier_that_names_no_decoder_builds_none():
    settings = _settings()
    policy = escalation_build.build_escalation_policy(settings.escalation)

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


def test_a_decoder_kind_that_names_no_row_is_refused():
    weak = decoder_settings.DecoderSettings(kind="oracle")
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(settings.escalation)

    with pytest.raises(ValueError) as refusal:
        decoder_build.build_decoder_unit(settings, "weak", policy)

    assert "oracle" in str(refusal.value)
