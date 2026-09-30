"""Building the two tiers' units and the pool the manager schedules.

A tier's kind names a row of decoders/settings.py's DECODERS and the
unit is that algorithm between its fetch and release stages. Which pools
the managers get is the escalation policy's declared fact, not its name:
a policy that may escalate gives the strong tier a unit and the host's
manager a pool of its own, and the chip's manager always has one pool,
of the tier that decodes the plan's windows. Each manager has its own
queue, scheduler and staging, and the two share one ledger of strong
requests.
"""

import dataclasses

import pytest

import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.config as config
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.decoder as union_find
import decsim.detector_error_model.detection_event_formation as event_formation
import decsim.detector_error_model.settings as event_settings
import decsim.escalation.settings as escalation_settings
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records
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
    engine = decoder_settings.EngineSettings(
        clock=clock,
        fetch_cycles_per_job=2,
        fetch_cycles_per_round=3,
        release_cycles_per_job=5,
        release_cycles_per_round=7,
    )
    weak = decoder_settings.DecoderSettings(kind="pymatching", engine=engine)
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
    engine = decoder_settings.EngineSettings(clock=clock)
    union_find_settings = union_find.UnionFindDecoder.Settings(weight_step=0.25)
    weak = decoder_settings.DecoderSettings(
        kind="union_find",
        engine=engine,
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


def test_a_weak_only_run_has_one_pool_and_no_strong_unit():
    weak = _preset(10.0)
    settings = _settings(weak=weak)

    pool = _pool(settings)

    assert pool.strong is None
    assert pool.chip.name == "default"
    assert pool.host is None


def test_a_run_that_may_escalate_gets_a_strong_unit_and_its_own_pool():
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

    assert pool.strong is not None
    assert pool.strong is not pool.active
    assert pool.host.name == decode_queue.STRONG_POOL


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

    assert pool.chip.capacity_bits == 12
    assert pool.host.capacity_bits == 30


def test_only_the_pool_that_decodes_the_windows_blocks_on_its_result():
    """A strong decode frees its unit at its end and waits in its output."""
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        threshold_source="fixed",
        gap_threshold_nats=2.0,
        confidence="complementary_gap",
    )
    weak_preset = _preset(10.0)
    weak = dataclasses.replace(weak_preset, result_blocks_unit=True)
    strong = _preset(30.0)
    settings = _settings(escalation=escalation, weak=weak, strong=strong)

    pool = _pool(settings)

    assert pool.chip.blocks_unit
    assert not pool.host.blocks_unit


def test_a_plan_whose_active_tier_names_no_decoder_is_refused():
    settings = _settings()

    with pytest.raises(ValueError) as refusal:
        _pool(settings)

    sentence = str(refusal.value)
    assert "names no decoder" in sentence
    assert "weak tier" in sentence


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

    assert formed_at_the_decoder.chip.formation is not None
    assert formed_at_the_controller.chip.formation is None


def test_a_decoder_kind_that_names_no_row_is_refused():
    weak = decoder_settings.DecoderSettings(kind="oracle")
    settings = _settings(weak=weak)
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )

    with pytest.raises(ValueError) as refusal:
        decoder_build.build_decoder_unit(settings, "weak", policy)

    assert "oracle" in str(refusal.value)


def test_a_weak_only_run_has_no_host_manager():
    machine = declared_run.weak_only_run()

    assert machine.decoders.strong_decoder_manager is None


def test_each_side_has_its_own_manager_pool_and_one_ledger():
    """The chip side opens a strong request and the host side serves it."""
    machine = declared_run.switching_run(escalates=True)
    decode_side = machine.decoders
    chip = decode_side.decoder_manager
    host = decode_side.strong_decoder_manager
    strong_decoder = decode_side.strong_decoder

    assert chip.pool.name == decode_queue.DEFAULT_POOL
    assert host.pool.name == decode_queue.STRONG_POOL
    assert chip.strong_requests is decode_side.strong_requests
    assert host.strong_requests is decode_side.strong_requests
    assert host.escalation_policy is machine.windows.escalation_policy
    assert chip.decoder is decode_side.primary_decoder
    assert host.decoder is strong_decoder
    assert machine.windows.models.strong_decoder is strong_decoder
    assert chip.queue is not host.queue
    assert chip.queue.scheduler is not host.queue.scheduler
    assert chip.service.staging is not host.service.staging


class _SeedRecordingScheduler(schedulers.FifoScheduler):
    """A FIFO that keeps the seed the run hands it."""

    def __init__(self):
        self.reserved_seeds = []

    def reserve_run_seed(self, seed):
        self.reserved_seeds.append(seed)
        return seed_records.RunSeedReservation("derived", seed, None)

    def commit_run_seed(self, reservation):
        del reservation

    def cancel_run_seed(self, reservation):
        del reservation


def test_each_managers_scheduler_is_seeded_on_its_own_path():
    weak, strong = declared_run.switching_decoders(False)
    weak_decoder = decoder_settings.DecoderSettings(decoder=weak)
    strong_decoder = decoder_settings.DecoderSettings(decoder=strong)
    manager_settings = decoder_settings.DecoderManagerSettings(
        scheduler=_SeedRecordingScheduler
    )
    switching = declared_run.switching_run(escalates=True)
    settings = dataclasses.replace(
        switching.settings,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        decoder_manager=manager_settings,
    )

    machine = machine_module.Machine.build(settings, 7)

    chip_scheduler = machine.decoders.decoder_manager.queue.scheduler
    host_scheduler = machine.decoders.strong_decoder_manager.queue.scheduler
    assert len(chip_scheduler.reserved_seeds) == 1
    assert len(host_scheduler.reserved_seeds) == 1
    assert chip_scheduler.reserved_seeds != host_scheduler.reserved_seeds
