"""The yaml surface of the decoder card, the sweep and the run folder.

These tests keep configs/reference.yaml and the loader from drifting
apart: every shipped config loads, an unknown key or a stale one is
refused with a sentence, the swept distance reaches the rounds policy,
and a run writes its manifest and its per-shot records.
"""

import dataclasses
import json

import pytest
import yaml

import decsim.build.escalation as escalation_build
import decsim.controller.settings as controller_settings
import decsim.experiments.collect_command as collect_command
import decsim.experiments.experiment as experiment
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import tests.experiments.yaml_configs as yaml_configs


def test_reference_config_defines_both_tiers_and_the_mode_picks_weak():
    reference_path = yaml_configs.CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(reference_path)
    settings = config.settings
    assert settings.weak_decoder.kind == "pymatching"
    assert settings.strong_decoder.kind == "belief_matching"
    assert escalation_build.primary_tier(settings.escalation) == "weak"
    # engine cycles price on a named domain, resolved once like the links
    assert settings.weak_decoder.engine.clock == settings.clocks.clock("fridge")
    assert settings.strong_decoder.engine.clock == settings.clocks.clock("room")


def test_the_reference_controller_charges_the_traced_issue_pipeline():
    """Eight cycles from the decision at the core to the pulse trigger.

    The trace on QubiC's core is in the reference yaml's own comment
    (Fruitwala 2404.15260 Sec. III and IV, gem5's MinorCPU stage delays
    where the paper is silent); a run that keeps the reference card
    charges it on every feedback decision.
    """
    reference_path = yaml_configs.CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(reference_path)
    controller = config.settings.controller
    assert controller.decision_to_pulse_cycles == 8
    assert controller.clock == config.settings.clocks.clock("fridge")


@pytest.mark.parametrize("name", yaml_configs.SHIPPED_CONFIGS)
def test_every_shipped_config_loads(name):
    config_path = yaml_configs.CONFIGS_DIR / name
    config = experiment.load_experiment(config_path)
    assert config.sweep


@pytest.mark.parametrize("name", yaml_configs.SHIPPED_CONFIGS)
def test_every_config_the_tests_name_is_shipped(name):
    config_path = yaml_configs.CONFIGS_DIR / name
    assert config_path.is_file()


def test_controller_cycle_card_reaches_both_runtime_paths(tmp_path):
    import decsim.config as config_module

    config_path = yaml_configs.write_config(
        tmp_path,
        {
            "clocks": {"fridge": 500.0, "room": 250.0},
            "controller": {
                "clock": "fridge",
                "readout_to_bits_cycles": 27,
                "packing_cycles_per_round": 0,
                "decision_to_pulse_cycles": 8,
            },
        },
    )
    config = experiment.load_experiment(config_path)
    controller = config.settings.controller
    assert controller.readout_to_bits_cycles == 27
    assert controller.decision_to_pulse_cycles == 8
    assert controller.clock.period_ticks == 2000

    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    completed = machine_module.Machine.build(settings, 0)
    built = completed.controller.settings
    assert built.clock.edge(27, 0) == config_module.microseconds_to_ticks(0.054)
    output = completed.instruction_output
    assert output.pulse_cycles == 8
    assert output.clock.edge(8, 0) == config_module.microseconds_to_ticks(0.016)


def test_the_packing_overflow_word_reaches_the_controller(tmp_path):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    controller["packing_overflow"] = "drop_round"
    config_path = yaml_configs.write_config(
        tmp_path, {"controller": controller}
    )
    config = experiment.load_experiment(config_path)
    assert config.settings.controller.packing_overflow is (
        controller_settings.PackingOverflowPolicy.DROP_ROUND
    )


def test_the_default_packing_overflow_is_backpressure(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    assert config.settings.controller.packing_overflow is (
        controller_settings.PackingOverflowPolicy.STALL
    )


def test_an_unknown_packing_overflow_word_is_refused(tmp_path):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    controller["packing_overflow"] = "overwrite"
    config_path = yaml_configs.write_config(
        tmp_path, {"controller": controller}
    )
    sentence = (
        "controller.packing_overflow must be one of "
        r"\('stall', 'drop_round'\), got 'overwrite'"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


@pytest.mark.parametrize("bound", [0.5, True, -1, 0, "x"])
def test_a_packing_bound_that_is_not_a_count_of_rounds_is_refused(
    tmp_path, bound
):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    controller["packing_rounds_in_flight"] = bound
    config_path = yaml_configs.write_config(
        tmp_path, {"controller": controller}
    )
    sentence = (
        "controller.packing_rounds_in_flight must be a whole count of "
        "rounds, at least one, or null for no bound"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_an_unknown_controller_key_is_refused(tmp_path):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    controller["packing_cycles"] = 3
    config_path = yaml_configs.write_config(
        tmp_path, {"controller": controller}
    )
    sentence = r"controller does not know \['packing_cycles'\]; its keys are"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_controller_without_its_clock_is_refused_naming_the_key(tmp_path):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    del controller["clock"]
    config_path = yaml_configs.write_config(
        tmp_path, {"controller": controller}
    )
    sentence = r"controller needs the keys \['clock'\]"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


@pytest.mark.parametrize(
    "key",
    [
        "readout_to_bits_cycles",
        "packing_cycles_per_round",
        "decision_to_pulse_cycles",
    ],
)
def test_a_controller_cycle_count_refusal_names_its_yaml_path(tmp_path, key):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    controller[key] = -1
    config_path = yaml_configs.write_config(
        tmp_path, {"controller": controller}
    )
    sentence = f"controller.{key} must not be negative"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


@pytest.mark.parametrize("key", ["latency_cycles", "cycles_per_round"])
def test_a_formation_cycle_count_refusal_names_its_yaml_path(tmp_path, key):
    config_path = yaml_configs.write_config(
        tmp_path, {"detection_events": {key: -1}}
    )
    sentence = f"detection_events.{key} must not be negative"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_formation_seat_off_the_path_is_refused_when_the_yaml_loads(
    tmp_path,
):
    section = {"formed_at": ["nowhere"]}
    config_path = yaml_configs.write_config(
        tmp_path, {"detection_events": section}
    )
    sentence = "detection_events.formed_at names 'nowhere', which is not a seat"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_the_idle_policy_kind_reaches_the_controller(tmp_path):
    config_path = yaml_configs.write_config(
        tmp_path, {"idle_policy": {"kind": "ignore"}}
    )
    config = experiment.load_experiment(config_path)
    assert config.settings.idle_policy.kind == "ignore"


def test_an_idle_policy_written_as_a_bare_word_is_refused_with_its_mapping(
    tmp_path,
):
    config_path = yaml_configs.write_config(tmp_path, {"idle_policy": "ignore"})
    sentence = (
        "the yaml section idle_policy holds 'ignore'; a section is a "
        "mapping of its keys, as in configs/reference.yaml, so write "
        "idle_policy: {kind: ignore}"
    )
    with pytest.raises(ValueError) as refusal:
        experiment.load_experiment(config_path)
    assert sentence in str(refusal.value)


def test_a_key_no_idle_policy_row_declares_is_refused(tmp_path):
    idle_policy = {"kind": "ignore", "every_nth_round": 2}
    config_path = yaml_configs.write_config(
        tmp_path, {"idle_policy": idle_policy}
    )
    sentence = (
        r"idle_policy does not know \['every_nth_round'\]; its keys are "
        r"\['kind'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_an_idle_policy_rows_own_key_reaches_its_settings(
    monkeypatch, tmp_path
):
    """gem5's shape: the row declares its key, the section hands it over."""
    monkeypatch.setitem(
        controller_settings.IDLE_POLICIES, "every_nth", _EveryNthIdlePolicy
    )
    idle_policy = {"kind": "every_nth", "every_nth_round": 2}
    config_path = yaml_configs.write_config(
        tmp_path, {"idle_policy": idle_policy}
    )

    config = experiment.load_experiment(config_path)

    row_settings = config.settings.idle_policy.row_settings
    assert row_settings == _EveryNthIdlePolicy.Settings(every_nth_round=2)


class _EveryNthIdlePolicy:
    """An idle row with one key of its own, for the section's split."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        every_nth_round: int = 1

        @classmethod
        def from_yaml(cls, section):
            return cls(**section)


def test_a_factory_row_named_in_the_yaml_is_built_with_its_own_keys(
    tmp_path,
):
    """The magic_state_factory section reads its row's keys like any other."""
    factory = {
        "kind": "multi_level",
        "levels": [{"unit_count": 2, "distance": 5}],
        "preparation_unit_count": 4,
    }
    config_path = yaml_configs.write_config(
        tmp_path, {"magic_state_factory": factory}
    )
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )

    machine = machine_module.Machine.build(settings, 0)

    built = machine.factory
    level = built.levels[0]
    assert built.card.preparation_unit_count == 4
    assert (level.unit_count, level.distance) == (2, 5)
    assert level.logical_cycles_per_round == 13


def test_a_key_no_factory_row_declares_is_refused(tmp_path):
    factory = {"kind": "infinite", "unit_count": 2}
    config_path = yaml_configs.write_config(
        tmp_path, {"magic_state_factory": factory}
    )
    sentence = (
        r"magic_state_factory does not know \['unit_count'\]; its keys are "
        r"\['kind'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_distillation_row_without_its_three_counts_is_refused(tmp_path):
    factory = {"kind": "distillation", "unit_count": 2}
    config_path = yaml_configs.write_config(
        tmp_path, {"magic_state_factory": factory}
    )
    sentence = (
        r"magic_state_factory kind distillation needs "
        r"\['attempt_ticks', 'correction_round_count'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_yaml_without_a_factory_section_runs_the_infinite_row(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    assert config.settings.magic_state_factory.kind == "infinite"


def test_the_bulk_strong_key_reaches_the_decoder_manager(tmp_path):
    config_path = yaml_configs.write_config(
        tmp_path, {"decoder_manager": {"bulk_strong": True}}
    )
    config = experiment.load_experiment(config_path)
    assert config.settings.decoder_manager.bulk_strong is True


def test_the_default_decoder_manager_does_not_batch(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    assert config.settings.decoder_manager.bulk_strong is False


@pytest.mark.parametrize("value", ["batch", 1, 0])
def test_a_bulk_strong_that_is_not_a_flag_is_refused(tmp_path, value):
    """A one or a zero equals a flag in Python and is still no flag."""
    config_path = yaml_configs.write_config(
        tmp_path, {"decoder_manager": {"bulk_strong": value}}
    )
    sentence = f"decoder_manager.bulk_strong {value!r} is not a boolean"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_an_unknown_decoder_manager_key_is_refused(tmp_path):
    config_path = yaml_configs.write_config(
        tmp_path, {"decoder_manager": {"units": 2}}
    )
    sentence = r"decoder_manager does not know \['units'\]"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_yaml_without_a_sweep_is_refused(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    text = config_path.read_text()
    sections = yaml.safe_load(text)
    del sections["sweep"]
    config_text = yaml.safe_dump(sections)
    config_path.write_text(config_text)
    with pytest.raises(ValueError, match="has no sweep"):
        experiment.load_experiment(config_path)


def test_a_sweep_axis_given_as_one_value_is_refused(tmp_path):
    block = {
        "physical_error_probability": [0.001],
        "distance": 3,
        "round_period_microseconds": [1.0],
        "shots": 1,
    }
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block]})
    with pytest.raises(ValueError, match="distance must be a list"):
        experiment.load_experiment(config_path)


def test_a_qpu_key_other_than_kind_is_refused(tmp_path):
    """The sweep sets the distance, so the section refuses one written here."""
    qpu = {"kind": "stim_device", "distance": 5}
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    sentence = (
        r"qpu does not know \['distance'\]; its keys are "
        r"\['kind', 'code_card'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


class _SplitBits(syndrome_devices.SyndromeBitDevice):
    """A source with a key of its own, for the section's split."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        one_payload_per_patch: bool = False

        @classmethod
        def from_yaml(cls, section):
            return cls(**section)

    def __init__(self, code, settings) -> None:
        syndrome_devices.SyndromeBitDevice.__init__(
            self, code, one_payload_per_patch=settings.one_payload_per_patch
        )
        self.settings = settings


def test_the_code_card_row_named_in_the_yaml_is_built_with_its_own_keys(
    tmp_path,
):
    """CUDA-Q QEC's get_code(name, options): a card by name, its own keys."""
    qpu = {
        "kind": "timing_only",
        "code_card": "bivariate_bicycle",
        "qubit_count": 30,
        "logical_qubit_count": 8,
    }
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=2,
        round_period_microseconds=1.0,
    )

    machine = machine_module.Machine.build(settings, 0)

    code = machine.syndrome_source.code
    assert code.name == "bivariate-bicycle code [[30,8,2]]"
    assert code.syndrome_bits_per_round(1) == 30


def test_a_card_key_on_a_card_that_declares_none_is_refused(tmp_path):
    qpu = {"kind": "timing_only", "qubit_count": 30}
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    sentence = (
        r"qpu does not know \['qubit_count'\]; its keys are "
        r"\['kind', 'code_card'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_code_card_off_its_table_is_refused_when_the_yaml_loads(tmp_path):
    qpu = {"kind": "timing_only", "code_card": "color"}
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    sentence = "qpu.code_card 'color' is not a row of its table"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_a_source_rows_own_key_reaches_the_built_source(monkeypatch, tmp_path):
    monkeypatch.setitem(qpu_settings.SYNDROME_SOURCES, "split", _SplitBits)
    qpu = {"kind": "split", "one_payload_per_patch": True}
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )

    machine = machine_module.Machine.build(settings, 0)

    own_settings = _SplitBits.Settings(one_payload_per_patch=True)
    assert machine.syndrome_source.settings == own_settings
    assert machine.syndrome_source.one_payload_per_patch is True


def test_the_burst_rows_keys_reach_the_source_that_runs(tmp_path):
    qpu = {
        "kind": "burst_stim",
        "burst_onset_round": 5,
        "burst_decay_rounds": 4.0,
        "burst_radius": 2.0,
        "burst_center": [3.0, 3.0],
        "burst_error_probability": 0.1,
        "burst_channels": ["idle", "measurement"],
    }
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )

    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()

    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=5,
        burst_decay_rounds=4.0,
        burst_radius=2.0,
        burst_center=(3.0, 3.0),
        burst_error_probability=0.1,
        burst_channels=("idle", "measurement"),
    )
    assert machine.syndrome_source.burst == burst
    assert len(result.operation_results) == 1


def test_a_mode_without_its_tier_is_refused(tmp_path):

    config_path = yaml_configs.write_config(
        tmp_path, {"escalation": {"kind": "strong_only"}}
    )  # only weak_decoder is defined
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    with pytest.raises(ValueError, match="strong tier, which names no decoder"):
        machine_module.Machine.build(settings)


def test_unknown_algorithms_and_stale_keys_fail_loudly(tmp_path):
    unknown_algorithm = yaml_configs.write_config(
        tmp_path,
        {
            "weak_decoder": {
                **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
                "kind": "lookup_table",
            }
        },
    )
    with pytest.raises(
        ValueError, match="weak_decoder.kind 'lookup_table' is not a row"
    ):
        experiment.load_experiment(unknown_algorithm)

    old_flat_decoder = yaml_configs.write_config(
        tmp_path,
        {
            "decoder": {
                "units": 1,
                "unit_memory": {"bits": None},
                "engine": {
                    "clock": "fridge",
                    "fetch_cycles_per_round": 1,
                    "fetch_cycles_per_job": 0,
                    "release_cycles_per_job": 1,
                    "release_cycles_per_round": 0,
                },
            }
        },
    )
    with pytest.raises(ValueError, match=r"no section \['decoder'\]"):
        experiment.load_experiment(old_flat_decoder)

    old_sweep_axis = yaml_configs.write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3],
                    "round_period_microseconds": [1.0],
                    "algorithm_latency_us": [0.028],
                    "shots": 1,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="decoder card"):
        experiment.load_experiment(old_sweep_axis)

    fixed_distance_key = yaml_configs.write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "round_period_microseconds": [1.0],
                    "shots": 1,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match=r"sweep block 1 lacks \['distance'\]"):
        experiment.load_experiment(fixed_distance_key)


def test_a_qpu_kind_off_its_table_is_refused_when_the_yaml_loads(tmp_path):
    misspelt_source = yaml_configs.write_config(
        tmp_path, {"qpu": {"kind": "stim_devic"}}
    )
    with pytest.raises(ValueError, match="qpu.kind 'stim_devic' is not a row"):
        experiment.load_experiment(misspelt_source)


def test_a_cycle_count_on_a_kind_that_is_not_union_find_is_refused(tmp_path):
    """cycle_count is the union_find row's own key: pymatching declares none.

    The refusal names the keys the pymatching tier does read, so the
    yaml is refused when it loads, before a machine is built.
    """
    counted_matching = yaml_configs.write_config(
        tmp_path,
        {
            "weak_decoder": {
                **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
                "kind": "pymatching",
                "cycle_count": {"clock": "fridge", "setup_cycles": 11},
            }
        },
    )
    sentence = (
        r"weak_decoder does not know \['cycle_count'\]; its keys are "
        r"\['kind', 'units', 'input', 'boundary_fold', 'result_blocks_unit', "
        r"'unit_memory', 'engine'\]"
    )
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(counted_matching)


def test_engine_clock_must_name_a_clock_domain(tmp_path):
    config_path = yaml_configs.write_config(
        tmp_path,
        {
            "weak_decoder": {
                **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
                "engine": {
                    "clock": "sfq",
                    "fetch_cycles_per_round": 1,
                    "fetch_cycles_per_job": 0,
                    "release_cycles_per_job": 1,
                    "release_cycles_per_round": 0,
                },
            }
        },
    )
    with pytest.raises(ValueError, match="clock 'sfq' is not a clocks entry"):
        experiment.load_experiment(config_path)


def test_report_rows_carry_the_algorithm_column(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    measurements = [
        yaml_configs.measure_point_shot(
            config,
            physical_error_probability=0.001,
            distance=3,
            round_period_microseconds=1.0,
            seed=seed,
        )
        for seed in range(2)
    ]
    record = report.record_of(measurements)
    rows = report.summarize(record.shots, record.window_samples)
    assert len(rows) == 1
    assert rows[0]["algorithm"] == 0.028
    per_link = report.link_rows(record.shot_links)
    assert per_link and all(row["algorithm"] == 0.028 for row in per_link)


def test_rounds_per_shot_scales_with_the_swept_distance(tmp_path):
    # "10d" is Toshio 2510.25222's memory-experiment convention: the shot
    # length follows the swept code distance.
    workload = yaml_configs.memory_workload("10d")
    config_path = yaml_configs.write_config(
        tmp_path,
        {
            "workload": workload,
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3, 5],
                    "round_period_microseconds": [1.0],
                    "shots": 1,
                }
            ],
        },
    )
    config = experiment.load_experiment(config_path)
    at_three = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
    at_five = config.point_settings(
        physical_error_probability=0.001,
        distance=5,
        round_period_microseconds=1.0,
    )
    assert at_three.workload.rounds_policy.rounds_by_operation[1] == 30
    assert at_five.workload.rounds_policy.rounds_by_operation[1] == 50
    measurement = yaml_configs.measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=5,
        round_period_microseconds=1.0,
        seed=0,
    )
    metadata = json.loads(measurement.metadata)
    assert metadata["distance"] == 5
    assert measurement.windows > 5


def test_a_run_writes_its_manifest_and_per_shot_records(tmp_path, monkeypatch):
    # One tiny run end to end: a timestamped run dir with manifest.json,
    # the config copy, shots.csv, sweep.csv and links.csv.
    import csv

    config_path = yaml_configs.write_config(tmp_path, {})
    monkeypatch.chdir(tmp_path)
    run_dir, rows = collect_command.run_experiment(config_path)

    manifest_path = run_dir / "manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    assert manifest["versions"]["packages"]["stim"]
    assert (
        manifest["resolved_config"]["settings"]["escalation"]["kind"]
        == "weak_baseline"
    )
    assert manifest["started_utc"] and manifest["finished_utc"]

    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path) as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)
    assert len(shots) == 1 and shots[0]["seed"] == "0"

    assert run_dir.name.endswith("-unit_test_config")
    assert (run_dir / "config" / "unit_test_config.yaml").exists()
    assert (run_dir / "sweep.csv").exists() and (run_dir / "links.csv").exists()


def test_a_run_with_an_online_threshold_records_its_trajectory(
    tmp_path, monkeypatch
):
    """The point's threshold is built per point, so every file is written."""
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    monkeypatch.chdir(tmp_path)
    run_dir, rows = collect_command.run_experiment(config_path)

    records = run_dir.glob("online_threshold_*.csv")
    assert len(rows) == 1 and len(list(records)) == 1
    assert (run_dir / "finished").exists()


def test_two_config_files_of_one_name_are_both_copied(tmp_path):
    """The chain's files keep their places, so a base of one name stays."""
    base_path = yaml_configs.write_config(tmp_path, {})
    child_folder = tmp_path / "child"
    child_folder.mkdir()
    child_path = child_folder / base_path.name
    child_path.write_text(f"extends: ../{base_path.name}\n")
    config = experiment.load_experiment(child_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_folder.snapshot_code_state(config, run_dir)

    child_copy = run_dir / "config" / "child" / base_path.name
    base_copy = run_dir / "config" / base_path.name
    assert child_copy.read_text() == child_path.read_text()
    assert base_copy.read_text() == base_path.read_text()
