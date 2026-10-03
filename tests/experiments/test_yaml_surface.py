"""The yaml surface of the decoder card, the sweep and the run folder.

These tests keep configs/reference.yaml and the loader from drifting
apart: every shipped config loads, an unknown key or a stale one is
refused with a sentence, the swept distance reaches the rounds policy,
and a run writes its run.json and its per-shot records.
"""

import csv
import dataclasses
import json
import pathlib
import re
import shutil

import pytest
import yaml

import decsim.collect as collect
import decsim.config as config_module
import decsim.controller.policies as idle_policies
import decsim.controller.settings as controller_settings
import decsim.experiments.collect_command as collect_command
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.settings as machine_settings
import tests.experiments.yaml_configs as yaml_configs

CONFIGS = yaml_configs.CONFIGS_DIR
REFERENCE_YAML = CONFIGS / "reference.yaml"
# the data-movement grid's fourth block, the one with the strong tier
DATA_MOVEMENT_SWITCHING_BLOCK = 3


def test_reference_config_defines_both_tiers_and_the_mode_picks_weak():
    """weak_baseline keeps the weak tier; the strong section is read, unused."""
    reference_path = yaml_configs.CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(reference_path)
    point = config.first_point_task()
    settings = point.settings
    assert settings.weak_decoder.algorithm.name == "pymatching"
    assert settings.strong_decoder is None
    assert settings.switching is None
    assert settings.window_tier.value == "weak"
    # engine cycles price on a named domain, resolved once like the links
    fridge_megahertz = config.sections["clocks"]["fridge"]
    fridge = config_module.Clock.from_megahertz(fridge_megahertz)
    assert settings.weak_decoder.engine.clock == fridge


def test_the_reference_controller_charges_the_traced_issue_pipeline():
    """Eight cycles from the decision at the core to the pulse trigger.

    The trace on QubiC's core is in the reference yaml's own comment
    (Fruitwala 2404.15260 Sec. III and IV, gem5's MinorCPU stage delays
    where the paper is silent); a run that keeps the reference card
    charges it on every feedback decision.
    """
    reference_path = yaml_configs.CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(reference_path)
    point = config.first_point_task()
    controller = point.settings.controller
    assert controller.decision_to_pulse_cycles == 8
    fridge_megahertz = config.sections["clocks"]["fridge"]
    fridge = config_module.Clock.from_megahertz(fridge_megahertz)
    assert point.settings.clock == fridge


def test_the_yaml_states_the_machine_clock_once():
    """The controller's clock is the machine's, so the record holds it once.

    A part on the machine's clock names none, as a gem5 ClockedObject
    takes its parent's domain (src/sim/ClockedObject.py:50).
    """
    reference_path = yaml_configs.CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(reference_path)

    task = config.first_point_task()

    settings = task.settings
    assert settings.clock is not None
    assert settings.controller.clock is None


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
    first_point = config.first_point_task()
    controller = first_point.settings.controller
    assert controller.readout_to_bits_cycles == 27
    assert controller.decision_to_pulse_cycles == 8
    assert first_point.settings.clock.period_ticks == 2000

    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    completed = machine_module.Machine.build(settings, 0)
    built = completed.readout.controller.settings
    assert built.clock.edge(27, 0) == config_module.microseconds_to_ticks(0.054)
    output = completed.control.instruction_output
    assert output.pulse_cycles == 8
    assert output.clock.edge(8, 0) == config_module.microseconds_to_ticks(0.016)


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
    first_point = config.first_point_task()
    assert first_point.settings.idle_policy == idle_policies.IgnoreSettings()


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


def test_an_idle_policy_records_own_key_reaches_it(monkeypatch, tmp_path):
    """gem5's shape: the record declares its key, the section hands it over."""
    monkeypatch.setitem(
        controller_settings.IDLE_POLICIES,
        "every_nth",
        _EveryNthIdlePolicySettings,
    )
    idle_policy = {"kind": "every_nth", "every_nth_round": 2}
    config_path = yaml_configs.write_config(
        tmp_path, {"idle_policy": idle_policy}
    )

    config = experiment.load_experiment(config_path)

    first_point = config.first_point_task()
    expected = _EveryNthIdlePolicySettings(every_nth_round=2)
    assert first_point.settings.idle_policy == expected


@dataclasses.dataclass(frozen=True)
class _EveryNthIdlePolicySettings:
    """An idle policy record with one key of its own."""

    every_nth_round: int = 1


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
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings

    machine = machine_module.Machine.build(settings, 0)

    built = machine.qpu.factory
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
    first_point = config.first_point_task()
    infinite = magic_state_factories.InfiniteFactory.Settings()
    assert first_point.settings.magic_state_factory == infinite


def test_the_bulk_strong_key_reaches_the_decoder_manager(tmp_path):
    config_path = yaml_configs.write_config(
        tmp_path, {"decoder_manager": {"bulk_strong": True}}
    )
    config = experiment.load_experiment(config_path)
    first_point = config.first_point_task()
    assert first_point.settings.decoder_manager.bulk_strong is True


def test_the_default_decoder_manager_does_not_batch(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    first_point = config.first_point_task()
    assert first_point.settings.decoder_manager.bulk_strong is False


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
    block = {"axes": {"qpu.distance": 3}, "collection": {"max_shots": 1}}
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block]})
    sentence = "sweep block 1 axis qpu.distance must be a list"
    with pytest.raises(ValueError, match=sentence):
        experiment.load_experiment(config_path)


def test_an_axis_path_that_is_not_text_is_refused(tmp_path):
    block = {"axes": {3: [1]}, "collection": {"max_shots": 1}}
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block]})
    sentence = (
        "sweep block 1 axis 3 is not a yaml path; an axis names the "
        "setting it sets by its dotted path, such as qpu.distance"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        experiment.load_experiment(config_path)


def test_a_yaml_key_that_is_not_text_is_refused(tmp_path):
    """A point's record and id are json, and json's keys are text.

    Python's json writes the key 1 as "1" (docs.python.org, json, "Keys in
    key/value pairs of JSON are always of the type str"), so a key of 1
    and a key of "1" would give two points one id.
    """
    clocks = {"fridge": 250.0, "room": 250.0, 1: 100.0}
    config_path = yaml_configs.write_config(tmp_path, {"clocks": clocks})
    sentence = (
        "the yaml key 1 among ['fridge', 'room', 1] is not text; a "
        "point's record "
        "and its id are json, whose keys are text, so 1 and '1' would "
        "name one point"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        experiment.load_experiment(config_path)


# A decoder row with keys of its own, one value of a weak_decoder axis.
PRICED_DECODER_ROW = {
    "kind": 0.05,
    "units": 2,
    "unit_memory": {"bits": None},
    "engine": {
        "clock": "fridge",
        "fetch_cycles_per_round": 2,
        "fetch_cycles_per_job": 0,
        "release_cycles_per_job": 1,
        "release_cycles_per_round": 0,
    },
}
MEMORY_X = "surface_code:rotated_memory_x"
MEMORY_Z = "surface_code:rotated_memory_z"


def test_every_axis_kind_resolves_to_the_settings_a_written_file_gives(
    tmp_path,
):
    """A scalar, a mapping that replaces a row, and a referenced value.

    The referent is the same machine written out by hand with no axis
    and no reference: Hydra's multi-run sets a node by its dotted path
    and a config-group override replaces a sub-config, so the swept
    point and the written file are one machine.
    """
    swept_block = {
        "axes": {
            yaml_configs.ERROR_RATE_PATH: [0.002],
            "qpu.distance": [5],
            "qpu.round_period_microseconds": [0.5],
            "windows.commit_rounds": [2],
            "weak_decoder": [PRICED_DECODER_ROW],
        },
        "collection": {"max_shots": 1},
    }
    swept_folder = tmp_path / "swept"
    swept_folder.mkdir()
    swept_path = yaml_configs.write_config(
        swept_folder, {"sweep": [swept_block]}
    )
    written_workload = yaml_configs.memory_workload(15)
    written_workload["arguments"]["distance"] = 5
    written_workload["arguments"]["physical_error_probability"] = 0.002
    written = {
        "qpu": {
            "kind": "stim_device",
            "distance": 5,
            "round_period_microseconds": 0.5,
        },
        "workload": written_workload,
        "windows": {
            "kind": "sliding",
            "commit_rounds": 2,
            "buffer_rounds": None,
        },
        "weak_decoder": PRICED_DECODER_ROW,
        "sweep": [{"axes": {}, "collection": {"max_shots": 1}}],
    }
    written_folder = tmp_path / "written"
    written_folder.mkdir()
    written_path = yaml_configs.write_config(written_folder, written)
    swept = experiment.load_experiment(swept_path)
    by_hand = experiment.load_experiment(written_path)

    swept_task = swept.first_point_task()
    written_task = by_hand.first_point_task()

    swept_settings = collect.json_value(swept_task.settings)
    written_settings = collect.json_value(written_task.settings)
    assert swept_settings == written_settings
    assert swept_task.settings.weak_decoder.unit_count == 2
    assert written_task.metadata == {}


def test_a_sweep_block_is_every_combination_of_its_axes_in_written_order(
    tmp_path,
):
    """itertools.product over the axes in the order the file writes them.

    The helper writes a yaml with its keys sorted, so weak_decoder.kind is
    the outer axis and the basis the inner one.
    """
    block = {
        "axes": {
            "weak_decoder.kind": [0.028, "pymatching"],
            "workload.arguments.code_task": [MEMORY_X, MEMORY_Z],
        },
        "collection": {"max_shots": 1},
    }
    workload = yaml_configs.memory_workload(15)
    workload["arguments"]["physical_error_probability"] = 0.001
    qpu = {"kind": "stim_device", "distance": 3}
    card = {"qpu": qpu, "workload": workload, "sweep": [block]}
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)

    tasks = config.tasks()

    metadata = [task.metadata for task in tasks]
    assert metadata == [
        {"weak_decoder.kind": 0.028, "workload.arguments.code_task": MEMORY_X},
        {"weak_decoder.kind": 0.028, "workload.arguments.code_task": MEMORY_Z},
        {
            "weak_decoder.kind": "pymatching",
            "workload.arguments.code_task": MEMORY_X,
        },
        {
            "weak_decoder.kind": "pymatching",
            "workload.arguments.code_task": MEMORY_Z,
        },
    ]
    kinds = [task.settings.weak_decoder.algorithm.name for task in tasks]
    assert kinds == [0.028, 0.028, "pymatching", "pymatching"]
    z_arguments = tasks[1].settings.workload.row_settings.arguments
    assert z_arguments["code_task"] == MEMORY_Z


def test_an_axis_under_a_section_that_is_not_there_is_refused(tmp_path):
    block = {
        "axes": {"windows.commit.rounds": [2]},
        "collection": {"max_shots": 1},
    }
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block]})
    sentence = (
        "the sweep axis windows.commit.rounds names windows.commit, but "
        "windows has no key commit; its keys are ['buffer_rounds', "
        "'commit_rounds', 'kind']"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        experiment.load_experiment(config_path)


def test_a_reference_to_a_path_that_is_not_there_is_refused(tmp_path):
    workload = yaml_configs.memory_workload(15)
    workload["arguments"]["distance"] = "${qpu.distanc}"
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    sentence = (
        "the reference ${qpu.distanc} names qpu.distanc, but qpu has no key "
        "distanc; its keys are ['kind', 'distance', "
        "'round_period_microseconds']"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        experiment.load_experiment(config_path)


def test_a_reference_cycle_is_refused_naming_its_paths(tmp_path):
    workload = yaml_configs.memory_workload("${workload.arguments.distance}")
    workload["arguments"]["distance"] = "${workload.arguments.rounds_per_shot}"
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    sentence = (
        "the references workload.arguments.rounds_per_shot -> "
        "workload.arguments.distance -> workload.arguments.rounds_per_shot "
        "form a cycle"
    )
    pattern = re.escape(sentence)

    with pytest.raises(ValueError, match=pattern):
        experiment.load_experiment(config_path)


def test_one_point_written_two_ways_has_one_id(tmp_path):
    """A reference and the value it names are one point.

    The id is over the resolved settings, as sinter's strong id is over
    the task's (sinter/_data/_task.py:167-204).
    """
    referenced_folder = tmp_path / "referenced"
    referenced_folder.mkdir()
    referenced_path = yaml_configs.write_config(referenced_folder, {})
    literal_workload = yaml_configs.memory_workload(15)
    literal_workload["arguments"]["distance"] = 3
    literal_folder = tmp_path / "literal"
    literal_folder.mkdir()
    literal_path = yaml_configs.write_config(
        literal_folder, {"workload": literal_workload}
    )
    referenced = experiment.load_experiment(referenced_path)
    literal = experiment.load_experiment(literal_path)

    referenced_task = referenced.first_point_task()
    literal_task = literal.first_point_task()

    assert referenced_task.strong_id() == literal_task.strong_id()


def test_a_later_point_leaves_an_earlier_points_values_as_they_were(
    tmp_path,
):
    """A section axis and an axis under it share no mapping across points.

    The qpu axis is the outer one, so both points place the same written
    qpu mapping and then set its distance; the first point keeps 3.
    """
    qpu = {"kind": "stim_device", "distance": 3}
    block = {
        "axes": {"qpu": [qpu], "qpu.distance": [3, 5]},
        "collection": {"max_shots": 1},
    }
    workload = yaml_configs.memory_workload(15)
    workload["arguments"]["physical_error_probability"] = 0.001
    card = {"workload": workload, "sweep": [block]}
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    first_alone = config.first_point_task()

    first, _second = config.tasks()

    assert first.metadata == {"qpu": qpu, "qpu.distance": 3}
    assert first.settings.qpu.distance == 3
    assert first.strong_id() == first_alone.strong_id()


def test_a_qpu_key_the_section_does_not_read_is_refused(tmp_path):
    qpu = {"kind": "stim_device", "colour": 5}
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    sentence = (
        r"qpu does not know \['colour'\]; its keys are "
        r"\['kind', 'code_card', 'distance', 'round_period_microseconds'\]"
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

        def build(self, code, circuit_arguments):
            del circuit_arguments
            return _SplitBits(code, self)

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
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 2,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings

    machine = machine_module.Machine.build(settings, 0)

    code = machine.qpu.syndrome_source.code
    assert code.name == "bivariate-bicycle code [[30,8,2]]"
    assert code.syndrome_bits_per_round(1) == 30


def test_a_card_key_on_a_card_that_declares_none_is_refused(tmp_path):
    qpu = {"kind": "timing_only", "qubit_count": 30}
    config_path = yaml_configs.write_config(tmp_path, {"qpu": qpu})
    sentence = (
        r"qpu does not know \['qubit_count'\]; its keys are "
        r"\['kind', 'code_card', 'distance', 'round_period_microseconds'\]"
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
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings

    machine = machine_module.Machine.build(settings, 0)

    own_settings = _SplitBits.Settings(one_payload_per_patch=True)
    assert machine.qpu.syndrome_source.settings == own_settings
    assert machine.qpu.syndrome_source.one_payload_per_patch is True


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
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings

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
    assert machine.qpu.syndrome_source.burst == burst
    assert len(result.operation_results) == 1


def test_a_mode_without_its_tier_is_refused(tmp_path):

    config_path = yaml_configs.write_config(
        tmp_path, {"escalation": {"kind": "strong_only"}}
    )  # only weak_decoder is defined
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    with pytest.raises(ValueError, match="the run names no decoder"):
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
                    "axes": {
                        "workload.arguments.physical_error_probability": [
                            0.001
                        ],
                        "qpu.distance": [3],
                        "qpu.round_period_microseconds": [1.0],
                    },
                    "algorithm_latency_us": [0.028],
                    "collection": {"max_shots": 1},
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="a block is axes"):
        experiment.load_experiment(old_sweep_axis)

    fixed_axis_key = yaml_configs.write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3],
                    "round_period_microseconds": [1.0],
                    "collection": {"max_shots": 1},
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="sweep block 1 is .*a block is axes"):
        experiment.load_experiment(fixed_axis_key)


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
    rows, run_dir = yaml_configs.folded_run(tmp_path, measurements)
    links_path = run_dir / "shot_links.csv"
    per_link = report.read_rows(links_path)
    assert len(rows) == 1
    assert rows[0]["algorithm"] == 0.028
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
                    "axes": {
                        "workload.arguments.physical_error_probability": [
                            0.001
                        ],
                        "qpu.distance": [3, 5],
                        "qpu.round_period_microseconds": [1.0],
                    },
                    "collection": {"max_shots": 1},
                }
            ],
        },
    )
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    at_three = point.settings
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 5,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    at_five = point.settings
    assert at_three.workload.rounds_policy.rounds_by_operation == ((1, 30),)
    assert at_five.workload.rounds_policy.rounds_by_operation == ((1, 50),)
    measurement = yaml_configs.measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=5,
        round_period_microseconds=1.0,
        seed=0,
    )
    assert measurement.decoded_windows > 5


def test_a_run_writes_its_run_record_and_per_shot_records(
    tmp_path, monkeypatch
):
    # One tiny run end to end: a dated results folder that holds
    # run.json, the config copy, shots.csv, sweep.csv and shot_links.csv.
    import csv

    config_path = yaml_configs.write_config(tmp_path, {})
    monkeypatch.chdir(tmp_path)
    run_dir, rows = collect_command.run_experiment(config_path)

    manifest_path = run_dir / "run.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    assert manifest["versions"]["packages"]["stim"]
    assert manifest["run_files"] == [str(config_path)]
    config_copy = run_dir / "config" / config_path.name
    assert config_copy.read_text() == config_path.read_text()
    assert manifest["started_utc"] and manifest["finished_utc"]

    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path) as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)
    assert len(shots) == 1 and shots[0]["seed"] == "0"

    assert run_dir.parent.name == "results"
    assert run_dir.name.endswith("_unit_test_config")
    assert (run_dir / "sweep.csv").exists()
    assert (run_dir / "shot_links.csv").exists()


def test_a_points_record_holds_the_sections_it_resolved_to(tmp_path):
    """The yaml of one point: its axes placed, its references resolved.

    Hydra keeps each job's composed config beside its output
    (.hydra/config.yaml, hydra.cc "Output/Working directory"); a point's
    record holds the same, so the point reruns from its record alone.
    """
    config_path = yaml_configs.write_config(tmp_path, {})
    run_dir = tmp_path / "run"

    collect_command.run_experiment(config_path, run_dir)

    records = run_folder.point_records(run_dir)
    (record,) = records.values()
    sections = record["sections"]
    assert sections["workload"]["arguments"]["distance"] == 3
    assert "sweep" not in sections


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


def test_show_names_each_file_of_an_extends_chain_by_its_plain_path():
    """A base in a sibling folder is shown without the `..` that reached it.

    configs/examples/two_tiers.yaml's base reads as
    configs/bases/weak_decoder_baseline.yaml.
    """
    config_path = yaml_configs.CONFIGS_DIR / "examples" / "two_tiers.yaml"
    base_path = (
        yaml_configs.CONFIGS_DIR / "bases" / "weak_decoder_baseline.yaml"
    )
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    settings = task.settings

    lines = experiment.resolved_description(config, settings)

    assert lines[0] == f"config: {config_path} <- {base_path}"


def test_a_base_past_a_linked_folder_is_the_one_the_filesystem_finds(
    tmp_path,
):
    """`..` after a symbolic link names the link target's parent.

    POSIX path resolution walks a link before the `..` that follows it
    (IEEE 1003.1, 4.13), so a yaml reached through a linked folder
    extends the base beside its real folder, as open() finds it.
    """
    nested = tmp_path / "real" / "nested"
    nested.mkdir(parents=True)
    logical = tmp_path / "logical"
    logical.mkdir()
    real_workload = yaml_configs.memory_workload(15)
    logical_workload = yaml_configs.memory_workload(30)
    base_path = yaml_configs.write_config(
        nested.parent, {"workload": real_workload}
    )
    yaml_configs.write_config(logical, {"workload": logical_workload})
    child_path = nested / "run.yaml"
    child_path.write_text(f"extends: ../{base_path.name}\n")
    linked = logical / "linked"
    linked.symlink_to(nested, target_is_directory=True)
    linked_child = linked / "run.yaml"

    config = experiment.load_experiment(linked_child)

    assert config.sections["workload"] == real_workload


def test_an_extends_chain_that_returns_to_a_file_is_refused(tmp_path):
    """A file is named once however the chain spells it."""
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    (tmp_path / "sub").mkdir()
    first_path.write_text("extends: second.yaml\n")
    second_path.write_text("extends: sub/../first.yaml\n")
    chain = f"{first_path} -> {second_path} -> {first_path}"
    sentence = f"the extends chain {chain} forms a cycle"
    pattern = re.escape(sentence)

    with pytest.raises(refusal.RefusalError, match=pattern):
        experiment.load_experiment(first_path)


def test_a_file_that_extends_itself_is_refused(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("extends: run.yaml\n")
    sentence = f"the extends chain {config_path} -> {config_path} forms"
    pattern = re.escape(sentence)

    with pytest.raises(refusal.RefusalError, match=pattern):
        experiment.load_experiment(config_path)


def test_two_config_files_of_one_name_are_both_copied(tmp_path):
    """The chain's files keep their places, so a base of one name stays."""
    base_path = yaml_configs.write_config(tmp_path, {})
    child_folder = tmp_path / "child"
    child_folder.mkdir()
    child_path = child_folder / base_path.name
    child_path.write_text(f"extends: ../{base_path.name}\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_folder.snapshot_code_state(child_path, run_dir)

    child_copy = run_dir / "config" / "child" / base_path.name
    base_copy = run_dir / "config" / base_path.name
    assert child_copy.read_text() == child_path.read_text()
    assert base_copy.read_text() == base_path.read_text()


def test_a_yaml_point_keeps_the_id_its_task_has(tmp_path):
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    (task,) = config.tasks()

    study = config.experiment()

    (point,) = study.points
    point_task = experiment.task_of(point)
    point_id = task.strong_id()
    assert point_task.strong_id() == point_id
    assert point.name == point_id[:12]


def test_a_yaml_point_two_blocks_name_is_one_point(tmp_path):
    block = yaml_configs.MINIMAL_CONFIG["sweep"][0]
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block, block]})
    config = experiment.load_experiment(config_path)

    study = config.experiment()

    assert len(study.points) == 1


def test_one_point_reads_no_point_past_the_one_it_chooses(
    tmp_path, monkeypatch
):
    """A point's read runs its maker, so a narrated shot reads no other.

    Without a name only the first point is read; with one, the points up
    to the named one, since a yaml point is named by its id.
    """
    axes = {
        "qpu.distance": [3, 5, 7],
        "workload.arguments.physical_error_probability": [0.001],
        "qpu.round_period_microseconds": [1.0],
    }
    card = {"sweep": [{"axes": axes, "collection": {"max_shots": 2}}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    study = config.experiment()
    second_name = study.points[1].name
    read_distances = []
    point_task = experiment.ExperimentConfig.point_task

    def counted_point_task(self, values):
        distance = values["qpu.distance"]
        read_distances.append(distance)
        return point_task(self, values)

    monkeypatch.setattr(
        experiment.ExperimentConfig, "point_task", counted_point_task
    )

    config.one_point()
    first_reads = list(read_distances)
    read_distances.clear()
    config.one_point(second_name)

    assert first_reads == [3]
    assert read_distances == [3, 5]


def _record_options_of(tmp_path, overrides: dict):
    """The record options and settings the yaml's first point reads to."""
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    return task.record_options, task.settings


def test_the_record_options_are_read_beside_what_they_record(tmp_path):
    """They are written under observation and burst_detector, run-owned.

    The machine never reads them, so they are on the task beside the
    settings and no part of the point's id.
    """
    overrides = yaml_configs.fixed_threshold_switching()
    overrides["observation"] = {"confidence_shot_count": "all"}
    overrides["burst_detector"] = {
        "kind": "masked_regional_cusum",
        "catch_deadline_rounds": 0,
    }
    written, settings = _record_options_of(tmp_path, overrides)
    unset, _settings = _record_options_of(tmp_path, {})

    assert written == collect.RecordOptions(None, 0)
    assert unset == collect.RecordOptions(100, 300)
    a_billion = 10**9
    assert written.samples_confidence_of(a_billion)
    assert unset.samples_confidence_of(99)
    assert not unset.samples_confidence_of(100)
    assert not hasattr(settings.observation, "confidence_shot_count")
    assert not hasattr(
        settings.switching.burst_detector, "catch_deadline_rounds"
    )


@pytest.mark.parametrize("written", [-1, True, "some", 2.5])
def test_a_confidence_shot_count_that_is_no_count_is_refused(tmp_path, written):
    """A count of shots from seed 0, or the word all; nothing else."""
    overrides = {"observation": {"confidence_shot_count": written}}

    with pytest.raises(ValueError, match="confidence_shot_count must be"):
        _record_options_of(tmp_path, overrides)


def test_a_catch_deadline_is_a_whole_number_of_rounds(tmp_path):
    detector = {"kind": "masked_regional_cusum", "catch_deadline_rounds": -1}

    with pytest.raises(ValueError, match="catch_deadline_rounds must be"):
        _record_options_of(tmp_path, {"burst_detector": detector})


def test_no_detector_takes_no_catch_deadline(tmp_path):
    """With no detector there is no flag to time, so the key is refused."""
    detector = {"kind": "none", "catch_deadline_rounds": 300}

    with pytest.raises(ValueError, match="burst_detector does not know"):
        _record_options_of(tmp_path, {"burst_detector": detector})


def test_a_swept_section_keeps_its_column_beside_every_measured_one(
    tmp_path,
):
    """A whole yaml section is an axis, and no measured column shares it.

    A row takes its swept values first and its measured values after
    (report._with_swept_values), so a measured column named as a section
    would overwrite the swept value in its cell. The swept windows cell
    is the section's compact json (run_folder.swept_values), and no csv
    column the run wrote is named as another yaml section.
    """
    windows = yaml_configs.MINIMAL_CONFIG["windows"]
    block = dict(yaml_configs.MINIMAL_CONFIG["sweep"][0])
    block["axes"] = dict(block["axes"], windows=[windows])
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block]})
    out_dir = tmp_path / "out"
    run_dir, _ = collect_command.run_experiment(config_path, out_dir)
    shots_path = run_dir / "shots.csv"
    shot, *_ = _csv_rows(shots_path)
    columns = _columns_of_every_csv(run_dir)
    sections_written = columns & set(machine_settings.SECTIONS)

    assert json.loads(shot["windows"]) == windows
    assert int(shot["decoded_windows"]) > 0
    assert sections_written == {"windows"}


def test_a_task_named_by_two_blocks_runs_once(tmp_path):
    raw = _reference_yaml()
    raw["sweep"] = [raw["sweep"][0], dict(raw["sweep"][0])]
    raw["collection"]["max_shots"] = 2
    twice_path = tmp_path / "twice.yaml"
    twice = _written_yaml(raw, twice_path)
    config = experiment.load_experiment(twice)
    tasks = config.tasks()
    out_dir = tmp_path / "out"

    run_dir, _ = collect_command.run_experiment(twice, out_dir)

    shots_path = run_dir / "shots.csv"
    shots = _csv_rows(shots_path)
    seeds = [shot["seed"] for shot in shots]
    assert len(tasks) == 2
    assert seeds == ["0", "1"]


def test_two_tasks_that_differ_only_in_bandwidth_are_two_tasks(tmp_path):
    """A yaml link rate is an exact Fraction and enters the id as its text.

    sinter's strong id is the sha256 of every value's json text
    (sinter/_data/_task.py strong_id_value), so two values give two ids.
    """
    narrow, wide = _bandwidth_tasks(tmp_path, (8, 64))

    unique = collect.unique_tasks([narrow, wide])
    narrow_shot = collect.run_shot(narrow, 0)
    wide_shot = collect.run_shot(wide, 0)

    assert narrow.strong_id() != wide.strong_id()
    assert len(unique) == 2
    assert _readout_rate(narrow_shot) == "2000"
    assert _readout_rate(wide_shot) == "16000"
    assert collect.json_value(narrow.settings) != collect.json_value(
        wide.settings
    )


def test_one_yaml_saved_under_two_names_names_its_points_alike(tmp_path):
    """A point is named by what it runs, not by the file that says it.

    sinter keeps a task's circuit_path out of its strong id and hashes
    the circuit's text (sinter/_data/_task.py:157, 167-204). The
    reference yaml saved as t2.yaml and as t6.yaml runs the same points,
    so its points' ids are equal, though the links card's labels name
    the two files.
    """
    first_path = tmp_path / "t2.yaml"
    second_path = tmp_path / "t6.yaml"
    shutil.copyfile(REFERENCE_YAML, first_path)
    shutil.copyfile(REFERENCE_YAML, second_path)

    first_ids = _point_ids_of(first_path)
    second_ids = _point_ids_of(second_path)

    assert first_ids
    assert first_ids == second_ids


def _reference_yaml() -> dict:
    text = REFERENCE_YAML.read_text()
    return yaml.safe_load(text)


def _written_yaml(raw: dict, path: pathlib.Path) -> pathlib.Path:
    text = yaml.safe_dump(raw)
    path.write_text(text)
    return path


def _columns_of_every_csv(run_dir) -> set:
    """Every column name any csv file under the folder holds."""
    columns = set()
    for path in run_dir.rglob("*.csv"):
        header = fold.header_of(path)
        columns.update(header)
    return columns


def _bandwidth_tasks(tmp_path, widths: tuple) -> list:
    """The data-movement grid's switching task, its readout hop at each width.

    The weak base the grid extends holds the hop, and that one file is
    edited in place between loads, so the tasks differ in the hop's rate
    and in nothing else, the path included.
    """
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    path = configs / "bases" / "weak_decoder_baseline.yaml"
    grid = configs / "experiments" / "data_movement" / "data_movement.yaml"
    text = path.read_text()
    unpriced = "clock: fridge, bits_per_cycle: null}"
    readout_hop = f"qpu_to_controller:  {{latency_cycles: 1, {unpriced}"
    assert readout_hop in text
    tasks = []
    for bits_per_cycle in widths:
        priced = f"clock: fridge, bits_per_cycle: {bits_per_cycle}}}"
        priced_hop = readout_hop.replace(unpriced, priced)
        priced_text = text.replace(readout_hop, priced_hop)
        path.write_text(priced_text)
        config = experiment.load_experiment(grid)
        task = _switching_task(config)
        tasks.append(task)
    return tasks


def _switching_task(config: experiment.ExperimentConfig):
    """The grid's switching block, its first point at p = 0.001."""
    switching_block = config.sweep[DATA_MOVEMENT_SWITCHING_BLOCK]
    points = switching_block.points()
    values = dict(points[0])
    values[yaml_configs.ERROR_RATE_PATH] = 0.001
    return config.point_task(values)


def _readout_rate(shot) -> str:
    """The readout hop's rate, the one setting two bandwidth tasks differ in."""
    capacity = shot.task.settings.links.qpu_to_controller.channel.capacity
    return str(capacity.input_bits_per_microsecond)


def _point_ids_of(config_path: pathlib.Path) -> list:
    config = experiment.load_experiment(config_path)
    point_ids = []
    for task, _collection in config.point_tasks():
        point_id = task.strong_id()
        point_ids.append(point_id)
    return point_ids


def _csv_rows(path: pathlib.Path) -> list:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)
