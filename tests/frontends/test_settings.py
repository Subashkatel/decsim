"""The workload section: a maker named in the yaml makes the workload.

A maker is a module:function and its arguments, loaded the way Python's
entry points load a named object (CPython Lib/importlib/metadata/
__init__.py:199-207) and called the way Hydra's instantiate calls a
target (hydra/_internal/instantiate/_instantiate2.py:76-82). The tests
below run a maker written outside decsim with no change to decsim's
tables, and pin the sentence every refusal reads as.
"""

import textwrap

import pytest

import decsim.experiments.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import tests.experiments.yaml_configs as yaml_configs

OUTSIDE_MAKER = '''
import decsim.records.program as program_records
import decsim.records.workload as workload_records


def two_patch_memory(patch_rounds, distance):
    """Two single-patch memory operations, d rounds apart."""
    first = program_records.Operation(1, "a", (1,), patches=(1,))
    second = program_records.Operation(2, "b", (2,), patches=(2,))
    rounds = {1: patch_rounds, 2: patch_rounds + distance}
    return workload_records.Workload((first, second), rounds)


def not_a_workload(distance):
    return [distance]


def rounds_from_keywords(distance, physical_error_probability, **options):
    """A maker that reads its round count out of its keyword arguments."""
    operation = program_records.Operation(1, "a", (1,), patches=(1,))
    rounds = options.get("rounds", distance)
    return workload_records.Workload((operation,), {1: rounds})
'''


def _outside_package(tmp_path, monkeypatch) -> None:
    """A package outside decsim that holds two makers."""
    package = tmp_path / "outside_makers"
    package.mkdir()
    init_path = package / "__init__.py"
    init_path.write_text("")
    module_path = package / "makers.py"
    maker_text = textwrap.dedent(OUTSIDE_MAKER)
    module_path.write_text(maker_text)
    monkeypatch.syspath_prepend(str(tmp_path))


def _point(config_path):
    config = experiment.load_experiment(config_path)
    return config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )


def _producer(function: str, arguments: dict) -> dict:
    return {"kind": "producer", "function": function, "arguments": arguments}


def test_a_maker_written_outside_decsim_runs_from_a_yaml(monkeypatch, tmp_path):
    """Its module:function and its arguments are the whole edit."""
    _outside_package(tmp_path, monkeypatch)
    workload = _producer(
        "outside_makers.makers:two_patch_memory", {"patch_rounds": 4}
    )
    card = {"workload": workload, "qpu": {"kind": "timing_only"}}
    config_path = yaml_configs.write_config(tmp_path, card)
    settings = _point(config_path)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    rounds = settings.workload.rounds_policy.rounds_by_operation

    assert result.terminal_status == "complete"
    assert len(machine.operations) == 2
    assert rounds == {1: 4, 2: 7}


@pytest.mark.parametrize("arguments, rounds", [({}, 3), ({"rounds": 9}, 9)])
def test_a_maker_with_keyword_arguments_takes_what_the_yaml_writes(
    monkeypatch, tmp_path, arguments, rounds
):
    """Python's own call rules: **options takes any argument the yaml has."""
    _outside_package(tmp_path, monkeypatch)
    workload = _producer(
        "outside_makers.makers:rounds_from_keywords", arguments
    )
    card = {"workload": workload, "qpu": {"kind": "timing_only"}}
    config_path = yaml_configs.write_config(tmp_path, card)

    settings = _point(config_path)

    policy = settings.workload.rounds_policy
    assert policy.rounds_by_operation == {1: rounds}


def test_a_maker_that_returns_no_workload_is_refused_at_the_point(
    monkeypatch, tmp_path
):
    _outside_package(tmp_path, monkeypatch)
    workload = _producer("outside_makers.makers:not_a_workload", {})
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    sentence = (
        "outside_makers.makers:not_a_workload returned a list; a maker "
        "returns a decsim.records.workload.Workload"
    )

    with pytest.raises(ValueError, match=sentence):
        _point(config_path)


def test_a_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    workload = {"kind": "memory_circuit", "rounds_per_shot": 6}
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})

    with pytest.raises(ValueError, match="workload.kind 'memory_circuit' is"):
        experiment.load_experiment(config_path)


def test_a_workload_without_a_kind_is_refused_naming_the_rows():
    """None is the kind of a Python-built workload, never a yaml's."""
    section = {"function": "decsim.producers:memory_circuit"}

    with pytest.raises(ValueError, match="workload.kind None is not a row"):
        workload_settings.WorkloadSettings.from_yaml(section, None)


@pytest.mark.parametrize("rounds_per_shot", [0, -1, True, "0d"])
def test_a_shot_of_fewer_than_one_round_is_refused_at_the_point(
    tmp_path, rounds_per_shot
):
    workload = yaml_configs.memory_workload(rounds_per_shot)
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})

    with pytest.raises(ValueError, match="a round count of at least 1"):
        _point(config_path)
