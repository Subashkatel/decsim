"""The workload section: a maker named in the yaml makes the workload.

A maker is a module:function and its arguments, loaded the way Python's
entry points load a named object (CPython Lib/importlib/metadata/
__init__.py:199-207) and called the way Hydra's instantiate calls a
target (hydra/_internal/instantiate/_instantiate2.py:76-82). The tests
below run a maker written outside decsim with no change to decsim's
tables, and pin the sentence every refusal reads as.
"""

import json
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


# The audit's merge probe: two memories, their merge, a measurement, as
# an operation list with kinds and no circuit, run on the code card's
# timing alone. The merge's predecessors come from patch order.
MERGE_OPERATIONS = {
    "schema": "decsim.ops/1",
    "operations": [
        {"id": 1, "name": "mem0", "patches": [0], "kind": "MEMORY"},
        {"id": 2, "name": "mem1", "patches": [1], "kind": "MEMORY"},
        {"id": 3, "name": "merge01", "patches": [0, 1], "kind": "MERGE"},
        {"id": 4, "name": "measure", "patches": [0], "kind": "MEASURE"},
    ],
}


def _write_json(folder, name: str, value) -> None:
    text = json.dumps(value)
    path = folder / name
    path.write_text(text)


def _files_config(tmp_path, workload: dict, qpu_kind="timing_only"):
    card = {"workload": workload, "qpu": {"kind": qpu_kind}}
    return yaml_configs.write_config(tmp_path, card)


def test_the_merge_probe_runs_from_an_operations_file(tmp_path):
    """Paths are read from the yaml's folder, not the working directory."""
    _write_json(tmp_path, "merge.json", MERGE_OPERATIONS)
    workload = {"kind": "files", "operations": "merge.json"}
    config_path = _files_config(tmp_path, workload)
    settings = _point(config_path)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    merge = machine.operations[2]
    policy = settings.workload.rounds_policy

    assert result.terminal_status == "complete"
    assert len(result.operation_results) == 4
    assert merge.predecessors == (1, 2)
    assert policy is None


def test_an_inherited_files_row_reads_beside_the_yaml_that_wrote_it(
    tmp_path,
):
    """A child that extends a base in another folder reads the base's files.

    A file of the same name beside the child is not the one read.
    """
    base_folder = tmp_path / "base"
    base_folder.mkdir()
    _write_json(base_folder, "merge.json", MERGE_OPERATIONS)
    _write_json(tmp_path, "merge.json", {"schema": "not decsim.ops/1"})
    workload = {"kind": "files", "operations": "merge.json"}
    base_path = _files_config(base_folder, workload)
    child_path = tmp_path / "child.yaml"
    child_path.write_text(f"extends: base/{base_path.name}\n")
    expected_path = base_folder / "merge.json"

    config = experiment.load_experiment(child_path)

    files = config.settings.workload.row_settings
    assert files.operations == expected_path


def test_one_circuit_under_two_operations_from_files_is_refused(tmp_path):
    """No merged circuit is built, so each needs its round range."""
    operations = {
        "schema": "decsim.ops/1",
        "operations": [
            {"id": 1, "patches": [0]},
            {"id": 2, "patches": [0, 1]},
        ],
    }
    _write_json(tmp_path, "ops.json", operations)
    circuit_path = tmp_path / "history.stim"
    circuit_path.write_text("M 0\nDETECTOR rec[-1]\n")
    _write_json(tmp_path, "rounds.json", {"0": 1})
    workload = {
        "kind": "files",
        "operations": "ops.json",
        "circuit": "history.stim",
        "measurement_rounds": "rounds.json",
    }
    config_path = _files_config(tmp_path, workload, "stim_device")

    with pytest.raises(ValueError, match="none names its round range"):
        _point(config_path)


def test_a_files_section_naming_two_physical_circuits_is_refused(tmp_path):
    """read_workload would read one and leave the other unread."""
    workload = {
        "kind": "files",
        "operations": "ops.json",
        "circuit": "c.stim",
        "measurement_rounds": "r.json",
        "fragments": "live",
    }
    config_path = _files_config(tmp_path, workload)

    with pytest.raises(ValueError, match="a workload carries one physical"):
        experiment.load_experiment(config_path)


def test_a_files_section_naming_a_missing_file_stops_at_the_point(tmp_path):
    """Python's own error names the file."""
    workload = {"kind": "files", "operations": "absent.json"}
    config_path = _files_config(tmp_path, workload)

    with pytest.raises(FileNotFoundError, match="absent.json"):
        _point(config_path)


def test_an_operations_file_of_another_schema_is_refused(tmp_path):
    """Another schema's fields could read as decsim.ops/1's and run wrong."""
    document = {"schema": "decsim.ops/2", "operations": []}
    _write_json(tmp_path, "ops.json", document)
    workload = {"kind": "files", "operations": "ops.json"}
    config_path = _files_config(tmp_path, workload)

    with pytest.raises(ValueError, match="is not a decsim.ops/1 operation"):
        _point(config_path)
