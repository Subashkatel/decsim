"""The workload section: a maker named in the yaml makes the workload.

A maker is a module:function and its arguments, loaded the way Python's
entry points load a named object (CPython Lib/importlib/metadata/
__init__.py:199-207) and called the way Hydra's instantiate calls a
target (hydra/_internal/instantiate/_instantiate2.py:76-82). The tests
below run a maker written outside decsim with no change to decsim's
tables, and pin the sentence every refusal reads as.
"""

import dataclasses
import json
import pathlib
import textwrap

import pytest
import stim

import decsim.collect as collect
import decsim.experiments.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.producers as producers
import decsim.records.circuits as circuit_records
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

# A point that sets the distance only, for a maker that takes no error
# rate, and one that sets both, for Stim's memory circuit.
AT_DISTANCE_3 = {"qpu.distance": 3}

AT_DISTANCE_3_AND_P = {
    "qpu.distance": 3,
    "workload.arguments.physical_error_probability": 0.001,
}

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
# One operation that runs the files row's finite circuit.
ONE_OPERATION_ON_A_CIRCUIT = {
    "schema": "decsim.ops/1",
    "operations": [{"id": 1, "patches": [0]}],
}


# One live fragment for every round: a single measurement.
ONE_MEASUREMENT = stim.Circuit("M 0")
LIVE_FRAGMENTS = circuit_records.RepeatedStimCircuit(
    ONE_MEASUREMENT, ONE_MEASUREMENT, ONE_MEASUREMENT, ONE_MEASUREMENT
)
# Every maker decsim ships, with the arguments of a small point.
SHIPPED_MAKERS = [
    (
        producers.memory_circuit,
        ("surface_code:rotated_memory_z", 6, 3, 0.001),
    ),
    (
        producers.memory_patches,
        ("surface_code:rotated_memory_z", 4, 2, 3, 0.001),
    ),
    (producers.deltakit_memory, (3, 3, 0.001)),
    (producers.deltakit_live_memory, (3, 0.001, 1.0, 2)),
    (producers.live_memory, (LIVE_FRAGMENTS, 2)),
]


@pytest.mark.parametrize("maker, arguments", SHIPPED_MAKERS)
def test_a_made_workload_hashes_and_writes_to_json(maker, arguments):
    """One record builds every shot of a point, so it is a value.

    Two records of one workload hash alike, and the record writes to
    json, as a point's id and its machine.json read it.
    """
    workload = maker(*arguments)
    settings = workload_settings.WorkloadSettings.running(workload)
    again = workload_settings.WorkloadSettings.running(workload)

    value = collect.json_value(settings)
    text = json.dumps(value)

    assert hash(settings) == hash(again)
    assert json.loads(text) == value


def test_a_workload_a_yaml_makes_hashes_and_writes_to_json(tmp_path):
    """The files row and its finite circuit, made at the point."""
    _write_json(tmp_path, "ops.json", ONE_OPERATION_ON_A_CIRCUIT)
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
    settings = _point(config_path, AT_DISTANCE_3)
    again = _point(config_path, AT_DISTANCE_3)

    value = collect.json_value(settings.workload)
    text = json.dumps(value)

    assert hash(settings.workload) == hash(again.workload)
    assert json.loads(text) == value


def test_making_a_workload_keeps_every_field_the_section_set():
    """The maker's workload fills the lowered fields and no other."""
    section = {
        "kind": "producer",
        "function": "decsim.producers:memory_circuit",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": 6,
            "distance": 3,
            "physical_error_probability": 0.001,
        },
    }
    read = workload_settings.WorkloadSettings.from_yaml(section, None)
    hand_set = dataclasses.replace(
        read, decode_operations=(), feedback_boundary_mode="measurement_closed"
    )

    made = hand_set.made()

    assert made.decode_operations == ()
    assert made.feedback_boundary_mode == "measurement_closed"
    assert made.kind == "producer"
    assert made.row_settings == hand_set.row_settings
    assert len(made.operations) == 1


def test_a_python_workload_is_the_point_the_yaml_maker_makes(tmp_path):
    """The maker's name and its "5d" are the yaml's labels, not the id.

    The lowered workload names the point, so a Python caller states the
    workload directly and names the same point.
    """
    yaml_workload = yaml_configs.memory_workload("5d")
    config_path = yaml_configs.write_config(
        tmp_path, {"workload": yaml_workload}
    )
    config = experiment.load_experiment(config_path)
    task = config.point_task(AT_DISTANCE_3_AND_P)
    made = producers.memory_circuit(
        "surface_code:rotated_memory_z", 15, 3, 0.001
    )
    python_workload = workload_settings.WorkloadSettings.running(made)
    python_settings = dataclasses.replace(
        task.settings, workload=python_workload
    )
    python_task = collect.Task(python_settings, task.metadata)

    assert python_workload == task.settings.workload
    assert python_task.strong_id() == task.strong_id()


def test_a_maker_written_outside_decsim_runs_from_a_yaml(monkeypatch, tmp_path):
    """Its module:function and its arguments are the whole edit."""
    _outside_package(tmp_path, monkeypatch)
    workload = _producer(
        "outside_makers.makers:two_patch_memory",
        {"patch_rounds": 4, "distance": "${qpu.distance}"},
    )
    card = {"workload": workload, "qpu": {"kind": "timing_only"}}
    config_path = yaml_configs.write_config(tmp_path, card)
    settings = _point(config_path, AT_DISTANCE_3)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    rounds = settings.workload.rounds_policy.rounds_by_operation

    assert result.terminal_status == "complete"
    assert len(machine.plan.all_operations) == 2
    assert rounds == ((1, 4), (2, 7))


@pytest.mark.parametrize("arguments, rounds", [({}, 3), ({"rounds": 9}, 9)])
def test_a_maker_with_keyword_arguments_takes_what_the_yaml_writes(
    monkeypatch, tmp_path, arguments, rounds
):
    """Python's own call rules: **options takes any argument the yaml has."""
    _outside_package(tmp_path, monkeypatch)
    arguments = {"distance": "${qpu.distance}", **arguments}
    workload = _producer(
        "outside_makers.makers:rounds_from_keywords", arguments
    )
    card = {"workload": workload, "qpu": {"kind": "timing_only"}}
    config_path = yaml_configs.write_config(tmp_path, card)

    settings = _point(config_path, AT_DISTANCE_3_AND_P)

    policy = settings.workload.rounds_policy
    assert policy.rounds_by_operation == ((1, rounds),)


def test_a_maker_that_returns_no_workload_is_refused_at_the_point(
    monkeypatch, tmp_path
):
    _outside_package(tmp_path, monkeypatch)
    arguments = {"distance": "${qpu.distance}"}
    workload = _producer("outside_makers.makers:not_a_workload", arguments)
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    sentence = (
        "outside_makers.makers:not_a_workload returned a list; a maker "
        "returns a decsim.records.workload.Workload"
    )

    with pytest.raises(ValueError, match=sentence):
        _point(config_path, AT_DISTANCE_3)


@pytest.mark.parametrize(
    "function, sentence",
    [
        ("decsim.no_such_module:maker", "No module named 'decsim.no_such"),
        ("decsim.producers:no_such_maker", "no attribute 'no_such_maker'"),
    ],
)
def test_a_maker_that_is_not_there_is_refused_at_the_point(
    tmp_path, function, sentence
):
    workload = _producer(function, {})
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})

    with pytest.raises(ValueError, match=sentence):
        _point(config_path, AT_DISTANCE_3)


@pytest.mark.parametrize(
    "arguments, sentence",
    [
        (
            {"code_task": "x", "rounds_per_shot": 3, "colour": 1},
            "got an unexpected keyword argument 'colour'",
        ),
        (
            {"code_task": "x", "distance": "${qpu.distance}"},
            "missing 1 required positional argument",
        ),
    ],
)
def test_a_bad_argument_stops_the_makers_call(tmp_path, arguments, sentence):
    """Python's own call names the argument; decsim adds no check."""
    workload = _producer("decsim.producers:memory_circuit", arguments)
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})

    with pytest.raises(TypeError, match=sentence):
        _point(config_path, AT_DISTANCE_3_AND_P)


def test_a_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    workload = {"kind": "memory_circuit", "rounds_per_shot": 6}
    card = {"workload": workload, "sweep": yaml_configs.QPU_ONLY_SWEEP}
    config_path = yaml_configs.write_config(tmp_path, card)

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
        _point(config_path, AT_DISTANCE_3_AND_P)


def test_the_merge_probe_runs_from_an_operations_file(tmp_path):
    """Paths are read from the yaml's folder, not the working directory."""
    _write_json(tmp_path, "merge.json", MERGE_OPERATIONS)
    workload = {"kind": "files", "operations": "merge.json"}
    config_path = _files_config(tmp_path, workload)
    settings = _point(config_path, AT_DISTANCE_3)
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    merge = machine.plan.all_operations[2]
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

    settings = _point(child_path, AT_DISTANCE_3)

    files = settings.workload.row_settings
    assert files.operations == expected_path


def test_one_files_row_read_from_two_folders_names_its_point_alike(
    tmp_path,
):
    """The files' folder is where they were read; their content is the point."""
    first_folder = tmp_path / "first"
    other_folder = tmp_path / "elsewhere"

    first_point_id = _point_id_of_the_merge_files_in(first_folder)
    other_point_id = _point_id_of_the_merge_files_in(other_folder)

    assert first_point_id == other_point_id


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
        _point(config_path, AT_DISTANCE_3)


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
        _point(config_path, AT_DISTANCE_3)


def test_an_operations_file_of_another_schema_is_refused(tmp_path):
    """Another schema's fields could read as decsim.ops/1's and run wrong."""
    document = {"schema": "decsim.ops/2", "operations": []}
    _write_json(tmp_path, "ops.json", document)
    workload = {"kind": "files", "operations": "ops.json"}
    config_path = _files_config(tmp_path, workload)

    with pytest.raises(ValueError, match="is not a decsim.ops/1 operation"):
        _point(config_path, AT_DISTANCE_3)


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


def _point(config_path, values):
    config = experiment.load_experiment(config_path)
    task = config.point_task(values)
    return task.settings


def _producer(function: str, arguments: dict) -> dict:
    return {"kind": "producer", "function": function, "arguments": arguments}


def _write_json(folder, name: str, value) -> None:
    text = json.dumps(value)
    path = folder / name
    path.write_text(text)


def _files_config(tmp_path, workload: dict, qpu_kind="timing_only"):
    card = {
        "workload": workload,
        "qpu": {"kind": qpu_kind},
        "sweep": yaml_configs.QPU_ONLY_SWEEP,
    }
    return yaml_configs.write_config(tmp_path, card)


def _point_id_of_the_merge_files_in(folder: pathlib.Path) -> str:
    """The point a files yaml in folder names, its operations merge.json."""
    folder.mkdir()
    _write_json(folder, "merge.json", MERGE_OPERATIONS)
    workload = {"kind": "files", "operations": "merge.json"}
    config_path = _files_config(folder, workload)
    config = experiment.load_experiment(config_path)
    task = config.point_task(AT_DISTANCE_3)
    return task.strong_id()
