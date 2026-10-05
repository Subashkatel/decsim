"""The workload record: a made workload is a value a point is built on.

Every shot of a point builds from one record, so the record hashes and
writes to json; a workload read from files is refused at the record when
its circuit cannot be laid out.
"""

import importlib.util
import json

import pytest
import stim

import decsim.experiments.collect as collect
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files
import decsim.producers as producers
import decsim.records.circuits as circuit_records

# One live fragment for every round: a single measurement.
ONE_MEASUREMENT = stim.Circuit("M 0")
LIVE_FRAGMENTS = circuit_records.RepeatedStimCircuit(
    ONE_MEASUREMENT, ONE_MEASUREMENT, ONE_MEASUREMENT, ONE_MEASUREMENT
)
# The Deltakit makers need the Deltakit packages, which no extra installs.
DELTAKIT_SPECIFICATION = importlib.util.find_spec("deltakit_explorer")
IS_DELTAKIT_ABSENT = DELTAKIT_SPECIFICATION is None
NEEDS_DELTAKIT = pytest.mark.skipif(
    IS_DELTAKIT_ABSENT, reason="could not import 'deltakit_explorer'"
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
    pytest.param(
        producers.deltakit_memory, (3, 3, 0.001), marks=NEEDS_DELTAKIT
    ),
    pytest.param(
        producers.deltakit_live_memory,
        (3, 0.001, 1.0, 2),
        marks=NEEDS_DELTAKIT,
    ),
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


def test_one_circuit_under_two_operations_from_files_is_refused(tmp_path):
    """No merged circuit is built, so each needs its round range."""
    operations = {
        "schema": "decsim.ops/1",
        "operations": [
            {"id": 1, "patches": [0]},
            {"id": 2, "patches": [0, 1]},
        ],
    }
    operations_path = _write_json(tmp_path, "ops.json", operations)
    circuit_path = tmp_path / "history.stim"
    circuit_path.write_text("M 0\nDETECTOR rec[-1]\n")
    rounds_path = _write_json(tmp_path, "rounds.json", {"0": 1})
    workload = workload_files.read_workload(
        operations_path,
        circuit_path=circuit_path,
        measurement_rounds_path=rounds_path,
    )

    with pytest.raises(ValueError, match="none names its round range"):
        workload_settings.WorkloadSettings.running(workload)


def test_an_operations_file_of_another_schema_is_refused(tmp_path):
    document = {"schema": "decsim.ops/2", "operations": []}
    operations_path = _write_json(tmp_path, "ops.json", document)

    with pytest.raises(ValueError, match="is not a decsim.ops/1 operation"):
        workload_files.read_workload(operations_path)


def _write_json(folder, name: str, value):
    """The value written as json into the folder, and the file's path."""
    text = json.dumps(value)
    path = folder / name
    path.write_text(text)
    return path
