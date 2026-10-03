"""A maker's workload on disk: decsim.ops/1 and the physical circuit.

read_workload and write_workload hold the operation list as a
decsim.ops/1 json, and the physical circuit as a finite .stim with its
measurement-to-round json or a folder of the four live fragments with
physical.json. The lowering of what they read is circuit_frontend.py.
"""

import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Optional

import stim

import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.workload as workload_records

OPERATIONS_SCHEMA = "decsim.ops/1"
# An operation's keys in decsim.ops/1. id and patches are required; a
# key left out takes records/program.py Operation's default, qubits the
# operation's patches and name "operation <id>". rounds is the
# operation's round count and circuit its own .stim, relative to the
# operations file.
OPERATION_KEYS = (
    "id",
    "patches",
    "qubits",
    "name",
    "kind",
    "rounds",
    "predecessors",
    "blocked_by",
    "stream_id",
    "stream_offset",
    "emits_detector_data",
    "scheduled_start_round",
    "clifford",
    "circuit",
)
# The four live fragments, each its own .stim in the fragments folder
# (records/circuits.py RepeatedStimCircuit), beside physical.json, which
# holds the period the fragments' noise was built for.
FRAGMENT_NAMES = (
    "first_round",
    "repeated_round",
    "final_round",
    "single_round",
)
PHYSICAL_FILE_NAME = "physical.json"
# The Operation fields decsim.ops/1 carries; any other field a maker sets
# has no file form, and writing it would lose it.
_FILE_FIELDS = frozenset(OPERATION_KEYS) - {"rounds"}


def read_workload(
    operations_path: pathlib.Path,
    circuit_path: Optional[pathlib.Path] = None,
    measurement_rounds_path: Optional[pathlib.Path] = None,
    fragments_path: Optional[pathlib.Path] = None,
) -> workload_records.Workload:
    """A maker's two outputs read from disk: the operations and the circuit.

    The physical circuit is a finite .stim with its measurement-to-round
    json, a folder of the four fragments, or neither.
    """
    operations, round_counts = _read_operations(operations_path)
    physical = None
    if fragments_path is not None:
        physical = read_fragments(fragments_path)
    if circuit_path is not None:
        physical = _read_finite_circuit(circuit_path, measurement_rounds_path)
    return workload_records.Workload(operations, round_counts, physical)


def write_workload(
    workload: workload_records.Workload, folder: pathlib.Path
) -> dict:
    """The workload as files in folder, which read_workload reads back.

    Returns the file row's keys for the files written: operations, and
    circuit with measurement_rounds, or fragments.
    """
    folder.mkdir(parents=True, exist_ok=True)
    round_counts = dict(workload.round_counts)
    entries = []
    for operation in workload.operations:
        entry = _operation_entry(operation, round_counts, folder)
        entries.append(entry)
    document = {"schema": OPERATIONS_SCHEMA, "operations": entries}
    operations_path = folder / "operations.json"
    _write_json(operations_path, document)
    written = {"operations": "operations.json"}
    physical = workload.physical
    if isinstance(physical, workload_records.FiniteCircuit):
        finite_keys = _write_finite_circuit(physical, folder)
        written.update(finite_keys)
    elif physical is not None:
        fragment_keys = _write_fragments(physical, folder)
        written.update(fragment_keys)
    return written


def read_fragments(
    folder: pathlib.Path,
) -> circuit_records.RepeatedStimCircuit:
    """The four live fragments and the period physical.json declares."""
    fragments = {}
    for name in FRAGMENT_NAMES:
        fragment_path = folder / f"{name}.stim"
        fragments[name] = stim.Circuit.from_file(fragment_path)
    physical_path = folder / PHYSICAL_FILE_NAME
    physical = _read_json(physical_path)
    return circuit_records.RepeatedStimCircuit(**fragments, **physical)


def _read_operations(path: pathlib.Path) -> tuple:
    """The operations in program order, and the rounds those that name them."""
    document = _read_json(path)
    entries = _operation_entries(document, path)
    operations = []
    round_counts = {}
    for entry in entries:
        operation, rounds = _operation_from_entry(entry, path)
        operations.append(operation)
        if rounds is not None:
            round_counts[operation.id] = rounds
    return tuple(operations), round_counts


def _operation_entries(document, path: pathlib.Path) -> list:
    """The file's operation list, once its schema is decsim.ops/1.

    Another schema's fields could read as this one's and run wrong.
    """
    if document.get("schema") != OPERATIONS_SCHEMA:
        raise ValueError(f"{path} is not a {OPERATIONS_SCHEMA} operation list")
    return document["operations"]


def _operation_from_entry(entry, path: pathlib.Path) -> tuple:
    """One entry as an Operation, and the rounds it names or None."""
    values = dict(entry)
    rounds = values.pop("rounds", None)
    patches = tuple(values["patches"])
    values["patches"] = patches
    qubits = values.get("qubits", patches)
    values["qubits"] = tuple(qubits)
    predecessors = values.get("predecessors", ())
    values["predecessors"] = tuple(predecessors)
    values.setdefault("name", f"operation {values['id']}")
    if "kind" in values:
        values["kind"] = program_records.OpKind[values["kind"]]
    if "circuit" in values:
        circuit_path = path.parent / values["circuit"]
        values["circuit"] = stim.Circuit.from_file(circuit_path)
    operation = program_records.Operation(**values)
    return operation, rounds


def _read_finite_circuit(
    circuit_path: pathlib.Path, measurement_rounds_path: pathlib.Path
) -> workload_records.FiniteCircuit:
    """The circuit, and its json of each measurement index's round."""
    circuit = stim.Circuit.from_file(circuit_path)
    document = _read_json(measurement_rounds_path)
    measurement_rounds = {}
    for index_text, round_index in document.items():
        measurement_rounds[int(index_text)] = round_index
    return workload_records.FiniteCircuit(circuit, measurement_rounds)


def _read_json(path: pathlib.Path):
    text = path.read_text()
    return json.loads(text)


def _operation_entry(
    operation: program_records.Operation, round_counts: Mapping, folder
) -> dict:
    """The operation's decsim.ops/1 entry, every key written out."""
    _refuse_fields_without_a_file_form(operation)
    entry = {
        "id": operation.id,
        "patches": list(operation.patches),
        "qubits": list(operation.qubits),
        "name": operation.name,
        "kind": operation.kind.name,
        "predecessors": list(operation.predecessors),
        "blocked_by": operation.blocked_by,
        "stream_id": operation.stream_id,
        "stream_offset": operation.stream_offset,
        "emits_detector_data": operation.emits_detector_data,
        "scheduled_start_round": operation.scheduled_start_round,
        "clifford": operation.clifford,
    }
    if operation.id in round_counts:
        entry["rounds"] = round_counts[operation.id]
    if operation.circuit is not None:
        circuit_name = f"operation_{operation.id}.stim"
        circuit_path = folder / circuit_name
        operation.circuit.to_file(str(circuit_path))
        entry["circuit"] = circuit_name
    return entry


def _refuse_fields_without_a_file_form(
    operation: program_records.Operation,
) -> None:
    """A field decsim.ops/1 does not carry is refused unless at its default."""
    unwritable = []
    for field in dataclasses.fields(operation):
        if field.name in _FILE_FIELDS:
            continue
        if getattr(operation, field.name) != field.default:
            unwritable.append(field.name)
    if unwritable:
        raise ValueError(
            f"operation {operation.id} sets {unwritable}, which "
            f"{OPERATIONS_SCHEMA} does not carry"
        )


def _write_finite_circuit(
    physical: workload_records.FiniteCircuit, folder: pathlib.Path
) -> dict:
    circuit_path = folder / "circuit.stim"
    physical.circuit.to_file(str(circuit_path))
    schedule = {}
    for index, round_index in physical.measurement_rounds:
        schedule[str(index)] = round_index
    schedule_path = folder / "measurement_rounds.json"
    _write_json(schedule_path, schedule)
    return {
        "circuit": "circuit.stim",
        "measurement_rounds": "measurement_rounds.json",
    }


def _write_fragments(
    program: circuit_records.RepeatedStimCircuit, folder: pathlib.Path
) -> dict:
    fragments_folder = folder / "fragments"
    fragments_folder.mkdir(exist_ok=True)
    for name in FRAGMENT_NAMES:
        fragment = getattr(program, name)
        fragment_path = fragments_folder / f"{name}.stim"
        fragment.to_file(str(fragment_path))
    physical = {"round_period_microseconds": program.round_period_microseconds}
    physical_path = fragments_folder / PHYSICAL_FILE_NAME
    _write_json(physical_path, physical)
    return {"fragments": "fragments"}


def _write_json(path: pathlib.Path, value) -> None:
    text = json.dumps(value, indent=2)
    file_text = text + "\n"
    path.write_text(file_text)
