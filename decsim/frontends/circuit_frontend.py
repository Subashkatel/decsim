"""A maker's workload: its file form, and its lowering into a program.

The file form is the workload on disk (read_workload, write_workload):
the operation list as a decsim.ops/1 json, and the physical circuit as
a finite .stim with its measurement-to-round json or a folder of the
four live fragments with physical.json. The lowering (lowered) wires
the operation list in program order: each operation's predecessors
come from program order on its patches, so two operations that share a
patch always carry a dependency edge between them. The stream a
physical circuit runs is derived from its segments, with its owner,
the protected region a waiting patch keeps measuring in, and the
rounds of every operation, so a maker writes none of them.
"""

import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Optional

import stim

import decsim.ports as ports
import decsim.qpu.round_policies as round_policies
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


@dataclasses.dataclass(frozen=True)
class WorkloadProgram:
    """A maker's workload in the machine's terms.

    physical_circuits maps the stream key that runs the workload's
    physical circuit (the stream's owner, or the one operation that runs
    it) to that circuit, for the syndrome source.
    """

    operations: tuple
    dynamic_streams: tuple
    protected_regions: tuple
    rounds_policy: Optional[ports.RoundsPolicy]
    physical_circuits: Mapping


def lowered(workload: workload_records.Workload) -> WorkloadProgram:
    """The workload's operations wired, its stream and rounds derived."""
    operations = _copied(workload.operations)
    _wire_circuit(operations)
    stream_id = _one_stream_id(operations)
    physical = workload.physical
    round_counts = dict(workload.round_counts)
    if stream_id is None:
        return _standalone_program(operations, round_counts, physical)
    return _stream_program(operations, round_counts, physical, stream_id)


def _wire_circuit(operations: list[program_records.Operation]) -> None:
    """Fill operation patches and predecessors in schedule order.

    An operation keeps the predecessors it declares and gains the last
    earlier user of each of its patches. A decoder boundary joins two
    operations that emit detector data, so only such an operation
    carries one.
    """
    _check_unique_qubits(operations)
    predecessors = _patch_order_predecessors(operations)
    emitters = _emitters(operations)
    decoded_ids = {operation.id for operation in emitters}
    for operation in operations:
        ordered = sorted(predecessors[operation.id])
        operation.predecessors = tuple(ordered)
        boundary_predecessors = _boundary_predecessors(operation, decoded_ids)
        operation.decoder_boundary_predecessors = boundary_predecessors


def _copied(operations) -> list:
    """Copies of the maker's operations, so the wiring leaves its own."""
    copies = []
    for operation in operations:
        copy = dataclasses.replace(operation)
        copies.append(copy)
    return copies


def _one_stream_id(operations: list) -> Optional[object]:
    """The stream the first segment names, or None.

    A workload has one physical circuit, so one stream; the machine
    refuses a segment of any other, whose stream has no owner.
    """
    for operation in operations:
        if operation.stream_id is not None:
            return operation.stream_id
    return None


def _standalone_program(
    operations: list, round_counts: dict, physical
) -> WorkloadProgram:
    """Operations with no stream: the physical circuit runs as one of them.

    That operation runs the whole circuit, so its rounds are the
    circuit's unless it names them.
    """
    physical_circuits = {}
    if physical is not None:
        runner = _the_one_runner(operations)
        runner.circuit = physical.circuit
        physical_circuits[runner.id] = physical
        circuit_round_count = _owner_round_count(physical)
        round_counts.setdefault(runner.id, circuit_round_count)
    rounds_policy = _rounds_policy(round_counts)
    return WorkloadProgram(
        operations=tuple(operations),
        dynamic_streams=(),
        protected_regions=(),
        rounds_policy=rounds_policy,
        physical_circuits=physical_circuits,
    )


def _stream_program(
    operations: list, round_counts: dict, physical, stream_id
) -> WorkloadProgram:
    """Segments of one stream: its owner, its region and its rounds derived.

    The owner holds every patch its segments do, runs the finite circuit
    for the circuit's rounds or stays open for live fragments, and the
    segments sample the owner's shot, so they carry its circuit too.
    """
    segments = _segments(operations, stream_id)
    circuit = _finite_circuit(physical)
    for segment in segments:
        segment.circuit = circuit
    owner = _owner(stream_id, segments, circuit)
    round_counts[stream_id] = _owner_round_count(physical)
    regions = _protected_regions(operations, segments, owner)
    physical_circuits = {}
    if physical is not None:
        physical_circuits[stream_id] = physical
    rounds_policy = _rounds_policy(round_counts)
    return WorkloadProgram(
        operations=tuple(operations),
        dynamic_streams=(owner,),
        protected_regions=regions,
        rounds_policy=rounds_policy,
        physical_circuits=physical_circuits,
    )


def _the_one_runner(operations: list):
    """The one operation that emits detector data, which runs the circuit.

    decsim never builds a merged circuit (docs/how-to/
    plug_in_a_workload_maker.md), so a circuit two operations share
    needs each one's round range.
    """
    emitters = _emitters(operations)
    if len(emitters) == 1:
        return emitters[0]
    raise ValueError(
        f"the workload's one circuit is the history of "
        f"{len(emitters)} operations that emit detector data, and "
        "none names its round range; give each a stream_id and a "
        "stream_offset into the circuit, since decsim builds no "
        "merged circuit"
    )


def _emitters(operations: list) -> list:
    emitters = []
    for operation in operations:
        if operation.emits_detector_data:
            emitters.append(operation)
    return emitters


def _segments(operations: list, stream_id) -> list:
    """The operations that run a slice of the stream, in program order."""
    segments = []
    for operation in operations:
        if operation.stream_id == stream_id:
            segments.append(operation)
    return segments


def _finite_circuit(physical):
    """The finite circuit the owner and its segments sample, or None."""
    if isinstance(physical, workload_records.FiniteCircuit):
        return physical.circuit
    return None


def _owner(stream_id, segments: list, circuit) -> program_records.Operation:
    """The stream's owner: every patch its segments hold, in first-use order."""
    patches = {}
    for segment in segments:
        segment_patches = dict.fromkeys(segment.patches)
        patches.update(segment_patches)
    return program_records.Operation(
        stream_id,
        f"stream {stream_id}",
        tuple(patches),
        patches=tuple(patches),
        circuit=circuit,
    )


def _owner_round_count(physical) -> int:
    """A finite circuit's rounds; zero leaves a live stream open-ended.

    The live source finishes a stream when its final readout executes
    (qpu/streaming_stim_device.py), so the owner declares no length.
    """
    if isinstance(physical, workload_records.FiniteCircuit):
        rounds = physical.measurement_rounds.values()
        return max(rounds)
    return 0


def _protected_regions(operations: list, segments: list, owner) -> tuple:
    """The region a waiting stream keeps measuring in, or none.

    A patch that waits keeps running rounds: "during this delay time
    ∆proc a new data record is generated" (Terhal 1302.3428, lines
    2697-2698 of the text), and "where our quantum system is idle,
    more syndromes are generated" (Holmes 2004.04794 line 451). So the
    operations that hold every patch of the stream after its last
    segment run inside one protected region, from the first of them to
    the last (controller/feedback_streams.py).
    """
    last_segment = segments[-1]
    last_index = operations.index(last_segment)
    after_the_stream = last_index + 1
    owner_patches = set(owner.patches)
    holders = []
    for operation in operations[after_the_stream:]:
        if owner_patches.issubset(operation.patches):
            holders.append(operation)
    if not holders:
        return ()
    first = holders[0]
    last = holders[-1]
    region = program_records.ProtectedRegion(owner.id, first.id, last.id)
    return (region,)


def _rounds_policy(round_counts: dict) -> Optional[ports.RoundsPolicy]:
    """The rounds each operation names; the rest take GateRounds'."""
    if not round_counts:
        return None
    fallback = round_policies.GateRounds()
    return round_policies.PerOperationRounds(round_counts, fallback)


def _boundary_predecessors(
    operation: program_records.Operation, decoded_ids: set
) -> tuple:
    if operation.id not in decoded_ids:
        return ()
    boundary_predecessors = []
    for predecessor_id in operation.predecessors:
        if predecessor_id in decoded_ids:
            boundary_predecessors.append(predecessor_id)
    return tuple(boundary_predecessors)


def _check_unique_qubits(operations: list[program_records.Operation]) -> None:
    """Refuse an operation that lists the same qubit twice."""
    for operation in operations:
        distinct = set(operation.qubits)
        if len(distinct) != len(operation.qubits):
            raise ValueError(
                f"{operation.name} lists the same qubit more than once: "
                f"{operation.qubits}"
            )


def _operation_patches(operation: program_records.Operation) -> tuple:
    """The patches an operation touches: its own, or its qubits."""
    if operation.patches:
        return operation.patches
    return tuple(operation.qubits)


def _patch_order_predecessors(
    operations: list[program_records.Operation],
) -> dict:
    """Each operation's predecessors: its own, and each patch's last user."""
    last_operation_on_patch = {}
    predecessors = {}
    for operation in operations:
        predecessors[operation.id] = set(operation.predecessors)
    for operation in operations:
        operation.patches = _operation_patches(operation)
        earlier_users = _claim_patches(operation, last_operation_on_patch)
        predecessors[operation.id].update(earlier_users)
    return predecessors


def _claim_patches(
    operation: program_records.Operation, last_operation_on_patch: dict
) -> set:
    """Mark the operation as each patch's last user; the users it displaces."""
    earlier_users = set()
    for patch in operation.patches:
        previous_id = last_operation_on_patch.get(patch)
        if previous_id is not None:
            earlier_users.add(previous_id)
        last_operation_on_patch[patch] = operation.id
    return earlier_users


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
        physical = _read_fragments(fragments_path)
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
    entries = []
    for operation in workload.operations:
        entry = _operation_entry(operation, workload.round_counts, folder)
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


def _read_fragments(folder: pathlib.Path):
    """The four live fragments and the period physical.json declares."""
    fragments = {}
    for name in FRAGMENT_NAMES:
        fragment_path = folder / f"{name}.stim"
        fragments[name] = stim.Circuit.from_file(fragment_path)
    physical_path = folder / PHYSICAL_FILE_NAME
    physical = _read_json(physical_path)
    return circuit_records.RepeatedStimCircuit(**fragments, **physical)


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
    for index, round_index in physical.measurement_rounds.items():
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
    if program.readout_partitions:
        raise ValueError(
            "the live fragments name readout_partitions, which "
            f"{PHYSICAL_FILE_NAME} does not carry"
        )
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
    lines = text + "\n"
    path.write_text(lines)
