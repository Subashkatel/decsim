"""A maker's workload lowered into a program the machine runs.

Each operation's predecessors come from program order on its patches,
so two operations that share a patch always carry a dependency edge.
The stream's owner, its protected region and every operation's rounds
are derived, so a maker writes none of them.
"""

import dataclasses
from typing import Optional

import decsim.ports as ports
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import decsim.records.workload as workload_records


@dataclasses.dataclass(frozen=True)
class WorkloadProgram:
    """A maker's workload in the machine's terms.

    physical_circuits pairs the stream key that runs the workload's
    physical circuit (the stream's owner, or the one operation that runs
    it) with that circuit, for the syndrome source; a workload carries
    one physical circuit at most.
    """

    operations: tuple
    dynamic_streams: tuple
    protected_regions: tuple
    rounds_policy: Optional[ports.RoundsPolicy]
    physical_circuits: tuple


def lowered(workload: workload_records.Workload) -> WorkloadProgram:
    """The workload's operations wired, its stream and rounds derived."""
    operations = _wired(workload.operations)
    stream_id = _one_stream_id(operations)
    physical = workload.physical
    round_counts = dict(workload.round_counts)
    if stream_id is None:
        return _standalone_program(operations, round_counts, physical)
    return _stream_program(operations, round_counts, physical, stream_id)


def _wired(operations: tuple) -> list:
    """The operations with their patches and predecessors, in schedule order.

    An operation keeps the predecessors it declares and gains the last
    earlier user of each of its patches. A decoder boundary joins two
    operations that emit detector data, so only such an operation
    carries one.
    """
    _check_unique_qubits(operations)
    placed = _placed(operations)
    predecessors = _patch_order_predecessors(placed)
    emitters = _emitters(placed)
    decoded_ids = {operation.id for operation in emitters}
    wired = []
    for operation in placed:
        ordered = sorted(predecessors[operation.id])
        ordered_predecessors = tuple(ordered)
        boundary_predecessors = _boundary_predecessors(
            operation.id, ordered_predecessors, decoded_ids
        )
        wired_operation = dataclasses.replace(
            operation,
            predecessors=ordered_predecessors,
            decoder_boundary_predecessors=boundary_predecessors,
        )
        wired.append(wired_operation)
    return wired


def _placed(operations: tuple) -> list:
    """Each operation on its patches: its own, or its qubits."""
    placed = []
    for operation in operations:
        patches = _operation_patches(operation)
        placed_operation = dataclasses.replace(operation, patches=patches)
        placed.append(placed_operation)
    return placed


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
    physical_circuits = ()
    if physical is not None:
        runner = _the_one_runner(operations)
        runner_ids = {runner.id}
        operations = _carrying(operations, runner_ids, physical.circuit)
        physical_circuits = ((runner.id, physical),)
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
    circuit = _finite_circuit(physical)
    segment_ids = _segment_ids(operations, stream_id)
    operations = _carrying(operations, segment_ids, circuit)
    segments = _segments(operations, stream_id)
    owner = _owner(stream_id, segments, circuit)
    round_counts[stream_id] = _owner_round_count(physical)
    regions = _protected_regions(operations, segments, owner)
    physical_circuits = ()
    if physical is not None:
        physical_circuits = ((stream_id, physical),)
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


def _segment_ids(operations: list, stream_id) -> set:
    segments = _segments(operations, stream_id)
    return {segment.id for segment in segments}


def _carrying(operations: list, operation_ids: set, circuit) -> list:
    """The operations, each one named carrying the circuit."""
    carried = []
    for operation in operations:
        if operation.id not in operation_ids:
            carried.append(operation)
            continue
        carrier = dataclasses.replace(operation, circuit=circuit)
        carried.append(carrier)
    return carried


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
        schedule = dict(physical.measurement_rounds)
        rounds = schedule.values()
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
    count_items = round_counts.items()
    count_pairs = tuple(count_items)
    return round_policies.PerOperationRounds(count_pairs, fallback)


def _boundary_predecessors(
    operation_id, predecessors: tuple, decoded_ids: set
) -> tuple:
    if operation_id not in decoded_ids:
        return ()
    boundary_predecessors = []
    for predecessor_id in predecessors:
        if predecessor_id in decoded_ids:
            boundary_predecessors.append(predecessor_id)
    return tuple(boundary_predecessors)


def _check_unique_qubits(operations: tuple) -> None:
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
