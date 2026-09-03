"""Lower a QLX schedule into decsim operations and decode streams.

The input is the per-operation schedule DAG that qlx.estimate.schedule()
exposes through SpaceTimeDiagram.entries: op ids, fabric op names,
dependencies, round durations, occupied (region, slot) cells, resource
produces and consumes, protocols. The output is a QLXProgram: decsim
Operations, a PerOperationRounds map, the protected allocation
generations as dynamic streams and, given a whole-program Stim circuit
with its detector metadata, one single-patch physical stream. Payloads
model detector events and wire sizes, not QLX's decoder ABI encoding.

Accepted input forms: a live SpaceTimeDiagram (objects with .entries),
its as_dict() form, or the frozen reflection capture under
tests/data/qlx, where every value is a Python repr string.

Mapping rules (tests/09_qlx_workloads asserts them):
  * entry order is kept; Operation.id is the position; the QLX op_id
    string is kept in QLXProgram.op_ids.
  * dependencies become workload-only Operation.predecessors.
  * duration becomes PerOperationRounds, zero-duration tasks included.
  * occupied cells become patches; a transport with no cell claims none.
  * fabric.mz, fabric.mx and fabric.measure* are OpKind.MEASURE,
    fabric.inject is OpKind.INJECT, fabric.merge* is OpKind.MERGE, the
    rest GENERIC.
  * a resource chain has one producer, zero or more transports and one
    non-Clifford inject; QLX owns that resource, so the inject sets
    consumes_magic_state=False rather than asking a second factory.
  * feedback is never wired by itself: operations depending on a
    measurement are listed in feedback_candidates and the caller decides
    which become blocked_by, since the schedule does not say which
    dependencies are classically conditioned.
"""

import ast
import dataclasses
import json
import types
from typing import Any, Optional

import decsim.message as message
import decsim.qpu.round_policies as round_policies

_KIND_BY_NAME = types.MappingProxyType(
    {
        "mz": message.OpKind.MEASURE,
        "mx": message.OpKind.MEASURE,
        "measure": message.OpKind.MEASURE,
        "inject": message.OpKind.INJECT,
        "merge": message.OpKind.MERGE,
        "measure_product": message.OpKind.MERGE,
    }
)

# Op names that produce a classical bit usable for measurement feedback.
# measure_product is a bit producer although its round-count kind is
# MERGE, so feedback keys off names, not OpKind.MEASURE.
_BIT_PRODUCER_NAMES = ("mz", "mx", "measure", "measure_product")

_GENERATION_END_NAMES = ("mz", "mx", "dealloc")

_TASK_NAMES = {
    "h",
    "s",
    "sdg",
    "x",
    "z",
    "t",
    "tdg",
    "reset",
    "cx",
    "cz",
    "prep_z",
    "prep_x",
    "mz",
    "alloc",
    "dealloc",
    "measure_syndrome",
    "transversal_cx",
    "merge",
    "split",
    "multi_measure",
    "measure_product",
    "rotate_product",
    "move",
    "idle",
    "barrier",
    "if",
    "produce_resource",
    "inject",
    "transport",
    "discard_resource",
    "tick",
}


@dataclasses.dataclass
class QLXProgram:
    """Structural decsim view of one QLX schedule."""

    operations: list
    rounds: round_policies.PerOperationRounds
    op_ids: dict  # decsim id -> original QLX op_id string
    patch_of_cell: dict  # (region, slot) -> patch int
    raw_durations: dict  # decsim id -> QLX duration (may be 0)
    resource_flows: list  # (resource, producer_id, ..., consumer_id)
    feedback_candidates: list = dataclasses.field(default_factory=list)
    protocols: dict = dataclasses.field(default_factory=dict)
    start_rounds: dict = dataclasses.field(default_factory=dict)
    # From an optional RealtimeArtifactTarget JSON, the authoritative
    # feedback source; None means no artifact was given.
    conditional_feedback: Optional[bool] = None
    decoder_latency_rounds: Optional[int] = None
    idle_rounds_for_decoder_wait: Optional[int] = None
    runtime_ops: list = dataclasses.field(default_factory=list)
    streams: tuple = ()  # protected allocation generations
    dynamic_streams: tuple = ()  # syndrome owners for those generations
    decoder_operations: tuple = ()
    detector_rounds_by_stream: dict = dataclasses.field(default_factory=dict)
    terminal_detector_ids_by_stream: dict = dataclasses.field(
        default_factory=dict
    )
    measurement_rounds_by_stream: dict = dataclasses.field(default_factory=dict)

    def build(self) -> list[message.Operation]:
        """The lowered workload, for the Workload port."""
        return self.operations

    def rounds_for(self, operation, code) -> int:
        """The exact QLX duration, for the rounds-policy port."""
        return self.rounds.rounds_for(operation, code)


def qlx_frontend(
    diagram: Any,
    *,
    feedback_from_measurements: bool = False,
    realtime_artifact: Any = None,
    physical_circuit: Any = None,
    detector_metadata: Any = None,
    decode_operation_id: Optional[int] = None,
) -> QLXProgram:
    """Build a structural decsim program from a QLX SpaceTimeDiagram.

    ``diagram`` may be the live object, its ``as_dict()`` form, or the raw
    entry list. With ``feedback_from_measurements=True``, every op whose
    dependencies include a bit producer gets ``blocked_by`` that producer
    (opt-in; see the module docstring).

    ``realtime_artifact`` (optional): a ``RealtimeArtifactTarget`` JSON
    string or parsed dict for the same program. It is the authoritative
    feedback source: when it reports ``conditional_feedback: false``,
    asking for ``feedback_from_measurements=True`` raises instead of
    fabricating feedback the program does not have. It also supplies
    decoder latency (rounds), the pre-feedback idle budget, runtime ops,
    and patch-local streams on the returned ``QLXProgram``. QLX does not
    export which op is classically conditioned, so per-op wiring stays
    the measurement heuristic even when the artifact confirms feedback
    exists; callers review ``feedback_candidates``.
    """
    artifact = _loaded_artifact(realtime_artifact)
    artifact_feedback = _artifact_feedback(artifact, feedback_from_measurements)
    if artifact is not None and diagram is None:
        diagram = artifact.get("schedule")
    entries = _entries_of(diagram)
    _check_physical_arguments(
        physical_circuit, detector_metadata, decode_operation_id
    )
    tasks, patch_of_cell = _parse_tasks(entries)
    generations = _link_cells(tasks, patch_of_cell)
    for task in tasks:
        _check_task_start(task, tasks)
    flows = _resource_flows(tasks)
    program = _lower(
        tasks, patch_of_cell, flows, generations, feedback_from_measurements
    )
    if physical_circuit is not None:
        _add_physical_stream(
            program, physical_circuit, detector_metadata, decode_operation_id
        )
    if artifact is not None:
        _attach_artifact(program, artifact, artifact_feedback)
    return program


@dataclasses.dataclass
class _Task:
    """One schedule entry, decoded."""

    position: int
    qlx_id: Any
    name: str
    dependencies: tuple
    cells: tuple
    duration: int
    consumes: Any
    produces: Any
    protocol: Any
    start_round: int


@dataclasses.dataclass
class _RoutingTables:
    """What the routing of every detector fills in."""

    detector_rounds: dict
    ordinary_ids_by_round: dict
    terminal_ids: list


@dataclasses.dataclass(frozen=True)
class _Routing:
    """What every detector location is checked against."""

    submission_count: int
    check_count: int
    x_check_count: int
    z_matrix: list
    x_matrix: list
    terminal_data_count: int
    records: list


def _loaded_artifact(realtime_artifact) -> Optional[dict]:
    """The artifact as a dict; a string is JSON."""
    artifact = realtime_artifact
    if isinstance(artifact, str):
        artifact = json.loads(artifact)
    if artifact is None:
        return None
    kind = artifact.get("kind")
    if kind != "qlx.realtime_artifact":
        raise ValueError(
            "realtime_artifact must be a RealtimeArtifactTarget emission "
            f"(kind='qlx.realtime_artifact'); got kind={kind!r}"
        )
    return artifact


def _artifact_feedback(
    artifact: Optional[dict], feedback_from_measurements: bool
) -> Optional[bool]:
    """Whether the artifact declares conditional feedback; None without one."""
    if artifact is None:
        return None
    execution_model = artifact.get("execution_model", {})
    declared = execution_model.get("conditional_feedback")
    has_feedback = bool(declared)
    if feedback_from_measurements and not has_feedback:
        raise ValueError(
            "feedback_from_measurements=True, but the realtime artifact "
            "says this program has NO conditional feedback "
            "(execution_model.conditional_feedback=false); refusing to "
            "fabricate feedback wiring"
        )
    return has_feedback


def _entries_of(diagram):
    if isinstance(diagram, dict):
        return diagram["entries"]
    if hasattr(diagram, "entries"):
        return diagram.entries
    return diagram


def _check_physical_arguments(
    physical_circuit, detector_metadata, decode_operation_id
) -> None:
    values = (physical_circuit, detector_metadata, decode_operation_id)
    given = []
    for value in values:
        is_given = value is not None
        given.append(is_given)
    if any(given) and not all(given):
        raise ValueError(
            "physical_circuit, detector_metadata, and decode_operation_id "
            "must be supplied together"
        )


def _parse_tasks(entries) -> tuple[list, dict]:
    """Every entry as a _Task, and the patch each occupied cell became."""
    position_by_qlx_id = {}
    for position, entry in enumerate(entries):
        source_id = _value(entry, "op_id")
        if source_id in position_by_qlx_id:
            raise ValueError("QLX source operation ids must be unique")
        position_by_qlx_id[source_id] = position
    patch_of_cell: dict = {}
    tasks = []
    for position, entry in enumerate(entries):
        task = _parse_task(position, entry, position_by_qlx_id, patch_of_cell)
        tasks.append(task)
    return tasks, patch_of_cell


def _parse_task(
    position: int, entry, position_by_qlx_id: dict, patch_of_cell: dict
) -> _Task:
    qlx_id = _value(entry, "op_id")
    cells = _cells_of(entry)
    for cell in cells:
        patch_of_cell.setdefault(cell, len(patch_of_cell))
    dependencies = []
    for dependency_id in _value(entry, "dependencies") or ():
        dependencies.append(position_by_qlx_id[dependency_id])
    full_name = _value(entry, "op_name") or ""
    name = _short_name(full_name)
    if name == "repeat":
        raise ValueError("fabric.repeat must be expanded before lowering")
    if name not in _TASK_NAMES:
        raise ValueError(f"unsupported QLX task fabric.{name}")
    raw_duration = _value(entry, "duration")
    duration = _nonnegative_int(raw_duration, "duration")
    consumes = _value(entry, "consumes")
    produces = _value(entry, "produces")
    protocol = _value(entry, "protocol")
    raw_start_round = _value(entry, "start_round")
    start_round = _nonnegative_int(raw_start_round, "start_round")
    return _Task(
        position=position,
        qlx_id=qlx_id,
        name=name,
        dependencies=tuple(dependencies),
        cells=cells,
        duration=duration,
        consumes=consumes,
        produces=produces,
        protocol=protocol,
        start_round=start_round,
    )


def _cells_of(entry) -> tuple:
    cells = []
    for cell in _value(entry, "occupied_slots") or ():
        cells.append(tuple(cell))
    return tuple(cells)


def _value(entry: Any, name: str):
    """A field of an entry object or dict, decoding repr-string captures."""
    if isinstance(entry, dict):
        raw = entry.get(name)
    else:
        raw = getattr(entry, name)
    if not isinstance(raw, str):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _short_name(op_name: str) -> str:
    """The name after the first dot: fabric.alloc is alloc."""
    _prefix, dot, rest = op_name.partition(".")
    if not dot:
        return op_name
    return rest


def _nonnegative_int(value, field_name: str) -> int:
    value_type = type(value)
    if value_type is not int:
        raise ValueError(f"QLX {field_name} must be an exact int")
    if value < 0:
        raise ValueError(f"QLX {field_name} must be nonnegative")
    return value


def _link_cells(tasks: list, patch_of_cell: dict) -> list:
    """Order each cell's tasks, add the cell-order edges, find generations.

    A generation is one allocation of a cell: from its alloc to the
    measurement or dealloc that ends it.
    """
    tasks_by_cell = {}
    for task in tasks:
        for cell in task.cells:
            cell_tasks = tasks_by_cell.setdefault(cell, [])
            cell_tasks.append(task)
    generations = []
    for cell, cell_tasks in tasks_by_cell.items():
        ordered = sorted(cell_tasks, key=_cell_order_key)
        _add_cell_order_edges(ordered)
        cell_generations = _generations_of(patch_of_cell[cell], ordered)
        generations.extend(cell_generations)
    return generations


def _cell_order_key(task: _Task) -> tuple:
    return (task.start_round, task.position)


def _add_cell_order_edges(ordered: list) -> None:
    """Consecutive tasks on one cell depend on each other, in cell order."""
    for prior, current in zip(ordered, ordered[1:]):
        prior_end = prior.start_round + prior.duration
        both_run = prior.duration > 0 and current.duration > 0
        if both_run and current.start_round < prior_end:
            raise ValueError("positive-duration QLX tasks overlap one cell")
        if prior.position not in current.dependencies:
            current.dependencies = current.dependencies + (prior.position,)


def _generations_of(patch, ordered: list) -> list:
    generations = []
    start = None
    for task in ordered:
        start, generation = _advance_generation(patch, start, task)
        if generation is not None:
            generations.append(generation)
    return generations


def _advance_generation(patch, start: Optional[int], task: _Task) -> tuple:
    """The open allocation after the task, and the generation it closed."""
    if task.name == "alloc":
        if start is not None:
            raise ValueError("QLX cell allocated twice")
        return task.position, None
    if start is not None and task.name in _GENERATION_END_NAMES:
        return None, (patch, start, task.position)
    return start, None


def _check_task_start(task: _Task, tasks: list) -> None:
    """A task starts no earlier than every predecessor ends."""
    for predecessor_id in task.dependencies:
        predecessor = tasks[predecessor_id]
        predecessor_end = predecessor.start_round + predecessor.duration
        if predecessor_end > task.start_round:
            raise ValueError("QLX task starts before its predecessor finishes")


def _resource_flows(tasks: list) -> list:
    """Every resource chain: (resource, producer id, ..., inject id)."""
    dependents = {}
    for task in tasks:
        dependents[task.position] = []
    for task in tasks:
        for dependency_id in task.dependencies:
            dependents[dependency_id].append(task.position)
    flows = []
    consumes_at = {}
    for task in tasks:
        if task.produces is None:
            continue
        flow = _follow_chain(task, tasks, dependents, consumes_at)
        flows.append(flow)
    _check_transports(tasks, flows, consumes_at)
    return flows


def _follow_chain(
    producer: _Task, tasks: list, dependents: dict, consumes_at: dict
) -> tuple:
    """The producer's chain of transports to its one inject."""
    chain = [producer.position]
    head = producer.position
    while True:
        transports = _dependents_named(head, "transport", tasks, dependents)
        if len(transports) > 1:
            raise ValueError("QLX resource transport chain forks")
        if not transports:
            break
        head = transports[0]
        if head in chain:
            raise ValueError("QLX resource transport chain cycles")
        chain.append(head)
    sinks = _dependents_named(head, "inject", tasks, dependents)
    if len(sinks) != 1:
        raise ValueError("QLX resource must terminate at one inject")
    sink = sinks[0]
    if sink in consumes_at:
        raise ValueError("QLX resource chains join at one inject")
    declared_kind = tasks[sink].consumes
    if declared_kind is not None and declared_kind != producer.produces:
        raise ValueError("QLX produced and consumed resource kinds differ")
    chain.append(sink)
    consumes_at[sink] = producer.produces
    qlx_ids = []
    for position in chain:
        qlx_ids.append(tasks[position].qlx_id)
    return (producer.produces, *qlx_ids)


def _dependents_named(head: int, name: str, tasks: list, dependents) -> list:
    named = []
    for position in dependents[head]:
        if tasks[position].name == name:
            named.append(position)
    return named


def _check_transports(tasks: list, flows: list, consumes_at: dict) -> None:
    """Every transport is on a chain and every inject ends one."""
    chained_transports = set()
    for flow in flows:
        transports = flow[2:-1]
        chained_transports.update(transports)
    for task in tasks:
        is_transport = task.name == "transport"
        if is_transport and task.qlx_id not in chained_transports:
            raise ValueError("orphan QLX resource transport")
        is_inject = task.name == "inject"
        if is_inject and task.position not in consumes_at:
            raise ValueError("QLX inject has no unique resource chain")


def _lower(
    tasks: list,
    patch_of_cell: dict,
    flows: list,
    generations: list,
    feedback_from_measurements: bool,
) -> QLXProgram:
    """The operations, streams and rounds of the parsed tasks."""
    has_explicit_if = False
    for task in tasks:
        if task.name == "if":
            has_explicit_if = True
    operations = []
    raw_durations = {}
    protocols = {}
    start_rounds = {}
    op_ids = {}
    feedback_candidates = []
    for task in tasks:
        bit_producers = _bit_producer_dependencies(task, tasks)
        candidate = _feedback_candidate(task, bit_producers, has_explicit_if)
        if candidate is not None:
            feedback_candidates.append(candidate)
        blocked_by = None
        if feedback_from_measurements:
            blocked_by = _blocker_of(task, bit_producers, has_explicit_if)
        operation = _operation_for(task, patch_of_cell, blocked_by)
        operations.append(operation)
        raw_durations[task.position] = task.duration
        if task.protocol is not None:
            protocols[task.position] = task.protocol
        start_rounds[task.position] = task.start_round
        op_ids[task.position] = task.qlx_id
    generations.sort()
    dynamic_streams = _dynamic_streams(len(tasks), generations)
    streams = _protected_regions(dynamic_streams, generations)
    rounds = round_policies.PerOperationRounds(dict(raw_durations))
    return QLXProgram(
        operations=operations,
        rounds=rounds,
        op_ids=op_ids,
        patch_of_cell=dict(patch_of_cell),
        raw_durations=raw_durations,
        resource_flows=flows,
        feedback_candidates=feedback_candidates,
        protocols=protocols,
        start_rounds=start_rounds,
        streams=streams,
        dynamic_streams=dynamic_streams,
    )


def _bit_producer_dependencies(task: _Task, tasks: list) -> list:
    producers = []
    for dependency_id in task.dependencies:
        if tasks[dependency_id].name in _BIT_PRODUCER_NAMES:
            producers.append(dependency_id)
    return producers


def _feedback_candidate(
    task: _Task, bit_producers: list, has_explicit_if: bool
) -> Optional[tuple]:
    """(task, producer) when the task may be classically conditioned.

    An explicit fabric.if entry is authoritative; without one, every
    task that depends on a bit producer is a candidate.
    """
    if not bit_producers:
        return None
    if has_explicit_if and task.name != "if":
        return None
    return (task.position, bit_producers[0])


def _blocker_of(
    task: _Task, bit_producers: list, has_explicit_if: bool
) -> Optional[int]:
    if not bit_producers:
        return None
    if has_explicit_if and task.name != "if":
        return None
    return bit_producers[0]


def _operation_for(
    task: _Task, patch_of_cell: dict, blocked_by: Optional[int]
) -> message.Operation:
    patches = []
    for cell in task.cells:
        patches.append(patch_of_cell[cell])
    patches = tuple(patches)
    kind = _KIND_BY_NAME.get(task.name, message.OpKind.GENERIC)
    is_clifford = task.name not in ("t", "tdg", "inject")
    return message.Operation(
        id=task.position,
        name=f"{task.name}[{task.qlx_id}]",
        qubits=patches,
        patches=patches,
        clifford=is_clifford,
        consumes_magic_state=False,
        predecessors=task.dependencies,
        decoder_boundary_predecessors=(),
        scheduled_start_round=task.start_round,
        emits_detector_data=False,
        blocked_by=blocked_by,
        kind=kind,
    )


def _dynamic_streams(task_count: int, generations: list) -> tuple:
    """One syndrome owner per protected allocation generation."""
    streams = []
    for index, (patch, _start, _end) in enumerate(generations):
        owner_id = task_count + index
        owner = message.Operation(
            owner_id,
            f"protected[{patch}]",
            (patch,),
            patches=(patch,),
        )
        streams.append(owner)
    return tuple(streams)


def _protected_regions(dynamic_streams: tuple, generations: list) -> tuple:
    regions = []
    for owner, (patch, start, end) in zip(dynamic_streams, generations):
        region = message.ProtectedRegion(patch, owner.id, start, end)
        regions.append(region)
    return tuple(regions)


def _attach_artifact(
    program: QLXProgram, artifact: dict, artifact_feedback: Optional[bool]
) -> None:
    timing = artifact.get("timing", {})
    idle = artifact.get("idle_policy", {})
    program.conditional_feedback = artifact_feedback
    program.decoder_latency_rounds = timing.get("decoder_latency_rounds")
    program.idle_rounds_for_decoder_wait = idle.get(
        "idle_rounds_for_decoder_wait"
    )
    runtime_ops = artifact.get("runtime_ops", [])
    program.runtime_ops = list(runtime_ops)


def _add_physical_stream(
    program: QLXProgram, circuit, metadata, decode_operation_id
) -> None:
    """Attach the reviewed single-patch physical stream to a QLX workload."""
    _check_physical_inputs(program, circuit, metadata, decode_operation_id)
    by_name = _operations_by_short_name(program)
    measurements = by_name.get("measure_syndrome", [])
    _check_measurement_tasks(program, measurements)
    patch = _one_measured_patch(measurements)
    routing = _prove_detector_routing(circuit, metadata, len(measurements))
    detector_rounds, terminal_ids, measurement_rounds = routing
    terminal_operation = None
    if terminal_ids:
        candidates = by_name.get("mz", [])
        terminal_operation = _terminal_operation(
            program, candidates, measurements, patch
        )
    _replace_stream_operations(
        program, measurements, terminal_operation, circuit, decode_operation_id
    )
    owner = message.Operation(
        decode_operation_id,
        "QLX physical decode stream",
        (patch,),
        circuit=circuit,
        patches=(patch,),
    )
    program.decoder_operations = (owner,)
    program.rounds.rounds_by_operation[decode_operation_id] = len(measurements)
    program.detector_rounds_by_stream = {decode_operation_id: detector_rounds}
    program.terminal_detector_ids_by_stream = {
        decode_operation_id: terminal_ids
    }
    program.measurement_rounds_by_stream = {
        decode_operation_id: measurement_rounds
    }


def _check_physical_inputs(
    program: QLXProgram, circuit, metadata, decode_operation_id
) -> None:
    import stim

    circuit_type = type(circuit)
    if circuit_type is not stim.Circuit:
        raise ValueError("physical_circuit must be an exact stim.Circuit")
    metadata_type = type(metadata)
    if metadata_type is not dict:
        raise ValueError("detector_metadata must be an exact dict")
    id_type = type(decode_operation_id)
    if id_type is not int:
        raise ValueError("decode_operation_id must be an exact int")
    for operation in program.operations:
        if operation.id == decode_operation_id:
            raise ValueError(
                "decode_operation_id collides with a schedule task"
            )


def _operations_by_short_name(program: QLXProgram) -> dict:
    by_name = {}
    for operation in program.operations:
        short_name, _, _qlx_id = operation.name.partition("[")
        named = by_name.setdefault(short_name, [])
        named.append(operation)
    return by_name


def _check_measurement_tasks(program: QLXProgram, measurements: list) -> None:
    """Syndrome measurements are unit-duration and in submission order."""
    if not measurements:
        raise ValueError(
            "physical QLX requires unit-duration measure_syndrome tasks"
        )
    for operation in measurements:
        if program.raw_durations[operation.id] != 1:
            raise ValueError(
                "physical QLX requires unit-duration measure_syndrome tasks"
            )
    for earlier, later in zip(measurements, measurements[1:]):
        if earlier.scheduled_start_round >= later.scheduled_start_round:
            raise ValueError("measure_syndrome submissions must be ordered")


def _one_measured_patch(measurements: list):
    patch_sets = set()
    for operation in measurements:
        patch_sets.add(operation.patches)
    if len(patch_sets) != 1:
        raise ValueError("physical QLX currently requires one occupied patch")
    patches = next(iter(patch_sets))
    if len(patches) != 1:
        raise ValueError("physical QLX currently requires one occupied patch")
    return patches[0]


def _terminal_operation(
    program: QLXProgram, candidates: list, measurements: list, patch
) -> message.Operation:
    """The one zero-duration mz on the patch after the last submission."""
    last_measurement = measurements[-1]
    matching = []
    for operation in candidates:
        if operation.patches != (patch,):
            continue
        if program.raw_durations[operation.id] != 0:
            continue
        if last_measurement.id not in operation.predecessors:
            continue
        matching.append(operation)
    if len(matching) != 1:
        raise ValueError("terminal detectors require one final dependent mz")
    return matching[0]


def _replace_stream_operations(
    program: QLXProgram,
    measurements: list,
    terminal_operation: Optional[message.Operation],
    circuit,
    decode_operation_id: int,
) -> None:
    """Each submission becomes a stream round; the terminal mz a fragment."""
    replacements = {}
    last_measurement = measurements[-1]
    for submission, operation in enumerate(measurements):
        fields = {
            "circuit": circuit,
            "stream_id": decode_operation_id,
            "stream_offset": submission,
            "emits_detector_data": True,
        }
        is_last = operation is last_measurement
        if terminal_operation is not None and is_last:
            fields["syndrome_fragment_index"] = 0
            fields["syndrome_fragment_count"] = 2
        replacements[operation.id] = dataclasses.replace(operation, **fields)
    if terminal_operation is not None:
        last_offset = len(measurements) - 1
        replacements[terminal_operation.id] = dataclasses.replace(
            terminal_operation,
            circuit=circuit,
            stream_id=decode_operation_id,
            stream_offset=last_offset,
            emits_detector_data=True,
            finalizes_stream_round=True,
            syndrome_fragment_index=1,
            syndrome_fragment_count=2,
        )
    replaced = []
    for operation in program.operations:
        current = replacements.get(operation.id, operation)
        replaced.append(current)
    program.operations = replaced


def _prove_detector_routing(circuit, metadata, submission_count: int):
    """Check every detector against the circuit and route it to a round."""
    _check_detector_counts(circuit, metadata)
    z_matrix, x_matrix = _parity_matrices(metadata)
    records, measurement_count = _absolute_detector_records(circuit)
    check_count = len(x_matrix) + len(z_matrix)
    syndrome_measurement_count = submission_count * check_count
    if measurement_count < syndrome_measurement_count:
        raise ValueError("circuit has too few syndrome measurements")
    terminal_data_count = measurement_count - syndrome_measurement_count
    routing = _Routing(
        submission_count=submission_count,
        check_count=check_count,
        x_check_count=len(x_matrix),
        z_matrix=z_matrix,
        x_matrix=x_matrix,
        terminal_data_count=terminal_data_count,
        records=records,
    )
    tables = _RoutingTables({}, {}, [])
    locations = metadata["dem_detector_locs"]
    for detector_id, location in enumerate(locations):
        _route_detector(routing, tables, detector_id, location)
    _check_detector_order(
        circuit,
        submission_count,
        tables.ordinary_ids_by_round,
        tables.terminal_ids,
    )
    measurement_rounds = _measurement_rounds(
        measurement_count, check_count, submission_count
    )
    terminal_ids = tuple(tables.terminal_ids)
    return tables.detector_rounds, terminal_ids, measurement_rounds


def _route_detector(
    routing: _Routing, tables: "_RoutingTables", detector_id: int, location
) -> None:
    """Check one detector's location against its record; note its round."""
    _check_location_shape(location)
    submission, _bit_or_check, _prior_submission = location
    if submission >= 0:
        expected = _ordinary_record(routing, location)
        round_index = submission + 1
        tables.detector_rounds[detector_id] = round_index
        round_ids = tables.ordinary_ids_by_round.setdefault(round_index, [])
        round_ids.append(detector_id)
    elif submission in (-1, -2):
        expected = _terminal_record(routing, location)
        tables.terminal_ids.append(detector_id)
        tables.detector_rounds[detector_id] = routing.submission_count
    else:
        raise ValueError("unknown terminal detector location tag")
    _check_record(expected, routing.records[detector_id], detector_id)


def _check_detector_counts(circuit, metadata) -> None:
    if circuit.num_detectors != metadata["dem_num_detectors"]:
        raise ValueError("detector count disagrees with the supplied circuit")
    if circuit.num_observables != metadata["dem_num_observables"]:
        raise ValueError("observable count disagrees with the supplied circuit")
    locations = metadata["dem_detector_locs"]
    if len(locations) != circuit.num_detectors:
        raise ValueError("detector locations must cover every circuit detector")


def _parity_matrices(metadata) -> tuple[list, list]:
    """(dem_hz, dem_hx) as exact int list matrices; dem_num_sx checked."""
    x_check_count = _nonnegative_int(metadata["dem_num_sx"], "dem_num_sx")
    matrices = {}
    for matrix_name in ("dem_hx", "dem_hz"):
        matrix = metadata[matrix_name]
        _check_int_matrix(matrix, matrix_name)
        matrices[matrix_name] = matrix
    if x_check_count != len(matrices["dem_hx"]):
        raise ValueError("QLX dem_num_sx must equal len(dem_hx)")
    return matrices["dem_hz"], matrices["dem_hx"]


def _check_int_matrix(matrix, matrix_name: str) -> None:
    matrix_type = type(matrix)
    if matrix_type is not list:
        raise ValueError(f"QLX {matrix_name} must be an exact list matrix")
    for row in matrix:
        _check_int_row(row, matrix_name)


def _check_int_row(row, matrix_name: str) -> None:
    row_type = type(row)
    if row_type is not list:
        raise ValueError(f"QLX {matrix_name} must be an exact list matrix")
    for index in row:
        index_type = type(index)
        if index_type is not int:
            raise ValueError(f"QLX {matrix_name} entries must be exact ints")


def _absolute_detector_records(circuit) -> tuple[list[tuple[int, ...]], int]:
    """Each detector's parity in absolute measurement-record indices."""
    measurement_count = 0
    records = []
    for instruction in circuit.flattened():
        if instruction.name != "DETECTOR":
            measurement_count += instruction.num_measurements
            continue
        indices = []
        for target in instruction.targets_copy():
            absolute_index = measurement_count + target.value
            indices.append(absolute_index)
        ordered = sorted(indices)
        records.append(tuple(ordered))
    return records, measurement_count


def _check_location_shape(location) -> None:
    location_type = type(location)
    if location_type is not list or len(location) != 3:
        raise ValueError("each detector location must have three integers")
    for value in location:
        value_type = type(value)
        if value_type is not int:
            raise ValueError("detector location entries must be exact integers")


def _ordinary_record(routing: _Routing, location: list) -> list:
    """The measurement records an ordinary (in-round) detector compares."""
    submission, check_index, prior_submission = location
    if not 0 <= submission < routing.submission_count:
        raise ValueError("detector submission is out of range")
    if not 0 <= check_index < routing.check_count:
        raise ValueError("detector syndrome bit is out of range")
    expected = [submission * routing.check_count + check_index]
    if prior_submission >= 0:
        if not 0 <= prior_submission < submission:
            raise ValueError(
                "detector baseline must precede current submission"
            )
        baseline = prior_submission * routing.check_count + check_index
        expected.append(baseline)
    elif prior_submission != -1:
        raise ValueError("detector baseline must be -1 or a submission")
    return expected


def _terminal_record(routing: _Routing, location: list) -> list:
    """The records a terminal detector compares: a check and data qubits."""
    submission, check_index, prior_submission = location
    if prior_submission != routing.submission_count - 1:
        raise ValueError("terminal detector must reference final submission")
    matrix = routing.x_matrix
    syndrome_bit = check_index
    if submission == -1:
        matrix = routing.z_matrix
        syndrome_bit = routing.x_check_count + check_index
    if not 0 <= check_index < len(matrix):
        raise ValueError("terminal check index is out of range")
    syndrome_count = routing.submission_count * routing.check_count
    expected = [prior_submission * routing.check_count + syndrome_bit]
    for data_qubit in matrix[check_index]:
        _check_data_qubit(data_qubit, routing.terminal_data_count)
        data_record = syndrome_count + data_qubit
        expected.append(data_record)
    return expected


def _check_data_qubit(data_qubit, terminal_data_count: int) -> None:
    data_qubit_type = type(data_qubit)
    if data_qubit_type is not int:
        raise ValueError("terminal parity indices must be exact ints")
    if not 0 <= data_qubit < terminal_data_count:
        raise ValueError("terminal data-qubit index is out of range")


def _check_record(expected: list, record: tuple, detector_id: int) -> None:
    ordered = sorted(expected)
    if tuple(ordered) != record:
        raise ValueError(
            f"detector {detector_id} disagrees with measurement records"
        )


def _check_detector_order(
    circuit, submission_count: int, ordinary_ids_by_round: dict, terminal_ids
) -> None:
    """Round by round, then terminal, the detectors keep the circuit's order."""
    emitted = []
    stop = submission_count + 1
    for round_index in range(1, stop):
        round_ids = ordinary_ids_by_round.get(round_index, ())
        emitted.extend(round_ids)
    emitted.extend(terminal_ids)
    if emitted != list(range(circuit.num_detectors)):
        raise ValueError(
            "detector fragments do not preserve global detector order"
        )


def _measurement_rounds(
    measurement_count: int, check_count: int, submission_count: int
) -> dict:
    """The QPU's packet schedule.

    Each submission's check bits travel in its round; the data readouts
    fold into the last round.
    """
    rounds = {}
    for index in range(measurement_count):
        submission_round = index // check_count + 1
        rounds[index] = min(submission_round, submission_count)
    return rounds
