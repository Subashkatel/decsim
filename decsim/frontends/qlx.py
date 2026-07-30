"""Lower a QLX schedule into exact decsim workload and decode streams.

Consumes the per-op schedule DAG that ``qlx.estimate.schedule()`` exposes via
``SpaceTimeDiagram.entries`` — op ids, fabric op names, dependencies, round
durations, occupied (region, slot) fabric cells, resource produces/consumes,
protocols — and emits decsim ``Operation``s plus a ``PerOpRounds`` map.

An optional whole-program Stim circuit and detector metadata attach one
single-patch physical stream. Payloads model detector events and wire sizes;
they do not claim byte-identical QLX decoder ABI encoding.

Input formats accepted:
  * a live ``SpaceTimeDiagram`` (objects with ``.entries`` attributes),
  * ``as_dict()``-style dicts, or
  * the reflection capture format used by
    ``docs/audits/qlx-output-sample/schedule_output.json``, where every
    value is a Python repr string ("'alloc_0'", "(('C0', 0),)", ...).

Mapping rules (each asserted by tests/test_qlx_frontend.py):
  * entry order is preserved; ``Operation.id`` = position; the original
    ``op_id`` string is kept in ``QLXProgram.op_ids``.
  * ``dependencies`` -> workload-only ``Operation.predecessors``.
  * ``duration`` -> ``PerOpRounds`` without changing zero-duration tasks.
  * occupied cells become patches; no-cell transports stay claim-free.
  * ``fabric.mz``/``fabric.measure*`` -> OpKind.MEASURE; ``fabric.inject``
    -> OpKind.INJECT; ``fabric.merge*`` -> OpKind.MERGE; others GENERIC.
  * a resource chain has one producer, zero or more transports, and one
    non-Clifford inject. QLX owns that resource, so the imported inject sets
    ``consumes_magic_state=False`` instead of requesting a second factory item.
  * feedback is NOT auto-wired: ops depending on a MEASURE op are listed in
    ``QLXProgram.feedback_candidates``; the caller decides which become
    ``blocked_by`` (decsim's feedback gating), because the schedule alone
    does not say which dependencies are classically conditioned.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ..message import Operation, OpKind
from ..planner import PerOpRounds

_KIND_BY_NAME = {"mz": OpKind.MEASURE, "mx": OpKind.MEASURE,
                 "measure": OpKind.MEASURE, "inject": OpKind.INJECT,
                 "merge": OpKind.MERGE, "measure_product": OpKind.MERGE}

# Op names that produce a classical bit usable for measurement feedback.
# measure_product is a bit producer even though its round-count KIND is MERGE,
# so the implicit-feedback heuristic must key off names, not OpKind.MEASURE.
_BIT_PRODUCER_NAMES = ("mz", "mx", "measure", "measure_product")

_TASK_NAMES = {
    "h", "s", "sdg", "x", "z", "t", "tdg", "reset", "cx", "cz",
    "prep_z", "prep_x", "mz", "alloc", "dealloc", "measure_syndrome",
    "transversal_cx", "merge", "split", "multi_measure",
    "measure_product", "rotate_product", "move", "idle", "barrier",
    "if", "produce_resource", "inject", "transport", "discard_resource",
    "tick",
}


@dataclass
class QLXProgram:
    """Structural decsim view of one QLX schedule."""

    operations: list
    rounds: PerOpRounds
    op_ids: dict                 # decsim id -> original QLX op_id string
    patch_of_cell: dict          # (region, slot) -> patch int
    raw_durations: dict          # decsim id -> QLX duration (may be 0)
    resource_flows: list         # (resource, producer_id, ..., consumer_id)
    feedback_candidates: list = field(default_factory=list)
    protocols: dict = field(default_factory=dict)
    start_rounds: dict = field(default_factory=dict)
    # From an optional RealtimeArtifactTarget JSON (the AUTHORITATIVE
    # feedback source — schedule dependency edges never imply feedback):
    conditional_feedback: Optional[bool] = None   # None = no artifact given
    decoder_latency_rounds: Optional[int] = None
    idle_rounds_for_decoder_wait: Optional[int] = None
    runtime_ops: list = field(default_factory=list)
    streams: list = field(default_factory=list)   # patch-local (region,slot)
    decoder_operations: tuple = ()
    detector_rounds_by_stream: dict = field(default_factory=dict)
    terminal_detector_ids_by_stream: dict = field(default_factory=dict)
    terminal_data_bits_by_stream: dict = field(default_factory=dict)

    def build(self) -> list[Operation]:
        """Return the lowered workload for the InputFrontend port."""
        return self.operations

    def rounds_for(self, op, code) -> int:
        """Return the exact QLX duration for the rounds-policy port."""
        return self.rounds.rounds_for(op, code)


def _value(entry: Any, name: str):
    """Read a field from an entry object/dict, decoding repr-string captures."""
    raw = entry.get(name) if isinstance(entry, dict) else getattr(entry, name)
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    return raw


def _short_name(op_name: str) -> str:
    return op_name.split(".", 1)[-1] if op_name else ""


def _nonnegative_int(value, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"QLX {field_name} must be an exact int")
    if value < 0:
        raise ValueError(f"QLX {field_name} must be nonnegative")
    return value


def _absolute_detector_records(circuit) -> tuple[list[tuple[int, ...]], int]:
    """Return each detector parity in absolute measurement-record indices."""
    measurement_count = 0
    records = []
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            records.append(tuple(sorted(
                measurement_count + target.value
                for target in instruction.targets_copy()
            )))
        else:
            measurement_count += instruction.num_measurements
    return records, measurement_count


def _prove_detector_routing(circuit, metadata, submission_count: int):
    locations = metadata["dem_detector_locs"]
    if circuit.num_detectors != metadata["dem_num_detectors"]:
        raise ValueError("detector count disagrees with the supplied circuit")
    if circuit.num_observables != metadata["dem_num_observables"]:
        raise ValueError("observable count disagrees with the supplied circuit")
    if len(locations) != circuit.num_detectors:
        raise ValueError("detector locations must cover every circuit detector")

    x_check_count = _nonnegative_int(
        metadata["dem_num_sx"], "dem_num_sx"
    )
    matrices = {}
    for matrix_name in ("dem_hx", "dem_hz"):
        matrix = metadata[matrix_name]
        if type(matrix) is not list or any(
            type(row) is not list for row in matrix
        ):
            raise TypeError(f"QLX {matrix_name} must be an exact list matrix")
        if any(
            type(index) is not int
            for row in matrix
            for index in row
        ):
            raise TypeError(f"QLX {matrix_name} entries must be exact ints")
        matrices[matrix_name] = matrix
    if x_check_count != len(matrices["dem_hx"]):
        raise ValueError("QLX dem_num_sx must equal len(dem_hx)")

    records, measurement_count = _absolute_detector_records(circuit)
    check_count = len(matrices["dem_hx"]) + len(matrices["dem_hz"])
    syndrome_measurement_count = submission_count * check_count
    if measurement_count < syndrome_measurement_count:
        raise ValueError("circuit has too few syndrome measurements")
    terminal_data_count = measurement_count - syndrome_measurement_count
    terminal_ids = []
    detector_rounds = {}
    ordinary_detector_ids_by_round = {}

    for detector_id, location in enumerate(locations):
        if type(location) is not list or len(location) != 3:
            raise ValueError("each detector location must have three integers")
        submission, bit_or_check, prior_submission = location
        if any(type(value) is not int for value in location):
            raise TypeError("detector location entries must be exact integers")
        if submission >= 0:
            if not 0 <= submission < submission_count:
                raise ValueError("detector submission is out of range")
            if not 0 <= bit_or_check < check_count:
                raise ValueError("detector syndrome bit is out of range")
            expected = [submission * check_count + bit_or_check]
            if prior_submission >= 0:
                if not 0 <= prior_submission < submission:
                    raise ValueError(
                        "detector baseline must precede current submission"
                    )
                expected.append(
                    prior_submission * check_count + bit_or_check
                )
            elif prior_submission != -1:
                raise ValueError("detector baseline must be -1 or a submission")
            detector_rounds[detector_id] = submission + 1
            ordinary_detector_ids_by_round.setdefault(
                submission + 1, []
            ).append(detector_id)
        elif submission in (-1, -2):
            if prior_submission != submission_count - 1:
                raise ValueError("terminal detector must reference final submission")
            matrix_name = "dem_hz" if submission == -1 else "dem_hx"
            matrix = matrices[matrix_name]
            if not 0 <= bit_or_check < len(matrix):
                raise ValueError("terminal check index is out of range")
            syndrome_bit = (
                x_check_count + bit_or_check
                if submission == -1 else bit_or_check
            )
            expected = [prior_submission * check_count + syndrome_bit]
            for data_qubit in matrix[bit_or_check]:
                if type(data_qubit) is not int:
                    raise TypeError("terminal parity indices must be exact ints")
                if not 0 <= data_qubit < terminal_data_count:
                    raise ValueError("terminal data-qubit index is out of range")
                expected.append(syndrome_measurement_count + data_qubit)
            terminal_ids.append(detector_id)
            detector_rounds[detector_id] = submission_count
        else:
            raise ValueError("unknown terminal detector location tag")
        if tuple(sorted(expected)) != records[detector_id]:
            raise ValueError(
                f"detector {detector_id} disagrees with measurement records"
            )
    emitted_detector_ids = tuple(
        detector_id
        for round_index in range(1, submission_count + 1)
        for detector_id in ordinary_detector_ids_by_round.get(round_index, ())
    ) + tuple(terminal_ids)
    if emitted_detector_ids != tuple(range(circuit.num_detectors)):
        raise ValueError(
            "detector fragments do not preserve global detector order"
        )
    return detector_rounds, tuple(terminal_ids), terminal_data_count


def _add_physical_stream(program, circuit, metadata, decode_operation_id):
    """Attach the reviewed single-patch physical stream to a QLX workload."""
    import stim

    if type(circuit) is not stim.Circuit:
        raise TypeError("physical_circuit must be an exact stim.Circuit")
    if type(metadata) is not dict:
        raise TypeError("detector_metadata must be an exact dict")
    if type(decode_operation_id) is not int:
        raise TypeError("decode_operation_id must be an exact int")
    if any(
        operation.id == decode_operation_id
        for operation in program.operations
    ):
        raise ValueError("decode_operation_id collides with a schedule task")

    by_name = {}
    for operation in program.operations:
        short_name = operation.name.split("[", 1)[0]
        by_name.setdefault(short_name, []).append(operation)
    measurements = by_name.get("measure_syndrome", [])
    if not measurements or any(
        program.raw_durations[op.id] != 1 for op in measurements
    ):
        raise ValueError(
            "physical QLX requires unit-duration measure_syndrome tasks"
        )
    if any(
        measurements[index].scheduled_start_round
        >= measurements[index + 1].scheduled_start_round
        for index in range(len(measurements) - 1)
    ):
        raise ValueError("measure_syndrome submissions must be ordered")
    patch_sets = {op.patches for op in measurements}
    if len(patch_sets) != 1 or len(next(iter(patch_sets))) != 1:
        raise ValueError("physical QLX currently requires one occupied patch")
    patch = next(iter(patch_sets))[0]

    detector_rounds, terminal_ids, terminal_data_count = (
        _prove_detector_routing(circuit, metadata, len(measurements))
    )
    terminal_operation = None
    if terminal_ids:
        candidates = [
            op for op in by_name.get("mz", [])
            if op.patches == (patch,)
            and program.raw_durations[op.id] == 0
            and measurements[-1].id in op.predecessors
        ]
        if len(candidates) != 1:
            raise ValueError("terminal detectors require one final dependent mz")
        terminal_operation = candidates[0]

    replacements = {}
    for submission, operation in enumerate(measurements):
        fields = {
            "circuit": circuit,
            "stream_id": decode_operation_id,
            "stream_offset": submission,
            "emits_detector_data": True,
        }
        if terminal_operation is not None and operation is measurements[-1]:
            fields.update(
                syndrome_fragment_index=0, syndrome_fragment_count=2
            )
        replacements[operation.id] = replace(operation, **fields)
    if terminal_operation is not None:
        replacements[terminal_operation.id] = replace(
            terminal_operation,
            circuit=circuit,
            stream_id=decode_operation_id,
            stream_offset=len(measurements) - 1,
            emits_detector_data=True,
            finalizes_stream_round=True,
            syndrome_fragment_index=1,
            syndrome_fragment_count=2,
        )
    program.operations = [
        replacements.get(operation.id, operation)
        for operation in program.operations
    ]

    owner = Operation(
        decode_operation_id,
        "QLX physical decode stream",
        (patch,),
        circuit=circuit,
        patches=(patch,),
    )
    program.decoder_operations = (owner,)
    program.rounds.rounds_by_op[decode_operation_id] = len(measurements)
    program.detector_rounds_by_stream = {
        decode_operation_id: detector_rounds
    }
    program.terminal_detector_ids_by_stream = {
        decode_operation_id: terminal_ids
    }
    program.terminal_data_bits_by_stream = {
        decode_operation_id: terminal_data_count
    }


def qlx_frontend(diagram: Any, *,
                 feedback_from_measurements: bool = False,
                 realtime_artifact: Any = None,
                 physical_circuit: Any = None,
                 detector_metadata: Any = None,
                 decode_operation_id: Optional[int] = None) -> QLXProgram:
    """Build a structural decsim program from a QLX SpaceTimeDiagram.

    ``diagram`` may be the live object, its ``as_dict()`` form, or the raw
    entry list. With ``feedback_from_measurements=True``, every op whose
    dependencies include a MEASURE op gets ``blocked_by`` that measurement
    (opt-in; see module docstring).

    ``realtime_artifact`` (optional): a ``RealtimeArtifactTarget`` JSON
    string or parsed dict for the SAME program. It is the authoritative
    feedback source: when it reports ``conditional_feedback: false``,
    requesting ``feedback_from_measurements=True`` raises instead of
    fabricating feedback the program does not have. It also supplies
    decoder latency (rounds), the pre-feedback idle budget, runtime ops,
    and patch-local streams on the returned ``QLXProgram``. QLX does not
    export WHICH op is classically conditioned, so per-op wiring remains
    the measurement heuristic even when the artifact confirms feedback
    exists — callers must review ``feedback_candidates``.
    """
    artifact = realtime_artifact
    if isinstance(artifact, str):
        import json
        artifact = json.loads(artifact)
    if artifact is not None and artifact.get("kind") != "qlx.realtime_artifact":
        raise ValueError(
            "realtime_artifact must be a RealtimeArtifactTarget emission "
            f"(kind='qlx.realtime_artifact'); got kind={artifact.get('kind')!r}")
    artifact_feedback = None
    if artifact is not None:
        artifact_feedback = bool(
            artifact.get("execution_model", {}).get("conditional_feedback"))
        if feedback_from_measurements and not artifact_feedback:
            raise ValueError(
                "feedback_from_measurements=True, but the realtime artifact "
                "says this program has NO conditional feedback "
                "(execution_model.conditional_feedback=false); refusing to "
                "fabricate feedback wiring")
        if diagram is None:
            diagram = artifact.get("schedule")

    entries = diagram
    if isinstance(diagram, dict):
        entries = diagram["entries"]
    elif hasattr(diagram, "entries"):
        entries = diagram.entries

    physical_values = (
        physical_circuit, detector_metadata, decode_operation_id
    )
    if any(value is not None for value in physical_values) and not all(
        value is not None for value in physical_values
    ):
        raise ValueError(
            "physical_circuit, detector_metadata, and decode_operation_id "
            "must be supplied together")

    ids = {}
    for pos, entry in enumerate(entries):
        source_id = _value(entry, "op_id")
        if source_id in ids:
            raise ValueError("QLX source operation ids must be unique")
        ids[source_id] = pos

    patch_of_cell: dict = {}
    parsed = []
    for pos, entry in enumerate(entries):
        cells = tuple(tuple(c) for c in (_value(entry, "occupied_slots") or ()))
        for cell in cells:
            patch_of_cell.setdefault(cell, len(patch_of_cell))
        deps = tuple(ids[d] for d in (_value(entry, "dependencies") or ()))
        name = _short_name(_value(entry, "op_name") or "")
        if name == "repeat":
            raise ValueError("fabric.repeat must be expanded before lowering")
        if name not in _TASK_NAMES:
            raise ValueError(f"unsupported QLX task fabric.{name}")
        parsed.append({
            "pos": pos, "qlx_id": _value(entry, "op_id"),
            "name": name,
            "deps": deps, "cells": cells,
            "duration": _nonnegative_int(
                _value(entry, "duration"), "duration"
            ),
            "consumes": _value(entry, "consumes"),
            "produces": _value(entry, "produces"),
            "protocol": _value(entry, "protocol"),
            "start_round": _nonnegative_int(
                _value(entry, "start_round"), "start_round"
            ),
        })

    tasks_by_cell = {}
    for task in parsed:
        for cell in task["cells"]:
            tasks_by_cell.setdefault(cell, []).append(task)
    for cell_tasks in tasks_by_cell.values():
        ordered = sorted(
            cell_tasks, key=lambda task: (task["start_round"], task["pos"])
        )
        for prior, current in zip(ordered, ordered[1:]):
            prior_end = prior["start_round"] + prior["duration"]
            if (
                prior["duration"] > 0
                and current["duration"] > 0
                and current["start_round"] < prior_end
            ):
                raise ValueError("positive-duration QLX tasks overlap one cell")
            if prior["pos"] not in current["deps"]:
                current["deps"] = current["deps"] + (prior["pos"],)

    for task in parsed:
        for predecessor_id in task["deps"]:
            predecessor = parsed[predecessor_id]
            predecessor_end = (
                predecessor["start_round"] + predecessor["duration"]
            )
            if predecessor_end > task["start_round"]:
                raise ValueError(
                    "QLX task starts before its predecessor finishes"
                )

    flows, consumes_at = [], {}
    dependents: dict = {p["pos"]: [] for p in parsed}
    for p in parsed:
        for dep in p["deps"]:
            dependents[dep].append(p["pos"])
    for p in parsed:
        if p["produces"] is None:
            continue
        chain = [p["pos"]]
        head = p["pos"]
        while True:
            nxt = [q for q in dependents[head]
                   if parsed[q]["name"] == "transport"]
            if len(nxt) > 1:
                raise ValueError("QLX resource transport chain forks")
            if not nxt:
                break
            head = nxt[0]
            if head in chain:
                raise ValueError("QLX resource transport chain cycles")
            chain.append(head)
        sinks = [q for q in dependents[head]
                 if parsed[q]["name"] == "inject"]
        if len(sinks) != 1:
            raise ValueError("QLX resource must terminate at one inject")
        sink = sinks[0]
        if sink in consumes_at:
            raise ValueError("QLX resource chains join at one inject")
        declared_kind = parsed[sink]["consumes"]
        if declared_kind is not None and declared_kind != p["produces"]:
            raise ValueError("QLX produced and consumed resource kinds differ")
        chain.append(sink)
        consumes_at[sink] = p["produces"]
        flows.append((p["produces"], *[parsed[i]["qlx_id"] for i in chain]))
    chained_transports = {
        item for flow in flows for item in flow[2:-1]
    }
    for task in parsed:
        if task["name"] == "transport" and task["qlx_id"] not in chained_transports:
            raise ValueError("orphan QLX resource transport")
        if task["name"] == "inject" and task["pos"] not in consumes_at:
            raise ValueError("QLX inject has no unique resource chain")

    operations, raw_durations, protocols, start_rounds = [], {}, {}, {}
    feedback_candidates, explicit_if_candidates = [], []
    has_explicit_if = any(p["name"] == "if" for p in parsed)
    for p in parsed:
        patches = tuple(patch_of_cell[c] for c in p["cells"])
        kind = _KIND_BY_NAME.get(p["name"], OpKind.GENERIC)
        bit_deps = []
        if p["name"] == "if":
            # EXPLICIT feedback: a fabric.if entry is classically
            # conditioned on a bit-producing dependency (mz/mx/measure/
            # measure_product). This is authoritative, unlike the
            # measurement heuristic below.
            bit_deps = [d for d in p["deps"]
                        if parsed[d]["name"] in _BIT_PRODUCER_NAMES]
            if bit_deps:
                explicit_if_candidates.append((p["pos"], bit_deps[0]))
        measure_deps = [d for d in p["deps"]
                        if parsed[d]["name"] in _BIT_PRODUCER_NAMES]
        if measure_deps and not has_explicit_if:
            feedback_candidates.append((p["pos"], measure_deps[0]))
        if has_explicit_if:
            # explicit fabric.if entries override the heuristic entirely
            blocked = (bit_deps[0] if p["name"] == "if" and bit_deps
                       else None) if feedback_from_measurements else None
        else:
            blocked = (measure_deps[0]
                       if feedback_from_measurements and measure_deps
                       else None)
        operations.append(Operation(
            id=p["pos"], name=f"{p['name']}[{p['qlx_id']}]",
            qubits=patches, patches=patches,
            clifford=p["name"] not in ("t", "tdg", "inject"),
            consumes_magic_state=False,
            predecessors=p["deps"],
            decoder_boundary_predecessors=(),
            has_successor=False,
            scheduled_start_round=p["start_round"],
            emits_detector_data=False,
            blocked_by=blocked,
            kind=kind))
        raw_durations[p["pos"]] = p["duration"]
        if p["protocol"] is not None:
            protocols[p["pos"]] = p["protocol"]
        start_rounds[p["pos"]] = p["start_round"]

    if has_explicit_if:
        feedback_candidates = explicit_if_candidates
    rounds = PerOpRounds({op.id: raw_durations[op.id]
                          for op in operations})
    program = QLXProgram(
        operations=operations, rounds=rounds,
        op_ids={p["pos"]: p["qlx_id"] for p in parsed},
        patch_of_cell=dict(patch_of_cell), raw_durations=raw_durations,
        resource_flows=flows, feedback_candidates=feedback_candidates,
        protocols=protocols, start_rounds=start_rounds)
    if physical_circuit is not None:
        _add_physical_stream(
            program, physical_circuit, detector_metadata, decode_operation_id
        )
    if artifact is not None:
        timing = artifact.get("timing", {})
        idle = artifact.get("idle_policy", {})
        program.conditional_feedback = artifact_feedback
        program.decoder_latency_rounds = timing.get("decoder_latency_rounds")
        program.idle_rounds_for_decoder_wait = idle.get(
            "idle_rounds_for_decoder_wait")
        program.runtime_ops = list(artifact.get("runtime_ops", []))
        program.streams = list(artifact.get("streams", []))
    return program
