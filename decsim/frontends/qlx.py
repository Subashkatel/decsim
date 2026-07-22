"""QLX structural frontend: SpaceTimeDiagram entries -> decsim Operations.

Phase 1 of the QLX integration plan (STRUCTURAL coupling only): consumes the
per-op schedule DAG that ``qlx.estimate.schedule()`` exposes via
``SpaceTimeDiagram.entries`` — op ids, fabric op names, dependencies, round
durations, occupied (region, slot) fabric cells, resource produces/consumes,
protocols — and emits decsim ``Operation``s plus a ``PerOpRounds`` map.

What this deliberately does NOT do (physical coupling, Phase 3): no stim
circuits, no DEMs, no syndromes. QLX today emits only whole-program circuits
(``emit_circuit``), so per-op physical coupling is blocked on a QLX-side
per-op slicer; any run built from this frontend is TIMING/STRUCTURE only,
and no QLX-origin logical-error-rate claim may be made from it.

Input formats accepted:
  * a live ``SpaceTimeDiagram`` (objects with ``.entries`` attributes),
  * ``as_dict()``-style dicts, or
  * the reflection capture format used by
    ``docs/audits/qlx-output-sample/schedule_output.json``, where every
    value is a Python repr string ("'alloc_0'", "(('C0', 0),)", ...).

Mapping rules (each asserted by tests/test_qlx_frontend.py):
  * entry order is preserved; ``Operation.id`` = position; the original
    ``op_id`` string is kept in ``QLXProgram.op_ids``.
  * ``dependencies`` -> ``Operation.predecessors`` / ``has_successor``.
  * ``duration`` (rounds) -> ``PerOpRounds``; zero-duration fabric ops are
    clamped to 1 round (decsim requires >= 1) with the raw value preserved
    in ``QLXProgram.raw_durations``.
  * ``occupied_slots`` (region, slot) pairs -> integer patches; ops that
    occupy no fabric cell (e.g. ``fabric.transport``) inherit their first
    dependency's patches (documented convenience so the chip can run them).
  * ``fabric.mz``/``fabric.measure*`` -> OpKind.MEASURE; ``fabric.inject``
    -> OpKind.INJECT; ``fabric.merge*`` -> OpKind.MERGE; others GENERIC.
  * resource flow: a ``produces`` value starts a resource edge; transports
    forward it; the op that depends on the end of the chain consumes it ->
    ``consumes_magic_state=True`` and ``clifford=False`` (T consumption).
    The full chain is recorded in ``QLXProgram.resource_flows``.
  * feedback is NOT auto-wired: ops depending on a MEASURE op are listed in
    ``QLXProgram.feedback_candidates``; the caller decides which become
    ``blocked_by`` (decsim's feedback gating), because the schedule alone
    does not say which dependencies are classically conditioned.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
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


def qlx_frontend(diagram: Any, *,
                 feedback_from_measurements: bool = False,
                 realtime_artifact: Any = None) -> QLXProgram:
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

    ids = {}
    for pos, entry in enumerate(entries):
        ids[_value(entry, "op_id")] = pos

    patch_of_cell: dict = {}
    parsed = []
    for pos, entry in enumerate(entries):
        cells = tuple(tuple(c) for c in (_value(entry, "occupied_slots") or ()))
        for cell in cells:
            patch_of_cell.setdefault(cell, len(patch_of_cell))
        deps = tuple(ids[d] for d in (_value(entry, "dependencies") or ()))
        parsed.append({
            "pos": pos, "qlx_id": _value(entry, "op_id"),
            "name": _short_name(_value(entry, "op_name") or ""),
            "deps": deps, "cells": cells,
            "duration": int(_value(entry, "duration") or 0),
            "consumes": _value(entry, "consumes"),
            "produces": _value(entry, "produces"),
            "protocol": _value(entry, "protocol"),
            "start_round": int(_value(entry, "start_round") or 0),
        })

    # resource flow: produce -> (transports...) -> first non-transport
    # dependent = the consumer
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
            if not nxt:
                break
            head = nxt[0]
            chain.append(head)
        sinks = [q for q in dependents[head]
                 if parsed[q]["name"] != "transport"]
        if sinks:
            chain.append(sinks[0])
            consumes_at[sinks[0]] = p["produces"]
        flows.append((p["produces"], *[parsed[i]["qlx_id"] for i in chain]))

    operations, raw_durations, protocols, start_rounds = [], {}, {}, {}
    feedback_candidates, explicit_if_candidates = [], []
    has_explicit_if = any(p["name"] == "if" for p in parsed)
    has_successor = {p["pos"]: bool(dependents[p["pos"]]) for p in parsed}
    for p in parsed:
        patches = tuple(patch_of_cell[c] for c in p["cells"])
        if not patches and p["deps"]:
            patches = tuple(
                patch_of_cell[c] for c in parsed[p["deps"][0]]["cells"])
        kind = _KIND_BY_NAME.get(p["name"], OpKind.GENERIC)
        consumed = consumes_at.get(p["pos"], p["consumes"])
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
            qubits=patches or (0,), patches=patches or (0,),
            clifford=consumed is None,
            consumes_magic_state=consumed is not None,
            predecessors=p["deps"], has_successor=has_successor[p["pos"]],
            blocked_by=blocked,
            kind=kind))
        raw_durations[p["pos"]] = p["duration"]
        if p["protocol"] is not None:
            protocols[p["pos"]] = p["protocol"]
        start_rounds[p["pos"]] = p["start_round"]

    if has_explicit_if:
        feedback_candidates = explicit_if_candidates
    rounds = PerOpRounds({op.id: max(1, raw_durations[op.id])
                          for op in operations})
    program = QLXProgram(
        operations=operations, rounds=rounds,
        op_ids={p["pos"]: p["qlx_id"] for p in parsed},
        patch_of_cell=dict(patch_of_cell), raw_durations=raw_durations,
        resource_flows=flows, feedback_candidates=feedback_candidates,
        protocols=protocols, start_rounds=start_rounds)
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
