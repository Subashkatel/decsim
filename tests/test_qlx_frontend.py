"""QLX structural frontend (Phase 1-2): the captured live fixture drives
every mapping rule, plus one end-to-end engine run.

Fixture: tests/data/qlx/schedule_h_then_t.json — a real
``qlx.estimate.schedule()`` dump (h_then_t on Steane compute + Surface[15]
factory) captured from the apptainer image (regenerate with
``./tools/qlx python3 decsim/tests/data/qlx/generate_qlx_fixtures.py``
from the workspace root). 8 fabric ops:
alloc -> prep_z -> h -> inject -> mz -> dealloc on compute cell (C0,0),
produce_resource('T') on factory cell (F0,0) -> transport -> inject.

STRUCTURAL coupling only: no stim, no syndromes, no QLX-origin LER claims
(the frontend module docstring states the Phase-3 blocker).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.frontends.qlx import qlx_frontend
from decsim.message import OpKind
from decsim.run_spec import RunSpec, simulate
from decsim.decoders import PerRoundDecoder

FIXTURE = pathlib.Path(__file__).resolve().parent / \
    "data/qlx/schedule_h_then_t.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="QLX fixture missing; regenerate with ./tools/qlx python3 "
           "decsim/tests/data/qlx/generate_qlx_fixtures.py")


@pytest.fixture(scope="module")
def program():
    return qlx_frontend(json.loads(FIXTURE.read_text()))


def test_operation_count_and_order_preserved(program):
    assert len(program.operations) == 8
    assert [program.op_ids[op.id] for op in program.operations] == [
        "alloc_0", "produce_resource_0", "prep_z_0", "transport_0",
        "h_0", "inject_0", "mz_0", "dealloc_0"]


def test_dependency_dag_matches_the_diagram(program):
    by_qlx = {qlx: i for i, qlx in program.op_ids.items()}
    deps = {program.op_ids[op.id]:
            tuple(program.op_ids[d] for d in op.predecessors)
            for op in program.operations}
    assert deps["alloc_0"] == ()
    assert deps["prep_z_0"] == ("alloc_0",)
    assert deps["h_0"] == ("prep_z_0",)
    assert deps["transport_0"] == ("produce_resource_0",)
    assert deps["inject_0"] == ("h_0", "transport_0")
    assert deps["mz_0"] == ("inject_0",)
    assert deps["dealloc_0"] == ("mz_0",)
    ops = {op.id: op for op in program.operations}
    assert ops[by_qlx["dealloc_0"]].has_successor is False
    assert ops[by_qlx["mz_0"]].has_successor is True


def test_durations_preserved_and_rounds_clamped(program):
    by_qlx = {qlx: i for i, qlx in program.op_ids.items()}
    assert program.raw_durations[by_qlx["produce_resource_0"]] == 120
    assert program.raw_durations[by_qlx["transport_0"]] == 1
    assert program.raw_durations[by_qlx["h_0"]] == 0
    ops = {op.id: op for op in program.operations}
    for op in program.operations:
        r = program.rounds.rounds_for(op, code=None)
        assert r == max(1, program.raw_durations[op.id])


def test_region_slot_cells_become_patches(program):
    assert program.patch_of_cell == {("C0", 0): 0, ("F0", 0): 1}
    ops = {program.op_ids[op.id]: op for op in program.operations}
    assert ops["alloc_0"].patches == (0,)
    assert ops["produce_resource_0"].patches == (1,)
    # transport occupies no fabric cell -> inherits its dependency's patches
    assert ops["transport_0"].patches == (1,)


def test_resource_flow_produce_transport_consume(program):
    assert program.resource_flows == [
        ("T", "produce_resource_0", "transport_0", "inject_0")]
    ops = {program.op_ids[op.id]: op for op in program.operations}
    assert ops["inject_0"].consumes_magic_state is True
    assert ops["inject_0"].clifford is False
    assert ops["inject_0"].kind is OpKind.INJECT
    assert ops["produce_resource_0"].consumes_magic_state is False
    assert ops["mz_0"].kind is OpKind.MEASURE


def test_feedback_is_opt_in_not_auto_wired(program):
    by_qlx = {qlx: i for i, qlx in program.op_ids.items()}
    assert program.feedback_candidates == [
        (by_qlx["dealloc_0"], by_qlx["mz_0"])]
    assert all(op.blocked_by is None for op in program.operations)
    wired = qlx_frontend(json.loads(FIXTURE.read_text()),
                         feedback_from_measurements=True)
    ops = {wired.op_ids[op.id]: op for op in wired.operations}
    assert ops["dealloc_0"].blocked_by == by_qlx["mz_0"]


def test_end_to_end_timing_run_through_the_engine(program):
    """The structural program executes through build_and_run (timing plane,
    magic-state demand served by the default infinite factory) and every
    operation completes; QLX's 120-round distillation dominates the
    makespan."""

    res = simulate(RunSpec(
              ops=program.operations,
              num_units=2,
              d=3,
              rounds_policy=program.rounds,
              round_us=1.0,
              decoder=PerRoundDecoder(tau_us=1.0),
          ), verbose=False)
    assert res.result.chip_done_ticks >= 120 * 1_000_000   # >= the 120-round distill
    assert res.result.fully_done_ticks >= res.result.chip_done_ticks
    cluster = res.window_manager
    assert len(cluster.committed_windows) == cluster.total_windows, \
        "not every QLX-derived window was decoded and committed"


# ---------------------------------------------------------------------------
# REAL-EXPERIMENT corpus: three schedules captured fresh from the actual QLX
# compiler (tests/data/qlx/generate_qlx_fixtures.py, run in the container).
# Validates the frontend against real compiler output beyond the
# single fixture, including multi-flow and multi-slot programs.
# ---------------------------------------------------------------------------

CORPUS = pathlib.Path(__file__).resolve().parent / "data/qlx"
_CORPUS_FILES = sorted(CORPUS.glob("schedule_*.json"))


@pytest.mark.skipif(not _CORPUS_FILES, reason="run ./tools/qlx python3 "
                    "decsim/tests/data/qlx/generate_qlx_fixtures.py first")
@pytest.mark.parametrize("path", _CORPUS_FILES,
                         ids=[p.stem for p in _CORPUS_FILES])
def test_corpus_structural_invariants(path):
    prog = qlx_frontend(json.loads(path.read_text()))
    ids = [op.id for op in prog.operations]
    assert ids == sorted(set(ids)), "ids must be unique and ordered"
    for op in prog.operations:
        assert all(0 <= d < len(ids) and d != op.id for d in op.predecessors)
        assert prog.rounds.rounds_for(op, code=None) >= 1
        assert op.patches, f"{op.name} has no patch"
    # every resource flow starts at a producer and ends at a consumer
    by_qlx = {q: i for i, q in prog.op_ids.items()}
    ops = {op.id: op for op in prog.operations}
    for flow in prog.resource_flows:
        resource, producer, *rest = flow
        assert ops[by_qlx[producer]].consumes_magic_state is False
        assert ops[by_qlx[flow[-1]]].consumes_magic_state is True


@pytest.mark.skipif(not (CORPUS / "schedule_h_then_2t.json").exists(),
                    reason="corpus not captured")
def test_two_t_program_has_two_flows_and_parallel_factory_slots():
    prog = qlx_frontend(json.loads(
        (CORPUS / "schedule_h_then_2t.json").read_text()))
    assert len(prog.operations) == 11
    assert len(prog.resource_flows) == 2
    assert {f[0] for f in prog.resource_flows} == {"T"}
    # the real compiler placed the two distillations on DIFFERENT factory
    # slots, concurrently — the patch mapping must preserve that
    by_qlx = {q: i for i, q in prog.op_ids.items()}
    ops = {op.id: op for op in prog.operations}
    p0 = ops[by_qlx["produce_resource_0"]].patches
    p1 = ops[by_qlx["produce_resource_1"]].patches
    assert p0 != p1, "parallel factory slots collapsed onto one patch"
    # both injections consume, and QLX serialized them via an explicit dep
    assert ops[by_qlx["inject_1"]].consumes_magic_state is True
    assert by_qlx["inject_0"] in ops[by_qlx["inject_1"]].predecessors


def _critical_path_rounds(prog):
    """Independent reference: longest path over dependency edges PLUS
    same-patch program order, with decsim's clamped >=1-round durations."""
    ops = prog.operations
    dur = {op.id: max(1, prog.raw_durations[op.id]) for op in ops}
    edges = {op.id: set(op.predecessors) for op in ops}
    last_on_patch: dict = {}
    for op in ops:                       # program order per patch
        for patch in op.patches:
            if patch in last_on_patch:
                edges[op.id].add(last_on_patch[patch])
            last_on_patch[patch] = op.id
    finish: dict = {}
    for op in ops:                       # ids are topological (QLX order)
        start = max((finish[d] for d in edges[op.id]), default=0)
        finish[op.id] = start + dur[op.id]
    return max(finish.values())


@pytest.mark.skipif(not _CORPUS_FILES, reason="corpus not captured")
@pytest.mark.parametrize("path", _CORPUS_FILES,
                         ids=[p.stem for p in _CORPUS_FILES])
def test_timing_cross_validation_against_real_qlx_schedules(path):
    """decsim's engine must reproduce the independent critical path EXACTLY
    on every real compiled program, and exceed QLX's own makespan only by
    the zero-duration ops decsim clamps to 1 round (delta reported)."""

    doc = json.loads(path.read_text())
    prog = qlx_frontend(doc)
    # QLX may schedule ops CONCURRENTLY on the same slot (produce_resource
    # in mem_surface_t; mz/dealloc siblings in ls_cx); decsim's chip
    # requires program-order wiring for patch-sharing ops, so chain each
    # op to the LAST op on its patch — the same rule as
    # frontends.circuit._wire_patch_dependencies — before running (G7P4
    # bring-up amendment 2, generalized for the ls_cx corpus program;
    # both the engine and the reference critical path see the same edges)
    last_on_patch: dict = {}
    for op in prog.operations:
        for patch in op.patches:
            prev = last_on_patch.get(patch)
            if prev is not None and prev not in op.predecessors:
                op.predecessors = tuple(op.predecessors) + (prev,)
                prog.operations[prev].has_successor = True
            last_on_patch[patch] = op.id
    res = simulate(RunSpec(
              ops=prog.operations,
              num_units=4,
              d=3,
              rounds_policy=prog.rounds,
              round_us=1.0,
              decoder=PerRoundDecoder(tau_us=1.0),
          ), verbose=False)
    decsim_rounds = res.result.chip_done_ticks / 1_000_000
    reference = _critical_path_rounds(prog)
    assert decsim_rounds == reference, \
        f"decsim {decsim_rounds} != independent critical path {reference}"
    qlx_makespan = max(prog.start_rounds[op.id] +
                       prog.raw_durations[op.id] for op in prog.operations)
    n_zero = sum(1 for op in prog.operations
                 if prog.raw_durations[op.id] == 0)
    assert qlx_makespan <= decsim_rounds <= qlx_makespan + n_zero, \
        (f"decsim {decsim_rounds} vs QLX {qlx_makespan} beyond the "
         f"zero-duration clamp budget ({n_zero})")


# ---------------------------------------------------------------------------
# Realtime-artifact feedback wiring (gap G2): the artifact is the
# AUTHORITATIVE feedback source. Fixtures: tests/data/qlx/realtime_*.json
# (regenerate with ./tools/qlx python3 decsim/tests/data/qlx/
# dump_realtime_artifacts.py).
# ---------------------------------------------------------------------------

ARTIFACTS = pathlib.Path(__file__).resolve().parent / "data/qlx"


@pytest.mark.skipif(
    not (ARTIFACTS / "realtime_mem_surface.json").exists(),
    reason="run dump_realtime_artifacts.py via ./tools/qlx first")
def test_artifact_refuses_fabricated_feedback():
    """mem_surface has decode_bit but NO fabric.if; asking for feedback
    wiring against its artifact must raise, not silently invent it."""
    art = json.loads((ARTIFACTS / "realtime_mem_surface.json").read_text())
    assert art["execution_model"]["conditional_feedback"] is False
    with pytest.raises(ValueError, match="NO conditional feedback"):
        qlx_frontend(None, feedback_from_measurements=True,
                     realtime_artifact=art)
    # without the wiring request the artifact enriches the program
    prog = qlx_frontend(None, realtime_artifact=art)
    assert prog.conditional_feedback is False
    assert prog.idle_rounds_for_decoder_wait == 0
    assert [o["op_name"] for o in prog.runtime_ops] == ["fabric.decode_bit"]
    assert all(op.blocked_by is None for op in prog.operations)


@pytest.mark.skipif(
    not (ARTIFACTS / "realtime_byproduct_ff.json").exists(),
    reason="run dump_realtime_artifacts.py via ./tools/qlx first")
def test_artifact_confirms_real_feedback_program():
    """byproduct_ff carries a real fabric.if: the artifact confirms
    feedback, budgets one idle round for the decoder wait, and the
    schedule embedded in the artifact drives the frontend directly."""
    art = json.loads((ARTIFACTS / "realtime_byproduct_ff.json").read_text())
    assert art["execution_model"]["conditional_feedback"] is True
    prog = qlx_frontend(None, feedback_from_measurements=True,
                        realtime_artifact=art)
    assert prog.conditional_feedback is True
    assert prog.decoder_latency_rounds == 1
    assert prog.idle_rounds_for_decoder_wait == 1
    assert {o["op_name"] for o in prog.runtime_ops} == \
        {"fabric.if", "fabric.decode_bit"}
    # the measurement heuristic wired the candidate under artifact consent
    wired = [op for op in prog.operations if op.blocked_by is not None]
    assert len(wired) == len(prog.feedback_candidates) > 0
    assert len(prog.streams) == 1                 # single (region,slot) patch


def test_artifact_kind_is_checked():
    with pytest.raises(ValueError, match="qlx.realtime_artifact"):
        qlx_frontend(None, realtime_artifact={"kind": "something_else"})
