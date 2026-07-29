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
    assert ops[by_qlx["mz_0"]].has_successor is False
    assert all(op.decoder_boundary_predecessors == ()
               for op in program.operations)


def test_durations_preserved_without_clamping(program):
    by_qlx = {qlx: i for i, qlx in program.op_ids.items()}
    assert program.raw_durations[by_qlx["produce_resource_0"]] == 120
    assert program.raw_durations[by_qlx["transport_0"]] == 1
    assert program.raw_durations[by_qlx["h_0"]] == 0
    ops = {op.id: op for op in program.operations}
    for op in program.operations:
        r = program.rounds.rounds_for(op, code=None)
        assert r == program.raw_durations[op.id]


def test_region_slot_cells_become_patches(program):
    assert program.patch_of_cell == {("C0", 0): 0, ("F0", 0): 1}
    ops = {program.op_ids[op.id]: op for op in program.operations}
    assert ops["alloc_0"].patches == (0,)
    assert ops["produce_resource_0"].patches == (1,)
    assert ops["transport_0"].patches == ()


def test_resource_flow_produce_transport_consume(program):
    assert program.resource_flows == [
        ("T", "produce_resource_0", "transport_0", "inject_0")]
    ops = {program.op_ids[op.id]: op for op in program.operations}
    assert ops["inject_0"].consumes_magic_state is False
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
        assert prog.rounds.rounds_for(op, code=None) >= 0
    # every resource flow starts at a producer and ends at a consumer
    by_qlx = {q: i for i, q in prog.op_ids.items()}
    ops = {op.id: op for op in prog.operations}
    for flow in prog.resource_flows:
        resource, producer, *rest = flow
        assert ops[by_qlx[producer]].consumes_magic_state is False
        assert ops[by_qlx[flow[-1]]].consumes_magic_state is False


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
    # QLX owns both injected states and serialized the injections explicitly.
    assert ops[by_qlx["inject_1"]].consumes_magic_state is False
    assert by_qlx["inject_0"] in ops[by_qlx["inject_1"]].predecessors


@pytest.mark.skipif(not _CORPUS_FILES, reason="corpus not captured")
@pytest.mark.parametrize("path", _CORPUS_FILES,
                         ids=[p.stem for p in _CORPUS_FILES])
def test_timing_cross_validation_against_real_qlx_schedules(path):
    """The engine reproduces every QLX absolute start, end, and makespan."""

    doc = json.loads(path.read_text())
    prog = qlx_frontend(doc)
    res = simulate(RunSpec(
              frontend=prog,
              decode_ops=list(prog.decoder_operations),
              num_units=4,
              d=3,
              rounds_policy=prog.rounds,
              round_us=1.0,
              decoder=PerRoundDecoder(tau_us=1.0),
          ), verbose=False)
    decsim_rounds = res.result.chip_done_ticks / 1_000_000
    qlx_makespan = max(prog.start_rounds[op.id] +
                       prog.raw_durations[op.id] for op in prog.operations)
    assert decsim_rounds == qlx_makespan
    for operation in prog.operations:
        assert res.chip.op_start_time[operation.id] == (
            prog.start_rounds[operation.id] * 1_000_000
        )
        assert res.chip.body_done_time[operation.id] == (
            (prog.start_rounds[operation.id]
             + prog.raw_durations[operation.id]) * 1_000_000
        )


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


def test_closed_vocabulary_rejects_repeat_and_unknown_tasks():
    base = {
        "op_id": "bad_0",
        "dependencies": (),
        "occupied_slots": (),
        "duration": 0,
        "consumes": None,
        "produces": None,
        "protocol": None,
        "start_round": 0,
    }
    for op_name, message in (
        ("fabric.repeat", "expanded"),
        ("fabric.future_task", "unsupported"),
    ):
        entry = dict(base, op_name=op_name)
        with pytest.raises(ValueError, match=message):
            qlx_frontend({"entries": [entry]})


def _resource_entry(op_id, name, dependencies=(), produces=None):
    return {
        "op_id": op_id, "op_name": f"fabric.{name}",
        "dependencies": dependencies, "occupied_slots": (),
        "duration": 0, "consumes": None, "produces": produces,
        "protocol": None, "start_round": 0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", 1.5),
        ("duration", True),
        ("start_round", 1.5),
        ("start_round", True),
    ],
)
def test_schedule_times_require_exact_integers(field, value):
    entry = _resource_entry("task", "alloc")
    entry[field] = value
    with pytest.raises(TypeError, match=field):
        qlx_frontend({"entries": [entry]})


def test_source_operation_ids_must_be_unique():
    entries = [
        _resource_entry("duplicate", "alloc"),
        _resource_entry("duplicate", "dealloc"),
    ]
    with pytest.raises(ValueError, match="unique"):
        qlx_frontend({"entries": entries})


def test_dependency_cannot_start_before_predecessor_finishes():
    first = _resource_entry("first", "alloc")
    first.update(duration=2, start_round=0)
    second = _resource_entry("second", "dealloc", ("first",))
    second.update(duration=0, start_round=1)
    with pytest.raises(ValueError, match="predecessor finishes"):
        qlx_frontend({"entries": [first, second]})


def test_dependency_may_start_at_predecessor_end():
    first = _resource_entry("first", "alloc")
    first.update(duration=2, start_round=0)
    second = _resource_entry("second", "dealloc", ("first",))
    second.update(duration=0, start_round=2)
    program = qlx_frontend({"entries": [first, second]})
    assert program.operations[1].predecessors == (0,)


def test_same_cell_zero_duration_tasks_may_share_one_tick():
    first = _resource_entry("first", "alloc")
    second = _resource_entry("second", "dealloc")
    for entry in (first, second):
        entry["occupied_slots"] = (("compute", 0),)
    program = qlx_frontend({"entries": [first, second]})
    assert program.operations[1].predecessors == (0,)


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([
            _resource_entry("p", "produce_resource", produces="T"),
            _resource_entry("t0", "transport", ("p",)),
            _resource_entry("t1", "transport", ("p",)),
        ], "forks"),
        ([_resource_entry("t", "transport")], "orphan"),
        ([
            _resource_entry("p0", "produce_resource", produces="T"),
            _resource_entry("p1", "produce_resource", produces="T"),
            _resource_entry("i", "inject", ("p0", "p1")),
        ], "join"),
    ],
)
def test_resource_chain_mutations_fail_closed(entries, message):
    with pytest.raises(ValueError, match=message):
        qlx_frontend({"entries": entries})
