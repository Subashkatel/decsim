"""Topology is the link matrix, not the component graph.

The same engine models a distributed data center (DecLat: chip -> controller ->
decoder cluster -> a SEPARATE orchestrator/HPC node -> back) and a single fused
control system (Rigetti arXiv:2410.05202: the FPGA decoder lives inside the
control box; there is no separate orchestrator). The difference is ONLY the seven
link latencies -- one shared LinkModel that every component reads its hop cost from
(wiring threads one instance to the controller and the cluster).

So "everything in one box" is just internal hops set to zero. A zero-latency hop
still happens -- engine.schedule(0, action) enqueues at the current tick and pops
at the SAME virtual time, but strictly after the event that scheduled it (a larger
insertion seq breaks the tie), so causal order survives while no time advances.

These tests assert the two consequences that make the fused model honest:
  - a path's latency is the SUM of its hops, ADDITIVE on top of the rest, and
    INVARIANT to how that sum is split (so collapsing wdo+oc+cq into one "fused
    feedback" number is a labelling choice, not a timing change);
  - zero hops add exactly nothing (fused == the bare decode/round timeline).
"""
from conftest import fixed_latency_link_config, trace_time  # noqa: F401

from decsim.config import us
from decsim.controllers import ModularController
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import DecodeResult, Operation
from decsim.schemes import NaiveOnlineScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


class _FixedLatency:
    """Timing-only decoder: a constant modelled decode latency, trivial result."""
    def __init__(self, latency_us=1.0):
        self.latency_ticks = us(latency_us)

    def latency(self, job):
        return self.latency_ticks

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(0,))


def _controller(engine, links):
    return ModularController(engine, links=links, log_syndromes=False)


def _links(t_qc=0.0, t_cwd=0.0, t_dd=0.0, t_wdo=0.0,
           t_oc=0.0, t_cq=0.0):
    return fixed_latency_link_config(
        qc=us(t_qc),
        cwd=us(t_cwd),
        dd=us(t_dd),
        wdo=us(t_wdo),
        oc=us(t_oc),
        cq=us(t_cq),
    )


def _first_round_arrival(t_qc, t_cwd):
    """When round 1 of a memory op reaches the decoder cluster = production time
    + the forward budget (t_qc + t_cwd). Everything else is zero."""
    op = Operation(0, "M(q0)", (0,), clifford=True, patches=(0,))
    res = simulate(RunSpec(
              ops=[op],
              num_units=1,
              d=3,
              rounds_policy=FixedRounds(3),
              round_us=1.0,
              decoder=_FixedLatency(1.0),
              links=_links(t_qc=t_qc, t_cwd=t_cwd),
              make_controller=_controller,
          ), verbose=False)
    return res.window_manager.windows[(0, 0)].t_first_round


def test_forward_latency_is_additive_and_split_invariant():
    """Forward path (qpu -> decoder): the round arrives at production + (t_qc+t_cd),
    independent of how the 1.4 us is split between the two hops; zero adds nothing."""
    split_all_qc = _first_round_arrival(1.4, 0.0)
    split_all_cd = _first_round_arrival(0.0, 1.4)
    split_even = _first_round_arrival(0.7, 0.7)
    assert split_all_qc == split_all_cd == split_even        # split-invariant

    fused = _first_round_arrival(0.0, 0.0)                    # no transmission at all
    assert split_all_qc == fused + us(1.4)                    # exactly additive


def _decode_release(t_wdo, t_oc, t_cq):
    """A feedback-blocked successor releases when op 0's correction returns through the
    feedback path (t_wdo + t_oc + t_cq). Returns release and decode time."""
    ops = CircuitFrontend([
        Operation(0, "T0", (0,), clifford=False, consumes_magic_state=False),
        Operation(1, "T1", (0,), clifford=False, blocked_by=0,
                  consumes_magic_state=False),
    ]).build()
    res = simulate(RunSpec(
              ops=ops,
              num_units=1,
              d=3,
              rounds_policy=FixedRounds(3),
              round_us=1.0,
              decoder=_FixedLatency(1.0),
              scheme=NaiveOnlineScheme(),
              links=_links(t_wdo=t_wdo, t_oc=t_oc, t_cq=t_cq),
              make_controller=_controller,
          ), verbose=False)
    return res.chip.decode_release_time[1], res.window_manager.windows[(0, 0)].t_done


def test_feedback_latency_is_additive_and_split_invariant():
    """Feedback path (result -> qpu): the gate releases at op0's decode-done + the
    feedback budget, INVARIANT to whether that budget sits on WDO, OC, or CQ -- which
    is exactly why collapsing WDO+OC+CQ into one 'fused control-system feedback'
    number changes no timing. Zero feedback hops => release the instant decode is
    done (the fully-fused, no-separate-orchestrator case)."""
    on_do, done_do = _decode_release(1.7, 0.0, 0.0)
    on_oc, done_oc = _decode_release(0.0, 1.7, 0.0)
    on_cq, done_cq = _decode_release(0.0, 0.0, 1.7)
    assert on_do == on_oc == on_cq                           # split-invariant
    assert done_do == done_oc == done_cq                     # decode unaffected by feedback hops
    assert on_do == done_do + us(1.7)                        # released a feedback-budget after decode

    fused_release, fused_done = _decode_release(0.0, 0.0, 0.0)  # everything in one box
    assert fused_release == fused_done                        # no feedback transmission time
