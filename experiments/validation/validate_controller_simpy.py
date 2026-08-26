"""Gate 7: the controller against outside models on the same data.

The controller has three parts and each is checked against an independent
reference fed the same inputs a decsim run saw:

1. Syndrome packing (reassembly, packing, route arbitration, C2B / CWD
   transmission). Reference: fragment reassembly completes when the last
   fragment of a round has arrived (RFC 815 hole list, one hole per missing
   fragment), packing is a fixed service time (a SimPy timeout), and each
   route is a store-and-forward channel with ns-3's point-to-point FIFO rule
   (TransmitStart at max(send, previous TransmitComplete), CalculateBitsTxTime,
   plus the channel delay). Inputs: the QC delivery ticks of every fragment
   from the link ledger (Gate 6 checked those) and the run's timing card.
   Checked: every round's packing tick and its Buffer 0 publication tick, and
   every feedback-memory delivery tick, on the QLX mem_surface program, on a
   stream whose last round arrives as two fragments (t_pack applies), on Stim
   memory with a priced C2B hop, and on feedback chains that carry both
   routes at once.

2. Feedback streams (protected regions). Reference: a SimPy periodic process
   per protected region: from the start operation's issue tick S with the
   patch cadence c, boundaries at S + k c (k >= 1), one stream round emitted
   at each boundary, the region sealed at the boundary on which the end
   operation's body finished, and every operation that touches the live
   patch starting on a boundary tick. Unprotected stream bindings are a bump
   allocator: an undeclared segment binds at the stream's next free round.
   Checked on the lock's protected chain (reopen and feedback), the live
   stream pair, and the two-fragment stream.

3. Controller proper: idle rounds and conditional releases. Reference for
   idle decode demand is the sliding-window rule (Skoric et al., one job of
   rcom + rbuf rounds per commit region of rcom rounds); reference for a
   release is ns-3 replayed over the OC and CQ channels: OC sent at the
   decision tick, CQ sent at OC delivery, the QPU instructed at CQ delivery.

    python -m experiments.validation.validate_controller_simpy
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tmp" / "references" / "code" / "simpy" / "src"))

import simpy  # noqa: E402

from experiments import refactor_lock as lock  # noqa: E402
from experiments.validation.validate_links_ns3_simpy import ns3_point_to_point  # noqa: E402

REPORT = REPO / "experiments" / "results" / "validation" / "controller_simpy.md"


# ---- a traced decsim run ---------------------------------------------------

class Trace:
    """What the run's controller did, recorded at the seams from outside the
    core: no core module changes."""

    def __init__(self):
        self.packed = {}          # round_key -> tick _finish_packing ran
        self.published = {}       # round_key -> tick the round reached Buffer 0
        self.feedback_delivered = []   # (tick, round_key)
        self.issued = []          # (tick, operation_id, patches)
        self.stream_rounds = []   # (tick, stream_id, global_round, patch)
        self.sealed = []          # (tick, stream_id, round_count)
        self.bindings = []        # (tick, operation_id, stream_id, stream_offset)
        self.idle_demands = []    # (tick, rounds, label)
        self.idle_rounds = []     # (tick, operation_id, patch, round_index)
        self.instructed = []      # (tick, target_operation_id)
        self.decided = []         # (tick, target_operation_id)


@contextmanager
def traced(trace: Trace):
    """Wrap the controller seams for one run; restored on exit."""
    from decsim.controller import controller as controller_module
    from decsim.controller import feedback_streams as streams_module
    from decsim.controller import syndrome_packing as packing_module
    from decsim.windows import window_manager as window_module
    originals = []

    def wrap(owner, name, before=None, after=None):
        original = getattr(owner, name)

        def wrapped(self, *args, **kwargs):
            if before is not None:
                before(self, *args, **kwargs)
            value = original(self, *args, **kwargs)
            if after is not None:
                after(self, value, *args, **kwargs)
            return value
        originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    Packing = packing_module.SyndromePacking
    wrap(Packing, "_finish_packing",
         before=lambda self, context: trace.packed.__setitem__(context.round_key, self.engine.now))
    wrap(Packing, "_deliver_window_input_round",
         before=lambda self, context: trace.published.setdefault(context.round_key, self.engine.now))
    wrap(Packing, "_deliver_feedback_memory_round",
         before=lambda self, context, source: trace.feedback_delivered.append(
             (self.engine.now, context.round_key)))
    Controller = controller_module.Controller
    wrap(Controller, "issue_operation",
         before=lambda self, operation, idle_rounds: trace.issued.append(
             (self.engine.now, operation.id, tuple(operation.patches))))
    wrap(Controller, "emit_idle_round",
         before=lambda self, op_id, patch, round_index: trace.idle_rounds.append(
             (self.engine.now, op_id, patch, round_index)))
    wrap(Controller, "relay_instruction",
         before=lambda self, decision, deliver: trace.decided.append(
             (self.engine.now, decision.target_operation_id)))
    Streams = streams_module.FeedbackStreams
    wrap(Streams, "_bind",
         before=lambda self, operation_id, stream_id, offset: trace.bindings.append(
             (self.engine.now, operation_id, stream_id, offset)))
    Windows = window_module.WindowManager
    wrap(Windows, "seal_stream",
         before=lambda self, stream_id, count: trace.sealed.append(
             (self.engine.now, stream_id, count)))
    wrap(Windows, "accept_idle_decode_demand",
         before=lambda self, **kw: trace.idle_demands.append(
             (self.engine.now, kw["rounds"], kw["label"])))
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def wrap_qpu_stream_rounds(trace):
    """The QPU's emit_idle_stream_round is a device call: wrap the class too."""
    from decsim.qpu import cycle_clock

    original = cycle_clock.QPUDevice.emit_idle_stream_round

    def wrapped(self, operation, stream_id, global_round, patch):
        trace.stream_rounds.append((self.engine.now, stream_id, global_round, patch))
        return original(self, operation, stream_id, global_round, patch)
    cycle_clock.QPUDevice.emit_idle_stream_round = wrapped
    return lambda: setattr(cycle_clock.QPUDevice, "emit_idle_stream_round", original)


def wrap_runtime_decisions(trace):
    from decsim.frontends import execution_runtime as runtime_module
    original = runtime_module.ExecutionRuntime.on_decision

    def wrapped(self, decision):
        trace.instructed.append((self.engine.now, decision.target_operation_id))
        return original(self, decision)
    runtime_module.ExecutionRuntime.on_decision = wrapped
    return lambda: setattr(runtime_module.ExecutionRuntime, "on_decision", original)


def run_traced(spec):
    trace = Trace()
    restore = [wrap_qpu_stream_rounds(trace), wrap_runtime_decisions(trace)]
    try:
        with traced(trace):
            done = spec.build()
    finally:
        for restore_one in restore:
            restore_one()
    return done, trace


# ---- reference models ------------------------------------------------------

def simpy_packing(fragment_arrivals, *, pack_ticks, processing_ticks):
    """The reference packing. ``fragment_arrivals``: round_key -> list of QC
    delivery ticks (one per fragment). Each round: complete when its last
    fragment has arrived plus controller processing (RFC 815: the hole list
    empties on the last fragment), then a pack service of ``pack_ticks`` when
    the round has more than one fragment, then it is offered to its channel
    in packing order (the channel is replayed by ``fifo_deliveries``).
    Returns round_key -> packed_tick."""
    environment = simpy.Environment()
    packed = {}

    def round_process(round_key, arrivals):
        completion = max(arrivals) + processing_ticks
        yield environment.timeout(completion)
        if len(arrivals) > 1:
            yield environment.timeout(pack_ticks)
        packed[round_key] = environment.now

    for round_key, arrivals in fragment_arrivals.items():
        environment.process(round_process(round_key, arrivals))
    environment.run()
    return packed


def fifo_deliveries(sends, *, bits_per_us, propagation_ticks):
    """ns-3 point-to-point over a channel: sends = [(send_tick, bits)] in
    order; returns the receive ticks."""
    replay = ns3_point_to_point(sends, bits_per_us=bits_per_us, propagation_ticks=propagation_ticks)
    return [receive for (_, _, receive) in replay]


def simpy_protected_region(start_tick, cadence, end_tick):
    """A SimPy periodic process: boundaries at start + k cadence, one round
    each, sealed at the boundary equal to the end tick."""
    environment = simpy.Environment(initial_time=start_tick)
    boundaries = []

    def region():
        while True:
            yield environment.timeout(cadence)
            boundaries.append(environment.now)
            if environment.now >= end_tick:
                return
    environment.process(region())
    environment.run()
    return boundaries


# ---- helpers on a completed run --------------------------------------------

def ledger_rows(done, path):
    return [row for row in done.result.link_traffic["transfers"] if row["path"] == path]


def channel_rate(done, path):
    """(bits_per_us or None, propagation_ticks) of the channel carrying path."""
    fabric = done.syndrome_packing.links.snapshot()
    edge = next(edge for edge in fabric.edges if edge.path.value == path)
    channel = next(ch for ch in fabric.channels if ch.alias == edge.physical_alias)
    capacity = channel.config.capacity
    rate = None if capacity is None else capacity.aggregate_bits_per_us
    return rate, channel.config.propagation_latency_ticks


# ---- part 1: syndrome packing ----------------------------------------------

def check_packing(done, trace, *, pack_ticks, processing_ticks):
    """Compare packing and publication ticks of every round with the reference."""
    problems = []
    arrivals = {}
    for row in ledger_rows(done, "qc"):
        round_key = _row_key(row)
        arrivals.setdefault(round_key, []).append(row["delivery_ticks"])
    packed_reference = simpy_packing(arrivals, pack_ticks=pack_ticks,
                                     processing_ticks=processing_ticks)
    checks = 0
    for round_key, tick in trace.packed.items():
        key = _plain_key(round_key)
        if key not in packed_reference:
            problems.append(("no reference for packed round", key))
            continue
        if packed_reference[key] != tick:
            problems.append(("packed tick", key, tick, packed_reference[key]))
        checks += 1
    # publication over C2B, in packing order, ns-3 FIFO
    c2b_rows = ledger_rows(done, "c2b")
    if c2b_rows:
        rate, propagation = channel_rate(done, "c2b")
        c2b_rows.sort(key=lambda r: (r["send_ticks"], r["physical_sequence"]))
        sends = [(r["send_ticks"], r["payload_bits"] or 0) for r in c2b_rows]
        expected = fifo_deliveries(sends, bits_per_us=rate, propagation_ticks=propagation)
        for row, receive in zip(c2b_rows, expected):
            key = _row_key(row)
            published = trace.published.get(_match_key(trace.published, key))
            if published != receive:
                problems.append(("publication tick", key, published, receive))
            if row["send_ticks"] != packed_reference.get(key):
                problems.append(("c2b send is not the packing tick", key, row["send_ticks"],
                                 packed_reference.get(key)))
            checks += 1
    else:
        for round_key, tick in trace.published.items():
            key = _plain_key(round_key)
            if packed_reference.get(key) != tick:
                problems.append(("publication without c2b", key, tick, packed_reference.get(key)))
            checks += 1
    # feedback-memory rounds over CWD (attributed per round, window_id None)
    cwd_rows = [r for r in ledger_rows(done, "cwd") if r["attribution"]["window_id"] is None]
    if cwd_rows:
        rate, propagation = channel_rate(done, "cwd")
        all_cwd = ledger_rows(done, "cwd")
        all_cwd.sort(key=lambda r: (r["send_ticks"], r["physical_sequence"]))
        sends = [(r["send_ticks"], r["payload_bits"] or 0) for r in all_cwd]
        expected = dict(zip((id(r) for r in all_cwd),
                            fifo_deliveries(sends, bits_per_us=rate, propagation_ticks=propagation)))
        delivered = {}
        for tick, round_key in trace.feedback_delivered:
            delivered.setdefault(_plain_key(round_key), []).append(tick)
        for row in cwd_rows:
            key = _row_key(row)
            ours = delivered.get(key, [])
            if expected[id(row)] not in ours:
                problems.append(("feedback delivery tick", key, ours, expected[id(row)]))
            checks += 1
    return checks, problems


def _row_key(row):
    return (_freeze(row["attribution"]["operation_id"]), row["attribution"]["round_lo"])


def _plain_key(round_key):
    operation_id, round_index = round_key
    return (_json_id(operation_id), round_index)


def _json_id(value):
    """The ledger's JSON identity, made hashable."""
    from decsim.message import stable_identity_json
    return _freeze(stable_identity_json(value))


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _match_key(mapping, key):
    for candidate in mapping:
        if _plain_key(candidate) == key:
            return candidate
    return None


# ---- part 2: feedback streams ----------------------------------------------

def check_protected_regions(done, trace, spec_regions):
    """Every protected region against the SimPy periodic process."""
    problems, checks = [], 0
    issue_tick = {operation_id: tick for tick, operation_id, _ in trace.issued}
    controller = done.controller
    for region in spec_regions:
        start_tick = issue_tick[region.start_operation_id]
        cadence = controller._resolved_patches[region.patch_id].round_ticks
        seal = next((s for s in trace.sealed if s[1] == region.stream_id), None)
        if seal is None:
            problems.append(("region never sealed", region.stream_id))
            continue
        seal_tick, _, sealed_rounds = seal
        boundaries = simpy_protected_region(start_tick, cadence, seal_tick)
        emitted = [(tick, global_round) for tick, stream_id, global_round, _ in trace.stream_rounds
                   if stream_id == region.stream_id]
        expected = list(zip(boundaries, range(1, len(boundaries) + 1)))
        if emitted != expected:
            problems.append(("stream rounds", region.stream_id, emitted[:6], expected[:6]))
        if sealed_rounds != len(boundaries):
            problems.append(("sealed round count", region.stream_id, sealed_rounds, len(boundaries)))
        if seal_tick != boundaries[-1]:
            problems.append(("seal off boundary", region.stream_id, seal_tick, boundaries[-1]))
        # every operation on the live patch, other than the region's start, began on a boundary
        for tick, operation_id, patches in trace.issued:
            on_patch = region.patch_id in patches
            live = start_tick < tick <= seal_tick
            if on_patch and live and operation_id != region.start_operation_id:
                if tick not in boundaries:
                    problems.append(("held operation started off boundary", operation_id, tick))
                checks += 1
        checks += 3
    return checks, problems


def check_bump_allocation(done, trace):
    """Unprotected stream bindings: an operation that declares an offset keeps
    it; one that declares none binds at the stream's next free round, which is
    the largest end of the segments bound before it or of the idle rounds the
    stream was extended by (ExtendStream) before that tick."""
    problems, checks = [], 0
    resolved = done.controller._resolved_operations
    protected = set(getattr(done.controller.streams, "_stream_owner_by_id", {}))
    segment_ends = {}
    for tick, operation_id, stream_id, offset in trace.bindings:
        if stream_id in protected:
            continue
        extended_to = max((global_round for round_tick, extended_stream, global_round, _
                           in trace.stream_rounds
                           if extended_stream == stream_id and round_tick <= tick), default=0)
        expected_offset = max(segment_ends.get(stream_id, 0), extended_to)
        declared = _declared_offset(done, operation_id)
        if declared is None and offset != expected_offset:
            problems.append(("bump offset", operation_id, offset, expected_offset))
        checks += 1
        if operation_id in resolved:
            end = offset + resolved[operation_id].round_count
            segment_ends[stream_id] = max(segment_ends.get(stream_id, 0), end)
    return checks, problems


def _declared_offset(done, operation_id):
    operations = getattr(done.execution_runtime, "operations", {})
    operation = operations.get(operation_id)
    return None if operation is None else operation.stream_offset


# ---- part 3: controller proper ---------------------------------------------

def check_idle_decode_demand(done, trace):
    """SeparateDecodeJobs: one load-only job of rcom + rbuf rounds per full
    commit region of idle rounds on a patch (Skoric et al. sliding window)."""
    problems = []
    geometry = done.controller._resolved_patches
    by_patch = {}
    for tick, op_id, patch, round_index in trace.idle_rounds:
        by_patch.setdefault(patch, []).append(round_index)
    expected_jobs = 0
    for patch, rounds in by_patch.items():
        commit = geometry[patch].code_geometry.commit_round_count
        expected_jobs += sum(1 for r in rounds if r % commit == 0)
    if len(trace.idle_demands) != expected_jobs:
        problems.append(("idle decode jobs", len(trace.idle_demands), expected_jobs))
    for tick, rounds, label in trace.idle_demands:
        patch = next(iter(by_patch))
        code = geometry[patch].code_geometry
        if rounds != code.commit_round_count + code.buffer_round_count:
            problems.append(("idle job size", rounds, code.commit_round_count + code.buffer_round_count))
    if done.controller.idle_rounds_emitted != len(trace.idle_rounds):
        problems.append(("idle rounds counted", done.controller.idle_rounds_emitted, len(trace.idle_rounds)))
    return len(trace.idle_demands) + 1, problems


def check_releases(done, trace):
    """Each release: OC sent at the decision tick, CQ sent at OC delivery,
    the runtime instructed at CQ delivery, both channels ns-3 FIFO."""
    problems, checks = [], 0
    oc_rows = ledger_rows(done, "oc")
    cq_rows = ledger_rows(done, "cq")
    for path, rows in (("oc", oc_rows), ("cq", cq_rows)):
        rows.sort(key=lambda r: (r["send_ticks"], r["physical_sequence"]))
        rate, propagation = channel_rate(done, path)
        sends = [(r["send_ticks"], r["payload_bits"] or 0) for r in rows]
        for row, receive in zip(rows, fifo_deliveries(sends, bits_per_us=rate, propagation_ticks=propagation)):
            if row["delivery_ticks"] != receive:
                problems.append((f"{path} fifo", row["send_ticks"], row["delivery_ticks"], receive))
            checks += 1
    decided = sorted(trace.decided)
    instructed = sorted(trace.instructed)
    if len(oc_rows) != len(decided) or len(cq_rows) != len(decided):
        problems.append(("one OC and one CQ transfer per decision", len(oc_rows), len(cq_rows), len(decided)))
    for (decision_tick, target), oc, cq, (instructed_tick, instructed_target) in zip(
            decided, oc_rows, cq_rows, instructed):
        if oc["send_ticks"] != decision_tick:
            problems.append(("oc send is decision tick", target, oc["send_ticks"], decision_tick))
        if cq["send_ticks"] != oc["delivery_ticks"]:
            problems.append(("cq send is oc delivery", target, cq["send_ticks"], oc["delivery_ticks"]))
        if instructed_tick != cq["delivery_ticks"] or instructed_target != target:
            problems.append(("instructed at cq delivery", target, instructed_tick, cq["delivery_ticks"]))
        checks += 3
    return checks, problems


# ---- scenarios -------------------------------------------------------------

def stim_memory_with_c2b():
    import stim
    from decsim.links.link_profiles import bandwidth_limited_profile, with_controller_to_buffer_edge
    from decsim.decoders.decoders import PresetLatencyDecoder
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.qpu.stim_device import StimDevice
    from decsim.run_spec import RunSpec
    rounds = 12
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=rounds, distance=3,
                                     after_clifford_depolarization=0.003)
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    links = with_controller_to_buffer_edge(bandwidth_limited_profile(capacity_scale=0.25),
                                           latency_us=0.1, aggregate_bits_per_us=8.0, source="gate7 c2b")
    return RunSpec(ops=[op], d=3, rounds_policy=FixedRounds(rounds), device=StimDevice(),
                   decoder=PresetLatencyDecoder(1.0), num_units=1, links=links, seed=3)


def timing_of(done):
    """(t_pack ticks, binary availability ticks) the run used."""
    return done.syndrome_packing.t_pack, done.controller.binary_availability_ticks


def main(argv=None) -> None:
    lines = ["# Gate 7: controller vs SimPy / ns-3 / RFC 815 references", ""]
    total_checks, total_problems = 0, []

    def report(title, checks, problems):
        nonlocal total_checks
        total_checks += checks
        total_problems.extend((title, p) for p in problems)
        state = "agree" if not problems else f"{len(problems)} disagreements"
        lines.append(f"- {title}: {checks} checks, {state}")

    lines.append("## 1. Syndrome packing")
    for title, make in (("QLX mem_surface program", lock.qlx_multi_fragment),
                        ("two-fragment stream", lock.two_fragment_stream),
                        ("Stim memory with a priced C2B hop", stim_memory_with_c2b),
                        ("feedback chain, both routes", lock.feedback_chain_trailing_buffer),
                        ("feedback chain, extend-stream idle rounds", lock.feedback_chain_extend_stream_fallback)):
        done, trace = run_traced(make())
        pack_ticks, processing_ticks = timing_of(done)
        report(title, *check_packing(done, trace, pack_ticks=pack_ticks,
                                     processing_ticks=processing_ticks))

    lines.append("")
    lines.append("## 2. Feedback streams")
    for title, make in (("protected chain, reopen", lock.protected_chain_reopen),
                        ("protected chain, feedback", lock.protected_chain_feedback)):
        spec = make()
        done, trace = run_traced(spec)
        report(title + ": periodic boundaries, rounds, seal, held starts",
               *check_protected_regions(done, trace, spec.protected_regions))
    for title, make in (("live stream pair", lock.live_stream_extend),
                        ("two-fragment stream", lock.two_fragment_stream),
                        ("QLX mem_surface program", lock.qlx_multi_fragment)):
        done, trace = run_traced(make())
        report(title + ": bump allocation of stream rounds", *check_bump_allocation(done, trace))

    lines.append("")
    lines.append("## 3. Controller")
    done, trace = run_traced(lock.feedback_chain_separate_decode_jobs())
    report("separate decode jobs: one job per commit region of idle rounds",
           *check_idle_decode_demand(done, trace))
    for title, make in (("feedback chain (fixed-latency links)", lock.feedback_chain_trailing_buffer),
                        ("protected chain, feedback", lock.protected_chain_feedback)):
        done, trace = run_traced(make())
        report(title + ": OC then CQ release timing", *check_releases(done, trace))

    verdict = "PASS" if not total_problems else "FAIL"
    lines.append("")
    lines.append(f"Verdict: {verdict}. {total_checks} checks, {len(total_problems)} disagreements.")
    for title, problem in total_problems[:40]:
        lines.append(f"  - {title}: {problem}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
