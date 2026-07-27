#==================================================================
# MODULARITY CONFORMANCE: the standard wiring runs end to end with the
# seams replaced by minimal from-scratch implementations that know nothing
# about the defaults. If a future change breaks an extension point, this
# fails first. (Per-port protocol conformance of the SHIPPED defaults lives
# in test_port_conformance.py.)
#==================================================================
from decsim.config import us
from decsim.message import DecodeResult, Decision, Operation, SyndromePayload
from decsim.planner import WindowPlanner, FixedRounds
from decsim.schemes import SlidingWindowScheme
from decsim.layouts import UniformLayout
from decsim.codes import SurfaceCodeModel
from decsim.run_spec import RunSpec, simulate
from decsim.decoders import PerRoundDecoder


# ---- a researcher's from-scratch stack: no defaults, port surface only ----

class MyDevice:
    def begin_operation(self, op):
        return None

    def round_payloads(self, op, r):
        return [SyndromePayload(op.id, op.patches[0], r)]

    def idle_round_payloads(self, op, stream_id, global_round, patch):
        return [SyndromePayload(stream_id, patch, global_round)]

    def register_dynamic_stream(self, stream_op, round_count, *, belief_matching=False):
        return None

    def validate_stream_length(self, stream_op, stream_round_count):
        return None

    def window_models_for_operation(self, op, windows, round_count,
                                    *, belief_matching=False):
        return []

    def window_model_for_stream(self, stream_id, window, *, is_last):
        return None

    def strong_window_model_for_operation(self, op, window, round_count,
                                          *, belief_matching=False,
                                          exclude_faults_touching=None):
        return None

class MyCode:
    name = "my-code"
    distance = 3
    def rounds_per_logical_cycle(self): return 3
    def commit_rounds(self): return 3
    def buffer_rounds(self): return 3
    def buffering_floor(self, scheme=None): return (3, 3)
    def spatial_nodes(self, n): return 9 * max(1, n)
    def syndrome_bits_per_round(self, n): return 8 * max(1, n)

class MyLayout:
    def __init__(self, code): self.code = code
    name = "my-layout"
    distance = 3
    def code_for_patch(self, p): return self.code
    def code_for_op(self, op): return self.code
    def spatial_nodes_for(self, op): return self.code.spatial_nodes(len(op.qubits))
    def codes(self): return [self.code]
    def resources_for(self, op): return []

class MyScheme:
    """One window per op committing everything.

    The scheme has no custom dependency hook, so the planner's chain fallback applies
    trivially to a single window.
    """
    def plan_windows(self, op_id, round_count, code):
        return [(1, round_count, round_count)]

    def data_complete(self, window, rounds_arrived, successor_rounds, memory_rounds,
                      round_count, has_successor, op=None, layout=None):
        return rounds_arrived >= window.commit_hi

    def validate_buffer(self, code):
        return None

class MyRounds:
    def rounds_for(self, op, code): return 7

class MyDecoder:
    def __init__(self): self.decodes = 0
    def latency(self, job): return us(0.2)
    def decode(self, job):
        self.decodes += 1
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(1,))

class MyScheduler:
    """LIFO -- a genuinely different policy than the default FIFO."""
    def insert(self, queue, job): queue.append(job)
    def pop(self, queue): return queue.pop()

class MyDeadline:
    def deadline(self, op, window, now, on_reaction_path): return now

class MyRouter:
    def __init__(self, decoder): self.decoder = decoder; self.calls = 0
    def route(self, job):
        self.calls += 1
        return self.decoder

class MyController:
    def __init__(self, engine): self.engine = engine
    def relay_syndrome(self, payload, deliver):
        self.engine.schedule(us(0.1), lambda: deliver(payload))
    def relay_instruction(self, decision, deliver):
        self.engine.schedule(us(0.1), lambda: deliver(decision))

class MyOrchestrator:
    """From-scratch Orchestrator: releases blocked ops without an effect."""
    def __init__(self):
        self.blocked = {}; self.controller = None; self.sink = None
        self.integrated = 0
    def connect(self, controller, decision_sink):
        self.controller = controller; self.sink = decision_sink
    def register_blocked_operation(self, blocked_op_id, blocking_op_id):
        self.blocked.setdefault(blocking_op_id, []).append(blocked_op_id)
    def integrate(self, op, result):
        self.integrated += 1
        for decision in self.on_result(op, result):
            self.controller.relay_instruction(decision, self.sink)
    def on_result(self, op, result):
        return [Decision(blocked_id)
                for blocked_id in self.blocked.pop(op.id, [])]

class MyFactory:
    def __init__(self, engine):
        self.engine = engine
        self.requests = 0
    def request(self, op_id, callback):
        self.requests += 1; callback()
    def shutdown(self):
        return None

class MyMetric:
    name = "my_events"
    def __init__(self): self.count = 0
    def observe(self, engine): self.count += 1
    def result(self): return self.count


def _blocked_ops():
    """CNOT, then a T whose decode releases a second T."""
    a = Operation(0, "CNOT(q0,q1)", (0, 1), clifford=True)
    b = Operation(1, "T(q1)", (1,), clifford=False)
    c = Operation(2, "T2(q1)", (1,), clifford=False, blocked_by=1)
    a.patches, b.patches, c.patches = (0, 1), (1,), (1,)
    b.predecessors, c.predecessors = (0,), (1,)
    a.has_successor = b.has_successor = True
    return [a, b, c]


def test_every_seam_accepts_a_from_scratch_implementation():
    """The standard wiring runs end to end with the seams replaced by the
    from-scratch stack above (device, code, layout, scheme, rounds, decoder,
    router, scheduler, deadline policy, controller, orchestrator, factory, metric
    -- all at once), with assertions that each custom piece participated."""
    decoder, metric = MyDecoder(), MyMetric()
    orchestrator = MyOrchestrator()
    router = MyRouter(decoder)
    code = MyCode()
    built_factories = []

    def make_factory(engine, cluster):
        factory = MyFactory(engine)
        built_factories.append(factory)
        return factory

    r = simulate(RunSpec(
            ops=_blocked_ops(),
            num_units=2,
            device=MyDevice(),
            layout=MyLayout(code),
            scheme=MyScheme(),
            rounds_policy=MyRounds(),
            decoder=decoder,
            router=router,
            scheduler=MyScheduler(),
            deadline_policy=MyDeadline(),
            make_controller=MyController,
            orchestrator=orchestrator,
            make_factory=make_factory,
            make_metrics=lambda e, cl, ch, f: [metric],
        ), verbose=False)
    factory = built_factories[0]
    chip = r["chip"]
    assert len(chip.done_bodies) == 3          # all ops ran, including the blocked T
    assert decoder.decodes >= 3                # the custom decoder decoded every window
    assert router.calls >= 3                   # routed per job
    assert factory.requests == 2               # both T gates drew a state
    assert metric.count > 0                    # the custom metric observed events
    assert orchestrator.integrated == 3        # the custom orchestrator saw every result
    assert r["fully_done"] > r["chip_done"] >= 0


def test_planner_parameter_swaps_the_planning_algorithm():
    """planner= replaces the WindowPlanner in the standard wiring."""
    class CountingPlanner:
        def __init__(self, inner):
            self.inner = inner
            self.scheme = inner.scheme
            self.layout = inner.layout
            self.rounds_policy = inner.rounds_policy
            self.calls = 0
        def plan(self, ops):
            self.calls += 1
            return self.inner.plan(ops)
    code = SurfaceCodeModel(d=3)
    planner = CountingPlanner(WindowPlanner(
        SlidingWindowScheme(), UniformLayout(code), FixedRounds(11)))
    r = simulate(RunSpec(
            ops=_blocked_ops(),
            planner=planner,
            decoder=PerRoundDecoder(tau_us=1.0),
        ), verbose=False)
    assert planner.calls == 1                  # OUR planner produced the plan
    assert r["cluster"].planner is planner
    assert r["cluster"].scheme is planner.scheme
    assert r["fully_done"] > 0
