#==================================================================
# MODULARITY CONFORMANCE: the standard wiring runs end to end with the
# seams replaced by minimal from-scratch implementations that know nothing
# about the defaults. If a future change breaks an extension point, this
# fails first. (Per-port protocol conformance of the SHIPPED defaults lives
# in test_port_conformance.py.)
#==================================================================
from decsim.config import us
from decsim.message import (
    DecodeResult,
    Decision,
    Operation,
    OperationWindowPlan,
    RetainedSyndromeFragment,
    SyndromePayload,
    SyndromeRoundPacket,
    WindowGeometry,
)
from decsim.planner import FixedRounds
from decsim.schemes import SlidingWindowScheme
from decsim.layouts import UniformLayout
from decsim.codes import SurfaceCodeModel
from decsim.run_spec import RunSpec, simulate
from decsim.decoders import PerRoundDecoder
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
from decsim.pauli_frame import PauliFrame


# ---- a researcher's from-scratch stack: no defaults, port surface only ----

class MyDevice:
    operation_circuit_scope = "none"

    def begin_operation(self, op, resolved_round_count):
        return None

    def round_payloads(self, op, r):
        return [SyndromePayload(op.id, op.patches[0], r)]

    def idle_round_payloads(self, op, stream_id, global_round, patch):
        return [SyndromePayload(stream_id, patch, global_round)]

    def register_dynamic_stream(
        self, stream_op, round_count, *, fault_model_requirement,
    ):
        return None

    def validate_stream_length(self, stream_op, stream_round_count):
        return None

    def window_models_for_operation(self, op, windows, round_count,
                                    *, fault_model_requirement):
        return []

    def window_model_for_stream(self, stream_id, window, *, is_last):
        return None

    def strong_window_model_for_operation(self, op, window, round_count,
                                          *, fault_model_requirement,
                                          exclude_faults_touching=None):
        return None

class MyCode:
    name = "my-code"
    distance = 3
    def rounds_per_logical_cycle(self): return 3
    def round_period_us(self): return None
    def commit_rounds(self): return 3
    def buffer_rounds(self): return 3
    def buffering_floor(self): return (3, 3)
    def buffer_floor_override_active(self): return False
    def spatial_nodes(self, n): return 9 * max(1, n)
    def syndrome_bits_per_round(self, n): return 8 * max(1, n)

class MyLayout:
    def __init__(self, code): self.code = code
    name = "my-layout"
    distance = 3
    def code_for_patch(self, p): return self.code
    def code_for_op(self, op): return self.code
    def spatial_nodes_for(self, op, *, base_spatial_node_count):
        return base_spatial_node_count
    def patch_spatial_nodes_for(self, patch, *, base_spatial_node_count):
        return base_spatial_node_count
    def codes(self): return [self.code]
    def resources_for(self, op): return []

class MyScheme:
    """One window per operation committing everything."""

    def plan_operation(
        self,
        operation_id,
        round_count,
        *,
        commit_round_count,
        buffer_round_count,
    ):
        return OperationWindowPlan(
            operation_id=operation_id,
            windows=(WindowGeometry(1, 1, round_count, round_count),),
            internal_dependencies=(),
            entry_window_indices=(0,),
            exit_window_indices=(0,),
            windowed=False,
            batch_preceding_idle_rounds=False,
        )

    def data_complete(
        self,
        window,
        *,
        rounds_arrived,
        successor_rounds,
        memory_rounds,
        round_count,
        has_successor,
        operation,
    ):
        return rounds_arrived >= window.commit_hi

    def validate_buffer(self, geometry):
        return None

class MyRounds:
    def rounds_for(self, op, code): return 7

class MyDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self): self.decodes = 0
    def latency(self, job): return us(0.2)
    def decode(self, job):
        self.decodes += 1
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(1,))

class MyScheduler:
    """LIFO -- a genuinely different policy than the default FIFO."""
    def insert(self, queue, job): queue.append(job)
    def pop(self, queue, now_ticks): return queue.pop()

class MyDeadline:
    def deadline(self, op, window, now, on_reaction_path): return now

class MyRouter:
    def __init__(self, decoder):
        self.decoder = decoder
        self.calls = 0

    def route(self, job):
        self.calls += 1
        return self.decoder

    def fault_model_requirement_for(self, code):
        return self.decoder.fault_model_requirement

class MyController:
    def __init__(self, engine, links):
        self.engine = engine
        self.links = links
    def relay_syndrome(self, payload, deliver):
        packet = SyndromeRoundPacket(
            operation_id=payload.operation_id,
            round_index=payload.round_index,
            fragments=(RetainedSyndromeFragment.from_payload(payload),),
        )
        self.engine.schedule(us(0.1), lambda: deliver(packet))
    def relay_instruction(self, decision, deliver):
        self.engine.schedule(us(0.1), lambda: deliver(decision))

class MyOrchestrator:
    """From-scratch Orchestrator: releases blocked ops without an effect."""
    def __init__(self, engine):
        self.engine = engine
        self.blocked = {}; self.controller = None; self.sink = None
        self.integrated = 0
        self.frame = PauliFrame()
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
    result_schema_version = 1
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
    orchestrators = []
    router = MyRouter(decoder)
    code = MyCode()
    built_factories = []

    def make_factory(engine, cluster):
        factory = MyFactory(engine)
        built_factories.append(factory)
        return factory

    def make_orchestrator(engine):
        orchestrator = MyOrchestrator(engine)
        orchestrators.append(orchestrator)
        return orchestrator

    r = simulate(RunSpec(
            ops=_blocked_ops(),
            num_units=2,
            device=MyDevice(),
            layout=MyLayout(code),
            scheme=MyScheme(),
            rounds_policy=MyRounds(),
            router=router,
            scheduler=MyScheduler(),
            deadline_policy=MyDeadline(),
            make_controller=MyController,
            make_orchestrator=make_orchestrator,
            make_factory=make_factory,
            make_metrics=lambda e, wm, dm, ch, f: [metric],
        ), verbose=False)
    factory = built_factories[0]
    orchestrator = orchestrators[0]
    chip = r.chip
    assert len(chip.done_bodies) == 3          # all ops ran, including the blocked T
    assert decoder.decodes >= 3                # the custom decoder decoded every window
    assert router.calls >= 3                   # routed per job
    assert factory.requests == 2               # both T gates drew a state
    assert metric.count > 0                    # the custom metric observed events
    assert orchestrator.integrated == 3        # the custom orchestrator saw every result
    assert r.result.fully_done_ticks > r.result.chip_done_ticks >= 0
