# Initialization flow

Verified against `decsim/run_spec.py` at commit 65660a0. The contract
form of this document is
`../validation/responsibility_audit_2026_08_30/initialization_contract.md`.

## One source, two views

```
                   SOURCE WORKLOAD (ops= or frontend=)
                              |
                              v
             resolve_run_configuration (run_configuration.py)
        copies + validates workload, selects every default,
        plans windows; nothing here touches the engine
                              |
              +---------------+----------------+
              |                                |
              v                                v
      execution view                    decoding view
      config.ops (private copies)       plan = _plan_execution(...)
      dependency DAG (validated)          plan.execution   (WindowPlan)
      resource_claims (layout)            plan.buffering   (BufferingPlan)
      scheduled starts, blockers          plan.code_geometry
      config.resource_claims              plan.resolved_operations
              |                           config.planned_operations
              v                                |
      ExecutionRuntime                         v
      (program sequencer)               WindowManager
```

`planned_operations` (the decode plan) is `decode_ops` when a static
decode plan is selected, otherwise the detector-emitting ops not bound
to a dynamic stream (`_decode_plan_operations`,
run_configuration.py:241-251). `_validate_workload_identity`
(frontends/planner.py:175-200) guarantees one object per operation id
across ops, decode_ops, and dynamic_streams.

## Exact call order (RunSpec._build_once, run_spec.py:184-346)

1. `resolve_run_configuration(self, root_seed)` (:200).
2. Construct: ConditionalRelease (:202), links (:205), SyndromeBuffer
   (:206), SyndromeBuffer1 only when the policy requires strong context
   or the primary tier is STRONG (:209-215), PauliFrame (:216),
   WindowManager (:221), SyndromePacking (:242), DecoderManager (:259),
   factory (:276), QPUDevice (:282), FeedbackStreams (:285), Controller
   (:292), ExecutionRuntime (:301).
3. Connect: window manager to decoder manager closures (:237-241,
   :273-275), `controller.connect_runtime` (:304), QPU receivers
   (:305-307). Seed everything (`bind_run_seed`, :315).
4. `conditional_release.connect(...)` and blocked-operation
   registration (:331-334).
5. `window_manager.register_op(op)` for every planned operation
   (:335-336).
6. `window_manager.load_execution_plan(plan.execution, plan.buffering)`
   (:337): installs the compile-time windows and their initial boundary
   states, builds window error models, validates Buffer 0 and SB1
   capacity against the plan's minimum live rounds, and registers the
   planned weak and potential holds.
7. `window_manager._register_dynamic_stream(stream, resolved)` for each
   dynamic stream (:339-340).
8. `controller.load_program(ExecutionProgram(ops, decode_ops,
   dynamic_streams, protected_regions))` (:343-345). Inside
   (controller/controller.py:52-58): `streams.load(program)`, then
   `window_manager.register_op(operation)` for EVERY program operation,
   then `runtime.load_program(program)`.
9. `ExecutionRuntime.load_program` (frontends/execution_runtime.py:
   101-122): index operations, build the dependency graph, schedule
   `scheduled_start_round` releases, then `_attempt_start` every root.
   A root that can start claims resources and reaches
   `controller.issue_operation`, which commands the QPU for the next
   cycle boundary.
10. `engine.run()` (:346). Post-run settledness checks: pending
    escalations, decoder work, packing contexts, SB1 (:347-356).

## Critical invariant

No operation that emits syndrome data starts before its operation or
stream is registered with the WindowManager, its static WindowPlan is
installed (or its dynamic-stream policy registered), and
buffer-capacity validation has succeeded. This holds statically: roots
start only inside step 8/9, strictly after steps 5-7, and the engine
runs only in step 10.

## Why operations are registered twice

Both registration passes are load-bearing; the overlap is idempotent.

- Step 5 must precede step 6: `_register_planned_holds` validates every
  hold's referenced operation is open on the store
  (syndrome_buffer.py:302-307), and `register_op` performs that opening
  (window_manager.py:141-149). With a static decode plan the planned
  set is disjoint from the program set, so step 8 alone would never
  register it.
- Step 8 must register the whole program: non-emitting and stream-bound
  operations also need `rounds_arrived` and `memory_rounds` accounts
  for idle and feedback-memory rounds.
- The overlap registers twice safely: one object per id, and the
  `if op.id not in self._ops` guard makes the second call a no-op apart
  from rewriting the same mapping entry. Deterministic setup tests pin
  this (`tests/21_stabilization/test_initialization_determinism.py`).

Dynamic streams use `_register_dynamic_stream`
(window_manager.py:151-203) because their windows are created at
runtime: the stream's feedback mode is resolved, the DynamicWindows
lifecycle takes over window creation, and the error-model provider may
bound the stream's source rounds.
