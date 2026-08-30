# Initialization contract

One resolved workload creates two setup views. This file records the
order that must hold and why; the narrative walkthrough lives in
`docs/architecture/INITIALIZATION_FLOW.md`.

## One source, two views

`resolve_run_configuration` (decsim/run_configuration.py) is the single
compiler step. From one validated workload it derives:

- the execution view: `config.ops` (private copies), the dependency DAG,
  resource claims, feedback blockers. Consumed by ExecutionRuntime.
- the decoding view: `plan` from `_plan_execution` (WindowPlan,
  BufferingPlan, code geometry, resolved round counts and cadence) plus
  `planned_operations` (the static decode plan: `decode_ops` when a
  static plan is selected, else the detector-emitting ops not bound to a
  dynamic stream). Consumed by WindowManager.

`_validate_workload_identity` (decsim/frontends/planner.py) guarantees
each operation id maps to exactly one object across ops, decode_ops and
dynamic_streams, and that a dynamic stream never doubles as either.

## The order that must hold (RunSpec._build_once)

1. resolve, construct, connect, seed. Nothing touches the engine.
2. `window_manager.register_op(op)` for every `planned_operations` op.
3. `window_manager.load_execution_plan(plan.execution, plan.buffering)`:
   installs windows and boundary states, builds window error models,
   validates Buffer 0 and SB1 capacity against the plan's minimum live
   rounds, registers planned weak and potential holds.
4. `window_manager._register_dynamic_stream(...)` for every dynamic
   stream.
5. `controller.load_program(ExecutionProgram(...))`: FeedbackStreams
   load, `register_op` for every program operation, then
   `ExecutionRuntime.load_program` starts root operations.
6. `engine.run()`.

Invariant: no operation that emits syndrome data can start before its
registration, its window plan (or dynamic-stream policy), and the
buffer-capacity validation, because roots start only inside step 5 and
the engine only runs in step 6.

## Duplicate registration is intentional; both loops are load-bearing

- Step 2 must precede step 3: `_register_planned_holds` validates every
  hold's referenced operation is open on Buffer 0
  (SyndromeBuffer.register_hold refuses closed references), and
  `register_op` is what opens the account. With a static decode plan the
  planned set is disjoint from the program set, so step 5 alone would
  never register it.
- Step 5 must register the full program: non-emitting and stream-bound
  operations also need `rounds_arrived` / `memory_rounds` accounts for
  idle and memory rounds.
- The overlap (emitting ops in the default plan) registers twice,
  safely: one object per id (identity validation) and the
  `if op.id not in self._ops` guard in `register_op` make the second
  call a no-op apart from rewriting the same mapping entry.
- Removing either loop changes behavior; neither is dead. Verdict: KEEP
  both, with the deterministic setup tests pinning idempotence.

## Dynamic streams have a genuinely separate path

`_register_dynamic_stream` resolves the stream's feedback mode,
registers it with the DynamicWindows lifecycle, obtains the source round
limit from the error-model provider, and pre-plans finite geometries
when the source is bounded. None of that exists in the static plan
install.
