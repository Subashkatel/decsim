# Initialization and program flow: findings

Companion to `docs/architecture/INITIALIZATION_FLOW.md` (the reference
walkthrough); this report records the questions the stabilization pass
was asked and the evidence that answers them. Verified at 65660a0 and
retested after the safe cleanups.

## Why are some operations registered before load_execution_plan and again in Controller.load_program?

Both passes are load-bearing; neither can be removed without changing
behavior.

- The early pass registers `planned_operations` (the decode-plan view)
  because `load_execution_plan` immediately registers the plan's weak
  and potential holds, and `SyndromeBuffer.register_hold` refuses a
  hold that references an operation not yet opened on the store
  (syndrome_buffer.py:302-307). `register_op` performs that opening.
- The `Controller.load_program` pass registers `program.operations`
  (the execution view) because non-emitting and stream-bound operations
  also need `rounds_arrived` / `memory_rounds` accounts for idle and
  feedback-memory rounds, and because with a static decode plan the two
  sets are disjoint.

## Is duplicate registration intentional and idempotent?

Yes. `_validate_workload_identity` (frontends/planner.py:175-200)
guarantees one object per operation id across the three workload roles,
and `register_op` guards its account initialization with
`if op.id not in self._ops` (window_manager.py:143-146), so the second
registration of an overlapping operation rewrites the same mapping
entry with the same object and touches nothing else. Pinned by
`tests/21_stabilization/test_initialization_determinism.py::test_registration_is_idempotent`.

## Do planned_operations and program.operations contain the same set?

Not in general. They are equal exactly when no static decode plan is
selected, there are no dynamic streams, and every program operation
emits detector data. With `spec.decode_ops` set, `planned_operations`
is the decode_ops tuple (disjoint from the program); with dynamic
streams, stream-bound emitters are excluded from the plan
(run_configuration.py:241-251).

## Do dynamic streams require a separate registration path?

Yes. `_register_dynamic_stream` (window_manager.py:151-203) resolves
the stream's feedback mode, hands window creation to the DynamicWindows
lifecycle, and obtains a source round limit from the error-model
provider; none of that exists in the static plan install.

## Can one path be removed without changing behavior?

No. Removing the early pass makes `load_execution_plan` raise
("hold ... references closed operation") for every planned hold and
leaves static decode plans unregistered. Removing the controller pass
leaves non-emitters and stream-bound operations without accounts
(KeyError on their first idle or memory round) and, under a static
decode plan, registers no program operation at all. The mutation matrix
exercises the second removal (M03): the setup suite detects it.

## Ordering guarantees

- Static: roots start only inside `ExecutionRuntime.load_program`,
  which runs strictly after registration, plan install, and buffer
  validation, and nothing executes before `engine.run()`
  (run_spec.py:335-346). Mutation M01 (installing the plan only after
  the run) is detected by the stabilization suite.
- Dynamic: streams are registered at step 7, before
  `controller.load_program` at step 8.
- Scheduled starts: a `scheduled_start_round` releases the operation on
  exactly that cycle boundary
  (`test_scheduled_start_round_delays_the_root`).

## Verdict

The one-source two-view flow is sound as built. KEEP both registration
passes; the contract is recorded in
`validation/responsibility_audit_2026_08_30/initialization_contract.md`.
