# Stabilization result (2026-08-30)

Audited commit: 65660a05ef91913f423843cfa4fdcf48d82d13f5
Actual starting commit: 65660a05 (identical; branch reorganize, tree clean)
Final commit: recorded in git; every change below is its own commit
Working tree clean: yes (untracked experiment results only)

Public-main comparison: origin/main 155a3e9 is an older snapshot of the
same line (merge base b67896f; the 11 main-only commits are the earlier
reorganize-core PR merges; local reorganize is 441 commits past that
base). No divergence blocks this work.
Audit-commit comparison: the audit commit IS the starting HEAD.

## External references

VERIFIED: swiper@1c00e097 (official Toffoli verify 17/17, rerun);
PECOS@0af1893 (cargo decoder-core 124/124 lib + 6 integration + 7
doctests, rerun); Helios@622d85d (own d=3 rsc bench 1000/1000 under
from-source iverilog, rerun); qubic distributed_processor@c22cce81
(official python suite 37 passed, 5 skipped, from its documented cwd;
software slice only, HDL benches not run).
READ_ONLY: XQsim@006c384 (crashes on its own example), QICK@0b1613e
(official tests need a physical ZCU111), RISC-Q@e7686d6 (toolchain
absent, NO license: cite only), qubic_emulator@8cc66588 (10/13 official
tests pass; 3 golden waveform comparisons exceed tolerance on this
host).
NOT REPRODUCED / BLOCKED: none beyond the READ_ONLY reasons above.
Pins, licenses, and commands: validation/responsibility_audit_2026_08_30/
reference_manifest.yaml and reproduction.md.

## Initialization

ExecutionProgram construction: resolve_run_configuration copies and
validates the workload; ExecutionProgram(ops, decode_ops,
dynamic_streams, protected_regions) is built at load (run_spec.py).
WindowPlan construction: _plan_execution inside the resolver.
BufferingPlan construction: same place; capacity validated at plan load.
Program Sequencer load: ExecutionRuntime.load_program, inside
Controller.load_program, strictly after the window-side setup.
WindowManager plan load: register planned ops, load_execution_plan,
register dynamic streams (run_spec.py steps 5-7).
Static ordering guarantee: roots start only in step 8/9; the engine
runs only in step 10. Mutation M01 detected.
Dynamic ordering guarantee: streams registered in step 7, before any
program load.
Duplicate registration verdict: intentional and idempotent; both passes
load-bearing (reports/DECSIM_INITIALIZATION_AND_PROGRAM_FLOW.md).

## Buffer 0

write point: SyndromePacking._finish_packing.
publication point: packing completion, or the CWB landing when priced.
notification: accept_window_input; publication verified at runtime.
readiness owner: WindowManager._count_arrival, weak-primary only.
retention owner: SyndromeBuffer refcounted holds.
release point: decoder-input landing, WBD delivery, or drop-on-arrival.

## Buffer 1

write point: the dual write in _finish_packing, once per round.
CSB behavior: priced when wired; capacity counts in-flight writes.
storage point: _store at the CSB landing.
callback: on_round_stored, strictly after storage.
callback payload: the operation id only; never the packet.
weak-primary behavior: wakes escalation only.
strong-primary behavior: drives window readiness (stored-through).
terminal escalation behavior: submits exactly once at context_hi;
exact round-set check guards the build.
far-boundary behavior: submits on the far weak commit; WSD reservation
must exist; SBD gates on ready_tick.
serial-switching lag behavior: fail-loud ("csb lag beyond the
escalation margin"); preserved, not converted to waiting.
parallel-switching lag behavior: the same fail-loud check fires at
strong-job build when the CSB margin does not hold
(test_parallel_requires_the_csb_margin).
ordered-arrival proof: Proof A chain (one clock, FIFO links, FIFO
engine, constant per-op fragment counts and t_pack); boundary cases
recorded.
gap-handling proof: Proof B (ready_tick raise, exact round-set
equality, retained-payload refusal); pinned by
test_sb1_gap_cannot_be_served and mutation M08.

## Strong request

creation: escalation policy directive at the weak outcome, or
submit_strong in parallel mode.
selection: prepare_strong_selection reserves WSD; ledger
begin_selection/select match exact request keys.
context gate: three exact checks over SB1 (build, exact set, ready_tick).
unit assignment: DecoderManager dispatch; unit assigned at DMA start.
SBD: reserve_transfer clamps to max(link, WSD, last stored round).
memory landing: DecoderInputStaging deposit; input_landed.
compute start: service gate (boundary) then the routed strong decoder.
result matching: StrongRequestLedger; stale or unconsumable results raise.
cancellation: cancel_strong is idempotent across queued, crossing,
running, held.
final delivery: DO transfer, priced frame write, one authoritative
correction per window.

## Safe cleanup

completed: (1) 353 unreachable lines behind the crossing-rephase
NotImplementedError deleted, refusal kept; (2) Controller.round_ticks
and Controller._code_geometry deleted, the single construction site
trimmed, three test files re-pointed at the owning components with
equal assertion strength (the audit's "never read" claim corrected:
core never read them, tests did); (3) five unused window-manager
imports removed; (4) PauliFrameSnapshot.window_count duplicate removed;
(5) the execution-runtime provisional start stamp got its reentry
invariant comment. Each step ran the full suite, the frozen suite, and
the independent verifier before its own commit.
deferred: nothing from the audit list.

## Ownership

defects proven: none that require moving state.
concerns not proven: none open.
changes made: documentation and the five cleanups only.
changes rejected: replacing the stored-through max counters (both
proofs hold; fail-loud checks pinned); converting the CSB-lag refusal
into waiting (architectural change, needs its own note).
Full field table: reports/DECSIM_RUNTIME_STATE_OWNERSHIP.md.

## Tests

full suite: 790 passed (752 pre-existing + 38 stabilization).
frozen exact: 4 weak points bit-identical including the full log hash.
frozen semantic: 2 strong + 3 switching points identical on the
semantic projection; independent verifier PASS at every gate.
initialization: 8 deterministic tests.
Buffer 0: covered in test_buffer_readiness (arithmetic, order,
publication, readiness authority).
Buffer 1: stored-through gap refusal, strong-primary readiness,
settledness.
switching: serial, parallel (including the CSB-margin refusal),
double-window terminal and far boundary, ledger generations.
feedback: no-feedback, exact OC+CQ chain, successor gating, structural
non-bypass.
mutation matrix: 22 of 23 detected (M09 and M21 by simulator stall);
the one undetected mutation is behaviorally inert by construction
(refcounted holds); two first-sweep misses exposed suite gaps that
were closed and re-detected. reports/DECSIM_MUTATION_MATRIX.md.
independent verifier: PASS at the final tree.

Scientific outputs changed: no.
Deterministic timestamps changed: no.
Selected tier changed: no.
Pauli-frame records changed: no.
Link traffic changed: no.
Buffer occupancy changed: no.
(All six: the frozen suite and the independent verifier passed after
every cleanup and at the final tree.)

## Remaining uncertainties

- Far-boundary deferral in a backlog regime refuses with a bare
  KeyError instead of its contract message (strong_escalation.py:716);
  flagged, not changed.
- Parallel switching requires the CSB margin; parameter-reference note
  suggested when parallel is next configured.
- qubic_emulator's three golden-waveform failures look like
  scipy/numpy drift; unresolved upstream.

## Next approved research task

None assumed. The commission ends before hardware models; candidate
next steps (ReferenceController, decoder units) each need their own
design note per docs and the audit.
