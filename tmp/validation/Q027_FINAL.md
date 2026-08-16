# Q-027 independent final review (smoke decoupling)

Reviewer: independent blind final reviewer (no implementation role in Q-027).
Date: 2026-08-16 (review session)
Repo: /scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/decsim
Branch: audit/evidence-first-rebuild
HEAD at review: 08fa8d46e704369abf53e53b5b620b7000ecd29e (matches Q027_REPORT.git.head)

VERDICT: PASS

Scope honored: I did not open, import, or cite decsim/frontends/circuit.py, git
history of it, decsim/experiments/, or any tests source. Evidence below is from
protocol/orientation, the owner records, the smoke artifacts, core module
declarations (decsim/message.py, decsim/run_spec.py, decsim/__init__.py header),
static grep, and freshly executed gates. I made no edits to any shared artifact,
production file, or test, and created no commit.

## 1. Inline reviewed core vocabulary + RunSpec

- tmp/validation/smoke_run.py builds its workload from three literal
  decsim.message.Operation records in a module-level tuple: ids 0/1/2, names
  "prepare patch 0" / "merge patches 0 and 1" / "measure patch 1", qubits
  (0,), (0,1), (1,), patches (0,), (0,1), (1,), predecessors (), (0,), (1,),
  kinds OpKind.MEMORY / MERGE / MEASURE.
- Operation (decsim/message.py:904) and OpKind (decsim/message.py:891) are in
  module 01_message, recorded committed in CLEANUP_STATE.completed_prefix, so
  the vocabulary is reviewed core. Every field used exists on the dataclass and
  every kind used exists in the enum; no non-core or ad-hoc workload type.
- Execution goes through the public entry RunSpec(...).build() (ORIENTATION:
  "RunSpec is the single public entry"), with ops=, decoder=, make_metrics=
  only. No private helper or internal builder is called.
- Protocol section 0 requirements for the smoke run are met: minimal
  timing-only simulation through the public surface, a few operations,
  .build(), asserts completion and returns metric results
  (terminal_status == "complete", event_queue_empty, decode_work_settled,
  execution_workload_complete, metric_results non-empty, decoder_utilization
  key present).

## 2. Minimal deterministic structure

- 105 lines, no branching workload construction, no randomness source, no
  wall-clock, no environment reads, no file writes. Only file input is the
  sibling SMOKE_BASELINE.yaml, guarded by a schema_version == 1 and snapshot
  mapping check.
- Determinism inputs are fixed: RunSpec.seed defaults to 0 (run_spec.py:132),
  device defaults to TimingOnlyDevice (run_spec.py:238, timing-only fast path),
  decoder is PerRoundDecoder(tau_us=0.1) (fixed latency model).
- Dependency graph is a minimal linear chain 0 -> 1 -> 2 with explicit patch
  identities; clifford defaults True, so magic-state factories stay inert.
- Empirical determinism: 4 independent processes (2 plain, 1 with
  PYTHONHASHSEED=1, 1 with PYTHONHASHSEED=98765) all exited 0 against the same
  exact snapshot.

## 3. No frontend dependency

- Static grep over tmp/validation/smoke_run.py for
  frontend|circuit|three_cnot|experiments|random|time\.|datetime returns no
  match (grep rc=1). Its only decsim imports are decsim.decoders
  (PerRoundDecoder), decsim.message (OpKind, Operation), decsim.metrics
  (DecoderUtilization), decsim.run_spec (RunSpec).
- The owner ruling's operative requirement ("no frontends import in
  tmp/validation/smoke_run.py"; the untrusted module "must not underpin
  verification") is satisfied: neither the workload, nor the asserted result,
  nor the metric snapshot derives from the untrusted frontend.
- Non-blocking observation (out of Q-027 scope, no action required here):
  importing any decsim submodule executes decsim/__init__.py, which at line 12
  imports .frontends.circuit, so decsim.frontends.circuit is present in
  sys.modules during the smoke process. This is pre-existing production
  coupling in decsim/__init__.py, not something Q-027 introduced, and removing
  it would require a production edit that Q-027 explicitly forbids itself.
  Nothing in the gate's semantics depends on that module's behavior or output.
  Recommendation (not a defect): add this import-level coupling as an extra
  evidence line under Q-029 for the pass-2 frontends review.

## 4. Exact baseline, rationale, interpreter, hashes

- The gate is a full-snapshot equality assertion
  (actual_snapshot == load_baseline()) covering workload operations, terminal
  status, the four completion booleans, execution_done_ticks,
  fully_done_ticks, per-operation result statuses, and metric_values. Not a
  tolerance or subset check.
- Sensitivity proof (run on private copies in /tmp, since removed; no shared
  file touched): 7 semantic single-field mutations of the baseline copy were
  each rejected with rc=1: busy fraction 0.07017543859649122 -> 0.07018,
  fully_done_ticks +1, execution_done_ticks +1, an operation name change,
  terminal_status complete -> incomplete, aggregate_total_units 1 -> 2,
  result_status no_logical_output -> logical_observables. Control copy passed
  (rc=0). The exact-snapshot claim is therefore real, not vacuous. The only
  insensitivity found is at sub-ULP float precision (last-digit change that
  parses to the same double), which is inherent to double comparison and not a
  defect.
- Rationale is recorded in both required places: SMOKE_BASELINE.yaml.rationale
  (workload / trust / supersession) and CLEANUP_STATE.current_smoke_baseline
  .rationale, both stating the untrusted legacy frontend no longer underpins
  verification and that the new metric intentionally supersedes rather than
  regresses against the old one. This matches the protocol's "record why in the
  state file" clause and reconciles the historical
  smoke_metrics_identical: true records of earlier modules.
- Interpreter recorded as /usr/bin/python, CPython 3.9.25 in both
  SMOKE_BASELINE.yaml and Q027_REPORT; verified: /usr/bin/python -VV reports
  Python 3.9.25 (CPython). Matches CLEANUP_STATE.python.executable.
- Recomputed sha256, all four match Q027_REPORT.artifact_sha256 exactly, and
  the smoke_run/baseline hashes also match
  CLEANUP_STATE.current_smoke_baseline:
    smoke_run.py        c9d4d23899234dcb224d248b5b6c21a2976fca00235d08d8b1dd35e8f77ad195
    SMOKE_BASELINE.yaml ba754e44bc86b55078e301b10647c4a31a61e251943550ba4ec4a0eb483a2307
    CLEANUP_STATE.yaml  72575a83b7888d79073d31ba7ae267e1aa68f54fca3bcd545a46a0e3c61ba1de
    OWNER_QUEUE.yaml    6cc212dea08d8fdd1c6bf846d75df4d9c1d77188ca1c7575ce953dd79289e156
- Recorded baseline numbers are internally consistent across Q027_REPORT,
  SMOKE_BASELINE.yaml, and CLEANUP_STATE: execution_done_ticks 11000000,
  fully_done_ticks 14250000, aggregate_busy_fraction 0.07017543859649122,
  aggregate_total_units 1, per-pool "default" identical.

## 5. Owner-record consistency

- Q-027 (owner_directive, ruling_status implemented_verified_awaiting_commit)
  clauses all satisfied: inline reviewed vocabulary, no frontends import,
  re-baselined snapshot, rationale in the state file, Q-029 queued as
  deferred_question/module frontends/pass-2 scope, full gates run, commit left
  to the parent batch (commit_created: false, and HEAD is still the Q-022 test
  commit 08fa8d4).
- Q-029 exists with kind deferred_question, module frontends, status
  pass_2_deferred_untrusted, and states nothing in this job may cite the
  untrusted module as evidence or use it in tests or smoke verification. Its
  evidence lines reference the historical dependency only, which is not a
  correctness citation of that module.
- Q-028 remains pending_after_Q027, consistent with Q-027 not yet committed.

## 6. Gates re-run independently by this reviewer

  /usr/bin/python -m compileall -q decsim              rc=0   PASS
  /usr/bin/python -m pytest -q tests/                  598 passed in 1.46s  PASS
  /usr/bin/python tmp/validation/smoke_run.py (run 1)  rc=0   PASS (exact snapshot)
  /usr/bin/python tmp/validation/smoke_run.py (run 2)  rc=0   PASS (exact snapshot)
  same, PYTHONHASHSEED=1 and =98765                    rc=0   PASS (exact snapshot)
  static grep, frontend/circuit refs in smoke_run.py   none   PASS
  git diff --check                                     rc=0   PASS
  git diff --name-only -- decsim tests                 empty  PASS
  git status --porcelain                               empty  PASS (tracked worktree clean)

Full-suite count matches the recorded 598; the recorded gate table in
Q027_REPORT is accurate in every line I could re-execute.

## 7. Production and test edits

None. HEAD is unchanged at 08fa8d4, the tracked worktree is clean, and
git diff --name-only -- decsim tests is empty, so the Q-027 change set is
confined to tmp/validation artifacts (which are git-ignored via /tmp/ in
.gitignore) exactly as Q027_REPORT.scope claims. No commit was created by the
implementer or by me.

## 8. Out-of-scope observation (informational only)

CLEANUP_STATE.owner_queue_count is 18 while OWNER_QUEUE.yaml holds 30 entries.
This field lies outside the smoke record under review and predates Q-027's
edits in kind; flagged only so the orchestrator can reconcile the counter
during its next state update. It has no bearing on the Q-027 verdict.

## Conclusion

PASS. The Q-027 smoke decoupling is correctly implemented, accurately recorded,
independently reproducible, deterministic across processes and hash seeds,
genuinely sensitive to drift, free of any frontend-derived verification input,
and free of production or test modifications. No blocking issue found; the two
observations above are informational and require no change to the Q-027 batch.
