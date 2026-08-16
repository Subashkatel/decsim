# Q-028 independent strict final review (QLX fixtures + workload/physical suite)

Reviewer: independent final reviewer, no implementation role in Q-028.
Repo: /scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/decsim
Branch: audit/evidence-first-rebuild
HEAD at review: d081472 (Q-027 smoke decoupling); Q-028 is an uncommitted batch.
Review round: 2 (round 1 returned FAIL with B1/B2/B3; corrections re-verified here).

VERDICT: PASS.

All three round-1 blockers are closed, the two round-1 non-blocking items were
also taken, and every claim I can execute reproduces exactly. One record-hygiene
line remains (R1 below): it touches no code, test, fixture, or gate, and does
not affect this verdict, but it should be corrected with the commit.

Scope honored: protocol, ORIENTATION, OWNER_QUEUE (Q-028/Q-029), qlx.py,
stim_device.py, detector_chronology.py, planner.py, the restored fixtures and
README, the new suite, Q028_RESTORE_REPORT.yaml, Q028_TEST_MAP.yaml, smoke
artifacts. The only history read was the authorized
43f4ffa^:tests/test_qlx_physical.py:610-614; fixture byte-identity was checked
by object id, which reads no historical content. Generator/dump/probe scripts
were never imported or executed. My probes ran from /tmp against the project
venv and are deleted. No shared edit, no commit.

## 1. Round-1 blockers: closed

B1 (missing test docstrings) CLOSED. Both tests now carry a one-line
plain-language docstring ("Pin frozen schedule structure and GateRounds-owned
runtime pricing." / "Prove detector routing completes without claiming decode
quality."). Tree-wide: 390/390 tests have docstrings.

B2 (undocumented load-bearing d=7) CLOSED. _detector_bearing_runtime's
docstring now states that d=7 "prices only the seven-round detector-bearing
stream owner; it is a runtime source length here, not a claim that the frozen
circuit has code distance seven". The disclaimer is exactly where a reader
meets the normalization, and Q028_TEST_MAP repeats it as a pinned contract.

B3 (TEST_MAP overstated the wiring) CLOSED. Schema 2 now pins
"detector_rounds is an order-preserving derivative with detector IDs unchanged
and each emitted round r mapped to r-1" and "terminal_detector_ids and
terminal_data_bits are passed verbatim from qlx_frontend". I verified both
statements literally: detector id set unchanged (all 56) and shifted[d] ==
orig[d] - 1 for every detector; the two terminal maps are passed through
untouched.

Round-1 non-blocking items also taken:
- baseline submission now gets kind=OpKind.MEASURE, so the workload is
  internally consistent: ops[1:9] all price at 1 round (test asserts the
  eight-tuple now, not seven), owner 7 rounds. Verified live.
- qlx.py line 19 now reads "Mapping rules (partially asserted by the
  tests/09_qlx_workloads suite)", which is accurate for the subset the suite
  exercises.

## 2. Restoration and provenance: still exact

- 25/25 fixtures byte-identical to 43f4ffa^ (git hash-object == historical blob
  ids, rechecked after the correction round); directory holds 26 entries
  (25 + README). README unchanged
  (c350360491b53d55c7819f20d3f90799dc44c1ffcae605615976949ce67ce47e) and
  records the literal revision, the resolved commit 811d3e1c..., frozen
  test-only status, no CI regeneration, scripts as provenance only, and the
  ban on importing/executing them.
- Collection sanity unchanged: 600 tests collected, zero items from
  tests/data; the suite imports only json, pathlib, dataclasses.replace, stim,
  and decsim modules.

## 3. No core edits; qlx.py still doc-only

git status -uall shows exactly one modified tracked file
(decsim/frontends/qlx.py) plus the untracked tests/09_qlx_workloads and
tests/data/qlx trees. The qlx.py diff is 4 docstring lines and the
docstring-stripped AST is IDENTICAL to HEAD, so the change is provably
semantics-free. Both original references were dangling; both replacements
exist; no other path-like reference remains.

## 4. Substance re-verified against the SHIPPED helper

I imported the shipped test module and traced a real run through its own
_detector_bearing_runtime (no reviewer reimplementation):

- Faithful relabel: detector ids unchanged (all 56), every emitted round
  mapped exactly r -> r-1; only the detector-free baseline leaves the stream.
- Real routing: 7 payloads of 8 bits, detectors 0-7, 8-15, ... 48-55 emitted
  by measure_syndrome_1..7 respectively; all 56 real detectors routed exactly
  once; the Stim shot itself is untouched.
- Still falsifiable: permuting two detectors across rounds (0 <-> 8) while
  preserving coverage raises the Q-019 canonical decoder-input row-layout
  check against the window error model built from mem_surface.stim.
- Still necessary (round-1 result, unchanged by the corrections): core
  detector_chronology.resolve_detector_rounds requires every emitted round to
  bear a detector, and the frozen circuit's first submission is a pure
  baseline, so an unnormalized program cannot be wired at all (unnormalized
  d=7/d=8, unshifted map, baseline-kept, and d=6 all raise).
- Scope discipline intact: timing-only PerRoundDecoder(tau_us=0.0), every
  operation result no_logical_output, no truth/accuracy/failure-rate
  assertion; wiring is verbatim 43f4ffa^:tests/test_qlx_physical.py:610-614.
- Structural test unchanged and exact: 11 ops, full predecessor chain, patch
  cardinality 1, monotone start_rounds, feedback_candidates [(10, 9)],
  GateRounds(merge_steps=2) resolved (3,3,3,3,3,3,3,3,3,1,3) and asserted
  different from the raw QLX durations. QLX 0/1 durations price nothing.

## 5. Gates re-run in round 2

  /usr/bin/python 3.9.25 (pinned): compileall rc=0; pytest -q tests/ 600
    passed in 1.71s; smoke_run.py rc=0 (exact baseline)
  .venv/bin/python 3.10.12: compileall -q decsim tests/09_qlx_workloads rc=0;
    600 passed in 1.57s
  focused tests/09_qlx_workloads: 2 passed, 3/3 identical repeats
  git diff --check rc=0
  Hashes recomputed and matching the record:
    suite  7c73555a16826710b2325dcc5d7acb6566e38befdafcf99597b490a6409be2e6
    qlx.py fed95cb158d034eb18f656528a326b081250c20f676d641adc93799b4ccdf307
    TEST_MAP ec62a96f9ef2cffd79b23b98c78b8a1fd0cfb1f70b962661d178f948e5624c74
    README c350360491b53d55c7819f20d3f90799dc44c1ffcae605615976949ce67ce47e
  TEST_MAP locators resolve: line 80 and line 115 are the two test defs.

## 6. Record hygiene and open ledger items

R1 (record only, no verdict impact): Q028_RESTORE_REPORT.yaml
qlx_module_docstring.after_sha256 is still 054024207fb0... from the
restoration round; the file is now fed95cb158d0... after the B2/6.6 doc
correction. Q028_TEST_MAP schema 2 carries the current hash, so the batch is
self-consistent apart from this one stale field. Update it with the commit so
every recorded hash resolves.

R2 (recommended, unchanged from round 1): queue an OWNER_QUEUE
deferred_question recording that core cannot represent a finite Stim source
whose first emitted round is detector-free
(detector_error_model/detector_chronology.py resolve_detector_rounds), since
real QLX-emitted memory circuits do exactly that and the workaround currently
lives only in this suite. Pass-2 frontends/detector work should own it.

R3 (informational, unchanged): importing any decsim submodule executes
decsim/__init__.py, which imports .frontends.circuit (Q-029, untrusted). The
suite does not use it.

## Conclusion

PASS. The restoration is byte-exact with recorded provenance, the only core
touch is a provably doc-only docstring fix, the structural test pins exactly
the ruled properties with explicit GateRounds pricing and no QLX raw
durations, and the physical test is an honest, necessary, order-preserving,
falsifiable routing-and-completion smoke that claims nothing about decode
quality. All gates pass on both interpreters, the suite is deterministic, and
every recorded contract in Q028_TEST_MAP is literally true of the shipped
code. Only R1 (one stale hash field in the restore report) should be corrected
with the commit.
