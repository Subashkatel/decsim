# Mutation matrix

Each mutation was applied to the working tree, the detector ran,
and the tree was restored, in sequence, from a clean checkout.
Detector "stabilization suite" is `pytest tests/21_stabilization`;
"frozen suite" is the independent verifier
(`validation/responsibility_audit_2026_08_30/verify.py`). Driver:
job tmp mutation_matrix.py; raw results: mutation_results_final.jsonl.

| # | Mutation | Detector | Detected | Actual failing check | Why this pins the invariant |
|---|---|---|---|---|---|
| M01 | install the window plan only after the run | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_weak_only_pipeline_arithmetic; tests/21_stabilization/test_buffer_readiness.py::test_unpriced_cwb_publishes_at_packing | pins the setup order: windows and holds exist before any round arrives |
| M02 | publish a round for an unregistered operation | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_weak_only_pipeline_arithmetic; tests/21_stabilization/test_buffer_readiness.py::test_unpriced_cwb_publishes_at_packing | pins registration as a precondition of publication |
| M03 | execution program and window plan disagree (drop one registration) | stabilization suite | yes | tests/21_stabilization/test_initialization_determinism.py::test_every_program_operation_is_registered | pins the execution-view registration pass as load-bearing |
| M04 | skip the Buffer 0 write (store then discard) | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_weak_only_pipeline_arithmetic; tests/21_stabilization/test_buffer_readiness.py::test_unpriced_cwb_publishes_at_packing | pins publication-before-notification on Buffer 0 |
| M05 | skip the Buffer 1 dual write | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_weak_primary_readiness_is_buffer0_not_sb1; tests/21_stabilization/test_buffer_readiness.py::test_strong_primary_readiness_is_sb1 | pins the dual write as the strong tier's only data source |
| M06 | advance the SB1 counter and fire the callback before the CSB crossing | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_strong_primary_readiness_is_sb1; tests/21_stabilization/test_switching_and_feedback.py::test_double_window_far_boundary_waits_for_the_restart_commit | pins the callback-after-storage contract |
| M07 | remove the SB1 stored-round callback | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_strong_primary_readiness_is_sb1 | pins the callback as the strong tier's wake signal |
| M08 | stored-through counter advances across a missing round | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_strong_primary_readiness_is_sb1; tests/21_stabilization/test_buffer_readiness.py::test_sb1_gap_cannot_be_served | pins the stored-through meaning of the max counter |
| M09 | Buffer 1 drives weak-primary readiness instead of Buffer 0 | stabilization suite | yes | DETECTOR TIMEOUT: the mutated simulator stalls (300 s) | pins Buffer 0 as the weak lane's readiness authority |
| M10 | build a strong job over missing SB1 context | stabilization suite | yes | tests/21_stabilization/test_switching_and_feedback.py::test_parallel_requires_the_csb_margin | pins the fail-loud missing-context refusal |
| M11 | start compute without waiting for the input transfer | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_weak_only_pipeline_arithmetic; tests/21_stabilization/test_buffer_readiness.py::test_strong_primary_readiness_is_sb1 | pins SBD/WBD landing before compute |
| M12 | release the upstream hold before the transfer lands | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_upstream_rounds_survive_until_the_input_transfer_lands; tests/21_stabilization/test_buffer_readiness.py::test_room_side_rounds_survive_until_the_final_strong_commit | pins upstream retention until the transfer lands |
| M13 | release the SB1 hold before the SBD landing | stabilization suite | NO | 38 passed in 0.21s (behaviorally inert: refcounted holds) | shows the refcounted holds absorb a redundant early release |
| M14 | commit the provisional weak result | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_stores_settle_empty; tests/21_stabilization/test_switching_and_feedback.py::test_unconfident_serial_escalates_every_window | pins that a provisional weak result never becomes final |
| M15 | route the strong request to the weak decoder | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_stores_settle_empty; tests/21_stabilization/test_switching_and_feedback.py::test_unconfident_serial_escalates_every_window | pins the weak/strong routing separation |
| M16 | accept a stale strong completion | stabilization suite | yes | tests/21_stabilization/test_switching_and_feedback.py::test_stale_strong_result_raises_when_a_newer_request_owns_the_destination | pins request-generation matching in the ledger |
| M17 | skip the WSD selection reservation | frozen suite | yes | FAIL {'config': 'switching_validation.yaml', 'd': 3, 'p': 0.008, 'round_period_us': 1.0, 'seed': 0} [semantic]: link_traffic_semantic; FAIL {'config': 'switching_validation.yaml', 'd': 3, 'p': 0.008, 'round_period_us': 1.0, 'seed': 1} [semantic]: link_traffic_semantic | pins the WSD selection leg as measured traffic |
| M18 | remove the SBD transfer delay | stabilization suite | yes | tests/21_stabilization/test_switching_and_feedback.py::test_serial_escalation_timeline_is_exact | pins the SBD leg in the strong start time |
| M19 | remove the weak output (WDO) delay | stabilization suite | yes | tests/21_stabilization/test_buffer_readiness.py::test_weak_only_pipeline_arithmetic | pins the WDO leg in the weak result path |
| M20 | remove the strong output (DO) delay | stabilization suite | yes | tests/21_stabilization/test_switching_and_feedback.py::test_serial_escalation_timeline_is_exact | pins the DO leg in the strong result path |
| M21 | write the Pauli frame twice | stabilization suite | yes | DETECTOR TIMEOUT: the mutated simulator stalls (300 s) | pins one authoritative correction per window |
| M22 | send the release directly, skipping OC and CQ | stabilization suite | yes | tests/21_stabilization/test_switching_and_feedback.py::test_release_travels_oc_then_cq_with_exact_cost | pins the OC then CQ release route through the controller |
| M23 | release the blocked successor without its decode release | stabilization suite | yes | tests/21_stabilization/test_switching_and_feedback.py::test_release_travels_oc_then_cq_with_exact_cost; tests/21_stabilization/test_switching_and_feedback.py::test_successor_cannot_start_before_its_release | pins the decode-release gate on blocked successors |

Detected: 22 of 23. Every mutation was reverted immediately after its
detector ran; the tree was verified clean at the end of the sweep.

Notes:

- M09 and M21 are detected by TIMEOUT: the mutated simulator stalls
  (the weak lane loses its readiness authority, or the frame's second
  accepted write never installs its commit continuation), so the
  detector run never finishes; a stall is the loudest possible failure.
- M13 is the one undetected mutation, and it is undetectable BY
  CONSTRUCTION: store holds are refcounted, so releasing one redundant
  holder early frees nothing while any other live holder protects the
  same rounds, and the store refuses to free held rounds outright
  (syndrome_buffer.py:415-418). The occupancy probe confirmed the
  room-side timeline is bit-identical under the mutation. The invariant
  the mutation aims at (rounds outlive their consumers) is carried by
  the refcount and pinned by
  tests/13_orchestrators/test_syndrome_buffer_1.py::test_refcounted_release_order
  and the two occupancy tests in tests/21_stabilization.
- M03 and M12 were misses on the first sweep; the suite gaps they
  exposed were closed (test_every_program_operation_is_registered and
  the two occupancy-probe tests), and the reruns detect them. Keeping
  the miss-then-fix history here is deliberate: the matrix tested the
  tests.

Expected detectors (recorded before the sweep):

- M01: any full-run stabilization test (windows missing at first arrival)
- M02: any full-run test via the unknown-operation refusal
- M03: full-run tests via the missing readiness account
- M04: test_weak_only_pipeline_arithmetic (publication check fails loudly)
- M05: switching and strong-only tests (strong context never stored)
- M06: test_strong_primary_readiness_is_sb1 (readiness runs ahead of storage)
- M07: strong-only and double-window tests (nothing wakes the strong tier)
- M08: test_sb1_gap_cannot_be_served (counter no longer stored-through)
- M09: test_weak_primary_readiness_is_buffer0_not_sb1 and weak-only runs
- M10: test_parallel_requires_the_csb_margin (the fail-loud check is gone)
- M11: test_weak_only_pipeline_arithmetic (transfer time vanishes)
- M12: test_upstream_rounds_survive_until_the_input_transfer_lands (occupancy probe)
- M13: behaviorally inert by construction: see the note under the table
- M14: test_unconfident_serial / test_provisional_weak_result_never_reaches_the_frame
- M15: switching tests (check_strong_route refuses the shared decoder)
- M16: test_stale_strong_result_raises_when_a_newer_request_owns_the_destination
- M17: frozen suite semantic link traffic (WSD transfers disappear)
- M18: test_serial_escalation_timeline_is_exact (strong start moves to 33)
- M19: test_weak_only_pipeline_arithmetic (frame accepted at t_done)
- M20: test_serial_escalation_timeline_is_exact and strong-only arithmetic
- M21: duplicate-drop assertions (frame accepts one write per window)
- M22: test_release_travels_oc_then_cq_with_exact_cost (release 4 us early)
- M23: test_successor_cannot_start_before_its_release
