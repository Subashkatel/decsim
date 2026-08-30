# Reproduction record (2026-08-30)

Rule: a reference is VERIFIED_REFERENCE only if its own official example
ran end to end here. Anything else is READ_ONLY_REFERENCE and is never
used as a numeric oracle. Local patches are listed in full; every patch
lives in a copy (job tmp), never in the clone.

Stabilization-pass reruns (same day, fresh processes): SWIPER verify
17/17 PASS; PECOS `cargo test -p pecos-decoder-core --lib` 124/124 plus
6 integration and 7 doctests; Helios d=3 rsc bench 1000/1000 under the
from-source iverilog. The QubiC 2.0 section below was added during the
stabilization pass.

## QubiC 2.0 (added 2026-08-30)

- distributed_processor @ c22cce81 (gitlab.com/LBL-QubiC), QubiC
  LBNL/Regents BSD-3 variant license. Official python test suite from
  its documented working directory `python/test`:
  37 passed, 5 skipped, 0 failed (python 3.12.13). VERIFIED for the
  distproc compiler/assembler software slice; the HDL cocotb benches
  were not run (no Verilator on this host), so RTL-level claims stay
  read-only.
- qubic_emulator @ 8cc66588, same license family. READ_ONLY_REFERENCE:
  the official `pytest test_suite` gives 10 passed, 3 FAILED here; the
  failures are the repo's own golden waveform comparisons
  (test_freq_amp_sweep, test_hundred_pulses, test_thousand_pulses)
  exceeding tolerance by 50.5 to 67.5 on a 32767 full scale (about 0.2
  percent), consistent with scipy/numpy filter-response drift. Running
  at all also requires python 3.12+ (PEP 701 nested f-strings in
  test_suite.py) and the qubic `software` repo (ab680657) for
  `qubic.toolchain`, which the emulator README does not declare as a
  test dependency. Companion pin: qubitconfig @ c766ddbc.

## SWIPER: VERIFIED_REFERENCE

- Command: tmp/swiper_timing_v1/run.sh (drives reference/swiper, the
  official DeviceManager + LatticeSurgerySchedule + DecodingSimulator on
  the pinned official Toffoli lattice-surgery schedule, d=15), then
  verify.py.
- Result: run exit 0; verify 17/17 PASS (schedule content matches the
  pinned official Toffoli; 86 instructions; 7 conditional corrections;
  reaction times and utilization recompute; more decoders reduce reaction
  time and backlog). Output refreshed at
  tmp/swiper_timing_v1/output/verification.txt (timestamp 2026-08-30).

## PECOS: VERIFIED_REFERENCE

- Build: previously built from source (pecos_build.log exit 0), package
  quantum-pecos 0.11.0.dev0 imports in tmp/reference-decoders/venv.
- Crate tests (cargo test -p pecos-decoder-core, 2026-08-30):
  pauli_frame 5/5, obs_mask 8/8, two_pass 2/2, adaptive 2/2,
  multi_decoder 5/5. Log: job tmp pecos_pf_test.log.

## Helios: VERIFIED_REFERENCE

- Toolchain: iverilog v12_0 built from source (no HDL simulator on the
  host); install at job tmp/iverilog-install.
- Bench: the repo's own single_FPGA_FIFO_verification_test_rsc.sv at
  CODE_DISTANCE=3 against shipped input_data_3_rsc.txt and golden
  output_data_3_rsc.txt, with the official file set from
  scripts/total.tcl (fifo_fwft, mem_communicator, control_node,
  neighbor_link_internal_v2, processing_unit_v2, serdes,
  decoding_graph_dynamic_rsc, tree_compare_solver, Helios_single_FPGA_core).
- Local patches (copies in job tmp/helios_repro): absolute paths for
  $fopen and the parameters.sv include; fifo_fwft.v comment idiom
  ("//*/" closes) rewritten for iverilog's nested-comment check;
  CODE_DISTANCE 13 -> 3. No logic touched.
- Result: Total 1000, Passed 1000, Failed 0 (helios_run_d3.log).
- Notes: the older single_FPGA_verification_test.sv targets a previous
  core interface (parallel measurements port, CODE_DISTANCE_X parameters)
  and no longer elaborates against the current byte-stream core: a stale
  TB upstream, recorded, not used. Golden rsc data ships only for d=3.

## XQsim: READ_ONLY_REFERENCE

- The released code cannot run its own README example:
  1. requirements.txt pins ibm-cloud-sdk-core==3.16.7, which no longer
     resolves; a minimal venv with their core pins (numpy 1.23.5,
     qiskit 0.42.1, ray 2.5.0, stim 1.11.0, python 3.9) was used instead.
  2. xq_simulator.py reads self.emulate (lines 170, 356) which no code
     path assigns: AttributeError on every invocation.
  3. With a one-line patch in a COPY (job tmp/xqsim_patched:
     self.emulate = self.skip_pqsim, mirroring the qxu wiring at line
     116), the official example
     `xq_simulator.py -c example_cmos_d5 -b pprIIZZZ_n5 -s 2048 -sp True`
     runs the full cycle simulation to completion: Last cycle 17679,
     810.158 sec, per-unit stats printed (QID, PDU, PIU, PSU, TCU, EDU,
     PFU, LMU) and then crashes in main()'s summary printing
     (pqsim_res.items() on None, line 547), a second released defect.
- Verdict per the audit rule: not reproducible as released, READ_ONLY.
  Used for microarchitectural reading only (unit decomposition and unit
  responsibilities from README section 2 and src/XQ-simulator/*.py).

## QICK: READ_ONLY_REFERENCE

- Official tests run on physical RFSoC boards
  (.github/workflows/test_zcu111.yml: runs-on [zcu111], sudo FPGA
  access); no board on this host.
- Software slice verified: qick 0.2.422 installs from the clone
  (job tmp/qick-venv); QickConfig parses the shipped
  firmware/testbench/qick_testbench/soccfg.json (ZCU216, tProc v2
  dispatcher timing 430.080 MHz, generator + readout channels printed).
- Used for reading the timed-command and readout-feedback ISA shape only.

## RISC-Q: READ_ONLY_REFERENCE

- Chisel/mill generator; testbenches are SpinalSim/Scala
  (src/main/scala/riscq/tester/). mill and Verilator are not on the host
  and ext/ submodules are not vendored in the shallow clone.
- NO LICENSE FILE at the pinned commit: default all rights reserved.
  Cite the design (paper arXiv:2505.14902); never copy code.

## decsim frozen reference suite (the differential baseline)

- tmp/decsim-responsibility-audit/frozen-suite/capture.py
  (run from the decsim repo with .venv/bin/python; `capture` then `check`).
- 9 points: 4 weak baseline (strict, bit-identical including the full log
  hash), 2 strong baseline + 3 switching (semantic tier).
- Verified 2026-08-30: `check` in a fresh process = PASS: behavior
  preserved. Golden: frozen-suite/golden.json at decsim 65660a0.
- Why two tiers: the strong tier and the switching weak tier price decode
  latency from the measured wall clock of the real decoder
  (decsim/decoders/belief_matching/decoder.py:82-87 and mwpm/decoder.py,
  time.perf_counter_ns), so tick-bearing fields legitimately vary between
  runs; the semantic projection (logical results, window structure and
  decode statuses, frame-record sequence with tiers and observables,
  tick-free link traffic, max queue depth, idle rounds, strong counters)
  is what a refactor must preserve there, and it was stable across three
  independent capture runs. If a future check ever flakes on frame-record
  ORDER at a wall-clock point, loosen order there deliberately; never
  accept a content change.
- Escalations are real in the suite: switching points commit 4/9/6 strong
  windows (frame tier column in golden.json).
