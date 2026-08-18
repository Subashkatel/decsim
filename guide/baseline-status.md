# Baseline status: target architecture vs decsim

Q-062 part (a), evidence snapshot for 2026-08-17. This document reports the
simplest owner-approved closed loop first. It does not treat file existence as
a completed review.

Status vocabulary:

- **DONE**: accepted review or ruling batch is committed, with an exact record below.
- **EXISTS, REVIEW PENDING**: the named source exists, but its cleanup module is pending.
- **BUILD PENDING**: owner-directed work has not landed.
- **LATER**: deliberately outside the minimal loop, not accidentally omitted.

## Current gate

- The committed review prefix is modules `00_engine` through `14_planner`, 15 of
  32 core modules. Source of truth: `tmp/validation/CLEANUP_STATE.yaml`,
  `module_order` and `modules`.
- Q-053 is complete. Links production commit `b538bc8`, test commit `207e3ee`.
- Q-062(b), the minimal Pauli-frame sink, is **DONE**: production commit
  `efe7270`, test commit `bdd0b1f`; the latest accumulated suite is **818 passed**,
  with compileall and the default smoke run green. The direct owner addendum in
  `tmp/validation/OWNER_QUEUE.yaml` accepted the verified semantics and required
  the short PECOS provenance header before commit.
- Q-062(c) is **REOPENED** by the 22:55/22:58 owner amendment. The committed
  card work (`ec0b17f`, `9d67f54`; tests `547802c`, `c507dd5`) passed 843 tests
  and smoke, but is not completion: ordered FETCH, DECODE, real EXECUTE, MEMORY,
  and WRITEBACK engine events adapted from XQsim must still land.
- Q-062(d), the baseline experiment and report, and Q-062(f), published-number
  validation, remain **BUILD PENDING** behind the reopened Part C. Evidence:
  `tmp/validation/OWNER_QUEUE.yaml`, Q-062. The module-review pipeline remains
  paused at module 15 until the Q-062 priority sequence completes.

## Minimal baseline closed loop

| # | Target component | decsim home | Status | Evidence and boundary |
|---|---|---|---|---|
| 1 | Frontend, operation DAG and real detector input | `decsim/frontends/qlx.py`; `decsim/adapters/stim_device.py` | EXISTS, REVIEW PENDING | Both sources exist at the hashes in the source ledger. They are adjacent code, not members of the committed 00-14 review prefix. Q-062(d) still must prove the large generated `rotated_memory_z` real-data path. |
| 2 | Orchestrator and planner, including window plan | `decsim/orchestrators.py`; `decsim/planner.py`; `decsim/rounds.py`; `decsim/window_manager.py` | DONE for orchestrator and planner; EXISTS, REVIEW PENDING for window manager | Orchestrator: `tmp/validation/13_orchestrators/REQUIREMENTS.yaml`, `tmp/validation/13_orchestrators/FINAL.md`, production `53cfb34`, tests `fa097a9`. Planner/round policies: `tmp/validation/14_planner/REQUIREMENTS.yaml`, `tmp/validation/14_planner/FINAL_BASE.md`, `tmp/validation/14_planner/FINAL_SPLIT.md`, production `323e321` then `b8d54d2`, tests `741b76a` then `7e19c56`. Window manager is pending module 27. |
| 3 | Controller, QPU-bound conversion, operations to pulses | `decsim/controller.py` | EXISTS, REVIEW PENDING | Source exists; module 23 is pending in `CLEANUP_STATE.yaml`. The symmetric `t_pulse_generation_us` cost is pending Q-055. No claim is made that this conversion is fully priced yet. |
| 4 | QPU round source and device | `decsim/qpu.py`; `decsim/devices.py`; `decsim/adapters/stim_device.py` | EXISTS, REVIEW PENDING | Sources exist. QPU is pending module 15 and devices module 20. Q-062(d) must exercise the real Stim device path. |
| 5 | Controller, decoder-bound conversion, pulses to binary | `decsim/controller.py`; `decsim/config.py` | EXISTS, REVIEW PENDING for controller; DONE for the existing timing card | Controller is pending module 23. The reviewed configuration contract records `t_binary_availability_us`: `tmp/validation/03_config/REQUIREMENTS.yaml` and `tmp/validation/03_config/FINAL.md`, production `c41b80b0798be81d9fd6dd77db517c437d821b5e`, tests `5e44ba132f8c5cb025e83ecf51be37a0e8635077`. |
| 6 | Syndrome packing, fragment reassembly and round packing | `decsim/syndrome_ingress.py` | EXISTS, REVIEW PENDING | Source exists; module 25 is pending. `tmp/validation/ORIENTATION.md` identifies reassembly and packing delay as this component's current role. Q-062(e) requires every configured cost to be explicit. |
| 7 | Syndrome Buffer 0 | `decsim/syndrome_buffer.py` | EXISTS, REVIEW PENDING | Source exists; module 18 is pending. The standalone FIFO baseline shape and separately priced controller-to-buffer hop are binding Q-055 work. |
| 8 | Window manager, fires decode when a window is ready | `decsim/window_manager.py` | EXISTS, REVIEW PENDING | Source exists; module 27 is pending. Q-057 binds zero weak/strong coordination in the baseline and keeps the parallel capability off, not deleted. |
| 9 | Decoder manager and weak scheduler, unit assignment and manager-side push | `decsim/decoder_manager.py`; `decsim/schedulers.py`; `decsim/decoder_input_transfer.py` | EXISTS, REVIEW PENDING | Sources exist; schedulers module 16, transfer module 22, and manager module 26 are pending. Push DMA and its priced trigger are later Q-056 work, so the current existence claim is not a Q-056 completion claim. |
| 10 | Weak decoder, memory plus compute engine | `decsim/decoders.py`; `decsim/decoder_cycle_model.py` | Q-062(c) REOPENED | The optional cycle card commits remain as partial work, but the 22:55/22:58 owner amendment rejects lump-cost-only completion. Ordered XQsim-adapted FETCH, DECODE, real EXECUTE, MEMORY, and WRITEBACK engine events plus the authorized decoder-manager dispatch edit are pending. |
| 11 | Pauli frame, minimal weak-correction commit sink | `decsim/pauli_frame.py` | DONE, Q-062(b) | The toggleable final-weak sink is committed at production `efe7270` and tests `bdd0b1f`; 818 accumulated tests, compileall, and the default smoke run passed. The adapted PECOS XOR core carries the owner-required short provenance header. Full Q-054 mapping, lookup/update bandwidth, storage, and capacity axes remain later work. |
| 12 | End-to-end throughput and per-point latency measurement | `decsim/views.py`; `decsim/metrics.py` | EXISTS, REVIEW PENDING; Q-062(d) BUILD PENDING | Both sources exist, but views module 29 and metrics module 30 are pending. `tmp/validation/ORIENTATION.md` defines them as the observation surface. The actual swept loop report is not done until Q-062(d) lands. |

The minimal baseline commit path is therefore: real Stim input, both controller
conversions, packing, Buffer 0, window readiness, manager dispatch, one weak
decode, then the minimal Pauli-frame sink. Per Q-062 and the confirmed meeting
memo, the baseline has no confidence estimator, escalation, reorder buffer, or
weak/strong coordination.

## Deliberately off or later

| Target component | Status | Exact owner source |
|---|---|---|
| Confidence estimator, complementary decode and offline G threshold | LATER, BUILD PENDING | `tmp/validation/OWNER_QUEUE.yaml`, Q-056. Not part of the Q-062 baseline. |
| Strong decoder and strong scheduler | EXISTS, REVIEW PENDING, unused in baseline | `decsim/decoders.py` and `decsim/schedulers.py` exist; modules 21 and 16 are pending. Q-057 and the meeting memo keep escalation off in the baseline. |
| Syndrome Buffer 1, room-temperature parallel copy | LATER, BUILD PENDING | `tmp/validation/OWNER_QUEUE.yaml`, Q-055; `guide/decoder-architecture-meeting-2026-08-17.md`, decisions 2-3. |
| Priced controller-to-Buffer-0 link path | LATER for the current source, BUILD PENDING | Q-053 is complete, but the accepted links record explicitly leaves the Q-055 controller-to-buffer member, edge, and reservation to future scope: `tmp/validation/12_links/REQUIREMENTS.yaml`. Binding directive: `tmp/validation/OWNER_QUEUE.yaml`, Q-055. |
| Reorder buffer and duplicate-drop retirement | LATER, memo only | `tmp/validation/OWNER_QUEUE.yaml`, Q-058. It is deprioritized until measurements justify it, and the ROB-to-window-manager signal is dropped. |
| Predecoding and weak-to-strong handoff | LATER, off by default | `tmp/validation/OWNER_QUEUE.yaml`, Q-060, after Q-056. |
| DMA doorbell-vs-polling trigger | LATER, research and build pending | `tmp/validation/OWNER_QUEUE.yaml`, Q-056. |
| Magic-state factory work for T gates | LATER | `guide/decoder-architecture-meeting-2026-08-17.md`, decision 11. `decsim/factories.py` exists, but module 28 is pending. |

## Evidence ledger

### Current reviewed-component records

| Component | Exact review records | Accepted commits |
|---|---|---|
| Configuration timing card | `tmp/validation/03_config/REQUIREMENTS.yaml`; `tmp/validation/03_config/FINAL.md` | production `c41b80b0798be81d9fd6dd77db517c437d821b5e`; tests `5e44ba132f8c5cb025e83ecf51be37a0e8635077` |
| Links, including Q-053 follow-up | `tmp/validation/12_links/REQUIREMENTS.yaml`; `tmp/validation/12_links/FINAL.md`; Q-053 fields in `tmp/validation/CLEANUP_STATE.yaml` and `tmp/validation/OWNER_QUEUE.yaml` | base production `0dbc4e1`; base tests `730b813`; Q-053 production `b538bc8`; Q-053 tests `207e3ee` |
| Orchestrator | `tmp/validation/13_orchestrators/REQUIREMENTS.yaml`; `tmp/validation/13_orchestrators/FINAL.md` | production `53cfb34`; tests `fa097a9` |
| Planner and rounds split | `tmp/validation/14_planner/REQUIREMENTS.yaml`; `tmp/validation/14_planner/FINAL_BASE.md`; `tmp/validation/14_planner/FINAL_SPLIT.md` | base production `323e321`; base tests `741b76a`; split production `b8d54d2`; split tests `7e19c56` |

### Current source existence and SHA-256

These are existence checks only. They do not promote pending modules to DONE.

| Source | SHA-256 or absence |
|---|---|
| `decsim/frontends/qlx.py` | `d4611425a77958885bbe0431f336b5827516428110392a219d1783c58edaea90` |
| `decsim/adapters/stim_device.py` | `1cd4290dee6ae2c149e848b722141c984230aab7b61aca3ed439f32b594381d5` |
| `decsim/orchestrators.py` | `ce678ba1bd17cf68f577db3f8ce4ec76e611f1aeeeb0e9bde661eb127fd576ad` |
| `decsim/planner.py` | `88963f657f54db88b2d7d3a899895d38bfed0d7ab64fc275e0e1dfb3ee8e2459` |
| `decsim/rounds.py` | `28a14fab95622e9b8c0c39b7979f1ea79baed3a776c74a4171ba999738c27ebd` |
| `decsim/window_manager.py` | `99c0d922e359475d463a59aaedc03f9155d03b45966fa074af1aa713745cd625` |
| `decsim/controller.py` | `07419197da6b80838358b6e80a7186642579c7b8e314bff4e5a6bde2cae2eeb5` |
| `decsim/config.py` | `55ca221be303a4c96ad26f59e42ecbd7903504f1a3865e60f53841be900cc639` |
| `decsim/qpu.py` | `6a99ebb366e692fe9f408d22a798c5c06065cd7ea699b62187b0d221d50877c2` |
| `decsim/devices.py` | `3b7544eeb171712a1795416d492e91e84c5942de569368606295d780e72aa00c` |
| `decsim/syndrome_ingress.py` | `ade686c8f5d6ddb9d9d256892e901f3228cdd70fca1e2fa7b76aa34943eb5c1e` |
| `decsim/syndrome_buffer.py` | `ea1c7874d1d5411b0919cbecc4fc908d983f0710e7bae1957445ed048450f827` |
| `decsim/decoder_manager.py` | `7521a78ec6275cbeffc253b27d1578bec842ffc0c75e3ecba0a9b0ddd8bbdfda` |
| `decsim/schedulers.py` | `1f44978718a8c7303f7089035c2fd1a728c491cfd181a940dc290c147f1acb81` |
| `decsim/decoder_input_transfer.py` | `7cdc4ca5ebeca9afe73b93b269ca7054edc53f6e93a4ea4534e7c1eabe3ffbaa` |
| `decsim/decoders.py` | `5a88d4dae99ac6839b6ab33f891d8b11739e109854ab4fd209533472663c51c2` |
| `decsim/decoder_cycle_model.py` | `e998c3c16f30e95b383395c2d1d4d222bdfaf327ee10f449ffe7bd504c0d1215` |
| `decsim/decoder_cycle_profiles.py` | `8815db9270783afde0486afba5a9afb99378bb69fae37b073ecb1ed42a0e50f4` |
| `decsim/views.py` | `71333dd928d8e7b2bc6e5880356a9eac6f11aaf0fbef3cacae09f1e556ebfb90` |
| `decsim/metrics.py` | `de07758d42ab8b2395e8f70496a8f0ea6e9d2f2d3d5c9146cfeefab182bf69fa` |
| `decsim/links.py` | `3c01b21ff91c7bf2495ed14ad432cb822a7e2435df341143ae6310ba30d38ce3` |
| `decsim/factories.py` | `d66bbb5bc64349dd3b49fc36d6f5d60bf005f8dae9e74872bfe255cfea875426` |
| `decsim/pauli_frame.py` | `a23d0a987a12fd6ab2c8d8e53636d44a6e3dcfd891487121e6595a3497d754aa` |

## Maintenance rule

Update this document at these exact points:

1. after each Q-062(b), Q-062(c), Q-062(d), or Q-062(f) batch lands;
2. after a Q-054 through Q-060 directive changes a component above;
3. after each module 15-31 reaches committed status; and
4. whenever the latest accumulated-suite count changes.

For every update: re-read `tmp/validation/CLEANUP_STATE.yaml`, the applicable
Q-053 through Q-062 entry in `tmp/validation/OWNER_QUEUE.yaml`, and the exact
module `REQUIREMENTS.yaml` plus accepted `FINAL*.md`. Record production and
test commits, refresh hashes only for sources named here, and update both the
component row and the current gate. A source file's existence is never enough
for DONE. Do not copy a test count forward without naming the batch that
produced it.
