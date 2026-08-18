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
- Q-062(c) is **DONE**: `decsim/decoder_engine.py` holds `DecoderEngine`,
  simulated timing around one functional decoder: configured stages before
  the algorithm (for a weak ASIC, fetching the window out of the decoder-side
  memory, cycles per round at the clock), the wrapped decoder's own latency
  with its result available when that time ends, then configured stages
  after (release). Stages are data with free names, one engine event and one
  record each; the manager dispatches through `run(job, engine, on_done)`.
  Owner-session implementation 2026-08-18. Suite 826 passed, smoke identical.
- Q-062(d) and (f) are **DONE** (owner session, 2026-08-18): `experiments/baseline_closed_loop.py`
  with its single config `experiments/baseline_closed_loop.yaml` runs real Stim
  rotated memory data through the whole loop and reports latency at thirteen
  points, throughput, utilization, backlog and LER per sweep point
  (`experiments/results/baseline_closed_loop/sweep.md`); `experiments/baseline_anchor.py`
  reproduces PyMatching v2's published microseconds per shot on this host and
  checks the windowed loop's logical error rate against whole-circuit PyMatching
  (`anchor.md`). The controller-to-Buffer-0 hop is a priced optional link (C2B).
  Finding: with the reference link cards the serial sliding-window chain (CWD
  2 us + decode + DD 0.5 us per window), not the ASIC decoder, bounds throughput:
  1.16 rounds/us sustained at d=3, knee near a 0.86 us round period (after Q-063).
- Decoder-side shape after the owner rulings of 2026-08-18 (commits `cd43fdb`,
  `1d8b77f`, `b3c8943`, `ebae88f`): assign-then-transfer (a job is queued, a
  numbered unit is assigned, then its input is transferred into that unit's
  own `DecoderMemory`, then service starts); decoder memory is per unit, never
  a pool; the software decoder row uses the measured wall clock of each real
  PyMatching call (`PyMatchingDecoder(latency_model=None)`), the only place
  wall clock enters as modeled time; `cancel` aborts stages. Suite 765 passed,
  smoke rebaselined for assign-then-transfer.
- Willow replication (owner request 2026-08-18, final at `a7336f2`):
  `RecordedStimDevice` replays Google's Zenodo 13273331 hardware detection
  events through the loop; `experiments/willow_replication.py` reports the loop
  identical shot for shot to whole-shot PyMatching (1999/2000, 1000/1000,
  500/500 at d = 3, 5, 7; the one difference is an equal-weight tie),
  epsilon per cycle 1.086 / 0.748 / 0.509 % (Lambda 1.47 vs 1.49 whole-shot;
  Google's released decoder 1.76, paper 2.14, a decoder-quality gap), and real
  time at d = 5, 1.1 us cycle: single-thread PyMatching measures 22 us median
  per window on this host, so serial windows sustain 0.19 rounds/us and
  parallel A/B with two units 0.56 against the 0.91 required, while a
  LILLIPUT-class ASIC keeps up at 0.90 with 3.1 us (serial) or 10 us
  (parallel) latency; the paper's 63 us is a multi-worker streaming decoder
  (`experiments/results/willow_replication/report.md` and four plots).
- External validation gates (`experiments/results/validation/`): Gate 1 qLDPC
  `SlidingWindowDecoder` (window 2d, stride d) vs decsim, same windows and commit
  sets, 2998/3000 shots identical (`validate_windows_qldpc.py`); Gate 2
  SWIPER-SIM sliding-window timing vs decsim with every link at zero latency,
  every interior window identical, tail convention differs by one window
  (`validate_timing_swiper.py`, commit `adfb522`); Gate 3 Fusion Blossom exact
  MWPM vs decsim's PyMatching on the same window graphs, 2700/2700 same weight
  and logical class (`validate_mwpm_fusion_blossom.py`). Gate 2 also restored
  the reference links one at a time: CWD and DD accumulate on the serial window
  chain, nothing else does.
- Q-063 (owner GO 2026-08-18, commits `2e4f068`, `b0d4ff2`): the boundary a
  window hands to its successor (residual defects at the commit edge, decoder
  state) leaves over DD at decode done; the WDO delivery and frame commit run
  downstream, off the window-to-window path. Grounded in Skoric 2209.08552
  App. D, Tan 2209.09219, LILLIPUT 2108.06569, qLDPC net_error, SWIPER's DAG.
  Baseline sweep and Willow real-time rows rerun; suite 768, smoke identical.
- Q-064 (owner GO 2026-08-18, commits `7a80d3a`, `5312f08`): the QPU runs one
  QEC cycle clock; every live patch yields a syndrome round every cycle, idle
  or not, tagged (patch, cycle); operations start on cycle boundaries and
  consume whole cycles. Grounded in Google readout per cycle (2207.06431,
  2408.13687), Skoric App. D, SWIPER device_manager `_generate_syndrome_round`,
  XQsim chip-wide ESM. Removed: per-op private round chains, feedback-memory
  idle loop, `max_idle_rounds`, `gates_start_on_round_boundaries`. Verified by
  tests/16_qpu, a decsim T-gate wait (14 idle rounds for a 14-round decode,
  successor starts on the boundary) beside SWIPER's RegularTSchedule rule,
  Gate 1 and the baseline sweep unchanged; suite 772, smoke identical.

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
| 9 | Decoder manager and weak scheduler, unit assignment and manager-side push | `decsim/decoder_manager.py`; `decsim/schedulers.py`; `decsim/decoder_memory_transfer.py` | EXISTS, REVIEW PENDING | Sources exist; schedulers module 16, transfer module 22, and manager module 26 are pending. Push DMA and its priced trigger are later Q-056 work, so the current existence claim is not a Q-056 completion claim. |
| 10 | Weak decoder, memory plus compute engine | `decsim/decoder_engine.py` (`DecoderEngine`, `DecoderTiming`, `DecoderStage`, stage records) | DONE, Q-062(c) | Stages before the algorithm (fetch from decoder-side memory, cycles per round), the real decoder latency and decode() at its completion, stages after (release); every stage an engine event with start/end ticks; one job per unit. Hardware decoders declare their own named stages as data. |
| 11 | Pauli frame, minimal weak-correction commit sink | `decsim/pauli_frame.py` | DONE, Q-062(b) | The toggleable final-weak sink is committed at production `efe7270` and tests `bdd0b1f`; 818 accumulated tests, compileall, and the default smoke run passed. The adapted PECOS XOR core carries the owner-required short provenance header. Full Q-054 mapping, lookup/update bandwidth, storage, and capacity axes remain later work. |
| 12 | End-to-end throughput and per-point latency measurement | `decsim/views.py`; `decsim/metrics.py`; `experiments/baseline_closed_loop.py` | DONE for the baseline experiment, Q-062(d); views/metrics review pending | Both sources exist, but views module 29 and metrics module 30 are pending. `tmp/validation/ORIENTATION.md` defines them as the observation surface. The actual swept loop report is not done until Q-062(d) lands. |

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
| `decsim/window_manager.py` | `27208fc1045c8700ef8bdaa25ec79235d08cb7e7d533fffeb991e4768ab0ec42` |
| `decsim/controller.py` | `07419197da6b80838358b6e80a7186642579c7b8e314bff4e5a6bde2cae2eeb5` |
| `decsim/config.py` | `55ca221be303a4c96ad26f59e42ecbd7903504f1a3865e60f53841be900cc639` |
| `decsim/qpu.py` | `6a99ebb366e692fe9f408d22a798c5c06065cd7ea699b62187b0d221d50877c2` |
| `decsim/devices.py` | `3b7544eeb171712a1795416d492e91e84c5942de569368606295d780e72aa00c` |
| `decsim/syndrome_ingress.py` | `ade686c8f5d6ddb9d9d256892e901f3228cdd70fca1e2fa7b76aa34943eb5c1e` |
| `decsim/syndrome_buffer.py` | `ee199a1bb7aeac7b79cd1bc44e2ae38df6ca5332b04c092bbbcab0bf67d7c6f2` |
| `decsim/decoder_manager.py` | `45954f75197e1f1939a055cb116c9f605883276b73d99272c24f326070808dea` |
| `decsim/schedulers.py` | `1f44978718a8c7303f7089035c2fd1a728c491cfd181a940dc290c147f1acb81` |
| `decsim/decoder_memory_transfer.py` | `171760e1c268b23ffe6e8f6e45e8c33bc091d34d6671f5e4bb4cdacbc1ad2f2c` |
| `decsim/decoders.py` | `63c2b7522aa65d977c5baac90d60450fcbb7e6ee38bb670a428cb56d71931cf8` |
| `decsim/decoder_engine.py` | `f465f213274503c9df4e5a19eb6d5800ad112161bf3e1518d393225d5aa10e15` |
| `decsim/views.py` | `09d776107fd0681e38ea49cd3fb6364ba2fc41c7108799793fc7aec890f6fe85` |
| `decsim/metrics.py` | `181ecb1a6e3d8e5fce1f583d60bd6963d78d7d17b408e3a0a63c5934728d9b03` |
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
