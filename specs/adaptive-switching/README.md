# Adaptive switching implementation plan

Status: proposed plan, revised against a full read of the current code and of
Toshio et al. (arXiv:2510.25222v1). No implementation slice has started.

Planning checkout: `decsim_adaptive`, branch `adaptive-decoding-plan`.
Run everything with `../venv_runs/bin/python` (Python 3.13, with the
`deltakit` extra SD6 needs) and `PYTHONPATH` set to this checkout, since that
venv's editable install points at `decsim_runs`. `../venv` cannot take the
extra: its constraints pin matplotlib 3.11.2 and `deltakit-explorer` 0.9.2
needs below 3.11. The `python` on PATH is 3.9 and cannot import `decsim`.

## Goal

Build a hybrid decoding system (union find + cluster gap on every window,
belief matching on escalated windows) whose escalation adapts so the system
keeps up with the QPU and delivers corrections before the operations that
need them, and report the correctness that costs. Every timing number comes
from supplied hardware timing, never from the host running the simulation.

## To-do list

Each step builds one piece and checks it before the next. Steps 1–6 are the
basic case. Tick a box only when its check has passed.

### Step 1: baseline on fixed data (slices 00, 01)

- [x] Environment with `deltakit` for SD6: `../venv_runs` (see above)
- [x] Dataset generator: `decsim/qpu/shot_datasets.py`, `experiments/basic_case/make_datasets.py`
- [x] Generate the 12 datasets (d = 5, 7, 9, 11 × p = 0.002, 0.003, 0.005), each sealed with its manifest (2026-10-19)
- [x] File-backed syndrome source `DatasetStimDevice`; a run on it equals the live error-model source, seed for seed (`tests/qpu/test_stim_device.py`)
- [x] Baseline config `experiments/basic_case/`: weak only, strong only, switching; smoke run on 20 shots matched digests and window layout across the three runs
- [x] Test: every window escalated equals strong only, shot for shot (`tests/escalation/test_switching_mode.py`)
- [x] Latency audit of one shot's timeline against hand formulas (link hops, fetch/release cycles, gap walk, verdict, strong service, next-window wait, same-seed repeat); three fixes and one parameter change came out of it (evidence ledger, 2026-10-19)
- [x] Plot script `experiments/basic_case/plots.py`: Toshio Figs. 15, 16, 4–6 style, window latency, backlog, and who answered each window
- [x] Pilot, the working set for steps 1–10: first 20,000 shots at d = 5 and 7, every p, all three runs, window outcomes recorded (`experiments/basic_case/pilot.py`; array 15815733, fold 15815734)
- [ ] Full runs: 1M shots, all 36 tasks, run here on Della (array 15816344, at most 16 jobs at once, fold 15816345)

### Step 2: timing table (slice 02)

- [x] Placeholder belief-matching times for now (1 us a BP iteration, 10 us a matching, labelled synthetic in `constants.yaml`); no part of the basic case reads the host's clock
- [x] Test: a slower strong tier changes when windows commit, never what they commit or which tier answers (`tests/escalation/test_switching_mode.py`), so error rates hold while the times are placeholders
- [ ] Later, when GPU numbers exist: timing table any decoder can read (exact match, missing cell refuses), and a calibration script

### Step 3: long runs (slice 03)

- [x] Synthetic escalation: `verdicts.Bernoulli`, each window escalating at a rate, drawn from the run's seed (Toshio Sec. III C's i.i.d. switching)
- [x] Simulator timestamps match the hand-computed queue: every commit is Lindley's recursion to the tick, services are Toshio's linear times plus fixed hops (`tests/escalation/test_redo_chain_queue.py`)
- [x] Long horizons without `decsim stream`: one long timing-only shot a stream, its windows read directly (`experiments/backlog/study.py`); stream mode needs a Stim source and stays as it is
- [x] Reproduce the paper's sliding-window backlog result (App. C, Figs. 20–21) and the weak-only bound (Eq. 7): `experiments/backlog/` (evidence ledger, 2026-10-19)

### Step 4: does the fixed threshold keep up? (slice 04)

- [x] Highest escalation rate the hardware sustains, and where 20 dB sits against it: q* = 9 to 17% on the placeholder times; 20 dB escalates 8 to 9% at p = 0.002, 24 to 57% above (evidence ledger)
- [x] Seed offset: `CollectionSettings.first_seed`, so calibration and evaluation shots never overlap
- [ ] Threshold sweep on calibration shots, evaluated on held-out shots

### Step 5: inner loop, hold a target escalation rate (slice 05)

- [x] Rate tracker without audits, warm start from the calibration table: `threshold_sources.RateThreshold`
- [x] Refuse the two broken combinations (extra cluster gap with a moving threshold; deadlines with a learning threshold): `build/escalation.py`, `tests/build/test_escalation.py`
- [x] Tracking and accuracy on held-out shots: achieved share against target, and error rate against the static threshold at the same share: `rate_controller.py`, `controller_plots.py` (evidence ledger)
- [x] Mid-stream change in p: `experiments/basic_case/drift.py`, `drift_plots.py` (evidence ledger)

### Step 6: outer loop, set the target from lag (slice 06)

- [x] Port giving the controller live state: `verdicts.BacklogBernoulli.source_queue`, bound like the deadline's
- [x] Keeps up where fixed settings fall behind when p rises or belief matching slows, timing-only: `throughput.py`, `throughput_plots.py` (evidence ledger)
- [x] The same loop on the real-data stream, setting RateThreshold's target: `drift.py` loop (evidence ledger)

### Later steps

- [ ] Step 7: T gates: gate wait measurement, two hang fixes, T-gate chain on one patch (slice 07)
- [ ] Step 8: lattice-surgery schedules (slice 08)
- [ ] Step 9: double window with belief matching (slice 09)
- [ ] Step 10: speculation (slice 10)

## The basic case in four questions

Everything after these depends on them. Each has its own slice and gate.

| # | Question | Slice | Answered when |
| --- | --- | --- | --- |
| Q1 | What exactly is the baseline, and is it decoded correctly? | 00, 01 | One pinned configuration; weak-only, strong-only and switching decode identical window geometry; equivalence tests pass |
| Q2 | Does the static fixed threshold work, for accuracy and for timing? | 02, 03, 04 | LER-versus-escalation frontier on held-out shots, and the escalation fraction the hardware can sustain, on the same setup |
| Q3 | Can a first-pass controller hold a target escalation fraction online? | 05 | Achieved fraction tracks the target under noise drift, with no simulator truth and no audits |
| Q4 | Can a slower loop set that target from throughput and latency feedback? | 06 | Adaptive beats the best static threshold when service time, noise or gate spacing changes |

## How the paper does each aspect, and how we do it

Section numbers refer to arXiv:2510.25222v1.

| Aspect | Paper | This plan |
| --- | --- | --- |
| Decoders | MWPM or UF weak; belief matching strong (Sec. IV) | Weighted union find weak; belief matching strong. No Relay-BP anywhere |
| Confidence | Cluster gap of Meister et al.: shortest path between inequivalent boundaries on the quotient graph G/C (Sec. II.3) | Same definition (`decsim/confidence/cluster.py`), verified against an independent Dijkstra in slice 01 |
| Protocol | Step 1 runs both decoders in parallel; the variant starts the strong decoder only after the weak verdict (Sec. III.1) | The variant (strong after verdict). Step 1 exists as `Switching(run_both_at_once=True)` and is kept for slice 10 |
| LER experiment | Whole-shot d-round memories, perfect time boundaries, p = 1e-3; Fig. 15 at 20 dB with 1e6 shots; Fig. 17 sweeps 0–50 dB with 1e8 shots | Windowed 10d-round memories (commit d, buffer d), 1e6 shots per point. The whole-shot reproduction (`experiments/switching_paper`) stays as the accuracy anchor |
| Threshold choice | Fixed 20 dB; brute-force sweep for g_op (LER within 1% of asymptote); thresholded LER P_L,th(g_th) ≤ ε·P_L,strong (Eqs. 5–7) | Fixed 20 dB baseline; sweep on calibration seeds, evaluate on held-out seeds; then an online rate tracker |
| Switching rate | γ per d rounds = ∫₀^g_th p(g) dg (Eq. 8); per shot in Figs. 16–17 | Escalation fraction per eligible window, plus per shot for comparison with the paper |
| Backlog | Not simulated with decoders. Recursion over non-Clifford gates, r_i rounds between gates (Eqs. 9–13), linear decode time, i.i.d. switching; Monte Carlo of 1e3 trajectories × 1e4 gates, diverged if r_i > 1e6 (Figs. 11, 14, 21) | Event simulation with each window's own decode time. First validated against the same abstraction (linear service, synthetic i.i.d. escalation), then run with real decodes |
| Windows | Naive sliding window: the weak decoder waits for the strong one (App. C, Fig. 19). Double window: r_strong = r_com + 2 r_buf, weak keeps going (Fig. 12) | Ordinary redo with held boundaries is exactly App. C's naive sliding window and is our baseline. Double window is slice 09 |
| Capacity condition | Weak: τ_weak(r_com + r_buf) ≤ τ_gen r_com (Eq. 16). Double window: τ_strong ≤ γ⁻¹ (d / r_strong) τ_gen (Eq. 14); keep γ under half of Eq. 18 (Fig. 14b). Many patches: γ → γ·N_window | Same weak condition; for ordinary redo, an exact per-window recursion (slice 03) instead of a mean-rate bound |
| Timing values | Ratios only: T_comm,weak = τ_gen, T_comm,strong = τ_strong = 10 τ_gen, d = 21, r_op = 9d | A sourced timing table per decoder, device and window size (slice 02) |

What the paper does not do, and this plan does: decode real windows and time
them in one simulation; measure LER under windowing at 10d rounds; adapt
the threshold online; tie escalation to gate deadlines and tail latency.

## Starting setup

The approved setup is in [parameters.md](parameters.md): union find with
cluster gap, belief matching (20 BP iterations) redoing windows below 20 dB,
sliding windows of d + d rounds with `lookahead` in all three runs, SD6,
10d-round shots at 1 µs per round, distances 5, 7, 9, 11, physical error rates
0.002, 0.003, 0.005, a GPU link of 4.5 µs round trip, and 1,000,000 shots per
point. A pilot on the first 2,000 shots of every point runs first.

## What the current code already does, and what must change

Inspected in the current tree; file references are current.

| Need | Current code | Change |
| --- | --- | --- |
| Ordinary-redo switching | Escalated window re-decoded on the same extent and predecessor boundary; held boundaries block the next window until the strong answer lands (`escalation/strong_regions.py`, `windows/window_boundaries.py`) | None. Add the equivalence tests (slice 01) |
| Equal terminal windows | `WindowSettings` defaults to `flush`; adding a strong tier forces `lookahead` (`windows/settings.py`) | Set `lookahead` explicitly for all three modes |
| Hardware timing | Union find: cycles from the decode's own work. Belief matching: host wall clock unless `IterationTime` is given; the shipped `STRONG_BASELINE_POOL` uses host time. Gap walk: host time unless `walk_microseconds` is set | Timing table and refusal of host time (slice 02) |
| Long-horizon capacity | Collect mode builds a new `Machine` every shot (`experiments/collect.py`), so queues reset every 10d rounds. `decsim stream` runs one long machine but scores only the end readout, refuses window outcomes, has one seed, and its throughput ignores the warmup | LER stays in collect mode; capacity and latency move to stream mode after fixes (slice 03) |
| Escalation metrics | `escalated_windows` counts strong-tier commits; false positives are not computed; seeds always run 0..N−1 | Candidate/admitted/denied counts, outcome classes, seed offset (slices 03–05) |
| Online threshold | `OnlineThreshold`: Gibbs–Candès rate tracker plus an audit lane whose label is weak/strong disagreement | Keep the tracker, drop the audit loop (slice 05) |
| Live state at decision time | `decide_keep(job, result)` sees only the job and its result; the deadline row's `source_queue` port is the only live read | A state port on the threshold source, bound like `source_queue` (slice 06) |
| Gate feedback | `blocked_by` release waits for every earlier window and charges frame write, frame→controller, pulse and controller→QPU; tested hop by hop | Stall measurement, two hang fixes (slices 07–08) |
| Lattice surgery | Operation records carry MERGE/INJECT/MEASURE, rounds, patches, `blocked_by` (`decsim.ops/1`); no merged circuits, so merges are timing-only | Slice 08 |

## Definitions used in every slice

- **τ_gen**: round period. **Commit stride**: `r_com · τ_gen`, the time the QPU
  takes to produce one window's worth of new rounds.
- **Eligible window**: a window's first weak result. **Candidate**: an eligible
  window whose gap is below the threshold. **Admitted**: a candidate that is
  actually sent to the strong decoder. Escalation fraction = admitted / eligible.
- **Outcome classes** on escalated windows (same input, both answers known):
  fixed (weak wrong, strong right), unneeded (both right), broken (weak right,
  strong wrong), both wrong. Kept windows need a strong counterfactual, which
  only diagnostic runs compute. Window labels are local; shot failure is the
  reported correctness metric.
- **Lag**: generated round frontier minus finalized round frontier, sampled over
  time. Throughput is sustained when lag stays bounded over a long horizon.
- **Reaction time**: a window's data complete to its frame commit.
- **Gate stall**: consumer start minus the time it was otherwise ready, for an
  operation released by a decode result. The paper's per-gate overhead is
  `r_i − r_op` rounds.
- **Capacity fraction q\***: the largest escalation fraction for which lag stays
  bounded on the declared hardware.

## Execution order

| Slice | Depends on | Answers | Status |
| --- | --- | --- | --- |
| [00 Baseline contract](slices/00-baseline-contract.md) | None | Q1 | Proposed |
| [01 Correctness](slices/01-correctness-audit.md) | 00 | Q1 | Proposed |
| [02 Timing table](slices/02-supplied-timing.md) | 00 | Q2 timing input | Proposed |
| [03 Capacity and backlog](slices/03-sustained-capacity.md) | 01, 02 | Q2 timing | Proposed |
| [04 Static threshold](slices/04-static-tradeoff.md) | 01, 03 | Q2 | Proposed |
| [05 Rate controller](slices/05-percentage-controller.md) | 04 | Q3 | Proposed |
| [06 Throughput and latency loop](slices/06-adaptive-capacity.md) | 03, 05; 07–08 for gate stalls | Q4 | Proposed |
| [07 Gate feedback](slices/07-gate-feedback.md) | 02, 03 | Gate deadlines on one patch | Proposed |
| [08 Lattice surgery schedules](slices/08-lattice-surgery-schedules.md) | 07 | Gate deadlines from real schedules | Proposed |
| [09 Double windows](slices/09-double-windows.md) | 01–06 | Paper's recovery scheme with belief matching | Proposed |
| [10 Speculation](slices/10-speculation.md) | 07, 08; 09 for double windows | Second contribution | Last, after 00–09 pass |

Slices 01 and 02 can proceed in parallel after 00. Slice 06 runs first on a
memory stream (lag and reaction time) and gains gate stalls as its main input
once 07 and 08 pass. Lattice surgery is on the critical path of the paper's
claims, not an optional extension.

## Global invariants

- Every LER run decodes a saved, hashed dataset (slice 00), so the data
  never changes between runs.
- Online decisions read only what is observable at decision time: no
  simulator truth, no future scores, no service time before completion.
- Correctness compares logical observables, not physical corrections.
- Every committed round has one owner. Overlapping reads add no throughput.
- A consumer waits for every result it depends on.
- Timing runs use stated timing for every component; host time is refused.
- Timing-only runs report no LER. Synthetic or replayed scores are labeled.
- Production LER uses the million-shot rule. Pilots, calibration seeds,
  warmup and timing streams are labeled separately.
- Each completed slice gets a coherent local commit. No push or merge.

## Open decisions

Before slice 00 runs: BP iteration cap, distance/error-rate grid, exact noise
channels, resource topology.

Before slice 04 production: what "the static threshold works" means for this
paper (proposed: switching LER within 10% of strong-only at an escalation
fraction at or below q\*), calibration seed allocation.

Before slice 06: correction-age or reaction budget, control cadence, step
sizes, priority (throughput first, correctness reported).

Before slice 08: whether the paper needs LER through lattice-surgery
operations, or timing and stalls through lattice surgery with LER from memory.

## Research anchors

- [Toshio et al., Decoder Switching](https://arxiv.org/html/2510.25222v1):
  Sec. III.1 protocol, III.2 thresholds, III.3 backlog and Theorem 1, IV
  numerical setup, App. C sliding-window backlog.
- [Meister et al., cluster gap](https://arxiv.org/abs/2405.07433): the soft output.
- [Skoric et al., Parallel window decoding](https://arxiv.org/abs/2209.08552):
  commit stride and boundary dependencies.
- [Helios](https://arxiv.org/abs/2301.08419): union-find cycle constants.
- [SWIPER](https://arxiv.org/abs/2412.05115): speculation comparison, slice 10.

## Evidence ledger

- Outer loop on real data on 2026-10-19 (`drift.py`, job 15844512,
  `results/basic_case_drift`: d = 5 and 7 streams of 40,000 rounds, 5
  seeds, p stepping from 0.002 to 0.003 at round 10,000; the fixed
  threshold, RateThreshold at a fixed 10%, and RateThreshold with the
  outer loop setting its target). After the step the fixed threshold
  escalates 16.9 and 17.0% and falls behind (mean reaction 5,798 and
  2,897 us, growing 0.38 and 0.18 us per us). The fixed 10% target
  escalates 10.4 and 10.7%: at d = 7 it keeps up (89 us, growth -0.003),
  at d = 5 it still falls behind (1,226 us, growth +0.046), q* there
  being 11.5%. The loop steps its target to 7 to 9% at d = 5 and 8 to 9%
  at d = 7, escalates 8.1 and 9.4%, and keeps up at both (217 and 68 us,
  growth -0.016 and -0.001): it holds the case the inner loop alone did
  not. Its threshold falls further than the rate controller's, to 9 to
  15 dB. Accuracy is read off the sweep at the share, as in the
  timing-only study; a stream's one end readout is too few to score.

- Outer loop on the backlog's trend, timing-only, on 2026-10-19
  (`throughput.py`, job 15835849; the same streams as the peak run
  below). The loop keeps up in every case, mean backlog 38 to 75 rounds
  against thousands for a fixed setting past q*, and holds near the
  share the chain sustains: steady, 9.0% at d = 5 (q* 13.5%) and 11.7% at
  d = 7 (q* 17.1%), the latter more than a fixed 10% and so 0.952
  against 0.965% a shot; when p rises to 0.003, 7.8 and 10.1% (q* 11.5
  and 14.2%) where the fixed 20 dB threshold escalates 25 and 27% and
  falls 5,900 rounds behind; when belief matching slows 1.5x, 6.8 and
  7.4% (q* 9.0 and 11.4%) where a fixed 10% falls behind at d = 5 (906
  rounds). Error rates are the sweep's at the share
  (`throughput_plots.py`). It settles under q*, not at it: at d = 7 a
  fixed 10% keeps up after the slowdown, the loop holds 7.4%. Every
  decode time is a placeholder (constants.yaml), so q* and the shares
  move with the GPU measurements.

- Outer loop, timing-only, on 2026-10-19 (`experiments/basic_case/throughput.py`,
  job 15831858: Toshio's random-escalation model at the basic case, each
  tier the pilot's mean time, kept 2.400 and escalated 21.783 us against
  the pilot's 2.386 and 21.769 at d = 5; 40,000 rounds, 10 seeds). The
  model follows q*: fixed 10% keeps up at both d (mean backlog 38 and 27
  rounds), 15% falls behind at d = 5 (q* 13.5%: 2,083 rounds) and keeps
  up at d = 7 (q* 17.1%: 89), 20% falls behind at both. When p rises to
  0.003 the fixed 20 dB threshold escalates 25 and 27% and falls behind
  (5,900 rounds); when belief matching slows 1.5x a fixed 10% falls
  behind at d = 5 (q* 9.0%: 906 rounds). The loop keeps up in every
  case (backlog 14 to 28 rounds), but at both watermark settings it
  ends at its lowest level, 5%, where a fixed 10% keeps up: its signal,
  the most windows waiting in a check, reads the queue's ordinary
  bursts (one escalation holds about 4 windows; at 10% the peak in 100
  windows is often 50) as falling behind. It costs accuracy for no
  throughput: 2.89 against 2.76% a shot at d = 5. Results in
  `results/basic_case_throughput_peak`; the loop now reads the trend
  instead (entry above).

- Rate controller on 2026-10-19 (`experiments/basic_case/rate_controller.py`,
  job 15823266: RateThreshold at 5 and 10%, from the calibrated threshold
  or cold from 20 dB, seeds 20,000 to 29,999, every d = 5 and 7 point).
  Over 100,000 windows a task escalates 4.99 to 5.08% and 9.99 to 10.06%,
  cold or warm alike: the cold start reaches the calibrated threshold in
  a few hundred windows. Paired against the static threshold on the same
  shots, where the two escalate the same share the failures match (d =
  5, p = 0.003, 10%: 929 against 925, 16 and 12 shots only one failed);
  where they differ the run escalating more fails less, as the sweep's
  frontier says (d = 7, p = 0.005, 5.0 against 4.5%: 3,450 against
  3,497). The controller holds a share no fixed threshold can (the gap's
  staircase, below) and costs nothing in accuracy at that share.

- Drift run on 2026-10-19 (`experiments/basic_case/drift.py`, job
  15824064, `results/basic_case_drift_v1`: a d = 5 and a d = 7 stream of
  20,000 rounds, 5 seeds each, p
  stepping from 0.002 to 0.003 at round 10,000, the decoders keeping
  0.002's models). The fixed threshold calibrated to escalate 10% at
  0.002 escalates 8 and 9% before the step and 17% after, past q* at
  0.003 (11.5 and 14.2%): the weak tier's reaction then grows 0.37 (d =
  5) and 0.20 (d = 7) us per us of stream and averages 1,980 and 1,108 us
  over the second half. RateThreshold (target 10%, step 0.25 dB) holds
  9.9 and 10.0% before; after, its threshold falls from about 21 dB
  toward the 15 dB calibrated at 0.003 over 2,000 to 5,000 rounds, the
  share is 11.2 and 11.8% over the second half, and the reaction
  averages 735 and 134 us. At d = 7 the reaction stops growing (-0.01);
  at d = 5 the 10% target sits just under q* = 11.5%, so the backlog the
  transient built still grows (0.08): a target the inner loop holds is
  not by itself a target the chain sustains, which is step 6's question.
  Before the step the fixed threshold escalates 8.0 and 9.0% on the
  stream, against 9.2 and 8.8% on the pilot shots' middle windows (a
  stream's windows are all middle windows; the last window of a shot,
  closed by the noiseless readout, escalates 4.2 and 1.4%). It is
  chance: 20 streams of 20,000 rounds at d = 5 (no drift, each window's
  gap recorded) escalate 9.08 +- 0.12% against the shots' 9.19 +- 0.05%,
  with 10.73 +- 0.02 detection events a window against 10.78 +- 0.02.
  Seeds 1 to 5, the drift run's, sit low (8.5 to 9.1%), and a stream's
  share spreads 0.54% from seed to seed, over the binomial 0.45%, since
  neighbouring windows share rounds. Shots and streams agree, so a
  share's accuracy measured on shots holds on a stream.

- Held-out thresholds on 2026-10-19 (`threshold_heldout.py`, seeds
  20,000 to 39,999, all 18 tasks 20,000 shots): the cluster gap takes
  only multiples of union find's weight step, 0.5 nats (2.17 dB), so a
  fixed threshold's escalated share is a staircase. At d = 5, p = 0.002
  the pilot's gaps put 8.7% of windows at or under 4.5 nats and 14.4% at
  or under 5.0: every threshold in (19.5, 21.7] dB escalates 8.7%, none
  10%. The thresholds interpolated for 5, 10 and 20% escalate 4.0 to
  7.0, 8.1 to 10.4 and 15.6 to 20.5% held out; where a target falls
  inside a step no fixed threshold reaches it. A rate tracker reaches it
  on average by moving across the step, which a fixed threshold cannot.

- Threshold sweep on 2026-10-19 (`experiments/basic_case/threshold_sweep.py`,
  seeds 0 to 19,999, d = 5 and 7, 0 to 50 dB): the 20 dB cell repeats the
  pilot's switching run shot for shot (120,000 of 120,000 digests,
  failures and escalations); 0 dB is union find alone and 40 to 50 dB is
  belief matching alone at every point. No threshold beats belief matching
  alone (Toshio reports a dip for UF + BM; none here). Within 1% of it
  (g_op) is about 35 dB, escalating 27 to 29% at p = 0.002. With the
  placeholder belief-matching times the chain sustains q* = 13.5% (d = 5)
  and 17.1% (d = 7) at p = 0.002, 11.5 and 14.2% at 0.003, 8.6 and 12.0%
  at 0.005; at q* switching fails 5, 8, 11, 24, 19 and 29% more shots
  than belief matching alone. A fixed threshold that keeps up costs
  accuracy that grows with p, the case for steps 5 and 6. The calibrated
  thresholds run on held-out seeds 20,000 to 39,999
  (`threshold_heldout.py`).

- Backlog study on 2026-10-19 (`experiments/backlog/`, job 15816899: d =
  21, Toshio's linear decoders, a timing-only stream of 2e5 rounds, 10
  seeds a point). Weak alone, the last window's backlog over the stream's
  rounds is the queue's load - 1 at every tau_weak, 0 up to Eq. (7)'s
  tau_gen / 2 (0.496: 0; 0.6: 0.201 against 0.200). With escalation the
  streams follow load - 1 above load 1. The gamma where load is 1, against
  Fig. 21 a's divergence threshold:

  | tau_weak / tau_gen | r_com | load 1 at gamma | Fig. 21 a | ratio |
  | --- | --- | --- | --- | --- |
  | 0 | 21 | 2.47e-2 | 2.5e-2 | 0.99 |
  | 0.1 | 21 | 1.97e-2 | 2e-2 | 0.99 |
  | 0.4 | 21 | 4.90e-3 | 5e-3 | 0.98 |
  | 0.7 | 59 | 9.74e-4 | 7e-4 | 1.39 |
  | 0.9 | 378 | 2.49e-4 | 1.5e-4 | 1.66 |

  This holds when gamma_switch is per d rounds the weak decoder reads,
  r_com + r_buf; Theorem 1's proof counts r_com only, which would put
  every threshold 2x higher at r_com = d. The figure's values are read off
  its grid. Where the paper is lower, r_com is 59 and 378 rounds: one
  escalation adds 800 to 4,000 us of strong decode against 5% slack, and
  Fig. 21 calls a trajectory diverged at 1e6 rounds within 1e4 gates with
  each gate waiting for its feedback; this stream has neither gates nor
  that cap (step 7).
- Pilot on 2026-10-19 (20,000 shots, d = 5 and 7): in every one of the
  20,000 switching shots at d = 5, p = 0.002 a shot fails exactly when an
  odd number of its windows disagree with their label, and 1,158 of the
  1,170 shots with two such windows have them adjacent: a chain across a
  commit boundary that the label and the decoder put in neighbouring
  windows. A window's "wrong" is that bookkeeping, not a logical failure;
  shot rates are the measure.

- Latency audit on 2026-10-19 (basic case, d = 5, p = 0.005, 6 seeds, all
  three runs): every link hop is latency plus bits over rate, union find's
  fetch is 1 + rounds cycles and its release 3 cycles at 100 MHz, the gap
  walk is 1 us, the verdict 1 chip cycle to keep and 2 to escalate, the
  frame write 1 chip cycle, the next window starts when its boundary lands,
  and a repeat of a seed repeats every timestamp. Four things it found:
  - A redo window's pin on a union-find commit left only after the
    escalation landed on the GPU, a 2.25 us leg the chip could not know of.
    It now leaves with the escalation (Toshio Fig. 12 caption, App. C); a
    pin on a GPU commit is still read when the escalation lands.
  - Belief matching always ran the matching. Higgott et al. 2203.04948
    Algorithm 1 and beliefmatching 0.2.0 (the library Toshio App. B used)
    return BP's own answer when BP converges; decsim now does too, agrees
    with the library on 900 of 900 shots, and charges the matching's time
    only when it runs.
  - With weight step 0.1 nats, edges are 32 to 89 ticks and union find
    alone falls behind the rounds at every point (mean decode 4.5 to 10.6
    us per window against d us of new rounds). Helios grows edges of 2 to
    w_max ticks and tests w_max up to 16 (2301.08419 Sec. VI-C). Step 0.5
    gives 6 to 16 ticks, fails as often (d = 5: 246 against 244 of 5,000;
    d = 7: 100 against 98 of 5,000, p = 0.002; d = 5, p = 0.005: 703
    against 691 of 2,000) and decodes about 4x faster. The basic case uses
    0.5. Union find then keeps up at every point: mean 1.1 to 3.5 us, the
    worst window under d us.
  - The cluster gap counted a walk that crosses the logical in the buffer
    and leaves by the window's open future boundary, which changes nothing
    the window commits (it commits its owned faults only). Escalation sat
    at about 20% of windows at both d = 5 and d = 7 (p = 0.002). Counting
    only owned faults' logical bits (Meister et al. 2405.07433 Definition 9:
    a path counts when it is a logical of what is decoded) gives 9.0% and
    7.1%, falling with d as Toshio Fig. 16 does, with the same failures
    (5 and 0 of 200).

- Current-tree review on 2026-10-18: escalation, windows, decoders, links,
  confidence, experiments, observe, controller and frontends tests pass
  (about 1,500 tests) with `../venv/bin/python`.
- Ordinary redo, every window escalated, matched strong-only shot for shot
  (0 of 215 shots differ, d = 3 and 5); never escalated matched weak-only.
  Scratch evidence, not yet a repo test.
- The d = 13, p = 0.005 result where switching failed more shots than either
  decoder came from Experiment 1 (double window, Relay-BP measured table,
  unconverged answers kept). It is not evidence about belief matching or
  ordinary redo.
