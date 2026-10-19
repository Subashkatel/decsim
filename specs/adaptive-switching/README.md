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
- [ ] Generate the 12 datasets (d = 5, 7, 9, 11 × p = 0.002, 0.003, 0.005): Slurm jobs 15812841 and 15812928, then seal each point
- [x] File-backed syndrome source `DatasetStimDevice`; a run on it equals the live error-model source, seed for seed (`tests/qpu/test_stim_device.py`)
- [x] Baseline config `experiments/basic_case/`: weak only, strong only, switching; smoke run on 20 shots matched digests and window layout across the three runs
- [x] Test: every window escalated equals strong only, shot for shot (`tests/escalation/test_switching_mode.py`)
- [x] Latency audit of one shot's timeline against hand formulas (link hops, fetch/release cycles, gap walk, verdict, strong service, next-window wait, same-seed repeat); three fixes and one parameter change came out of it (evidence ledger, 2026-10-19)
- [x] Plot script `experiments/basic_case/plots.py`: Toshio Figs. 15, 16, 4–6 style, window latency, backlog, and who answered each window
- [ ] Pilot: first 2,000 shots of every dataset, all three runs; window layout, failure counts, run time per point
- [ ] Full runs: 1M shots, 36 jobs, run here on Della

### Step 2: timing table (slice 02)

- [x] Placeholder belief-matching times for now (1 us a BP iteration, 10 us a matching, labelled synthetic in `constants.yaml`); no part of the basic case reads the host's clock
- [x] Test: a slower strong tier changes when windows commit, never what they commit or which tier answers (`tests/escalation/test_switching_mode.py`), so error rates hold while the times are placeholders
- [ ] Later, when GPU numbers exist: timing table any decoder can read (exact match, missing cell refuses), and a calibration script

### Step 3: long runs (slice 03)

- [ ] `decsim stream` fixes: warmup excluded, lag over time, p50/p99, several seeds
- [ ] Simulator timestamps match the hand-computed queue on the same inputs
- [ ] Reproduce the paper's sliding-window backlog result (App. C) with synthetic escalation

### Step 4: does the fixed threshold keep up? (slice 04)

- [ ] Highest escalation rate the hardware sustains, and where 20 dB sits against it
- [ ] Threshold sweep on calibration shots, evaluated on held-out shots

### Step 5: inner loop, hold a target escalation rate (slice 05)

- [ ] Rate tracker without audits, warm start from a lookup table
- [ ] Refuse the two broken combinations (extra cluster gap with a moving threshold; deadlines with the online controller)

### Step 6: outer loop, set the target from lag (slice 06)

- [ ] Port giving the controller live state
- [ ] Beats fixed thresholds when p drifts or belief matching slows

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
