# Per-window experiment: log

This file records everything about this experiment: why we run it,
what we decided and why, what we ran, with which code and parameters,
where the data is, and how to run it again. Newest entries go at the
bottom of each section. The plan is discussed step by step on the page
"Per-Window Experiment Plan" and copied here once decided.

## Why this experiment

The meeting of 2026-10-08 asked us to understand the data behind two
plots before going further:

- the logical error rate plot of Experiment 1 (switching against
  union-find alone, per round, d=5 to 13);
- the per-shot bar chart (how often union-find alone, Relay-BP-5 alone
  and switching fail on the same shots).

The two plots looked very different, and at d=13 switching failed many
more shots than union-find alone. The meeting asked: is the cluster gap
making wrong switching decisions, or is something else going on? To
answer it we need, for every window, the true answer and what each
decoder gave. The old runs do not save that, so we run again.

## What we knew before this experiment

Details and code are on branch `exp1-switching-baseline`
(`EXPERIMENTS.md`, `results/`).

- The two plots use the same runs and the same shots. They differ in
  what they show: the line plot is the error rate per round on a log
  scale over all shots of each run; the bars are % of shots failed
  (100 rounds each) on the shots the three runs share. Example, d=13 at
  0.005: union-find alone fails 152 of 4,000 shots (3.8%) in the line
  plot's data and 39 of 1,000 (3.9%) in the bars'.
- At d=13 with physical error rate 0.005, 17% of the windows sent to
  Relay-BP-5 do not converge, and decsim uses the unconverged answer.
  In 121 of the 122 shots that only switching fails, one of these
  windows is present.
- These windows are large (39 rounds, about 6,500 detectors by 131,000
  fault columns). A valid answer exists for each one we checked;
  Relay-BP-5 stops before it finds it (it finds it with 3,000 legs).
- The strong decoder falls behind at every setting of Experiment 1.

## Decisions

All decisions by the owner.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-10-08 | Stop Experiments 1, 2 and 3, fold them, push the final tables, then clean up disk space. | The points left were the slowest and least useful, and the runs save no per-window truth. |
| 2026-10-08 | A new run, not the old data. | The old runs save no true answer per window. |
| 2026-10-08 | Three configurations on the same shots: union-find alone, Relay-BP-5 alone, switching (option A). | Union-find alone and Relay-BP-5 alone use the same windows, so their answers line up window by window, without running Relay-BP on every kept window. |
| 2026-10-08 | Switching keeps Relay-BP's answer, even when it did not converge. No fallback to union-find. | We want to see today's behavior. The tables show what each decoder gave. |
| 2026-10-08 | Save Tesseract's answer on every window as a check. The machine never uses it. | A third opinion on each window. |
| 2026-10-08 | Work on a new branch from main, with this log. | Anyone can repeat the run. |
| 2026-10-08 | Four configurations on the same shots: union-find alone, Relay-BP-5 alone, switching, Tesseract alone. This replaces "Tesseract as a check on every window". | Each configuration saves its own answer per window. Tesseract alone shows what a more accurate decoder gives on the same windows and shots; it is not part of the switching design. |
| 2026-10-08 | d = 5, 7, 9, 11 and physical error rates 0.002, 0.003, 0.004, 0.005. d=13 later. | Relay-BP-5 starts to not converge at d=9. The configuration id leaves out the sweep and the stop rule (`decsim/experiments/run_folder.py`), so new points can be added later with the same commit. |
| 2026-10-08 | No pilot. | The time per shot at d=5 to 11 is already measured in Experiments 1 and 3. |
| 2026-10-08 | All parameters in one experiment file, `experiments/per_window/run.py`, as main does since the yaml path was deleted (`526f7f64`): named constants at the top, each with its source. The analysis scripts hold no numbers. | One place to read and change every parameter. A task's id hashes its machine and metadata, not its stop rule (`decsim/experiments/collect.py`), so new distances or rates can be added later with the same commit. |
| 2026-10-08 | Stop rule, option A: at each setting, one fixed number of shots N, the same for all four configurations; N gives union-find alone about 100 failures (N = 100 / its rate in Experiment 1). A rate is shown only with at least 20 failures, with 95% Wilson intervals. Keep d=11 at 0.002. | Papers that compare decoders use one fixed shot count for all of them (see Stop rule research). 100 failures gives about +-20% at 95%. Estimated cost about 9,100 core-hours, 6,750 of them at d=11, 0.002. |
| 2026-10-08 | Keep Experiment 1's machine and every latency as they are, on today's main (newer code than Experiments 1 and 3). | The results compare with Experiment 1. The run is about correctness, so the timing estimates stay. |
| 2026-10-09 | The true label of a window comes from the errors that fired: a new shot source draws each shot from Stim's error model with `sample(return_errors=True)` and emits the raw row that forms exactly those events. | 2509.03815 labels windows by the flipped edges (lines 497-508); Gong et al. sample their windows from Stim's error model (`SlidingWindowDecoder/osd.py:124-125`). A measurement sample cannot say which errors fired. The shots are new, so they are not Experiment 1's shots. |
| 2026-10-09 | `decsim run --slurm` packs fixed-shot pieces into one job array: pieces of at most a twentieth of a core's budget, dealt longest first to the least loaded core, every job asking the busiest core's work over 0.8 (at least 61 minutes). 3 cores a job. | Princeton Research Computing asked for 80% use of requested walltime (our jobs used 28%) and walltimes of at least 1:01:00. Graham 1969 bounds each core within one piece of the mean. Della's short QOS: 400 jobs and 1000 cores per user. |
| 2026-10-09 | Run with the host venv `tmp/deltakit-integration/upstream/.venv` and Tesseract from `tmp/runs/2026-10-07_strong_only_pydeps`, as Experiments 1 to 3 did. | The sbatch line holds the launching interpreter; `decsim/.venv/bin/python` is an Apptainer wrapper whose interpreter path does not exist on compute nodes. |
| 2026-10-09 | Tesseract alone uses the package's exact short-beam profile: beam 15, climbing, no revisits, queue 200,000, 16 orders by index, merge errors, order seed 2384753. | `tesseract-decoder` 6a260b0, `src/tesseract_sinter_compat.pybind.h:468-474`, the profile the Tesseract paper runs. |

## Plan

Each step is decided before the next. Status on 2026-10-09:

| Step | What we decide | Status |
| --- | --- | --- |
| 1. Questions | what the run must answer | discussing |
| 1b. Literature check | which papers ground each piece | done, see below |
| 2. Definitions | what right and wrong mean for a window | not started |
| 3. Records | what decsim saves per shot, per window, per strong window | built, see Records |
| 4. Configurations and settings | decoders, distances, error rates, stop rule, every latency | decided |
| 5. Checks before the run | how we prove the records are right | done, see Records; outside review running |
| 6. Cost | time per shot (from Experiments 1 and 3), core-hours, job plan | done, see Cost; no pilot |
| 7. Run system | how the run is started, folded, logged, pushed | built, see Cost; waits for the owner's go |
| 8. Analysis | the plots, named before the run | not started |

### Step 1. Questions (draft)

| # | Question | Level |
| --- | --- | --- |
| Q1 | How many windows does each configuration decode per shot? Are union-find alone and Relay-BP alone exactly equal? | per shot |
| Q2 | How many windows does union-find get right, and how many wrong? | per window |
| Q3 | How often does switching escalate a window? | per window |
| Q4 | Is each switching decision right? Escalated and union-find wrong: correct escalation. Escalated and union-find right: false positive. Kept and union-find wrong: false negative. Why: the gap value and detection events of each case. | per window |
| Q5 | Is Relay-BP right on the windows it gets? Did it fix union-find's error or break a right answer? Did it converge? | per strong window |
| Q6 | Which wrong windows make the shot fail? | window to shot |
| Q7 | Latency of kept and escalated windows, and strong decoder load (second priority). | per window |

## Literature grounding (2026-10-08)

Every piece we plan to measure follows a published method. No paper
combines them: per-window truth inside sliding windows, under
circuit-level noise, with weak-to-strong escalation. That combination is
our own, built only from these grounded pieces. Line numbers refer to
the text copies in the sandbox (tmp/papers/txt/<arXiv id>.txt), each
quote checked word for word.

| What we measure | Method we follow | Source (lines) |
| --- | --- | --- |
| True answer of a window (form A) | parity of the true errors in the window's own region on the logical operator; the windows' labels add up to the shot's | Zhang et al. 2509.03815, Eq. 1 (499-508, 576-580) |
| Per-window error rate | Pr[window answer differs from its label] | 2509.03815 (1073-1077) |
| The seam trap | a window can be wrong on its own label while the shot is right | 2509.03815 (1130-1138); Tan et al. 2209.09219 (921-929) |
| Prefix right (form B) | after each commit, the residual (true errors plus all corrections so far) is judged by an ideal decoder | Huang and Puri 2311.03307 (214-225); Sriram et al. 2608.10081 (350-354) |
| Window error that makes the shot fail | window decoding fails where global decoding does not | Mishima, Toshio et al. 2605.14637 (474-479, 904-908) |
| Weak error rate per gap | gap labelled by whether the decoder is right | Toshio 2510.25222 (803-808, 820-828); Gidney et al. 2312.04522 (894-900) |
| Kept but wrong (false negative) | accepted by the decoder but corrected into a logical error | Smith et al. 2405.03766 (1187-1193); Meister et al. 2405.07433 (1312-1315); Toshio P_th (880-897) |
| How well the gap ranks failures | AUC against true failures | Dentelski 2606.08758 (534-540) |
| The words false positive and false negative | as there, but judged against the truth, not a reference decoder | Viszlai et al. 2412.05115 (574-583) |

Known limit: neighbouring windows overlap, so they are not independent
(Dinca et al. 2512.15689, 1076-1088); Toshio's per-window rates assume
they are. We report both.

## Records (2026-10-09)

What decsim keeps per window, and how (branch `per-window-experiment`):

- `ErrorModelStimDevice` (`decsim/qpu/stim_device.py`): draws each shot
  from the circuit's error model and keeps the errors that fired; the
  raw measurement row it emits is solved back from the drawn events
  (`detector_formation.measurements_forming`), so the rest of the
  machine runs unchanged.
- A fired error belongs to the window whose commit rounds hold the
  round of its earliest detector, the rule decsim's windows already use
  to own faults (`window_placement.py`).
- `observation.record_window_outcomes` writes `window_outcomes.csv`:
  one row per window that delivered the shot's answer (the logical
  ledger's tiling, so a double window is one row). Columns: the window,
  its rounds, tier, decode status (converged or not), detection events,
  answer, true label, right or wrong, and on switching runs the gap,
  escalated, and the weak answer and label of its own rounds.
- Checks: Stim's own converter turns the emitted rows back into the
  drawn events (test); the labels of every shot add up to its truth
  (asserted in every shot, 2509.03815 Eq. 2); a shot fails exactly when
  its answers and labels differ in total (test, and the smoke run).

## Cost (2026-10-09)

`experiments/per_window/run.py` holds N per setting (100 / union-find
alone's Experiment 1 rate) and each task's core seconds a shot
(Experiment 1 or 3, same configuration and setting). Core-hours by
setting, all four configurations:

| d \ physical error rate | 0.002 | 0.003 | 0.004 | 0.005 |
| --- | --- | --- | --- | --- |
| 5 | 13 (N 4,041) | 7 (1,504) | 4 (705) | 3 (420) |
| 7 | 106 (17,836) | 30 (3,774) | 14 (1,274) | 8 (616) |
| 9 | 890 (88,496) | 138 (10,321) | 47 (2,516) | 27 (996) |
| 11 | 6,699 (392,157) | 740 (34,800) | 170 (5,240) | 114 (1,527) |

Total about 9,000 core-hours. Dry run (`--slurm --dry-run`): 15,185
pieces; with `--hours 24`, 165 jobs of 3 cores at 22:47:00 (495 cores);
with `--hours 12`, 330 jobs at 11:24:00 (990 cores). Every job is
planned at 80% use. The largest Experiment 1 piece used 4.6 GB, so three
processes fit in 16 GB.

## Machine and latencies (2026-10-08)

Every value is Experiment 1's, written with its source in
`experiments/switching_baseline/run.py` on main. Values marked
estimate or chosen have no paper behind them.

| Part | Value | Source |
| --- | --- | --- |
| Chip clock | 250 MHz | 2605.04892 line 1063 |
| Round period | 1 us | chosen in Experiment 1, no source |
| Rounds per shot | 100 | Experiment 1 |
| Packing a round | 8 cycles (32 ns) | 2603.16203 lines 894-895 |
| Forming detection events | 5 cycles, then 1 a round | 2605.04892 lines 1273-1275 |
| Uplink to the weak buffer | 40 cycles, 4 lanes of 38.79 bits | 2603.16203 lines 895-897, 971-975 |
| Weak buffer to union-find | 1 cycle | estimate (2603.16203 lines 668-670) |
| Window decision | 5 cycles (20 ns) | 2603.16203 lines 897-899 |
| Union-find | Helios cycle law at 100 MHz, delay 3 cycles | Helios RTL 2dda998; 2406.08491 line 1230 |
| Union-find weight step | 0.5 | estimate |
| Correction to the frame | 41 cycles (164 ns); frame write 1 cycle | 2603.16203 lines 903-904; 2605.04892 Table I |
| Cluster gap walk | 1.0 us | chosen, estimate (2602.03336 lines 17-19) |
| Threshold | 20 dB on the cluster gap; compare and switch 1 cycle each | chosen; cycles estimate |
| Strong window | double window, re-reads 1 buffer region | 2510.25222 Sec. III C, Fig. 12 |
| Each cable leg to the strong host (4 legs) | 1.151 us | 2609.09270 Table III, lines 1611, 1627 |
| Relay-BP-5 keys | gamma0 0.35, interval [-0.254, 0.985], 600 legs of 60 after 80, stop at 5 solutions | 2506.01779 lines 307, 332, 343 |
| Relay-BP-5 time | measured A100-SXM4-80GB line in iterations, nearest region in detectors | `decsim/decoders/measured_table/measurements.py`, Slurm job 14676845 |
| Tesseract | its own wall clock; only its answers and error rate are results | as Experiment 3 |
| Decision to pulse | 8 cycles | QubiC, estimate |

The A100 table was measured on regions of 3d rounds at physical error
rate 0.001. A window of another size is priced by the nearest measured
region, at its own iteration count.

## Stop rule research (2026-10-08)

| Source | Rule | Lines |
| --- | --- | --- |
| Toshio 2510.25222 | one fixed shot count for every decoder: 10^6 (threshold plot), 10^7 to 10^8 (gap plots) | 741, 793, 1690, 1733 |
| Dentelski 2606.08758 | the same 10^8 shots per configuration; 95% intervals; a point needs 20 failures | 381-383, 1165-1167 |
| Gidney 2312.04522; Dinca 2512.15689 | fixed 10^9 or 10^8 shots; Wilson intervals | 864, 896; 348-351, 494 |
| Liang 2406.08491 (Helios) | fixed 10^6 to 10^8 trials | 1312, 1322 |
| sinter (Stim) | stop a task after max_errors failures; its README uses 1,000 | `_collection.py:79`, README 85 |
| Tesseract and Chromobius READMEs | max_errors 100 | README 363; README 89 |
| Fowler 1110.5133; Bravyi 2208.04660 | until 10,000 failures; until 10 | 416; 443 |

No paper uses exactly 100 failures; 100 is the value of Experiment 1.

## Runs

None yet. Each run gets an entry here before it starts: the date, the
commit, the configs, the parameters, the Slurm job ids, the data folder,
and the exact commands to repeat it.

## Results

None yet.
