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
| 2026-10-08 | All parameters in yaml files in `experiments/per_window/`, one per configuration, each extending `configs/experiments/switching_baseline/switching_baseline.yaml`. Scripts hold no numbers. | One place to read and change every parameter. |
| 2026-10-08 | The stop rule comes from the papers. | Research in progress. |

## Plan

Each step is decided before the next. Status on 2026-10-08:

| Step | What we decide | Status |
| --- | --- | --- |
| 1. Questions | what the run must answer | discussing |
| 1b. Literature check | which papers ground each piece | done, see below |
| 2. Definitions | what right and wrong mean for a window | not started |
| 3. Records | what decsim saves per shot, per window, per strong window | not started |
| 4. Configurations and settings | decoders, distances, error rates, stop rule | decided except the stop rule |
| 5. Checks before the run | how we prove the records are right | not started |
| 6. Cost | time per shot (from Experiments 1 and 3), core-hours, job plan | not started; no pilot |
| 7. Run system | how the run is started, folded, logged, pushed | not started |
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

## Runs

None yet. Each run gets an entry here before it starts: the date, the
commit, the configs, the parameters, the Slurm job ids, the data folder,
and the exact commands to repeat it.

## Results

None yet.
