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

## Plan

Each step is decided before the next. Status on 2026-10-08:

| Step | What we decide | Status |
| --- | --- | --- |
| 1. Questions | what the run must answer | discussing |
| 2. Definitions | what right and wrong mean for a window | not started |
| 3. Records | what decsim saves per shot, per window, per strong window | not started |
| 4. Configurations and settings | decoders, distances, error rates, stop rule | partly decided |
| 5. Checks before the run | how we prove the records are right | not started |
| 6. Pilot and cost | time per shot, core-hours, job plan | not started |
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

## Runs

None yet. Each run gets an entry here before it starts: the date, the
commit, the configs, the parameters, the Slurm job ids, the data folder,
and the exact commands to repeat it.

## Results

None yet.
