# Experiment log

One entry per experiment, newest first: the question, the design
decisions and who made them, the code and parameters, how it ran on
Slurm, where the data is, and how to run it again. Findings are
tentative until an experiment has stopped and been checked.

Why stopped: on 2026-10-08 the owner stopped all three online
experiments. The points left were the slowest (low error rates), and
the runs save no true answer per window, which the next experiment
needs. Finished points are complete; stopped points keep every shot
they saved.

Every path under `/scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox` is on
Princeton's Della cluster. Full run folders (every shot's pieces) stay
there; they are too large for GitHub (211 GB for Experiment 1). GitHub
holds the code, the configs, the folded tables (`status.csv` and the
CSVs beside it), the plots, and the scripts that make them.

## Common setup

- **Code.** github.com/Subashkatel/decsim. Each entry names the commit
  its run used. The online experiments (1, 2, 3) ran from a checkout
  pinned at that commit that nobody edits (a "run copy"); every piece
  records the commit, whether the tree was clean, and every Python
  package's version in its `piece.json`.
- **Python.** 3.11 with stim 1.16.0, sinter 1.16.0, numpy 2.2.6, scipy
  1.17.1, relay-bp 0.2.2, ldpc 2.4.1, pymatching 2.3.1, matplotlib
  3.10.9, PyYAML 6.0.3, and tesseract_decoder 0.1.1.dev20260822 for the
  Tesseract runs. On Della this is the venv
  `tmp/deltakit-integration/upstream/.venv` (set `DECSIM_PYTHON` to its
  `bin/python`).
- **Union-find.** Compiled C: `tools/build_union_find.sh` builds
  `decsim/decoders/union_find/union_find.so` in the checkout.
- **Running an online experiment.** `slurm/run_all_rounds.sh <run
  folder> --tasks N --cores 1 --hours H --memory-mb M <yamls>` plans
  batch 1 (`decsim plan`), submits it (`slurm/round.sh`), and queues
  itself after the batch (`afterany`); each next step plans and submits
  the next batch until every point has stopped. The Slurm job ids of
  batch k are in `<run folder>/round<k-1>/next_step.log` (batch 1's in
  the shell that started it). A point stops at 100 logical failures or
  5,000,000 shots.
- **Folding.** `python -m decsim status <run folder>` folds every saved
  piece into `<run folder>/status.csv` and `<run folder>/combined/`. The
  loop does not run it; run it by hand, as one Slurm job for
  Experiment 1 (about 4 hours, 16 GB).
- **Same shots.** A shot's Stim sample is drawn from its seed, and every
  point starts at seed 0, so the same seed is the same shot in every
  configuration and experiment that shares a setting. Checked on 796,762
  switching shots with no escalation: every one fails or passes exactly
  as union-find alone does.
- **Slurm on Della.** The queue (QOS) is chosen from the time limit:
  up to 24 h 1 min is `short` (400 running jobs, 1,000 cores, 1,000
  queued jobs per user), up to 72 h is `medium` (200 jobs, 400 cores).
  Naming a QOS does not override this (`/etc/slurm/job_submit.lua`).

## Planned: per-window decoder comparison

**Status.** Being designed on branch `per-window-experiment` (from
`main`), with its own detailed log in `experiments/per_window/LOG.md`.

**Question.** For the logical error rate plot of Experiment 1: how many
windows each decoder decodes, how many union-find gets right and wrong,
how often switching happens, and when a switching decision is right or
wrong (escalated though union-find was right, or kept though it was
wrong) and why.

**Design decisions (owner, 2026-10-08).**
- A new run, not the old data: decsim saves no true answer per window,
  and the true answer needs Stim's sampled errors, so the shots are
  sampled a new way and are not the old runs' shots.
- Switching uses Relay-BP-5's answer as now, no fallback to union-find.
- Option A: union-find alone, Relay-BP-5 alone and switching run as
  separate configurations on the same shots. Union-find alone and
  Relay-BP-5 alone use the same windows, so their rows line up window by
  window; switching has its own windows.
- A per-window table: shot, window, rounds, gap, escalated, detection
  events, true answer, union-find's answer, Relay-BP-5's answer and
  whether it converged, and Tesseract's answer on the same input as a
  check the machine never uses (`observation.check_windows_with:
  tesseract`).

## 2026-10-08: per-shot comparison

**Question.** On the same shots, which of union-find alone, Relay-BP-5
alone and switching fails, and are switching's extra failures at d=13
the Relay-BP windows that did not converge?

**Code.** `experiments/per_shot_comparison/compare.py` on this branch.
No new runs: it reads the saved pieces of Experiments 1 and 3.

**Run it again.**
```
python experiments/per_shot_comparison/compare.py \
  <exp1 run folder> <exp3 run folder> results/2026-10-08_per_shot_comparison
```
with the run folders of Experiments 1 and 3 below (about 5 minutes).

**Results.** `results/2026-10-08_per_shot_comparison/`:
`comparison.csv` (one row per setting: shared shots, each decoder's
failures, and every combination), `plots/failure_rate_per_decoder.png`,
`plots/only_switching_failed.png`. The plots draw settings with at
least 500 shared shots; Experiment 3 was still running.

**What we see (tentative, 2026-10-08).** At d=5 to 9 switching fails
about as often as Relay-BP-5 alone and less often than union-find alone;
shots that only switching fails are about as common as shots only
Relay-BP-5 fails, and none had a window that did not converge. At d=13,
physical error rate 0.005, switching fails 168 of 1,000 shared shots,
Relay-BP-5 alone 69 and union-find alone 39; in 121 of the 122 shots
only switching fails, a Relay-BP window did not converge. At d=11, 0.005,
45 of 73 such shots had one.

## 2026-10-07: Relay-BP probe at d=13

**Question.** Why do Relay-BP-5 windows not converge at d=13, physical
error rate 0.005 (Experiment 1: 432 of 2,535 escalated windows)? Is the
window or its syndrome wrong, or is the search too short?

**Code.** `experiments/relay_bp_probe/` on this branch: `probe.yaml`
(Experiment 1's switching config at that one setting, 20 shots; same
configuration id, 1d1aa311), `capture.py` (saves every Relay-BP window's
check matrix, priors, syndrome, status and iterations), `analyze.py`
(three checks on each window that did not converge).

**Run it again** (on any node with 10 cores, about 25 minutes):
```
PYTHONPATH=. python experiments/relay_bp_probe/capture.py <windows> \
  collect experiments/relay_bp_probe/probe.yaml --out <run> --processes 10
python experiments/relay_bp_probe/analyze.py <windows> <out>
```

**Results.** `results/2026-10-07_relay_bp_probe/`: `windows.csv` and
`failed_windows.csv`, from a second run of the committed scripts on
2026-10-08 (the same 5 windows fail, with the same outcomes as the
first probe).

**What we found.** An escalated window at d=13 is a double window: 39
rounds, about 6,500 detectors by 131,000 fault columns, 500 to 600
detection events. The Relay-BP paper (2506.01779 lines 280-302) tested
the surface code only at d=11 with 11 rounds, a matrix of about 1,000 by
24,000. In the first probe 5 of 50 windows did not converge, and many
that converged used the whole 36,080-iteration budget. For all 5,
BP+OSD-0 finds a correction that explains the syndrome, so a valid
answer exists. With 600 legs and other gamma seeds, 4 of 10 tries
converge; with 3,000 legs, 5 of 5. decsim commits a nonconverged answer
as is (`decsim/decoders/backend_outcome.py`).

## 2026-10-07: Experiment 3, strong decoder alone

**Status.** Stopped by the owner on 2026-10-08 during batch 5. The
final fold (job 15247572, done 2026-10-08 20:12) has 21 of 40 points at
their stop rule; the 19 others keep their partial shots.

**Question.** What logical error rate does a strong decoder reach alone
on the same machine, windows and shots as Experiment 1, so switching
can be compared with it honestly (not with a whole-shot decoder)?

**Design decisions.**
- New windowed runs for both Relay-BP-5 and Tesseract, not the
  whole-shot decoder baseline (owner, 2026-10-07).
- Only `escalation.kind: strong_only` changes from Experiment 1's
  switching config: every window is decoded once by the strong decoder,
  on the same sliding windows (commit d rounds, buffer d rounds).
  Tesseract uses the package's short-beam profile.
- The settings where whole-shot Tesseract reached 100 failures: d=5 all
  six error rates; d=7 from 0.001; d=9 from 0.002; d=11 from 0.003; d=13
  0.004 and 0.005. 20 settings per decoder. The lower error rates at
  large distance would run to the 5,000,000-shot cap (about 124,000 and
  193,000 core-hours). Upper estimate for the 40 points: 4,700 core-h
  (Relay-BP-5) and 7,300 (Tesseract).
- 25-hour jobs, so the run uses the `medium` queue and does not wait
  behind Experiment 1 in `short` (owner, 2026-10-07).

**Code.** Branch `strong-only-baseline-run` at `05702f9f` (Experiment
1's code `815e987a` plus two configs):
`configs/experiments/strong_only_baseline/relay_bp5_alone.yaml` and
`tesseract_alone.yaml`. Configuration ids 37cc8bd5 (Relay-BP-5) and
f052df19 (Tesseract).

**Job shape.** 1 core, 8 GB, 25 h per task, up to 200 tasks per batch.
Batch 1: array 15194145, loop step 15194184.

**Run it again.**
```
git checkout 05702f9f && tools/build_union_find.sh
export SBATCH_TIMELIMIT=25:00:00   # every later batch inherits it
slurm/run_all_rounds.sh results/2026-10-07_strong_only_baseline/run \
  --tasks 200 --cores 1 --hours 25 --memory-mb 8192 \
  configs/experiments/strong_only_baseline/relay_bp5_alone.yaml \
  configs/experiments/strong_only_baseline/tesseract_alone.yaml
```
The Tesseract package must be importable (`PYTHONPATH` to a folder
holding `tesseract_decoder` and `_tesseract_py_util`).

**Data.** `tmp/runs/2026-10-07_strong_only_code/results/2026-10-07_strong_only_baseline/run`.
On GitHub: `results/2026-10-07_strong_only_baseline/` (`status.csv`
final fold, `configurations.csv`, `commit.txt`). Experiment 1's
`plot.py` draws these points in
`results/2026-10-01_switching_baseline/plots/logical_error_rate_with_strong_alone.png`.

## 2026-10-03: Experiment 2, threshold sweep

**Status.** Stopped by the owner on 2026-10-08 during batch 11. The
final fold (job 15247571, done 2026-10-08 21:10) has 10 of 20 points at
their stop rule (every point at physical error rate 0.003); the points
at 0.001 keep their partial shots.

**Question.** How does the switching threshold change the share of
windows sent to the strong decoder, the logical error rate, the latency
of kept and escalated windows, and whether the strong decoder keeps up?

**Design decisions.**
- Experiment 1's machine with the cluster-gap threshold swept over 5,
  10, 15, 20 and 30 dB; d=9 and 11; physical error rate 0.001 and 0.003
  (owner kept 20 dB in the sweep, "let's redo 20").
- Union-find alone is not run again: it is Experiment 1's weak_alone on
  the same shots, the same at every threshold.

**Code.** Commit `71385257` (Experiment 1's decsim plus a plotting
change), config `configs/experiments/threshold_sweep/threshold_sweep.yaml`.

**Job shape.** 1 core, 8 GB, 24 h per task, up to 400 tasks per batch.

**Run it again.**
```
git checkout 71385257 && tools/build_union_find.sh
slurm/run_all_rounds.sh results/2026-10-03_threshold_sweep/run \
  --tasks 400 --cores 1 --hours 24 --memory-mb 8192 \
  configs/experiments/threshold_sweep/threshold_sweep.yaml
python -m decsim status results/2026-10-03_threshold_sweep/run
python experiments/threshold_sweep/plot.py results/2026-10-03_threshold_sweep
```

**Data.** `tmp/runs/2026-10-03_threshold_sweep_code/results/2026-10-03_threshold_sweep/run`
(29 GB). On GitHub: `results/2026-10-03_threshold_sweep/` (`status.csv`
final fold of 2026-10-08, `configurations.csv`, `commit.txt`, `plots/`).

## 2026-10-01: Experiment 1, switching baseline

**Status.** Stopped by the owner on 2026-10-08 during batch 12, with 44
of 60 points at their stop rule; the 16 others (physical error rates
0.0005 to 0.002) keep their partial shots. Final fold running.

**Question.** Does switching (union-find on every window, Relay-BP-5 on
the windows union-find is unsure of) reach a lower logical error rate
than union-find alone on the same shots, and at what latency and strong
decoder load?

**Design decisions** (owner, 2026-09-29 unless noted).
- Rotated surface code memory Z, 100 rounds, one physical error rate on
  all four Stim noise channels; d=5, 7, 9, 11, 13; physical error rates
  0.0005, 0.001, 0.002, 0.003, 0.004, 0.005; a round every 1 us.
- Weak decoder: union-find on the Helios cycle law (100 MHz).
  Confidence: cluster gap. Threshold: fixed 20 dB.
- Strong decoder: Relay-BP-5 (2506.01779 surface code values, X and Z
  decoded together), its time from the A100 measurement (Slurm job
  14676845; `decsim/decoders/measured_table/measurements.py`).
- Windows: sliding, commit d rounds, buffer d rounds. Strong window:
  double window (Toshio et al. 2510.25222 Sec. III C).
- Link and compute latencies held at one sourced value each, in the
  config's comments.
- Reference: union-find alone (`weak_alone.yaml`) on the same shots.
- No pilot batch and no time cap (owner, 2026-10-01).

**Code.** Commit `815e987a` on `main`. Configs
`configs/experiments/switching_baseline/switching_baseline.yaml`
(configuration id 1d1aa311) and `weak_alone.yaml` (45a586ca).

**Job shape.** 1 core, 8 GB, 24 h per task. Batch 1: array 14843105
(60 tasks), loop step 14843106. Later batches up to 1,000 tasks; the
loop was restarted with `--tasks 450` once Experiment 2 shared the
1,000-job queue limit.

**Run it again.**
```
git checkout 815e987a && tools/build_union_find.sh
slurm/run_all_rounds.sh results/2026-10-01_switching_baseline/run \
  --tasks 1000 --cores 1 --hours 24 --memory-mb 8192 \
  configs/experiments/switching_baseline/switching_baseline.yaml \
  configs/experiments/switching_baseline/weak_alone.yaml
python -m decsim status results/2026-10-01_switching_baseline/run
cp results/2026-10-01_switching_baseline/run/status.csv results/2026-10-01_switching_baseline/
python experiments/switching_baseline/timing_tables.py \
  results/2026-10-01_switching_baseline/run/combined results/2026-10-01_switching_baseline
python experiments/switching_baseline/plot.py results/2026-10-01_switching_baseline
```
The two plot steps run from this branch's checkout.

**Data.** `tmp/runs/2026-10-01_switching_baseline_code/results/2026-10-01_switching_baseline/run`
(211 GB). On GitHub: `results/2026-10-01_switching_baseline/`
(`status.csv` folded 2026-10-07 20:27, `configurations.csv`,
`commit.txt`, `decode_time.csv`, `stage_means.csv`, `plots/`).
`plots/logical_error_rate.png` is switching against union-find alone;
`logical_error_rate_with_tesseract.png` adds the whole-shot Tesseract of
the decoder baseline, a best case, not a windowed decoder.

**What we see (tentative, fold of 2026-10-07).** From d=5 to 11,
switching is below union-find alone wherever both have failures. At
d=13, 0.004 and 0.005, switching is above union-find alone (0.005: 1.9e-3
against 4.2e-4 per round); see the Relay-BP probe and the per-shot
comparison. The strong decoder does not keep up at any setting: its mean
service time is above Toshio's bound (Theorem 1) by 1.3 times at d=5,
0.0005, up to about 27,000 times at d=13, 0.005.

## 2026-09-28: burst detection

**Question.** How fast does the masked regional CUSUM detector catch a
burst of extra noise, and how many false alarms does it raise on quiet
shots?

**Code.** Commit `e566bccf`, `experiments/burst_detection/run.py` and
`plot.py`. The burst detector was later removed from `main` (parked);
it can be re-added from git.

**Data.** `tmp/job_1f7e5d61/baseline_runs/2026-09-28_burst_detection`.
On GitHub: `results/2026-09-28_burst_detection/` (`trials.csv`,
`quiet.csv`, `alarm_levels.csv`, `plots/`).

## 2026-09-27: decoder baseline (offline, whole shots)

**Question.** The logical error rate of each decoder alone on whole
100-round shots: union-find, PyMatching, Tesseract (short beam) and
XYZ-Relay-BP-5, d=5 to 15, six error rates, X and Z.

**Design decisions** (owner, 2026-09-27). Offline with sinter and each
decoder's own adapter; stop at 100 errors; the Slurm time limit is the
budget (a timed-out task keeps what it saved, and the same submission
continues it). Relay-BP-1 was replaced by XYZ-Relay-BP-5.

**Code.** `experiments/decoder_baseline/run.py`: commit `e5f821d4` for
union-find, PyMatching and Tesseract, `7024df9d` for XYZ-Relay-BP-5.

**Job shape.** `slurm/run.sbatch`, one point per array task, 16 cores,
32 GB. Batch 1: array 14598433 (ids 0-287, 3 h); batch 2: 14619453 (the
73 timed-out points, 24 h).

**Run it again.**
```
python experiments/decoder_baseline/run.py --list
sbatch --array 0-287 slurm/run.sbatch <checkout>/experiments/decoder_baseline/run.py <out folder>
python experiments/decoder_baseline/run.py combine --out <out folder>
```

**Data.** `tmp/job_1f7e5d61/baseline_runs/2026-09-27_decoder_baseline`
and `2026-09-28_decoder_baseline` (Relay-BP-5). On GitHub:
`results/2026-09-27_decoder_baseline/` (`stats.csv`, `plots/`).

**What we see.** Whole-shot XYZ-Relay-BP-5 is far worse than union-find
at d of 11 and above from physical error rate 0.003 (d=13 at 0.005: 105
errors in 239 shots).
