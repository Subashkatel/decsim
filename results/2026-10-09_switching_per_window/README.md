# Experiment 4: switching per window

Four configurations decode the same shots. Shot n has seed n in every
configuration.

- union-find alone
- switching: union-find decodes every window. A window whose cluster gap
  is under 20 dB goes to Relay-BP-5 (escalated). The others keep
  union-find's answer (kept).
- Relay-BP-5 alone
- Tesseract alone

Surface-code memory, 100 rounds per shot, d = 5, 7, 9, 11, physical
error rate 0.002, 0.003, 0.004, 0.005. Slurm array 15259257, all 330
jobs completed; fold job 15276821. 2,264,892 shots in all: each setting
has the same shot count in all four configurations.

## Where the numbers come from

- The run: decsim at commit 1785bafa (see commit.txt), run folder
  `tmp/runs/2026-10-09_switching_per_window_code/results/2026-10-09_switching_per_window/run`
  in the sandbox. It is not in the repository (window_outcomes.csv is
  3.5 GB).
- This folder: `experiments/switching_per_window/` at commit c9257a56.

```
python experiments/switching_per_window/tables.py <run folder> <this folder>
python experiments/switching_per_window/plot.py <this folder>
python experiments/switching_per_window/comparison.py <this folder>
```

The scripts only read the run folder. The run folder's file sizes and
times were the same before and after the scripts ran. Every number here
is a count from the run's csv files, or a rate made from those counts.
tables.py stops with an error if a shot's failed flag differs from the
parity of its window answers against its window labels, or if the
windows of a shot do not cover its rounds.

An independent recount (Codex, its own code on the raw csv files)
matched every table count at d = 5, 0.005; d = 7, 0.003; and d = 11,
0.005, all shot totals and seed sets in all 16 settings, and every cell
of comparison.csv.

## Files

| File | What it holds |
|---|---|
| shot_summary.csv | per configuration and setting: shots, failed shots, failure rate with 95% Wilson interval, windows decoded, shots with a wrong window |
| window_summary.csv | per configuration and setting: windows by final answer class |
| switching_summary.csv | switching only: escalated and kept windows by union-find's class, and Relay-BP-5's outcomes on the escalated windows |
| union_find_by_gap.csv | switching only: union-find's windows by cluster gap (1 dB bins) and class |
| kept_by_detection_events.csv | switching only: kept windows by detection events and class |
| comparison.csv | the meeting table: one row per setting, all four configurations side by side |
| plots/ | the figures below |

## Words

- **Window label.** The parity of the sampled errors whose first round is
  in the window's commit rounds (Zhang et al. 2509.03815, Eq. 1). A
  window is right when its answer bit equals its label. A shot fails when
  the parity of its answers differs from the parity of its labels (Eq. 2).
- **Wrong in a seam pair.** Two neighbour windows are both wrong. Their
  flips cancel, so the shot does not fail. One error near the seam is
  owned by one window and corrected in the next one. Pairs are taken left
  to right and do not overlap.
- **Wrong alone.** A wrong window that is not in a seam pair. Its flip
  stays.
- **Wrong beside a skipped window.** An escalated window commits its own
  rounds and the rounds of the next two windows. Union-find never decodes
  those two windows. A wrong window beside them can have its seam partner
  there, so we cannot tell "alone" from "pair".
- **False positive.** Escalated, and union-find was right on the
  window's own rounds.
- **False negative.** Kept, and union-find was wrong alone.
- **Fixed / broke.** For an escalated window, union-find alone (the same
  shot, folded onto the same rounds) is compared with Relay-BP-5. Fixed:
  union-find wrong, Relay-BP-5 right. Broke: union-find right, Relay-BP-5
  wrong. Seam pairs count as right on both sides. Windows where
  Relay-BP-5 did not converge keep their committed answer and stay in
  these counts; they are also counted on their own (not_converged).
  The `label_` columns count every wrong label as wrong, with no seam
  correction.
- **Per round.** A shot's failure rate p over 100 rounds is turned into a
  rate per round e by 1 - 2p = (1 - 2e)^100, as in Zhang et al. and
  Experiment 1. comparison.csv also gives the rate per d rounds, the
  form Toshio et al. 2510.25222 report.

No paper we checked scores each window right or wrong inside a
sliding-window run, so figures 03 to 13 use the definitions above.

## comparison.csv columns

For each configuration (prefix `union_find_alone_`, `switching_`,
`relay_bp5_alone_`, `tesseract_alone_`): `shots`, `failed_shots`,
`windows_per_shot`, `per_round` and `per_d_rounds` (each with `_low`
and `_high`, the 95% Wilson bounds), `windows`, `wrong_alone`,
`wrong_paired`.

For switching: `switching_decided_windows` (all windows union-find
decided), `switching_escalated`, `switching_correct_escalations`
(escalated, union-find wrong), `switching_false_positives`,
`switching_false_negatives`, `switching_kept_wrong_paired`,
`switching_kept_wrong_partner_unseen`, and Relay-BP-5's
`relay_fixed`, `relay_broke`, `relay_both_right`, `relay_both_wrong`,
`relay_not_converged`.

## Figures

Each figure answers one question. Bars and error bars are 95% Wilson
intervals. Figure 01 does not draw a point with fewer than 20 failed
shots. The share figures do not draw a point with fewer than 20 windows
in its total.

### 01 What is each configuration's logical error rate?

- How to read: one panel per distance, one line per configuration,
  logical error rate per round on a log axis.
- What we see: union-find alone is highest at every setting. Switching
  is close to Relay-BP-5 alone up to 0.004 (d = 11, 0.002: both
  6e-7 per round, union-find alone 2.5e-6). At 0.005 switching moves
  toward union-find alone (d = 11: switching 5.8e-4, Relay-BP-5 3.6e-4,
  union-find 6.4e-4). Tesseract alone is lowest where it is drawn.
  Tesseract has fewer than 20 failed shots at d = 7 (0.002, 0.004) and at
  d = 9 and 11 (0.002 to 0.004), so those points are not drawn.

### 02 How many windows does each configuration decode in a shot?

- How to read: one bar per configuration and physical error rate, mean
  windows per 100-round shot.
- What we see: the three "alone" configurations decode the same windows:
  20, 15, 12 and 10 at d = 5, 7, 9, 11. Switching decodes fewer, because
  one escalated window commits the rounds of three windows. It drops as
  more windows escalate (d = 11: 6.9 at 0.002, 4.6 at 0.005).

### 03 What share of union-find's windows are wrong?

- How to read: union-find alone. Two bars per physical error rate: wrong
  alone and wrong in a seam pair, in % of all windows.
- What we see: most wrong windows are seam pairs. d = 11, 0.002: 0.016%
  wrong alone, 1.67% in seam pairs. Wrong alone grows fast with the
  physical error rate (d = 11: 0.016% to 0.93%).

### 04 What share of each configuration's windows are wrong alone?

- How to read: one bar per configuration, windows wrong alone in % of
  that configuration's windows. For switching this is the final answer.
- What we see: Tesseract lowest, then Relay-BP-5, then union-find.
  Switching has fewer and larger windows, so its share per window is not
  a like-for-like comparison with the others.

### 05 What share of union-find's windows does switching escalate?

- How to read: one line per distance, escalated windows in % of all
  windows union-find decided.
- What we see: 12.3% (d = 5, 0.002) to 66.0% (d = 11, 0.005). It grows
  with distance and with the physical error rate.

### 06 Of the escalated windows, what share were false positives?

- How to read: one line per distance, escalated windows that union-find
  had right, in % of escalated windows.
- What we see: 93.4% to 99.2%. Almost every escalation is a false
  positive.

### 07 Of union-find's windows, what share are wrong, by decision?

- How to read: bars in % of all windows union-find decided. Red: kept,
  wrong alone (false negative). Orange: kept, wrong in a seam pair.
  Grey: kept, wrong beside a skipped window. Purple: escalated, wrong
  (correct escalation).
- What we see: false negatives are very rare: 1 to 131 windows per
  setting (131 at d = 11, 0.002, out of 2.7 million windows decided).
  Kept wrong windows are nearly all seam pairs. Correct escalations grow
  with the physical error rate.

### 08 Does the gap separate union-find's wrong windows from right ones?

One file per physical error rate.

- How to read: two lines, the gap of right windows and of wrong windows,
  each in % of its own windows, 2 dB bins. Dashed line: the 20 dB
  threshold. The last bin holds 60 dB and above.
- What we see: the two lines almost match, so the cluster gap does not
  separate wrong windows from right ones here. Many windows share one gap
  value, and that value moves with the physical error rate: 43 dB at
  0.002 (46% to 66% of windows), 39 dB at 0.003, 34 dB at 0.004. So it
  comes from the error weights, not from a fixed cap. We have not checked
  more than this.

### 09 Do the kept windows union-find gets wrong have more detection events?

One file per physical error rate.

- How to read: kept windows only. One line per class, each in % of its
  own windows. A class with fewer than 20 windows is not drawn.
- What we see: seam-pair windows have the same detection events as right
  windows. Windows beside a skipped window have more (d = 11, 0.003:
  near 200 against near 130). Wrong-alone kept windows reach 20 only at
  some settings (d = 7 to 11 at 0.002); there they sit near the right
  windows. The peak near 0 is the short last window of each shot.

### 10 Of the escalated windows union-find got wrong, what share did Relay-BP-5 fix?

- How to read: escalated windows where union-find alone was wrong on the
  same rounds (seam pairs count as right). One line per distance. This
  is not the purple bar of figure 07: that bar uses switching's own
  union-find answer on the window's own rounds only.
- What we see: 56% (d = 11, 0.002) to 83% (d = 7, 0.002), most points
  between 63% and 78%. The rest are "both wrong".

### 11 Of the escalated windows union-find got right, what share did Relay-BP-5 break?

- How to read: escalated windows where union-find alone was right. One
  line per distance.
- What we see: 0.002% (d = 11, 0.002) to 2.75% (d = 5, 0.005). At small
  d, broken windows are a large part of all changes: d = 5, 0.002 has 58
  fixed and 35 broken.

### 12 On what share of the escalated windows did Relay-BP-5 not converge?

- How to read: one line per distance, not-converged windows in % of
  escalated windows.
- What we see: zero at d = 5 and 7. At d = 11 it grows to 2.26% at
  0.005 (105 windows). This and the broken windows match where switching
  moves away from Relay-BP-5 alone in figure 01.

### 13 Of the shots with a wrong window, what share fail?

- How to read: of the shots with at least one wrong window, the % that
  failed. One line per configuration.
- What we see: at low physical error rate most do not fail, because seam
  pairs cancel. d = 11, 0.002: union-find alone has 32,111 shots with a
  wrong window and 98 failed (0.3%). The share grows with the physical
  error rate. Switching is highest at 0.005 (d = 11: 31%).

## Open

- The common gap value in figure 08: not studied beyond its position.
- Why switching's share in figure 13 is highest at 0.005: not studied.
