# 05 Rate controller

## Question

Can the threshold be adjusted online so the system escalates a target
fraction of windows, without simulator truth and without audits?

## Deliverable and ownership

An audit-free threshold source built from the existing rate tracker, warm
started from the slice 04 calibration table, plus an optional hard cap.
It integrates at the existing decision boundary (`ThresholdSource.decide_keep`).

## Current state

`OnlineThreshold` (`escalation/threshold_sources.py`) combines
`EscalationRateTracker` (escalate if `gap < threshold`, then
`threshold += step × (target − escalated)`, unclipped) with an audit lane
whose label is weak/strong disagreement and a slow loop driven by those
audits. One instance persists across a task's shots; each shot rebuilds the
machine. Known problems:
- extra cluster gap freezes its growth limit at the starting threshold, so a
  rising online threshold reads gaps as infinite and under-escalates;
- escalations forced by a late result under the deadline row skip the
  tracker, so realized duty exceeds the target;
- the audit label is disagreement, not correctness.

## Work

1. New source: rate tracker only, with bounds on the threshold, start value
   from the calibration table for (distance, p, target).
2. Count eligible, candidate, admitted and denied windows separately.
   Every path that starts strong work counts toward the budget.
3. Optional hard cap with integer credit: start at zero, earn one unit per
   eligible window, spend `1 / target` units per admission, hold at most an
   allowed burst. A candidate without credit keeps the weak result.
4. Refuse the combinations that break it: extra cluster gap with a moving
   threshold, deadline-forced escalation without counting.
5. Run in stream mode (state lives with the machine) and in collect mode
   (state shared across shots) and compare.
6. Migrate callers and tests off the audit source before removing it.

## Gate and evidence

Exact decisions on small synthetic score sequences. Tracking error against
target on held-out replayed scores, including a mid-stream change in p.
At steady state the controller's LER equals the static threshold's LER at the
same achieved fraction (the controller costs no accuracy). The cap bound
`admissions(N) ≤ floor(N × target) + burst` holds.

## Scope boundary

A target is not a cap; a window fraction is not strong-unit busy time; the
tracker cannot know which future windows are worst. No audits, no truth.
