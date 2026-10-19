# 05 Percentage controller

## Question

Can we target a specified fraction of windows online, and separately enforce
the precise admission limit the user chooses?

## Deliverable and ownership

An audit-free candidate-rate tracker and a separately testable optional quota.
Integrate at the existing escalation decision boundary. Reuse the useful rate
recursion if verified; do not rebuild the engine or window manager.

## Work

1. Define an eligible window as one first weak result per original window.
   Retries and changed geometry require explicit later accounting.
2. Exercise the tracker on synthetic scores before real held-out score replay.
   For `escalate if gap < threshold`, use `threshold += step *
   (target_fraction - candidate_escalated)` after each observation.
3. Keep rate targets fixed first. Track score ties, startup, abrupt score shifts
   and unachievable fractions from infinite/missing scores without claiming
   distribution-free exact finite-interval control.
4. If a hard 1% cap is selected, use integer credit: start at zero, earn one
   unit per eligible window, spend 100 per admission, and retain at most
   `100 * allowed_burst` units. A candidate without credit keeps the weak result.
5. Count candidate, admitted and budget-denied windows separately. Scope state
   across a task or continuous stream, not independently per short shot.
6. Migrate callers/tests before removing the old audit-based online source.
   Keep one supported owner per behavior and no deprecated compatibility path.

## Gate and evidence

Verify exact decisions on small score sequences and the fixed-rate prefix
bound `admissions(N) <= floor(N / 100)` with zero startup credit. Verify the
separate burst bound, not a false promise of 1% in every rolling interval.
Record unused quota and startup bias. Empirically evaluate approximate tracking
with a declared tolerance and duration on calibration-independent traces, then
run the live simulator to expose changed downstream window inputs.

## Scope boundary

A target is not a hard cap, a window fraction is not service-time utilization,
and the online rule cannot know the worst future 1%. No audit mechanism or
simulator-truth feedback. Budget denials must not corrupt candidate-rate
learning or silently become strong-correctness labels.
