# 00 Baseline contract

## Question

What exactly is the experiment, and what input does each decoder receive?

## Deliverable and ownership

One run configuration in the existing experiment system, plus a small window
manifest. Use the baseline choices and open decisions in the [plan](../README.md).
Keep workload construction in the frontend, window geometry in the window
planner, and decoder/resource settings in the existing machine graph.

## Work

1. Pin memory basis, noise channels, terminal noise, decoder settings, seeds,
   round clock, worker counts, links and buffers.
2. Configure `10d` physical rounds, commit `d`, buffer `d`, ordinary redo and
   fixed 20 dB. Inspect automatic switching changes to the terminal policy.
3. Export each window's read/commit ranges and predecessor boundary identities.
4. Specify finite-shot and continuous-workload state lifetimes separately.
5. Stage the million-shot production stop rule without launching the grid.

## Gate and evidence

For the pilot, record exact circuit/model hashes and software versions. Check
that commit extents cover the shot exactly once, read extents stay legal, and
the final readout contributes to the intended observable. Count actual windows
rather than assume ten identical windows. Record all proposed choices that the
user has not yet settled. The output must be understandable without the chat.

## Scope boundary

No new controller, timing fit, double windows or speculation. Do not change
noise or shot length while diagnosing a decoder discrepancy.
