# 04 Static threshold

## Question

Does a fixed threshold work? For accuracy: how close does switching get to
strong-only, at what escalation fraction? For timing: is that fraction
within what the hardware sustains (q\* from slice 03)?

## Deliverable and ownership

A held-out LER-versus-escalation frontier with the capacity line drawn on it,
and a calibration table mapping (distance, p) and a target fraction to a
threshold. Experiment collection owns seed partitioning; the threshold
source owns selection.

## Work

1. **Fixed 20 dB.** Weak-only, strong-only, switching on the slice 00 setup,
   1e6 shots per point. Report shot LER with intervals, escalation fraction,
   outcome classes on escalated windows, and per-shot switching rate for
   comparison with the paper's Fig. 16(b).
2. **Diagnostic counterfactual.** On a smaller sample, run the strong decoder
   on every window without using its answer, so kept windows also get
   outcome classes. `run_both_at_once` with `NoCancel`
   (`windows/revision_rules.py`) already lets the strong decode finish on a
   keep, but drops its result; recording that dropped answer per window is a
   small addition. This gives P(fail | gap) per window for both decoders, the
   analogue of the paper's Figs. 5–6.
3. **Threshold sweep.** 0–50 dB as in Fig. 17, on calibration seeds. Find
   g_op (LER within 1% of the strong-only asymptote) and check whether the
   minimum dips below strong-only as the paper reports for UF + belief
   matching. Note the 20 dB value came from whole-shot complementary gaps;
   per-window cluster gaps on 2d-round windows have a different distribution.
4. **Seed offset.** Add a seed-range start to collection (seeds currently
   always run 0..N−1) so calibration and evaluation never share shots.
5. **Calibration table.** For targets 1, 2, 5, 10, 20%, the threshold that
   achieves each on calibration seeds; evaluate achieved fraction and LER on
   held-out seeds.
6. **Overlay timing.** For each threshold, the stream-mode lag and reaction
   percentiles from slice 03. Mark q\* on the frontier.

## Gate and evidence

Plots: LER versus escalation fraction (with q\*), LER versus threshold,
reaction p99 versus escalation fraction. The static threshold "works" at a
point if switching LER meets the agreed tolerance at a fraction at or below
q\*. Where it does not (expected at higher p, where 20 dB escalates more),
that is the motivation for slices 05–06, recorded as a result.

## Scope boundary

A gap quantile is not a failure probability. Offline top-k selection by
truth is an upper-bound diagnostic, not an online policy. Negative results
are kept.
