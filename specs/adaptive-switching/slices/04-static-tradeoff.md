# 04 Static trade-off

## Question

Which static score thresholds and hardware configurations give useful
correctness at acceptable work and delay?

## Deliverable and ownership

A calibration table and a held-out static frontier. The threshold source owns
selection; experiment collection owns sample partitioning and uncertainty.
Timing profiles and threshold-calibration tables are different inputs.

## Work

1. Measure score distributions on a separate calibration seed range, keyed by
   the circuit, noise, distance, window/boundary geometry and decoder settings.
2. Derive candidate thresholds for proposed targets of 1%, 2%, 5%, 10% and 20%.
   Keep 20 dB as the original baseline. Record strict comparison and tie policy,
   including zero, infinite and missing scores.
3. Evaluate achieved fractions on held-out samples; do not pick using test
   labels or assume quantiles transfer perfectly.
4. Measure final shot LER, beneficial/harmful escalations where comparable,
   strong work, finalized throughput, age, end-of-stream response and tails.
5. Expand from pilot to proposed three-distance validation, then the agreed
   production grid and million-shot stop rule after runtime/resource estimates.

## Gate and evidence

Report actual counts, confidence intervals, calibration/test separation and
hardware assumptions. Produce LER-versus-escalation, LER-versus-strong-work,
and throughput/latency-versus-escalation plots. A gap quantile does not certify
failure probability. Preserve negative results when no useful operating region
exists; adaptation is not a remedy for absent static opportunity.

## Scope boundary

No hard cap yet. Offline top-score selection may be a diagnostic benchmark,
but cannot be represented as an online decision without its buffering delay.
