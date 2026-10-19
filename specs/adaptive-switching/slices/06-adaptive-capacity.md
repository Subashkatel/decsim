# 06 Adaptive capacity

## Question

Can feedback adjust escalation to sustain a required useful-round rate and
control correction age under changing load or hardware service?

## Deliverable and ownership

A slower budget controller consuming causal observations from existing queues,
decoder completion, finalized frame coverage and later gate-consumer events.
The score tracker and quota remain separate responsibilities.

## Work

1. Define inputs: required round rate, allowed reaction/age bound where supplied,
   worker capacity, observed work/age, escalation bounds and update cadence.
2. Start throughput-first: reduce the target when outstanding work/age grows;
   cautiously increase it after a sustained headroom interval; otherwise hold.
   Specify units, smoothing, step size and hysteresis before implementation.
3. Include unresolved earlier windows and upstream blocking, not just ready
   queue length. Use estimated work when service lengths vary.
4. Expose only information available at the update instant. A measured service
   time is usable after completion, not as foreknowledge at admission.
5. Define reconfiguration: distinguish a fresh distance/geometry experiment
   from an actual mid-run change, which needs versioned state and a safe boundary.
6. Verify adaptive quota changes do not create unexplained startup credit;
   specify how saved permits behave when a target or resource budget decreases.

## Gate and evidence

Compare fixed 20 dB, the best calibrated static policy and adaptive policy on
the same workloads/resources. Test steady input, service slowdown/recovery,
score-distribution change and deliberately infeasible weak-only capacity.
Measure settling time, overshoot, denied candidates, useful throughput,
age/tails and LER across held-out distances/window sizes and independent runs.
Report late work already admitted: lowering the target cannot undo its delay.

## Scope boundary

Do not fabricate an online LER guarantee or infer correctness from decoder
agreement. A miss is reported and the consumer stalls under the baseline;
no silent substitution or cancellation of an admitted strong result.
