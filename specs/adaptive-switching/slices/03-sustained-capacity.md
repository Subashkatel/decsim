# 03 Sustained capacity

## Question

Does weak-only decoding sustain arrivals, and what work and delay does static
switching add on the same declared hardware?

## Deliverable and ownership

A continuous-workload capacity report built from existing queue, stage,
coverage and stream measurements. Inspect `measure.py` and `stream_measures.py`
endpoints before choosing output columns. Add missing observations at their
owning component, not in a second simulated pipeline.

## Work

1. Check a deterministic serial case against `service < commit_stride *
   round_period`; account for sequential transfer and boundary costs.
2. Sweep supplied service around the analytical capacity boundary and verify
   growth above capacity. Validate weak-only first.
3. Add fixed-20-dB switching with fixed and then variable strong service times.
4. Run increasing lengths, initially 1,000, 10,000 and 100,000 rounds, extending
   when needed. Separate warmup, observation interval and drain.
5. Record generated/finalized round frontiers, unresolved holes, upstream
   buffers, strong outstanding work/age, resource occupancy and final drain.

## Gate and evidence

Require reconciled useful-round coverage and exact deterministic fixtures.
Distinguish offered load, which can exceed 100%, from single-resource busy
fraction, which cannot. Measure correction age and per-tier service as well as
queue waits. Repeats and longer runs must support any claimed stability region;
finite runs do not prove a worst-case guarantee. Bounded queues may move backlog
upstream, so an empty decoder queue alone cannot establish capacity.

## Scope boundary

No adaptive thresholds, speculation or T-gate circuit claims. Finite-shot
million-sample correctness does not establish continuous-stream stability.
