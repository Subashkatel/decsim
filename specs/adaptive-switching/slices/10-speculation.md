# 10 Speculation

Status: last slice, started after 00–09 pass. Part of the paper's second
contribution.

## Question

Does speculative hybrid decoding improve gate response and throughput after
paying for verification, wasted work and recovery?

## Current state

- `Switching(run_both_at_once=True)` is the paper's Step 1: the strong decode
  starts beside the weak one and is cancelled on keep (`escalation/policies.py`,
  `decoders/decoder_manager.py`).
- The deadline row (`escalation/deadlines.py`) starts a strong decode early
  when slack runs low or the source queue is deep (DART-Q triggers). It
  cannot be combined with `run_both_at_once`.
- Serial switching refuses provisional boundaries, so no later window moves
  ahead on a weak answer; operation results are delivered only when every
  window is final.
- A speculative start cannot depend on the gap, because the gap exists only
  after the weak decode ends.
- Aborting a running strong job frees its unit instantly, which overstates
  capacity; a cancellation cost belongs in the timing table.

## Work

1. Define what is provisional: the boundary passed to the next window, the
   frame update, or the gate release. Start with classical boundaries only.
2. Provisional boundaries in serial mode, with invalidation and re-decode of
   dependent windows when the strong answer differs.
3. Selective speculation: speculate only on ambiguous windows, using a
   signal available before or early in the weak decode, or on slack.
4. Charge verification traffic, duplicated compute, cancellation, recovery
   and the last unresolved result at each gate.
5. Compare with non-speculative switching and with SWIPER under matched
   schedules and resources, after reading SWIPER's method.

## Gate and evidence

Injected boundary mismatches recover to the correct final outcome. No gate
consumes unresolved speculative state. Report useful throughput, wasted
work, recovery frequency, stall tails and LER.

## Scope boundary

No speculative execution of quantum operations in the first study. A
negative result is kept if recovery erases the gain.
