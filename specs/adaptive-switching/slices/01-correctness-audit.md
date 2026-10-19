# 01 Correctness audit

## Question

Are the weak, strong and switching answers correct on the intended inputs,
and can every reported failure and escalation be accounted for?

## Deliverable and ownership

A paired diagnostic report using the frontend's sampled input identities,
decoder outcomes, window ledger and final logical observable. Existing anchors
include `tests/observe/test_window_outcomes.py` and
`tests/escalation/test_strong_redecode.py`; inspect their contracts before reuse.

## Work

1. Begin with noiseless and hand-selected error fixtures, then a small fixed
   random sample at the pilot point.
2. Validate belief matching against the pinned reference implementation on
   identical adapted window inputs. Compare union-find evidence with an
   independent existing reference where available.
3. Run weak alone, strong alone and fixed switching on the same physical samples
   and documented geometry. Record boundary differences rather than hide them.
4. For identical-input counterfactuals, count both correct, weak-only correct,
   strong-only correct and both wrong. Record final shot outcomes separately.
5. Check forced-keep and forced-escalate single-window cases against direct
   decodes. Diagnose disagreements before increasing sample size.

## Gate and evidence

Require exact agreement on logical observables for deterministic reference
fixtures, no missing/duplicate commits, and reconciled window/shot counts.
Store seeds, sample/model hashes, gap, threshold, policy decision and output.
An error identity is not a unique correction. Per-window labels do not replace
the final shot failure metric. Report disagreements and statistical uncertainty.

## Scope boundary

Simulator truth is for evaluation only. No strong-perfect assumption, online
audit labels, controller tuning or claim that independently windowed results
are interchangeable. Whole-shot decoding is an accuracy reference with a
different decoding problem, not an equality oracle for windowed decoding.
