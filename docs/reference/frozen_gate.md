# The frozen gate

Every change to decsim also passes a behaviour gate: a frozen set of
runs whose results and logs are hashed, so a change that moves a tick or
a bit fails loudly. `STYLE.md` opens by naming it, and this page says
what it is, where it lives and how to run it.

## What it is

The gate is a **characterization test**, also called a golden master
(Michael Feathers, *Working Effectively with Legacy Code*). It pins what
the code does, not what it should do. That is a deliberate choice with
three consequences a reader should know:

- It pins bugs as readily as it pins correct behaviour. A bug found
  while working on something else gets a referent and its own commit
  rather than a quiet move of the golden file.
- It is blind past its own points. "Reachable" in `STYLE.md` means
  reachable by a gate point, a test, or a yaml key.
- It is brittle to incidental text. The golden file therefore carries
  two hashes per point, one over results and one over the log, so a
  change to a log line's wording regenerates only the log hash, and the
  commit that does it says the results hash is unchanged.

It runs twenty-six points. Twenty-one are **strict**: every field must
match, including the full log hash. Five are **semantic**: two strong
points and three switching points, where the strong tier and the
switching weak tier price a decode from the measured wall clock of a
real decoder, so tick-bearing fields legitimately differ between runs.
Those five are compared on a projection instead: the logical results,
the window structure and decode statuses, the frame record sequence with
its tiers and observables, tick-free link traffic, the maximum queue
depth, the idle rounds, and the packing and strong counters.

## Where it lives

Not in this tree. The gate lives beside the checkout, in the research
sandbox, at

```
validation/responsibility_audit_2026_08_30/
```

with `frozen_suite/capture.py` (which captures or checks the points),
`golden/golden.json` (the frozen capture, whose sha256 `verify.py` pins),
`verify.py` (an independent verifier with its own comparison code), and
one design note per regeneration of the golden file.

It is outside the tree because it is evidence about the tree rather than
part of it, and because regenerating it needs an approved design note,
never a quiet edit. The README of that folder is its own reference.

## How to run it

From the repo root, with the checkout's own interpreter:

```
.venv/bin/python ../validation/responsibility_audit_2026_08_30/verify.py
```

The verifier first checks the golden file's recorded sha256, so a
corrupted or silently edited golden is caught before anything runs. It
then re-runs every point and compares. It prints one line per point and
a final count.

## One known weakness

On the switching points the order of the transfer list inside
`link_traffic_semantic` is not a property of the code: the weak tier
prices its decode from the measured wall clock, so the interleaving of
transfers moves from run to run while the multiset of rows does not. The
verifier compares that list in order, so a switching point can fail with
no change to the code at all. Two runs of the verifier on this page's
own commit failed the same two switching points on
`link_traffic_semantic` alone, and passed the other twenty-four,
including all twenty-one strict points.

If the verifier fails on a switching point on `link_traffic_semantic`
alone, that is this weakness. The fix belongs to the verifier: compare
those transfer rows as a multiset on wall-clock points. It is recorded
in the sandbox as an open item.

## Read next

- `STYLE.md`: the rules the gate backs up.
- `docs/explanation/time.md`: why a wall-clock decoder makes some points
  semantic rather than strict.
- `docs/how-to/run_a_timing_only_study.md`: how to get a run whose ticks
  do not depend on the host.
