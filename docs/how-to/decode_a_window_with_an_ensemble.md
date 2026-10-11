[decsim docs](../README.md) › [How-to guides](README.md)

# How to decode a window with an ensemble

An ensemble is a decoder row whose members decode every window at once
on the pool's unit; a pooling row picks the answer
(`decsim/decoders/ensembles.py`). It follows the ensembles the sources
build, each one decoder with parallel members and a pooling step:
Harmony (Shutty, Newman and Villalonga 2401.12434), VibeLSD
(2508.15743) and the Coset Ensemble (2606.11076).

## 1. Make the ensemble the pool's algorithm

```python
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.ensembles as ensembles
import decsim.presets.experiment_one as experiment_one
import decsim.presets.machines as machines

base = experiment_one.switching(5, 0.004)
union_find = base.part(machines.WEAK_POOL).settings.algorithm
ensemble = ensembles.EnsembleDecoder.Settings(
    members=(union_find, belief_matching.BeliefMatchingDecoder.Settings()),
    pooling=ensembles.MostLikely.Settings(),
)
machine = base.with_settings(machines.WEAK_POOL, algorithm=ensemble)
```

`members` lists at least two decoder rows, set in Python; `--set`
reaches the pooling row but not inside the list.

## 2. Choose the pooling

| Row | Source | The answer | When the unit frees |
|---|---|---|---|
| `vote` | Harmony Sec. III.2 | the prediction most members give; a tie to the earliest listed | when the slowest member finishes |
| `sum_likelihood` | Harmony Sec. III.2 | the prediction whose members' likelihoods sum largest | when the slowest member finishes |
| `most_likely` | Harmony Sec. III.2 | the most likely member's | when the slowest member finishes |
| `first_converged` | VibeLSD Sec. III.4.2 | the most likely of the first `count` members to converge | when the count-th converges; the rest are stopped |

A member's likelihood is its whole selected error set's, priced on the
window model as the sum of log((1 - p) / p) over its faults. Pricing
needs every member in one fault representation; members of two are
refused, not implemented yet (stage 5), unless they vote. With no member
converged, `first_converged` answers best-effort, with the most likely
member and its status.

## 3. Escalate when the members disagree

Each answer carries the ensemble's agreement, the share of members
voting with the majority. Harmony's layered decoding runs a second pass
"if and only if the first pass ensemble fails to yield a consensus
prediction" (Sec. V):

```python
import decsim.confidence.consensus as consensus
import decsim.escalation.verdicts as verdicts

layered = machine.with_settings(
    machines.SWITCHING,
    confidence=consensus.Consensus.Settings(),
    verdict=verdicts.Agreement.Settings(),
)
```

`agreement` keeps an answer at or above `minimum`, 1.0 by default (any
dissent escalates). The consensus row reports a share, not a gap, so it
is refused beside a threshold verdict, and the agreement verdict beside
any other row; a pool that is not an ensemble is refused the consensus
row.

## 4. Know what is not built

- Harmony makes its members disagree by perturbing correlated
  matching's priors (Sec. III.1). Here members differ only as their
  rows do.
- The Coset Ensemble votes among the smallest corrections only.
- No source merges one window's results across separate pools; an
  ensemble is one pool's decoder.
