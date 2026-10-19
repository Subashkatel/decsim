# 00 Baseline contract

## Question

What exactly is the experiment, and do weak-only, strong-only and switching
decode the same windows?

## Deliverable and ownership

One run configuration in the existing experiment system (a new folder under
`experiments/`, built like `experiments/switching_paper/config.py`), plus a
window manifest for each pilot point. Workload construction stays in the frontend,
window geometry in the window planner, decoder and resource settings in the
machine graph.

## Two anchors

1. **Paper anchor (exists):** `experiments/switching_paper` reproduces the
   paper's Fig. 15(b): whole-shot d-round memories, one window a shot, cluster
   gap, 20 dB, belief matching with 20 BP iterations. Record where its
   results live and the numbers it produced; it is the accuracy reference and
   is not re-run unless the decoders change.
2. **System baseline (new):** the windowed 10d-round setup below. It is not
   expected to match the paper's numbers; it is the setup every later slice
   uses.

## Fixed datasets

Every run decodes saved syndrome data, never freshly sampled data, so the
data cannot change over time and a changed result always comes from a code
change. When something goes wrong, the exact shot can be found again.

- For each (distance, p) in [parameters.md](../parameters.md), sample
  1,000,000 shots once and save each shot's detection events and true
  observable flips, beside the circuit file and a sha256 manifest.
  Roughly 75 MB per point at d = 5 to 0.8 GB at d = 11.
- A file-backed syndrome source reads shot k from the dataset instead of
  sampling it. It implements the existing syndrome source port, so nothing
  else in the machine changes.
- Every result records the dataset's hash. The existing per-shot
  `sample_digest` (`experiments/measure.py`) must match the dataset.
- A dataset belongs to one circuit. Changing decoders, windows, thresholds,
  timing or controllers reuses it; changing the noise model, shot length or
  workload needs a new dataset.
- Calibration and evaluation use separate shot ranges of the same dataset.

## Work

1. Pin: Z memory, rotated surface code, SD6 at p, noiseless time boundaries,
   `10 * d` rounds, τ_gen = 1 µs, union find with `CycleCount`, cluster gap,
   belief matching (product-sum, iteration cap from the open decision), one
   weak unit, one strong unit, link cards, seeds.
2. Windows: sliding, commit d, buffer d, ordinary redo, held boundaries,
   threshold 20 dB. Set `terminal_policy: lookahead` explicitly on all three
   machines. `WindowSettings` defaults to `flush` and adding a strong tier
   forces `lookahead`, so leaving it implicit gives weak-only and strong-only
   a different last window from switching whenever the shot exceeds 3d rounds.
3. Build strong-only from the same weak machine (as `machines.strong_alone`
   does), so geometry is shared by construction.
4. Export each pilot point's window manifest: each window's read and commit ranges
   and its predecessor. Count windows rather than assume ten.
5. Stage the million-shot stop rule (`CollectionSettings(max_shots=N)`, no
   `max_failures`) without launching the grid.

## Gate and evidence

The three manifests are identical. Commit ranges cover the shot exactly once.
The final readout feeds the scored observable. Circuit hash, package
versions and every still-open choice are recorded in the experiment folder,
readable without this chat.

## Scope boundary

No controller, no timing claims, no double windows. Do not change noise or
shot length while diagnosing a decoder discrepancy.
