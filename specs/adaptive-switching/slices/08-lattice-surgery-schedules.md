# 08 Lattice surgery schedules

## Question

Can real lattice-surgery schedules drive the decoding system, so gate
deadlines, decoder load and stall tails come from actual programs rather than
a synthetic chain?

## Current state

- A schedule lands as a `decsim.ops/1` operation list
  (`frontends/workload_files.py`): each operation has `kind`
  (MERGE/INJECT/MEASURE), `rounds`, `patches`, `predecessors` and
  `blocked_by`. Round counts per kind come from `qpu/round_policies.py`.
- One operation is one decoding problem, so a merge is decoded jointly.
- decsim builds no merged circuit (`frontends/circuit_frontend.py`). With no
  circuit, a merge is timing-only, sized by patch count plus a seam.
- A workload holds one stream with one protected region; the critical-path
  scheduler refuses streams.
- Patches waiting on a decision send rounds with no syndrome values: decoder
  load without real syndromes.
- In a timing-only run there are no gaps, so escalation decisions need a
  score source: replayed gaps from memory runs or synthetic gaps.

## Work

### 08a: timing through lattice surgery

1. Choose a compiler by what its export actually contains: operation ids,
   patch placement, merge/split durations in rounds, and which dependencies
   are classically conditioned. Only conditioned dependencies become
   `blocked_by`; a plain dependency graph does not say which ones wait on a
   decode.
2. Importer at the frontend boundary into `decsim.ops/1`.
3. Score source for timing-only runs: replay per-window gaps from slice 04
   runs at the matching distance and p, labeled as replayed.
4. Validate a small published schedule (a lattice-surgery CNOT, then a T
   injection with its conditioned S) against an ideal-classical-cost reference,
   then add the slice 02 timing and explain every extra wait.
5. Run a small algorithm schedule with the slice 06 loop reading gate stalls.

### 08b: real syndromes through merges

Only if the paper needs LER through lattice-surgery operations (open
decision). Needs a merged physical circuit from an external tool, fed as
stream segments, and support for more than one stream or protected region.

## Gate and evidence

Patch occupancy, rounds and dependencies match the source schedule. Every
wait beyond the ideal reference is attributed to a decoder, link or factory.
Compiler name and revision are recorded.

## Scope boundary

A CNOT example or a Stim export does not establish T-gate support. Replayed
gaps are not real gaps of the merged problem.
