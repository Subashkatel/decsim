# 02 Timing table

## Question

Does every simulated time come from timing we supply, so results describe the
modeled hardware and not the host?

## Deliverable and ownership

A timing table that any decoder row can take, a calibration script that
fills it from runs on a target device, and a refusal of host time in timing
runs. The table plugs in where each decoder already sets its service time
(`ticks_after_decode` in `decsim/decoders/decoder.py`). Precedent for a
table loaded before the machine is built: `TableThreshold` in
`decsim/escalation/threshold_sources.py`.

## Current state

- Union find: `CycleCount` computes cycles from the decode's own grow, merge
  and peel work at a clock (`union_find/cycle_count.py`); constants from
  Helios RTL. No host time.
- Belief matching: `timing=None` (the default) charges host wall clock;
  `IterationTime` charges µs per BP iteration × the iterations BP ran, plus
  µs for the matching. `STRONG_BASELINE_POOL` uses host time.
- Cluster gap: `walk_microseconds=None` charges host time.
- `measured_table` holds Relay-BP numbers only and picks the nearest cell
  with no bound. Not used.
- Links are priced by default (store, forward, answer, decision, command
  hops) with FIFO channels.

## Work

1. Table columns: `decoder`, `device`, `distance`, `window_rounds`, `model`,
   values, `physical_error_probability` (only where the model needs it),
   provenance (device, software, date, method).
   - `fixed`: one service time per cell.
   - `per_iteration`: `per_iteration_us × iterations + matching_us`; the
     decode's own iteration count makes it follow p.
   - `samples`: measured per-decode times, drawn per decode, so tails come
     from data.
2. Exact match only. A missing cell refuses the run. Window size is a key
   because ordinary redo reads 2d rounds and the double window 3d.
3. Refuse host time in timing runs: belief matching without a table or
   `IterationTime`, a gap walk without `walk_microseconds`.
4. Calibration script: run the decoder on the target device over sampled
   windows at each distance, window size and p in the grid; write rows with
   provenance. The decoder's answer still comes from decsim's belief matching;
   only its time comes from the table.
5. Ask the collaborator to run the calibration for belief matching on the
   GPU farm (BP iteration time and matching time per window size).

## Gate and evidence

Hand-calculated fixtures: one job, two jobs FIFO, link serialization, a
dependency wait, a feedback return, each matching to the tick after the
documented clock rounding. Changing one table cell changes only the stages
it should. On a fixed nonadaptive memory run, decoder answers are unchanged
by any timing change.

## Scope boundary

Synthetic test values are labeled synthetic. Table values describe assumed
hardware; whether they are realistic needs separate sourcing and a
sensitivity check before claims.
