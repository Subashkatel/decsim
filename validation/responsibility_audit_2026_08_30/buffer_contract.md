# Buffer contract: Buffer 0 and syndrome buffer 1

Two stores, two different readiness authorities. The mode diagrams live
in `docs/architecture/BUFFER_0_AND_BUFFER_1.md`; this file records the
invariants and where they are enforced.

## Buffer 0 (upstream store, weak lane)

- Written by SyndromePacking at round completion; publication tick is
  the completion tick, or the CWB landing when CWB is priced.
- Notification is `accept_window_input(packet)`, and it may not precede
  publication: `WindowManager._store_payload` raises unless the packet
  is already retained AND its publication tick equals the current tick.
- In weak-primary modes, Buffer 0 arrival advances the readiness counter
  (`_count_arrival`). In strong-primary plans it does not; readiness is
  driven from syndrome buffer 1.
- CWB keeps round order explicitly: FIFO route queue, a refused round
  blocks the rounds behind it (`_refused_round_ahead_of`).
- Rounds stay retained until their consumers' holds release (decoder
  input landing, WBD delivery, or drop-on-arrival when nothing holds
  them).

## Syndrome buffer 1 (room-side store, strong tier)

- Dual write: every packed round is written exactly once toward SB1 in
  parallel with its Buffer 0 publication; the CSB hop is priced when the
  card wires it. Capacity is checked before the link reservation,
  counting in-flight writes.
- `on_round_stored(operation_id)` fires strictly AFTER storage; the
  callback carries the operation id only, never the payload.
  WindowManager reads `syndrome_buffer_1.rounds_arrived[op]`. Do not
  turn this into a packet callback.
- The data stay owned by SB1 until a strong request exists, a unit is
  assigned, SBD is reserved, and the selected rounds land in decoder
  memory.
- `_on_room_round_stored` wakes `escalation.after_arrival` in every
  mode; it advances window readiness only when the strong tier is
  primary.

## Ordered arrival

The per-operation `rounds_arrived` counters (WindowManager's for Buffer
0, SB1's own) advance with max(previous, round_index) and are read as
"stored through". This is sound because per-operation arrival is
in order for every supported configuration (Proof A), and any violation
is fail-loud, never silent (Proof B).

Proof A (in-order): the QPU emits rounds on one cycle clock, round index
strictly increasing; links are strict FIFOs (`Link.reserve` rejects a
send tick earlier than the prior reservation and serializes with a
monotone next-free tick); the engine fires same-tick events in insertion
order; controller binary availability is a constant per-fragment delay;
per-operation fragment counts are constant in every shipped device
model, so the uniform t_pack preserves completion order; SB1 writes
issue in completion order and land FIFO over CSB.

Boundary of Proof A, documented: a device model emitting varying
fragment counts per round of one operation with t_pack > 0, or the
opt-in DROP_ROUND overflow policy, can create a gap under the counter.

Proof B (fail-loud exactness): the counters only decide WHEN submission
is attempted; every consumer verifies exact rounds before compute:

- `SyndromeBuffer1.ready_tick` raises for any unstored round ("never
  silently served early") and is called inside every strong
  `reserve_transfer`, so an SBD DMA cannot start over a gap.
- Deferred strong job build checks the exact covered round set equals
  the needed set and raises otherwise.
- The immediate serial path raises when context reached Buffer 0 but is
  not stored in SB1 ("csb lag beyond the escalation margin"). This
  fail-loud check is deliberate; replacing it with waiting would be an
  architectural change needing its own design note and timing
  validation.
- `_require_retained_payloads` refuses consumers of released rounds.

## Notification role by mode

- weak-only: SB1 is not constructed; Buffer 0 is the only authority.
- strong-only: SB1 stored-round notification drives normal readiness;
  primary window jobs read SB1 and price SBD; no escalation.
- serial switching: Buffer 0 drives weak readiness; an untrusted weak
  result builds the strong job immediately (fail-loud CSB-lag check) or
  defers it (double-window phases).
- parallel switching: weak and strong submitted together; the strong
  DMA start clamps to the last CSB landing (`ready_tick` inside
  `reserve_transfer`), so Buffer 0 readiness ahead of Buffer 1 only
  delays the DMA, never reads unstored data; a trusted weak result
  cancels the strong work.
- double-window terminal: every SB1 store wakes `after_arrival`; the
  strong job submits exactly once when the stored-through counter
  reaches context_hi (single-shot registry take), and the exact
  round-set check still guards the build.
- double-window far boundary: submission triggers on the far weak
  commit, not on an SB1 callback; the WSD reservation must already
  exist, and SBD still gates on `ready_tick`.
