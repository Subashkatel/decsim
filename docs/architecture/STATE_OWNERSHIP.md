# State ownership

The rule: every mutable runtime fact has exactly one owner; everyone
else reads through the owner or receives a notification. The exhaustive
field-by-field table (created by / writers / readers / invariant /
verdict) lives in `reports/DECSIM_RUNTIME_STATE_OWNERSHIP.md`; this file
records the ownership boundaries a change must not blur.

## Owners and their state

- **ExecutionRuntime**: `operations`, `dependencies_remaining`,
  `successors`, `schedule_released`, `requested`, `state_ready`, and the
  four timestamp maps (`op_start_time`, `body_done_time`,
  `decode_release_time`, `result_return_time_by_operation`).
  ResourceLedger inside it owns `busy_claims` (one holder per
  resource). Nobody else writes these.
- **Controller**: stateless apart from `idle_rounds_emitted` and its
  immutable resolved maps. It relays; it does not decide.
- **SyndromePacking**: `_contexts` (PARTIAL / PACKED_WAIT / DRAINING),
  `_route_queues`, `_packed_rounds`, `_dropped_rounds`, drop and
  timeout counters. The stores never see a partial round.
- **SyndromeBuffer (Buffer 0)**: round slots, publication ticks,
  refcounted holds, open/closed operation sets, tombstones. Holds are
  the ONLY retention mechanism; consumers never flag rounds directly.
- **SyndromeBuffer1**: the CSB crossing (`_in_flight_writes`,
  `_written`), room-side `rounds_arrived`, `copied_bits_total`, and its
  composed SyndromeBuffer store. WindowManager reads `rounds_arrived`
  through the stored-through convention; nothing outside writes it.
- **WindowManager**: `windows`, `op_windows`, `window_count`,
  `rounds_arrived` (Buffer 0 authority), `memory_rounds`,
  `committed_windows`, `op_results`, `ledger` (LogicalLedger),
  `courier` (BoundaryCourier), retention-hold bookkeeping, and the
  lifecycle (DynamicWindows). StrongEscalation is the one object
  outside the manager allowed to work these tables directly; it is
  constructed with the manager and documented as such.
- **DecoderManager**: pool queues, `_free_units`, `_unit_residents`,
  `_computing`, `_parked_service`, `decoder_memories`, `queue_log`,
  gap joins. `StrongRequestLedger` inside it owns request admission,
  selection, held completions, and the strong counters.
- **StrongEscalation**: `_EscalationRegistry` (pending deferred strong
  windows and their single-shot readiness indexes).
- **PauliFrame**: the XOR frame, its record list, duplicate-drop count.
  Exactly one authoritative correction per window may reach it.
- **ConditionalRelease**: blocked-operation registry and delivered
  decisions; the only path from decode results back to the Controller.

## Documented delegations (not violations)

- StrongEscalation reading WindowManager tables: deliberate, single
  named exception (strong_escalation.py class docstring).
- WindowManager holding `submit_fn` / `withdraw_decode` /
  `release_service` closures into DecoderManager: the two managers are
  mutually recursive by construction (run_spec.py:219-275); the
  closures are the seam.
- SyndromePacking writing both stores: the dual write is the packing
  stage's single responsibility boundary crossing, priced per store.

## Change rules

- A new consumer of rounds must take a store hold; never copy payloads
  out of a store into private state.
- A new readiness signal must be a notification (id, never payload) and
  must not perform decoding work in the callback.
- Nothing may write another owner's counters; expose a method on the
  owner instead.
