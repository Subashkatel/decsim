# Terminology

The words this codebase uses, and the ones it deliberately does not.

## Stores

- **Buffer 0** (`SyndromeBuffer`, `decsim/syndrome_buffer/`): the
  upstream store the weak lane reads. Refcounted holds, orphan until
  arrival, tombstoned late writes, refuse before mutation.
- **Syndrome buffer 1 / SB1** (`SyndromeBuffer1`): the room-side store
  the strong tier reads; owns the CSB crossing and the stored-round
  arrival gate. "Room-side" is physical: the strong context lives at
  room temperature.

## Link segments (`LinkPath`, decsim/links/links.py:239-252)

| Segment | Meaning |
|---|---|
| QC | QPU to controller: syndrome readout leaving the QPU |
| CWB | controller to syndrome buffer 0: a completed binary round published |
| CSB | controller to syndrome buffer 1: the packed round's second (dual) write |
| WBD | weak buffer to weak decoder: syndrome data reaching the weak tier, or a feedback-memory round straight off packing |
| WSD | weak decoder to strong decoder: the escalation selection handing a window to the strong tier |
| SBD | strong buffer to strong decoder: the strong window's input, assembled from syndrome buffer 1 |
| WDO | weak decoder to Pauli frame: the weak correction leaving the weak tier |
| DD | decoder to decoder: a committed window boundary handed to a dependent window |
| DO | strong decoder to Pauli frame: the strong correction leaving the strong tier |
| OC | Pauli frame to controller: the conditional release returning |
| CQ | controller to QPU: the instruction delivered back |

The segment vocabulary is the fixed measurement decomposition, closed
in code; a new measured segment is added by a declared three-line
extension, never by run configuration (tmp/validation/ORIENTATION.md).

## Tiers and switching

- **weak / strong**: the two decoder tiers (Toshio 2510.25222). The
  weak tier is fast and gives soft output; the strong tier is accurate.
- **complementary gap**: the weak decoder's confidence measure; below
  `gap_threshold_db` the weak result is untrusted.
- **serial switching**: escalate after the weak result; **parallel
  switching**: strong submitted alongside the weak; **double-window**:
  the strong window is deferred (terminal data or far boundary).
- **forward strong window**: the strong window that absorbs the windows
  it covers and restarts the weak chain past it.
- **absorbed window**: a window skipped by the weak chain because a
  strong window covers it.

## Windows

- **commit region / buffer region**: the rounds a window commits vs the
  lookahead it reads (Skoric 2209.08552; SWIPER 2412.05115).
- **window plan**: compile-time windows (`WindowPlan`); **dynamic
  stream**: windows created at runtime by `DynamicWindows`.
- **boundary courier**: ships a committed window's boundary to its
  dependents (`BoundaryCourier`).
- **held boundary**: a boundary policy that defers publishing until
  permitted (`Held`).

## Requests and units

- **request key / run sequence**: one decode request's identity and its
  global admission order (`DecoderRequestKey`).
- **destination**: the weak window a strong result is for
  (`strong_decode_for`).
- **two-slot unit**: depth-1 decoupled access-execute decoder unit; two
  input slots, one compute; a **parked** job holds a slot but released
  its compute claim (boundary owed).
- **CsdInput / PotentialStrong / PendingStrong**: typed hold tokens on
  SB1 for a submitted, planned, or deferred strong request's context.
- **external job**: a decode with `on_done` and no syndrome data
  (factory corrections, separate idle decodes).

## Rounds

- **idle round**: syndrome extraction of a patch nobody operates on;
  never stops; the idle policy routes it.
- **feedback-memory round**: an idle round traveling WBD as a
  timing-only round of an operation.
- **memory-filled buffer**: a trailing buffer region satisfied by
  memory rounds alone (time-only, no syndrome content); flagged, never
  hidden.
- **stored-through**: the per-operation `rounds_arrived` convention:
  the counter names the highest arrived round and, because arrival is
  in order (see buffer contract), reads as "stored through round N".

## Words deliberately not used

- "cache" for the stores (they are retention buffers with holds).
- "ASIC latency" for measured Python wall clock: the strong tier and
  the switching weak tier price latency from the measured wall clock of
  the room-side software decoder; it is never presented as hardware
  timing.
