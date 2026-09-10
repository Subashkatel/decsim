[decsim docs](../README.md) › [Explanation](README.md)

# The design decisions

Twelve decisions shape what decsim charges and where it charges it. Each is
recorded here with what was decided, why, and the source the answer came
from, because a modelling question is answered by reading the referent
rather than by choosing (`STYLE.md` rule 8). The last section says what
is not modelled yet.

## D1. Whoever executes a send is an end of that hop

**Decided.** A component may name a link path only if it is one of that
path's two ends. The decoder manager schedules, says when inputs move
and returns results; it executes no send. The frame publisher lives on
the decoder side rather than in the window manager.

**Why.** The tempting alternative, "the producer pushes", is not a
general law: in Caune's system the decoder writes a register that the
sequencer reads, so the reader is the mover. The narrow rule holds
everywhere and can be checked mechanically.

**Sources.** OMNeT++ refuses at runtime a module that sends a message it
does not own (`src/sim/csimplemodule.cc:333-334`, omnetpp-6.1.0); gem5
bills a transfer to the port it left by rather than to whoever arranged
it (`packet.hh:424-431`); Yang et al. arXiv:2605.04892 book the frame
update inside the decoder's own subtotal.

**Where to see it.** `tests/test_send_ends.py`, whose `ENDS_OF_PATH`
table is the rule, and which walks `decsim/` to enforce it.

## D2. The complementary gap is two forced-class jobs of one window

**Decided.** The second gap solve is not a job of its own kind. The
window side submits the window's two forced-class jobs at one instant,
both into the weak pool, and the two weights are subtracted in the
confidence component rather than in the decoder manager. The base case
is one decoder unit running both solves one after the other. Whether the
pair overlaps is decided by the unit count and by
`escalation.run_both_at_once`, and by nothing else.

**Why.** The two forced classes are unequal work, measured at roughly
two to four times apart, so "two units finishing together" is not a
shape that exists. A scheduler that computed or combined a confidence
would also be doing a component's job.

**Sources.** Toshio et al. arXiv:2510.25222 for the unequal work; the
classical referents for fine-grained parallel work all fork in the
requester, feed each worker from the source, and join in the requester
rather than in the scheduler, with one code path parameterised by the
worker count (Cilk's serial elision, an MCS barrier at one processor,
CUDA blocks in any order).

## D3. The complementary gap is the exact reference; the cluster gap is the real-time default

**Decided.** The complementary gap stays as the exact reference metric.
Meister's cluster gap is a row of the same port beside it, and it is
what a real-time system would run.

**Why.** Toshio's own device computes its soft output by Meister's
method rather than by running two matchings. The complementary gap also
does not survive a change of decoder: its in-class weight is unreliable
for a non-matching decoder, while the cluster gap "is suitable for both
the MWPM decoder and the Union-Find Decoder".

**Sources.** Toshio arXiv:2510.25222 Fig. 3(c,d) and its reference to
Meister; Meister arXiv:2405.07433 Algorithm 2; Lee, English and Bartlett
arXiv:2510.05795 on why the logical gap method is generally not
compatible beyond the MWPM decoder.

**Where to see it.** `CONFIDENCE_SIGNALS` in
`decsim/confidence/signals.py`.

## D4. One decoder manager over both pools, and its work is not free

**Decided.** A single decoder manager schedules over both the weak and
the strong pool. Its own work is charged: `decoder_manager.dispatch_cycles`
on a named clock, defaulting to zero.

**Why.** Every classical referent read runs one scheduler over many
workers rather than one scheduler per class of worker. And a component
that costs nothing is inconsistent with a simulator that prices every
other component: a zero was a defect, not a modelling choice.

**Sources.** Skoric's parallel-decoder condition, StarPU's scheduling
handbook, and Caune's measured 250 to 370 cycles of dispatch work.

**Where to see it.** `configs/reference.yaml`, the `decoder_manager`
section.

## D5. The strong buffer hop is a priced link like every other

**Decided.** `controller_to_strong_buffer` gets the same card as every
other link: latency, clock, bits per cycle, channels, setup. Both
buffer-to-decoder hops price the actual bits. "Unwired" is not a state a
fabric can be in, so no hop can silently be free.

**Why.** A hop that could be unwired was a hop that could be free
without anyone saying so.

**Sources.** Caune et al. arXiv:2410.05202 Fig. 1a, stage D for the
weak-side write (40 nanoseconds) and stage F for the inter-node
broadcast (240 to 260 nanoseconds at the stated worst case).

**Where to see it.** `decsim/links/link_profiles.py`, and
[The data path, hop by hop](data_path.md) hops 2 and 3.

**Flagged with the decision.** The reference card's weak-side sum
exceeds Toshio's own communication time for the weak side, because the
card's nine latencies are taken from one source's table rather than
derived from one referent. Re-deriving the fabric card from a single
referent is open work.

## D6. Pair placement is deferred

**Decided.** Whether the two forced-class jobs of a window are kept
together on one unit or spread across two stays a scheduler row, to be
added when it matters.

**Why.** The record gives timing as the reason and no more. It carries
no source, and this page does not invent one.

## D7. A priced weak card is charged once per forced solve

**Decided.** When a weak tier is priced by a card rather than named, the
card is charged once per forced solve. A card weak tier under switching
therefore costs two cards per window.

**Why.** A card prices one decode call, and a forced pair is two calls.
Charging once per window would make two units no faster than one, which
the measurements contradict.

**Source and a warning.** Toshio's weak-decoder time is per window
including the soft output, so a reader reproducing the paper's number
sets the card to half of it.

**Where to see it.** `configs/reference.yaml`, the `weak_decoder`
section's comment on the forced solve.

## D8. The cluster gap's own walk is charged on the weak unit's clock

**Decided.** The time spent walking the cluster structure to produce a
cluster gap is charged on the weak decoder unit, attributed to the unit
that produced the evidence. `escalation.confidence_walk_microseconds` is
the card; leaving it null puts each row on its own cost model.

**Why.** Toshio's device computes the soft output on the weak decoder,
so that is where the work happens. Before this decision the walk was
charged to nothing at all, which is a component running for free.

**Source.** Toshio arXiv:2510.25222, Fig. 1 and its caption, for where
the soft output is computed; Meister arXiv:2405.07433 Algorithm 2 for
what the walk is.

## D9. The boundary payload is priced dense by default

**Decided.** The message crossing a window seam is priced as the seam
layer's whole detector count, `d*d - 1`, with a sparse row beside it.

**Why.** Both compiled implementations that were read carry a mask over
the layer, so the dense cost is independent of the noise, which is what
a parameter sweep wants. The sparse form is kept as a second row
precisely so that the bandwidth claim can be tested rather than assumed.

**Sources.** Stim's own generated circuits give `d*d - 1` detectors per
round layer, which is 8, 24 and 48 at distances 3, 5 and 7; quits and
cuda-q QEC both update one layer; Skoric et al. arXiv:2209.08552 and
Bombin et al. arXiv:2303.04846 for the sparse form.

**Where to see it.** `BOUNDARY_PAYLOADS` in
`decsim/windows/settings.py`, and the two rows in
`decsim/windows/boundary_payloads.py`.

## D10. The policy instance is the authority, not the settings row

**Decided.** Which tier a policy decodes on is answered by the policy
object itself, through facts declared on the port, never by a settings
key beside it and never by the object's class.

**Why.** The settings row and the object could disagree, and did: the
same run named one way and built the other way routed to different
decoders and finished at different ticks.

**Sources.** sinter resolves a caller's own object first and its table
second; gem5's port API is what a port promises not to reveal about a
row's class.

**Where to see it.** `decsim/build/escalation.py`, `escalation_row`,
which returns the built policy when there is one; and
`tools/check_row_recognition.py`, which fails on any module that tests
against a row's class.

## D11. The receiving end handles the landing

**Decided.** At a priced hop the sender's delivery callback does one
thing: it calls one method of the receiving end, or its own accounting.
Everything the receiver owns, its record, its publication tick, its
announcement to whoever waits on it, happens inside that call, in the
receiving package. On `controller_to_weak_buffer` the controller's
transmitter sends and hears the landing for its count of the rounds on
their route; Buffer 0's own incoming port stamps the publication tick on
the store's record and tells the window manager. On
`controller_to_strong_buffer` the rule already held: the strong writer
is the receiving end and stores each round at its landing. The record of
a Buffer 0 round is still written before the wire is used, because a
store answers for room before a round leaves; what the landing adds is
the publication, and the announcement never precedes it.

**Why.** The tempting alternative, "the sender that arranged the
transfer finishes the job", puts one component's state in another
component's callback: the controller stamped Buffer 0's record and woke
the window manager, so a reader of Buffer 0 could not see when its own
rounds became readable, and the two ends of one hop could drift apart
without either file changing. The narrow rule is the one every referent
keeps, and it can be checked mechanically.

**Sources.** gem5's requesting port hands the packet to the peer's own
receive method rather than writing the peer's state
(`src/mem/port.hh:603-614`, whose `src/mem/protocol/timing.cc:49-53`
calls `peer->recvTimingReq(pkt)`, declared at
`src/mem/protocol/timing.hh:170-172`), and bills a transfer to the port
it left by (`packet.hh:424-431`). OMNeT++ hands the message's ownership
to the destination module before that module's `handleMessage` runs
(`src/sim/csimplemodule.cc:777-799`, `take(msg)` at 783) and refuses a
send of a message the sender does not own (`333-334`). ns-3's
point-to-point channel schedules `PointToPointNetDevice::Receive` on the
destination device (`point-to-point-channel.cc:88-92`), and that receive
is the destination device's own method
(`point-to-point-net-device.cc:324`). That the room for a round is
answered before the wire is used is gem5's queue, which reserves an
entry before the send (`src/mem/cache/queue.hh:150-152`); that the
publication never precedes the store is the buffer contract of the
behaviour gate. The two hops keep the referents they had: Caune
arXiv:2410.05202 Fig. 1a stage D for hop 2's latency, and Toshio
arXiv:2510.25222 line 1248 for what Buffer 1 is assigned.

**Where to see it.** `decsim/syndrome_buffer/round_input.py`, the
`RoundStoreInput` port in `decsim/ports.py`, and `tests/test_send_ends.py`,
whose receive-end law reads every delivery callback out of the tree. The
same rule decides who writes a structure at a handoff off the wire: a
window's boundary mask is computed by the window side and written into
the decoder unit's memory, or into the masked duplicate the unit reads,
by the decoder side that owns both (`DecoderInputFold`).

## D12. A hop's ends are the packages that hold the objects at them

**Decided.** `ENDS_OF_PATH` names, for each hop, the packages that hold
the object the message leaves and the object it lands in, not the
hardware the card is named after. On `decoder_to_decoder` both of those
are the window side: the boundary is a record the courier keeps for a
committed window, and it lands in the destination window's own record,
where it waits until the decode that reads it starts. So the courier
executes that send and handles that landing, and the decoders package,
which held a three-line pass-through with no state of its own, holds
nothing of this hop any more. The card is unchanged: it still prices the
0.5 microsecond on-chip wire between two decoders.

**Why.** The rule that decides it is ownership. A component may send
only what it owns, and nothing in the decoders package owned the
boundary: `DecoderOutput.send_boundary` forwarded an attribution and a
bit count it had not made, to a delivery callback that belonged to the
window side, and no decoder-side object heard the landing at all. The
alternative, moving the boundary's state into the decoders package, was
turned down because it would move the window model with it: what a
boundary is, how two boundaries merge, which version wins and when a
window may start are the windowing scheme's laws, and a decoder unit in
decsim is a plug-in that decodes one job and keeps no window state.

**Sources.** OMNeT++ refuses at runtime a module that sends a message it
does not own (`src/sim/csimplemodule.cc:333-334`), and gem5's requesting
port names a peer that receives (`src/mem/protocol/timing.cc:49-53`),
which on this hop did not exist. What crosses is Skoric's exchange
between decoding blocks: "Once DA_i finishes decoding, it sends the
artificial defects and unresolved syndromes from the bottom d rounds to
DB_{i-1} ... When the data from DA_i and DA_{i+1} has been received, the
DB_i block can start decoding" (arXiv:2209.08552,
`2209.08552.txt` lines 1038-1046). decsim's model of a decoding block's
own state, what it has received and whether it may start, is the window
record, which is what these two ends hold.

**Where to see it.** `decsim/windows/window_boundaries.py`, the
`ENDS_OF_PATH` row in `tests/test_send_ends.py`, and hop 7 of
[The data path, hop by hop](data_path.md).

## What is not modelled yet

These are open, recorded rather than hidden, so that a reader does not
mistake a gap for a result.

- **O1. The switching gate points still run on the host wall clock.**
  The three switching points of the behaviour gate price two real
  decoders from the measured clock, at weak 12 to 119 microseconds and
  strong 0.94 to 46 milliseconds against a one microsecond round. That
  is the whole source of order and queue-depth variance in the gate. The
  proposal on the table is a latency key on the decoder section, with
  the two points priced at Toshio's generation time and ten times it.
- **O2. The `bandwidth_limited` link row cannot be named from a yaml.**
  Its card is built before the sweep point sets the geometry, so
  reaching it from a config would mean building the links card inside
  the per-point settings.
- **O6. `MAGIC_STATE_FACTORIES` has a table and no yaml section.** Its
  rows cannot be selected from a config the way every other table's rows
  can.
- **O7. The parallel windowing scheme refuses the decoder-side formation
  row.** Skoric's A and B blocks read disjoint round ranges, and the
  formation component forms in round order, so it is asked for a later
  round while standing at an earlier one and refuses. What it wants is a
  former that forms in arrival order from the tier's own store, which is
  also how LILLIPUT's block and Yang's stage run: on the stream, not on
  the window.
- **O8. `decsim collect` refuses a `timing_only` device.** The device
  builds and runs as a machine, but the front's per-shot measurement
  compares the loop's prediction against PyMatching on the sampled shot,
  and a timing-only device samples none, so `collect` raises `KeyError`.
  A priced card on a real device is the way to a host-independent run
  today.
- **O9. A decoder section ignores a key it does not know.** The
  `weak_decoder` and `strong_decoder` sections do not yet refuse an
  unknown key by name the way `decoder_manager` does, so a misspelt key
  there runs the default in silence.

A sixth row, O4, was a real mispricing of a backward hand-off in the
parallel scheme, and it is closed: the two layers that differ are now
tested. A seventh, O10, said that a timing-only round's landing reached
no object of the receiving package; it is closed too, by the decoders'
own end for such a round (`decsim/decoders/memory_rounds.py`). An
eighth, O5, said the boundary fold was executed by the window gate
because moving it would reorder the copy trace sources; it is closed as
well, and no trace source moved: the gate hands the mask and the
decoder side writes it (`decsim/decoders/decoder_memory_transfer.py`,
D11). A ninth, O3, held two payload-source strings that named no real
field until the golden file next moved; it is closed by naming what the
sends carry, `QPUReadout.size_bits` on the readout hop and no payload at
all on the escalation hop, under one regeneration note.

## Read next

- [The principles behind the shape](principles.md): the ideas these decisions were made
  under.
- [The data path, hop by hop](data_path.md): D1, D5 and D9 as they appear on the
  wire.
- `STYLE.md` rule 8: why a modelling question is answered from a source.
