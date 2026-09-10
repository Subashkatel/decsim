[decsim docs](../README.md) › [Explanation](README.md)

# The design decisions

Ten decisions shape what decsim charges and where it charges it. Each is
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
[`docs/explanation/data_path.md`](data_path.md) hops 2 and 3.

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
- **O3. Two payload-source strings name no real field.**
  `SyndromePayload.size_bits` and `switching decision payload_bits`
  travel into the gate's link traffic and are held until the next time
  the golden file moves, so the correction is not made twice.
- **O5. The boundary fold executes in the window gate.** Moving it where
  the design would otherwise put it reorders the copy trace sources,
  which the gate pins.
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

A sixth row, O4, was a real mispricing of a backward hand-off in the
parallel scheme, and it is closed: the two layers that differ are now
tested.

## Read next

- [`docs/explanation/principles.md`](principles.md): the ideas these decisions were made
  under.
- [`docs/explanation/data_path.md`](data_path.md): D1, D5 and D9 as they appear on the
  wire.
- `STYLE.md` rule 8: why a modelling question is answered from a source.
