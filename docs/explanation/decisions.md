[decsim docs](../README.md) › [Explanation](README.md)

# The design decisions

Nineteen decisions shape what decsim charges, where it charges it, and
where a reader finds a thing. Each is recorded here with what was
decided, why, and the source the answer came from, because a modelling
question is answered by reading the referent rather than by choosing
(`STYLE.md` rule 8). The last section says what is not modelled yet.

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

## D4. One decoder manager per side, and its work is not free

**Decided.** Two instances of one decoder manager class: the chip's
over the weak pool, and, in a run whose windows may escalate, the
host's over the strong pool. The escalation side and a window's strong
sibling submit to the host's; the ledger of strong requests is one seat
both take, since the chip's side opens a request and the host's serves
it; a kept weak result halts its request through the escalation side.
Each manager's own work is charged: `decoder_manager.dispatch_cycles`
on a named clock, defaulting to zero, the one card both read.

**Why.** The systems that put a fast decoder beside the control
electronics and a slow one on a host give each side its own scheduling:
LATTE's local decoder on the control FPGA has no scheduler of its own
and the host's Global Dynamic Scheduler owns the decode queue and the
thread pool (2509.03954 lines 24-25 and 705-720). The rack drawing puts
the strong decoder manager on the host, and the tree shows what the
drawing shows. A component that costs nothing is inconsistent with a
simulator that prices every other component: a zero was a defect, not a
modelling choice. Until this decision one manager served both pools
(every single-place referent runs one scheduler over many workers).

**Sources.** LATTE 2509.03954; Toshio 2510.25222 lines 606-614 (the
strong computation halted on a confident weak result); Skoric's
parallel-decoder condition, StarPU's scheduling handbook, and Caune's
measured 250 to 370 cycles of dispatch work.

**Where to see it.** `decsim/assembly.py`, the `decoder_manager` and
`strong_decoder_manager` rows; `configs/reference.yaml`, the
`decoder_manager` section.

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

**Narrowed since.** A switching run sends nothing on this hop: its
rounds stay in the weak syndrome buffer and the escalation carries the
strong window's rounds over `weak_decoder_to_strong_decoder` (hop 5),
the transport Toshio arXiv:2510.25222 lines 1247 to 1250 describe and
the one a cold weak tier exists for (Battistel arXiv:2303.00054 lines
342 to 347). The hop stays priced for the strong-only run, whose one
transport it is.

**Flagged with the decision.** The reference card's weak-side sum
exceeds Toshio's own communication time for the weak side, because the
card's six latencies are taken from one source's table rather than
derived from one referent. Re-deriving the fabric card from a single
referent is open work; D14 prices the strong tier's four off-board hops
from one measured round trip instead, on two rows a config can name.

## D6. Pair placement is deferred

**Decided.** Whether the two forced-class jobs of a window are kept
together on one unit or spread across two stays a scheduler row, to be
added when it matters.

**Why.** Timing. No referent answers this one, and this page does not
invent a source.

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
so that is where the work happens, and a walk charged to nothing is a
component running for free.

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

**Why.** The settings row and the object can disagree: one run named
one way and built the other way routes to different decoders and
finishes at different ticks.

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
their route; the weak syndrome buffer's own incoming port stamps the publication tick on
the store's record and tells the window manager. On
`controller_to_strong_buffer` the strong receiver is the receiving end and
stores each round at its landing. What a store does with a round is the
store's, at its own door. D13 settles when that door takes a slot.

**Why.** The tempting alternative, "the sender that arranged the
transfer finishes the job", puts one component's state in another
component's callback: a controller that stamps the weak syndrome buffer's record and
wakes the window manager leaves a reader of the weak syndrome buffer unable to see when
its own rounds become readable, and lets the two ends of one hop drift
apart without either file changing.

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
(`point-to-point-net-device.cc:324`). That the publication never precedes
the store is a law of the store, and its tests hold it. Each hop keeps
its own latency source: Caune arXiv:2410.05202 Fig. 1a stage D for hop
2, and Toshio arXiv:2510.25222 line 1248 for what the strong syndrome buffer is
assigned.

**Where to see it.** `decsim/syndrome_buffer/weak_syndrome_round_receiver.py`, the
`WeakSyndromeRoundReceiver` port in `decsim/ports.py`, and `tests/test_send_ends.py`,
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
executes that send and handles that landing, and the decoders package
holds nothing of this hop. The card is unchanged: it still prices the
0.5 microsecond on-chip wire between two decoders.

**Why.** The rule that decides it is ownership. A component may send
only what it owns, and nothing in the decoders package owns the
boundary: a send from that side would forward an attribution and a bit
count it had not made, to a delivery callback that belongs to the window
side, and no decoder-side object would hear the landing. The
alternative, moving the boundary's state into the decoders package, is
turned down because it would move the window model with it: what a
boundary is, how two boundaries merge, which version wins and when a
window may start are the windowing scheme's laws, and a decoder unit in
decsim is a plug-in that decodes one job and keeps no window state.

**Sources.** OMNeT++ refuses at runtime a module that sends a message it
does not own (`src/sim/csimplemodule.cc:333-334`), and gem5's requesting
port names a peer that receives (`src/mem/protocol/timing.cc:49-53`).
What crosses is Skoric's exchange between decoding blocks: "Once DA_i
finishes decoding, it sends the artificial defects and unresolved
syndromes from the bottom d rounds to DB_{i-1} ... When the data from
DA_i and DA_{i+1} has been received, the DB_i block can start decoding"
(arXiv:2209.08552, `2209.08552.txt` lines 1038-1046). decsim's model of
a decoding block's own state, what it has received and whether it may
start, is the window record, which is what these two ends hold.

**Where to see it.** `decsim/windows/window_boundaries.py`, the
`ENDS_OF_PATH` row in `tests/test_send_ends.py`, and hop 7 of
[The data path, hop by hop](data_path.md).

## D13. A round occupies a slot when its bits are in the store

**Decided.** One rule for both stores. A round takes a slot in a store
at the landing of the hop that carries its bits, in the receiving
package, and it is readable at that same instant: the store and the
publication are one call at one tick. The sender still refuses before it
sends, by asking that same end for room against the rounds it holds plus
the writes it has in flight, and reserving one before the round leaves.
`WeakSyndromeRoundReceiver` owns the weak syndrome buffer's room, its slot, its intake line and
its announcement, the shape `StrongSyndromeRoundReceiver` has, and the publication
tick is the `controller_to_weak_buffer` landing.

**Why.** The tempting alternative, "book the slot when the sender
commits the round", makes the refusal simple but makes every occupancy
figure mean something the name does not say: a store whose bits are not
there yet is reported full. Keeping the two counts apart, occupancy at
the landing and the in-flight writes on the end that answers for room,
gives the same admission decision with both numbers true. Credits would
give the same decision again, since each store has exactly one writer,
and are not built because they would add a return-path model nothing
needs.

**Sources.** Every referent that models storage writes it at the
landing. ns-3's channel schedules the destination device's own `Receive`
after the transmission and the propagation
(`point-to-point-channel.cc:88-92` into `point-to-point-net-device.cc:324`).
OMNeT++ takes ownership into the destination module and inserts inside
that module's handler (`src/sim/csimplemodule.cc:782-783`, `:799`, with
`queueinglib/Queue.cc:84-94` checking the capacity and inserting there).
Ciw counts the individual in the destination's own `accept`
(`ciw/node.py:602` into `:102-103`), and blocks the sender upstream when
the destination is full (`:470-473`). Ruby's arrival time is the tick
the message may be read (`MessageBuffer.cc:243`). In the quantum control
literature the store is on the far side of the wire too: Caune
arXiv:2410.05202 lines 1243-1247 stores the outcomes "in the decoder
sequencer's memory" only after a 1.4 microsecond propagation; Google
arXiv:2408.13687 lines 471-477 puts the shared memory buffer "inside the
workstation" reached "via low-latency Ethernet"; Maurer arXiv:2510.21600
lines 593-597 fills the syndrome FIFO at the decoder FPGA after the
serial link. That the room counts the writes in flight is gem5's queue,
whose `isFull` counts reserved entries (`src/mem/cache/queue.hh:150-153`,
the reserve at `:87-93`), its cache blocking the port when the write
buffer fills (`src/mem/cache/base.cc:255-257`, `:266-271`) and the
refusal being the receiver's answer (`src/mem/port.hh:244-255`); and
Ruby's `areNSlotsAvailable`, which sums the queue and the stalled
messages (`MessageBuffer.cc:181`, the two sizes read at `:155-158`).

**Where to see it.** `decsim/syndrome_buffer/weak_syndrome_round_receiver.py`, the
`SyndromeBuffer` and `WeakSyndromeRoundReceiver` ports in `decsim/ports.py`,
`tests/syndrome_buffer/test_weak_syndrome_round_receiver.py`, and hop 2 of
[The data path, hop by hop](data_path.md).

## D14. The strong tier's off-board path can be priced by a measured round trip

**Decided.** Two rows of `LINK_FABRICS`, `roce_v2_cpu` and `roce_v2_gpu`,
are the default card with four hops repriced from one measurement:
Backline's steady-state round trip from an FPGA controller to a CPU
coprocessor over RoCE v2, 2.305 microseconds in the median, and to a GPU
coprocessor, 4.5 microseconds. The controller's write into the strong
syndrome buffer (hop 3), the escalation request (hop 5) and the strong
decoder's reply to the frame (hop 9) are each half of the round trip; the strong
store's read into the strong decoder (hop 6) is zero, because the
coprocessor polls a slot in its own memory. On either row the escalation
round trip, hops 5, 6 and 9, is the measured median exactly. Every other
hop keeps the default card's number and source, and the default row is
unchanged.

**Why.** On the default card hop 5 has no source: its string reads
"repository weak-to-strong model choice". The three cited hops around it
come from one table each. A measured cable is a better kind of fact for
that path than a table row, and a config that asks "strong tier on a CPU
or on a GPU" needs a card behind each answer. The split is decsim's rule,
stated in the docstring, because the measurement is one number and
neither the paper nor its published runtime gives a per-direction
figure: one hardware timer starts at the controller's doorbell and stops
when the reply lands. The rows price the median; the first, warm-up
round trip (4.64 and 9.27 microseconds) and the tails are on the
docstring and not on the card.

**Sources.** Backline, Lee et al. arXiv:2609.09270, Sec. V-C1 lines 1607
to 1633 for the measurement (a 16-byte payload with an 8-byte syndrome
as a one-sided RDMA write, the coprocessor polling its buffer, the reply
as a one-sided write, the GPU signalling a CPU thread that writes back,
the FPGA timing the round trip in its own clock), Table III lines 1736
to 1745 for the echo rows the card takes, footnote 3 lines 1629 to 1631
for the excluded warm-up trip. The published runtime confirms the shape:
one 16-byte frame in a 64-byte slot of a 256-slot ring, the receiver
spinning on the slot's sequence number, the reply posted as an inline
RDMA write, the round trip read from the engine's own timer.

**Where to see it.** `roce_v2_measured_profile` and the two rows in
`decsim/links/link_profiles.py`, the `links.kind` comment in
`configs/reference.yaml`, and `tests/links/test_link_profiles.py`, which
holds the split, the exact escalation sum, the unchanged remainder of the
card, the four Backline citations and both rows running from a yaml.

## D15. The park before a decode is two points, by what it waited for

**Decided.** `dep_block` keeps its name and means the dependency wait
only: from the committing decode's input landing in its unit's memory,
or from the tick a unit took it when the input was already there, to the
first tick that decode may compute, which is where its window's last
boundary arrived and is the landing itself when nothing was owed. The
rest of the park, from that tick to the compute starting, is its own
point, `compute_wait`: the unit's compute was busy with another decode.
The tick that divides them is stamped per decode where the decode
becomes startable, on the job and on its stage records beside the
dispatch tick, so a window decoded more than once divides each decode's
own park. The identity closes to the tick with both points.

**Why.** One number for both hides the two answers a reader of a sweep
wants apart. A dependency wait is the windowing's doing and shrinks by
changing the geometry or the boundary policy; a wait on a busy unit is
the machine's doing and shrinks by adding units. Reported as one park,
a run with a long dependency chain and a run with too few units look
the same, and the second unit that fixes one of them moves the number
in a way nothing explains. The name stays `dep_block` because that is
what it now means exactly; renaming a column that keeps its meaning
would break every recorded sweep for nothing. `compute_wait` is a new
name and takes the tree's own word for the resource: the units claim,
release and free their `compute`.

**Widened since.** An escalated window's rounds cross with the
escalation and land in the strong store before the strong input hop can
start, so `dep_block` now runs from the verdict to the first startable
tick, less the input hop itself; the wait for the rounds is a
dependency wait by the same reasoning, the windowing's doing and not
the machine's. Every earlier number is unchanged, because until then
every input hop started where the decode's path did.

**Sources.** gem5's instruction queue keeps the two waits apart. An
instruction reaches the ready list only when its operands are there
(`src/cpu/o3/inst_queue.cc:1536-1562`, `addIfReady`, woken by
`wakeDependents` at `:1074`), and a ready instruction that finds no
functional unit is counted on its own line
(`src/cpu/o3/inst_queue.cc:1009-1014`, `FUPool::NoFreeFU` into
`statFuBusy` and `fuBusy`, the stats declared at `:306-316`). Bombin et
al. arXiv:2303.04846 names the first wait as the decoder's data
dependency: a decoder unit "needs to wait for said outcomes to be
available before it can begin solving its task" (lines 932-934), and
modules "lay idle waiting for other modules to complete their tasks
which are needed for input boundary conditions" (lines 1346-1349). Ciw's
per-customer record keeps `waiting_time` apart from `service_time` and
from `time_blocked` (`ciw/data_record.py:3-21`), so the whole
pre-service wait is a sum of named parts and never one number.

**Where to see it.** `POINTS` and `window_points_us` in
`decsim/experiments/measure.py`, the ready tick stamped in
`decsim/decoders/decode_service.py` (`mark_startable`, and the landing
itself when no boundary is owed), carried on `decsim/records/decoding.py`
and on the stage record in `decsim/decoders/staged_decoder.py`, the two
rows in [The run folder](../reference/run_folder.md), and
`tests/experiments/test_measure.py`, which holds the one-unit run where the
whole park is dependency and the forced-class pair where the two trade
places.

## D16. The confidence step is a point of the window's path

**Decided.** A new latency point, `confidence`: the committing decode's
end to the verdict on the window's answer, for a window whose weak
result committed. It is the sibling forced-class solve's remaining time
under `complementary_gap`, the walk under `cluster_gap`, and zero under
`weak_baseline`, whose verdict needs no signal. For a window that
escalated it is zero too: its committing decode is the strong one,
which answers after the verdict, and `weak_attempt` already runs from
the window's first dispatch to that verdict. The chain identity closes
on every window of every config this repository ships, and
`run_both_at_once` is the only run outside the sum. `chain_load` counts
the step as the unit's occupancy, because it is the unit's time.

**Why.** A span that is on the reaction time and in no column makes a
real cost invisible in exactly the runs it is largest in: on
`two_tiers.yaml` it is the second forced solve, and on a `cluster_gap`
run with a priced walk it is that walk on every window the weak tier
answered. A reader who sums the columns of such a window finds less
than its reaction time and no column to blame. The alternative, folding
the step into `service`, would make service stop meaning the decode's
own compute. The name is the tree's own word: the `ConfidenceSignal`
port, the `escalation.confidence` key, D3 and D8 all call this the
confidence.

**Sources.** Toshio et al. arXiv:2510.25222 makes the signal part of
the weak decoder's per-window work: the weak decoder "simultaneously
generates a soft output g for each decoding window" (lines 657-663),
the device computes it on the weak decoder (Fig. 1 caption, lines
152-158), and the Response Time the whole sum must reach is "the total
time elapsed from the generation of the final syndrome to the
application of the correction" (lines 360-363). In the tree the step is
already a component with an owner: D3 puts the signal behind the
`ConfidenceSignal` port and D8 charges its computation on the weak unit
that produced the evidence, which is why the same span is the unit's
occupancy in `load`.

**Where to see it.** `POINTS`, `window_points_us` and `chain_load` in
`decsim/experiments/measure.py`, the row in
[The run folder](../reference/run_folder.md),
`configs/cluster_gap_switching.yaml`, and the four shipped-config
identity tests in `tests/experiments/test_measure.py`.

## D17. The Union-Find growth, forest and peeling run in C

**Decided.** The one decoder decsim implements itself decodes in C.
`decsim/decoders/union_find/union_find.c` grows the clusters, takes the
minimum weight contact forest and peels it; the Python beside it builds
the graph of a placed model, turns a syndrome into a residual one, and
turns the outcome back into the evidence record a confidence signal
reads. The two are bound by ctypes, the library is a build artifact
that `tools/build_union_find.sh` makes and the suite makes when it is
missing, and the C must make the same decisions the Python made, not
merely correct ones: the same selected faults, the same intervals, the
same contacts in the same order, the same forest, the same unmatched
detectors.

**Why.** decsim's timing comes from the latency card and never from how
long a decoder runs, so a faster decoder must move the bill and no
result. The bill is the reason: the experiments in
`configs/experiments_2026_09` run union find at distances up to 15,
where
the Python row cost 48 seconds a shot. Identity is held by a property
test rather than by review: the Python growth, forest and peeling live
on as the oracle at `tests/decoders/union_find_oracle.py`, and
`tests/decoders/test_union_find_compiled_decoder.py` puts the two side
by side on Stim's rotated surface code circuits and on random graphs
that carry the shapes a surface code never makes.

Every other decoder row is an adapter over an installed package, so
this is the only row where the algorithm is decsim's to write, and the
only row where a compiled artifact enters the tree. The cost of that is
a build step; the row says so in one sentence when its library is
missing, and names the command.

**Sources.** Delfosse and Nickerson arXiv:1709.06218 give the two
algorithms and the data structure the growth uses: a cluster keeps a
list of its boundary, and "To grow a cluster, we must then simply
iterate over this list and grow the incident edges" (lines 506-507),
fusion appends one list to the other (lines 528-531), and a last pass
removes what is no longer on the boundary (lines 537-539). Huang,
Newman and Brown arXiv:2004.04693 give the weighted growth that this
row implements and iterate over the same boundary edges: "we first
iterate over the boundary edges to identify the smallest boundary edge
weight wmin, and then again iterate over the boundary edges to grow the
radius of the cluster by wmin" (lines 88-94). The C follows the LLVM
Coding Standards in the points `STYLE.md` lists under rule 9, which is
where this tree's rule for C lives. The naming is the exception and is
deliberate: the file is `snake_case`
like the Python beside it, not LLVM's capitalization.

**Where to see it.** `decsim/decoders/union_find/union_find.c` and its
header, `decsim/decoders/union_find/compiled_decoder.py`,
`tools/build_union_find.sh`, `tests/conftest.py`, and
`tests/decoders/test_union_find_compiled_decoder.py`, whose corpus test
is the identity claim.

## D18. The cluster gap's walk runs in C beside the growth it reads

**Decided.** The walk that turns one Union-Find growth into a gap is
`decsim/decoders/union_find/cluster_gap.c`, in the same library as the
decoder and reached through the same binding.
`decsim/confidence/cluster.py` keeps the signal row: what the gap is
defined on, the two refusals, the reading back into natural-log weight,
and what the step costs the run. The C returns the same half ticks the
Python returned for every growth, and that Python walk lives on as the
oracle at `tests/confidence/cluster_gap_oracle.py`.

**Why.** The walk was the larger part of a switching shot: about three
quarters of one at distance nine, one search from every node of the
quotient graph, where the whole union find row cost 13.7 seconds a shot
at distance 15. Nothing about the value changes: the gap is a minimum over
the quotient graph's nodes of a doubled-state Dijkstra distance, so
neither the order the nodes are visited in nor the cutoff each search
carries can move it, and a switching run makes the same decisions on
the same shots.

**What the measured time means.** A run that declares
`escalation.confidence_walk_microseconds` charges that number and is
untouched by this. A run that leaves it null charges what the walk cost
on the host clock, the way a decoder with no latency card is charged,
so its confidence term is what the C walk costs on that host, and
every span that waits on the confidence step carries it. That number is
the host's, which is what open issue O1 records for a real decoder; it
is not a hardware estimate.

**The uses order.** `decsim/confidence` imports
`decsim/decoders/union_find/compiled_decoder.py`, so it sits at level 4
rather than level 3. The relation stays a partial order, since no
decoder module imports the confidence package, and the cluster gap is
defined on a Union-Find growth alone, so the dependency is the one the
signal already had in prose.

**Sources.** Meister et al. arXiv:2405.07433 Definition 9 quotients the
decoding graph by the grown clusters and takes "the length of the
shortest path that covers a logical operator" (`2405.07433.txt` lines
518-521); Algorithm 2 line 2 is the search, "Run Dijkstra's algorithm
on G'_D" (lines 531-532), "with runtime O(|ED| +|VD| log|VD|)" (line
536). The C runs one such search per node over the parity-doubled node
set, with a binary heap, an array of distances two states wide, and a
generation stamp in place of a pass over every state between sources.

**Where to see it.** `decsim/decoders/union_find/cluster_gap.c` and its
header, `cluster_gap` and `cluster_gap_entry_point` in
`decsim/decoders/union_find/compiled_decoder.py`, `_cluster_gap` in
`decsim/confidence/cluster.py`, and
`tests/confidence/test_compiled_cluster_gap.py`, whose corpus test is
the identity claim.

## D19. A thing is named for what it is, and the name is experiment

**Decided.** The package that holds the yaml experiment, the sweep, the
collected rows, the figures, the trace viewer and the `decsim` command
is `decsim/experiments`, its tests are `tests/experiments`, the sixteen
decoder runs of 2026-09 are `configs/experiments_2026_09`, and the
Slurm array script is `slurm/experiment_run.sh`. Nothing inside any of
them moved: every module, class and function keeps its name, every yaml
key and every number is what it was, and a run charges exactly what it
charged before.

**Why.** A name should say what the thing is for. `front` said only
where the package sat in the uses order, and it collided with
`decsim/frontends`, the program readers and the planner, which is a
different thing at a different level. The 2026-09 folder and its Slurm
script carried a second word for what the tree already calls an
experiment, one yaml and the shards it is cut into, and two words for
one thing make a reader ask what the difference is when there is none.

**Where to see it.** `decsim/experiments/`, `tests/experiments/`, level
9 of the uses order in `decsim/machine.py`, the generated
[The module map](../reference/map.md),
`configs/experiments_2026_09/PLAN.md` and `slurm/experiment_run.sh`.

## What is not modelled yet

These are open, recorded rather than hidden, so that a reader does not
mistake a gap for a result.

- **O1. A switching run on two real decoders runs on the host wall clock.**
  Such a run prices both decoders from the measured clock, at weak 12
  to 119 microseconds and strong 0.94 to 46 milliseconds against a one
  microsecond round, which is the whole source of order and queue-depth
  variance between two runs of one seed. The proposal on the table is a
  latency key on the decoder section, with the two points priced at
  Toshio's generation time and ten times it.
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
  builds and runs as a machine, but the experiments layer's per-shot
  measurement compares the loop's prediction against PyMatching on the
  sampled shot, and a timing-only device samples none, so `collect`
  raises `KeyError`. A priced card on a real device is the way to a
  host-independent run today.
- **O9. A decoder section ignores a key it does not know.** The
  `weak_decoder` and `strong_decoder` sections do not yet refuse an
  unknown key by name the way `decoder_manager` does, so a misspelt key
  there runs the default in silence.

O3, O4, O5 and O10 are closed: the sends name what they carry
(`QPUReadout.size_bits` on the readout hop, nothing on the escalation
hop), the backward hand-off of the parallel scheme is priced and tested,
the boundary fold is written by the decoder side from the gate's mask
(`decsim/decoders/decoder_memory_transfer.py`, D11), and a timing-only
round ends in the decoders' own end for it
(`decsim/decoders/memory_rounds.py`).

## Read next

- [The principles behind the shape](principles.md): the ideas these
  decisions were made under.
- [The data path, hop by hop](data_path.md): D1, D5 and D9 as they
  appear on the wire.
- `STYLE.md` rule 8: why a modelling question is answered from a source.
