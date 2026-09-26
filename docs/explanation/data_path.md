[decsim docs](../README.md) › [Explanation](README.md)

# The data path, hop by hop

A round of syndrome data leaves the fridge and comes back as an
instruction. This page follows it, one hop at a time: what crosses, how
many bits, what the transfer does to the bits, and where the hop's
numbers came from.

## Three words for what a transfer does

The traffic ledger (`decsim/observe/data_movement.py`) keeps three
accounts, and the vocabulary is gem5's.

- A **move** is bits leaving one structure and arriving in another over
  a link. Every one of the eleven link paths is a move, and only a move:
  `transfer_delivered` books every delivered transfer into the move
  account.
- A **copy** is bits duplicated into a structure the receiver owns. A
  copy is booked where it lands, by the name of the structure it landed
  in, not by the path that brought it. A hop can therefore be a move
  whose landing is also a copy, and most of them are.
- A **reference** is a handle to bits that stay where they are. In
  decsim a reference is a **hold**: a token one consumer registers on a
  store's rounds so the store may not drop them. The rounds do not move
  and are not duplicated; what is paid for is the store capacity they
  occupy while the hold is live.

The distinction matters because the cost of moving data is a property of
the memory crossed rather than of the act of copying. A copy inside a
register file is nearly free; a copy that crosses a board is not. Each
path is therefore also given a **memory class**: on chip, on board, or
off board (`MEMORY_CLASS_BY_LINK_PATH`). A study that adds up bits adds
them up per class, never in one total.

## Every hop rides a link

Between two components a call is not just a call: it is a transfer with
a card. The card gives the path a propagation latency, optionally a
bandwidth in bits per cycle of a named clock, the channel it shares with
other paths, and a per-transfer setup cost. `LINK_FABRICS` has four rows:
`logical_reference`, the default, which charges each hop a latency plus
its bits over a rate, both from a system of decsim's scale;
`bandwidth_limited`, which gives the channels finite
rates provisioned from the code geometry; and `roce_v2_cpu` and
`roce_v2_gpu`, the default card with the strong tier's off-board hops
priced by a measured round trip ([D14](decisions.md#d14-the-strong-tiers-off-board-path-can-be-priced-by-a-measured-round-trip)).

Whoever executes a send is an end of that hop. `tests/test_send_ends.py`
holds the rule: its `ENDS_OF_PATH` table names each path's two ends as
packages, and the test walks `decsim/` to check that no other component
names that path.
[D1](decisions.md#d1-whoever-executes-a-send-is-an-end-of-that-hop) has
the referents it comes from.

## The eleven hops

The paths are the `LinkPath` values in `decsim/records/transfers.py`.
The default latencies below are the `logical_reference` card in
`decsim/links/link_profiles.py`; they are the card's numbers, and a
config that sets its own replaces them. The weak loop is Yang
arXiv:2605.04892 Table I hop for hop, a d=3 surface code whose readout
modules, decoder FPGA and pulse generators the table times one by one
at 250 MHz; the strong node is one inter-chassis hop away, Caune
arXiv:2410.05202 Fig. 1a stage F; the on-chip hops are one 250 MHz
cycle, a stated assumption, because no referent measures a wire inside
a chip. A bounded hop moves a 32-bit word a cycle on the decoder's bus
and on chip (Caune lines 998-1008), a 128-bit instruction word a cycle
into the pulse generator (QubiC arXiv:2404.15260), and 100 Gb/s off
board (Backline arXiv:2609.09270 lines 1229-1230). Hops 1 and 7 are
unbounded, because every qubit has its own readout channel and every
graph edge its own link.

### 1. `qpu_to_controller`

The readout electronics to the control workstation. Ends: `qpu` to
`controller`; the send is executed in `decsim/controller/controller.py`,
by the controller's intake.

What crosses: one acquisition fragment of classified measurements for a round.
`patch_ids` names its contributing patch group; `fragment_index` preserves
measurement order when a source emits several acquisitions.
Python fabric cards can set `readout_routes` to select a `PathSettings` by
that complete footprint. Unmatched groups use the default `qpu_to_controller`
card. Equal channel names share one wire; different names give separate queues.
These delays start when the physical round emits its outcomes, so they model
post-measurement transport. They are not independent analog acquisition times.
The controller announces round order before sending; the assembler waits for
prior emitted rounds in the same stream before forming or forwarding later
ones. Each arrived round occupies packing capacity through detector formation.
The count is `readout.size_bits`, set where the device builds the
readout (`decsim/qpu/stim_device.py`, `decsim/qpu/syndrome_devices.py`).
For a distance 3 rotated surface code that is 8 bits per round, the
`d*d - 1` stabilizers of the patch (`decsim/qpu/code_geometry.py`,
`syndrome_bits_per_round`).

A round's width is not one number over a whole stream. The last round of
a memory experiment also reads the patch's `d*d` data qubits, so it
carries `2*d*d - 1` bits: 17 at distance 3, 49 at distance 5. A hop that
carries detection events instead of raw outcomes sees `(d*d - 1)/2` of
them on the first round, `d*d - 1` in the bulk and `3*(d*d - 1)/2` on
the last, since the first round has no round before it to difference
against and the last closes on the data readout
(`tests/links/test_data_through.py` derives each count from the circuit).
A bounded link therefore serializes the last round longer than the rest.

Move, off board, and the landing is also a copy into the controller's
intake register. Default latency 48 nanoseconds, unbounded: Yang's ADC
chip 12, IQ demodulation 32 and classification 4 nanoseconds, counted
from the end of the acquisition window, which the round period holds.
The default card therefore includes turning the signal into bits; a
config's own card times the wire alone and prices the classification
separately
(`readout_to_bits_cycles`, whose
sources are in `decsim/controller/controller.py`: 40 nanoseconds of
in-FPGA discrimination, Fermilab arXiv:2406.18807, and 20 nanoseconds to
a syndrome, Yang arXiv:2605.04892).

### 2. `controller_to_weak_buffer`

The packed round into the weak syndrome buffer. Ends: `controller` to
`syndrome_buffer`; the send is executed by the controller's transmitter
(`decsim/controller/round_transmission.py`), and the landing is handled
by the weak syndrome buffer's own incoming port
(`decsim/syndrome_buffer/weak_syndrome_round_receiver.py`), which stores the round with
the landing tick as its publication tick, narrates the copy and the
intake, and announces the published round to the window manager. The
same end decides whether the buffer has space. It counts the bits
already stored and the bits reserved for the writes still crossing the
link, and the controller's sender reserves the round's bits before it
leaves. The transmitter is told when the round lands, and only so that
it can keep its own count of the rounds in flight.

What crosses: one **packed round**, every fragment that leaves the
controller. The bit count is `PackedRound.wire_bits`, computed in
`decsim/controller/round_assembly.py`. Under the `controller` row of
`DETECTION_EVENT_FORMATION` that width is the detection events, which is
narrower than the raw outcomes; under the `decoder` row it is the raw
outcomes and each tier forms its own events.

Move, on board, with a copy into the weak syndrome buffer's record at the landing. The
round occupies a slot when its bits are in the store, and it is readable
at that same instant; the room it will need is reserved before the wire
is used, so the store can still refuse a round before it leaves the
controller. Default latency 18 nanoseconds and a 32-bit word a cycle:
half of the 36 nanoseconds of digital communication in Yang's Table I,
which cover the loop's two links (readout module to decoder FPGA, and
decoder FPGA to the pulse side) and give no split. The weak syndrome
buffer sits on the decoder chip, so this is the first of the two.

### 3. `controller_to_strong_buffer`

A strong-only run's packed round into the strong syndrome buffer, its
one transport. Ends: `controller` to `syndrome_buffer`; the send is
executed by the controller's round sender
(`decsim/controller/syndrome_round_sender.py`), and the landing is
handled by the room side
(`decsim/syndrome_buffer/strong_syndrome_round_receiver.py`), which
reserves the room the round will take before it leaves and then stores
it with the landing tick, or drops it at the door when its operation
closed while it crossed. A switching run sends nothing on this hop: its
rounds stay in the weak syndrome buffer, and the strong side gets the
escalated window's rounds on hop 5.

What crosses: the same round, the same bit count.

Move, off board, with a copy into the strong syndrome buffer at the landing. Default
latency 0.26 microseconds and 100 Gb/s, from Caune Fig. 1a stage F, the inter-node
broadcast between control system chassis, 240 to 260 nanoseconds at the
stated worst case. It is off board because the strong tier is a separate
machine, room side in this tree, which is given its assigned data:
"we assign the syndrome data of rstrong rounds, which includes the
region with the small soft output, to the strong decoder" (Toshio
arXiv:2510.25222, line 1248 of the text extraction the code's
docstrings cite).

### 4. `weak_buffer_to_weak_decoder`

A window's rounds into a weak unit's memory. Ends: `syndrome_buffer` to
`decoders`; the send is executed by the store's own outgoing port,
`decsim/syndrome_buffer/round_output.py`.

What crosses: the whole input of one decode job, every payload round of
the window at once. The bit count is `job.payload_bits()`
(`decsim/records/decoding.py`). The same path also carries a
timing-only feedback-memory round. That is an idle patch's round: it
carries no syndrome a decoder reads, and it travels so that the buffer's
slot, the link and the decoder's stream stage are charged for it. It is
sent by the weak syndrome buffer on the
controller's ask and lands at the decoders' own end for it
(`decsim/decoders/memory_rounds.py`): nothing is deposited in a unit's
memory, and what the end does is count the round its stream stage was
handed and tell the window side.

Move, on board, with a copy into the unit's own memory at the landing.
That is the `copy` row of the tier's `input` key, and it is the default,
because a hardware decoder loads the syndrome into its storage elements
before it decodes, which `decsim/decoders/settings.py` cites Collision
Clustering's Init unit for (arXiv:2309.05558, `2309.05558.txt` lines
268-272: the Init unit "loads the input syndrome data and appropriate
data into the storage elements"). The other row, `in_place`, sends
nothing at all and books a reference instead: the unit reads the rounds
where they sit, which is what a decoder with its input on chip does,
and the same comment cites AFS for it (arXiv:2001.06598,
`2001.06598.txt` lines 520-535: "the processing elements can directly
access the data stored on-chip"). Default latency one 250 MHz cycle and
a 32-bit word a cycle, a stated assumption: Yang keep the syndrome in
registers, fully pipelined (lines 1273-1275), and a compiled on-chip
memory moves one 32-bit word per access.

### 5. `weak_decoder_to_strong_decoder`

An escalation. Ends: `decoders` to `syndrome_buffer`; the send is
executed by `decsim/decoders/decoder_output.py`, asked for by
`decsim/escalation/strong_redecode.py`, and the region's landing is
handled by the room side
(`decsim/syndrome_buffer/strong_syndrome_round_receiver.py`).

What crosses: first a selection, which window escalates and nothing
else, with no payload; then the strong window's rounds, read out of the
weak syndrome buffer, `r_com + 2 r_buf` of them under Toshio's
assumption less any the strong side already has, at the width each
round left the controller (`EscalatedRegion.wire_bits`). A window whose
rounds are all measured at the verdict carries them with the selection;
a forward window carries them at the far commit, when its extent is
known, and a terminal one, a window at the end of the stream whose last
rounds are still being measured, carries the rest as they arrive (Toshio
arXiv:2510.25222 lines 1247 to 1250: the syndrome data of `r_strong`
rounds is assigned to the strong decoder at the switch, after both
boundaries are determined).

Move, off board, with a copy into the strong syndrome buffer at the
landing. Default latency 0.26 microseconds and 100 Gb/s: the strong
node is one inter-chassis hop from the weak chip, Caune Fig. 1a stage F,
the same kind of hop as hop 3.

### 6. `strong_buffer_to_strong_decoder`

The strong region into the strong unit. Ends: `syndrome_buffer` to
`decoders`; the same store output port as hop 4, bound to this path for
the strong syndrome buffer (`decsim/build/stores.py`).

What crosses: the strong window's assigned rounds, `r_com + 2 r_buf` of
them under Toshio's assumption, in one transfer. The bit count is again
`job.payload_bits()`.

Under strong-only, this path also carries each timing-only feedback-memory
round after it lands in the strong store. The round occupies its slot until
delivery to `MemoryRoundArrivals`, exactly as hop 4 does for a weak-primary run.
These timing-only rounds are counted there and occupy no decoder-unit input
memory. No weak-buffer hop or escalation is involved.

Window input moves on board, with the same copy into the unit's memory. Default
latency one 250 MHz cycle and a 32-bit word a cycle, as hop 4. It is on board rather than off board because
the rounds already crossed boards into the strong syndrome buffer, at
hop 3 or hop 5: the strong syndrome buffer sits beside the strong
decoder.

### 7. `decoder_to_decoder`

One committed window's boundary to the next window. Ends: both
`windows`; the send is executed by the boundary courier
(`decsim/windows/window_boundaries.py`), which holds the record that
leaves, and the landing is handled by the same component, which owns the
destination window's record. The card still prices the wire between two
decoders; what the ends name is which package holds the objects at them
([The design decisions](decisions.md) D12).

What crosses: the residual defects on the seam layer the two windows
share, which are the detection events the committed correction leaves
unexplained at the window's edge (Skoric's artificial defects). Under the default `dense_seam_mask` row of `BOUNDARY_PAYLOADS`
the cost is the seam layer's whole detector count, `d*d - 1`, because
both compiled implementations carry a mask whatever the noise did. Under
`sparse_seam_list` it is the flipped detectors and their index width,
which is how Skoric's blocks exchange the artificial defects themselves
(arXiv:2209.08552, `2209.08552.txt` lines 269-281, where they are named
and their creation described) and how
`decsim/windows/boundary_payloads.py` cites Bombin arXiv:2303.04846 for
bounding the update to a small number of check generators.

Move, on chip. Default latency one 250 MHz cycle, unbounded: Helios
gives every graph edge its own link in one clock domain
(arXiv:2301.08419 lines 764-769 and 801), so the seam's detectors cross
in parallel. It is on chip because
one decoder's boundary reaches another through shared memory the
processing elements write, which
`decsim/observe/data_movement.py` cites Helios arXiv:2301.08419 for.

### 8 and 9. `weak_decoder_to_frame`, `strong_decoder_to_frame`

A correction to the Pauli frame. Ends: `decoders` to `pauli_frame`; the
send is executed by `decsim/decoders/decoder_output.py`, which picks the
path by the tier from `FRAME_PATH_BY_TIER`.

What crosses: one bit per logical observable
(`result_payload_bits`). That is the logical-frame convention: Caune
arXiv:2410.05202 returns one Boolean per decode and Google
arXiv:2408.13687 an observable bitmask per block. A decoder that fed a
physical frame instead would emit a per-qubit correction vector, which
is a different card.

Both are moves. The weak one is on board, because the frame is the
controller's; the strong one is off board, because the strong decoder is
not. Default latency 18 nanoseconds and a 32-bit word a cycle for the
weak one, the second half of Yang's digital communication, and 0.26
microseconds and 100 Gb/s for the strong one, Caune's stage F. The strong answer's way home
runs through the chip's window side before it leaves: the decoder
manager returns the result to the verdict (`accept_strong_result` in
`decsim/windows/window_commits.py`), the committer publishes it, and
the chip's decoder output executes the send; the one card prices the
whole way from the strong decoder to the frame. An escalated window's
weak answer never reaches the frame (the verdict escalates instead of
publishing, `_apply_verdict` in the same file), so the frame takes one
correction per window and no difference between the two answers is
formed.

### 10. `frame_to_controller`

The conditional release. Ends: `pauli_frame` to `controller`; the send
is executed by `decsim/pauli_frame/decision_dispatch.py`, which the
controller's `decsim/controller/conditional_release.py` asks once a
result is final, and the landing is
`decsim/controller/instruction_output.py`. The decision leaves by the
frame's port, so that end executes the send and narrates it, the way
gem5 bills a transfer to the port it left by (`packet.hh:424-431`).

What crosses: a decision, no data. The card charges one 32-bit control
bus word, the decoder sequencer's WISHBONE interface width (Caune
arXiv:2410.05202, Methods). Move, on chip, no propagation and a 32-bit
word a cycle: the frame sits in the controller and hands the core one
32-bit word with a ready signal (QubiC arXiv:2404.15260 lines 186-190),
so the hop costs that word's one cycle.

### 11. `controller_to_qpu`

The instruction back. Ends: `controller` to `qpu`; the send is executed
by `decsim/controller/instruction_output.py`.

What crosses: one instruction, no data payload. The card charges one
128-bit control-processor instruction word, which is QubiC's width
(Fruitwala arXiv:2404.15260, Sec. III and IV). The decision-to-pulse
cost is charged separately (`decision_to_pulse_cycles`): the control
processor's issue pipeline from the decision at the core to the pulse
trigger, 8 cycles traced on QubiC's core in `configs/reference.yaml`,
against QICK's measured 16 clocks for the conditional evaluation and
the jump and 20 for the next pulse (arXiv:2110.00557 lines 893-900).
Move, off board, default latency 88 nanoseconds and a 128-bit word a
cycle: Yang's trigger propagation to the pulse generator 16, waveform
generation 32 and DAC chip 40 nanoseconds (QICK measures a 45
nanosecond DAC). The coax down to the QPU is outside Yang's
measurement and no source gives it.

## Why the copies are where they are

decsim copies at almost every landing, and that is a modelled choice
rather than an implementation accident.

The reason is not the energy of the copy. It is that a reference has a
precondition a real system has to pay for: the data must not be mutated
while the reader still holds it. That is exactly what a hold is, and
what a hold costs is store capacity. A design that references instead of
copying trades bandwidth for buffer occupancy, and decsim can price both
sides of that trade because both are modelled.

The second reason is determinism. Shared memory between a controller and
a decoder introduces contention and unpredictable access delays, which
is what a real-time control loop cannot have. A copy into a structure
one component owns is predictable, and predictability is the property
the loop is being designed for.

Where a real system does read in place, decsim has a row for it: the
`in_place` value of a tier's `input` key on hops 4 and 6.

## Reading a real path

Every hop above appears in a trace, and `decsim trace follow` prints one
round's or one window's hops in tick order, with the transfer word and
the bit count on each line.
[How to read a trace and follow one round or one window](../how-to/read_a_trace.md) shows how.

## Read next

- [Glossary](../reference/glossary.md): the path names in one list.
- [Architecture](architecture.md): the components at the ends.
- [Windows and boundaries](windows_and_boundaries.md): what hop 7 carries, in
  detail.
