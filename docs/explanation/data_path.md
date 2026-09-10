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
other paths, and a per-transfer setup cost. `LINK_FABRICS` has two rows:
`logical_reference`, the default, which charges propagation only and
never queues, and `bandwidth_limited`, which gives the channels finite
rates provisioned from the code geometry.

Whoever executes a send is an end of that hop. `tests/test_send_ends.py`
holds the rule: its `ENDS_OF_PATH` table names each path's two ends as
packages, and the test walks `decsim/` to check that no other component
names that path. The rule is OMNeT++'s, which refuses at runtime a
module that sends a message it does not own
(`tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334`), and gem5's,
which bills a transfer to the port it left by rather than to whoever
arranged it (`packet.hh:424-431`).

## The eleven hops

The paths are the `LinkPath` values in `decsim/records/transfers.py`.
The default latencies below are the `logical_reference` card in
`decsim/links/link_profiles.py`; they are the card's numbers, and a
config that sets its own replaces them.

### 1. `qpu_to_controller`

The readout electronics to the control workstation. Ends: `qpu` to
`controller`; the send is executed in `decsim/controller/controller.py`,
by the controller's intake.

What crosses: one patch's classified measurement bits for one round.
The count is `readout.size_bits`, set where the device builds the
readout (`decsim/qpu/stim_device.py`, `decsim/qpu/syndrome_devices.py`).
For a distance 3 rotated surface code that is 8 bits per round, the
`d*d - 1` stabilizers of the patch (`decsim/qpu/code_geometry.py`,
`syndrome_bits_per_round`).

Move, off board, and the landing is also a copy into the controller's
intake register. Default latency 0.15 microseconds. The hop is modelled
on measurement signals classified into bits and then sent to a
workstation over a low-latency link, which is what
`decsim/observe/data_movement.py` cites Google arXiv:2408.13687 for,
with the classification itself priced separately
(`readout_to_bits_cycles`, whose
sources are in `decsim/controller/controller.py`: 40 nanoseconds of
in-FPGA discrimination, Fermilab arXiv:2406.18807, and 20 nanoseconds to
a syndrome, Yang arXiv:2605.04892).

### 2. `controller_to_weak_buffer`

The packed round into syndrome buffer 0. Ends: `controller` to
`syndrome_buffer`; the send is executed by the controller's transmitter
(`decsim/controller/round_transmission.py`).

What crosses: one **packed round**, every fragment that leaves the
controller. The bit count is `PackedRound.wire_bits`, computed in
`decsim/controller/round_assembly.py`. Under the `controller` row of
`DETECTION_EVENT_FORMATION` that width is the detection events, which is
narrower than the raw outcomes; under the `decoder` row it is the raw
outcomes and each tier forms its own events.

Move, on board, with a copy into Buffer 0 at the landing. Default
latency 0.04 microseconds, taken from Caune arXiv:2410.05202 Fig. 1a
stage D, "result message handled and prepared for broadcast", 40
nanoseconds. Buffer 0 sits with the controller, which is why that stage
is the right one.

### 3. `controller_to_strong_buffer`

The same packed round into syndrome buffer 1, in parallel. Ends:
`controller` to `syndrome_buffer`; the send is executed by
`decsim/syndrome_buffer/strong_round_writer.py`.

What crosses: the same round, the same bit count.

Move, off board, with a copy into Buffer 1 at the landing. Default
latency 0.26 microseconds, from Caune Fig. 1a stage F, the inter-node
broadcast between control system chassis, 240 to 260 nanoseconds at the
stated worst case. It is off board because the strong tier is a separate
machine, room side in this tree, which is given its assigned data:
"we assign the syndrome data of rstrong rounds, which includes the
region with the small soft output, to the strong decoder" (Toshio
arXiv:2510.25222, `2510.25222.txt` line 1261).

### 4. `weak_buffer_to_weak_decoder`

A window's rounds into a weak unit's memory. Ends: `syndrome_buffer` to
`decoders`; the send is executed by the store's own outgoing port,
`decsim/syndrome_buffer/round_output.py`.

What crosses: the whole input of one decode job, every payload round of
the window at once. The bit count is `job.payload_bits()`
(`decsim/records/decoding.py`). The same path also carries a
timing-only feedback-memory round.

Move, on board, with a copy into the unit's own memory at the landing.
That is the `copy` row of the tier's `input` key, and it is the default,
because a hardware decoder loads the syndrome into its storage elements
before it decodes, which `decsim/decoders/settings.py` cites Collision
Clustering's Init unit for (arXiv:2309.05558). The other row,
`in_place`, sends nothing at all and books a reference instead: the unit
reads the rounds where they sit, which is what a decoder with its input
on chip does, and the same comment cites AFS (arXiv:2001.06598) for it.
Neither paper has a text file in the sandbox, so those two citations are
the code's, carried here rather than re-checked. Default latency 2.0
microseconds.

### 5. `weak_decoder_to_strong_decoder`

An escalation. Ends: both `decoders`; the send is executed by
`decsim/decoders/decoder_output.py`, asked for by
`decsim/escalation/strong_redecode.py`.

What crosses: a selection, not data. Which window escalates, and nothing
else. On the default card the transfer carries no stated payload at all;
on the bounded card it is charged one bit.

Move, off board. Default latency 0.5 microseconds. This is the one card
number in the fabric that names no paper: its source string reads
"repository weak-to-strong model choice". No system read during the
design work had a weak-to-strong data handoff, because the strong tier
takes its data from the store rather than from the weak decoder.

### 6. `strong_buffer_to_strong_decoder`

The strong region into the strong unit. Ends: `syndrome_buffer` to
`decoders`; the same store output port as hop 4, bound to this path for
Buffer 1 (`decsim/build/stores.py`).

What crosses: the strong window's assigned rounds, `r_com + 2 r_buf` of
them under Toshio's assumption, in one transfer. The bit count is again
`job.payload_bits()`.

Move, on board, with the same copy into the unit's memory. Default
latency 2.0 microseconds. It is on board rather than off board because
the write into Buffer 1 already crossed boards at hop 3: Buffer 1 sits
beside the strong decoder.

### 7. `decoder_to_decoder`

One committed window's boundary to the next window. Ends: both
`decoders`; the send is executed by `decsim/decoders/decoder_output.py`,
asked for by `decsim/windows/window_boundaries.py`.

What crosses: the residual defects on the seam layer the two windows
share. Under the default `dense_seam_mask` row of `BOUNDARY_PAYLOADS`
the cost is the seam layer's whole detector count, `d*d - 1`, because
both compiled implementations carry a mask whatever the noise did. Under
`sparse_seam_list` it is the flipped detectors and their index width,
which is how Skoric's blocks exchange the artificial defects themselves
(arXiv:2209.08552, `2209.08552.txt` lines 272-284, where they are named
and their creation described) and how
`decsim/windows/boundary_payloads.py` cites Bombin arXiv:2303.04846 for
bounding the update to a small number of check generators.

Move, on chip. Default latency 0.5 microseconds. It is on chip because
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
not. Default latency 1.0 microseconds each.

### 10. `frame_to_controller`

The conditional release. Ends: `pauli_frame` to `controller`; the send
is executed by `decsim/controller/instruction_output.py`, asked for by
`decsim/controller/conditional_release.py`.

What crosses: a decision, no data. The card charges one 32-bit control
bus word, the decoder sequencer's WISHBONE interface width (Caune
arXiv:2410.05202, Methods). Move, on chip, default latency 4.0
microseconds.

### 11. `controller_to_qpu`

The instruction back. Ends: `controller` to `qpu`; the send is executed
by `decsim/controller/instruction_output.py`.

What crosses: one instruction, no data payload. The card charges one
128-bit control-processor instruction word, which is QubiC's width
(Fruitwala arXiv:2404.15260, Sec. III and IV). The decision-to-pulse
cost is charged separately (`decision_to_pulse_cycles`, which
`decsim/controller/instruction_output.py` sources to QICK's 42
nanosecond conditional jump and 52 nanosecond next pulse,
arXiv:2110.00557 Table II). Move, off board, default latency 0.15 microseconds.

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
`docs/how-to/read_a_trace.md` shows how.

## Read next

- `docs/reference/glossary.md`: the path names in one list.
- `docs/explanation/architecture.md`: the components at the ends.
- `docs/explanation/windows_and_boundaries.md`: what hop 7 carries, in
  detail.
