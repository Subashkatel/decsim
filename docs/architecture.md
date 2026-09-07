# Architecture

decsim is one root and a row of components, in the order a readout
travels. Each component depends on the port of the next, never on its
class, so a component can be replaced without any other one knowing.
`decsim/ports.py` is that list of ports, top to bottom in pipeline order,
with the record that crosses each; `decsim/machine.py` builds them in the
same order and wires them the way gem5's config script assigns ports.

## The readout's path

```mermaid
graph TD
    QPU -->|"ReadoutReceiver.accept_qpu_readout"| Controller
    Controller -->|"RoundStore.accept_packed_round"| Buffer0["Syndrome buffer 0"]
    Controller -->|"StrongRoundStore.write"| Buffer1["Syndrome buffer 1"]
    Buffer0 -->|"WindowInput.accept_window_input"| Windows["Window manager"]
    Windows -->|"DecodeQueue.enqueue"| Decoders["Decoder manager"]
    Decoders -->|"Decoder.start"| Unit["Decoder unit"]
    Decoders -->|"WindowInputGate.may_start"| Windows
    Decoders -->|"EscalationPolicy.verdict_for_weak_result"| Escalation
    Windows -->|"Frame.commit_correction"| Frame["Pauli frame"]
    Windows -->|"ReleaseReceiver.release_waiters"| Release["Conditional release"]
    Release -->|"InstructionReceiver.relay_instruction"| Controller
    Controller -->|"Qpu.issue"| QPU
```

Each arrow is one port method, and the component at its tail knows the
one at its head only through that port. Buffer 1 and the decoder unit
answer the decoder manager: it stages a window's rounds out of a buffer
into a unit's input slot, and starts the unit when the window's own gate
allows it.

Every arrow between two components also rides a link card
(`Link.send`, `decsim/links/`), which charges the hop its serialization
and propagation and books it under one path name. The path names are the
`LinkPath` values in `decsim/records/transfers.py`; the traffic ledger
reports one row per path.

## What each component owns

| Component | Package | It is handed | It hands on |
| --- | --- | --- | --- |
| QPU device | `qpu/` | one operation body per patch | one readout per round, on the cycle boundary |
| Controller | `controller/` | readouts | one packed round per round, with its detection events formed |
| Syndrome buffer 0 | `syndrome_buffer/` | packed rounds | the rounds a window reads, kept until every hold releases |
| Syndrome buffer 1 | `syndrome_buffer/` | the same packed rounds, in parallel | the rounds a strong window reads |
| Window manager | `windows/` | published rounds | one decode job per closed window, and each window's boundary to the next |
| Detector error model | `detector_error_model/` | the circuit's whole-circuit model | one fault model per window |
| Decoder manager | `decoders/` | decode jobs | one result per request, after its input landed and its unit computed |
| Decoder unit | `decoders/` | one input per slot | the correction its backend found, priced at the unit's clock |
| Escalation | `escalation/` | a weak result and its confidence | the verdict: keep it, or redecode the region on the strong tier |
| Pauli frame | `pauli_frame/` | one correction per window | the folded frame per stream, and the release of what waited |
| Observation | `observe/` | callbacks every component fires | the metrics, the traffic ledger, the trace |

The syndrome source, the round store, the decoder, the escalation
policy, the confidence signal, the windowing scheme, the idle policy
and the workload are the pluggable parts a table picks: each has its
abstract class in `decsim/ports.py` and one table of rows at the top
of `decsim/machine.py`. Two parts are pluggable through their own
settings section instead. The link fabric is built from the `links`
section, a number card read by `link_profiles.from_yaml` and wired by
`fabric.LinkFabric`, so a card of your own is a card file and not a
row. The threshold source is `escalation.threshold_source`: fixed and
table both reach the root as a threshold in nats, resolved per sweep
point by the front, and online reaches it as the calibrator object
itself (`_threshold_source`).

## The hops, one row each

A hop is one pair of neighbours and one link path. The traffic ledger
books every transfer as one of three words
(`decsim/observe/data_movement.py`): a move leaves the bits behind, a
copy ends with both sides holding them, and a reference hands over an
object both sides read, which is what a store's hold books rather
than a hop of its own.

| Hop | Path | Bits |
| --- | --- | --- |
| Readout electronics to controller | `qpu_to_controller` | move, one bit per measure qubit per round, and on the last round the data readout too, one bit per data qubit |
| Controller to syndrome buffer 0 | `controller_to_weak_buffer` | move, the packed round |
| Controller to syndrome buffer 1 | `controller_to_strong_buffer` | copy, the same round in parallel |
| Buffer 0 to a weak unit's input slot | `weak_buffer_to_weak_decoder` | copy, the window's rounds |
| Weak decoder to strong decoder | `weak_decoder_to_strong_decoder` | the selection only, no payload |
| Buffer 1 to the strong decoder | `strong_buffer_to_strong_decoder` | copy, the strong region at once |
| Decoder to decoder | `decoder_to_decoder` | the committed window boundary |
| Weak or strong decoder to the frame | `weak_decoder_to_frame`, `strong_decoder_to_frame` | one bit per logical observable |
| Frame to controller | `frame_to_controller` | the release, no payload |
| Controller to the QPU | `controller_to_qpu` | the instruction, no payload |

The hardware each hop stands for is named in the module that models it:
readout over a low-latency link and syndrome formation at the controller
(Google, arXiv:2408.13687), the streamed buffer the decoder reads, the
strong decoder's stored region and its two-sided boundary condition
(Toshio, arXiv:2510.25222), and the conditional pulse back to the QPU
(Yang, arXiv:2605.04892).

## Time

One `Engine` (`decsim/engine.py`) orders every action by integer ticks;
one microsecond is a million ticks. A component never sleeps and never
polls: it schedules the action that follows the cost it just charged.
The QPU is the one clocked component, and every round lands on a cycle
boundary of its clock.

## Read next

- `decsim/ports.py`: the ports in pipeline order, one method per handoff.
- `decsim/machine.py`: the build, in that same order, and the tables.
- `docs/plug_in_a_component.md`: how to fill one of those ports yourself.
