# Syndrome data-path refactor target

Status: implementation target under validation. This document distinguishes decsim policy from implementation evidence; see `evidence_catalog.md`.

## Stable ownership boundaries

```text
Controller --command--> QPUDevice
QPUDevice --QPUReadout (raw measurement fragments)--> Controller --raw fragments--> SyndromeIngress
SyndromeIngress --complete round, formed into detection events--> SyndromeBuffer
SyndromeBuffer --retained-round availability--> WindowManager
WindowManager --window input requirement--> DecoderManager
DecoderManager --transfer request--> DecoderInputTransfer
DecoderInputTransfer --delivered request--> DecoderInputStoreStager
DecoderInputStoreStager --materialized input--> DecoderInputStore
DecoderManager --ready DecodeJob--> Scheduler/Decoder
```

- `SyndromeBuffer` owns exactly one modeled upstream allocation per live round, fragment assembly, packing state, upstream capacity, immutable retained packets, typed consumer holds, and last-consumer release.
- `WindowManager` owns window bounds, readiness, dependencies, boundary transformations, and finality. It does not own payload storage, transfer scheduling, or decoder-input store allocations.
- `DecoderManager` owns request admission, transfer coordination, post-transfer ready queues, service, cancellation, and completion.
- `DecoderInputTransfer` owns mechanism-specific delay and cancellation only. The generic contract is not named DMA.
- `DecoderInputStoreStager`, owned by `DecoderManager`, is the storage-admission boundary after every transfer: round demand, admission, materialization, upstream-hold release, waiting, and credit return.
- `DecoderInputStore` owns one pool's round credits and its materialized immutable inputs. It is not a second scheduler.

## Baseline

The default is weak-only, one weak decoder, FIFO, unbounded upstream capacity, one shared unbounded decoder-input store, and explicit fixed-latency delivery with materialization on arrival. Strong/replay tokens are absent unless the selected strategy needs them.

## Core invariants

1. A round is removed from fragment admission before retained-ready publication.
2. `ASSEMBLING -> PACKING -> PACKED_RETAINED` does not charge an unsupported staging-to-retention copy.
3. A decoder cannot observe payload bytes before decoder-input transfer completion.
4. Every overlapping consumer owns a distinct hold; upstream release follows the last required transfer completion or cancellation.
5. Decoder-local input is a distinct modeled allocation/container.
6. Request admission, data readiness, queue readiness, dispatch, and completion are distinct timestamps.
7. Strong/replay retention is a typed owner token on the same upstream allocation, not a physical SB1.
8. Mechanism-specific streaming, ring, DMA, and instruction profiles remain replaceable implementations.
9. Detection events are formed once per complete round at Buffer 0 intake from the
   raw packet and the circuit's formation table (records, reference parity, layer
   kind); QC and C2B carry raw measurement bits, everything from Buffer 0 onward
   carries detection events.

## Deletions

The integrated target removes `PayloadStore`, `EndpointRole.SB0`, `EndpointRole.SB1`, endpoint ledgers, pair reservations, `complete_input_transfer`, direct shared `DecodeJob.payloads`, and `FixedDelayDirectValueTransport`. It retains scientifically meaningful patch/fragment identity and boundary transformation semantics.
