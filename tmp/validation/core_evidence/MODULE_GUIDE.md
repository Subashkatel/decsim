# Phase A core module guide

This guide answers two questions:

1. Why does each module exist?
2. Where should a developer make a change?

## The normal data flow

```
Controller --command--> QPUDevice
QPUDevice --QC/readout--> Controller --binary availability--> SyndromeIngress
SyndromeIngress -> SyndromeBuffer -> WindowManager
                                           |
                                  DecoderInputTransfer
                                           |
                                   DecoderLocalMemory
                                           |
                                     DecoderManager
                                           |
                                        decoder
```

`RunSpec` creates these objects and connects them. It does not own their live
state.

## Module responsibilities

### `controller.py`

**Keeps:** operation sequencing, feedback decisions, idle-round policy, QPU
commands, and the timed availability of validated binary readout data.

**Does not keep:** retained syndrome data, decoder queues, analog waveforms, or
classifier internals.

Change this file when you change controller decisions or command sequencing.
Do not add decoder scheduling or syndrome storage here.

### `execution_runtime.py`

**Keeps:** which operations are admitted, resource claims, dependency readiness,
and execution timestamps.

It is separate from the controller to keep program-DAG/resource state out of
stream-cadence code. It is not currently an injected part: the two objects call
each other directly. If that coupling grows, merging them into one sequencing
owner is preferable to adding another interface.

Change this file when you change program-DAG admission or resource ownership.

### `qpu.py`

**Keeps:** the physical round clock for issued commands. It is the only object
that asks the configured device to produce operation-body QPU readout events.

Change this file when you change how an issued QPU command produces rounds.

### `syndrome_ingress.py`

**Keeps:** controller-side QC receipt, fragment reassembly timing, route arbitration, packing
delay, timeout, and finite-ingress overflow behavior.

It does not own retained syndrome lifetime. That belongs to `SyndromeBuffer`.

Change this file for transport timing, reassembly, routing, or ingress policy.

### `syndrome_buffer.py`

**Keeps:** the one upstream allocation per round, its state, and typed consumer
holds.

It deliberately has no `Engine`, link, decoder queue, DMA, or placement state.

Change this file for upstream capacity, retention, or hold lifetime rules.

### `window_manager.py`

**Keeps:** window readiness, boundaries, dependencies, finality, and which
syndrome rounds each decode request needs.

It does not own physical payload storage or decoder service queues.

Change this file for window algorithms, strong/replay dependencies, or boundary
behavior. This is still the largest runtime module and the main remaining
complexity hotspot.

### `decoder_input.py`

**Keeps:** the replaceable action that turns an admitted request into ready,
decoder-local input after a modeled delay.

The default implementation is intentionally small. A DMA or streaming model
would be a separate implementation of the same protocol.

Change this file to add or select a decoder-input transfer profile.

### `decoder_local.py`

**Keeps:** immutable input values owned by the decoder side and their local
allocation lifetime.

It has no scheduler. `DecoderManager` remains the only ready-queue owner.

Change this file for decoder-local capacity or materialized input format.

### `decoder_manager.py`

**Keeps:** request admission, ready queues, service units, dispatch,
cancellation, and terminal request/service records.

Change this file for decoder scheduling or service lifecycle. Do not put
window readiness or upstream retention here.

### `run_spec.py`

**Keeps:** construction only. It chooses implementations, wires them together,
runs the engine, and returns `CompletedRun`.

Change this file only when adding or removing a supported construction seam.

## Comment and docstring style

Core source comments should help a reader change behavior:

- Say what state this code owns or why an ordering check is required.
- Use one short sentence when the code already shows the mechanics.
- Define an acronym before using it in prose.
- Do not put paper titles, evidence IDs, audit paths, or implementation history
  in Python files. Put that material in `README.md` beside this guide.
- Do not describe deleted classes or compatibility behavior in current source.
- A module docstring should answer “what does this file own?” and “what does it
  not own?” in a few lines.

## Common changes

| Goal | Start here | Usually also test |
|---|---|---|
| Change QPU round production | `qpu.py`, configured device | `test_qpu.py`, `test_device_wire_sizes.py` |
| Change QC/reassembly timing | `syndrome_ingress.py` | `test_syndrome_ingress.py` |
| Change upstream capacity/retention | `syndrome_buffer.py` | `test_syndrome_buffer.py`, `test_syndrome_buffer_holds.py` |
| Change window overlap/readiness | `window_manager.py`, scheme | `test_kernel_window_manager.py`, switching tests |
| Add DMA/ring/streaming input | new `DecoderInputTransfer` implementation | `test_decoder_input_transfer.py` |
| Change decoder-local format/capacity | `decoder_local.py` | `test_decoder_local.py` |
| Change FIFO or decoder service | `decoder_manager.py`, scheduler | decoder-manager and queue tests |
| Change program dependencies/resources | `execution_runtime.py` | execution-order/runtime tests |

## Why these are separate modules

The split follows ownership, not hardware placement. Each mutable fact has one
owner:

- execution admission -> `ExecutionRuntime`;
- command/feedback sequencing -> `Controller`;
- physical round production -> `QPUDevice`;
- transport/reassembly timing -> `SyndromeIngress`;
- upstream lifetime -> `SyndromeBuffer`;
- window readiness -> `WindowManager`;
- transfer completion -> `DecoderInputTransfer`;
- decoder-local lifetime -> `DecoderLocalMemory`;
- queue/service lifetime -> `DecoderManager`.

The small `qpu.py` and `decoder_input.py` files are real replaceable or command
boundaries, not utility layers. `ExecutionRuntime` is a state-ownership split,
not a replaceable seam; it and `Controller` are tightly coupled and should be
merged rather than wrapped if that split stops making the sequencing easier to
follow. `decoder_local.py` stays separate because it defines the input value and
capacity model independently of a transfer mechanism. The module set should not
grow unless a new component owns different mutable state or has more than one
supported implementation.

## Remaining design concern

`window_manager.py` is much larger than the other core modules. Do not split it
by creating forwarding wrappers. A future split is justified only if a coherent
state owner can move out with a narrow API, for example the already-contained
strong/replay registry. Until then, keeping that state together is simpler than
adding more layers.
