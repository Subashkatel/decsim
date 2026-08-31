# decsim architecture overview

decsim is a discrete-event co-design simulator for the classical control
path of quantum error correction: it measures full reaction time from
syndrome emission through decoding to conditional feedback, under
configurable links, buffers, decoder tiers, and windowing. The
scientific outputs of one run are the logical operation results, the
event log, link traffic, buffer occupancy, per-window decode tiers, and
the Pauli-frame record sequence; the frozen suite in
`validation/responsibility_audit_2026_08_30/` pins all of them.

The architecture has two phases, documented separately:

- setup and compilation: `INITIALIZATION_FLOW.md`. One resolved
  workload creates two views: the execution view for the program
  sequencer, the decoding view for the window manager.
- runtime: `RUNTIME_FLOW.md`. Readouts travel controller-side packing
  into two stores; windows interpret readiness; decoder units are
  assigned, fed by DMA, and gated; exactly one authoritative correction
  reaches the Pauli frame; releases travel back through the controller.

## The eight owners

| Component | Owns |
|---|---|
| ExecutionRuntime (program sequencer) | program DAG, readiness, timestamps; ResourceLedger: one holder per resource |
| Controller | operation to QPU command seam; readout to packing relay; OC then CQ release relay |
| SyndromePacking | fragment reassembly, detector formation, dual write into the stores, CWB/WBD route arbitration |
| WindowManager | window lifecycle, commit and buffer regions, readiness interpretation, boundary courier, retention holds, results |
| DecoderManager | unit pools, two-slot decoupled access-execute units, DMA staging, parking, dispatch |
| StrongEscalation + StrongRequestLedger | strong-job building, deferred submission, request generations, selection and completion matching |
| PauliFrame | priced-write XOR frame; one authoritative correction per window |
| ConditionalRelease | the SWIPER rule: a blocked successor releases on its blocker's final decode |

All eight earned KEEP in the 2026-08-30 responsibility audit; the
decomposition matches SWIPER-SIM's own (DeviceManager, WindowBuilder,
WindowManager, DecoderManager) plus the two-store strong tier.

## The two stores

Buffer 0 (`SyndromeBuffer`) is the upstream store the weak lane reads;
syndrome buffer 1 (`SyndromeBuffer1`) is the room-side store the strong
tier reads. They are different readiness authorities with different
notification contracts: `BUFFER_0_AND_BUFFER_1.md`.

## Reading order

1. `TERMINOLOGY.md` for the link segments and tier vocabulary.
2. `INITIALIZATION_FLOW.md`, then `RUNTIME_FLOW.md`.
3. `BUFFER_0_AND_BUFFER_1.md` and `STRONG_REQUEST_LIFECYCLE.md`.
4. `STATE_OWNERSHIP.md`; the exhaustive field table lives in
   `reports/DECSIM_RUNTIME_STATE_OWNERSHIP.md`.
