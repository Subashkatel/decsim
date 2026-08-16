# Syndrome data path: historical-to-integrated symbol map

Traceability map from the deleted pre-Phase-A architecture to the integrated
mechanism-neutral runtime selected after the combined syndrome-buffer
implementation audit
(`tmp/validation/combined_syndrome_buffer_implementation_audit_v1`). Evidence
IDs refer to `docs/architecture/evidence_catalog.md`. Deleted symbols appear
only in the explicitly historical column; they are not compatibility APIs.

## Integrated shape (mechanism-neutral)

```
Controller --command--> QPUDevice --QPUReadout--> Controller
           --binary detector data--> SyndromeIngress -> SyndromeBuffer
           -> WindowManager -> DecoderInputTransfer
           -> DecoderInputStoreStager -> DecoderInputStore
           -> DecoderManager ready queue -> decoder
```

The controller availability boundary has a configurable fixed cost. The default is
zero additional cost because physical acquisition and classification may already
be included in the round cadence. The QC transfer reaches the controller-side receive boundary before the optional
binary-availability delay. This models causal cost without
claiming analog-waveform, ADC, classifier, or universal-placement fidelity.

Evidence: QuMA Secs. 4.2.1 and 5.1.2; eQASM Secs. 2.3.6--2.3.7;
Khalid et al. Fig. 2 and Table II; [EV-POLICY-NAMED-TRANSFER-PROFILES,
EV-POLICY-DIRECT-VALUE-BASELINE]

## Historical-to-integrated symbol map

| Deleted pre-Phase-A symbol | Integrated symbol/concept | Evidence | Notes |
|---|---|---|---|
| `SyndromeStaging`, `_StagingSlot`, `_ControllerSlotState` | `SyndromeIngress` plus `SyndromeBuffer` states `ASSEMBLING -> PACKING -> PACKED_RETAINED` | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION, EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH | `SyndromeIngress` owns timing, routing, and arbitration; `SyndromeBuffer` owns one allocation across assembly and retention. |
| `SyndromeStagingPolicy` | `SyndromeIngressPolicy`, `ReassemblyQueueAdmission`, `IngressOverflowPolicy` | EV-QEC-HELIOS-INPUT-READY-BACKPRESSURE, EV-QEC-MICROBLOSSOM-BUS-BACKPRESSURE | Overflow and admission remain explicit policies; they are not universal hardware claims. |
| `PayloadStore`, `_EndpointLedger`, `EndpointRole.SB0/SB1` | `SyndromeBuffer` bounded slot and typed-hold ledger | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION, EV-QEC-HELIOS-UPSTREAM-FIFO | The SB0/SB1 split was a decsim ledger, not a physical claim. |
| `PayloadStore.register_owner` / `release_owner` | `SyndromeBuffer.register_hold`, `replace_hold`, `transfer_hold`, `release_hold` | EV-CLS-DPDK-MBUF-OWNERSHIP, EV-POLICY-LAST-TRANSFER-RELEASE | Weak, strong, replay, and rephase lifetimes use typed holds on the same backing. |
| `complete_input_transfer` and `_TransportOwner` | `DecoderInputHold` released by `DecoderInputStoreStager` on storage admission | EV-POLICY-LAST-TRANSFER-RELEASE, EV-QEC-CUDAQ-RING-SLOT-LIFECYCLE | Upstream lifetime ends only after decoder-input store input exists or the request is cancelled. |
| `FixedDelayDirectValueTransport` | `DecoderInputTransfer` protocol and `FixedLatencyDecoderInputTransfer` baseline | EV-POLICY-DIRECT-VALUE-BASELINE, EV-POLICY-NAMED-TRANSFER-PROFILES | The baseline is mechanism-neutral; DMA, rings, MMIO, and streaming require named profiles. |
| Shared pre-transfer `DecodeJob.payloads` as decoder input | Immutable `DecoderInput` in `DecoderInputStore` | EV-QEC-CUDAQ-HOST-STAGING-RETAINED-SPLIT, EV-QEC-HELIOS-PE-M-STATE, EV-QEC-XQSIM-ESM-LOCAL-HISTORY | The stager materializes decoder-input store rounds on transport arrival, before ready-queue admission. A temporary payload view remains only for existing decoder adapters after materialization. |
| Admission conflated with input readiness | `request_admitted_ticks`, transfer completion/`ready_time`, dispatch, and completion | EV-POLICY-REQUEST-ADMISSION-DATA-READINESS-SPLIT, EV-QEC-CUDAQ-DECODE-TRIGGER-RESET | These are distinct lifecycle events. |
| Per-endpoint capacity/exhaustion | Independent `SyndromeBufferingConfig.upstream_packet_slots` (rounds upstream) and `RunSpec.decoder_input_store` (rounds per decoder unit pool, with STALL or FAIL_STOP overflow) | EV-QEC-MICROBLOSSOM-BUS-BACKPRESSURE, EV-QEC-HELIOS-INPUT-READY-BACKPRESSURE | Both capacities default to the unbounded baseline and are independently configurable. |
| `SyndromeRoundPacket`, `RetainedSyndromeFragment` | Same immutable, identity-preserving packet/fragment types | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION, EV-CLS-DPDK-MBUF-OWNERSHIP | Patch, operation, round, fragment, bit, code, and size identity remain available across the boundary. |

## Corrections and quarantine

- XQsim `srmem`: earlier decsim reasoning treated XQsim `srmem` banks as
  syndrome storage. The pinned-code audit shows `srmem_double` holds
  instruction/patch metadata only; the syndrome-bearing state is the per-cell
  two-slot raw buffer plus `esm_reg` history. Cite
  EV-QEC-XQSIM-SRMEM-METADATA for the corrected reading; do not cite XQsim as
  a central or banked syndrome-buffer precedent
  (EV-QEC-XQSIM-NO-JOB-QUEUE-NO-DMA).
- Astrea: the archived Astrea PDF is the wrong paper and is quarantined as
  EV-QUARANTINE-ASTREA-PDF. No decsim docstring may cite it; the traceability
  test enforces this.
- "PayloadStore" is not a literature entity. The stable cross-system concepts
  are input ownership, completeness/load granularity, local materialization,
  lifetime, and backpressure (combined audit REPORT.md). The rename to
  `SyndromeBuffer` removes the fabricated term.

## Citation convention

Production Python contains only behavior and ownership documentation. Evidence
IDs and research citations remain in this catalog and the validation reports;
`tests/test_evidence_traceability.py` enforces that separation.
