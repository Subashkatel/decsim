# Core source evidence map

This file keeps research citations out of the simulator source. The source
files explain behavior only. This map explains why the behavior exists and
where it came from.

## How to read this map

- **DIRECT_QEC_CODE**: inspected code from a QEC implementation.
- **DIRECT_CLASSICAL_CODE**: inspected classical systems code with the same
  ownership or lifetime pattern.
- **DECSIM_POLICY**: a simulator design choice. It is not a hardware claim.
- Evidence IDs are defined in `docs/architecture/evidence_catalog.md`.
- The detailed implementation audit is in
  `tmp/validation/combined_syndrome_buffer_implementation_audit_v1/`.

## Source-to-evidence map

| Source symbol | Why it exists | Evidence IDs |
|---|---|---|
| `message.QPUReadout` | Marks the effective QPU-to-controller return separately from controller-accepted detector data. The baseline begins after measurement-to-detection formation and does not claim analog fidelity. | DECSIM_POLICY; QuMA/eQASM validate only the preceding physical measurement-bit boundary |
| `controller.accept_qpu_readout` and `TimingConfig.t_binary_availability_us` | Detector-data availability follows a modeled controller-side receive boundary. The optional fixed cost is DECSIM_POLICY; zero means no additional post-QC cost. | DECSIM_POLICY; named system profiles must define what the effective cost includes |
| `syndrome_ingress.relay_qpu_readout` | The controller-side receive port charges QPU-to-controller QC once, then exposes validated binary data after the optional availability cost. | Khalid Fig. 2/Table II; DECSIM_POLICY |
| `syndrome_buffer.SyndromeBufferingConfig` | Upstream and decoder-local capacities are independent. Both are unbounded by default. | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION; EV-POLICY-NAMED-TRANSFER-PROFILES |
| `syndrome_buffer.SyndromeBufferRoundState` | One modeled allocation changes state during assembly, packing, retention, and release. | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION; EV-CLS-LINUX-REASSEMBLY-OWNERSHIP; EV-CLS-DPDK-MBUF-OWNERSHIP; EV-CLS-LWIP-PBUF-CHAIN |
| `syndrome_buffer.SyndromeBuffer` | One component owns upstream allocation and retention. It does not own timing, links, or decoder-local memory. | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION; EV-POLICY-LAST-TRANSFER-RELEASE; EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH |
| `syndrome_buffer.accept_fragment` | The first fragment claims the round allocation. Later fragments fill the same allocation. | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION; EV-CLS-LINUX-REASSEMBLY-OWNERSHIP |
| `syndrome_buffer.finish_packing` | Fragment admission closes before a retained packet is published. | EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH; EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION |
| `syndrome_buffer.register_hold` and hold transfers | Shared rounds stay live until every consumer has released or transferred its hold. | EV-POLICY-LAST-TRANSFER-RELEASE; EV-CLS-DPDK-MBUF-OWNERSHIP |
| `window_manager.accept_window_input` | The window manager accepts only packets already published by the upstream buffer. | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION; EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH |
| `window_manager._bind_decoder_input_hold` | A decoder request owns a typed upstream hold until input materialization finishes. | EV-POLICY-LAST-TRANSFER-RELEASE; EV-QEC-CUDAQ-RING-SLOT-LIFECYCLE |
| `decoder_input.FixedLatencyDecoderInputTransfer` | Request admission and input readiness are separate events. The baseline says only “materialize after this latency.” | EV-POLICY-DIRECT-VALUE-BASELINE; EV-POLICY-REQUEST-ADMISSION-DATA-READINESS-SPLIT; EV-QEC-CUDAQ-HOST-STAGING-RETAINED-SPLIT |
| `decoder_local.MaterializedSyndromeRound` | Decoders read decoder-owned round state rather than the upstream allocation. | EV-QEC-XQSIM-DIRECT-LOCAL-MATERIALIZATION; EV-QEC-HELIOS-PE-M-STATE |
| `decoder_local.DecoderInput` and `materialize_decoder_input` | A completed transfer creates an immutable decoder-local value for one request. | EV-QEC-CUDAQ-HOST-STAGING-RETAINED-SPLIT; EV-POLICY-DIRECT-VALUE-BASELINE; EV-POLICY-NAMED-TRANSFER-PROFILES |
| `decoder_local.DecoderLocalMemory` | Local allocation lifetime is separate from the upstream buffer and from scheduling. | EV-QEC-XQSIM-DIRECT-LOCAL-MATERIALIZATION; EV-QEC-MICROBLOSSOM-VERTEX-STATE-COMMIT; EV-QEC-CUDAQ-HOST-STAGING-RETAINED-SPLIT |
| `planner._plan_syndrome_buffering` | Planning reports one upstream capacity/retention witness instead of SB0/SB1 endpoint ledgers. | EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION; EV-POLICY-LAST-TRANSFER-RELEASE |

## Important limits

1. One allocation across assembly and retention is a **DECSIM_POLICY**. It is
   not a universal statement about QEC hardware.
2. `DecoderInputTransfer` is mechanism-neutral. It does not imply DMA, a ring,
   MMIO, a pointer handoff, or any placement.
3. XQsim `srmem_double` is instruction/patch metadata, not syndrome storage.
4. The archived Astrea PDF with SHA-256 prefix `e5b70432` is unrelated and is
   quarantined. It supports no claim.
5. LILLIPUT, AFS, Collision Clustering, and Caune remain paper-only because no
   official public implementation was verified.

## Pinned audit inputs

The detailed report and exact repository commits are recorded in:

- `tmp/validation/combined_syndrome_buffer_implementation_audit_v1/REPORT.md`
- `tmp/validation/combined_syndrome_buffer_implementation_audit_v1/SOURCE_MANIFEST.json`
- `tmp/validation/combined_syndrome_buffer_implementation_audit_v1/SHA256SUMS.json`

## Other core algorithms

### General simulator behavior

| Source area | Grounding |
|---|---|
| `run_spec.LogicalOperationResult.logical_failure` | arXiv:2303.15933v2 §2.1 and Stim/Sinter v1.16.0: failure means any predicted observable differs from sampled truth. |
| `codes.SurfaceCodeModel.buffering_floor` | Skoric et al., arXiv:2209.08552 (`n_buf=d`); Bombín et al., arXiv:2303.04846 (`b>=d`). |
| `codes.BBCodeModel` | Bravyi et al., arXiv:2308.07915. |
| `schemes.SlidingWindowScheme` terminal flushing | Huang–Puri §III; Tan et al. supplement §S2.C; QUITS §5.2. |
| `schemes.TanSandwichScheme` seam layout | Tan et al., Fig. 2(c), supplement §S2.D, seam offset `t=0`. |
| `detector_error_model.has_leading_buffer` and `decode_windowed` | Skoric et al., arXiv:2209.08552 §I.B–I.C; Tan et al., arXiv:2209.09219 supplement §S2.C–S2.D. |
| `metrics.ConditionalReactionTime` | SWIPER reaction-time convention; average includes every conditional operation. |

### Switching and soft output

| Source area | Grounding |
|---|---|
| `switching.Switching(double_window=True)`, deferred escalation, and slab validation | Toshio et al., arXiv:2510.25222 §III.C, Fig. 12. |
| `decoder_manager.StrategyServicesImpl.defer_strong_escalation` | Same double-window far-boundary rule. |
| `window_manager._build_pending_strong_job` complete-slab check | Same stored-block readiness rule. |
| `soft_output.SoftOutputMetric` | Toshio et al., arXiv:2510.25222 §II.B. |
| `soft_output.complementary` | Toshio et al., arXiv:2510.25222 §II.C. |
| `soft_output.cluster.union_find_cluster_gap_source` | Huang–Newman–Brown, arXiv:2004.04693 §II; arXiv:2405.07433v2 Definition 9/Algorithm 2; Toshio et al., arXiv:2510.25222 §II.C. |
| Exact cluster-gap likelihood interpretation | Only the uniform repetition-code setting in arXiv:2405.07433v2, Theorem 10. The surface-code value is a confidence score, not a calibrated failure probability. |

### Decoder implementations

| Source area | Grounding |
|---|---|
| `relay_bp_decoder.window_decoder` | Müller et al., arXiv:2506.01779v2, Algorithm 1. |
| `union_find_decoder.decoder` growth and peeling | Huang–Newman–Brown, arXiv:2004.04693; Delfosse–Nickerson, arXiv:1709.06218v3, Algorithms 1–2. |

