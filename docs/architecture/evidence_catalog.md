# Evidence catalog for the syndrome/decoder data path

Canonical registry of stable evidence IDs for decsim's syndrome buffering and
decoder-input architecture. Production Python does not contain research tags or
citations. `tests/test_evidence_traceability.py` validates this external map,
rejects unknown or quarantined IDs, and enforces the source/evidence separation.

Classifications:

- `DIRECT_QEC_CODE`: read directly from a pinned official QEC implementation.
- `DIRECT_CLASSICAL_CODE`: read directly from a pinned classical networking or
  kernel implementation.
- `PAPER_ONLY`: primary paper claims with no inspectable official code found.
- `DECSIM_POLICY`: an explicit decsim simulator policy choice, not a claim
  about what hardware universally does.

Audit provenance: `tmp/validation/combined_syndrome_buffer_implementation_audit_v1`
(REPORT.md, reports/, SOURCE_MANIFEST.json, baseline decsim head
`2b3c838916fb76e8c074edbc2042538151fbb544`) and
`tmp/validation/new_decoder_input_literature_v1` (REPORT.md,
SOURCE_MANIFEST.json). All file:line anchors below are stable because they
refer to the pinned commits listed per entry, not to a moving branch.

Pinned source commits:

| Source | Remote | Commit |
|---|---|---|
| XQsim | https://github.com/SNU-HPCS/XQsim.git | `006c38474c4caf8d51e2065ad3f3a5144b251bfc` |
| CUDA-Q QEC (cudaqx) | https://github.com/NVIDIA/cudaqx.git | `4ab085d049c1f1027fdeea20087c3bebfa945506` |
| Micro Blossom | https://github.com/yuewuo/micro-blossom.git | `a3530d9e58ff790dafc93d56c9a2769ceac84731` |
| Helios | https://github.com/NamiLiy/Helios_scalable_QEC.git | `622d85dac2c78006e1e2aea940725ef800d853c0` |
| Linux | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git | `3d6d817622b0a9721e3cc404df3469171582be13` |
| DPDK | https://github.com/DPDK/dpdk.git | `c1a46b9d9243e922428e8a5f87fa3c6ac177dc5a` |
| lwIP | https://git.savannah.nongnu.org/git/lwip.git | `3d896ba0a37ff3ce73270ca5e230707fe47f60e3` |

## DIRECT_QEC_CODE

### EV-QEC-XQSIM-DIRECT-LOCAL-MATERIALIZATION
- Classification: DIRECT_QEC_CODE
- Source: XQsim @ `006c38474c4caf8d51e2065ad3f3a5144b251bfc`
- Anchors: `src/XQ-estimator/EDU/baseline/rtl/EDU.v:3-45`,
  `src/XQ-estimator/EDU/baseline/rtl/EDU.v:614-643`,
  `src/XQ-estimator/EDU/baseline/rtl/edu_cell.v:156-179`,
  `src/XQ-estimator/EDU/baseline/rtl/edu_cell.v:277-292`,
  `src/XQ-simulator/error_decode_unit.py:1897-1909`,
  `src/XQ-simulator/xq_simulator.py:238-257`
- Claim: the whole packed ancilla-measurement array is wired directly from the
  QCI boundary into distributed EDU cells; each cell captures one bit in a
  two-entry local raw-measurement buffer. There is no central combined
  syndrome buffer and no inter-unit syndrome FIFO. This is the
  streaming/direct local materialization profile.

### EV-QEC-XQSIM-ESM-LOCAL-HISTORY
- Classification: DIRECT_QEC_CODE
- Source: XQsim @ `006c38474c4caf8d51e2065ad3f3a5144b251bfc`
- Anchors: `src/XQ-estimator/EDU/baseline/rtl/edu_cell.v:190-203`,
  `src/XQ-estimator/EDU/baseline/rtl/edu_cell.v:308-319`,
  `src/XQ-estimator/EDU/baseline/rtl/edu_cell.v:320-342`,
  `src/sim_param.py:277-290`,
  `src/XQ-simulator/error_decode_unit.py:1918-1959`
- Claim: derived error-syndrome bits live in a per-cell `AQMEAS_TH=3` entry
  `esm_reg` history that is reused in place; raw measurements and derived
  syndromes are separate local registers. The closest literal "syndrome
  buffer" in XQsim is this distributed per-cell register, not a shared memory.

### EV-QEC-XQSIM-SRMEM-METADATA
- Classification: DIRECT_QEC_CODE
- Source: XQsim @ `006c38474c4caf8d51e2065ad3f3a5144b251bfc`
- Anchors: `src/XQ-estimator/LMU/baseline/rtl/srmem_double.v:34-87`,
  `src/XQ-estimator/LMU/baseline/rtl/srmem_double.v:90-149`,
  `src/XQ-estimator/LMU/baseline/rtl/LMU.v:377-414`,
  `src/XQ-simulator/srmem.py:2-55`,
  `src/XQ-simulator/logical_measurement_unit.py:129-147`
- Claim (correction of an earlier decsim assumption): XQsim `srmem_double` is
  a two-bank metadata memory holding logical-measurement instruction info and
  patch info in PSU/PFU/LMU. No `srmem` instance stores QCI measurements, ESM
  bits, or routed syndrome messages. Any prior decsim reading of XQsim
  `srmem` as a syndrome buffer is wrong and must not be cited as such.

### EV-QEC-XQSIM-NO-JOB-QUEUE-NO-DMA
- Classification: DIRECT_QEC_CODE
- Source: XQsim @ `006c38474c4caf8d51e2065ad3f3a5144b251bfc`
- Anchors: `src/XQ-simulator/error_decode_unit.py:329-364`,
  `src/XQ-estimator/EDU/baseline/rtl/EDU.v:600-612`,
  `src/XQ-estimator/EDU/baseline/rtl/edu_ctrl.v:234-238`,
  `src/XQ-estimator/EDU/baseline/rtl/fifo_ctrl.v:26-59`
- Claim: XQsim's EDU is a round-count-triggered state machine; the only
  top-level FIFO (`edu_pibuf`, depth 2) holds patch info, not syndrome data
  or decoder jobs. Repository-wide searches found no DMA identifiers.
  Modeling XQsim as a queued decoder-job accelerator is an unsupported
  inference.

### EV-QEC-MICROBLOSSOM-INSTRUCTION-STREAM
- Classification: DIRECT_QEC_CODE
- Source: Micro Blossom @ `a3530d9e58ff790dafc93d56c9a2769ceac84731`
- Anchors: `src/cpu/blossom/src/cli.rs:479-490`,
  `src/cpu/embedded/src/mains/benchmark_decoding.rs:112-120`,
  `src/cpu/embedded/src/dual_driver.rs:19-34`,
  `src/cpu/blossom-nostd/src/instruction.rs:9-29`,
  `src/fpga/microblossom/MicroBlossomBus.scala:199-221`,
  `src/fpga/microblossom/DualConfig.scala:13-20`
- Claim: sparse defects become one 32-bit `AddDefectVertex` instruction each,
  passing through a generic depth-4 instruction FIFO. There is no external
  retained syndrome buffer and no bitmap assembly; this is the instruction
  streaming profile.

### EV-QEC-MICROBLOSSOM-VERTEX-STATE-COMMIT
- Classification: DIRECT_QEC_CODE
- Source: Micro Blossom @ `a3530d9e58ff790dafc93d56c9a2769ceac84731`
- Anchors: `src/fpga/microblossom/modules/Vertex.scala:73-95`,
  `src/fpga/microblossom/modules/Vertex.scala:189-200`,
  `src/fpga/microblossom/combinatorial/VertexPostExecuteState.scala:50-63`,
  `src/fpga/microblossom/types/VertexState.scala:16-29`
- Claim: each defect commits directly into per-vertex decoder-local
  `VertexState`; reset clears occupation in place. The first structure that
  retains syndrome meaning is decoder-local algorithm state, not an upstream
  buffer.

### EV-QEC-MICROBLOSSOM-BUS-BACKPRESSURE
- Classification: DIRECT_QEC_CODE
- Source: Micro Blossom @ `a3530d9e58ff790dafc93d56c9a2769ceac84731`
- Anchors: `src/fpga/microblossom/MicroBlossomBus.scala:332-364`,
  `src/fpga/microblossom/MicroBlossomBus.scala:797-838`,
  `src/fpga/microblossom/modules/MicroBlossomLooper.scala:84-115`
- Claim: bus-to-command-queue backpressure is explicit and lossless; the bus
  holds a write transaction until the instruction FIFO is ready. Admission is
  gated on hazard checks, an existing-code example of request admission
  separate from data movement.

### EV-QEC-HELIOS-UPSTREAM-FIFO
- Classification: DIRECT_QEC_CODE
- Source: Helios @ `622d85dac2c78006e1e2aea940725ef800d853c0`
- Anchors: `scripts/total.tcl:833-850`, `scripts/total.tcl:1601-1619`,
  `design/generics/fifo_fwft.v:60-71`, `design/generics/fifo_fwft.v:89-117`,
  `test_benches/full_tests/single_FPGA_FIFO_verification_test_rsc.sv:67-113`,
  `design/wrappers/mem_communicator.v:41-79`
- Claim: the shipped Helios FPGA integration has an independent retained
  upstream 128x8-bit FIFO (127 usable entries) between the producer and the
  decoder core. It retains a serialized command-plus-syndrome byte stream,
  not a parsed whole-shot bitmap; it is elastic decoupling, not complete-shot
  buffering, and it is integration-owned, not decoder-core-owned.

### EV-QEC-HELIOS-ROUND-ASSEMBLY
- Classification: DIRECT_QEC_CODE
- Source: Helios @ `622d85dac2c78006e1e2aea940725ef800d853c0`
- Anchors: `design/stage_controller/control_node_single_FPGA.v:35-39`,
  `design/stage_controller/control_node_single_FPGA.v:53-64`,
  `design/stage_controller/control_node_single_FPGA.v:131-182`,
  `design/wrappers/Helios_single_FPGA_core.v:36-44`
- Claim: the core controller allocates exactly one aligned spatial round of
  assembly state (`measurements[ALIGNED_PU_PER_ROUND-1:0]`), shifts payload
  bytes into it, and consumes it into distributed PE state every round. The
  assembly register is staging inside the decoder controller, not an upstream
  retained buffer.

### EV-QEC-HELIOS-PE-M-STATE
- Classification: DIRECT_QEC_CODE
- Source: Helios @ `622d85dac2c78006e1e2aea940725ef800d853c0`
- Anchors: `design/pe/processing_unit_single_FPGA_v2.v:95-120`,
  `design/pe/processing_unit_single_FPGA_v2.v:139-176`,
  `design/wrappers/single_FPGA_decoding_graph_dynamic_rsc.sv:54-60`,
  `design/wrappers/single_FPGA_decoding_graph_dynamic_rsc.sv:112-125`
- Claim: the only complete-window syndrome image in the Helios core is the
  distributed one-bit-per-PE `m` register plane, decoder-local state rewritten
  in place by the per-round shift load. Decoder-local materialization is round
  streaming, not window DMA.

### EV-QEC-HELIOS-INPUT-READY-BACKPRESSURE
- Classification: DIRECT_QEC_CODE
- Source: Helios @ `622d85dac2c78006e1e2aea940725ef800d853c0`
- Anchors: `design/stage_controller/control_node_single_FPGA.v:170-204`,
  `design/stage_controller/control_node_single_FPGA.v:235-245`,
  `design/wrappers/mem_communicator.v:61-66`,
  `design/wrappers/mem_communicator.v:109-113`
- Claim: core input backpressure is coarse (`input_ready` only in idle or
  measurement-preparing stages); decode triggers after the final round load
  with no next-shot overlap. Both shipped producers can advance without
  testing FIFO readiness, a real backpressure-compliance gap upstream of the
  FIFO.

### EV-QEC-CUDAQ-HOST-STAGING-RETAINED-SPLIT
- Classification: DIRECT_QEC_CODE
- Source: cudaqx @ `4ab085d049c1f1027fdeea20087c3bebfa945506`
- Anchors: `libs/qec/lib/realtime/rpc_producer.cpp:217-256`,
  `libs/qec/lib/realtime/rpc_producer.cpp:114-143`,
  `libs/qec/lib/realtime/qec_realtime_session.cpp:174-219`,
  `libs/qec/lib/decoder.cpp:30-50`, `libs/qec/lib/decoder.cpp:314-352`,
  `libs/qec/lib/decoder.cpp:366-379`
- Claim: CUDA-Q QEC HOST mode is an explicit staging-plus-retained split:
  caller bytes -> temporary packed vector -> reusable RX ring slot ->
  temporary unpacked vector -> decoder-owned retained `msyn_buffer`. The ring
  slot is never the retained combined-syndrome buffer.

### EV-QEC-CUDAQ-RING-SLOT-LIFECYCLE
- Classification: DIRECT_QEC_CODE
- Source: cudaqx @ `4ab085d049c1f1027fdeea20087c3bebfa945506`
- Anchors: `libs/qec/lib/realtime/rpc_producer.cpp:69-111`,
  `libs/qec/lib/realtime/rpc_producer.cpp:145-188`,
  `libs/qec/lib/realtime/rpc_producer.cpp:273-308`,
  `libs/qec/lib/realtime/qec_realtime_session.cpp:923-951`
- Claim: ring slots are transient transport frames owned by the producer from
  acquire through release; combined syndrome state survives slot release only
  because it was already copied into decoder-owned state. Slot release on the
  final consuming transfer is existing-code precedent for last-transfer
  release semantics.

### EV-QEC-CUDAQ-DEVICE-MAPPED-RING
- Classification: DIRECT_QEC_CODE
- Source: cudaqx @ `4ab085d049c1f1027fdeea20087c3bebfa945506`
- Anchors: `libs/qec/lib/realtime/qec_realtime_session.cpp:602-655`,
  `libs/qec/lib/realtime/qec_realtime_session.cpp:657-696`,
  `libs/qec/lib/realtime/qec_realtime_session.cpp:780-838`,
  `libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_bridge.cpp:27-40`,
  `libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_bridge.cpp:249-267`
- Claim: DEVICE mode uses pinned-mapped RX slots read directly by the GPU (no
  syndrome H2D memcpy), yet still accumulates into a separate registered
  plugin `GpuDecoderState`. Mapped aliasing removes a copy but not the logical
  capture into decoder-owned state. The plugin state layout is proprietary and
  outside the pinned tree.

### EV-QEC-CUDAQ-PIPELINE-H2D-DMA
- Classification: DIRECT_QEC_CODE
- Source: cudaqx @ `4ab085d049c1f1027fdeea20087c3bebfa945506`
- Anchors: `libs/qec/lib/realtime/realtime_pipeline.cu:116-170`,
  `libs/qec/lib/realtime/realtime_pipeline.cu:377-416`,
  `libs/qec/lib/realtime/host_side_dispatcher_design.md:34-42`,
  `libs/qec/lib/realtime/host_side_dispatcher_design.md:180-198`,
  `libs/qec/include/cudaq/qec/realtime/pipeline.h:43-58`
- Claim: the generic realtime pipeline's optional worker `pre_launch_fn` does
  `cudaMemcpyAsync(HostToDevice)` from a pinned ring slot to a separate
  decoder/TRT input allocation. DMA into decoder-local memory is a concrete,
  named, optional profile, not the universal syndrome-input mechanism.

### EV-QEC-CUDAQ-DECODE-TRIGGER-RESET
- Classification: DIRECT_QEC_CODE
- Source: cudaqx @ `4ab085d049c1f1027fdeea20087c3bebfa945506`
- Anchors: `libs/qec/lib/decoder.cpp:381-389`, `libs/qec/lib/decoder.cpp:557-563`,
  `libs/qec/lib/decoder.cpp:570-586`, `libs/qec/lib/decoder.cpp:607-629`
- Claim: decode triggers when decoder-owned retained input becomes full (or on
  round-based sliding-window conditions); after decode, indices reset for
  in-place reuse without freeing retained buffers. Data readiness, not request
  arrival, fires the decode.

## DIRECT_CLASSICAL_CODE

### EV-CLS-LINUX-REASSEMBLY-OWNERSHIP
- Classification: DIRECT_CLASSICAL_CODE
- Source: Linux @ `3d6d817622b0a9721e3cc404df3469171582be13`
- Anchors: `net/ipv4/ip_fragment.c:359-369`, `net/ipv4/ip_fragment.c:397-467`,
  `net/ipv4/inet_fragment.c:499-600`, `net/ipv4/inet_fragment.c:602-673`
- Claim: reassembly completion performs descriptor surgery and pointer
  chaining; the returned datagram is the input fragment allocation. Completion
  changes ownership and state, it does not copy into a second packed buffer.

### EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH
- Classification: DIRECT_CLASSICAL_CODE
- Source: Linux @ `3d6d817622b0a9721e3cc404df3469171582be13`
- Anchors: `net/ipv4/ip_fragment.c:407`, `net/ipv4/ip_fragment.c:271-275`,
  `net/ipv4/inet_fragment.c:261-285`, `include/net/inet_frag.h:41-45`
- Claim: the reassembly identity is retired (marked complete and unhashed)
  before the completed datagram is published downstream; a late fragment is
  rejected as a duplicate. Retire-before-publish is the classical pattern for
  a packet's assembly-to-retained state transition.

### EV-CLS-LWIP-PBUF-CHAIN
- Classification: DIRECT_CLASSICAL_CODE
- Source: lwIP @ `3d896ba0a37ff3ce73270ca5e230707fe47f60e3`
- Anchors: `src/core/ipv4/ip4_frag.c:86-103`, `src/core/ipv4/ip4_frag.c:619-675`
- Claim: lwIP overlays reassembly bookkeeping on fragment header bytes,
  chains pbufs on completion, removes the reassembly context, and returns the
  first fragment's allocation. One-allocation ownership transition without a
  second payload store.

### EV-CLS-DPDK-MBUF-OWNERSHIP
- Classification: DIRECT_CLASSICAL_CODE
- Source: DPDK @ `c1a46b9d9243e922428e8a5f87fa3c6ac177dc5a`
- Anchors: `lib/ip_frag/ip_frag_internal.c:192-196`,
  `lib/ip_frag/ip_frag_internal.c:247`,
  `lib/ip_frag/rte_ipv4_reassembly.c:43-48`,
  `lib/ip_frag/rte_ipv4_reassembly.c:63-68`,
  `lib/mbuf/rte_mbuf.h:1860-1883`, `lib/mbuf/rte_mbuf.h:359-445`
- Claim: DPDK expresses ownership transfer by storing the mbuf handle and
  nulling the local one; chaining moves pointers and metadata only. Retention
  beyond first consumption uses refcount lifetime, not a second copy.

### EV-CLS-LINUX-DMA-OWNERSHIP
- Classification: DIRECT_CLASSICAL_CODE
- Source: Linux @ `3d6d817622b0a9721e3cc404df3469171582be13`
- Anchors: `Documentation/core-api/dma-api.rst:550-585`,
  `Documentation/core-api/dma-api.rst:174`,
  `Documentation/core-api/dma-api-howto.rst:722-780`,
  `Documentation/core-api/dma-api-howto.rst:679-717`
- Claim: DMA APIs require explicit CPU-versus-device buffer ownership, and
  sync calls transfer that ownership; one mapped buffer can alternate owners
  without remapping or copying. Authoritative ownership-token precedent, not
  evidence that QEC inputs universally use DMA.

## PAPER_ONLY

### EV-PAPER-LILLIPUT
- Classification: PAPER_ONLY
- Source: arXiv:2108.06569; DOI 10.1145/3503222.3507707
- Claim: readout buffer overwritten every cycle; the most recent `m` rounds
  retained in a FIFO; sliding window advances per cycle. No official code or
  code-availability statement found; the only GitHub match is an explicitly
  third-party reconstruction and is excluded.

### EV-PAPER-AFS
- Classification: PAPER_ONLY
- Source: arXiv:2001.06598; DOI 10.1109/HPCA53966.2022.00027
- Claim: describes quantum-to-decoder transfer/compression and decoder-local
  pipeline structures; does not describe fragment reassembly or an archival
  upstream retained buffer. No public repository found.

### EV-PAPER-COLLISION-CLUSTERING
- Classification: PAPER_ONLY
- Source: arXiv:2309.05558v2; DOI 10.1038/s41928-024-01319-5
- Claim: decoder-local data structures; streaming/sliding-window
  implementation explicitly left as future work. Zenodo 11621877 holds CSV
  results and Stim circuits, not decoder HDL/software.

### EV-PAPER-CAUNE
- Classification: PAPER_ONLY
- Source: arXiv:2410.05202v1
- Claim: latest round stored in decoder-sequencer memory, round buffers pushed
  to a data stack, combined into 32-bit strings, transferred to the decoder.
  Does not establish a separate retained archival upstream buffer. Control
  system is proprietary; no code or data link found.

## DECSIM_POLICY

### EV-POLICY-ONE-ALLOCATION-OWNERSHIP-TRANSITION
- Classification: DECSIM_POLICY
- Grounding: EV-CLS-LINUX-REASSEMBLY-OWNERSHIP, EV-CLS-LWIP-PBUF-CHAIN,
  EV-CLS-DPDK-MBUF-OWNERSHIP, EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH;
  combined audit REPORT.md "Least-claim decsim direction"
- Claim: in decsim's least-claim baseline, an upstream syndrome payload may
  transition `ASSEMBLING -> PACKED_RETAINED` on one physical allocation
  without charging a fabricated byte copy. This is a simulator ownership
  policy grounded in classical reassembly practice, explicitly NOT a claim
  that QEC hardware generally shares one buffer (the audited open QEC systems
  mostly do not implement fragments at all).

### EV-POLICY-DIRECT-VALUE-BASELINE
- Classification: DECSIM_POLICY
- Grounding: new_decoder_input_literature_v1 REPORT.md "Policy"
- Claim: use direct-value decoder-input delivery as the baseline until a named
  hardware profile is selected. Do not label the generic buffer-to-decoder
  path DMA or pointer-based.

### EV-POLICY-NAMED-TRANSFER-PROFILES
- Classification: DECSIM_POLICY
- Grounding: EV-QEC-XQSIM-DIRECT-LOCAL-MATERIALIZATION,
  EV-QEC-MICROBLOSSOM-INSTRUCTION-STREAM, EV-QEC-HELIOS-PE-M-STATE,
  EV-QEC-CUDAQ-HOST-STAGING-RETAINED-SPLIT, EV-QEC-CUDAQ-PIPELINE-H2D-DMA,
  EV-CLS-LINUX-REASSEMBLY-OWNERSHIP
- Claim: `DecoderInputTransfer` must remain mechanism-neutral with selectable
  named profiles: direct local/streaming load (XQsim, Micro Blossom, Helios);
  ring plus decoder-owned accumulation (CUDA-Q QEC); ring plus H2D DMA into
  decoder-local memory (CUDA-Q generic pipeline); shared-allocation ownership
  transition (classical reassembly analogue). No universal mechanism exists in
  the audited implementations.

### EV-POLICY-REQUEST-ADMISSION-DATA-READINESS-SPLIT
- Classification: DECSIM_POLICY
- Grounding: EV-QEC-CUDAQ-DECODE-TRIGGER-RESET,
  EV-QEC-HELIOS-INPUT-READY-BACKPRESSURE, EV-QEC-MICROBLOSSOM-BUS-BACKPRESSURE
- Claim: decsim separates request admission (a `WindowInputRequirement` or
  `DecodeJob` may be admitted before its data exists) from data readiness (the
  decode fires only when required input is materialized decoder-locally). In
  the audited code, admission gates and readiness triggers are distinct
  mechanisms.

### EV-POLICY-LAST-TRANSFER-RELEASE
- Classification: DECSIM_POLICY
- Grounding: EV-QEC-CUDAQ-RING-SLOT-LIFECYCLE,
  EV-CLS-LINUX-RETIRE-BEFORE-PUBLISH, EV-CLS-DPDK-MBUF-OWNERSHIP
- Claim: the upstream retained allocation is released when its last
  outstanding transfer or retention owner token completes, using
  ownership/refcount lifetime rather than eager copies. Simulator ownership
  policy modeled on ring-slot release and classical retire/refcount practice.

## QUARANTINED

Quarantined entries MUST NOT be cited by any decsim docstring `Evidence:` tag.
`tests/test_evidence_traceability.py` enforces this.

### EV-QUARANTINE-ASTREA-PDF
- Classification: PAPER_ONLY (QUARANTINED)
- Status: QUARANTINED
- Reason: the archived file
  `tmp/validation/new_decoder_input_literature_v1/papers/astrea.pdf` (sha256
  `e5b70432427c574914f10351ba9e81554f565f7fb0c2ac7d2fd285d6eb751a7a`) is NOT
  the Astrea paper (DOI 10.1145/3579371.3589037, arXiv:2305.06040). Its
  extracted text is a silicon-photonics neural-network paper (Zazzi et al.,
  RWTH Aachen). The combined audit SOURCE_MANIFEST notes: "Do not treat
  incorrect archived Astrea PDF as evidence." No Astrea-derived claim may be
  cited until a verified correct PDF replaces the archive and a new,
  non-quarantined ID is registered.
