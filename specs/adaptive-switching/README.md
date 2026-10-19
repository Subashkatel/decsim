# Adaptive switching implementation plan

Status: proposed implementation plan. No simulator changes or experimental
validation have been performed for this plan.

Planning checkout: `decsim_adaptive`, branch `adaptive-decoding-plan`, based on
local `main` at `75f157bed3e4d89a67d07491e0fdb25e38b89e7b`.

## Goal

Establish whether union-find and belief-matching switching has a useful
correctness and timing trade-off, then build an understandable controller
that adjusts escalation to the available capacity and feedforward deadlines.
Use editable hardware timing inputs and the same event path for timing-only
and real-decoding experiments. Advance only after each slice's gate passes.

## Next Agent Prompt

The user requested planning before implementation. Start by reviewing
[slice 00](slices/00-baseline-contract.md) with the user. Confirm unresolved
baseline choices, then materialize its configuration when implementation is
requested. Do not launch the production grid, remove the old online controller,
or implement later slices merely because this plan exists. Read the active
checkout's `STYLE.md` before changes. Record actual commands and evidence here
as work proceeds. No implementation slice is complete yet.

## Starting setup

| Item | Choice | Status |
| --- | --- | --- |
| Weak decoder | Weighted union find | User selected |
| Confidence | Cluster gap from the weak decode | User selected |
| Strong decoder | Belief matching | User selected |
| Shot length | Exactly `10 * distance` QEC rounds | User selected |
| Production samples | Exactly 1,000,000 shots per distance/error-rate point and decoding configuration | User selected; no early stop at a failure count |
| Initial scheme | Ordinary sequential sliding windows | User selected |
| Regular window | Commit `d` rounds, read `d` buffer rounds | Proposed starting geometry |
| Initial threshold | 20 dB, escalate below it, keep at equality | Existing baseline to retain |
| Syndrome cadence | One round per microsecond per active patch | Proposed baseline from the meetings |
| Circuit | Stim rotated surface-code Z memory | Proposed; record exact circuit and version |
| Noise | Existing SD6 reproduction profile, no bursts, noiseless preparation/readout boundaries | Proposed; resolve channels and boundary convention before running |
| Strong parameters | Product-sum BP, maximum 20 iterations | Proposed reproduction setting, not a known parameter from the paper |
| Resources | One weak worker and one strong worker, explicit links and buffers | Proposed baseline; no implicit extra concurrency |
| First pilot | `d=5`, physical error probability `0.001` | Proposed diagnostic point, not the production grid |

The distance/error-rate grid remains open. Begin with a single point and a
small fixed seed range. Preserve seeds as a pilot grows. Production correctness
uses finite shots; sustained-capacity studies use a continuous workload and
do not clear queues between artificial shot boundaries.

Terminal windows are part of the scientific setup. Inspect the actual planner
and list read/commit extents before choosing a shared terminal policy. Switching
currently selects `lookahead`; weak-only and strong-only comparisons must use
the intended common convention. A shorter final window needs a declared timing
entry or an explicit pricing rule. Do not add physical rounds to the promised
`10d` shot without changing and reporting the workload definition.

## Evidence from the existing checkout

These are inspected implementation entry points, not claims of validation:

| Capability | Existing entry point | Limit relevant to this plan |
| --- | --- | --- |
| Threshold choices | `decsim/escalation/threshold_sources.py` | Fixed and table-selected thresholds exist; the online source combines rate tracking with disagreement audits |
| Online state | `decsim/experiments/collect.py` | Threshold state can persist across shots; each shot still builds a new machine |
| Weak timing | `decsim/decoders/union_find/cycle_count.py` | A hardware model exists; it needs validation for the selected architecture |
| Strong timing | `decsim/decoders/belief_matching/decoder.py` | Host measurement or supplied iteration/matching times; no general hardware table contract established here |
| Cluster-gap timing | `decsim/confidence/cluster.py` | Supplied walk time or host measurement; currently charged on the weak unit |
| Supplied service time | `decsim/decoders/decoders.py` | A timing-only fixed-latency decoder exists |
| Timing-only source | `decsim/qpu/syndrome_devices.py` | Generates timing traffic without real logical-failure truth |
| Windowing | `decsim/windows/schemes/sliding.py` | Geometry and terminal behavior affect ownership and work |
| Re-decoding shapes | `decsim/escalation/strong_window_shapes.py` | Ordinary redo and double-window decoding have different dependencies and ownership |
| Measurements | `decsim/experiments/measure.py`, `stream_measures.py` | Existing throughput and reaction metrics need their endpoints checked |
| Program dependencies | `decsim/records/program.py` | Operations declare dependencies, streams, feedback blocking and result return |
| Feedback path | `decsim/pauli_frame/decision_dispatch.py`, `decsim/controller/feedback_streams.py` | Reuse and verify these owners rather than create a second runtime |

## Meaning of the requirements

- Throughput is sustained advancement of finalized, consecutively covered
  syndrome rounds. Decoding overlapping windows repeatedly is not extra useful
  throughput. Record per-patch coverage as well as aggregate rates.
- Decoder service time is resource occupancy, separate from input/dependency
  waits, transfers, and consumer response time. Model initiation interval
  separately if hardware overlaps multiple jobs.
- Correction age is the gap between the QPU's progress and finalized coverage;
  lookahead and unresolved earlier windows contribute to it.
- Feedback response ends when the relevant result reaches the consuming
  operation's control path. A Pauli-frame commit is an intermediate endpoint.
- A reaction budget for memory is an explicit experimental bound. A gate
  dependency supplies an operation-specific required result and desired start.
- Table values initially describe assumed hardware. Correct simulation of
  those values and realism of those values require separate evidence.
- A 1% escalation fraction is a fraction of eligible windows, not GPU busy
  time, and not a guarantee that deadlines will be met.

## Execution order

| Slice | Depends on | Deliverable | Gate | Status |
| --- | --- | --- | --- | --- |
| [00 Baseline contract](slices/00-baseline-contract.md) | None | One reproducible baseline configuration and window manifest | Inputs and terminal extents are explicit | Proposed |
| [01 Correctness audit](slices/01-correctness-audit.md) | 00 | Paired outcomes and counts at one point | Logical accounting and independent references agree | Proposed |
| [02 Supplied timing](slices/02-supplied-timing.md) | 00 | Editable timing profile and deterministic timing fixture | Analytical event times agree; no host time leaks | Proposed |
| [03 Sustained capacity](slices/03-sustained-capacity.md) | 01, 02 | Weak-only and fixed-switching long-run traces | Capacity boundary, upstream backlog and drain are accounted for | Proposed |
| [04 Static trade-off](slices/04-static-tradeoff.md) | 01, 03 | Held-out correctness/timing frontier | Achieved escalation and uncertainty reported | Proposed |
| [05 Percentage controller](slices/05-percentage-controller.md) | 04 | Audit-free rate tracker and separate optional hard admission cap | Rate tracking, quota accounting and denied escalations verified | Proposed |
| [06 Adaptive capacity](slices/06-adaptive-capacity.md) | 03, 05 | Budget adaptation from causal pipeline feedback | Stable behavior, overload recovery and infeasibility measured | Proposed |
| [07 Gate feedback](slices/07-gate-feedback.md) | 02, 05; 06 for adaptive tests | Small timing-only feedforward workload | Full result-to-consumer release path matches a reference schedule | Proposed |
| [08 Lattice surgery schedules](slices/08-lattice-surgery-schedules.md) | 07 | One compiler import with preserved dependencies | Geometry, rounds and dependencies match an external fixture | Proposed |
| [09 Double windows](slices/09-double-windows.md) | 01 through 07 | Separately validated double-window experiment | New geometry, recovery and resource accounting verified | Proposed |
| [10 Speculation](slices/10-speculation.md) | 07, 08; 09 only if speculating double windows | Defined speculation and recovery study | Dependent recovery, useful completion and tail costs accounted for | Deferred |

Slices 01 and 02 are independent after configuration is fixed. This is a
dependency fact, not a request to delegate work. Present each result before
widening scope. Gate-feedback design can be reviewed early; full compiler
integration is not a prerequisite for the memory baseline.

## Controller design boundaries

Keep four responsibilities distinguishable:

1. Cluster gap produces a score and has a charged computation cost.
2. Threshold selection identifies candidate windows, using either fixed
   calibration or a simple online target-rate tracker.
3. Admission enforces an explicitly defined quota or resource guard.
4. A slower capacity loop adjusts the allowed escalation target using observed
   work, correction age and, later, actual feedforward slack or stalls.

The first experiments have no online audits. Weak/strong agreement is not a
correctness label. Existing audit code may remain unused until its callers and
tests are migrated; retirement is a focused change after replacement behavior
is established, not a wholesale rewrite of the simulator.

Candidate and admitted fractions are different metrics. A quota denial must
not masquerade as a decoder success or silently feed a different decision into
the candidate-rate tracker. Every path that starts strong work counts toward
the applicable budget, including any future retries or audits under a stated
accounting rule.

Calibrate score thresholds offline on separate samples first. A score
percentile, an empirical failure probability, and an escalation fraction are
different quantities. Percentage confidence is optional later research; it is
not needed for the first rate controller.

## Global invariants

- Online decisions cannot read simulator truth, future scores, eventual strong
  answers, or eventual service times before those values are observable.
- Logical correctness compares observable outcomes, not equality with a unique
  physical error or correction. Different physical corrections may be valid.
- Every committed round has one logical owner. Overlapping reads do not create
  duplicate correction commits or extra throughput.
- A feedback consumer waits for all its required results; later completed
  windows do not hide an earlier unresolved dependency.
- During a feedback stall, continuing protected patches still generate their
  required syndrome work. A stalled workload must not look stable by pausing
  the arrival source without reporting it.
- Timing inputs have explicit units, inclusion boundaries, resource profiles
  and provenance. Missing timing cells fail or use a declared fallback, never
  silently borrow RelayBP timing for belief matching.
- A timing-profile-only change preserves the answers for a fixed, nonadaptive
  memory workload and routing policy. Feedback-driven routing or altered
  physical schedules may legitimately change selected work and outcomes.
- Timing-only runs report no measured logical error rate. Synthetic scores are
  labeled synthetic; replayed scores identify their source configuration.
- No controller promises a feasible deadline when weak-only service, required
  buffering, transport or a committed strong job already makes it impossible.
- Production correctness uses the agreed million-shot stopping rule. Early
  pilots, calibration samples, warmup and timing streams are labeled separately.
- Each completed slice gets a coherent local commit. No push or merge.

## Verification and comparison strategy

Use three layers of evidence:

1. Exact fixtures: deterministic arrivals and service times; a separately
   calculated queue/dependency schedule, including zero backlog and overload.
   Agree at simulator tick precision after the documented rounding rule.
2. Reference behavior: Stim's logical outcomes/detector conversion; the pinned
   beliefmatching package on identical adapted inputs; existing qLDPC/window
   referents where their boundary conventions match. Record revisions.
3. Scientific comparisons: weak alone, strong alone and fixed-20-dB switching;
   then calibrated static fractions, online fractions and resource adaptation.
   Hold hardware resources and workload assumptions explicit and constant.

Single-window forced-keep and forced-escalate fixtures must reproduce their
corresponding decoder answers. Independently windowed weak-only and strong-only
runs may supply different predecessor boundaries: do not claim per-shot union
bounds or identical-input counterfactuals unless those inputs actually match.
On a live switching trajectory, counterfactual decodes can use the same adapted
input for diagnosis, but an offline selector on that fixed trajectory is only
an optimistic diagnostic. Changing decisions can change downstream inputs.

For steady workloads, record warmup, a measurement region and drain separately.
Compare increasing stream lengths rather than declare stability from one flat
short trace. Retain generated-to-finalized coverage, upstream buffers, queue
work and resource occupancy so backpressure cannot conceal overload.

Report shot failures with binomial uncertainty, including nonzero upper bounds
for zero observed failures. Window outcomes within a shot are correlated;
use shots as the sampling unit for correctness comparisons and uncertainty.
Tail estimates need sample counts and uncertainty; a sample maximum is not a
worst-case guarantee. Correlated timing streams require independent repeats or
an appropriate block-based analysis. Keep initialization and compilation costs
separate from steady service, charging them where the proposed system needs them.

Full traces are for small diagnostics. Production retains reproducible input
identities, compact outcomes and aggregates, with sampled detailed records.
Do not store a million full syndrome vectors by default.

## Research anchors

- [Toshio et al., Decoder Switching, sections III.B, III.C and IV](https://arxiv.org/html/2510.25222v1):
  fixed-threshold correctness, double-window backlog assumptions and threshold sweeps.
- [Skoric et al., Parallel window decoding](https://arxiv.org/abs/2209.08552):
  commit-stride capacity, boundary dependencies and feedforward response.
- [Caune et al., real-time QEC, section VI](https://arxiv.org/abs/2410.05202):
  measured separation of decoding and communication/control response costs.
- [Helios, section VI](https://arxiv.org/abs/2301.08419):
  hardware-specific cycles and service-time distributions.
- [RFC 3290, Appendix A](https://www.rfc-editor.org/rfc/rfc3290.html#appendix-A):
  rate/burst accounting as a reference for a separate admission limiter.
- [TQEC logical CNOT example](https://tqec.github.io/tqec/auto_examples/cnot.html):
  candidate external lattice-surgery fixture, not proof of complete T-gate support.
- [SWIPER](https://arxiv.org/abs/2412.05115):
  later speculation comparison; select matched schedules and assumptions first.

Do not infer novelty, an optimal hardware design, or replication of an entire
paper from these limited comparisons.

## Open decisions

Resolve before slice 00 runs: memory basis, exact noise channels and terminal
noise, initial point, decoder parameters, resource topology and shared finite
tail convention. The table above proposes defaults rather than final approval.

Resolve before slice 04 production: distance/error-rate grid, calibration
sample allocation, hardware profiles, and which portion of the static frontier
is useful enough to justify adaptation. No target LER tolerance is agreed yet.

Resolve before slice 05: approximate fraction versus hard cap, quota scope,
startup credit, allowed burst and policy for denied requests. Recommended first
cap study: fixed 1% across a task or continuous stream, zero startup credit,
one stored escalation permit; report the resulting startup bias.

Resolve before slice 06: throughput-first versus correctness-constrained
operation, allowed correction age, control cadence and recovery rule. Start
with throughput-first and report correctness; do not claim an online LER bound.

Resolve before slice 08: compiler selection based on actual schedule exports
and feedforward dependencies. Do not assume a tool supports T-gate dependencies
because it can compile a CNOT or produce a Stim memory circuit.

## Scope and deferred work

Core scope ends with supplied timing, a verified sliding-window baseline,
static and online escalation fractions, causal capacity feedback, and a full
timing-only gate-feedback path. Lattice-surgery integration extends workload
realism. Double windows and speculation are separate later experiments.

Bursts, new weak decoders, neural estimators, multiple strong tiers, spatially
distributed decoding and algorithmic speed-of-light models stay deferred until
the core experiment identifies a need. Confidence-computation parallelism is
an explicit architectural variant with resource costs, not free overlap.

## Evidence ledger and current pickup

- Inspected code and existing tests while planning. No tests were executed and
  no new performance or correctness measurements were produced.
- No implementation slice has started. The next review is slice 00.
- Keep this README as the current handoff; add run evidence under this spec
  only as it is produced. Large result data belongs in normal result folders.
