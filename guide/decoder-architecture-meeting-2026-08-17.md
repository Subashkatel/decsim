# Architecture walkthrough decisions, 2026-08-17

Source: owner meeting with senior collaborator, relayed into the shared checkout
by the owner's architecture-stream session on 2026-08-17. STATUS: CONFIRMED by
the owner in the primary session on 2026-08-17; queue entries Q-055 through
Q-058 are binding owner directives.
Decisions separate baseline shape from deferred optimizations. Route into the
cleanup pipeline so pending module reviews implement the agreed shape
(syndrome_buffer 18, controller 23, syndrome_ingress 25, decoder_manager 26,
window_manager 27, factories 28).

## Component decisions

1. Controller has two internal roles, one component: QPU-bound (ops to pulses)
   and decoder-bound (pulses to binary). Conversion only; the syndrome buffer is
   NOT inside the controller.
2. Syndrome buffer is its own component, FIFO for baseline, may packetize.
   Separate so controller-to-buffer bandwidth and latency are measurable axes.
3. TWO syndrome buffers, fridge (0) and room-temperature (1), BOTH fed directly
   in parallel by the controller, not chained. Identical contents in baseline;
   buffer 1 may drop; infinite capacity initially.
4. Data movement to decoders is PUSH (DMA-style by the manager side), never
   ASIC pull. A trigger mechanism (doorbell or polling) must be modeled; which
   one is an open research question.
5. Confidence estimation is a named component (complementary-gap second decode)
   running IN PARALLEL with the weak decode, both fed from buffer 0. The
   G-threshold is an explicit input, offline for baseline. The escalation
   signal originates from the confidence estimator, not the window manager.
6. Window management: one module; weak and strong managed as separate
   components; ZERO weak/strong coordination in baseline; same window size to
   both. Duplicate strong corrections are dropped at the reorder buffer.
   Coordination is added only if measured drop counts later justify it.
7. Weak and strong schedulers are separate classes (shared base allowed) under
   decoder_manager. Deadline-aware strong scheduling is a later optimization.
8. Baseline escalation is escalated-window decode ONLY. The partial/parallel
   weak path is OFF in baseline. RESOLVED by the owner 2026-08-17: this means
   toggle-off in baseline with the capability PRESERVED (never deleted);
   ORIENTATION.md's serial/parallel escalation sweep axis stays intact.
9. Reorder buffer modeled on classical CPU ROB literature; tolerates
   duplicates by dropping.
10. Results land in the Pauli frame; the controller unblocks T-gate-blocked
    operations from the result frame (aligns with Q-054).
11. Magic-state factory started by the controller on T gates, same QPU,
    developed later.

## Baseline experiment (Friday walkthrough target)

Single weak ASIC; strong decoder present but ~10x slower and unused
(G-threshold set so nothing escalates); infinite buffers; full flow from
frontend DAG through the Pauli frame and controller unblock. Goal: convince
ourselves every component is right before escalation exists.

## Open questions (research, not improvisation)

- ASIC trigger mechanism: doorbell vs polling, priced from literature.
- Offline G-threshold calibration.
- Buffer pointer advancement so no round is skipped (possible C++ buffer later).
- ROB sizing and retirement policy from literature.

## Target diagram

The canonical target-architecture diagram is
tmp/references/ref_arch_diagram/arch_target_from_notes24.svg (supersedes
notes23). It is a text-readable SVG; consult it for the component layout,
box inputs/outputs, and which edges are baseline (solid/dotted) versus
future (sparse-dashed).

## Diagram rules

Component boxes with explicit inputs and outputs; no flow-chart diamonds.
Color only with a stated reason. Dotted lines are signals, solid lines are
data. Two internal controller boxes.
