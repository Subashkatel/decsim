# Package layout and module names

Applied 2026-08-18 (owner: "apply all the change"). Companion to
guide/core-design-audit.md and guide/core-rewrite-plan.md. Differences from
the tree below: policies.py and decoders.py kept their names (review);
run_defaults is run_configuration.py (review); strong_escalation.py holds
both owners; DecoderInputStaging lives in decoders/decoder_memory_transfer.py.
2026-08-19: program/ became orchestrator/ (Khalid Fig. 2), pauli_frame
joined it, orchestrators.py is logical_measurement.py, and
magic_state_factories and round_policies moved to qpu/.

Why. Ousterhout ch. 7 (different layer, different abstraction) and tef step
6 (isolate by likelihood of change) argue for a tree that mirrors the
architecture, not the kind of code. A reader of the target diagram (notes24)
sees boxes: QPU, controller,
syndrome buffer, window manager, decoder manager and decoders, Pauli frame,
links, program orchestration, observation. The package tree should be that
diagram, two levels deep, so a box maps to a folder and a file name only has
to say what it does inside its box. Run-level vocabulary and wiring stay at
the top because every folder imports them.

Rules
- Two levels at most: decsim/<component>/<module>.py. Exceptions, stated
  here once: the six decoder backend packages under decoders/, stimcircuits/
  under qpu/, and detector_error_model/ keep their own internal structure
  (a third level) because they move as sealed wholes.
- A file keeps its name when the name is already exact (controller,
  syndrome_buffer, window_manager, decoder_manager, decoder_memory,
  pauli_frame, links, engine, message). It is renamed only when the current
  name does not say what the module does.
- Moves-only commits, one per component, each right before that
  component's refactor phase (rewrite plan section C): git mv, import paths
  updated everywhere (150 internal imports, 37 test and experiment files),
  an import-surface test, no re-exports, no shims, no content edits. History
  follows the files; every later refactor diff stays readable. Not one
  global move first: that would move code that Phase 1 deletes and put the
  whole import blast radius in one commit.
- detector_error_model/, the six decoder backend packages and soft_output/
  keep their internal structure; they only move as a whole.

## 1. Target tree

    decsim/
      engine.py                 discrete event engine (unchanged name)
      run_spec.py               RunSpec: configuration and composition root
      run_configuration.py      NEW: resolve_run_configuration, defaults and
                                compatibility rules as one resolver
      message.py                the shared vocabulary (unchanged name)
      protocols.py              seams (unchanged name)
      seeding.py                run seed binding (unchanged name)
      config.py                 TimingConfig, TICKS_PER_US (unchanged name)

      qpu/                      the box that produces syndromes every cycle
        cycle_clock.py            was qpu.py: QPUDevice, one QEC cycle clock
        syndrome_devices.py       was devices.py: TimingOnly, SyndromeBit devices
        stim_device.py            was adapters/stim_device.py: Stim and recorded data
        code_geometry.py          was codes.py: distances, rounds, syndrome bits
        layouts.py                patch layouts (unchanged name)
        stimcircuits/             surface-code circuit generators (as is)

      program/                  what the QPU is asked to run and when
        planner.py                window plan and resolved cadence (unchanged)
        orchestrators.py          feedback decisions (unchanged)
        execution_runtime.py      op DAG, readiness, completion (unchanged)
        round_policies.py         was rounds.py: FixedRounds, GateRounds, ...
        qlx_frontend.py           was frontends/qlx.py
        circuit_frontend.py       was frontends/circuit.py
        magic_state_factories.py  was factories.py

      controller/               pulses to binary, operations to commands
        controller.py             (unchanged name)
        feedback_streams.py       NEW: protected regions and feedback streams
                                  (today inside controller.py)
        policies.py               boundary policies Eager, Held and idle
                                  policies Ignore, ExtendStream,
                                  SeparateDecodeJobs (unchanged name: it
                                  holds both axes)
        syndrome_ingress.py       packing and C2B arbitration (unchanged)

      syndrome_buffer/
        syndrome_buffer.py        Buffer 0 (unchanged name)

      windows/                  the window manager box
        window_manager.py         (unchanged name)
        windowing_schemes.py      was schemes.py: sliding, parallel A/B, sandwich
        window_interactions.py    boundary and defect policy (unchanged)
        window_boundaries.py      NEW: BoundaryCourier (today window_manager 2392-2567)
        committed_rounds.py       NEW: LogicalLedger, who owns which committed
                                  rounds (today window_manager 2122-2312)
        dynamic_windows.py        real-time streams (unchanged)
        speculative_recovery.py   Eager replay (unchanged)

      decoders/                 decoder manager, memory, engine, algorithms
        decoder_manager.py        (unchanged name)
        decoder_memory.py         per-unit input memory (unchanged)
        decoder_memory_transfer.py transfer and staging into unit memory (unchanged)
        decoder_engine.py         stages around one algorithm (unchanged)
        decoders.py               router, latency models, sampled confidence
                                  (unchanged name until those three owners
                                  separate; decoder_routing would be false)
        schedulers.py             queue order (unchanged)
        weak_strong_switching.py  was switching.py: Baseline, Switching strategies
        strong_escalation.py      NEW: StrongEscalation and StrongRequestLedger
                                  (today spread over window_manager and decoder_manager)
        window_decode_results.py  was adapters/window_decode_results.py
        mwpm/                     was mwpm_decoder/        (internals unchanged)
        union_find/               was union_find_decoder/  (internals unchanged)
        tesseract/                was tesseract_decoder/   (internals unchanged)
        relay_bp/                 was relay_bp_decoder/    (internals unchanged)
        belief_matching/          was belief_matching_decoder/ (internals unchanged)
        bposd/                    was bposd_decoder/       (internals unchanged)

      confidence/               the confidence estimator box of the memo
        (was soft_output/, internals unchanged: decoder.py, cluster.py,
         complementary.py, the SoftOutputMetric protocol)

      detector_error_model/     (as is)

      pauli_frame/
        pauli_frame.py            (unchanged name)

      links/
        links.py                  vocabulary, cards, FIFO, reserve (unchanged)
        link_profiles.py          the number cards (unchanged)
        link_traffic_report.py    NEW: JSON reports (today links.py 828-972)

      observe/
        metrics.py                observers (unchanged)
        run_views.py              was views.py: frozen views of a run

## 2. Renames, with the reason each time

| today | proposed | reason |
|---|---|---|
| qpu.py | qpu/cycle_clock.py | the module is the QEC cycle clock; "qpu" is the folder |
| devices.py | qpu/syndrome_devices.py | it holds the syndrome-producing device models |
| codes.py | qpu/code_geometry.py | it answers geometry questions (distance, rounds, bits), it is not a code library |
| adapters/stim_device.py | qpu/stim_device.py | it is a device; "adapters" said nothing |
| rounds.py | program/round_policies.py | it holds the round-count policies |
| frontends/qlx.py, circuit.py | program/qlx_frontend.py, circuit_frontend.py | one word says the role |
| factories.py | program/magic_state_factories.py | "factories" collides with the design-pattern meaning |
| schemes.py | windows/windowing_schemes.py | "schemes" alone is opaque |
| switching.py | decoders/weak_strong_switching.py | it is the weak-to-strong strategy |
| adapters/window_decode_results.py | decoders/window_decode_results.py | it belongs to the decoders |
| mwpm_decoder/, union_find_decoder/, tesseract_decoder/, relay_bp_decoder/, belief_matching_decoder/, bposd_decoder/ | decoders/mwpm/, union_find/, tesseract/, relay_bp/, belief_matching/, bposd/ | the folder already says decoders; each package moves whole, internals unchanged |
| soft_output/ | confidence/ | the memo names confidence estimation as its own component; it is not a decoder |
| views.py | observe/run_views.py | it is the frozen views of one run |
| NEW protected_streams | controller/feedback_streams.py | it is the feedback-stream state of the controller |
| NEW boundaries | windows/window_boundaries.py | says whose boundaries |
| NEW logical_ledger | windows/committed_rounds.py | says what it tracks |
| NEW strong_tier | decoders/strong_escalation.py | says what happens, not which tier |
| NEW defaults | run_configuration.py | one resolver, resolve_run_configuration, not a file of default functions |
| NEW link_reports | links/link_traffic_report.py | says what it reports |

Unchanged names (already exact): engine, run_spec, message, protocols,
seeding, config, planner, orchestrators, execution_runtime, layouts,
controller, policies, syndrome_ingress, syndrome_buffer, window_manager,
window_interactions, dynamic_windows, speculative_recovery, decoder_manager,
decoder_memory, decoder_memory_transfer, decoder_engine, decoders,
schedulers, pauli_frame, links, link_profiles, metrics,
detector_error_model, stimcircuits.

Not in the tree on purpose: the reorder buffer between the syndrome buffer
and the decoders that the meeting memo defers (decoder-architecture-meeting
2026-08-17, lines 33-34). It gets a folder when it gets code; no empty
module holds its place.

## 3. What each folder's __init__ contains

Nothing but a one-line docstring naming the component. No re-exports: a
module is imported by its full path (from decsim.windows.window_manager
import WindowManager). This keeps "where is it" answerable from the import
line and avoids the shim your rule forbids.

## 4. Order

No global move. After the lock (rewrite plan Phase 0) and the dead
deletions (Phase 1), each component gets its moves-only commit immediately
before its refactor phase: qpu/, controller/, syndrome_buffer/, program/
before the front path; decoders/, confidence/ before the decoder side;
windows/ before the window manager; links/, observe/ before reports and
observers. Steps per commit: git mv; sed the import paths in decsim/,
tests/, experiments/; import-surface test; pytest and the differential
matrix; commit "layout: <component> as a folder; moves only".

## 5. Open questions for the owner

1. Folder names: qpu, program, controller, syndrome_buffer, windows,
   decoders, pauli_frame, links, observe. Alternatives considered:
   "orchestration" instead of program (rejected: orchestrators.py lives
   there and would read twice), "decoding" instead of decoders (either
   works), "reports" instead of observe.
2. Whether syndrome_ingress belongs to controller/ (it models the
   controller's decoder-bound side and C2B) or to syndrome_buffer/ (it feeds
   the buffer). Placed under controller because the meeting memo puts
   packing on the controller's decoder-bound role.
3. Whether message.py should be renamed (vocabulary.py, values.py). Kept as
   message.py: it is imported in 20 modules and the name is established.
