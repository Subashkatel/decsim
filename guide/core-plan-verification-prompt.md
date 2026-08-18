# Prompt: independent verification of the core refactor plan

Paste the text below to a fresh agent (no prior context) in the decsim
repository. It should have read access to the repo and the web.

----------------------------------------------------------------------

You are reviewing a refactor plan for a Python discrete-event simulator
before any code is changed. Repository root: the current directory (decsim,
branch audit/evidence-first-rebuild). Read these three documents fully:

  guide/core-design-audit.md        the audit (findings with file:line)
  guide/core-rewrite-plan.md        the plan, file by file
  guide/core-layout-and-names.md    the folder layout and renames

Then read the sources the plan claims to follow, and check the plan against
what they actually say (not against your memory of them):

  https://lamb.github.io/a-philosophy-of-software-design/  (Ousterhout; at
    least chapters 4, 5, 7, 9, 10, 13, 14, 16, 18, 19)
  https://python-patterns.guide/  (Rhodes; at least the Composition Over
    Inheritance page and the Python-specific patterns)
  https://programmingisterrible.com/post/139222674273/write-code-that-is-easy-to-delete-not-easy-to  (tef)
  https://github.com/johnousterhout/aposd-vs-clean-code  (Ousterhout vs
    Martin on method length, comments, TDD)

Answer these questions, each with evidence (file:line in the repo, or a
quoted sentence from a source). Do not soften; a "no" with a reason is more
useful than a polite "yes".

1. Findings. Pick ten findings at random from guide/core-design-audit.md
   (entanglement chains, leaks, dead code, ceremony counts, lump sizes) and
   verify each against the code. Report any that are wrong or overstated.

2. Split criteria. For each of the six proposed new modules
   (decoders/strong_escalation.py, windows/window_boundaries.py,
   windows/committed_rounds.py, controller/feedback_streams.py,
   run_defaults.py, links/link_traffic_report.py): does it satisfy
   Ousterhout ch. 9 (shared information kept together, general-purpose
   separated from special-purpose, deeper interface than before) and tef
   step 6 (isolated by likelihood of change)? For each, name the interface
   the plan gives it and say whether it is deep (small interface, much
   hidden) or shallow. Flag any extraction that is splitting for length.

3. Things kept together. For each file the plan says "not broken down" or
   "leave" (message.py, links.py mechanism, syndrome_buffer.py, the
   detector_error_model package, the long methods _on_decode_done and
   Switching.on_decode_outcome), argue whether keeping it together is right
   by the same criteria. Would you split any of them? Would you merge any
   two files the plan keeps apart?

4. Rhodes. Does the plan anywhere still rely on scattered "if feature:"
   checks (his "if statement dodge"), implementation inheritance, mixins, or
   module-level mutable state? Are the no-op collaborators (NoStrongTier,
   NoFeedbackStreams) the right composition, or is there a simpler
   composition he would suggest?

5. Ousterhout ch. 19. Does the plan add getters, setters or accessor
   methods anywhere? Does it introduce design patterns by name where a plain
   object would do? Does the "tests as a lock" approach match what he says
   about unit tests and TDD?

6. Comments. Does the plan's comment rule (module header states what it
   hides and its invariants, present tense, one-line contract per public
   method, no history) match ch. 13 and 15 and the aposd-vs-clean-code
   discussion? Point to any place where the plan would rely on long names
   instead.

7. Ceremony. The owner decided to delete __post_init__ blocks and internal
   validation guards in Phase 0 and keep only six invariants (listed in the
   plan's Rules). Read those six and the deleted categories: is any deleted
   check actually guarding a real invariant whose failure would produce a
   silently wrong result rather than an exception? Name it if so.

8. Layout. Is the proposed folder tree (qpu, program, controller,
   syndrome_buffer, windows, decoders, confidence, detector_error_model,
   pauli_frame, links, observe; two levels deep) the right partition of this
   codebase, given the architecture in guide/decoder-architecture-meeting-
   2026-08-17.md and the components the meeting names? Are the renames
   justified by their stated reasons? Would you rename anything else, or
   keep any name the plan renames?

9. Behavior preservation. The lock is: tests/ (772 tests), tmp/validation/
   smoke_run.py, experiments/baseline_closed_loop.py, and Gates 1 to 5 under
   experiments/results/validation/. Is that lock sufficient to prove "what
   the code does did not change" for each phase? What would slip through
   (name at least two concrete risks, e.g. timing changes on paths the
   baseline does not exercise) and what would you add?

10. Order. Is the phase order (0 deletions and ceremony, 1 front path,
    2 decoder side, 3 window manager with boundaries and ledger before
    escalation, 4 wiring and vocabulary) the right dependency order? Would
    you do the moves-only layout commit first (the plan says yes) or last?

Deliver: a verdict per question (agree / disagree / agree with change), the
evidence, and a final list of concrete edits to the three documents. Under
2000 words. No em dashes.
