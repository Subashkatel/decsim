# Core rewrite plan review

Review date: 2026-08-18  
Reviewed code: `b443995` on `audit/evidence-first-rebuild`  
Documents reviewed: `core-design-audit.md`, `core-rewrite-plan.md`, `core-layout-and-names.md`

External sources checked directly:

- [A Philosophy of Software Design](https://lamb.github.io/a-philosophy-of-software-design/), chapters 4, 5, 7, 9, 10, 13, 14, 16, 18, and 19
- [Python Patterns Guide](https://python-patterns.guide/), including Composition Over Inheritance and the Python-specific patterns
- [Write code that is easy to delete, not easy to extend](https://programmingisterrible.com/post/139222674273/write-code-that-is-easy-to-delete-not-easy-to)
- [A Philosophy of Software Design vs Clean Code](https://github.com/johnousterhout/aposd-vs-clean-code)

## Overall verdict

Disagree with the plan as written

I reviewed HEAD `b443995`, all four requested sources, the three proposal documents, and the meeting memo. The architecture direction is mostly sound. Phase 0 and the claimed verification lock are not safe enough to begin implementation.

## 1. Findings, agree with corrections

I used a seeded random sample of two findings from each requested category.

| sampled finding | verification |
|---|---|
| Start-gating cycle | Correct: `execution_runtime.py:134-144` calls `controller.can_start`; `controller.py:154-168,263-276` opens the boundary and retries runtime. |
| Strong-deferral chain | Correct: `switching.py:228`, `decoder_manager.py:943-945`, `window_manager.py:1169,1733,1869,1886,1921,1957,1996`. |
| `views` private reads | Correct, and understated: `_ops`, `_selected_request_keys`, controller, engine, and device internals occur at `views.py:206,212,251,268,353,391,394`. |
| Eight `run_spec` post-installs | Wrong count: there are nine at `run_spec.py:319,349-355,406`, not eight. |
| `BurstEscalationDetector` dead | Correct: only its definition remains at `metrics.py:410`; no production/test consumer exists. |
| Provenance-as-data only serves reports | Overstated. `RequestTransferRelation` participates in core validation and request matching at `links.py:789-795,926`; `SoftOutputSource` participates in switching compatibility at `switching.py:34-47,151-176`. Only fields such as `references` look report-only. |
| Engine has 18 raises | Count correct, classification wrong. Phase 0 must not delete time and phase invariants at `engine.py:45-65,90-99,172-185`. |
| QLX has 46 raises | Count correct, classification badly wrong. This is an input boundary; detector routing, resource-chain, and overlap checks at `frontends/qlx.py:140-234,407-518` are semantic validation, not internal ceremony. |
| Protected-region lump about 200 lines | Fair: the cited controller ranges total about 213 lines and share stream state. |
| Multi-fragment lump about 60 lines | Fair: the cited ranges in `syndrome_ingress.py:282-289`, `syndrome_buffer.py:138-170`, and `qpu.py:143-168` total about 62 lines. |

## 2. Split criteria, agree with changes

Ousterhout ch. 9 says, “Subdivision usually results in more interfaces, and every new interface adds complexity,” and concludes to choose “the best information hiding, the fewest dependencies, and the deepest interfaces.” Tef says to isolate “difficult design decisions or design decisions which are likely to change.”

| proposed module and planned interface | verdict |
|---|---|
| `strong_escalation`: two six-method interfaces at `core-rewrite-plan.md:117-125` | Medium-depth, acceptable as one deletable feature, but it is not one 3-to-6-method module. Document its two owners explicitly. Not length-driven. |
| `window_boundaries`: `send, merge_available, hold, take_held, invalidate` at `:127-132` | Deep. Five methods hide versioning, DD scheduling, late delivery, and held state. Keep. |
| `committed_rounds`: five methods at `:134-138` | Mostly deep. Remove or generalize `slab_candidate`, which leaks strong-tier policy into the general ledger. |
| `feedback_streams`: six methods at `:167-171` | Deep enough. It hides several maps, cadence boundaries, activation, and sealing; it changes with feedback policy. |
| `run_defaults`: eight default functions plus unrelated compatibility checks at `:181-190` | Shallow grab bag. It hides no coherent state and changes for unrelated reasons. Replace it with a deep `resolve_run_configuration(...) -> ResolvedRunConfiguration`, or keep rules with their owners. This extraction is decluttering `run_spec`, not modularity. |
| `link_traffic_report`: three functions at `:192-195` | Shallow but justified by ch. 9’s general-purpose versus special-purpose rule. Reporting schema changes independently. It must consume a frozen link snapshot, not link internals. |

## 3. Things kept together, agree with change

Keep `links.py` mechanism, `syndrome_buffer.py`, and `detector_error_model/`. Their state and rules are shared, and the DEM package already declares a one-way L0-L3 lattice at `detector_error_model/__init__.py:1-38`. Keep `_on_decode_done` and `Switching.on_decode_outcome` intact after collaborators leave; today they are coherent orderings at `decoder_manager.py:541-638` and `switching.py:217-239`. Keep `message.py` for this refactor, but “imported jointly” is not proof of cohesion; revisit only after ownership moves stabilize. I would merge `run_defaults.py` back into `run_spec.py` unless it becomes the single deep resolver above. I would not merge boundaries with committed rounds: delivery/version knowledge differs from interval/logical ownership.

## 4. Rhodes, disagree

Rhodes says, “Favor object composition over class inheritance,” and explains that an `if` forest loses locality and deletability. The plan removes scheme implementation inheritance and proposes no mixins, which is good. `NoStrongTier` and `NoFeedbackStreams` are appropriate Null Objects because they remove repeated feature checks. But the plan contradicts itself with “strong ledger or None” and “ProtectedStreamRegions or None” at `core-rewrite-plan.md:148,165`. Use the Null Objects consistently. For a single hook, inject a no-op callable rather than a class. Static tables such as mutable `_PATH_RULES` at `links.py:238` and `_KIND_BY_NAME` at `frontends/qlx.py:45` remain module-level mutable objects; make them immutable mappings or explicitly exclude constant tables from the rule.

## 5. Chapter 19, agree with change

No conventional setters are proposed. However, `assembling_fragment_count` (`core-rewrite-plan.md:174-176`) is a shallow accessor exposing slot representation; return the needed count in `FragmentAdmission` instead. `holder_of`, `binding_for`, and typed snapshots are domain queries, but a generic snapshot that mirrors every field is merely a bulk getter. Specify purpose-built frozen view types. Pattern naming is restrained.

“Tests, particularly unit tests, … facilitate refactoring” supports tests as a lock. It does not support the claim that this lock proves identical behavior. The plan correctly does not use TDD as the design driver; ch. 19 says TDD “focuses attention on getting specific features working, rather than finding the best design.”

## 6. Comments, disagree

Module abstraction, invariants, present tense, and no duplicated implementation are right. “Every public method has a one-line contract” (`core-rewrite-plan.md:24-27`) is not. Ch. 13 requires behavior, arguments, return value, side effects, exceptions, and preconditions. Ch. 16 also says subtle motivation needed by future developers belongs in code, not merely commit history. Therefore “no history” should mean no changelog narration, not deletion of why a nonobvious rule exists. Names such as `assembling_fragment_count`, `release_service_members`, and `observables_for_interval` do not replace those contracts. The Ousterhout/Martin exchange explicitly records Ousterhout’s preference for shorter names supplemented by comments.

## 7. Ceremony, disagree strongly

At least two omitted invariants can silently corrupt results:

* `Engine.schedule` requires an exact, nonnegative integer delay (`engine.py:94-99`) and `run` requires nondecreasing event time (`:183-185`). Delete both and simulation time can move backward or become fractional.
* `RunSeedPathSegment.kind` is restricted at `message.py:93-95`; without it, an unknown kind falls through to the string-key encoding at `:99-110`, creating silent canonical-path collisions and correlated seeds.

QLX semantic checks must also remain at the external boundary. Replace blanket category deletion with an enumerated guard-by-guard decision.

## 8. Layout, agree with changes

The component partition mostly matches the meeting: controller has two roles (`decoder-architecture-meeting-2026-08-17.md:10-13`), buffer is separate (`:14-17`), confidence is independent (`:22-24`), and results land in the frame (`:35-37`). Fix four points:

1. The “two levels” rule conflicts with preserved backend and `stimcircuits` subpackages. State the exception.
2. `policies.py` contains boundary policies `Eager/Held` at `policies.py:12-27` and idle policies at `:31-57`; renaming all of it `idle_policies.py` is false. Split those axes or keep `policies.py`.
3. `decoder_routing.py` is too narrow: current `decoders.py` also owns latency and sampled-confidence models (`decoders.py:51-156,283-380`). Keep `decoders.py` until those owners move, or use `decoder_models.py`.
4. Record the meeting’s deferred reorder buffer (`meeting:33-34`) explicitly; do not create an empty module.

Other stated renames are justified. I would add no others now.

## 9. Behavior preservation, disagree

The 772-test collection is useful, but static markdown Gates 1-5 are evidence artifacts, not executable gates. The baseline omits strong cancellation/escalation, dynamic/protected streams, finite buffer exhaustion, multi-fragment packing, and many link paths. Those behaviors can change while the lock stays green. Also byte-identical baseline output is impossible because `baseline_closed_loop.py:129-131,216` records wall time and its measured decoder row varies by host.

Add executable differential traces, old versus new, for: strong cancel/batch/reentry; protected and dynamic streams; finite buffer and decoder memory; multi-fragment timeout/drop; and every link path. Compare ordered events, ticks, reservations, outcomes, and snapshots. Add import-surface tests for moves. Use tolerances and distribution records for wall time, not byte equality.

## 10. Order, disagree

Phase 3’s boundaries and ledger before escalation is right; front path before decoder side is reasonable. The rest is wrong. Phase 0 is not zero-risk, and constructor/result contracts needed by Phases 1-3 should not wait until Phase 4.

Use: (0) executable characterization lock and guard inventory; (1) proven dead deletions; (2) result callbacks, constructor injection, and resolved configuration; (3) front path; (4) decoder service and strong ledger; (5) boundary/ledger/escalation; (6) reports, vocabulary, and observers. Do neither one giant move first nor last. After safe deletion, move each component in a moves-only commit immediately before refactoring that component. This avoids moving dead code and limits each import blast radius.

## Concrete document edits

### `core-design-audit.md`

- Correct eight post-construction installs to nine.
- Narrow the provenance-only claim to the fields actually used only by reports.
- Remove “zero-risk” from Phase 0.
- Correct the claim that all listed engine and QLX checks are internal ceremony.

### `core-rewrite-plan.md`

- Replace blanket Phase 0 guard deletion with a reviewed keep/delete table covering engine time, seed encoding, QLX boundary validation, and the six already-listed invariants.
- Replace `run_defaults` with a resolved-configuration interface, or keep the unrelated rules with their owners.
- Remove `slab_candidate` from the general committed-round ledger.
- Eliminate both `or None` contradictions and use the named Null Objects consistently.
- Specify purpose-built typed observer snapshots rather than generic state dumps.
- Expand the comment rule beyond one line and preserve nonobvious rationale.
- Replace byte-identical sweep and gate claims with executable differential matrices and performance tolerances.
- Move result callbacks and constructor injection before the feature extractions that depend on them.

### `core-layout-and-names.md`

- State the depth-rule exceptions for backend and `stimcircuits` packages.
- Do not rename mixed boundary and idle policies as `idle_policies.py`.
- Do not call the mixed routing, latency, and confidence module `decoder_routing.py` until those owners separate.
- Record the meeting’s deferred reorder buffer without creating an empty module.
- Replace the global moves-first commit with safe deletion followed by per-component moves immediately before each component’s refactor.
