# How decsim is written

This is the contract for the rewrite and for every line written after it.
The plan that schedules the work is in the sandbox at docs/rewrite/PLAN.md.
The behavior gate every change must pass is described there.

## The one sentence

A reader opens any file and understands it without decoding a name,
unpacking a line, or guessing what happens somewhere else.

## Rule 1. One line does one thing

A line calls one function, or does one arithmetic step, or makes one
decision. A line may walk attributes and indexes on a named value
(`snapshot.records[0]`, `record.window.tier`); it may not take anything
from a call's result. A value that is needed twice, or that is the result
of a step, gets a name on its own line.

Not this:

    return PauliFrame(engine, commit_ticks=self.commit_ticks())
    record = frame.snapshot().records[0]
    first = ticks[0] if ticks else None
    record = Record(due=self.now + delay)

This:

    commit_ticks = self.commit_ticks()
    return PauliFrame(engine, commit_ticks=commit_ticks)

    snapshot = frame.snapshot()
    record = snapshot.records[0]

    first = None
    if ticks:
        first = ticks[0]

    due = self.now + delay
    record = Record(due=due)

When the value that would be hoisted has no honest name, keep the call in
the condition: `if not math.isfinite(cost) or cost < 0:` is one decision
and needs no `cost_is_a_number`.

`tools/check_one_action.py` enforces the rule. It reports: a call inside
another call's arguments; arithmetic, a comparison or a boolean inside a
call's arguments; an attribute, index or call taken from a call's result;
an inline conditional; an `and` or `or` with more than two operands, or
`and` and `or` mixed in one expression; a comprehension whose element
does two things, or that calls while it filters; a lambda that is more
than one call on plain values or one f-string; a call inside an f-string;
a walrus; a function longer than 40 lines; blocks nested deeper than two;
a class whose `__init__` sets more than six attributes.

These built-ins may appear inside a call's arguments or an f-string,
because they read as part of the phrase: len, str, repr, int, float, bool,
tuple, list, set, dict, sorted, range, enumerate, zip, min, max, abs,
isinstance, next, iter, reversed, any, all, sum. Nothing else, including
`math.isfinite` and the project's own helpers. Adding a name to that list
is a structural commit with a reason.

A lambda passed to `schedule` or `log_io` is one call on plain values, or
one f-string of names and attribute walks. Anything more becomes a named
method. A lambda is never built inside a loop.

A function longer than 40 lines is split. A block nested deeper than two
levels is hoisted into a named method.

## Rule 2. Names are full words that say what the thing is

No abbreviations. No acronyms except these, which are words in this field
and stay: qpu, id, io, xor, yaml. The glossary in the sandbox at
docs/architecture/TERMINOLOGY.md maps every plain name to the exact term
the literature uses, so `minimum_weight_perfect_matching` is listed
beside "MWPM".

    wm            -> window_manager
    dem           -> detector_error_model
    mwpm          -> minimum_weight_perfect_matching
    seq           -> sequence_number
    res, val, tmp -> the thing's real name in that place
    cfg           -> the specific settings object
    us(x)         -> microseconds_to_ticks(x)
    fmt(ticks)    -> format_ticks(ticks)
    TICKS_PER_US  -> TICKS_PER_MICROSECOND

A single-letter name is allowed only inside a function whose docstring
cites the paper and the equation it implements; `p` and `d` may then stay.
An uppercase symbol from a paper is spelled out (`qubit_count`, never
bare `N`), which also keeps pep8-naming quiet.

A method is named for what it does, not for the phase it belongs to:
`release_waiters`, not `integrate`; `decisions_for`, not `on_result`. A
collection is named for what it holds, keyed by what looks it up:
`waiting_by_blocker`, `pending_by_window`, `windows_by_stream`. A boolean
starts with `is_`, `has_`, `can_`, or reads as a question (`verbose` and
`idle` are fine; `flag` is not). A duration field ends in `_microseconds`
or `_ticks`; a count ends in `_count`.

Link paths and yaml keys are renamed to plain words too. The owner chooses
the final names; the list lives on the sample page and in PLAN.md. So
far, accepted by the owner on 2026-09-02: qpu_to_controller,
controller_to_weak_buffer, controller_to_strong_buffer,
weak_buffer_to_weak_decoder, strong_buffer_to_strong_decoder,
weak_decoder_to_strong_decoder, decoder_to_decoder, weak_decoder_to_frame,
strong_decoder_to_frame, frame_to_controller, controller_to_qpu; and the
config keys readout_to_bits_cycles, decision_to_pulse_cycles,
packing_cycles_per_round, weak_buffer_rounds, strong_buffer_rounds,
packing_rounds_in_flight, unit_memory_rounds, write_cycles,
setup_cycles_per_transfer, circuit, log_component_io, check_windows_with,
decode_path. They land in the surface commit of the links slice.

## Rule 3. Comments say why, in the present tense

A comment explains a decision, a unit, an invariant, or a source. It never
narrates the syntax below it, never tells a defect story, never cites a
finding number. If a workaround needs a paragraph to justify it, the code
is wrong; fix the code.

Every module starts with a docstring that says, in plain words, what the
component is and what it does. When the component follows a paper or a
reference implementation, the docstring names it. Docstrings follow the
Google style: a one-line summary, a blank line, then the detail; Args,
Returns and Raises only when the signature does not already say it.
Review checks this; no tool can.

## Rule 4. Checks live where input enters, and where a contract would break

An entry point is where something from outside the machine arrives: a
yaml file, a data file, a device reading. It is checked once, loudly, with
a message that reads as a sentence, and raises ValueError.

Inside the machine a component trusts what a collaborator already
guaranteed and never re-checks it. It does refuse a call that breaks its
own contract (a negative delay, a second correction for the same window,
a duplicate metric name), because a wrong caller is a bug and a silent
wrong answer is worse than a loud stop; that raises RuntimeError. A check
that no caller can trigger is deleted, together with the test that forced
the state by hand.

## Rule 5. No compatibility layer

A rename is a rename. Nothing keeps an old name alive: no alias, no
wrapper, no deprecated path. Callers move in the same commit. Old tests
are edited for renames only, never for behavior, until they are deleted.

Dead code goes with the rewrite of the file that holds it. When a file is
rewritten, every function, class, branch, field and parameter that
nothing calls or reads is deleted, and the commit message lists what was
removed. Git remembers it; nothing is kept "in case". A decoder or a
study knob that is unused today but has research value is not dead
code; it stays, with the reason on its checklist row.

## Rule 6. One reason per component

A class owns one job: its docstring's first sentence says it without an
"and". Its state is set in `__init__` and is at most six attributes; the
checker reports more as wide state, which a mechanical slice records in
the checklist for the structural slice that splits the class. Every
attribute has its final value when `__init__` returns; a collaborator
arrives through the constructor. Public methods read top to bottom in the
order a reader meets them; private classes and functions come after every
public one in the module and carry a leading underscore. A package's
`__init__.py` holds the package docstring and nothing else.

A record used by one component lives next to it. A record shared by two
or more components lives in `message.py`.

The target shape, taken from gem5: a component owns its settings and its
state, exposes a few named methods other components call, and is wired by
one root object. A mechanical rewrite keeps a file's current shape,
including a `connect` step that exists today, and only makes it read
well; a structural change moves toward the target shape and is planned
and gated on its own.

Every public signature is annotated. An `engine` parameter is typed
`Engine`. `Any` is used only for an opaque identity, with a comment that
says so.

## Rule 7. Google Python style, enforced

The Google Python Style Guide applies wherever the rules above are
silent. Import modules, not names: `import decsim.message as message`,
then `message.Decision`; the exceptions are `typing`, `dataclasses`,
`collections.abc` and `enum`. No module uses `from __future__ import
annotations`; `Optional[X]` is written out because the package runs on
Python 3.9.

`ruff` enforces the rest with the settings in `pyproject.toml`: 80
columns, pycodestyle (E, W), pyflakes (F), pep8-naming (N), Google
docstrings (D), bugbear (B), import order (I), pyupgrade (UP) without the
`X | None` rules, unused arguments (ARG). The `SIM` family is off because
it asks for inline conditionals and merged conditions, the opposite of
rule 1. `ruff format` runs before `ruff check` and settles line breaks
and trailing commas.

Run, from the repo root, before every commit:

    PYTHONPATH=.pydeps .venv/bin/python -m ruff format decsim experiments tests tools
    PYTHONPATH=.pydeps .venv/bin/python -m ruff check decsim experiments tests tools
    .venv/bin/python tools/check_one_action.py decsim experiments tests tools

`tools/check.sh` runs all three. The ruff binary lives at
`.pydeps/bin/ruff`; if `-m ruff` reports RuffNotFound, copy it from the
ruff wheel's `data/scripts/ruff` into that folder.

## Tests

New tests are written from sources, not copied from the old ones. A test
file's docstring names the paper, the reference code, or the closed form
it checks against; ruff keeps the module docstring required under
`tests/`. A test name is a sentence that states the behavior:
`test_a_second_correction_for_a_window_is_refused`. One new test file per
module, at `tests/<component>/test_<module>.py`.

A test checks one behavior through the public surface, and checks the
state it leaves, not the calls it made. The values that matter sit
inside the test, even when that repeats a line; a helper builds only
what the test does not care about. A test has no branches; a loop is
allowed only in a property test that runs many random programs against
an oracle, and the test says so in its name. A test that must change
when the implementation changes, but the behavior does not, is a
brittle test and is rewritten. The old tests keep
running until the new ones cover the same behavior, then they are deleted
in one commit per test directory.

Log lines are part of the pinned behavior: the gate hashes the full log
of every strict point. A change to a log line's text or source name is a
surface change (below), never part of a mechanical slice.

## Slices and commits

Every commit is one of three kinds, never two:

- mechanical: a file reads better, its shape is unchanged, the gate passes
  unchanged;
- structural: a split, a new component, a new wiring; planned, reviewed,
  and gated on its own;
- surface: a config key, a link path name, or a log line changes;
  reference.yaml, the parameter reference, the frozen suite configs and
  the golden move in the same commit, under a design note.

A bug found on the way gets its own commit with a test and a referent. The
golden is regenerated only under a design note.

Before a slice commit: gate green, component harnesses exact, old and new
tests green, `tools/check.sh` clean on the touched files. The file
checklist in docs/rewrite/PLAN.md moves in the same commit.

## Review

Each slice has one implementer and at least two adversarial reviewers,
each in its own context, each seeing only the diff and told to assume it
is wrong (owner directive 2026-09-02). A reviewer's only job is to find
bugs and reasons the code does not work; a reviewer never implements,
and the implementer never reviews. Every finding goes back to the
implementer, who fixes it and resubmits; a slice is done when both
reviewers find nothing. Beyond bugs, each reviewer answers this list
with yes or no for every touched file:

1. Does every line do one thing, and does `tools/check.sh` pass?
2. Is every name a full word that says what the thing is, with no
   abbreviation and no acronym outside qpu, id, io, xor, yaml?
3. Does every comment say why, and does every module docstring say what
   the component is in plain words?
4. Is every check either at an entry point or a contract of the method it
   sits in, and is nothing re-checked?
5. Did every rename move its callers and its tests, with no alias left?
6. Does every class own one job, with at most six attributes, all final
   at the end of `__init__`, public before private?
7. Are imports by module, signatures annotated, and Optional written out?
8. Is the commit one kind only, and does the checklist row move with it?
9. Is the gate green, are the harnesses exact, and are the tests green?

A file the owner finds unreadable is not done.

## Where these rules come from

Google's own rules for rules (Software Engineering at Google, chapter 8,
on disk under the sandbox tmp/resources/style_guides): a rule must pull
its weight, favor the reader over the writer, keep the code consistent,
and be enforced by a tool wherever a tool can. Chapter 20's bar for a
checker is a false-positive rate under ten percent; when the one-action
checker flags a line that reads well, the fix is to the checker's rules,
recorded here, never a per-file exception. Chapter 22's rule for a
large change: past a few hundred edits, write the tool that makes the
edit (the tick-helper rename was one), and add a check so the old form
cannot come back. Chapter 12 gives the test rules above.

## Where the tree stands

Recorded when the tooling landed, at decsim aad5747, by running the three
commands above over decsim, experiments and tests. The numbers are
findings, not lines; one line can carry several. The current numbers are
in docs/rewrite/PLAN.md, section 3, and go to zero file by file.
