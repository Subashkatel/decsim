# How decsim is written

This is the contract for the rewrite and for every line written after it.
The plan that schedules the work is in the sandbox at docs/rewrite/PLAN.md.
The behavior gate that every change must pass is described there too.

## The one sentence

A reader opens any file and understands it without decoding a name,
unpacking a line, or guessing what happens somewhere else.

## Rule 1. One line does one thing

A line calls one function, or reads one attribute, or does one arithmetic
step, or makes one decision. When a value is needed twice or is the result
of a step, it gets a name on its own line.

Not this:

    return PauliFrame(engine, commit_ticks=self.commit_ticks())
    record = frame.snapshot().records[0]
    first = ticks[0] if ticks else None

This:

    commit_ticks = self.commit_ticks()
    return PauliFrame(engine, commit_ticks=commit_ticks)

    snapshot = frame.snapshot()
    record = snapshot.records[0]

    first = None
    if ticks:
        first = ticks[0]

`tools/check_one_action.py` enforces this. It reports a call used as an
argument of another call, an attribute or index taken from a call's
result, an inline conditional, a comprehension that both filters and
calls, an `and` or `or` with more than two operands, and a walrus. Small
built-ins that read as part of a phrase are allowed inside a call: len,
str, tuple, sorted, enumerate, zip, min, max, isinstance and their kin,
listed at the top of the script. Calls inside f-strings and lambdas are
allowed, since a log line or a scheduled action is one thing.

## Rule 2. Names are full words that say what the thing is

No acronyms. No abbreviations. No single letters, except inside a formula
that matches a paper, where `p` and `d` may stay because the reader has
the paper open.

    wm            -> window_manager
    dem           -> detector_error_model
    mwpm          -> minimum_weight_matching
    seq           -> sequence_number
    res, val, tmp -> the thing's real name in that place
    cfg           -> settings, or the specific settings object
    us(x)         -> microseconds_to_ticks(x)
    fmt(ticks)    -> format_ticks(ticks)

A method is named for what it does, not for the phase it belongs to:
`release_waiters`, not `integrate`; `decisions_for`, not `on_result`.
A collection is named for what it holds, keyed by what looks it up:
`waiting_by_blocker`, `pending_by_window`, `windows_by_stream`.

Link paths and yaml keys are renamed to plain words too. The owner chooses
the final names; the current list is on the sample page and in
docs/rewrite/PLAN.md. So far: `qc` becomes `qpu_to_controller`, and every
path is named source to destination by component.

A glossary in the docs maps each plain name to the paper's term, for a
reader arriving from the literature.

## Rule 3. Comments say why, in the present tense

A comment explains a decision, a unit, an invariant, or a source. It never
narrates the syntax below it, never tells a defect story, never cites a
finding number. If a workaround needs a paragraph to justify it, the code
is wrong; fix the code.

Every module starts with a docstring that says, in plain words, what the
component is and what it does, and names the source it follows. Two to
four short paragraphs. Docstrings follow the Google style: a one-line
summary, a blank line, then the detail; Args, Returns and Raises sections
only when the signature does not already say it.

## Rule 4. Checks live where input enters

A yaml loader, a device reading, a file being read: checked once, loudly,
with a message that reads as a sentence. Inside the machine a component
trusts its collaborators. A check that no input can trigger is deleted,
together with the test that forced the state by hand. A check that guards
an entry point stays.

## Rule 5. No compatibility layer

A rename is a rename. Nothing keeps an old name alive: no alias, no
wrapper, no deprecated path. Callers and tests move in the same commit.

## Rule 6. One reason per component

A class owns one job. Its state is listed in `__init__` and fits on one
hand. Its public methods read top to bottom, in the order a reader meets
them. Private helpers come after the public methods. Records are grouped
next to the component that owns them.

The target shape, taken from gem5: a component owns its settings and its
state, exposes a few named methods other components call, and is wired by
one root object. A mechanical rewrite keeps a file's current shape and
only makes it read well; a structural change moves toward this shape and
is planned and gated on its own.

## Rule 7. Google Python style, enforced

The Google Python Style Guide applies wherever the rules above are
silent. `ruff` enforces it with the settings in `pyproject.toml`: 80
columns, pycodestyle, pyflakes, pep8-naming, Google docstrings, import
by module. Run both checks before every commit:

    .venv/bin/python -m ruff check decsim experiments tests
    .venv/bin/python tools/check_one_action.py decsim experiments tests

## Tests

New tests are written from sources, not copied from the old ones. A test
file's docstring names the paper, the reference code, or the closed form
it checks against. A test name is a sentence that states the behavior:
`test_a_second_correction_for_a_window_is_refused`. The old tests keep
running until the new ones cover the same behavior, then they are deleted
in one commit per test directory.

## Slices and commits

Every commit is one of two kinds, never both:

- mechanical: a file reads better, its shape is unchanged, the gate passes
  unchanged;
- structural: a split, a new component, a new wiring; planned, reviewed,
  and gated on its own.

A bug found on the way gets its own commit with a test and a referent. The
golden is regenerated only under a design note.

Before a slice commit: gate green, component harnesses exact, old and new
tests green, ruff clean, one-action checker clean on the touched files.
The file checklist in docs/rewrite/PLAN.md moves in the same commit.

## Review

Each slice has an implementer, a reviewer who sees only the diff and is
told to assume it is wrong, and a fixer. The reviewer's checklist is this
file. A file the owner finds unreadable is not done.

## Where the tree stands today

At decsim aad5747 the one-action checker reports 1,170 lines in 82 files:
604 nested calls, 261 call chains, 258 inline conditionals, 40 long
conditions, 7 busy comprehensions. That count goes to zero file by file.
