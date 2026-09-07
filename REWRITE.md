# How decsim is written

This is the contract for the rewrite and for every line written after it.
The plan that schedules the work is in the sandbox at docs/rewrite/PLAN.md.
The behavior gate every change must pass is described there.

## The target

Three things, in this order, and a file is done only when all three hold.

1. A reader opens any file and understands it without decoding a name,
   unpacking a line, or guessing what happens somewhere else.
2. A reader follows one readout through the whole machine by reading
   named calls: the QPU emits it, the controller receives it, the buffer
   holds it, the window manager closes a window, the decoder manager
   schedules a decode, the decoder returns a correction, the frame
   commits it, the frame releases the controller, the controller
   instructs the QPU. Each handoff is one method with a plain name, and
   the reader can say, at every step, which component holds the data and
   where it is stored.
3. A new component plugs in. A better QPU model, another decoder, a
   different buffer or link, arrives as one class that implements the
   port the machine already has, is named in the yaml, and nothing else
   changes. That is gem5's shape: a component owns its settings and its
   state, talks to other components only through named ports, and one
   root object wires them.

The old code was correct. The rewrite keeps it correct, proven by the
gate at every commit, and makes it read, flow and plug in as above. It
is not a rewrite of what decsim computes.

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

This rule makes dense code longer. That is intended. A file that grows
because its lines were unpacked is right; a file that grows for the
reasons listed under "Size" is wrong.

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

A function longer than 40 lines is split where its meaning splits: each
piece has a name a reader would look for. A helper that exists only to
get a function under the cap, with a name like `_second_half`, is worse
than the long function; then the function stays long and the checklist
row says why. The 40 comes from Google's Python and C++ guides, where it
is a prompt to think, not a limit; the checker lists functions over it
and classes over six attributes as reports, not failures. The six is
ours. A block nested deeper than two levels is hoisted into a named
method (LLVM's early exits and predicate functions); that one is a
failure.

## Rule 2. Names are full words that say what the thing is

No abbreviations. No acronyms except these, which are words in this field
and stay: qpu, id, io, xor, yaml, json. The glossary in the sandbox at
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

Link paths and yaml keys are renamed to plain words too. Accepted by the
owner on 2026-09-02: qpu_to_controller, controller_to_weak_buffer,
controller_to_strong_buffer, weak_buffer_to_weak_decoder,
strong_buffer_to_strong_decoder, weak_decoder_to_strong_decoder,
decoder_to_decoder, weak_decoder_to_frame, strong_decoder_to_frame,
frame_to_controller, controller_to_qpu; and the config keys
readout_to_bits_cycles, decision_to_pulse_cycles,
packing_cycles_per_round, weak_buffer_rounds, strong_buffer_rounds,
packing_rounds_in_flight, unit_memory_rounds, write_cycles,
setup_cycles_per_transfer, circuit, log_component_io, check_windows_with,
decode_path. They land in the surface commit of the links slice.

## Rule 3. Comments say why, in the present tense

A comment explains a decision, a unit, an invariant, or a source. It never
narrates the syntax below it, never tells a defect story, never cites a
finding number. If a workaround needs a paragraph to justify it, the code
is wrong; fix the code.

Every module starts with a docstring that says, in one or two sentences,
what the component is and what it does. When the component follows a
paper or a reference implementation, the docstring names it, with the
section or the file. A docstring never repeats what the code below it
already says; a function whose name and signature say everything has no
docstring at all. Google style for the rest: a one-line summary, a blank
line, then the detail; Args, Returns and Raises only when the signature
does not already say it. Review checks this; no tool can.

## Rule 4. Only the checks that are necessary

A check is necessary in exactly two places.

Where input enters decsim: a yaml file, a call on the front, a Stim
circuit, a QLX program, a data file, a device reading. That input is
checked once, at that boundary, loudly, with a message that reads as a
sentence, and raises ValueError.

Where a silent wrong answer would corrupt a result: a component refuses a
call that breaks its own contract (a negative delay, a second correction
for the same window, a physical fault kept past its own component) and
raises RuntimeError, because a wrong caller is a bug and a loud stop is
better than a wrong number. This is gem5's split: `fatal` for the
user's mistake, `panic` for the machine's.

An invariant a component keeps for itself (a queue that is never empty
here, a tick that never goes backwards) is a bare `assert` with a
message. It costs one line, it carries no test, and no reviewer asks
for one; LLVM's "assert liberally" and Google's rule that an assert is
never application logic both apply.

Everything else is not necessary and is deleted, with its test. Inside
the machine a function trusts what its callers send. A function in a
package that only decsim's own components call refuses nothing that
those components cannot send; whether a hand-written call could break it
is not a reason for a check. A check no caller can trigger is deleted
together with the test that forced the state by hand. A reviewer does
not ask for a check on an input that no yaml, no front call and no
runtime path produces; such a finding is out of scope, not a defect.

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
or more components lives in `decsim/records/`, one module per record family.

Every public signature is annotated. An `engine` parameter is typed
`Engine`. `Any` is used only for an opaque identity, with a comment that
says so.

## Rule 7. Components plug in through ports

This is the target shape. The ownership and the name table are gem5's
(a SimObject's Python class is its params; `allClasses` maps a name to
a class); the table plus one abstract class per pluggable part is
sinter's (`BUILT_IN_DECODERS` and `Decoder`); the wiring is by
constructor, not gem5's late port bind, because Python needs no second
step. It applies to every component the structural slices produce.

A port is a small Protocol in `decsim/ports.py` (today's
`decsim/protocols.py`, renamed and trimmed under rule 5) with the
methods one component needs from another, named for what they do:
`SyndromeSource.begin_operation`, `SyndromeSource.round_payloads`,
`Decoder.decode`, `Link.send`, `Frame.commit_correction`. A component
depends on ports, never on another component's class.

Which ports exist is decided by the pipeline, not by taste: one port per
neighbour in the list under "The target", point 2, one method per
handoff, and the record that crosses it defined in `decsim/records/`.
Observation (metrics, ledgers, the traffic report) is reached through
callbacks a component fires, never through a port, so a component can
run with no observer at all. The port file is therefore the map of the
pipeline, and a reader who wants to follow a readout starts there.

One root object, `Machine`, builds every component from its settings and
wires them by constructor; no component builds or looks up another. The
yaml has one section per component, each section builds one settings
dataclass, and a pluggable component's section carries one `kind` key
naming a row in the root's table (`qpu: {kind: stim_device, ...}`,
`weak_decoder: {kind: union_find, ...}`). That surface does not exist
today (the decoder algorithm is fixed on a card); it arrives as a
surface commit under the root slice's design note.

Adding a component means: write one class that implements the port, add
one row to that table, add its section to the yaml reference. Nothing
else changes. A slice that makes adding a component take more than that
is not done. The port also lets an old and a new implementation coexist
while callers move (Fowler's branch by abstraction), so a structural
slice never needs a long-lived branch.

A mechanical rewrite keeps a file's current shape, including a `connect`
step that exists today, and only makes it read well; a structural slice
moves the component to this shape and is planned and gated on its own.

## Rule 8. Decisions come from sources

A modeling question (how a shared channel serializes, how a DMA engine
queues, what a distillation level costs, how a window commits) is
answered by reading the referent, not by choosing. The referents are on
disk in the sandbox: the papers under tmp/papers, gem5 under
tmp/resources/gem5, ns-3 and the other buffer oracles under
tmp/resources/l5_buffers, the style guides and the Software Engineering
at Google book under tmp/resources/style_guides, qLDPC, Stim, PyMatching
and beliefmatching under the environment. Compilers, simulators and
schedulers outside this field are referents too when they solve the same
shape of problem. The design note of a structural slice names the
referent for each decision, and the code's docstring names it again.
When the referents disagree or are silent, the note says so and picks the
simplest law that the gate can pin.

## Rule 9. Google Python style, enforced

The Google Python Style Guide applies wherever the rules above are
silent. Import modules, not names: `import decsim.records.program as
program_records`, then `program_records.Decision`; the exceptions are
`typing`, `dataclasses`, `collections.abc` and `enum`. No module uses `from __future__ import
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

    PYTHONPATH=.pydeps .venv/bin/python -m ruff format decsim tests tools
    PYTHONPATH=.pydeps .venv/bin/python -m ruff check decsim tests tools
    .venv/bin/python tools/check_one_action.py decsim tests tools

`tools/check.sh` runs all three, on the whole tree when given no paths.
The ruff binary lives at `.pydeps/bin/ruff`; if `-m ruff` reports
RuffNotFound, copy it from the ruff wheel's `data/scripts/ruff` into
that folder. A worktree has no `.venv`, so pass the main checkout's
interpreter and dependency folder to the script:

    DECSIM_PYTHON=/path/to/decsim/.venv/bin/python \
    DECSIM_PYDEPS=/path/to/decsim/.pydeps tools/check.sh

For anything you run by hand from a worktree, set `PYTHONPATH=.` and
confirm `decsim.__file__` is the worktree before trusting any result.

## Tests

The best test of a component is its output beside a referent's output on
the same input: qLDPC's sliding windows on the same detector error
model, Stim's detector converter on the same circuit, PyMatching on the
same graph, the Tan and Skoric window harnesses, gem5's DMA law for the
links, the closed-form queue results for the decoder pool. Every
component has at least one such test, and when a referent exists for a
law, the test compares against the referent rather than against a number
the author computed by hand. The validation harnesses under the sandbox
are these tests at the system level, and they stay exact at every commit.

Beyond the referent tests, a test pins one law of the component through
its public surface, named as a sentence:
`test_a_second_correction_for_a_window_is_refused`. The values that
matter sit inside the test, even when that repeats a line; a helper
builds only what the test does not care about. A test has no branches; a
loop is allowed only in a property test that runs many random programs
against an oracle, and the test says so in its name. A test file's
docstring names the paper, the reference code, or the closed form it
checks against, and cites only what is on disk. One test file per
module, at `tests/<component>/test_<module>.py`.

At a boundary the refusal is tested, with the sentence asserted, because
a user will send that input. Inside the machine, tests are not written
for inputs nothing produces, and a test of a check that rule 4 deletes
goes with it. A test that must change when the
implementation changes, but the behavior does not, is brittle and is
rewritten. The old tests keep running until the new ones cover the same
behavior, then they are deleted in one commit per test directory.

The gate is a characterization test (Feathers; also called a golden
master): it pins what the code does, not what it should do. Its three
known failure modes are handled in this order. It pins bugs, so a bug
found on the way gets a referent and its own commit rather than a
golden move. It is blind beyond its points, so "reachable" in this
document means reachable by the sixteen gate points, the harnesses, or
a yaml key. It is brittle to incidental text, so the golden carries two
hashes, one over results and one over the log; a surface commit
regenerates only the log hash and states that the results hash is
unchanged (the verify tool gains the split in the next gate commit). A
change to a log line's text or source name is a surface change (below),
never part of a mechanical slice.

## Size

Every checklist row carries three numbers for the file: lines before,
lines after, and lines after with blank lines, comments and docstrings
removed; and one line saying where any growth came from. Growth that is
right: a dense line unpacked under rule 1, a value named, a check at a
boundary that was missing. Growth that is wrong, and is removed before
the slice closes: a check for an input no caller produces, a law for a
shape nothing runs, a helper that exists to dodge the line cap, a
docstring that repeats the code, a record or alias that has one reader.
A file may end larger than it began when all of its growth is the right
kind; it may not end larger for the wrong kind.

The row also carries the gate's wall time before and after. One action
per line adds loads and stores, and a sweep multiplies the engine, the
buffers, the links and the decoder unit by shots and points. A hot path
may keep a dense line when the row shows the cost and says so in a
comment; that is Google's "concede to practicalities", and it is the
only exception to rule 1.

## Slices and commits

Every commit is one of three kinds, never two:

- mechanical: a file reads better, its shape is unchanged, the gate passes
  unchanged;
- structural: a split, a new component, a new port, a new wiring;
  planned in a design note, reviewed, and gated on its own;
- surface: a config key, a link path name, or a log line changes;
  reference.yaml, the parameter reference, the frozen suite configs and
  the golden move in the same commit, under a design note.

A bug found on the way gets its own commit with a test and a referent; an
old test that pinned the bug is corrected or deleted in that same commit,
and the message says so. Any change to what decsim computes, however
small, is its own commit that says so. The golden is regenerated only
under a design note.

Before a slice commit: gate green, component harnesses exact, old and new
tests green, `tools/check.sh` clean on the touched files. The file
checklist in docs/rewrite/PLAN.md moves in the same commit, with the
size numbers.

Work is committed often, so that a cut-off loses little (owner,
2026-09-04). Inside a structural commit the implementer commits a
checkpoint whenever the tree imports and the touched tests pass, then
squashes the checkpoints into the one structural commit with
`git reset --soft <base>` and one commit when the gate is green. Never
more than about an hour of work sits uncommitted.

A file is touched in one slice, in two commits: the mechanical commit
first, with the gate unchanged, then the structural commit under the
design note. Mixing the two in one diff hides the move behind the
re-wording, and neither a reviewer's differential nor a reader can
separate them (Fowler's two hats; Google's one-thing-per-change).

## Review

Each slice has one implementer and reviewers who only find defects; a
reviewer never implements, and the implementer never reviews.

Reviewer A runs the differential: the same inputs through the old tree
and the new tree, every output compared, and reads the diff for anything
the differential cannot reach. Reviewer A checks every slice.

Reviewer B runs mutation testing against the new tests and reads the
files against the rules. Reviewer B checks the first round of a slice
only.

A reviewer's scope is what a yaml, a front call, or a runtime path can
produce. A finding about an input none of those produces is out of
scope: it is written on the checklist row if the reviewer thinks it
matters, and it is not fixed. A reviewer does not ask for a check, a
test or a law for such an input.

Each finding carries a severity: bug, a test that cannot fail, untested
law, weak test, rule violation, nit. A slice closes when reviewer A's
differential is exact and no in-scope bug and no test-that-cannot-fail
remains; everything else is a nit and goes on the checklist row. That is
Google's standard: approve when the change improves the code's health,
not when it is perfect, and "nit" for what need not block. Two rounds is
the backstop, not the target; a third round needs the coordinator to
say why. For a structural slice reviewer B also reads the design note
against the code. Beyond bugs, each reviewer answers this list with yes
or no for every touched file:

1. Does every line do one thing, and does `tools/check.sh` pass?
2. Is every name a full word that says what the thing is, with no
   abbreviation and no acronym outside qpu, id, io, xor, yaml, json?
3. Does every comment say why, and does every module docstring say what
   the component is in one or two plain sentences?
4. Is every check at a boundary where input enters or a contract of the
   method it sits in, and is nothing re-checked?
5. Did every rename move its callers and its tests, with no alias left?
6. Does every class own one job, with at most six attributes, all final
   at the end of `__init__`, public before private?
7. Does the component talk to others through ports, and could a new
   implementation plug in with one class and one table row?
8. Is every decision cited to its source?
9. Is the commit one kind only, and does the checklist row move with it,
   with the size numbers?
10. Is the gate green, are the harnesses exact, and are the tests green?

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
cannot come back. Chapters 11 to 14 give the test rules above: test behaviors through the
public surface, keep tests obvious and unchanging, prefer real
implementations, and use A/B diffs across a migration (reviewer A's
differential is chapter 14's). The scope sentence in "Review" is
Google's code-review guidance: solve the problem that needs solving
now, not one the developer speculates might come.
The component shape is gem5's (src/sim/sim_object.hh and the Python
params), the front is sinter's, the engine is SimPy's.

## What the first two lanes taught

Recorded 2026-09-03 so the rules above are not read as abstract. The
detector-model lane took eight review rounds and grew the package from
2,200 to 3,500 lines. Rounds one and two found four real bugs. Rounds
three to eight found inputs no yaml or runtime path produces, and each
was fixed with a check, a law or a test. That growth was the wrong kind
under "Size", the review had no scope bound, and the loop had no round
cap. Rule 4, the scope sentence in "Review", and the round backstop are
the corrections. The lane's mechanical rewrite, its four bug fixes and
its referent tests are kept.

A research pass on 2026-09-03 checked this document against Google's
guides, the Software Engineering at Google chapters, LLVM, gem5, ns-3,
sinter, SimPy, Fowler and Feathers. It found the approach recognized in
every part and changed five things: the line and attribute caps are
prompts with their origin stated; internal invariants are bare asserts;
the port rule says which ports exist and how the yaml names a
component; a slice is two commits, mechanical then structural, never
one; and a review closes on the differential with nits on the row.

## Where the tree stands

Recorded when the tooling landed, at decsim aad5747, by running the three
commands above over decsim, experiments and tests. The numbers are
findings, not lines; one line can carry several. The current numbers are
in docs/rewrite/PLAN.md, section 3, and go to zero file by file.
