# How decsim is written

The rules every line of this package is written to. Every change also
passes the behavior gate: a frozen set of points whose results and seed
paths are hashed, so a change that moves a tick or a bit fails loudly.

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
than the long function; then the function stays long and its docstring
says why. The 40 comes from Google's Python and C++ guides, where it
is a prompt to think, not a limit; the checker lists functions over it
and classes over six attributes as reports, not failures. The six is
ours. A block nested deeper than two levels is hoisted into a named
method (LLVM's early exits and predicate functions); that one is a
failure.

### Classes exempt from the six-attribute report

Some classes are one responsibility with genuinely many collaborators:
splitting them would put one job in two places. `tools/check_one_action.py`
reads this list and reports every other wide class, so the exemptions are
visible here, beside the rule, rather than buried in the checker. The list
stays short; past eight names the rule is wrong, not the list.

- `WindowManager` (`decsim/windows/window_manager.py`): the windows
  package's facade over that package's components, plus the escalation
  package's strong re-decode it wakes and the workload's feedback mode.
  A caller asks the facade for what it wants (`window_sources`,
  `copy_sources`, `planned_windows`, `reads_windows_from`) rather than
  for a component of it; what an observer still reaches for through it
  is a component's own trace group, which by design no port carries.
- `FeedbackStreams` (`decsim/controller/feedback_streams.py`): one
  protected cycle, which needs the qpu it releases, the windows it hears
  from, and the three tables the program declares it with.
- `StrongRedecode` (`decsim/escalation/strong_redecode.py`): one strong
  re-decode of a window, which crosses both send ends, the decode queue
  and the committer's return path in a single flow.
- `DecodeRequester` (`decsim/windows/decode_requests.py`): one request per
  complete window, which needs the window state, the builder, the queue,
  the escalation verdict and the store's outgoing port to place it.
- `IdleRoundAccounting` (`decsim/controller/idle_rounds.py`): one idle
  round routed by the policy, which needs the geometry, the streams and
  the qpu the round belongs to as well as the queue it charges.
- `RoundWriter` (`decsim/controller/round_writes.py`): one finished round
  written to its stores or held, with both stores, the hold and the
  transmitter that publishes the landing.
- `OperationResults` (`decsim/windows/operation_results.py`): one final
  result per operation, which reads the plan, the window state, the
  retention and the ledger before it releases a conditional operation.
- `DecodeRequestBuilder` (`decsim/windows/decode_requests.py`): one decode
  job built from a window, stamped with the input gate and the run-wide
  request ordinal.

## Rule 2. Names are full words that say what the thing is

No abbreviations. No acronyms except these, which are words in this field
and stay: qpu, id, io, xor, yaml, json. docs/glossary.md maps every plain
name to the exact term the literature uses, so
`minimum_weight_perfect_matching` is listed beside "MWPM", and it carries
the renamed modules too.

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

Link paths and yaml keys are plain words too: qpu_to_controller,
controller_to_weak_buffer, controller_to_strong_buffer,
weak_buffer_to_weak_decoder, strong_buffer_to_strong_decoder,
weak_decoder_to_strong_decoder, decoder_to_decoder,
weak_decoder_to_frame, strong_decoder_to_frame, frame_to_controller,
controller_to_qpu; and the config keys readout_to_bits_cycles,
decision_to_pulse_cycles, packing_cycles_per_round, weak_buffer_rounds,
strong_buffer_rounds, packing_rounds_in_flight, unit_memory_rounds,
write_cycles, setup_cycles_per_transfer, circuit, log_component_io,
check_windows_with, decode_path.

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

Dead code goes with the next change to the file that holds it. When a
file is rewritten, every function, class, branch, field and parameter
that nothing calls or reads is deleted, and the commit message lists
what was removed. Git remembers it; nothing is kept "in case". A decoder
or a study knob that is unused today but has research value is not
dead code; it stays, with the reason in its docstring.

## Rule 6. One reason per component

A class owns one job: its docstring's first sentence says it without an
"and". Its state is set in `__init__` and is at most six attributes; the
checker reports more as wide state, a prompt to split the class. Every
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

This is the shape of every component. The ownership and the name table
are gem5's (a SimObject's Python class is its params; `allClasses` maps
a name to a class); the table plus one abstract class per pluggable part
is sinter's (`BUILT_IN_DECODERS` and `Decoder`); the wiring is by
constructor, not gem5's late port bind, because Python needs no second
step. It applies to every component.

A port is a small Protocol in `decsim/ports.py` with the
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

A Protocol that stays outside `decsim/ports.py` is a seam inside one
package: every implementation of it and every caller of it sit in that
package, so no component learns it and the package may change it alone.
Its module docstring names it as that package's own seam. The one
exception is `decsim/seeding.py`, a shared module beside `ports` in the
package order whose two Protocols cannot move without making the two
depend on each other. Everything a second package implements or calls is
a port and lives in the port file.

One root object, `Machine`, builds every component from its settings and
wires them by constructor; no component builds or looks up another. The
yaml has one section per component, each section builds one settings
dataclass, and a pluggable component's section carries one `kind` key
naming a row in the root's table (`qpu: {kind: stim_device, ...}`,
`weak_decoder: {kind: union_find, ...}`).

Adding a component means: write one class that implements the port, add
one row to that table, add its section to the yaml reference. Nothing
else changes. A change that makes adding a component take more than that
is not done. The port also lets an old and a new implementation coexist
while callers move (Fowler's branch by abstraction), so a restructuring
never needs a long-lived branch.

## Rule 8. Decisions come from sources

A modeling question (how a shared channel serializes, how a DMA engine
queues, what a distillation level costs, how a window commits) is
answered by reading the referent, not by choosing. The referents are the
papers, cited by arXiv number; gem5, ns-3 and the other queue oracles;
the Google style guides and the Software Engineering at Google book; and
qLDPC, Stim, PyMatching and beliefmatching as installed. Compilers,
simulators and schedulers outside this field are referents too when they
solve the same shape of problem. The code's docstring names the referent
for each decision. When the referents disagree or are silent, the
docstring says so and the code picks the simplest law that the gate can
pin.

## Rule 9. Google Python style, enforced

The Google Python Style Guide applies wherever the rules above are
silent. Import modules, not names: `import decsim.records.program as
program_records`, then `program_records.Decision`; the exceptions are
`typing`, `dataclasses`, `collections.abc` and `enum`. No module uses
`from __future__ import annotations`; `Optional[X]` is written out
because the package runs on Python 3.9.

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

## Rule 10. The package order, the rows, and the ports

The packages import each other in one direction only: the `uses`
relation is a partial order, so the top levels can be cut off and the
rest still runs (Parnas 1972 lines 505-529; Dijkstra's THE levels,
dijkstra_the.txt 52-57). `tools/check_uses_graph.py`, run by
`tools/check.sh`, fails on any cycle and prints the levels, which
machine.py's docstring names. No component recognises another
component's row by its class: a fact a caller needs about a row is
declared on the port and answered by every row, never read off the
row's type, because a class is what the port promises not to reveal
(gem5's port API, arXiv 2007.03152 lines 489-491). And `decsim/ports.py`
is the slowest layer of all: a port method added, renamed or removed
needs a design note saying why, the way a golden move does.

## Tests

The best test of a component is its output beside a referent's output on
the same input: qLDPC's sliding windows on the same detector error
model, Stim's detector converter on the same circuit, PyMatching on the
same graph, the Tan and Skoric window harnesses, gem5's DMA law for the
links, the closed-form queue results for the decoder pool. Every
component has at least one such test, and when a referent exists for a
law, the test compares against the referent rather than against a number
the author computed by hand.

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
document means reachable by the gate points, the tests, or a yaml key.
It is brittle to incidental text, so the golden carries two hashes, one
over results and one over the log; a change to a log line's text or
source name regenerates only the log hash and states in its commit
message that the results hash is unchanged.

## Size

A file's size is three numbers: lines before a change, lines after, and
lines after with blank lines, comments and docstrings removed; and one
line in the commit message saying where any growth came from. Growth
that is right: a dense line unpacked under rule 1, a value named, a check at a
boundary that was missing. Growth that is wrong, and is removed before
the change lands: a check for an input no caller produces, a law for a
shape nothing runs, a helper that exists to dodge the line cap, a
docstring that repeats the code, a record or alias that has one reader.
A file may end larger than it began when all of its growth is the right
kind; it may not end larger for the wrong kind.

The commit message also carries the gate's wall time before and after.
One action per line adds loads and stores, and a sweep multiplies the
engine, the buffers, the links and the decoder unit by shots and points.
A hot path may keep a dense line when the measured cost says so in a
comment; that is Google's "concede to practicalities", and it is the
only exception to rule 1.

## Where these rules come from

Google's own rules for rules (Software Engineering at Google, chapter
8): a rule must pull its weight, favor the reader over the writer, keep
the code consistent, and be enforced by a tool wherever a tool can.
Chapter 20's bar for a checker is a false-positive rate under ten
percent; when the one-action checker flags a line that reads well, the
fix is to the checker's rules, recorded here, never a per-file
exception. Chapter 22's rule for a large change: past a few hundred
edits, write the tool that makes the edit (the tick-helper rename was
one), and add a check so the old form cannot come back. Chapters 11 to
14 give the test rules above: test behaviors through the public surface,
keep tests obvious and unchanging, prefer real implementations, and use
A/B diffs across a migration (a differential review is chapter 14's).
Google's code-review guidance holds: solve the problem that needs
solving now, not one the developer speculates might come. The component
shape is gem5's (src/sim/sim_object.hh and the Python params), the front
is sinter's, the engine is SimPy's.

