[decsim docs](../README.md) › [Explanation](README.md)

# The principles behind the shape

Eleven ideas explain why decsim's tree looks the way it does. Each is
somebody else's, each is quoted from a source on disk, and each has a
consequence you can see in the code. They are the reasons behind
`STYLE.md`, which is the same ideas written as rules a tool can check.

Each source is cited by title and, where it has one, arXiv number; a
line number is into the plain-text extraction of that source, the same
one the code's docstrings cite.

## 1. A module hides one decision that is likely to change

"We propose instead that one begins with a list of difficult design
decisions or design decisions which are likely to change. Each module is
then designed to hide such a decision from the others"
(`parnas1972.txt` lines 547-550). And the warning that goes with it: "it
is almost always incorrect to begin the decomposition of a system into
modules on the basis of a flowchart" (lines 545-546).

**In decsim.** The list of decisions the simulator exists to vary is the
list of tables: which decoder, which windowing scheme, which link
fabric, which escalation policy, where detection events are formed,
whether a hop copies or references. Each of those lives in one place,
and everything else asks that place through a port. The packages are not
named after the stages of a flowchart; they are named after the
decisions they own.

## 2. The `uses` relation is a partial order

"We have a hierarchical structure if a certain relation may be defined
between the modules or programs and that relation is a partial ordering.
The relation we are concerned with is 'uses' or 'depends upon'"
(`parnas1972.txt` lines 504-511), and with the hierarchy "we are able to
cut off the upper levels and still have a usable and useful product"
(lines 518-520). Dijkstra's THE builds the same order one level at a
time, each level an abstraction the levels above rely on and the levels
below know nothing of (`dijkstra_the.txt` lines 52-57).

**In decsim.** Twenty-five packages on eleven levels, printed by
`tools/check_uses_graph.py` and enforced by `tools/check.sh`, which
fails on any cycle. Nothing at level 3 or below imports `decsim/build/`
or `decsim/machine.py`, so the decoders' own tests decode a window on a
store with no root at all. The levels are listed in
[The map of the package](../reference/map.md).

## 3. An interface reveals as little as it can, and never an order it does not need

"By prescribing the order for the shifts we have given more information
than necessary and so unnecessarily restricted the class of systems that
we can build without changing the definitions ... must clearly be
classified as a design error" (`parnas1972.txt` lines 370-379).

**In decsim.** A port method promises a set where a set is enough. The
same rule applies to the tests: a comparison that pins an order the
code does not promise fails for reasons that are not defects, which a
run that prices two real decoders from the host clock shows whenever
their order flips.

## 4. An interface is the set of assumptions two programs make about each other

"The interface between two programs consists of the set of assumptions
that each programmer needs to make about the other program in order to
demonstrate the correctness of his program" (`lampson1983.txt` lines
79-81). Lampson's hints follow from it: "Do one thing at a time, and do
it well. An interface should capture the minimum essentials of an
abstraction. Don't generalize; generalizations are generally wrong"
(lines 100-101), and "service must have a fairly predictable cost, and
the interface must not promise more than the implementer knows how to
deliver" (lines 157-158).

**In decsim.** `decsim/ports.py` is the slowest-changing layer in the
tree. A port method added, renamed or removed needs a design note saying
why. Every port
method has a caller outside its own package, or it should not be a port.

## 5. An update is a function of its arguments, so a run can be replayed

"The update procedure must be a true function: Its result does not
depend on any state outside its arguments" (`lampson1983.txt` lines
947-948), which is what lets a log of entries be re-executed and produce
the same objects.

**In decsim.** A run is a function of the yaml and the seed. Every
stochastic component derives its generator from the root seed and its
own path in the machine, so one seed reproduces one shot exactly
(`decsim/seeding.py`). The one exception is deliberate and named: a tier
that prices its decode from the host wall clock is not a function of its
arguments, which is why [Time](time.md) exists and why a regression comparison
of such a run must leave the tick-bearing fields out.

## 6. Model objects, a separate configuration script, a port API, timing apart from function

"the user writes a Python script that describes the system under test by
instantiating model objects" (`gem5_v20.txt` lines 342-346); "gem5
provides a modular port interface which allows any component that
implements the port API to be connected to any other component
implementing the same API" (lines 489-491); "gem5 separates the
functional execution from the timing in most of its models" (lines
369-370).

**In decsim.** A yaml file is the configuration script and
`decsim/machine.py` is the root that reads it; a component owns its
settings and its state and talks to its neighbours only through ports.
Function and timing are separate in the same way: a decoder row returns
a correction, and what that decode costs is the unit's card or its
measured clock, charged around the call.

## 7. A record that crosses a boundary lives in one shared place

"A data structure, its internal linkings, accessing procedures and
modifying procedures are part of a single module. They are not shared by
many modules as is conventionally done" (`parnas1972.txt` lines
384-387).

**In decsim.** A record used by one component lives next to it; a record
shared by two or more lives in `decsim/records/`, one module per record
family, and that is `STYLE.md` rule 6. It is why a reader can learn the
vocabulary of the whole machine by reading one folder.

## 8. Different parts change at different rates, and the fast must not force the slow

"Different artifacts change at different rates" (`bigballofmud.txt`
lines 1689-1690, the Shearing Layers pattern). And the reason to be
careful about what you expose while they do: "With a sufficient number
of users of an API, it does not matter what you promise in the contract:
all observable behaviors of your system will be depended on by somebody"
(`hyrum.txt` lines 6-9).

**In decsim.** Adding a table row is the fast layer and touches no port
and no root line. Adding a yaml key touches its settings record and
`configs/reference.yaml` and nothing above them. A port change is the
slow layer and carries a design note. Hyrum's half is a rule for the
tests: a test that pins a tick, an order or a string the port does not
promise has made itself a consumer of the implementation, and it is
rewritten rather than protected.

## 9. Complexity is essential, so remove the accidental and grow the rest

"The complexity of software is an essential property, not an accidental
one" (`brooks_nsb.txt` line 113). What follows is a method: "the system
should first be made to run, even though it does nothing useful except
call the proper set of dummy subprograms. Then, bit-by-bit it is fleshed
out" (lines 660-663). And: "The most radical possible solution for
constructing software is not to construct it at all" (lines 547-548).

**In decsim.** The essential complexity is the set of states a window
and a decode job can be in, and it is written down as records with
phases rather than spread across flags. The parts that are bought
instead of built are the decoders (PyMatching, union find, belief
matching, Relay-BP, Tesseract, BP-OSD), Stim for the circuits and their
error models, Chrome's trace format and its viewers, and sinter's shape
for the collect layer.

## 10. The system copies the organisation that builds it

"organizations which design systems ... are constrained to produce
designs which are copies of the communication structures of these
organizations" (`conway1968.txt` line 117), with the practical advice
that follows: "Ways must be found to reward design managers for keeping
their organizations lean and flexible" (line 119).

**In decsim.** The port file is written from the design record rather
than grown from whatever was convenient at the call site, because every
port method is a negotiation between two components and therefore
between two pieces of work. [The design decisions](decisions.md) is where
those negotiations are recorded.

## 11. Local and remote calls differ in kind, and the interface must say which

"The major differences between local and distributed computing concern
the areas of latency, memory access, partial failure, and concurrency"
(`waldo1994.txt` lines 302-304), and therefore "an additional part of
the definition of a class of objects will be the specification of
whether those objects are meant to be used locally or remotely" (lines
852-855).

**In decsim.** The eleven priced hops are that line. A call across a hop
has a card, a payload a record names, and a send at one end; a call
inside a unit is never priced. Memory access is made explicit per hop by
the memory class and by the copy-or-reference key. Partial failure is
not modelled at all, and that is written into `decsim/machine.py`'s own
docstring as a stated scope rather than left for a reader to discover:
no hop drops, duplicates or reorders what it carries, and nothing
retries.

## What is not here

Three sources that belong on this list were not read when it was
written and are therefore not cited: Saltzer and Kaashoek's *Principles
of Computer System Design* (chapter 1), Ousterhout's *A Philosophy of
Software Design*, and Liskov and Zilles 1974.

## Read next

- `STYLE.md`: these eleven as rules, with the tools that check them.
- [The design decisions](decisions.md): the thirteen modelling decisions made
  under them.
- [Architecture](architecture.md): the shape they produced.
