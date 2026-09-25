[decsim docs](../README.md) › [How-to guides](README.md)

# How to plug in a workload maker

A workload maker is anything that produces the work a run does: Stim's
generator, Deltakit, a lattice-surgery scheduler, your own script. Every
maker hands decsim at most two things (`decsim/records/workload.py`):

- a physical circuit: one finite Stim circuit and the round each of its
  measurements lands in, or the four live fragments of a repeated
  memory (`RepeatedStimCircuit`) with their declared round period;
- an operation list: operations over patches, each with its kind, its
  predecessors, the operation whose decision releases it (`blocked_by`),
  its rounds, and either its own circuit or a segment (`stream_id`,
  `stream_offset`) of the one physical circuit.

A yaml names a maker with one entry, a Python function and its
arguments. There is no plugin framework and no registration.

## 1. Write the function

```python
import decsim.records.program as program_records
import decsim.records.workload as workload_records


def two_patch_memory(patch_rounds, distance):
    first = program_records.Operation(1, "a", (1,), patches=(1,))
    second = program_records.Operation(2, "b", (2,), patches=(2,))
    rounds = {1: patch_rounds, 2: patch_rounds + distance}
    return workload_records.Workload((first, second), rounds)
```

A parameter named `physical_error_probability`, `distance` or
`round_period_microseconds` receives the sweep point's value; every
other parameter is an argument the yaml writes. The function runs once
per sweep point, and every shot of the point runs what it returned.

## 2. Name it in the yaml

```yaml
workload:
  kind: producer
  function: my_package.makers:two_patch_memory
  arguments:
    patch_rounds: 4
```

The module is imported by name, so it needs to be importable where
decsim runs (installed, or on `PYTHONPATH`). A module that does not
import or a function that is not there is refused with one sentence
when the point calls the maker, and an argument the function does not
take, one it needs and is not given, or a sweep value written as an
argument stops the call with Python's own error, which names the
argument.

The makers decsim ships are named the same way, from
`decsim/producers.py`: `decsim.producers:memory_circuit` (Stim's
generated memory), `decsim.producers:memory_patches` (several such
memories at once), and, with the `deltakit` extra installed,
`decsim.producers:deltakit_memory` (Deltakit's finite memory) and
`decsim.producers:deltakit_live_memory` (a live memory decoded after
some rounds, then read out).

## 3. Know what decsim derives

You write the operations; decsim fills in what follows from them
(`decsim/frontends/circuit_frontend.py`):

- the predecessors from program order on each operation's patches,
  beside the ones you declare;
- the rounds of an operation you leave out of the rounds mapping, from
  its kind (`round_policies.GateRounds`), and the whole circuit's rounds
  for the one operation that runs a finite circuit;
- for segments of a stream: the stream's owner, and the protected
  region in which the stream's patches keep measuring while the
  operations after the last segment wait.

## Limits

decsim never builds a merged circuit. For real syndromes across a
merge, the maker supplies one circuit for the whole history and gives
every operation that emits detector data its round range in it, a
`stream_id` and a `stream_offset`. A circuit shared by several
operations without those ranges is refused with a sentence.
