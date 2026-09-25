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

## Or read the two outputs from files

A maker that runs elsewhere writes its two outputs to disk, and the
`files` row reads them, paths relative to the folder of the yaml that
writes the `workload` section (a base's section, inherited through
`extends`, reads beside the base):

```yaml
workload:
  kind: files
  operations: merge.json            # required
  # circuit: history.stim           # a finite circuit, with
  # measurement_rounds: rounds.json  # each measurement index's round
  # fragments: live/                # or the four live fragments
```

The operation list is json with the schema `decsim.ops/1`:

```json
{
  "schema": "decsim.ops/1",
  "operations": [
    {"id": 1, "name": "mem0", "patches": [0], "kind": "MEMORY"},
    {"id": 2, "name": "mem1", "patches": [1], "kind": "MEMORY"},
    {"id": 3, "name": "merge01", "patches": [0, 1], "kind": "MERGE"},
    {"id": 4, "name": "measure", "patches": [0], "kind": "MEASURE"}
  ]
}
```

`id` and `patches` are required. The other keys are `qubits`, `name`,
`kind` (a member of `records/program.py` `OpKind`: IDLE, MEMORY, MERGE,
MEASURE, INJECT or GENERIC),
`rounds`, `predecessors`, `blocked_by`, `stream_id`, `stream_offset`,
`emits_detector_data`, `scheduled_start_round`, `clifford` and
`circuit` (the operation's own
`.stim`, relative to the operations file). A key left out takes the
default of `records/program.py` `Operation`. `measurement_rounds` maps
each measurement index, as a string, to its one-based round. A
`fragments` folder holds `first_round.stim`, `repeated_round.stim`,
`final_round.stim`, `single_round.stim` and `physical.json`, which holds
`round_period_microseconds`, the period the fragments' noise was built
for. `decsim show` builds the first point, which reads the files, so a
missing or malformed one stops it with Python's or Stim's own error,
and an operations file of another schema is refused.
`decsim.frontends.circuit_frontend.write_workload` writes a Python
maker's workload in this form.

## Run a live memory from a yaml

A live memory runs on `qpu: {kind: streaming_stim}`: the stream's
first rounds are decoded, the patch keeps measuring while it waits for
the decoded answer, one round resumes it, and the readout ends it. With
Deltakit:

```yaml
qpu: {kind: streaming_stim}
workload:
  kind: producer
  function: decsim.producers:deltakit_live_memory
  arguments: {decode_after_rounds: 3}
```

or with fragments on disk, the four operations written out:

```json
{
  "schema": "decsim.ops/1",
  "operations": [
    {"id": 1, "name": "prefix", "patches": ["memory-patch"],
     "stream_id": 100, "stream_offset": 0, "rounds": 3},
    {"id": 2, "name": "protect", "patches": ["memory-patch"],
     "emits_detector_data": false, "rounds": 0},
    {"id": 3, "name": "resume", "patches": ["memory-patch"],
     "blocked_by": 1, "emits_detector_data": false, "rounds": 1},
    {"id": 4, "name": "readout", "patches": ["memory-patch"],
     "emits_detector_data": false, "rounds": 0}
  ]
}
```

Both run what `tools/live_memory_example.py` builds by hand in Python
(`tests/test_live_memory_example.py` and `tests/test_producers.py`
compare them field by field). The stream's owner, its protected region
and its rounds are derived, as the next section says.

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
