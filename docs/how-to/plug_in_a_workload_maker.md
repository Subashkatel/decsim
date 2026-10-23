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

A run file calls the maker and hands its workload to the machine.
There is no plugin framework and no registration.

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

The arguments are yours: the run file calls the function once per
task, so a sweep over `patch_rounds` calls it with each value, and
every shot of a task runs what it returned.

## 2. Hand it to the machine

The machine's `workload` part is the workload, lowered for the
machine by `WorkloadSettings.running` (`decsim/frontends/settings.py`).
On a base such as `machines.weak_baseline(distance, 0.001, 1.0)`
(`decsim/presets/machines.py`), put it in place of the base's own:

```python
workload = two_patch_memory(4, distance)
running = workload_settings.WorkloadSettings.running(workload)
workload_part = graph.Part("workload", running)
machine = base.with_part(workload_part)
```

The makers decsim ships are in `decsim/producers.py`: `memory_circuit`
(Stim's generated memory), `memory_patches` (several such memories at
once), and, with the `deltakit` extra installed, `deltakit_memory`
(Deltakit's finite memory) and `deltakit_live_memory` (a live memory
decoded after some rounds, then read out). `memory_workload` in
`decsim/presets/machines.py` is `memory_circuit` on the rotated surface
code, running.

## Or give the workload a maker row, set by path

A workload lowered in the run file was made at the distance the run
file gave its maker, so the distance is not set by path
([set parameters](set_parameters.md)). A maker row holds the maker's
parameters instead and is made when each task is made, at the task's
qpu (`decsim/producers.py`):

| Row | Makes | Its parameters |
|---|---|---|
| `memory` | Stim's memory at the qpu's distance (`memory_circuit`, `memory_patches`), or patches with no circuit (`memory_without_circuit`) | `physical_error_probability`; `rounds` or `rounds_per_distance`; `code_task`, a Stim task of the qpu's card or `none`; `patches`; `noise`, a noise row ([use your own noise model](use_your_own_noise_model.md)); `decoded`; `noiseless_time_boundaries` |
| `live_memory` | a live memory over a fragments folder (`live_memory`) | `fragments`, `decode_after_rounds`, `patch`, `feedback_boundary_mode`; the fragments fix the distance |

Every preset base holds a `memory` row, so on them:

```bash
decsim run run.py --set qpu.distance=7 --set workload.rounds_per_distance=none --set workload.rounds=200
```

In Python, a row is a part like any other:

```python
memory = producers.MemorySettings(0.001, rounds=100, patches=2)
machine = base.with_part(graph.Part("workload", memory))
```

A task's id is that of the workload the row lowers to, so a row and
the same workload lowered in Python are one task. A row of your own is
a record with `name`, `fixes_the_distance` and `lowered_for(qpu)`,
which returns `WorkloadSettings.running` of your maker's workload
(`ports.WorkloadMakerSettings`); it is written by its import path,
`workload.type=mylab.makers.SurgerySettings`.

## Or read the two outputs from files

A maker that runs elsewhere writes its two outputs to disk, and
`read_workload` (`decsim/frontends/workload_files.py`) reads them:

```python
operations_path = pathlib.Path("merge.json")
# a finite circuit with each measurement index's round (left out when
# its DETECTOR time coordinates give the rounds), or fragments_path, a
# folder of the four live fragments
circuit_path = pathlib.Path("history.stim")
measurement_rounds_path = pathlib.Path("rounds.json")
workload = workload_files.read_workload(
    operations_path,
    circuit_path=circuit_path,
    measurement_rounds_path=measurement_rounds_path,
)
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
each measurement index, as a string, to its one-based round; without
it, each measurement takes the round its DETECTOR group announces, the
time coordinate plus one, and a circuit with an untimed detector is
refused. A
`fragments` folder holds `first_round.stim`, `repeated_round.stim`,
`final_round.stim`, `single_round.stim` and `physical.json`, which holds
`round_period_microseconds`, the period the fragments' noise was built
for. A missing or malformed file stops the read with Python's or
Stim's own error, and an operations file of another schema is refused.
`decsim.frontends.workload_files.write_workload` writes a Python
maker's workload in this form.

## Run a live memory

A live memory runs on the streaming Stim source, the QPU's source
`StreamingStimDevice.Settings()`: the stream's first rounds are
decoded, the patch keeps measuring while it waits for the decoded
answer, one round resumes it, and the readout ends it. With Deltakit,
the maker is `deltakit_live_memory`, called with the decode after its
first rounds, `decode_after_rounds=3`; with fragments on disk, the four
operations are written out:

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

Both run what `examples/live_memory_example.py` builds by hand in Python
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

## Decode a lattice-surgery circuit on real syndromes

Run the whole circuit as one operation. A tqec circuit (tqec 0.2.0,
Apache-2.0) times its detectors by coordinate, so it needs no rounds
file. The operations file holds one operation over every patch:

```json
{
  "schema": "decsim.ops/1",
  "operations": [{"id": 1, "name": "cnot", "patches": [0, 1, 2]}]
}
```

and the run file reads it beside the circuit:

```python
workload = workload_files.read_workload(
    pathlib.Path("cnot.json"), circuit_path=pathlib.Path("cnot.stim")
)
machine = machines.weak_baseline(3, 0.002, 1.0)
machine = machine.with_held(
    "workload", workload_settings.WorkloadSettings.running(workload)
)
```

`decsim run` then writes each shot's `logical_failure` to `shots.csv`:
the windows slide over the whole program's rounds, and a shot fails
when any of the circuit's observables is wrong. Beside whole-circuit
PyMatching on the same 4000 shots, with sliding windows, decsim's
prediction is the same on every shot of a tqec ZZ merge and of a
T teleport's Clifford part, and on 3999 of a CNOT (364 failures
against 363). With one window over every round it is the same shot
for shot (tests/frontends/test_circuit_frontend.py).

## Limits

decsim never builds a merged circuit. For real syndromes across a
merge, the maker supplies one circuit for the whole history and gives
every operation that emits detector data its round range in it, a
`stream_id` and a `stream_offset`. A circuit shared by several
operations without those ranges is refused with a sentence.

Not supported yet on real syndromes: a decode stream per patch (patch
streams are timing only), a wait on a decode inside a finite circuit,
whose timeline is fixed, and switching on region windows.
