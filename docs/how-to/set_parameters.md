[decsim docs](../README.md) › [How-to guides](README.md)

# How to set any parameter without editing code

Every field of a machine is a parameter at a dotted path: a part's
name, then the field names down its record (`decsim/parameters/tree.py`);
the hops' prices sit under `fabric`, by role.
A unit leaves the field's name and is written with the value, so
`fetch_cycles_per_round` is set as `fetch_per_round=8ns` or
`fetch_per_round=2 cycles`, and converts exactly as decsim converts it
(`decsim/parameters/fields.py`).

## 1. See what can be set

```bash
decsim params experiments/switching/run.py
decsim params experiments/switching/run.py --match "weak.**"
```

Each line is `path = value  [record default]  (set by)`, for the first
task, or the one `--only NAME` names. A `.type` line is a part you can
swap.

## 2. Set one value, or many

```bash
decsim run experiments/switching/run.py --set weak.unit_count=3
decsim run experiments/switching/run.py --set "**.engine.clock=500MHz" \
    --set "fabric.*.latency=20ns"
```

`*` stands for one name and `**` for any number of them. The flags
apply to every task, in order, a later one winning. A value is a
number, `true` or `false`, a word, a quantity with its unit (`157ns`,
`1.1us`, `250MHz`, `3 cycles`, `64 bits`, `20dB`, `100Gbps`), `none`,
`unlimited`, or a list `[a, b]`. A text that would read as something
else is quoted: `name='3'`.

A path no task has is refused with the paths it was near, and a value
decsim refuses is refused by its path, before any shot runs.

## 3. Swap a part

```bash
decsim info
decsim info DecoderPool
decsim run run.py --set weak.algorithm.type=union_find \
    --set weak.algorithm.timing.type=HostMeasuredTime
decsim run run.py --set fabric.forward.bits_per_cycle=400
```

`decsim info` lists every kind of part and every slot below one, as
`record.field`, with the records it takes; `decsim info NAME` lists one record's parameters
and, for each part, the records its slot takes. NAME may be an import
path, `decsim info mylab.parts.NeuralDecoder.Settings`. A part keeps the old record's fields of the
same name and takes the assignments that reach the rest; a required
one left unset is refused. A record of your own is written by its
import path, `weak.algorithm.type=mylab.decoders.NeuralSettings`,
and is taken if the slot's type admits it. `type=none` empties an
optional slot; in a slot that must hold a record it names the record
called `none`, as `switching.confidence.type=none` does.

## 4. Check every task before a job

```bash
decsim check experiments/switching/run.py
```

`check` builds every task's machine once, as a run builds it, and runs
no shot. A task whose build refuses a setting is printed with the
reason, the rest are still built, and the command exits 1. Run it with
the same `--set` flags as the run.

## 5. Run it on Slurm

The flags are carried in the environment variable DECSIM_SET, `;`
between assignments, which you can also set yourself; the flags come
after it and win. Export it in the shell that submits the jobs, which
sbatch passes on, or give each job the same `--set` flags as the
record, so every job builds the tasks the folder recorded; a job that
builds another is refused. Each run's `run.json` records it.

## 6. Know what stays in the run file

A changed value is a new machine, so its task has a new id. The run
file still names the task, so a name such as `cluster_gap_d3` keeps
saying what the run file built.

The distance follows the workload. A workload made at the task, the
`memory` row every preset base holds, is made again at the distance
you set, and its own fields are paths:

```bash
decsim run run.py --set qpu.distance=7 --set workload.physical_error_probability=0.004
```

A workload lowered in the run file, or a live memory read from
fragments, was made at one distance, so then `qpu.distance` and the
lowered fields are refused by name (`Machine.set_in_the_run_file`,
`decsim/topology/graph.py`): an assignment would leave the circuit
behind.
`decsim params` marks them `(run file only)`, and a wildcard passes
them by. [A maker row](plug_in_a_workload_maker.md) makes such a
workload settable.

## 7. Set a constant

A run built from a constants file takes `constants.<name>` as a path,
and `decsim params` lists those names after the machine's:

```bash
decsim run run.py --set constants.escalation.threshold.threshold=25dB
```

A `--set` to a value the run file sweeps is refused, since every cell
would run it while its name said its own. [How to sweep a machine over a
grid](sweep_a_machine_over_a_grid.md) writes such a run.
