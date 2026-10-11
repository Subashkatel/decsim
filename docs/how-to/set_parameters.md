[decsim docs](../README.md) › [How-to guides](README.md)

# How to set any parameter without editing code

Every field of a machine's settings is a parameter at a dotted path:
the field names from the machine down (`decsim/parameters/tree.py`).
A unit leaves the field's name and is written with the value, so
`fetch_cycles_per_round` is set as `fetch_per_round=8ns` or
`fetch_per_round=2 cycles`, and converts exactly as decsim converts it
(`decsim/parameters/fields.py`).

## 1. See what can be set

```bash
decsim params experiments/switching/run.py
decsim params experiments/switching/run.py --match "weak_decoder.**"
```

Each line is `path = value  [record default]  (set by)`, for the first
task, or the one `--only NAME` names. A `.type` line is a part you can
swap.

## 2. Set one value, or many

```bash
decsim run experiments/switching/run.py --set weak_decoder.unit_count=3
decsim run experiments/switching/run.py --set "**.engine.clock=500MHz" \
    --set "links.*.channel.propagation_latency=20ns"
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
decsim info DecoderPool
decsim run run.py --set weak_decoder.algorithm.type=union_find \
    --set weak_decoder.algorithm.timing.type=HostMeasuredTime
decsim run run.py --set "links.weak_decoder_to_strong_decoder.channel.capacity={type: Capacity, input: 100Gbps, source: my estimate}"
```

`decsim info NAME` lists a record's parameters and, for each part, the
records its slot takes. A part keeps the old record's fields of the
same name and takes the assignments that reach the rest; a required
one left unset is refused. A record of your own is written by its
import path, `weak_decoder.algorithm.type=mylab.decoders.NeuralSettings`,
and is taken if the slot's type admits it. `type=none` empties an
optional slot.

## 4. Run it on Slurm

The flags are carried in the environment variable DECSIM_SET, `;`
between assignments, which you can also set yourself; the flags come
after it and win. `--slurm` writes it into `run.sbatch`, so every job
builds the tasks the plan holds, and each run's `run.json` records it.

## 5. Know what stays in the run file

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
lowered fields are refused by name (`MachineSettings.set_in_the_run_file`,
`decsim/settings.py`): an assignment would leave the circuit behind.
`decsim params` marks them `(run file only)`, and a wildcard passes
them by. [A maker row](plug_in_a_workload_maker.md) makes such a
workload settable.
