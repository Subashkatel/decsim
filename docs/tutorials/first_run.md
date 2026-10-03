[decsim docs](../README.md) › [Tutorials](README.md)

# Your first run

This lesson takes about ten minutes. By the end you will have run one
shot of the reference configuration, opened the folder it wrote, read a
figure and followed one round of syndrome data through the machine.

You do not need to know anything about decsim, and the words you need
are defined as they appear. You do need a terminal and a Python 3.10 or
newer interpreter.

## What decsim is, in one paragraph

A quantum computer that corrects its own errors runs in a loop. The
quantum processing unit, the QPU, measures a small block of qubits over
and over. Each pass is a **round**. A round produces a handful of
classical bits called the **syndrome**: not the state of the qubits,
which measuring would destroy, but a set of parity checks that say where
something looks wrong. Those bits leave the fridge, cross a control
computer, and reach a **decoder**, which is a piece of classical
software or hardware that reads them and works out which correction to
apply. The correction comes back, and the next quantum operation can
proceed. decsim simulates that whole classical loop and measures two
things: how long the loop takes, and how often the decoder gets the
answer wrong.

## Step 1. Install

From the checkout:

```bash
python -m pip install -e ".[run]" -c constraints.txt
```

The `run` extra brings Stim (which simulates the quantum circuit and
produces the syndrome), PyMatching (the default decoder), ldpc (the
BP-OSD and belief matching decoders), PyYAML (which reads the config),
numpy, scipy and matplotlib. Without it decsim imports but cannot run a
shot.

## Step 2. Run one shot

```bash
decsim run configs/reference.yaml --seed 0 --trace --out results/first_shot
```

A **shot** is one complete run of the workload from start to finish,
with one random seed. `configs/reference.yaml` is the reference
configuration: it carries every key the yaml layer reads, and its sweep
is deliberately tiny so that it runs in seconds. `--trace` asks for a
record of the data path, which step 6 reads, and `--out` names the
folder it goes in.

The output:

```
config: reference
point: {"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.001} seed 0
terminal status: complete
execution done: 15000000 ticks
fully done: 37356000 ticks
operation 1: logical_observables, observables (0,), truth (0,)
run dir: results/first_shot
```

Line by line:

- `point: {...} seed 0`. One point of a sweep is one machine, and the
  braces hold the settings its sweep block set, each by its path in the
  yaml. `qpu.distance` is the code distance, the size of the error
  correcting code: distance 3 corrects one error.
  `qpu.round_period_microseconds` is how long one round of measurement
  takes on the QPU, one microsecond here.
  `workload.arguments.physical_error_probability` 0.001 says each
  physical operation on the QPU fails with probability one in a
  thousand.
- `execution done: 15000000 ticks`. A **tick** is the engine's integer
  unit of time, and one microsecond is a million ticks
  (`decsim/config.py`). So the QPU finished its quantum work after 15
  microseconds: 15 rounds at one microsecond each.
- `fully done: 37356000 ticks`. The classical loop finished 37
  microseconds in. The gap between the two numbers is the point of the
  whole simulator: the decoder was still working long after the QPU
  stopped.
- `operation 1: logical_observables, observables (0,), truth (0,)`. The
  workload was one logical operation. Its **logical observable** is the
  one bit of information the encoded qubit was holding. `observables
  (0,)` is what the decoder concluded, `truth (0,)` is what Stim knows
  it really was. They agree, so this shot did not fail.

Your `fully done` number will differ from the one above; `execution
done` will not, since the QPU's rounds do not wait on the decoder here.
The default decoder is a real PyMatching call, and decsim charges the
decoder unit the wall-clock time that call actually took on your machine
(`decsim/decoders/decoder.py`, `decode_timed`). A faster computer gives
a faster machine. [Time](../explanation/time.md) says what that means and
how to run a timing study that does not depend on your hardware.

## Step 3. Run the sweep and get a run folder

With `--seed`, `decsim run` runs one shot: it prints to the terminal
and keeps that shot's record in a folder. Without it, `decsim run` runs
every point of the yaml's sweep until its collection stops it:

```bash
decsim run configs/reference.yaml --out results/reference
```

It first prints what the yaml resolved to, one line per component, then
one line per point as it finishes:

```
{"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.001}: 2 shots done (cap)
```

The numbers are in the folder's `sweep.csv`, one row per point, which
step 5 reads. Two new words for its columns:

- A **window** is a slice of rounds that one decode covers. A decoder
  does not wait for the whole run before deciding: it takes the rounds a
  few at a time, decodes that slice, commits the part of it it is sure
  about, and slides on. Here each shot's 15 rounds became four windows.
- A **frame commit** is the moment the correction for a window is
  written into the **Pauli frame**, the running record of every
  correction decided so far. The `buffer0_ready_to_frame_median_us`
  column is therefore the time from a window having all its rounds to
  its correction being recorded, which is the reaction time this
  configuration achieves.

The `load` column is the ratio of the time a window spends being
decoded to the time between windows arriving. Above 1 the decoder
cannot keep up, and work queues. PyMatching in Python on a small window
is slow compared to one microsecond a round, so it is often above 1
here; the value is this host's, like every tick of the row.

A `_p99_us` column is the 99th percentile: ninety-nine windows in a
hundred finished within it.

## Step 4. Open the run folder

```bash
ls results/reference
```

```
config
latency_samples.csv
pieces
points
run.json
shot_links.csv
shots.csv
sweep.csv
timeline.png
trace
window_samples.csv
```

`--out` names the results folder. `pieces/` holds each point's shots
in pieces, each saved whole the moment it ends, and running the command
again into the same folder runs only the pieces it has not saved. The
csv files at the top are folded from the pieces. Without `--out`,
`decsim run` writes a new folder, `results/<date>_<name>/`, and a
second one the same day gets `_2`, so no two runs share one.

```bash
ls results/reference/points/*
```

```
inputs
machine.json
```

`points/` holds one folder per point. A point of a yaml is named by the
first twelve characters of its id. The folder is written under
`results/`, which is output and is not tracked by git. `config/` holds
a verbatim copy of the yaml files that produced it, `run.json` the git
commit and the command line, each `machine.json` every value its point
ran with and the function that made its workload, `inputs/` the
workload the point ran, and the csv files the facts.
[The run folder](../reference/run_folder.md) has one row per file.

## Step 5. Read one row and one figure

`sweep.csv` has one row per sweep point and more than a hundred
columns. The distance and the first counts:

```bash
cut -d, -f3,6,8,11-14 results/reference/sweep.csv
```

```
qpu.distance,shots,logical_failures,state,logical_error_rate_estimate,logical_error_rate_low,logical_error_rate_high
3,2,0,cap,,,0.841886116991581
```

`state` says why the point stopped: `cap`, at the two shots its
`collection` allows. `logical_error_rate_estimate` is the fraction of
scored shots whose decoded observable did not match the truth, and the
two limits bracket the true rate with 95 percent confidence. With no
failure in two shots there is no fraction to quote, only an upper
limit: the true rate is below 0.84. Two shots say almost nothing. The
next tutorial, [Your first sweep](first_sweep.md), explains the limits
and runs enough shots to make them narrow.

The row's first columns name its point. `point_id`, left out here, is a
hash of every setting the point ran with, so two points that differ in
any setting have two ids. After it comes one column per yaml path the
sweep sets, named by that path, holding the point's value there: the
values in braces on the point's progress line. Every other
csv file of the folder names its rows the same way, so a table of any
of them groups by a setting with no parsing.

`decsim run` also drew a figure. Draw a second one:

```bash
decsim plot results/reference --figure stage_breakdown
```

```
results/reference/stage_breakdown.png
```

`stage_breakdown.png` shows where a window's time went, stage by stage:
the buffer filling, the queue, the link into the decoder unit, the
fetch, the algorithm, the release, the boundary handed to the next
window, the link out and the frame commit, one bar per sweep point.
These two figures read decsim's own records, a trace and the stage
columns in pipeline order. A figure of the sweep's numbers against a
setting is yours to draw from the csv files, since only you know
which setting belongs on the axis and what the figure should look
like.

## Step 6. Follow one round

The `--trace` in step 2 and the `trace: chrome` key in the reference
yaml both ask for the same thing: one file per traced shot recording
where every round and every window sat, for how long, and what moved it.
The file is in the Chrome Trace Event Format, so
[ui.perfetto.dev](https://ui.perfetto.dev) opens it as a timeline with
one lane per component. Drag the file in and zoom.

For one round, decsim prints the path itself:

```bash
decsim trace follow \
  results/reference/trace/*_seed0.trace.json \
  --round 1:1
```

```
round 1:1 of decsim weak_baseline d3 seed0

tick (us)  where                        what                                                                 dur (us)  transfer   bits
0.000      weak syndrome buffer         hold registered                                                                reference
1.000      QPU                          emitted round 1                                                                           8
1.000      qpu_to_controller            move                                                                 0.004     move       8
1.000      Controller                   controller intake copy                                                         copy       8
1.004      Controller                   controller assembler copy                                                      copy       8
1.004      Controller                   residence, unbounded, freed at packed                                0.000     copy       8
1.004      controller_to_weak_buffer    move                                                                 0.006     move       4
1.010      weak syndrome buffer         weak syndrome buffer copy                                                      copy       4
1.010      weak syndrome buffer         residence, unbounded, data ready 1.010, freed at last hold released  5.006     copy       4
6.012      Window planner               W0 ready
6.012      weak_buffer_to_weak_decoder  move, with W0 rounds 1:1..6                                          0.004     move       44
6.016      Decoder unit default#0       unit default#0 memory copy                                                     copy       44
6.016      Decoder unit default#0       residence, unbounded, data ready 6.016, freed at decode done         11.548    copy       44

copies 4, references 1 job and 1 hold, moves 3
longest residence: 11.548 us in Decoder unit default#0 (residence, unbounded, data ready 6.016, freed at decode done)
longest queue wait: none
```

`1:1` is operation 1, round 1. Read it top to bottom as one round's
life. The QPU emitted 8 raw measurement bits at 1 microsecond. They
moved to the controller, which turned them into **detection events**: a
detection event is a check whose value changed from the round before,
which is what a decoder actually reads. Four bits left the controller
rather than eight, because in the first round of a memory experiment
only half the checks have a value to compare against
(`decsim/detector_error_model/detection_event_formation.py`,
`form_at`, which the controller's assembler calls with its own seat). That
round moved into the weak syndrome buffer, the store the decoder reads from, and sat
there 5 microseconds waiting for the rest of its window. At 6.012 microseconds window 0 had all six of its rounds, so
all 44 bits moved together into the decoder unit's memory, and the
decode held that unit for the wall clock PyMatching took, 11.5
microseconds on this host.

The `transfer` column is the vocabulary decsim uses for data movement: a
**move** leaves the bits behind, a **copy** ends with both sides holding
them, and a **reference** hands over an object both sides read. The
summary line counts two kinds of reference. A **job** is one window's
request to a decoder unit. A **hold** is a note a reader puts on a round
saying it may still need it, so the buffer may not drop the round; the
round itself does not move.
[The data path, hop by hop](../explanation/data_path.md) walks every hop.

The same command follows a window instead of a round:

```bash
decsim trace follow \
  results/reference/trace/*_seed0.trace.json \
  --window 1:0
```

That path ends where the round's path leaves off: the decode's stages,
the verdict, the boundary passed to the next window, and the commit into
the frame.

## What you learned

- One round, one window, one shot, one sweep point.
- The reaction time is the interval from a window having its rounds to
  its correction reaching the Pauli frame, and the run folder measures
  it.
- The trace is the data path, and `decsim trace follow` reads it.

## Read next

- [Your first sweep](first_sweep.md): run a real sweep and read the error
  bars.
- [Architecture](../explanation/architecture.md): what each component is.
- `configs/reference.yaml`: every key you can set, with its source.
