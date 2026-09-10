[decsim docs](../README.md) › [Tutorials](README.md)

# Your first run

This lesson takes about ten minutes. By the end you will have run one
shot of the reference configuration, opened the folder it wrote, read a
figure and followed one round of syndrome data through the machine.

You do not need to know anything about decsim, and the words you need
are defined as they appear. You do need a terminal and a Python 3.9 or
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
python -m pip install -e ".[run]"
```

The `run` extra brings Stim (which simulates the quantum circuit and
produces the syndrome), PyMatching (the default decoder), numpy, scipy
and matplotlib. Without it decsim imports but cannot run a shot on real
syndrome data.

## Step 2. Run one shot

```bash
decsim run configs/reference.yaml --seed 0 --trace
```

A **shot** is one complete run of the workload from start to finish,
with one random seed. `configs/reference.yaml` is the reference
configuration: it carries every key the yaml layer reads, and its sweep
is deliberately tiny so that it runs in seconds. `--trace` asks for a
record of the data path, which step 6 reads.

The output, from this page's own run at commit `172da23`:

```
config: reference
point: p0.001 d3 round period 1 us seed 0
terminal status: complete
execution done: 15000000 ticks
fully done: 60969000 ticks
operation 1: logical_observables, observables (0,), truth (0,)
run dir: results/2026-09-10T02-29-32Z-reference
```

Line by line:

- `point: p0.001 d3 round period 1 us seed 0`. One point of a sweep is
  one machine. `p0.001` is the physical error probability: each physical
  operation on the QPU fails with probability one in a thousand. `d3` is
  the code distance, the size of the error correcting code: distance 3
  corrects one error. `round period 1 us` is how long one round of
  measurement takes on the QPU, one microsecond here.
- `execution done: 15000000 ticks`. A **tick** is the engine's integer
  unit of time, and one microsecond is a million ticks
  (`decsim/config.py`). So the QPU finished its quantum work after 15
  microseconds: 15 rounds at one microsecond each.
- `fully done: 60969000 ticks`. The classical loop finished 61
  microseconds in. The gap between the two numbers is the point of the
  whole simulator: the decoder was still working long after the QPU
  stopped.
- `operation 1: logical_observables, observables (0,), truth (0,)`. The
  workload was one logical operation. Its **logical observable** is the
  one bit of information the encoded qubit was holding. `observables
  (0,)` is what the decoder concluded, `truth (0,)` is what Stim knows
  it really was. They agree, so this shot did not fail.

Your two tick numbers will differ from the ones above. The default
decoder is a real PyMatching call, and decsim charges the decoder unit
the wall-clock time that call actually took on your machine
(`decsim/decoders/decoder.py`, `decode_timed`). A faster computer gives
a faster machine. [Time](../explanation/time.md) says what that means and
how to run a timing study that does not depend on your hardware.

## Step 3. Run the sweep and get a run folder

`decsim run` is one shot and prints to the terminal. To get a folder of
results, use `collect`, which runs every point of the yaml's sweep:

```bash
decsim collect configs/reference.yaml
```

It first prints what the yaml resolved to, one line per component, then
the summary of each point:

```
p 0.001, d 3, round period 1.0 us: 2 shots done
distance: 3
physical error rate: 0.001
algorithm: pymatching
round period: 1 us
load (service per window / window inter-arrival): 7.31
logical failures: 0 of 2 shots
mismatches vs direct PyMatching: 0
throughput: 0.267 rounds per us
queue wait, mean: 7.150 us
service time per window, mean: 21.922 us
ready to frame commit: median 32.910 us, p99 42.246 us

every column: results/2026-09-10T02-29-33Z-reference/sweep.csv
```

Two new words:

- A **window** is a slice of rounds that one decode covers. A decoder
  does not wait for the whole run before deciding: it takes the rounds a
  few at a time, decodes that slice, commits the part of it it is sure
  about, and slides on. Here each shot's 15 rounds became four windows.
- A **frame commit** is the moment the correction for a window is
  written into the **Pauli frame**, the running record of every
  correction decided so far. `ready to frame commit` is therefore the
  time from a window having all its rounds to its correction being
  recorded, which is the reaction time this configuration achieves.

`load: 7.31` is the ratio of the time a window spends being decoded to
the time between windows arriving. Above 1 the decoder cannot keep up,
and work queues. It is 7.31 here because PyMatching in Python on a small
window is slow compared to one microsecond a round.

## Step 4. Open the run folder

```bash
ls results/2026-09-10T02-29-33Z-reference
```

```
config
latency_samples.csv
links.csv
manifest.json
shot_links.csv
shots.csv
sweep.csv
timeline.png
trace
window_samples.csv
```

Your folder has a different name: it is stamped with the UTC time the
run started, so no two collects ever share one. Substitute yours in the
commands below.

The folder is written under `results/`, which is output and is not
tracked by git. `config/` holds a verbatim copy of the yaml files that
produced it, `manifest.json` holds the git commit, the library versions
and the command line, and the csv files hold the facts. Every summary is
computed when a file is read, never stored, so two folders of the same
sweep can be added together. [The run folder](../reference/run_folder.md) has one row
per file.

## Step 5. Read one row and one figure

`sweep.csv` has one row per sweep point and 83 columns. The first few:

```bash
cut -d, -f1-10 results/2026-09-10T02-29-33Z-reference/sweep.csv
```

```
distance,physical_error_probability,algorithm,round_period_us,shots,windows_per_shot,logical_failures,logical_error_rate,ler_wilson_low,ler_wilson_high
3,0.001,pymatching,1.0,2,4.0,0,0.0,0.0,0.6576280471103807
```

`logical_error_rate` is the fraction of shots whose decoded observable
did not match the truth: zero out of two here. `ler_wilson_low` and
`ler_wilson_high` bracket it. Two shots say almost nothing, which is why
the interval runs from 0 to 0.66. The next tutorial,
[Your first sweep](first_sweep.md), explains that interval and runs enough
shots to make it narrow.

`collect` also drew a figure. Draw a second one:

```bash
decsim plot results/2026-09-10T02-29-33Z-reference --figure stage_breakdown
```

```
results/2026-09-10T02-29-33Z-reference/stage_breakdown.png
```

`stage_breakdown.png` shows where a window's time went, stage by stage:
waiting in the queue, crossing the link into the decoder unit, fetching,
running the algorithm, and releasing the answer. `timeline.png`, which
`collect` drew for you, shows one traced shot as a timeline. `collect`
draws a figure only when its input is there, which is why this run has a
timeline and no others: it traced a shot, but its sweep has one physical
error rate and one distance.

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
  results/2026-09-10T02-29-33Z-reference/trace/p0.001_d3_algopymatching_round1us_seed0.trace.json \
  --round 1:1
```

```
round 1:1 of decsim weak_baseline d3 p0.001 seed0

tick (us)  where                        what                                                                 dur (us)  transfer   bits
0.000      Buffer 0                     hold registered                                                                reference
1.000      QPU                          emitted round 1                                                                           8
1.000      qpu_to_controller            move                                                                 0.004     move       8
1.000      Controller                   controller intake copy                                                         copy       8
1.004      Controller                   controller assembler copy                                                      copy       8
1.004      Controller                   residence, unbounded, freed at packed                                0.000     copy       8
1.004      controller_to_weak_buffer    move                                                                 0.006     move       4
1.004      Buffer 0                     Buffer 0 copy                                                                  copy       4
1.004      Buffer 0                     residence, unbounded, data ready 1.010, freed at last hold released  5.012     copy       4
6.012      Window planner               W0 ready
6.012      weak_buffer_to_weak_decoder  move, with W0 rounds 1..6                                            0.004     move       44
6.016      Decoder unit default#0       unit default#0 memory copy                                                     copy       44
6.016      Decoder unit default#0       residence, unbounded, data ready 6.016, freed at decode done         16.926    copy       44

copies 4, references 1 job and 1 hold, moves 3
longest residence: 16.926 us in Decoder unit default#0 (residence, unbounded, data ready 6.016, freed at decode done)
longest queue wait: none
```

`1:1` is operation 1, round 1. Read it top to bottom as one round's
life. The QPU emitted 8 raw measurement bits at 1 microsecond. They
moved to the controller, which turned them into **detection events**: a
detection event is a check whose value changed from the round before,
which is what a decoder actually reads. Four bits left the controller
rather than eight, because in the first round of a memory experiment
only half the checks have a value to compare against
(`decsim/controller/round_assembly.py`, `form_before_departure`). That
round moved into Buffer 0, the store the decoder reads from, and sat
there 5 microseconds waiting for the rest of its window. At 6.012 microseconds window 0 had all six of its rounds, so
all 44 bits moved together into the decoder unit's memory, and the
decode held that unit for 16.9 microseconds.

The `transfer` column is the vocabulary decsim uses for data movement: a
**move** leaves the bits behind, a **copy** ends with both sides holding
them, and a **reference** hands over an object both sides read.
[The data path, hop by hop](../explanation/data_path.md) walks every hop.

The same command follows a window instead of a round:

```bash
decsim trace follow \
  results/2026-09-10T02-29-33Z-reference/trace/p0.001_d3_algopymatching_round1us_seed0.trace.json \
  --window 1:0
```

That path ends where the round's path leaves off: the decode's stages,
the verdict, the boundary passed to the next window, and the commit into
the frame at 22.950 microseconds.

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
