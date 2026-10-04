[decsim docs](../README.md) › [Tutorials](README.md)

# Your first run

This lesson takes about ten minutes. By the end you will have read a
run file, run one shot of it, collected one point into a results folder
and followed one round of syndrome data through the machine.

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
BP-OSD and belief matching decoders), numpy and scipy. Without it
decsim imports but cannot run a shot.

## Step 2. Read the run file

An experiment is a Python file that builds a list of machines. This
lesson's ships with decsim as `examples/priced_cards_example.py`. Its
heart is one loop:

```python examples/priced_cards_example.py
points = []
for distance in DISTANCES:
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        ROUND_PERIOD_MICROSECONDS,
    )
    card = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=DECODE_MICROSECONDS
    )
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=card)
    machine = dataclasses.replace(base, weak_decoder=weak_decoder)
```

- `weak_decoder_baseline` returns a `MachineSettings`: one record for
  each part of one machine, the QPU, the controller, the links, the
  syndrome buffer, the windows, the decoder and the Pauli frame. It has
  one decoder, which decodes everything. Its shot is ten rounds for each
  unit of distance, 30 at d = 3;
  `dataclasses.replace(base, workload=machine_settings.memory_workload(3, 0.001, 300))`
  makes it 300.
- `dataclasses.replace` changes one field of a record and keeps the
  rest. Here the decoder's algorithm becomes a **priced card**:
  PyMatching still decodes every window, but each decode is charged
  `DECODE_MICROSECONDS`, 1.0 microseconds, however fast your computer
  is. So every tick on this page is the same on your machine.

After the loop, each machine becomes a `decsim.Point` with a name, `d3`
for distance 3, and the points become a `decsim.Experiment`. Its
`COLLECTION` says when a point stops: here at 20 shots.

## Step 3. Run one shot

```bash
decsim run examples/priced_cards_example.py --seed 0 --trace --out results/first_shot
```

A **shot** is one complete run of the workload from start to finish,
with one random seed. `--seed` runs one shot of the first point.
`--trace` asks for a record of the data path, which step 7 reads, and
`--out` names the folder it goes in.

The output:

```
config: priced_cards_example
point: {"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.001} seed 0
terminal status: complete
execution done: 30000000 ticks
fully done: 31084000 ticks
operation 1: logical_observables, observables (0,), truth (0,)
run dir: results/first_shot
```

Line by line:

- `point: {...} seed 0`. The point's settings. `qpu.distance` is the
  code distance, the size of the error correcting code: distance 3
  corrects one error. `qpu.round_period_microseconds` is how long one
  round of measurement takes on the QPU. The physical error probability
  0.001 says each physical operation on the QPU fails with probability
  one in a thousand.
- `execution done: 30000000 ticks`. A **tick** is the engine's integer
  unit of time, and one microsecond is a million ticks. So the QPU
  finished its quantum work after 30 microseconds: 30 rounds at one
  microsecond each.
- `fully done: 31084000 ticks`. The classical loop finished 1.084
  microseconds later. That gap, the decoder still working after the QPU
  stopped, is what decsim exists to measure.
- `operation 1: logical_observables, observables (0,), truth (0,)`. The
  workload was one logical operation. Its **logical observable** is the
  one bit of information the encoded qubit was holding. `observables
  (0,)` is what the decoder concluded, `truth (0,)` is what Stim knows
  it really was. They agree, so this shot did not fail.

To watch the shot as it runs, add `--log print`. decsim then prints
the engine's log: one line for each step a part takes, with its time.

```bash
decsim run examples/priced_cards_example.py --seed 0 --log print --out results/first_shot_log
```

The log starts:

```
[  0.000 us] Controller: START memory  (Clifford, qubits (0,))
[  1.000 us] QPU: memory fires round 1/30
[  1.008 us] Decoder manager: round 1 of memory arrived (op now has rounds 1..1)
```

## Step 4. Collect one point

Without `--seed`, `decsim run` runs every point until its collection
stops it. `--only` picks one point by its name:

```bash
decsim run examples/priced_cards_example.py --only d3 --out results/d3
```

It first prints the folder it writes, then one line per point as the
point finishes:

```
run dir: results/d3

d3: 20 shots done (cap)
```

The numbers are in the folder's `sweep.csv`, one row per point. These
are the columns this step reads:

```bash
cut -d, -f5,35,73,97,130,131 results/d3/sweep.csv
```

```
algorithm,load,queue_wait_mean_us,service_mean_us,buffer0_ready_to_frame_median_us,buffer0_ready_to_frame_p99_us
1.0,0.35585185185185186,0.0,1.064,1.076,1.076
```

Two new words for these columns:

- A **window** is a slice of rounds that one decode covers. A decoder
  does not wait for the whole run before deciding: it takes the rounds a
  few at a time, decodes that slice, commits the part of it it is sure
  about, and slides on.
- A **frame commit** is the moment the correction for a window is
  written into the **Pauli frame**, the running record of every
  correction decided so far. `buffer0_ready_to_frame_median_us` is the
  median time from a window having all its rounds to its correction
  being recorded: the reaction time this machine achieves.

`algorithm` is the card, 1.0 microseconds. `service_mean_us` adds the
decoder unit's own cycles to it: one cycle a round to fetch the six
rounds of a window and ten to release the result, at 4 nanoseconds a
cycle, 0.064 microseconds in all. `load` is that service time divided
by the time between windows arriving, about three microseconds. Above 1
the decoder cannot keep up and work queues; at 0.36 nothing waits, so
`queue_wait_mean_us` is zero and every window's reaction time is the
same. A `_p99_us` column is the 99th percentile: ninety-nine windows in
a hundred finished within it.

The run counted no copies, references or moves, so the folder has no
`shot_data_movement.csv`: the `data_movement` field of the machine's
`ObservationSettings` is off by default.

## Step 5. Open the results folder

```bash
ls results/d3
```

```
pieces
points
priced_cards_example.py
run.json
shot_links.csv
shots.csv
sweep.csv
window_samples.csv
```

`pieces/` holds the point's shots in pieces, each saved whole the moment
it ends, and running the command again into the same folder runs only
the pieces it has not saved. The csv files are folded from the pieces.
`priced_cards_example.py` is a copy of the run file, and `run.json`
holds the git commit and the command line. A checkout with changes git
has not committed also gets `code_state.patch`, those changes as a
patch, and `run.json` records the patch's sha256. Without `--out`,
`decsim run` writes a new folder,
`results/<date>_<name>/`.

```bash
ls results/d3/points/*
```

```
inputs
machine.json
```

`points/` holds one folder per point, named by the point's name.
`machine.json` holds every value the point ran with, and `inputs/` the
workload it ran. [The run folder](../reference/run_folder.md) has one
row per file.

## Step 6. Read one row

`sweep.csv` has one row per point and more than a hundred columns. The
distance and the first counts:

```bash
cut -d, -f3,6,8,11-14 results/d3/sweep.csv
```

```
qpu.distance,shots,logical_failures,state,logical_error_rate_estimate,logical_error_rate_low,logical_error_rate_high
3,20,0,cap,,,0.1684334709830853
```

`state` says why the point stopped: `cap`, at the 20 shots its
collection allows. `logical_error_rate_estimate` is the fraction of
scored shots whose decoded observable did not match the truth, and the
two limits bracket the true rate with 95 percent confidence. With no
failure in 20 shots there is no fraction to quote, only an upper limit:
the true rate is below 0.17. The next tutorial,
[Your first sweep](first_sweep.md), explains the limits and runs enough
shots to make them narrow.

The row's first columns name its point: `point_id`, a hash of every
setting the point ran with, then the point's settings as its metadata
names them. Every other csv file of the folder names its rows the same
way.

## Step 7. Follow one round

The `--trace` in step 3 wrote one file recording where every round and
every window sat, for how long, and what moved it. The file is in the
Chrome Trace Event Format, so [ui.perfetto.dev](https://ui.perfetto.dev)
opens it as a timeline with one lane per component.

For one round, decsim prints the path itself:

```bash
decsim trace follow \
  results/first_shot/trace/*_seed0.trace.json \
  --round 1:1
```

```
round 1:1 of point d3, decsim weak_baseline d3 seed0

tick (us)  where                        what                                                                 dur (us)  transfer   bits
0.000      weak syndrome buffer         hold registered                                                                reference
1.000      QPU                          emitted round 1                                                                           8
1.000      qpu_to_controller            move                                                                 0.004     move       8
1.000      Controller                   controller intake copy                                                         copy       8
1.004      Controller                   controller assembler copy                                                      copy       8
1.004      Controller                   residence, unbounded, freed at packed                                0.000     copy       8
1.004      controller_to_weak_buffer    move                                                                 0.004     move       4
1.008      weak syndrome buffer         weak syndrome buffer copy                                                      copy       4
1.008      weak syndrome buffer         residence, unbounded, data ready 1.008, freed at last hold released  5.004     copy       4
6.008      Window planner               W0 ready
6.008      weak_buffer_to_weak_decoder  move, with W0 rounds 1:1..6                                          0.004     move       44
6.012      Decoder unit default#0       unit default#0 memory copy                                                     copy       44
6.012      Decoder unit default#0       residence, unbounded, data ready 6.012, freed at decode done         1.064     copy       44

copies 4, references 1 job and 1 hold, moves 3
longest residence: 5.004 us in weak syndrome buffer (residence, unbounded, data ready 1.008, freed at last hold released)
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
`form_at`). That round moved into the weak syndrome buffer, the store
the decoder reads from, and sat there 5 microseconds waiting for the
rest of its window. At 6.008 microseconds window 0 had all six of its
rounds, so all 44 bits moved together into the decoder unit's memory,
and the decode held that unit for 1.064 microseconds.

The `transfer` column says how data moves: a **move** leaves the bits
behind, a **copy** ends with both sides holding them, and a
**reference** hands over an object both sides read. The summary line
counts two kinds of reference. A **job** is one window's request to a
decoder unit. A **hold** is a note a reader puts on a round saying it
may still need it, so the buffer may not drop the round; the round
itself does not move.

The same command follows a window instead of a round:

```bash
decsim trace follow \
  results/first_shot/trace/*_seed0.trace.json \
  --window 1:0
```

That path ends where the round's path leaves off: the decode's stages,
the correction and the commit into the frame.

## What you learned

- A run file is Python: a list of points, each a machine.
- One round, one window, one shot, one point.
- The reaction time is the interval from a window having its rounds to
  its correction reaching the Pauli frame, and the results folder
  measures it.
- The trace is the data path, and `decsim trace follow` reads it.

## Read next

- [Your first sweep](first_sweep.md): run a real sweep and read the
  error bars.
- [Two tiers](two_tiers.md): a machine with two decoders.
- [Build a machine step by step](build_a_machine.md): the parts of one
  machine, built by hand.
