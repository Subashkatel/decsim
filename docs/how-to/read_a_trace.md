# How to read a trace and follow one round or one window

A **trace** is one shot's data path written out: where every round and
every window sat, for how long, what moved it, and whether each hop
copied the bits or referenced them.

## 1. Ask for one

Either in the yaml's `observation` section:

```yaml
observation:
  trace: chrome
```

or on the command line for a single shot:

```bash
decsim run configs/reference.yaml --seed 0 --trace
```

A traced shot writes one file, `results/<run>/trace/<shot>.trace.json`,
and only the shots `observation.trace_shots` names are traced.

Turning the trace on changes nothing about the run. The writer schedules
nothing and calls no component, so the ticks, the log and the results
are the same either way.

## 2. For the whole shot, use a viewer

The file is the Chrome Trace Event Format. Open
[ui.perfetto.dev](https://ui.perfetto.dev) and drop the file in. You get
one lane per component and one per wired link path, in pipeline order,
so the lanes read top to bottom the way the data flows: the QPU,
`qpu_to_controller`, the controller, `controller_to_weak_buffer`, Buffer
0, `controller_to_strong_buffer`, Buffer 1, the window planner, the
decoder links, one lane per decoder unit, the frame, and the two paths
back to the QPU. Perfetto gives zoom, nested slices, counter tracks, the
flow arrows of a selected slice, an argument pane, and SQL over the
whole file.

## 3. For one round, use `trace follow`

A viewer makes you assemble a path by clicking through flow arrows. The
command prints it:

```bash
decsim trace follow \
  results/<run>/trace/<shot>.trace.json --round 1:1
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

A round key is `operation_id:round_index`, counted from 1. Read the
table in tick order as one round's life. Under it come the counts: how
many hops copied the bits, how many referenced them as a job or a hold,
how many moved them, and the longest residence and queue wait on that
path.

## 4. For one window, the same command

```bash
decsim trace follow \
  results/<run>/trace/<shot>.trace.json --window 1:0
```

```
window 1:0 of decsim weak_baseline d3 p0.001 seed0

tick (us)  where                        what                                                          dur (us)  transfer  bits
6.012      Window planner               W0 ready
6.012      Window planner               queued, dispatched to default#0                               0.000
6.012      weak_buffer_to_weak_decoder  move, with W0 rounds 1..6                                     0.004     move      44
6.016      Decoder unit default#0       unit default#0 memory copy                                              copy      44
6.016      Decoder unit default#0       stage fetch                                                   0.024
6.016      Decoder unit default#0       decode service                                                16.926
6.016      Decoder unit default#0       residence, unbounded, data ready 6.016, freed at decode done  16.926    copy      44
6.040      Decoder unit default#0       stage algorithm                                               16.898
22.938     Decoder unit default#0       stage release                                                 0.004
22.942     Window planner               verdict
22.942     decoder_to_decoder           move, with W0 rounds 1..6                                     0.004     move      8
22.942     weak_decoder_to_frame        move, with W0 rounds 1..6                                     0.004     move      1
22.946     Frame                        residence, unbounded, committed 22.950, freed at end of run   34.115    copy
22.950     Window planner               W0 committed
22.950     Frame                        1:0 committed

copies 1, references 1 job and 0 holds, moves 3
longest residence: 34.115 us in Frame (residence, unbounded, committed 22.950, freed at end of run)
longest queue wait: 0.000 us in Window planner (queued, dispatched to default#0)
```

A window key is `operation_id:window_id`. The unit's internal stages
(fetch, algorithm, release) are nested inside its service slice, so the
decode's own time is visible apart from the time its input took to
arrive.

A key the trace does not carry is refused with the keys it does carry.

## 5. Write the path as a page

```bash
decsim trace follow \
  results/<run>/trace/<shot>.trace.json --round 1:1 --html path.html
```

`--html` writes the same path as one self-contained page, one lane per
component.

## What is in the file

| In the file | What it is |
| --- | --- |
| a complete event (`X`) on a component's lane | a residence or a service: a round in a store, a job in a unit, a decode on a unit's compute. Its arguments carry the capacity, when the slot was taken, when the data was ready, and why it was freed |
| a complete event on a link path's lane | a move, from its send tick to its delivery tick, with the bits it carried and the request it served |
| an instant event (`i`) | something with no duration: a hold registered, transferred or released, a verdict, a selection, a copy made |
| a counter event (`C`) | an occupancy at every change: Buffer 0's rounds, Buffer 1's rounds, a unit's memory rounds, the ready queue's depth |
| flow events (`s`, `t`, `f`) | one round's hops joined into a chain, and one window's chain from queue to frame |

Every event carries `args.tick`, the exact integer tick. The `ts` field
is that tick in microseconds, for the viewer to place it; every reader
that computes anything uses `args.tick`.

Residences overlap by design, because a store holds many rounds at once,
and a viewer stacks overlapping slices on extra rows. The counter track
is the exact occupancy; the rows are the viewer's arrangement.

`decsim collect` draws `timeline.png` from a trace file, not from the
machine, so a point that traced no shot draws no timeline.

`residence.csv` is read from the same files: one row per traced shot
per structure, with the mean and longest a round or window sat there,
and one per link path with the longest a move waited on the wire. A
point that traced no shot writes no row, for the same reason.

## Read next

- `configs/reference.yaml`, the `observation` section: every trace knob
  and what it costs.
- `decsim/observe/trace_writer.py`: which event each source becomes.
- `docs/explanation/data_path.md`: what each hop in the table is.
