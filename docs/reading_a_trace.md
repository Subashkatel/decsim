# Reading a trace

A trace is one shot's data path written out: where every round and every
window sat, for how long, what moved it, and whether each hop copied the
bits or referenced them. Ask for one with `trace: chrome` in the yaml's
`observation` section, or with `--trace` on `decsim run`. A traced shot
writes one file, `results/<run>/trace/<shot>.trace.json`, and only the
shots `trace_shots` names are traced.

The file is the Chrome Trace Event Format, the format Perfetto and
`chrome://tracing` read. Nothing about a run changes when the trace is
on: the writer schedules nothing and calls no component, so the ticks,
the log and the results are the same either way.

## Open it

Two ways, and they answer different questions.

**The whole shot, as a timeline.** Open the file at
[ui.perfetto.dev](https://ui.perfetto.dev) and drop it in. You get one
lane per component and one per wired link path, in pipeline order, so
the lanes read top to bottom the way the data flows: QPU,
`qpu_to_controller`, Controller, `controller_to_weak_buffer`, Buffer 0,
`controller_to_strong_buffer`, Buffer 1, Window planner, the decoder
links, one lane per decoder unit, Frame, and the two paths back to the
QPU. Perfetto gives zoom, nested slices, counter tracks, the flow arrows
of one selected slice, an args pane, and SQL over the whole file.

**One round or one window, as a path.** The command prints the hops as a
table, in tick order, which is what Perfetto would make you assemble by
clicking through flow arrows:

```bash
decsim trace follow results/<run>/trace/<shot>.trace.json --round 1:1
decsim trace follow results/<run>/trace/<shot>.trace.json --window 1:0
decsim trace follow results/<run>/trace/<shot>.trace.json --round 1:1 --html path.html
```

The keys are the code's own: a round is `operation_id:round_index`
(counted from 1), a window is `operation_id:window_id`. A key the trace
does not carry is refused with the keys it does carry. `--html` writes
the same path as one self-contained page, one lane per component.

Under the table come the counts: how many hops copied the bits, how many
referenced them as a job or a hold, how many moved them, and the longest
residence and the longest queue wait on that path.

## What is in it

| In the file | What it is |
| --- | --- |
| a complete event (`X`) on a component's lane | a residence or a service: a round sitting in a store, a job in a unit, a decode on a unit's compute. Its `args` carry the capacity, when the slot was taken, when the data was ready, and why it was freed |
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

## Two things worth knowing

- A decoder unit's internal stages (fetch, algorithm, release) are
  nested inside its service slice, so a decode's own time is visible
  apart from the time its input took to arrive.
- `decsim collect` draws `timeline.png` from a trace file, not from the
  machine, so a point that traced no shot draws no timeline.

## Read next

- `configs/reference.yaml`, the `observation` section: every trace knob
  and what it costs.
- `decsim/observe/trace_writer.py`: which event each source becomes.
- `decsim/front/trace_follow.py`: how the table is built.
