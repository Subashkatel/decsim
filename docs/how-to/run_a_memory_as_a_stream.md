[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a memory as a stream

A memory task runs its circuit for a fixed number of rounds, shot after
shot. `decsim stream` runs the same memory as one long stream of rounds,
read out once at the end, and measures the decoders as the stream goes.
The machine lets each window go once it is final, so a stream of a
million rounds runs in the memory a stream of ten thousand does. Use it
to see whether a decoder system keeps up over a long computation: how
its reaction time and its queue grow, and when it first misses a
deadline.

## 1. Pick a memory task

Any task whose workload is one decoded Stim memory patch streams, as
every task of `examples/my_first_sweep.py` is:

```
decsim stream examples/my_first_sweep.py --only d3 --rounds 3000 --warmup 300 --out d3_stream.json
```

The task's own machine runs: its decoders, links, stores and verdicts.
Two things change. The rounds come from the flip stream source
(`decsim/qpu/flip_stream_device.py`), which samples the task's circuit
one round at a time, and the memory's circuit is cut into the rounds a
stream repeats (`decsim/frontends/memory_stream.py`). A task with no
Pauli frame gets the default one, since a window is measured when the
frame commits it.

## 2. Choose the rounds and the warmup

- `--rounds N` is the stream's length. The readout is scheduled at round
  N, and its logical outcome is the stream's one outcome.
- `--warmup W` leaves the first W rounds out of the region of interest.
  The statistics restart after round W, and the latencies are sampled
  from the windows that commit after it. It is below N.
- `--seed S` is the stream's root seed, 0 by default.
- `--set PATH=VALUE` sets any parameter, as `decsim run` does. A budget
  on the reaction time, for instance, counts the windows over it:
  `--set observation.reaction_deadline_microseconds=3`.

Without `--out` the record is printed.

## 3. Read the record

The record is one json object: the task's name and id, the rounds, the
warmup, the seed, and the measurement
(`decsim/experiments/stream_measures.py`, StreamMeasurement):

| Field | What it holds |
|---|---|
| `rounds` | the rounds read out |
| `logical_failure` | whether the end readout's prediction was wrong |
| `decoded_windows` | every window decoded, the warmup's too |
| `recorded_windows` | the windows of the region of interest |
| `values` | each latency's count, mean and maximum, in microseconds |
| `window_tiers` | the recorded windows by the pool that answered |
| `weak_reaction_growth_rate` | the reaction time's growth per microsecond of the stream; above 0 the tier falls behind |
| `deadline_missed_windows` | the windows over the reaction budget, when one is set |
| `max_queued_windows` | the most windows waiting at once |
| `statistics` | every part's statistic rows over the region of interest |

The means are exactly those `decsim run` reports for the same windows,
and so is every other number both compute
(`tests/experiments/test_stream_measures.py`).

## What a stream does not build yet

Each of these is refused, "not implemented yet", with its reason:

- a workload other than one decoded Stim memory patch, such as several
  patches, a memory with no circuit, a circuit with no REPEAT block
  (Stim writes one from 3 rounds on), or a memory that decodes nothing;
- a source other than the Stim device;
- the trace, the log, and the data movement and window outcome records,
  which grow with every round;
- batched or bulk strong decoding, whose merged batch the per-window
  measures do not read;
- a graph of a new shape;
- the double window on a stream, which absorbs windows a stream has not
  laid yet;
- a logical failure counted inside the stream, which no source defines.

[How to build each switching pattern](build_each_switching_pattern.md#what-is-refused-for-now)
lists every refusal.
