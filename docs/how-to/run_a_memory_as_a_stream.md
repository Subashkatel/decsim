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

Any task whose workload is decoded Stim memory patches streams, as
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

Several patches (`MemorySettings(patches=N)`) stream side by side, each
its own stream with its own readout, on whatever graph the task builds:
a pool per patch behind a router, a shared strong pool, a chain
([How to route windows to several pools](route_windows_to_several_pools.md)).

## 2. Choose the rounds and the warmup

- `--rounds N` is the stream's length. Each patch's readout is
  scheduled at round N, and the stream fails when any patch's outcome
  is wrong.
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
| `logical_failure` | whether any patch's end readout was predicted wrong |
| `decoded_windows` | every window decoded, the warmup's too |
| `recorded_windows` | the windows of the region of interest |
| `values` | each latency's count, mean and maximum, in microseconds |
| `window_tiers` | the recorded windows by the pool that answered |
| `weak_reaction_growth_rate` | the reaction time's growth per microsecond of the stream; above 0 the tier falls behind |
| `deadline_missed_windows` | the windows over the reaction budget, when one is set |
| `max_queued_windows` | the most windows waiting at once |
| `pools` | each tier's queue peak, busy fraction and offered load, its pools read together |
| `pool_rows` | each pool's own queue peak, busy fraction, offered load, committed and escalated windows past the warmup, as `shot_pools.csv` holds for a shot |
| `statistics` | every part's statistic rows over the region of interest |
| `escalation_series` | every hundredth committed window: the windows committed, the escalated among them and its first committed round |
| `checkpoints` | one row a checkpoint, when the run closes checkpoints (section 4); empty otherwise |

A task whose threshold moves (a `RateThreshold` or `OnlineThreshold`)
adds `online_threshold`: the source's summary and its trajectory of
(windows, threshold in nats, event).

The means are exactly those `decsim run` reports for the same windows,
and so is every other number both compute (`escalation_series` is the
stream's alone)
(`tests/experiments/test_stream_measures.py`).

## 4. Measure the logical error rate during the stream

A stream has one readout, so it has one logical outcome. A checkpoint
gives the outcome the stream would have had if it had been read out at
an earlier round, and the stream goes on unchanged. Ask for one every
K rounds:

```
decsim stream run.py --rounds 2000 --seed 1 --set observation.checkpoint_every_rounds=50 --out s1.json
```

At each checkpoint the source copies its state and reads the copy out
without noise (`decsim/records/checkpoints.py`). The windows that read
only earlier rounds keep their final answers, because they are the
same in a stream that ends there. PyMatching decodes the rest of the
rounds as one last window, with the copy's readout and the boundary
the last kept window hands on (`decsim/observe/checkpoints.py`). This
takes no simulated time, and no decoder or threshold reads it. Tan et
al. measure their per-round rates the same way (2209.09219, Section
S3 C).

Each row of `checkpoints` holds:

- `round` is the checkpoint's round. The stream's last round is always
  a checkpoint, and that row's outcome is the stream's own.
- `failed` holds one outcome for each patch, true when the prediction
  was wrong, or null when a kept window gave no prediction.
- `windows` and `escalated_windows` count the kept windows and the
  ones the policy sent to the strong decoder. Their ratio is the
  escalated share up to that round.

Give the memory noiseless time boundaries
(`noiseless_time_boundaries`), so that the stream's own last readout is
as clean as the checkpoints' readouts. Then run many seeds and fit the
per-round rate with its interval. The interval comes from a bootstrap
over whole streams, because the checkpoints of one stream share its
rounds:

```python
streams = [json.loads(path.read_text()) for path in sorted(folder.glob("s*.json"))]
rows = [stream["measurement"]["checkpoints"] for stream in streams]
rounds = [row["round"] for row in rows[0]]
failed = [[any(row["failed"]) for row in stream_rows] for stream_rows in rows]
fit = failure_statistics.per_round_rate_over_checkpoints(rounds, failed)
```

`fit.per_round_rate` is the rate, and `fit.low` and `fit.high` bound its
95 percent interval (`decsim/experiments/failure_statistics.py`). A
checkpoint round with no failure in any stream is refused, so leave the
early rounds out. With a moving threshold, the rate is the average over
the rounds fitted, and `online_threshold` gives the threshold at each
window count.

## What a stream does not build yet

Each of these is refused, "not implemented yet", with its reason:

- a workload other than decoded Stim memory patches, such as a memory
  with no circuit, a circuit with no REPEAT block (Stim writes one from
  3 rounds on), or a memory that decodes nothing;
- a source other than the Stim device;
- the trace, the log, and the data movement and window outcome records,
  which grow with every round;
- batched or bulk strong decoding, whose merged batch the per-window
  measures do not read;
- the double window on a stream, which absorbs windows a stream has not
  laid yet;
- a logical failure counted inside the stream other than at a
  checkpoint (section 4).

[How to build each switching pattern](build_each_switching_pattern.md#what-is-refused-for-now)
lists every refusal.
