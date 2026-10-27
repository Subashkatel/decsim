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

A checkpoint's outcome is the machine's own answer if the stream had
ended at that round. Tan et al. measure their per-round rates the same
way (2209.09219, Section S3 C):

- The source copies its state and reads the copy out without noise
  (`decsim/records/checkpoints.py`).
- The windows that read only earlier rounds keep their final answers,
  because they are the same in a stream that ends there.
- The windows after them that commit a round up to the checkpoint are
  the stream's last windows, each cut to end at the checkpoint. Each
  one gets the boundary the window before it hands on, and the copy's
  readout.
- The machine's own decoders decide each of these last windows, as
  they decide a final window: the entry pool's decoder, then the
  verdict on its confidence and, on an escalation, the strong
  decoder's answer on the same window
  (`decsim/observe/closing_decisions.py`).

This takes no simulated time. The verdict reads a copy of the
threshold, so the threshold learns nothing and no audit runs, and each
decoder is a fresh one, so the stream's numbers are the same with
checkpoints and without.

Each row of `checkpoints` holds:

- `round` is the checkpoint's round. The stream's last round is always
  a checkpoint.
- `failed` holds one outcome for each patch, true when the prediction
  was wrong, or null when a kept window gave no prediction.
- `windows` and `escalated_windows` count the kept windows and the
  ones the policy sent to the strong decoder. Their ratio is the
  escalated share up to that round.
- `threshold` holds, for each patch, the threshold its escalate edge
  read when the checkpoint closed, in nats, or null with no edge. It
  comes from the same summary as the record's `online_threshold`.

On the same seed, a checkpoint at round t has the outcome of a separate
stream of t rounds. One difference can remain: a separate stream's
moving threshold learns from its next-to-last window before it decides
its last one, and a checkpoint decides all of its last windows on the
threshold of its close.

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
the rounds fitted, and each row's `threshold` gives the threshold at
that round.

## 5. Check each window's weak and strong answers

A switching machine keeps the weak decoder's answer on a window, or
the strong decoder decodes the window again. Ask for each window's two
answers checked against the truth, on a switching machine:

```
decsim stream run.py --rounds 70 --seed 1 --set 'observation.answer_check_paths=("weak", "strong", "best_of_two")' --out s1.json
```

A window's check is a checkpoint (section 4) at the round after the
last round the window reads, or at the stream's last round if the
window reads it. At that round, a stream that ended there keeps the
window's own answer. The listener keeps its own chain of answers for
each path, takes no simulated time, and never changes the machine
(`decsim/observe/answer_checks.py`):

- `weak` keeps every weak answer, so its outcome is the weak decoder
  alone on the same shot.
- `strong` keeps every strong answer, so its outcome is the strong
  decoder alone.
- `best_of_two` keeps the weak answer unless that answer fails its
  check and the strong answer passes. It is a perfect chooser of each
  window: the per-window best-of-two line. Compare it with the
  machine's own `logical_failure`.

Each window's two answers come from the machine's own decoders, with
the chain's boundary folded in. The windows after it, up to the
check's round, are decided as in a checkpoint. A row of
`answer_checks` holds one path, and for each patch its outcome
(`failed`) and its windows' rows (`windows`):

- `window_index` and `check_round` name the window and its check.
- `weak_gap` is the weak answer's confidence in nats, or null when it
  gave none.
- `weak_failed` and `strong_failed` are each answer's check.
- `same_logical_class` says whether the two answers flip the same
  logical observables on the window, or null when one gave no answer.
- `would_escalate` is the machine's verdict on the weak answer, on a
  copy of the threshold at that time.
- `kept` is the answer the chain kept.

`decsim/experiments/answer_check_bins.py` bins the rows by the weak
gap, for the confidence plots. In each bin it gives how often the
weak answer fails where the strong one passes in another logical class,
the reverse, one fails and the other passes in the same class, and both
fail. Two answers in the same class differ only in the boundary they
hand on, so the windows the check cuts decide which one fails: this is
not a fault in the weak answer's class.

### Static chains, one a threshold

Ask for one more chain at each fixed threshold, in decibels:

```
decsim stream run.py --rounds 70 --seed 1 --set 'observation.answer_check_thresholds=(0.0, 10.0, 20.0)' --out s1.json
```

A static chain keeps the weak answer unless the machine's own verdict
row escalates it at the chain's threshold, and then keeps the strong
answer. Only the threshold is fixed: the row is the machine's. Static
chains share each window's answers with the paths on the same boundary.
A static chain is checked only at its last window, so it closes no
windows after a check and keeps no window rows. A row of
`threshold_checks` holds one threshold (`threshold_decibels`), and for
each patch:

- `failed` is the chain's outcome;
- `escalated_windows` counts the windows the row escalated;
- `window_count` counts the windows the chain decided.

One switching run then gives the failure rate and the escalated share
of every static threshold on the same shots. The chain at the machine's
own fixed threshold ends and escalates as the machine does. A run with
static chains is refused when the edge's verdict row reads no
threshold, such as `status`, `always` or `never`.

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
  checkpoint (section 4);
- checkpoints or answer checks beside an escalate edge past the entry
  pool, or beside a decoder or a verdict that draws from the run's
  seed;
- answer checks on a machine whose entry pool has no escalate edge to
  a pool that decodes.

[How to build each switching pattern](build_each_switching_pattern.md#what-is-refused-for-now)
lists every refusal.
