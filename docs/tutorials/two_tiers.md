[decsim docs](../README.md) › [Tutorials](README.md)

# Two tiers

This lesson runs a machine with two decoders and watches one window get
decoded twice. It takes about ten minutes, four of which are the machine
running.

It assumes you have done [Your first run](first_run.md), so you know what
a round, a window and a commit are.

## What escalation is

A fast decoder keeps up with the machine and gets some windows wrong. An
accurate decoder gets more of them right and cannot keep up. **Decoder
switching** is the idea of having both: run the fast one on every window,
and call the accurate one only on the windows the fast one was unsure
about.

decsim calls the two the **weak tier** and the **strong tier**, which
are the words Toshio et al. use (arXiv:2510.25222, Sec. III A).
"Unsure" has to be a number, and that number is the window's
**confidence**: this machine's signal decodes the window twice, each
solve forced into one of the two logical answers, and subtracts the two
weights, so a small gap means the decoder had almost no reason to prefer
the answer it picked.

## Step 1. Read the run file

This lesson's run file ships with decsim as `examples/two_tiers.py`.
Both tiers are priced by cards rather than measured, so every tick below
is the same on your machine as on this page's. The switching record and
the two decoders come first:

```python examples/two_tiers.py
complementary_gap = complementary.ComplementaryGap.Settings()
threshold = threshold_sources.FixedThreshold.Settings(
    threshold_decibels=THRESHOLD_DECIBELS
)
switching = escalation_settings.SwitchingSettings(
    confidence=complementary_gap, threshold=threshold
)
weak_decoder = decoder_settings.linear_decoder_pool(
    WEAK_DECODE_MICROSECONDS_PER_ROUND,
    machine_settings.FRIDGE_CLOCK,
    solves_per_window=2,
)
strong_decoder = decoder_settings.linear_decoder_pool(
    STRONG_DECODE_MICROSECONDS_PER_ROUND,
    machine_settings.ROOM_CLOCK,
    solves_per_window=1,
)
```

- `SwitchingSettings` turns the weak base into a two-tier machine: the
  weak tier decodes every window first, and a window escalates to the
  strong tier when its confidence is below the threshold.
- `ComplementaryGap.Settings()` is the confidence signal: the two forced
  solves and the difference of their weights.
- `FixedThreshold.Settings(threshold_decibels=20.0)` is the confidence
  below which a window escalates, written in the paper's decibels.
  decsim converts it once into natural-log weight, the unit the decoder
  compares in (`decsim/escalation/threshold_sources.py`,
  `decibels_to_nats`).
- The switching record's `strong_window` is left at its default,
  `RedoWindow.Settings()`: the strong decoder re-reads the escalated
  window's commit region and one buffer region ahead of it, with its
  past face pinned on the correction the window before it committed.

Both decoders are Toshio et al.'s linear decoder: a decode of r rounds
costs a fixed time per round, tau_dec r. This sweep's round period is
one microsecond, one **syndrome generation time** (tau_gen).
`linear_decoder_pool` puts that time on the decoder unit's fetch stage
and prices the algorithm at zero.

- The weak tier's tau_weak_dec is 0.4 tau_gen a round. Its confidence
  signal decodes each window twice, so `solves_per_window=2` makes each
  solve 0.2 microseconds a round. A signal that decodes a window once
  takes `solves_per_window=1`.
- The strong tier's tau_strong_dec is 10 tau_gen, 10 microseconds a
  round.

Then, for each distance, the loop moves the paper's T_comm, the time
from a round's measurement to its decoder, onto two links:

```python examples/two_tiers.py
    weak_side = link_profiles.with_path_latency(
        base.links, "controller_to_weak_buffer", WEAK_COMMUNICATION_MICROSECONDS
    )
    strong_side = machine_settings.one_cycle_strong_side(weak_side)
    links = link_profiles.with_path_latency(
        strong_side,
        "weak_decoder_to_strong_decoder",
        STRONG_COMMUNICATION_MICROSECONDS,
    )
```

One tau_gen on `controller_to_weak_buffer`, ten on
`weak_decoder_to_strong_decoder`, the hop an escalation crosses, and
one cycle on each other hop of the strong side.

The cards price time and nothing else: both tiers still decode for
real, on the minimum-weight perfect matching path, so the logical
failures below are measured and only the time is stated.

Two syndrome buffers, not one: `weak_syndrome_buffer` streams to the
weak tier and keeps every round a strong re-decode might still ask for;
`strong_syndrome_buffer` holds the rounds an escalation carries up to
the strong decoder, and nothing else in this run.

## Step 2. Run the sweep

```bash
decsim run examples/two_tiers.py --out results/two_tiers
```

The command prints its folder, then one line per task as it finishes.
The numbers are in `sweep.csv`; these are the columns this step reads:

```bash
cut -d, -f3,8,9,35,79,103,136,137 results/two_tiers/sweep.csv
```

```
qpu.distance,logical_failures,scored_shots,load,queue_wait_mean_us,service_mean_us,buffer0_ready_to_frame_median_us,buffer0_ready_to_frame_p99_us
3,15,50,20.337608,0.0,31.8336,218.924,567.788
5,16,50,24.0966912,0.0,64.718,441.088,986.208
```

Every value is the same on any host: the decoders are priced by cards,
and the decode time is on the unit's fetch stage, as Step 1 said.

Read `service_mean_us` against the weak tier's own cost.
A window here is six rounds, so its two weak decodes cost 2.4
microseconds. The average window costs 31.8 microseconds at distance 3,
because some windows were decoded a third time, on a strong decoder
that costs 60 microseconds for the same six rounds. That average is the
price of switching, and putting a number on it is what a study of
switching is for.

`load` is the service time per window divided by the time between
windows arriving. Above 1 the decoders cannot keep up, the undecoded
backlog grows, and the tail of the reaction time grows with it. This
configuration is far over its head, at a load near 20, which is why
`buffer0_ready_to_frame_p99_us` is between two and three times the
median beside it. The mean queue wait is zero all the same: each window
waits for the boundary of the window before it, and a unit is free by
the time that boundary lands. The results put that wait in `dep_block`,
the wait for what a decode depends on, and `queue_wait` holds only the
wait for a free unit.

## Step 3. Trace one shot

A sweep gives averages. To see one window escalate you need the trace,
so run a single shot with `--trace`.

```bash
decsim run examples/two_tiers.py --seed 1 --trace --out results/two_tiers_shot
```

```
config: two_tiers
task: {"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.008} seed 1
terminal status: complete
execution done: 30000000 ticks
fully done: 401736000 ticks
operation 1: logical_observables, observables (1,), truth (1,)
run dir: results/two_tiers_shot
```

## Step 4. A window that was kept

```bash
decsim trace follow \
  results/two_tiers_shot/trace/*_seed1.trace.json \
  --window 1:0
```

```
window 1:0 of task d3, decsim switching d3 seed1

tick (us)  where                        what                                                          dur (us)  transfer  bits
7.004      Window planner               W0 ready
7.004      Window planner               queued, dispatched to default#0                               0.000
7.004      Window planner               queued, dispatched to default#0                               0.000
7.004      weak_buffer_to_weak_decoder  move, with W0 rounds 1:1..6                                   0.004     move      44
7.008      Decoder unit default#0       unit default#0 memory copy                                              copy      44
7.008      Decoder unit default#0       stage fetch                                                   1.200
7.008      Decoder unit default#0       decode service                                                1.200
7.008      Decoder unit default#0       residence, unbounded, data ready 7.008, freed at decode done  2.400     copy      44
8.208      Window planner               solve held
8.208      Decoder unit default#0       stage algorithm                                               0.000
8.208      Decoder unit default#0       stage release                                                 0.000
8.208      Decoder unit default#0       stage fetch                                                   1.200
8.208      Decoder unit default#0       decode service                                                1.200
9.408      Window planner               verdict
9.408      decoder_to_decoder           move, with W0 rounds 1:1..6                                   0.004     move      8
9.408      weak_decoder_to_frame        move, with W0 rounds 1:1..6                                   0.004     move      1
9.408      Decoder unit default#0       stage algorithm                                               0.000
9.408      Decoder unit default#0       stage release                                                 0.000
9.412      Frame                        residence, unbounded, committed 9.416, freed at end of run    392.324   copy
9.416      Window planner               W0 committed
9.416      Frame                        1:0 committed

copies 1, references 2 jobs and 0 holds, moves 3
longest residence: 392.324 us in Frame (residence, unbounded, committed 9.416, freed at end of run)
longest queue wait: 0.000 us in Window planner (queued, dispatched to default#0)
```

Two things here are new since [Your first run](first_run.md).

The window was **dispatched twice**, and the decode ran twice: two
`stage fetch` lines of 1.200 microseconds, at 7.008 and again at 8.208,
each six rounds at the weak tier's 0.2 microseconds a round. That is
the confidence being computed: two solves of one window, each forced
into one of the two logical answers. `solve held` at 8.208 is the first
result waiting for the other class's solve, so that the two weights can
be subtracted.

Then, at 9.408, `verdict`. This window's confidence was at or above the
threshold, so the weak answer was kept: the boundary went to the next
window on `decoder_to_decoder`, the correction went to the Pauli frame
on `weak_decoder_to_frame`, and the window committed at 9.416.

Note the cost. Two solves is 2.4 microseconds of decoding, not 1.2, on
every window, whether or not it escalates. That is what this confidence
signal costs.

`W0 ready` at 7.004 includes the microsecond each round spends on
`controller_to_weak_buffer`, the weak tier's T_comm.

## Step 5. A window that escalated

```bash
decsim trace follow \
  results/two_tiers_shot/trace/*_seed1.trace.json \
  --window 1:3
```

```
window 1:3 of task d3, decsim switching d3 seed1

tick (us)  where                            what                                                           dur (us)  transfer  bits
16.004     Window planner                   W3 ready
16.004     Window planner                   queued, dispatched to default#0                                0.000
16.004     Window planner                   queued, dispatched to default#0                                0.000
16.004     weak_buffer_to_weak_decoder      move, with W3 rounds 1:10..15                                  0.004     move      48
16.008     Window planner                   masked view copy                                                         copy      48
16.008     Decoder unit default#0           unit default#0 memory copy                                               copy      48
16.008     Decoder unit default#0           stage fetch                                                    1.200
16.008     Decoder unit default#0           decode service                                                 1.200
16.008     Decoder unit default#0           residence, unbounded, data ready 16.008, freed at decode done  2.400     copy      48
17.208     Window planner                   solve held
17.208     Window planner                   masked view copy                                                         copy      48
17.208     Decoder unit default#0           stage algorithm                                                0.000
17.208     Decoder unit default#0           stage release                                                  0.000
17.208     Decoder unit default#0           stage fetch                                                    1.200
17.208     Decoder unit default#0           decode service                                                 1.200
18.408     Window planner                   verdict
18.408     Window planner                   W3 committed
18.408     Strong tier                      W3 strong window held                                          10.000
18.408     weak_decoder_to_strong_decoder   move, with W3 rounds 1:10..15                                  10.000    move      65
18.408     weak_decoder_to_strong_decoder   move, with W3 rounds 1:10..15                                  10.000    move      112
18.408     Decoder unit default#0           stage algorithm                                                0.000
18.408     Decoder unit default#0           stage release                                                  0.000
28.408     Window planner                   queued, dispatched to strong#0                                 10.000
28.408     weak_decoder_to_strong_decoder   move, with W3 rounds 1:10..15                                  10.000    move      72
38.408     strong_buffer_to_strong_decoder  move, with W3 rounds 1:10..15                                  0.004     move      48
38.412     Window planner                   masked view copy                                                         copy      48
38.412     Decoder unit strong#0            unit strong#0 memory copy                                                copy      48
38.412     Decoder unit strong#0            stage fetch                                                    60.000
38.412     Decoder unit strong#0            residence, unbounded, data ready 38.412, freed at decode done  60.000    copy      48
38.412     Decoder unit strong#0            decode service                                                 60.000
98.412     strong_decoder_to_frame          move, with W3 rounds 1:10..15                                  0.004     move      65
98.412     Decoder unit strong#0            stage algorithm                                                0.000
98.412     Decoder unit strong#0            stage release                                                  0.000
98.416     Frame                            residence, unbounded, committed 98.420, freed at end of run    303.320   copy
98.420     strong_decoder_to_weak_decoder   move, with W3 rounds 1:10..15                                  0.004     move      72
98.420     Frame                            1:3 committed

copies 5, references 3 jobs and 0 holds, moves 7
longest residence: 303.320 us in Frame (residence, unbounded, committed 98.420, freed at end of run)
longest queue wait: 10.000 us in Window planner (queued, dispatched to strong#0)
```

The first half is nearly the same: two weak solves, 1.2 microseconds
each, finishing at 18.408. Then the `verdict` goes the other way, and
eight things happen that did not happen for window 0.

- **`masked view copy`, three times.** Window 0 had no window before
  it; window 3 does, so each of its two weak solves reads a duplicate
  of its rounds with window 2's boundary folded in, and so does the
  strong solve at 28.416. The fold happens on every window that has a
  predecessor, whatever that predecessor's seam carried.
- **`W3 strong window held`, 10 microseconds.** The strong tier has the
  window but not its rounds yet, so the strong request waits for them
  to land.
- **`weak_decoder_to_strong_decoder`, twice, 10 microseconds each.** The
  escalation crosses this hop as two transfers, and each takes the
  strong tier's T_comm, ten tau_gen. The first is the selection, 65
  bits: which window to decode again, the request's 64-bit name, and
  one bit of what the weak decode committed behind the window, which the
  strong host joins to its own answer. That is the default
  `strong_answer_route`, `direct`; under `through_weak_chip` the weak
  chip keeps that bit and joins the answer itself, so the selection is
  the name alone. The second is 112 bits: that name and 48 bits of
  rounds, read out of the weak syndrome buffer. They are six rounds, `10..15`, the escalated
  window's commit region and the buffer region ahead of it. The rounds
  cross once, when a window escalates, and never before.
- **`queued, dispatched to strong#0`.** A third decode job, on the other
  pool's unit, queued at 28.408 when the rounds land in the strong
  syndrome buffer.
- **`weak_decoder_to_strong_decoder` again at 28.408, 72 bits.** Window
  2's committed boundary, shipped to the strong window: `redo_window`
  pins the strong window's past face on it, and the strong solve folds
  it in (`decsim/windows/window_boundaries.py`, `pin_strong_face`).
  Window 2 committed on the chip and the strong window decodes on the
  host, so the boundary crosses the same hop as the escalation, 8 bits
  of seam behind the request's 64-bit name. The job takes a unit only
  once that boundary lands, which is the 10 microseconds it waits in
  the queue.
- **`strong_buffer_to_strong_decoder` at 38.408, 48 bits.** The strong
  decoder's input comes from the strong syndrome buffer, where the
  escalation just put it, and not from the weak decoder.
- **The strong decode costs 60 microseconds**, six rounds at ten tau_gen
  each, against 1.2 for each weak solve.
- **The correction reaches the frame at 98.420**, on
  `strong_decoder_to_frame`, and there is no `weak_decoder_to_frame` at
  all. Only a final answer is published to the frame.

`W3 committed` at 18.408 is worth reading carefully. The window
provisionally commits on the weak answer as soon as the verdict is in
(`decsim/windows/window_commits.py`, `commit`). What it does not do is
ship its boundary: this run's boundary policy is held, the one the redo
window's record hands `switching_windows`, because the redo window does
not absorb the windows it covers. A strong window absorbs a
weak window when it decodes the same rounds again and replaces that
window's answer, so the weak window never ships a boundary of its own.
The held boundary ships only when the strong answer lands, which is the
`strong_decoder_to_weak_decoder` move at 98.420
(`decsim/windows/window_commits.py`, `finish_strong`): the strong
decode computed it on the host, and the next window decodes on the
chip.

The window behind it waits for that boundary. Follow window 4; these
are its first lines:

```bash
decsim trace follow \
  results/two_tiers_shot/trace/*_seed1.trace.json \
  --window 1:4
```

```
window 1:4 of task d3, decsim switching d3 seed1

tick (us)  where                            what                                                            dur (us)  transfer  bits
19.004     Window planner                   W4 ready
19.004     Window planner                   queued, dispatched to default#0                                 79.420
19.004     Window planner                   queued, dispatched to default#0                                 79.420
98.424     weak_buffer_to_weak_decoder      move, with W4 rounds 1:13..18                                   0.004     move      48
98.428     Window planner                   masked view copy                                                          copy      48
98.428     Decoder unit default#0           unit default#0 memory copy                                                copy      48
98.428     Decoder unit default#0           stage fetch                                                     1.200
98.428     Decoder unit default#0           decode service                                                  1.200
98.428     Decoder unit default#0           residence, unbounded, data ready 98.428, freed at decode done   2.400     copy      48
99.628     Window planner                   solve held
```

Window 4 has its rounds at 19.004, but it waits in the queue for 79.420
microseconds: it takes a unit only at 98.424, the tick window 3's
boundary lands, and its rounds then cross to the unit, so its first
solve starts at 98.428. Decoding the next window waits on the held
boundary, and its rounds stay in the weak syndrome buffer until then
(`decsim/decoders/decode_dispatch.py`). So under the redo
window one stream has at most one strong decode in flight: the next
window gets no verdict, and cannot escalate, until the strong answer
lands. The double window keeps the weak chain committing, so one
stream's escalations can overlap there.

That last number is the whole trade in one line. Window 3's strong
answer was more likely to be right, and it arrived about 80
microseconds after the weak answer, on a machine whose rounds are 1
microsecond apart.

The listing goes on with window 4's own escalation. Its second
transfer carries only rounds `16..18`, 88 bits, because rounds `13..15`
are already in the strong syndrome buffer from window 3.

## What you learned

- Switching runs the weak tier on everything and the strong tier on the
  windows the weak tier was unsure about.
- The confidence costs two decodes per window, always.
- The rounds cross between the tiers once, with the escalation, and
  only for the windows that escalate.
- Escalating buys accuracy and pays latency, and both are on the trace.

## Read next

- [Build a machine step by step](build_a_machine.md): this lesson's
  machine, built part by part.
- [The parts](../reference/parts.md): every settings record a machine
  is built from.
