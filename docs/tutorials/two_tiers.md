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
**confidence**: this config's signal decodes the window twice, each
solve forced into one of the two logical answers, and subtracts the two
weights, so a small gap means the decoder had almost no reason to prefer
the answer it picked. [Two tiers](../explanation/two_tiers.md) explains
the signals and the shapes behind the knobs.

## Step 1. Read the config

This lesson's config ships with decsim, as
`configs/examples/two_tiers.yaml`. Both tiers are priced by cards rather
than measured, so every tick below is the same on your machine as on
this page's.

```yaml configs/examples/two_tiers.yaml
# Two tiers on priced cards, so the whole switching loop is
# deterministic and the same on every host. Both tiers are Toshio et
# al.'s linear decoder (arXiv:2510.25222 lines 968-971): a decode of r
# rounds costs tau_dec r on the unit, and a round reaches the decoder
# T_comm after it is measured. Their backlog simulations set
# T_weak_comm = tau_gen and T_strong_comm = tau_strong_dec = 10 tau_gen
# (lines 1109-1114), tau_gen being this sweep's 1.0 us round period at
# 250 cycles. tau_weak_dec is 0.4 tau_gen, one of the values the paper
# sweeps (lines 1125-1126); the complementary gap decodes each window
# twice, so each decode costs 0.2 tau_gen a round. The weak T_comm sits
# on controller_to_weak_buffer and the strong one on
# weak_decoder_to_strong_decoder, which carries the escalated rounds up.
# docs/tutorials/two_tiers.md runs this config. Every key is documented
# in reference.yaml.
extends: ../bases/weak_decoder_baseline.yaml

escalation:
  kind: switching
  gap_threshold_db: 20.0
  strong_window: redo_window

links:
  qpu_to_controller:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  controller_to_weak_buffer: {latency_cycles: 250, clock: fridge, bits_per_cycle: null}   # T_weak_comm
  controller_to_strong_buffer: {latency_cycles: 1, clock: room, bits_per_cycle: null}
  weak_buffer_to_weak_decoder: {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  weak_decoder_to_strong_decoder: {latency_cycles: 2500, clock: room, bits_per_cycle: null}   # T_strong_comm
  strong_buffer_to_strong_decoder: {latency_cycles: 1, clock: room, bits_per_cycle: null}
  decoder_to_decoder:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  weak_decoder_to_frame: {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  strong_decoder_to_frame:  {latency_cycles: 1, clock: room, bits_per_cycle: null}
  frame_to_controller:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  controller_to_qpu:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}

weak_decoder:
  kind: 0.0                         # the matching answers; the engine times it
  units: 1
  unit_memory:
    bits: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 50      # 0.2 tau_gen a decode, two decodes a window
    fetch_cycles_per_job: 0
    release_cycles_per_job: 0
    release_cycles_per_round: 0
strong_decoder:
  kind: 0.0
  units: 1
  unit_memory:
    bits: null
  engine:
    clock: room
    fetch_cycles_per_round: 2500    # tau_strong_dec, ten tau_gen
    fetch_cycles_per_job: 0
    release_cycles_per_job: 0
    release_cycles_per_round: 0

sweep:
  - axes:
      workload.arguments.physical_error_probability: [0.008]
      qpu.distance: [3, 5]
      qpu.round_period_microseconds: [1.0]
    collection: {max_shots: 50}
```

The `escalation` section is the new part.

- `kind: switching` is the word of `ESCALATION_KINDS` that runs the weak tier
  first and escalates. The other two rows are `weak_baseline`, one tier
  only, and `strong_only`, the accurate decoder on everything.
- `gap_threshold_db: 20.0` is the confidence below which a window is
  escalated, written in the paper's decibels. decsim converts it once,
  at load, into natural-log weight, which is the unit the decoder
  compares in (`decsim/escalation/settings.py`, `decibels_to_nats`).
- `strong_window: redo_window` says which rounds the strong
  decoder re-reads: the escalated window's commit region and one buffer
  region ahead of it, with its past face pinned on the correction the
  window before it committed.

Both decoder sections are Toshio et al.'s linear decoder: a decode of r
rounds costs a fixed time per round, tau_dec r. This sweep's round
period is one microsecond, one **syndrome generation time** (tau_gen),
which is 250 cycles of the 250 MHz clocks.

- `kind: 0.0` prices the algorithm stage at zero, so the time sits on
  the engine's fetch stage.
- The weak tier's `fetch_cycles_per_round: 50` is 0.2 microseconds a
  round. Its confidence signal decodes each window twice, so the two
  decodes cost tau_weak_dec = 0.4 tau_gen a round between them.
- The strong tier's `fetch_cycles_per_round: 2500` is 10 microseconds a
  round, tau_strong_dec = 10 tau_gen.
- Two link cards carry the paper's T_comm, the time from a round's
  measurement to its decoder: 250 cycles (one tau_gen) on
  `controller_to_weak_buffer`, and 2500 cycles (ten tau_gen) on
  `weak_decoder_to_strong_decoder`, the hop an escalation crosses.

The cards price time and nothing else: both tiers still decode for
real, on the minimum-weight perfect matching path, so the logical
failures below are measured and only the time is stated
(`decsim/decoders/settings.py`, `DecoderSettings`).
[How to run a timing study whose numbers do not depend on your computer](../how-to/run_a_timing_only_study.md) says more about cards.

`decsim show` prints the config as decsim resolved it:

```bash
decsim show configs/examples/two_tiers.yaml
```

```
config: configs/examples/two_tiers.yaml <- configs/bases/weak_decoder_baseline.yaml
qpu: kind stim_device
windows: kind sliding
escalation: kind switching
burst_detector: kind none
workload: kind producer
links: card two_tiers.yaml
sweep block 1: workload.arguments.physical_error_probability [0.008], qpu.distance [3, 5], qpu.round_period_microseconds [1.0]; max_shots 50, min_shots 0, piece_rounds 20000
log: off
trace: off
values:
clock.period_ticks = 4000
qpu.round_period_microseconds = [1.0]  [sweep, configs/examples/two_tiers.yaml:58-63]
qpu.distance = [3, 5]  [sweep, configs/examples/two_tiers.yaml:58-63]
qpu.error_model_provider = null
controller.clock = null
```

Below `values:` the list goes on to every value the machine is built
with, one per line: the layer that set it (your file, a preset it
extends, the sweep, or the default) and the yaml lines it came from. A
line with no source is a value decsim works out from others, such as
`clock.period_ticks`, a 250 MHz cycle in ticks. That is the machine's
clock, the domain the yaml's `controller.clock` names, so the
controller's own clock is null: it runs on the machine's.

Two syndrome buffers, not one: `weak_syndrome_buffer` streams to the weak tier and
keeps every round a strong re-decode might still ask for;
`strong_syndrome_buffer` holds the rounds an escalation carries up to
the strong decoder, and nothing else in this run.

## Step 2. Run the sweep

```bash
decsim run configs/examples/two_tiers.yaml --out results/two_tiers
```

The command prints the same resolved config, then one line per point as
it finishes, then the summary. This is the summary:

```
workload.arguments.physical_error_probability: 0.008
qpu.distance: 3
qpu.round_period_microseconds: 1.0
algorithm: 0 us
load (service per window / window inter-arrival): 19.90
logical failures: 15 of 50 scored shots
logical error rate among scored shots: 0.3, 95% 0.179 to 0.446 (cap)
unscored shots: 0 of 50 (0)
throughput: 0.101 rounds per us
queue wait, mean: 136.378 us
service time per window, mean: 31.834 us
ready to frame commit: median 210.056 us, p99 558.328 us

workload.arguments.physical_error_probability: 0.008
qpu.distance: 5
qpu.round_period_microseconds: 1.0
algorithm: 0 us
load (service per window / window inter-arrival): 23.87
logical failures: 16 of 50 scored shots
logical error rate among scored shots: 0.32, 95% 0.195 to 0.467 (cap)
unscored shots: 0 of 50 (0)
throughput: 0.071 rounds per us
queue wait, mean: 289.609 us
service time per window, mean: 64.718 us
ready to frame commit: median 439.072 us, p99 986.144 us

data movement: observation.data_movement was off, so this run counted no copies, references or moves

every column: results/two_tiers/sweep.csv
```

`algorithm: 0 us` is the card; the decode time is on the engine, as
Step 1 said.

Read `service time per window, mean` against the weak tier's own cost.
A window here is six rounds, so its two weak decodes cost 2.4
microseconds. The average window costs 31.8 microseconds at distance 3,
because some windows were decoded a third time, on a strong decoder
that costs 60 microseconds for the same six rounds. That average is the
price of switching, and putting a number on it is what a study of
switching is for.

`load` is the service time per window divided by the time between
windows arriving. Above 1 the decoders cannot keep up, the undecoded
backlog grows, and the queue wait and the tail of the reaction time grow
with it. This configuration is far over its head, at a load near 20,
which is why the mean queue wait is over a hundred microseconds and the
p99 of `ready to frame commit` is between two and three times its
median.

## Step 3. Trace one shot

A sweep gives averages. To see one window escalate you need the trace,
so run a single shot with `--trace`.

```bash
decsim run configs/examples/two_tiers.yaml --seed 1 --trace --out results/two_tiers_shot
```

```
config: two_tiers
point: {"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.008} seed 1
terminal status: complete
execution done: 30000000 ticks
fully done: 381700000 ticks
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
window 1:0 of decsim switching d3 seed1

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
9.412      Frame                        residence, unbounded, committed 9.416, freed at end of run    372.288   copy
9.416      Window planner               W0 committed
9.416      Frame                        1:0 committed

copies 1, references 2 jobs and 0 holds, moves 3
longest residence: 372.288 us in Frame (residence, unbounded, committed 9.416, freed at end of run)
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
window 1:3 of decsim switching d3 seed1

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
18.408     weak_decoder_to_strong_decoder   move, with W3 rounds 1:10..15                                  10.000    move      64
18.408     weak_decoder_to_strong_decoder   move, with W3 rounds 1:10..15                                  10.000    move      112
18.408     Decoder unit default#0           stage algorithm                                                0.000
18.408     Decoder unit default#0           stage release                                                  0.000
28.408     Window planner                   queued, dispatched to strong#0                                 0.000
28.408     strong_buffer_to_strong_decoder  move, with W3 rounds 1:10..15                                  0.004     move      48
28.408     decoder_to_decoder               move, with W3 rounds 1:10..15                                  0.004     move      8
28.412     Window planner                   masked view copy                                                         copy      48
28.412     Decoder unit strong#0            unit strong#0 memory copy                                                copy      48
28.412     Decoder unit strong#0            stage fetch                                                    60.000
28.412     Decoder unit strong#0            residence, unbounded, data ready 28.412, freed at decode done  60.000    copy      48
28.412     Decoder unit strong#0            decode service                                                 60.000
88.412     strong_decoder_to_frame          move, with W3 rounds 1:10..15                                  0.004     move      65
88.412     Decoder unit strong#0            stage algorithm                                                0.000
88.412     Decoder unit strong#0            stage release                                                  0.000
88.416     Frame                            residence, unbounded, committed 88.420, freed at end of run    293.284   copy
88.420     decoder_to_decoder               move, with W3 rounds 1:10..15                                  0.004     move      8
88.420     Frame                            1:3 committed

copies 5, references 3 jobs and 0 holds, moves 7
longest residence: 293.284 us in Frame (residence, unbounded, committed 88.420, freed at end of run)
longest queue wait: 0.000 us in Window planner (queued, dispatched to default#0)
```

The first half is nearly the same: two weak solves, 1.2 microseconds
each, finishing at 18.408. Then the `verdict` goes the other way, and
eight things happen that did not happen for window 0.

- **`masked view copy`, three times.** Window 0 had no window before
  it; window 3 does, so each of its two weak solves reads a duplicate
  of its rounds with window 2's boundary folded in, and so does the
  strong solve at 28.412. The fold happens on every window that has a
  predecessor, whatever that predecessor's seam carried.
- **`W3 strong window held`, 10 microseconds.** The strong tier has the
  window but not its rounds yet, so the strong request waits for them
  to land.
- **`weak_decoder_to_strong_decoder`, twice, 10 microseconds each.** The
  escalation crosses this hop as two transfers, and each takes the
  strong tier's T_comm, ten tau_gen. The first carries only the
  selection, which window to decode again, the request's 64-bit name.
  The second is 112 bits: that name and 48 bits of rounds, read out of
  the weak syndrome buffer. They are six rounds, `10..15`, the escalated
  window's commit region and the buffer region ahead of it. The rounds
  cross once, when a window escalates, and never before.
- **`queued, dispatched to strong#0`.** A third decode job, on the other
  pool's unit, dispatched at 28.408 when the rounds land in the strong
  syndrome buffer.
- **`strong_buffer_to_strong_decoder`, 48 bits.** The strong decoder's
  input comes from the strong syndrome buffer, where the escalation just
  put it, and not from the weak decoder.
- **`decoder_to_decoder` at 28.408, 8 bits.** Window 2's committed
  boundary, shipped to the strong window: `redo_window` pins the
  strong window's past face on it, and the strong solve folds it in
  (`decsim/windows/window_boundaries.py`, `pin_strong_face`).
- **The strong decode costs 60 microseconds**, six rounds at ten tau_gen
  each, against 1.2 for each weak solve.
- **The correction reaches the frame at 88.420**, on
  `strong_decoder_to_frame`, and there is no `weak_decoder_to_frame` at
  all. Only a final answer is published to the frame.

`W3 committed` at 18.408 is worth reading carefully. The window
provisionally commits on the weak answer as soon as the verdict is in
(`decsim/windows/window_commits.py`, `commit`). What it does not do is
ship its boundary: this run's boundary policy is `held`, chosen for you
because the escalation may escalate and `redo_window` does not
absorb the windows it covers. A strong window absorbs a weak window when
it decodes the same rounds again and replaces that window's answer, so
the weak window never ships a boundary of its own
(`decsim/windows/settings.py`,
`BOUNDARY_POLICIES`, and the `boundaries` key's docstring). The held
boundary ships only when the strong answer lands, which is the
`decoder_to_decoder` move at 88.420
(`decsim/windows/window_commits.py`, `finish_strong`).

The window behind it waits for that boundary. Follow window 4; these
are its first lines:

```bash
decsim trace follow \
  results/two_tiers_shot/trace/*_seed1.trace.json \
  --window 1:4
```

```
window 1:4 of decsim switching d3 seed1

tick (us)  where                            what                                                            dur (us)  transfer  bits
19.004     Window planner                   W4 ready
19.004     Window planner                   queued, dispatched to default#0                                 0.000
19.004     Window planner                   queued, dispatched to default#0                                 0.000
19.004     weak_buffer_to_weak_decoder      move, with W4 rounds 1:13..18                                   0.004     move      48
19.008     Decoder unit default#0           unit default#0 memory copy                                                copy      48
19.008     Decoder unit default#0           residence, unbounded, data ready 19.008, freed at decode done   71.816    copy      48
88.424     Window planner                   masked view copy                                                          copy      48
88.424     Decoder unit default#0           stage fetch                                                     1.200
88.424     Decoder unit default#0           decode service                                                  1.200
89.624     Window planner                   solve held
```

Window 4 has its rounds at 19.004 and they are staged in the decoder
unit's memory at 19.008, but its first solve starts at 88.424, the tick
window 3's boundary lands. Staging goes on while window 3 is escalated;
decoding the next window waits on the held boundary.

That last number is the whole trade in one line. Window 3's strong
answer was more likely to be right, and it arrived about 70
microseconds after the weak answer, on a machine whose rounds are 1
microsecond apart.

The listing goes on with window 4's own escalation. Its second
transfer carries only rounds `16..18`, 88 bits, because rounds `13..15`
are already in the strong syndrome buffer from window 3.

To price the strong tier's off-board hops with a measured cable instead
of the room-clock cards above, set `links.kind` to
`roce_v2_cpu` or `roce_v2_gpu` and delete the four strong-side cards, so
the row's numbers stand. They come from Backline (arXiv:2609.09270),
which measured a real round trip from a controller to a CPU or GPU over
Ethernet: half of that round trip on the write into the strong syndrome
buffer, on the escalation and on the reply, and
zero on the strong store's own read
([D14](../explanation/decisions.md#d14-the-strong-tiers-off-board-path-can-be-priced-by-a-measured-round-trip)).

## What you learned

- Switching runs the weak tier on everything and the strong tier on the
  windows the weak tier was unsure about.
- The confidence costs two decodes per window, always.
- The rounds cross between the tiers once, with the escalation, and
  only for the windows that escalate.
- Escalating buys accuracy and pays latency, and both are on the trace.

## Read next

- [Two tiers](../explanation/two_tiers.md): the four tables behind the knobs
  above, the other two confidence signals, and the other strong window
  shapes.
- [Windows and boundaries](../explanation/windows_and_boundaries.md): what a commit region, a
  buffer region and a seam are.
- [How to read a trace and follow one round or one window](../how-to/read_a_trace.md): the trace format and the other ways to
  read it.
