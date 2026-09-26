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

This lesson's config ships with decsim, as `configs/two_tiers.yaml`.
Both tiers are priced by cards rather than measured, so every tick
below is the same on your machine as on this page's.

```yaml configs/two_tiers.yaml
# Two tiers on priced cards, so the whole switching loop is
# deterministic and the same on every host. The weak card is one
# syndrome generation time at this sweep's round period and the strong
# card is ten of them, the ratio Toshio et al. use in their own backlog
# simulations (arXiv:2510.25222, tau_strong_dec = 10 tau_gen).
# docs/tutorials/two_tiers.md runs this config. Every key is documented
# in reference.yaml.
extends: weak_decoder_baseline.yaml

escalation:
  kind: switching
  gap_threshold_db: 20.0
  strong_window: near_seam_pinned

links:
  qpu_to_controller:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  controller_to_weak_buffer: {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  controller_to_strong_buffer: {latency_cycles: 1, clock: room, bits_per_cycle: null}
  weak_buffer_to_weak_decoder: {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  weak_decoder_to_strong_decoder: {latency_cycles: 1, clock: room, bits_per_cycle: null}
  strong_buffer_to_strong_decoder: {latency_cycles: 1, clock: room, bits_per_cycle: null}
  decoder_to_decoder:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  weak_decoder_to_frame: {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  strong_decoder_to_frame:  {latency_cycles: 1, clock: room, bits_per_cycle: null}
  frame_to_controller:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}
  controller_to_qpu:  {latency_cycles: 1, clock: fridge, bits_per_cycle: null}

weak_decoder:
  kind: 1.0                         # one tau_gen
  units: 1
  unit_memory:
    bits: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 1
    fetch_cycles_per_job: 0
    release_cycles_per_job: 10
    release_cycles_per_round: 0
strong_decoder:
  kind: 10.0                        # ten tau_gen
  units: 1
  unit_memory:
    bits: null
  engine:
    clock: room
    fetch_cycles_per_round: 1
    fetch_cycles_per_job: 0
    release_cycles_per_job: 10
    release_cycles_per_round: 0

sweep:
  - physical_error_probability: [0.008]
    distance: [3, 5]
    round_period_microseconds: [1.0]
    shots: 50
```

The `escalation` section is the new part.

- `kind: switching` is the row of `ESCALATIONS` that runs the weak tier
  first and escalates. The other two rows are `weak_baseline`, one tier
  only, and `strong_only`, the accurate decoder on everything.
- `gap_threshold_db: 20.0` is the confidence below which a window is
  escalated, written in the paper's decibels. decsim converts it once,
  at load, into natural-log weight, which is the unit the decoder
  compares in (`decsim/escalation/settings.py`, `decibels_to_nats`).
- `strong_window: near_seam_pinned` says which rounds the strong
  decoder re-reads: the escalated window's commit region and one buffer
  region ahead of it, with its past face pinned on the correction the
  window before it committed.

The two decoder sections are priced cards: `kind: 1.0` says the weak
decode costs one microsecond, which is this sweep's round period and so
one **syndrome generation time**, and `kind: 10.0` says the strong
decode costs ten of them, the ratio the yaml comment above cites.

A card prices the algorithm stage and nothing else: both tiers still
decode for real, on the minimum-weight perfect matching path, so the
logical failures below are measured and only the time is stated
(`decsim/decoders/settings.py`, `DecoderSettings`).
[How to run a timing study whose numbers do not depend on your computer](../how-to/run_a_timing_only_study.md) says more about cards.

One row is chosen for you and matters below. `decsim show` prints it:

```bash
decsim show configs/two_tiers.yaml
```

```
config: configs/two_tiers.yaml <- configs/weak_decoder_baseline.yaml
qpu: kind stim_device
idle_policy: kind separate_decode_jobs
links: kind logical_reference
weak_syndrome_buffer: kind syndrome_buffer
strong_syndrome_buffer: kind syndrome_buffer
windows: kind sliding
weak_decoder: kind 1.0
strong_decoder: kind 10.0
escalation: kind switching
burst_detector: kind none
pauli_frame: kind logical_register
workload: kind producer
magic_state_factory: kind infinite
links: card two_tiers.yaml
sweep block 1: p [0.008], d [3, 5], round period [1.0] us, 50 shots
log: off
trace: off
values:
clocks.fridge = 250.0  [preset weak_decoder_baseline.yaml, configs/weak_decoder_baseline.yaml:50]
clocks.room = 250.0  [preset weak_decoder_baseline.yaml, configs/weak_decoder_baseline.yaml:51]
qpu.kind = "stim_device"  [preset weak_decoder_baseline.yaml, configs/weak_decoder_baseline.yaml:4]
qpu.code_card = "rotated_surface"  [default, configs/reference.yaml:81]
qpu.round_period_microseconds = [1.0]  [sweep, configs/two_tiers.yaml:51-55]
qpu.distance = [3, 5]  [sweep, configs/two_tiers.yaml:51-55]
```

Below `values:` the list goes on to every value the machine is built
with, one per line: the layer that set it (your file, a preset it
extends, the sweep, or the default) and the yaml lines it came from.

Two syndrome buffers, not one: `weak_syndrome_buffer` streams to the weak tier and
keeps every round a strong re-decode might still ask for;
`strong_syndrome_buffer` holds the rounds an escalation carries up to
the strong decoder, and nothing else in this run.

## Step 2. Run the sweep

```bash
decsim collect configs/two_tiers.yaml --out results/two_tiers
```

The command prints the same resolved config, then one line per point as
it finishes, then the summary. This is the summary:

```
distance: 3
physical error rate: 0.008
algorithm: 1 us
round period: 1 us
load (service per window / window inter-arrival): 3.67
logical failures: 15 of 50 shots
mismatches vs direct PyMatching: 0
throughput: 0.390 rounds per us
queue wait, mean: 16.643 us
service time per window, mean: 5.941 us
ready to frame commit: median 28.900 us, p99 79.656 us

distance: 5
physical error rate: 0.008
algorithm: 1 us
round period: 1 us
load (service per window / window inter-arrival): 2.54
logical failures: 16 of 50 shots
mismatches vs direct PyMatching: 0
throughput: 0.525 rounds per us
queue wait, mean: 14.916 us
service time per window, mean: 6.964 us
ready to frame commit: median 28.368 us, p99 70.340 us

data movement: observation.data_movement was off, so this run counted no copies, references or moves

every column: results/two_tiers/sweep.csv
```

Read `service time per window, mean` against the weak card of one
microsecond. The average window costs several times the weak decode,
because some windows were decoded a second time on a decoder that costs
ten. That average is the price of switching, and putting a number on it
is what a study of switching is for.

`load` is the service time per window divided by the time between
windows arriving. Above 1 the decoders cannot keep up, the undecoded
backlog grows, and the queue wait and the tail of the reaction time grow
with it. This configuration is deliberately over its head, which is why
the p99 of `ready to frame commit` is between two and three times its
median.

## Step 3. Trace one shot

A sweep gives averages. To see one window escalate you need the trace,
so run a single shot with `--trace`.

```bash
decsim run configs/two_tiers.yaml --seed 1 --trace --out results/two_tiers_shot
```

```
config: two_tiers
point: p0.008 d3 round period 1 us seed 1
terminal status: complete
execution done: 30000000 ticks
fully done: 80316000 ticks
operation 1: logical_observables, observables (1,), truth (1,)
run dir: results/two_tiers_shot
```

## Step 4. A window that was kept

```bash
decsim trace follow \
  results/two_tiers_shot/trace/p0.008_d3_algo1.0_round1us_seed1.trace.json \
  --window 1:0
```

```
window 1:0 of decsim switching d3 p0.008 seed1

tick (us)  where                        what                                                          dur (us)  transfer  bits
6.008      Window planner               W0 ready
6.008      Window planner               queued, dispatched to default#0                               0.000
6.008      Window planner               queued, dispatched to default#0                               0.000
6.008      weak_buffer_to_weak_decoder  move, with W0 rounds 1..6                                     0.004     move      44
6.012      Decoder unit default#0       unit default#0 memory copy                                              copy      44
6.012      Decoder unit default#0       stage fetch                                                   0.024
6.012      Decoder unit default#0       decode service                                                1.064
6.012      Decoder unit default#0       residence, unbounded, data ready 6.012, freed at decode done  2.128     copy      44
6.036      Decoder unit default#0       stage algorithm                                               1.000
7.036      Decoder unit default#0       stage release                                                 0.040
7.076      Window planner               solve held
7.076      Decoder unit default#0       stage fetch                                                   0.024
7.076      Decoder unit default#0       decode service                                                1.064
7.100      Decoder unit default#0       stage algorithm                                               1.000
8.100      Decoder unit default#0       stage release                                                 0.040
8.140      Window planner               verdict
8.140      decoder_to_decoder           move, with W0 rounds 1..6                                     0.004     move      8
8.140      weak_decoder_to_frame        move, with W0 rounds 1..6                                     0.004     move      1
8.144      Frame                        residence, unbounded, committed 8.148, freed at end of run    72.172    copy
8.148      Window planner               W0 committed
8.148      Frame                        1:0 committed

copies 1, references 2 jobs and 0 holds, moves 3
longest residence: 72.172 us in Frame (residence, unbounded, committed 8.148, freed at end of run)
longest queue wait: 0.000 us in Window planner (queued, dispatched to default#0)
```

Two things here are new since [Your first run](first_run.md).

The window was **dispatched twice**, and the algorithm ran twice, at
6.036 and again at 7.076. That is the confidence being computed: two
solves of one window, each forced into one of the two logical answers.
`solve held` at 7.076 is the first result waiting for its sibling, so
that the two weights can be subtracted.

Then, at 8.140, `verdict`. This window's confidence was at or above the
threshold, so the weak answer was kept: the boundary went to the next
window on `decoder_to_decoder`, the correction went to the Pauli frame
on `weak_decoder_to_frame`, and the window committed at 8.148.

Note the cost. Two solves is 2 microseconds of decoding, not 1, on every
window, whether or not it escalates. That is what this confidence signal
costs.

## Step 5. A window that escalated

```bash
decsim trace follow \
  results/two_tiers_shot/trace/p0.008_d3_algo1.0_round1us_seed1.trace.json \
  --window 1:3
```

```
window 1:3 of decsim switching d3 p0.008 seed1

tick (us)  where                            what                                                           dur (us)  transfer  bits
15.008     Window planner                   W3 ready
15.008     Window planner                   queued, dispatched to default#0                                0.000
15.008     Window planner                   queued, dispatched to default#0                                0.000
15.008     weak_buffer_to_weak_decoder      move, with W3 rounds 10..15                                    0.004     move      48
15.012     Window planner                   masked view copy                                                         copy      48
15.012     Decoder unit default#0           unit default#0 memory copy                                               copy      48
15.012     Decoder unit default#0           stage fetch                                                    0.024
15.012     Decoder unit default#0           decode service                                                 1.064
15.012     Decoder unit default#0           residence, unbounded, data ready 15.012, freed at decode done  2.128     copy      48
15.036     Decoder unit default#0           stage algorithm                                                1.000
16.036     Decoder unit default#0           stage release                                                  0.040
16.076     Window planner                   masked view copy                                                         copy      48
16.076     Window planner                   solve held
16.076     Decoder unit default#0           stage fetch                                                    0.024
16.076     Decoder unit default#0           decode service                                                 1.064
16.100     Decoder unit default#0           stage algorithm                                                1.000
17.100     Decoder unit default#0           stage release                                                  0.040
17.140     Window planner                   verdict
17.140     Window planner                   W3 committed
17.140     Strong tier                      W3 strong window held                                          0.004
17.140     weak_decoder_to_strong_decoder   move, with W3 rounds 10..15                                    0.004     move      0
17.140     weak_decoder_to_strong_decoder   move, with W3 rounds 10..15                                    0.004     move      48
17.144     Window planner                   queued, dispatched to strong#0                                 0.000
17.144     strong_buffer_to_strong_decoder  move, with W3 rounds 10..15                                    0.004     move      48
17.144     decoder_to_decoder               move, with W3 rounds 10..15                                    0.004     move      8
17.148     Window planner                   masked view copy                                                         copy      48
17.148     Decoder unit strong#0            unit strong#0 memory copy                                                copy      48
17.148     Decoder unit strong#0            stage fetch                                                    0.024
17.148     Decoder unit strong#0            residence, unbounded, data ready 17.148, freed at decode done  10.064    copy      48
17.148     Decoder unit strong#0            decode service                                                 10.064
17.172     Decoder unit strong#0            stage algorithm                                                10.000
27.172     Decoder unit strong#0            stage release                                                  0.040
27.212     strong_decoder_to_frame          move, with W3 rounds 10..15                                    0.004     move      1
27.216     Frame                            residence, unbounded, committed 27.220, freed at end of run    53.100    copy
27.220     decoder_to_decoder               move, with W3 rounds 10..15                                    0.004     move      8
27.220     Frame                            1:3 committed

copies 5, references 3 jobs and 0 holds, moves 7
longest residence: 53.100 us in Frame (residence, unbounded, committed 27.220, freed at end of run)
longest queue wait: 0.000 us in Window planner (queued, dispatched to default#0)
```

The first half is nearly the same: two weak solves, one microsecond
each, finishing at 17.140. Then the `verdict` goes the other way, and
eight things happen that did not happen for window 0.

- **`masked view copy`, three times.** Window 0 had no window before
  it; window 3 does, so each of its two weak solves reads a duplicate
  of its rounds with window 2's boundary folded in, and so does the
  strong solve at 17.148. The fold happens on every window that has a
  predecessor, whatever that predecessor's seam carried.
- **`W3 strong window held`.** The strong tier has the window but not
  its rounds yet, so the strong request waits for them to land.
- **`weak_decoder_to_strong_decoder`, twice.** The escalation crosses
  this hop as two transfers. The first carries only the selection, which
  window to decode again, so it carries zero bits. The second
  carries the window's rounds, 48 bits, read out of the weak syndrome
  buffer: six rounds, `10..15`, the escalated window's commit region
  and the buffer region ahead of it. The rounds cross once, when a
  window escalates, and never before.
- **`queued, dispatched to strong#0`.** A third decode job, on the other
  pool's unit, dispatched at 17.144 when the rounds land in the strong
  syndrome buffer.
- **`strong_buffer_to_strong_decoder`, 48 bits.** The strong decoder's
  input comes from the strong syndrome buffer, where the escalation just
  put it, and not from the weak decoder.
- **`decoder_to_decoder` at 17.144, 8 bits.** Window 2's committed
  boundary, shipped to the strong window: `near_seam_pinned` pins the
  strong window's past face on it, and the strong solve folds it in
  (`decsim/windows/window_boundaries.py`, `pin_strong_face`).
- **The strong decode costs 10 microseconds**, its card, against the
  weak tier's 1.
- **The correction reaches the frame at 27.220**, on
  `strong_decoder_to_frame`, and there is no `weak_decoder_to_frame` at
  all. Only a final answer is published to the frame.

`W3 committed` at 17.140 is worth reading carefully. The window
provisionally commits on the weak answer as soon as the verdict is in
(`decsim/windows/window_commits.py`, `commit`). What it does not do is
ship its boundary: this run's boundary policy is `held`, chosen for you
because the escalation may escalate and `near_seam_pinned` does not
absorb the windows it covers. A strong window absorbs a weak window when
it decodes the same rounds again and replaces that window's answer, so
the weak window never ships a boundary of its own
(`decsim/windows/settings.py`,
`BOUNDARY_POLICIES`, and the `boundaries` key's docstring). The held
boundary ships only when the strong answer lands, which is the
`decoder_to_decoder` move at 27.220
(`decsim/windows/window_commits.py`, `finish_strong`).

The window behind it waits for that boundary. Follow window 4:

```bash
decsim trace follow \
  results/two_tiers_shot/trace/p0.008_d3_algo1.0_round1us_seed1.trace.json \
  --window 1:4
```

```
window 1:4 of decsim switching d3 p0.008 seed1

tick (us)  where                            what                                                           dur (us)  transfer  bits
18.008     Window planner                   W4 ready
18.008     Window planner                   queued, dispatched to default#0                                0.000
18.008     Window planner                   queued, dispatched to default#0                                0.000
18.008     weak_buffer_to_weak_decoder      move, with W4 rounds 13..18                                    0.004     move      48
18.012     Decoder unit default#0           unit default#0 memory copy                                               copy      48
18.012     Decoder unit default#0           residence, unbounded, data ready 18.012, freed at decode done  11.340    copy      48
27.224     Window planner                   masked view copy                                                         copy      48
27.224     Decoder unit default#0           stage fetch                                                    0.024
27.224     Decoder unit default#0           decode service                                                 1.064
27.248     Decoder unit default#0           stage algorithm                                                1.000
28.248     Decoder unit default#0           stage release                                                  0.040
```

Window 4 has its rounds at 18.008 and they are staged in the decoder
unit's memory at 18.012, but its first solve starts at 27.224, the tick
window 3's boundary lands. Staging goes on while window 3 is escalated;
decoding the next window waits on the held boundary.

That last number is the whole trade in one line. This window's answer
was more likely to be right, and it arrived 10 microseconds later than
the weak answer would have, on a machine whose rounds are 1 microsecond
apart.

To price the strong tier's off-board hops with a measured cable instead
of the one-cycle room-clock cards above, set `links.kind` to
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
