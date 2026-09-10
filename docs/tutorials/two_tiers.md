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
**confidence**, or soft output: decode the window twice, each solve
forced into one of the two possible logical answers, and subtract the
two weights. If the two answers were nearly equally likely, the decoder
had almost no reason to prefer the one it picked, and the window is
escalated.

## Step 1. Read the config

This lesson's config ships with decsim, as `configs/two_tiers.yaml`.
It is the shape of the behaviour gate's own switching point: both
tiers priced by cards rather than measured, so every tick below is the
same on your machine as on this page's.

```yaml
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
  strong_window: two_sided_context

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
  frame_to_controller:  null
  controller_to_qpu:  null

weak_decoder:
  kind: 1.0                         # one tau_gen
  units: 1
  unit_memory_rounds: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 1
    release_cycles_per_job: 10
strong_decoder:
  kind: 10.0                        # ten tau_gen
  units: 1
  unit_memory_rounds: null
  engine:
    clock: room
    fetch_cycles_per_round: 1
    release_cycles_per_job: 10

sweep:
  - physical_error_probability: [0.008]
    distance: [3, 5]
    round_period_us: [1.0]
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
- `strong_window: two_sided_context` says which rounds the strong
  decoder re-reads: the escalated window's commit region with one
  buffer region of raw context on each side.

The two decoder sections are priced cards: `kind: 1.0` says the weak
decode costs one microsecond and `kind: 10.0` says the strong decode
costs ten. Those numbers are the paper's ratio. One microsecond is this
sweep's round period, which is one **syndrome generation time**, the
time the machine takes to produce a round of syndrome
(arXiv:2510.25222, `2510.25222.txt` lines 186-187 in the sandbox's
`tmp/papers/txt/` extraction), and Toshio's own backlog simulations
set the strong decoding time to ten of them (lines 1110-1112).

A card prices the algorithm stage and nothing else: both tiers still
decode for real, on the minimum-weight perfect matching path, so the
logical failures below are measured and only the time is stated
(`decsim/decoders/settings.py`, `DecoderSettings`). Priced rather than
measured means every tick on this page is the same on your machine.
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
round_store: kind round_store
strong_round_store: kind round_store
windows: kind sliding
weak_decoder: kind 1.0
strong_decoder: kind 10.0
escalation: kind switching
pauli_frame: kind logical_register
workload: kind memory_circuit
magic_state_factory: kind infinite
links: card two_tiers.yaml
sweep block 1: p [0.008], d [3, 5], round period [1.0] us, 50 shots
log: off
trace: off
```

Two round stores, not one: `round_store` streams to the weak tier and
`strong_round_store` keeps the same rounds in case a strong re-decode
asks for them later.

## Step 2. Run the sweep

```bash
decsim collect configs/two_tiers.yaml
```

The command prints the same resolved config, then one line per point as
it finishes, then the summary. This is the summary:

```
distance: 3
physical error rate: 0.008
algorithm: 1 us
round period: 1 us
load (service per window / window inter-arrival): 2.25
logical failures: 15 of 50 shots
mismatches vs direct PyMatching: 0
throughput: 0.420 rounds per us
queue wait, mean: 15.950 us
service time per window, mean: 6.748 us
ready to frame commit: median 21.440 us, p99 77.976 us

distance: 5
physical error rate: 0.008
algorithm: 1 us
round period: 1 us
load (service per window / window inter-arrival): 1.56
logical failures: 16 of 50 shots
mismatches vs direct PyMatching: 0
throughput: 0.537 rounds per us
queue wait, mean: 12.850 us
service time per window, mean: 7.799 us
ready to frame commit: median 24.000 us, p99 65.312 us

every column: results/2026-09-10T03-22-57Z-two_tiers/sweep.csv
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
the p99 of `ready to frame commit` is several times its median.

## Step 3. Trace one shot

A sweep gives averages. To see one window escalate you need the trace,
so run a single shot with `--trace`.

```bash
decsim run configs/two_tiers.yaml --seed 0 --trace
```

```
config: two_tiers
point: p0.008 d3 round period 1 us seed 0
terminal status: complete
execution done: 30000000 ticks
fully done: 70268000 ticks
operation 1: logical_observables, observables (0,), truth (0,)
run dir: results/2026-09-10T03-13-39Z-two_tiers
```

The timestamp in your run folder's name will be your own. Substitute it
in the two commands below.

## Step 4. A window that was kept

```bash
decsim trace follow \
  results/2026-09-10T03-13-39Z-two_tiers/trace/p0.008_d3_algo1.0_round1us_seed0.trace.json \
  --window 1:0
```

```
window 1:0 of decsim switching d3 p0.008 seed0

tick (us)  where                        what                                                          dur (us)  transfer  bits
6.008      Window planner               W0 ready
6.008      Window planner               queued, dispatched to default#0                               0.000
6.008      Window planner               queued, dispatched to default#0                               0.000
6.008      weak_buffer_to_weak_decoder  move, with W0 rounds 1..6                                     0.004     move      44
6.012      Decoder unit default#0       unit default#0 memory copy                                              copy      44
6.012      Decoder unit default#0       stage fetch                                                   0.024
6.012      Decoder unit default#0       decode service                                                1.064
6.012      Decoder unit default#0       residence, unbounded, data ready 6.012, freed at decode done  2.128     copy      44
6.012      Decoder unit default#0       residence, unbounded, data ready 6.012, freed at end of run   64.256    copy      44
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
8.144      Frame                        residence, unbounded, committed 8.148, freed at end of run    62.124    copy
8.148      Window planner               W0 committed
8.148      Frame                        1:0 committed

copies 1, references 2 jobs and 0 holds, moves 3
longest residence: 64.256 us in Decoder unit default#0 (residence, unbounded, data ready 6.012, freed at end of run)
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
  results/2026-09-10T03-13-39Z-two_tiers/trace/p0.008_d3_algo1.0_round1us_seed0.trace.json \
  --window 1:3
```

```
window 1:3 of decsim switching d3 p0.008 seed0

tick (us)  where                            what                                                           dur (us)  transfer  bits
15.008     Window planner                   W3 ready
15.008     Window planner                   queued, dispatched to default#0                                0.000
15.008     Window planner                   queued, dispatched to default#0                                0.000
15.008     weak_buffer_to_weak_decoder      move, with W3 rounds 10..15                                    0.004     move      48
15.012     Decoder unit default#0           unit default#0 memory copy                                               copy      48
15.012     Decoder unit default#0           stage fetch                                                    0.024
15.012     Decoder unit default#0           decode service                                                 1.064
15.012     Decoder unit default#0           residence, unbounded, data ready 15.012, freed at decode done  2.128     copy      48
15.012     Decoder unit default#0           residence, unbounded, data ready 15.012, freed at end of run   55.256    copy      48
15.036     Decoder unit default#0           stage algorithm                                                1.000
16.036     Decoder unit default#0           stage release                                                  0.040
16.076     Window planner                   solve held
16.076     Decoder unit default#0           stage fetch                                                    0.024
16.076     Decoder unit default#0           decode service                                                 1.064
16.100     Decoder unit default#0           stage algorithm                                                1.000
17.100     Decoder unit default#0           stage release                                                  0.040
17.140     Window planner                   queued, dispatched to strong#0                                 0.000
17.140     Window planner                   verdict
17.140     Window planner                   W3 committed
17.140     strong_buffer_to_strong_decoder  move, with W3 rounds 7..15                                     0.004     move      72
17.140     weak_decoder_to_strong_decoder   move, with W3 rounds 10..15                                    0.004     move
17.144     Decoder unit strong#0            unit strong#0 memory copy                                                copy      72
17.144     Decoder unit strong#0            stage fetch                                                    0.036
17.144     Decoder unit strong#0            residence, unbounded, data ready 17.144, freed at decode done  10.076    copy      72
17.144     Decoder unit strong#0            decode service                                                 10.076
17.180     Decoder unit strong#0            stage algorithm                                                10.000
27.180     Decoder unit strong#0            stage release                                                  0.040
27.220     strong_decoder_to_frame          move, with W3 rounds 10..15                                    0.004     move      1
27.224     Frame                            residence, unbounded, committed 27.228, freed at end of run    43.044    copy
27.228     decoder_to_decoder               move, with W3 rounds 10..15                                    0.004     move      8
27.228     Frame                            1:3 committed

copies 2, references 3 jobs and 0 holds, moves 5
longest residence: 55.256 us in Decoder unit default#0 (residence, unbounded, data ready 15.012, freed at end of run)
longest queue wait: 0.000 us in Window planner (queued, dispatched to default#0)
```

The first half is the same: two weak solves, one microsecond each,
finishing at 17.140. Then the `verdict` goes the other way, and five
things happen that did not happen for window 0.

- **`queued, dispatched to strong#0`.** A third decode job, on the other
  pool's unit.
- **`strong_buffer_to_strong_decoder`, 72 bits.** The strong decoder's
  input comes from the strong round store, which has been keeping these
  rounds all along, and not from the weak decoder. It is nine rounds,
  `7..15`, where the weak window read six: the escalated window's commit
  region plus one buffer region of raw context on each side, which is
  what `strong_window: two_sided_context` asked for.
- **`weak_decoder_to_strong_decoder`, and the bits column is empty.**
  The escalation itself crosses this hop, and it carries only the
  selection, which window to decode again. No syndrome data passes
  between the tiers.
- **The strong decode costs 10 microseconds**, its card, against the
  weak tier's 1.
- **The correction reaches the frame at 27.228**, on
  `strong_decoder_to_frame`, and there is no `weak_decoder_to_frame` at
  all. Only a final answer is published to the frame.

`W3 committed` at 17.140 is worth reading carefully. The window
provisionally commits on the weak answer as soon as the verdict is in,
so the windows behind it are not blocked
(`decsim/windows/window_commits.py`, `commit`). What it does not do is
ship its boundary: this run's boundary policy is `held`, chosen for you
because the escalation may escalate and `two_sided_context` does not
absorb the windows it covers (`decsim/windows/settings.py`,
`BOUNDARY_POLICIES`, and the `boundaries` key's docstring). The held
boundary ships only when the strong answer lands, which is the
`decoder_to_decoder` move at 27.228
(`decsim/windows/window_commits.py`, `finish_strong`).

That last number is the whole trade in one line. This window's answer
was more likely to be right, and it arrived 10 microseconds later than
the weak answer would have, on a machine whose rounds are 1 microsecond
apart.

## What you learned

- Switching runs the weak tier on everything and the strong tier on the
  windows the weak tier was unsure about.
- The confidence costs two decodes per window, always.
- The strong tier reads its rounds from its own store, and only the
  selection crosses between the tiers.
- Escalating buys accuracy and pays latency, and both are on the trace.

## Read next

- [Two tiers](../explanation/two_tiers.md): the four tables behind the knobs
  above, the other confidence signal, and the other strong window
  shapes.
- [Windows and boundaries](../explanation/windows_and_boundaries.md): what a commit region, a
  buffer region and a seam are.
- [How to read a trace and follow one round or one window](../how-to/read_a_trace.md): the trace format and the other ways to
  read it.
