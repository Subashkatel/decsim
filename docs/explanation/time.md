# Time

Everything decsim reports is a time, so it is worth being exact about
what a time is here, where the numbers come from, and which of them
depend on the computer you run on.

## The tick

The engine counts in **ticks**, whole integers, and one microsecond is
one million ticks (`decsim/config.py`, `TICKS_PER_MICROSECOND`). Every
duration in the machine is an integer number of ticks. Nothing is stored
as a floating-point microsecond, so no cost can drift by a rounding
error, and two runs that do the same work land on the same tick.

A yaml file states its costs in microseconds or in cycles of a named
clock; `config.microseconds_to_ticks` rounds each to whole ticks once,
at load. A duration that is positive but rounds to zero ticks is refused
by name rather than silently becoming free (`config.check_duration`).

## The engine

`decsim/engine.py` is a discrete-event engine. It holds a queue of
actions, each due at a tick, and runs them in order. Nothing polls and
nothing sleeps: a component that has just charged itself some cost
schedules the action that follows that cost, and the engine jumps
straight to the next due tick. Simulated time therefore advances in
jumps, never in steps, and an idle machine costs nothing to simulate.

When two actions fall on the same tick, the one with the lower priority
number runs first; when they share a priority, the one scheduled first
runs first. That is time, then priority, then arrival, which is SimPy's
ordering (`simpy.core.Environment`).

Time never runs backwards. A negative delay is refused, because a
component that asks for one is a bug and a loud stop is better than a
wrong number.

## Clocks

A cost can be written two ways.

In microseconds, when a paper states it in seconds:
`round_period_us` on the sweep, `confidence_walk_microseconds` on the
escalation section.

In cycles of a **clock domain**, when a paper states it as a cycle
count: `readout_to_bits_cycles`, `packing_cycles_per_round`,
`write_cycles`, `decision_to_pulse_cycles`. The `clocks` section of the
yaml gives each domain a frequency in megahertz, and the settings layer turns cycles
into microseconds once, at load. The two shipped domains are `fridge`
and `room`, both at 250 megahertz, which is LILLIPUT's FPGA clock
(arXiv:2108.06569, Table 4). A machine with an SFQ decoder at 4 kelvin
and a room-temperature controller is one more entry in that section, not
a code change.

The one component that is clocked rather than event-driven is the QPU.
It runs syndrome extraction on every live patch every cycle whether or
not an operation is using that patch, so every round lands on a cycle
boundary, and a patch nobody is using still emits an idle round
(`decsim/qpu/cycle_clock.py`, which cites Google arXiv:2207.06431 and
arXiv:2408.13687 for the cadence). What those idle rounds cost the
decoder is the idle policy's question, and `IDLE_POLICIES` is the table
of answers.

## Priced cards, and what a card is

A **card** is a small record of numbers that prices one thing. A link
card prices one hop: a propagation latency, in microseconds or in cycles
of a named clock; a bandwidth, as bits per cycle, or none for an
unbounded wire; the channel the path rides, since two paths naming one
channel share it; and a per-transfer setup cost
(`decsim/links/settings.py`). A decoder card prices one decode: a fixed
latency in microseconds.

A card is how you get a run whose ticks are a property of the
configuration and nothing else. Write a number where a decoder's `kind`
would go and that tier is charged that number per decode:

```yaml
weak_decoder:
  kind: 0.028                       # LILLIPUT card
```

`configs/weak_decoder_baseline.yaml` does exactly this, and its comment
says where the number came from.

## The wall-clock decoder, and what it costs you

Name a decoder instead, and decsim runs it for real. PyMatching, belief
matching, union find, Relay-BP, Tesseract and BP-OSD all decode the
window's actual detection events and return an actual correction, which
is how a run can report a logical error rate at all.

The price of that decode is the time the call really took on your
computer. `decode_timed` in `decsim/decoders/decoder.py` reads
`time.perf_counter_ns` on each side of the call, and the unit is held
for exactly that long.

Three things follow, and they matter.

- **A named decoder makes the run depend on the host.** Two runs of the
  same yaml with the same seed produce the same logical results and the
  same window structure, but not the same ticks. A faster computer makes
  a faster machine.
- **The measured number is a software wall clock, not a hardware
  latency.** PyMatching in a Python process is not an ASIC. A run that
  wants to say what a decoder of a given speed would do sets a card.
- **The behaviour gate has to know the difference.** Twenty-one of its
  twenty-six points are priced by cards and are compared bit for bit;
  the five that name a decoder are compared on a projection that leaves
  the tick-bearing fields out. `docs/reference/frozen_gate.md` says
  which and why.

The two are not exclusive. A study of accuracy names a decoder and reads
the logical error rate; a study of the reaction time sets cards and
reads the ticks; a study of both runs the accuracy sweep for the error
rate and the timing sweep for the time, and says so.

## Where the reaction time is measured

The run folder's latency points are the pieces of the loop, and the four
totals are the ways of naming the whole of it. Two of them start the
clock when a window's rounds are available to the decoder
(`buffer0_ready_to_frame`, `buffer0_first_round_to_frame`), and two
start it when the physics happened, as the round left the QPU
(`qpu_last_round_to_frame`, `qpu_first_round_to_frame`), so those two
also carry the link out of the fridge, the controller's own work and the
write into the buffer. `docs/reference/run_folder.md` lists all of them.

## Read next

- `docs/reference/run_folder.md`: every latency point, defined.
- `docs/reference/frozen_gate.md`: strict points and semantic points.
- `docs/how-to/run_a_timing_only_study.md`: a run whose ticks are
  configuration alone.
