[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a timing study whose numbers do not depend on your computer

By default a decoder row in decsim runs for real and is charged the wall
clock the call took on your machine. That is right for accuracy, and
wrong for timing: it makes the reaction time a property of your laptop.
For a timing study, price the decoder with a **card** instead.

## 1. Write the config

A tier's `kind` may be a number instead of a name. The number is the
decode's core latency in microseconds, charged on the
minimum-weight-perfect-matching path. `configs/priced_cards_example.yaml`
is a worked one:

```yaml
# Priced cards: the decoder costs a stated number, not a measured one.
extends: weak_decoder_baseline.yaml

weak_decoder:
  kind: 1.0
  units: 1
  unit_memory_rounds: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 1
    release_cycles_per_job: 10

sweep:
  - physical_error_probability: [0.001]
    distance: [3, 5, 7]
    round_period_us: [1.0]
    shots: 20
```

Two things to know before you copy it.

**A section replaces its base's section whole.** `extends` does not
merge inside a section, so the `weak_decoder` block above repeats
`units`, `unit_memory_rounds` and `engine` even though the base already
had them. Leaving `engine` out raises `KeyError: 'engine'` at load.

**The links must be able to price what crosses them.** A bounded channel
needs a payload size, so a run whose device emits payloads without bits
is refused on the first bounded hop:
`controller_to_weak_buffer has no payload size and its channel is
bounded; a bounded wire needs a size to serialize`. The weak baseline's
channels are unbounded, which is why the example extends it.

## 2. Run it

```bash
decsim collect configs/priced_cards_example.yaml
```

```
p 0.001, d 3, round period 1.0 us: 20 shots done
distance: 3
physical error rate: 0.001
algorithm: 1 us
round period: 1 us
load (service per window / window inter-arrival): 0.36
logical failures: 0 of 20 shots
mismatches vs direct PyMatching: 0
throughput: 0.997 rounds per us
queue wait, mean: 0.000 us
service time per window, mean: 1.068 us
ready to frame commit: median 1.076 us, p99 1.076 us
```

`algorithm: 1 us` is the card. `load: 0.36` says the decoder is
comfortably ahead of the round rate, which is what a 1 microsecond
decode against a 1 microsecond round with a distance 3 commit region
should give. Compare that with the same sweep on a named decoder, where
the load was above 7.

## 3. Check that it is deterministic

Run it a second time and compare:

```bash
decsim collect configs/priced_cards_example.yaml
diff <(cut -d, -f1-20 results/<first>/sweep.csv) \
     <(cut -d, -f1-20 results/<second>/sweep.csv)
```

Every column matches except `sim_wall_seconds_per_shot`, which is how
long the simulation took to run and not a property of the machine being
simulated. That is the whole point: the ticks are now a function of the
configuration and the seed.

## What still runs for real

The decode does. A priced tier still decodes the window through
PyMatching and still returns a correction, so the logical error rate is
still measured; only the time it is charged comes from the card. That is
why `mismatches vs direct PyMatching: 0` is still meaningful above.

If you want a run with no syndrome data at all, `qpu.kind: timing_only`
emits payloads without bits. It builds and runs as a machine, but
`decsim collect` currently raises `KeyError` on it, because the front's
per-shot measurement always compares the loop's prediction against
PyMatching on the device's sampled shot, and a timing-only device
samples none. Use a priced card on a real device instead.

## Read next

- [`docs/explanation/time.md`](../explanation/time.md): ticks, clocks and the wall-clock decoder.
- [`docs/how-to/compare_two_runs.md`](compare_two_runs.md): putting a card run beside a
  measured one.
