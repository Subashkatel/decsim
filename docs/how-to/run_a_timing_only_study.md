[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a timing study whose numbers do not depend on your computer

By default a decoder row in decsim runs for real and is charged the wall
clock the call took on your machine. That is right for accuracy, and
wrong for timing: it makes the reaction time a property of your laptop.
For a timing study, price the decoder with a **card** instead.

## 1. Write the config

A tier's `kind` may be a number instead of a name. The number is the
decode's core latency in microseconds, charged on the
minimum-weight-perfect-matching path.
`configs/examples/priced_cards_example.yaml` is a worked one:

```yaml configs/examples/priced_cards_example.yaml
# Priced cards: the decoder costs a stated number, not a measured one.
extends: ../bases/weak_decoder_baseline.yaml

weak_decoder:
  kind: 1.0
  units: 1
  unit_memory:
    bits: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 1
    fetch_cycles_per_job: 0
    release_cycles_per_job: 10
    release_cycles_per_round: 0

sweep:
  - axes:
      workload.arguments.physical_error_probability: [0.001]
      qpu.distance: [3, 5, 7]
      qpu.round_period_microseconds: [1.0]
    collection: {max_shots: 20}
```

Two things to know before you copy it.

**A section replaces its base's section whole.** `extends` does not
merge inside a section, so the `weak_decoder` block above repeats
`units`, `unit_memory` and `engine` even though the base already
had them. Leaving `engine` out raises `KeyError: 'engine'` at load.

**The links must be able to price what crosses them.** A bounded channel
needs a payload size, so a run whose device emits payloads that state
no size stops on the first bounded hop with a `TypeError`: the wire
cannot add a missing size to the header's size. A syndrome buffer
or a unit memory sized in bits refuses such a round for the same reason.
Every `qpu.kind` row states its rounds' size, so this bites only a
device you build in Python.

## 2. Run it

```bash
decsim run configs/examples/priced_cards_example.yaml
```

```
{"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.001}: 20 shots done (cap)
```

The point's `sweep.csv` row holds the numbers. Its `algorithm` column
is the card, 1 microsecond. Its `load` of 0.36 says the decoder is
comfortably ahead of the round rate, which is what a 1 microsecond
decode against a 1 microsecond round with a distance 3 commit region
should give. Compare that with the same sweep on a named decoder, whose
load is whatever the host's wall clock gives.

## 3. Check that it is deterministic

Run it a second time and compare:

```bash
decsim run configs/examples/priced_cards_example.yaml
diff <(cut -d, -f1-29 results/<first>/sweep.csv) \
     <(cut -d, -f1-29 results/<second>/sweep.csv)
```

Every column matches except `sim_wall_seconds_per_shot`, which is how
long the simulation took to run and not a property of the machine being
simulated. That is the whole point: the ticks are now a function of the
configuration and the seed.

## What still runs for real

The decode does. A priced tier still decodes the window through
PyMatching and still returns a correction, so the logical error rate is
still measured; only the time it is charged comes from the card, so the
logical failures in `sweep.csv` are real decodes.

If you want a run with no syndrome data at all, `qpu.kind: timing_only`
emits payloads that state the code's size per round and carry no values,
so links and memories are still charged in bits. It builds and runs as a
machine, but
`decsim run` refuses it with a sentence, because every shot is
scored against the observables its source sampled, and a timing-only
device samples none. Use a priced card on a real device instead.

## Read next

- [Time](../explanation/time.md): ticks, clocks and the wall-clock decoder.
- [How to compare two runs](compare_two_runs.md): putting a card run beside a
  measured one.
