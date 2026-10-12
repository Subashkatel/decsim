[decsim docs](../README.md) › [How-to guides](README.md)

# How to use your own noise model

A memory's noise is a row in its `noise` field (`producers.py`
`MemorySettings`). The row has two hooks (`ports.CircuitNoise`), which
`memory_circuit` (`decsim/frontends/settings.py`) calls once per task:

- `generator_noise(probability)` returns the four channels Stim's
  generator puts in (`StimNoise`);
- `noisy(circuit, probability)` returns the generated circuit with the
  row's own noise added.

Stim's own channels stay inside the generator, because Stim draws them
in its own way: noise added afterwards to the same channels samples
different shots from the same seed.

| Row | Generator | Added afterwards |
|---|---|---|
| none (the default) | p on all four channels | nothing |
| `StimNoise(a, b, c, d)` | the four channels at these probabilities | nothing |
| `sd6` (`frontends/deltakit.py` `SD6Noise`) | nothing | Deltakit Explorer's SD6 at p |

SD6 is the standard depolarising model of Gidney et al. 2108.10457
(Tables 1 and 2): DEPOLARIZE2(p) after each two-qubit gate,
DEPOLARIZE1(p) after each one-qubit gate and on each idle qubit, and a
flip of p after each reset and before each measurement. It needs the
`deltakit` extra.

In Python, or by path on any base that holds a `memory` row:

```python
memory = producers.MemorySettings(
    0.001, rounds_per_distance=1, noise=deltakit.SD6Noise()
)
```

```bash
decsim run run.py --set workload.noise.type=sd6
```

`noiseless_time_boundaries=True` prepares and reads out the data qubits
without noise, whichever row made it: the "perfect terminal time
boundaries" of Toshio et al. 2510.25222.

A row of your own is a frozen dataclass with a `name` and the two hooks,
written by its import path (`workload.noise.type=mylab.noise.MyNoise`).
Its `noisy` keeps the generated circuit's qubits, detectors and
observable; `circuit.without_noise().detector_error_model()` raises if
a detector stops being deterministic.
