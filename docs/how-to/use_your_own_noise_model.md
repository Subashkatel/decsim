[decsim docs](../README.md) › [How-to guides](README.md)

# How to use your own noise model

A memory's noise is a row in its `noise` field (`producers.py`
`MemorySettings`). The row has two hooks (`ports.CircuitNoise`), and
`memory_circuit` (`decsim/frontends/settings.py`) calls them in order:

- `generator_noise(probability)` returns the four channels
  `stim.Circuit.generated` puts in (`StimNoise`), at the memory's p;
- `noisy(circuit, probability)` returns the generated circuit with the
  row's own noise added.

The generator's noise stays inside the generator. Stim draws the
generator's noise in one way, merging the data and ancilla resets, so a
pass that rebuilt that noise afterwards would sample different shots
from the same seed. A row that adds all its noise afterwards asks the
generator for none (`NO_GENERATOR_NOISE`). Both hooks run once per task,
when the workload is lowered, so a row costs nothing per shot.

| Row | Generator | Added afterwards |
|---|---|---|
| none (the default) | p on all four channels | nothing: the circuit is `stim.Circuit.generated`'s, byte for byte |
| `StimNoise(a, b, c, d)` | the four channels at these probabilities; p is only the label | nothing |
| `sd6` (`frontends/deltakit.py` `SD6Noise`) | nothing | Deltakit Explorer's SD6 at p |
| your own record | what you choose | what you choose |

SD6 is the standard depolarising model of Gidney et al. 2108.10457
(Tables 1 and 2):

- each two-qubit gate is followed by DEPOLARIZE2(p);
- each one-qubit Clifford is followed by DEPOLARIZE1(p);
- each reset is followed by a flip of p;
- each measurement is preceded by a flip of p;
- each qubit idle in a time step gets DEPOLARIZE1(p).

SD6's gate set measures and resets in separate steps, so its
surface code cycle takes 8 time steps (2108.10457, Table 2's caption).
Stim's generated memory instead uses one combined step, MR, and takes
7. `SD6Noise` therefore splits each MR into a measurement and, one time
step later, a reset before it adds the noise. On Stim's rotated memories
the 8-step cycle reproduces the MWPM logical error rates of Toshio et al.
2510.25222's Fig. 15a. The 7-step cycle gives fewer errors.
`SD6Noise(measure_and_reset_apart=False)` keeps the 7-step cycle.

SD6 needs the `deltakit` extra. `SD6Noise(probability=0.003)` fixes
its p whatever the memory's.

## 1. Choose a row

In Python:

```python
import decsim.frontends.deltakit as deltakit
import decsim.producers as producers

memory = producers.MemorySettings(
    0.001, rounds_per_distance=1, noise=deltakit.SD6Noise()
)
machine = base.with_held("workload", memory)
```

By path, on any base that holds a `memory` row:

```bash
decsim run run.py --set workload.noise.type=sd6
```

`noiseless_time_boundaries=True` prepares and reads out the data qubits
without noise, whichever row made the noise. It removes the data's flip
after their first reset and their flip before the last readout,
including a readout's own flip argument, `M(p)`. This is the
"perfect time boundary" setting of Toshio et al. 2510.25222 and Gidney
et al. 2312.04522.

## 2. Or write your own row

A row is a frozen dataclass with the two hooks and a `name`. The
`noise` field admits any record with both hooks, taking the same
parameter names. decsim finds the record class by itself, with no
registration step:

```python
import dataclasses

import stim

import decsim.frontends.settings as workload_settings


@dataclasses.dataclass(frozen=True)
class MeasurementHeavyNoise:
    """Stim's gate noise at p, and five times p on every measurement."""

    name = "measurement_heavy"

    def generator_noise(self, probability: float) -> workload_settings.StimNoise:
        heavy = 5 * probability
        return workload_settings.StimNoise(
            probability, probability, heavy, probability
        )

    def noisy(self, circuit: stim.Circuit, probability: float) -> stim.Circuit:
        del probability
        return circuit
```

`noisy` may return any Stim circuit that keeps the generated circuit's
qubits, detectors and observable. To add noise of your own, rebuild the
circuit there, as `SD6Noise.noisy` does through Deltakit. A row in your
own package is written by its import path:
`workload.noise.type=mylab.noise.MeasurementHeavyNoise`.

## 3. Check it

```python
circuit = workload_settings.memory_circuit(
    "surface_code:rotated_memory_z", 3, 3, 0.001, noise=MeasurementHeavyNoise()
)
circuit.without_noise().detector_error_model()  # deterministic detectors
```

`tests/frontends/test_deltakit.py` checks SD6 in the same way. It
checks that SD6 keeps Stim's qubit numbers, detectors and observable,
adds noise at the row's p, and loses both boundary flips.
