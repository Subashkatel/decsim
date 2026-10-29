r"""Long memory streams at fixed thresholds: sliding and speculation.

Each task streams one memory at a steady p on FlipStreamDevice (a
stream samples its rounds as it goes, not from a saved dataset), a
scheme of speculation.py at one fixed threshold. The cluster gap takes
multiples of 0.5 nats, so each threshold sits half-way between two
levels and is one distinct policy. The levels are the capacity study's
(about 3% to 30% escalated) and three more above them, since
speculation is expected to carry a larger share. The double window is
not in this run file, though a stream runs it too
(escalation/strong_window_shapes.py DoubleWindow). Task names:
{scheme}_d{distance}_p{probability}_{nats}.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md):

```
export PYTHONPATH=$PWD/../..
python -m decsim stream speculation_streams.py \
  --only speculation_d7_p0.003_4.75 --rounds 20000 --warmup 1400 --seed 1
```
"""

import dataclasses

import decsim
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device

speculation = sweeps.load_beside(__file__, "speculation.py")
STREAM_SCHEMES = ("sliding", "speculation")
# the capacity study's levels, then three more 0.5 nats apart
LEVELS_NATS = {
    (5, 0.002): (3.25, 4.75, 5.25, 6.25, 7.25, 7.75, 8.25, 8.75, 9.25),
    (5, 0.003): (2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25),
    (5, 0.005): (0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25),
    (7, 0.002): (3.25, 4.75, 5.25, 6.25, 7.25, 7.75, 8.25, 8.75, 9.25),
    (7, 0.003): (2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25),
    (7, 0.005): (0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25),
}


def task(scheme: str, point: tuple, nats: float) -> decsim.Task:
    """One steady stream of a scheme at one fixed threshold."""
    distance, probability = point
    decibels = threshold_sources.nats_to_decibels(nats)
    threshold = threshold_sources.FixedThreshold.Settings(decibels)
    shaped = speculation.machine(scheme, distance, probability)
    machine = shaped.with_settings(machines.SWITCHING, threshold=threshold)
    qpu = machine.held(machines.QPU)
    source = flip_stream_device.FlipStreamDevice.Settings()
    steady = dataclasses.replace(qpu, source=source)
    metadata = {
        "scheme": scheme,
        "threshold_nats": nats,
        "workload.arguments.physical_error_probability": probability,
        "qpu.distance": distance,
    }
    name = f"{scheme}_d{distance}_p{probability}_{nats:g}"
    streamed = machine.with_held(machines.QPU, steady)
    return decsim.Task(name, streamed, metadata)


tasks = [
    task(scheme, point, nats)
    for scheme in STREAM_SCHEMES
    for point, levels in LEVELS_NATS.items()
    for nats in levels
]
stop = decsim.CollectionSettings(max_shots=1)
experiment = decsim.Experiment("speculation_streams", tasks, stop)
