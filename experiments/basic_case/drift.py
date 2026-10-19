r"""The rate controller against the static threshold as the noise drifts.

Each task streams one memory (decsim stream) whose noise moves from
p = 0.002 to 1.5 p = 0.003 at round 10,000 (FlipStreamDevice's drift;
SD6 is linear in p, so the rounds after are the memory at 0.003), the
decoders keeping p = 0.002's window models, as a device drifting from
its last calibration does. The static threshold is the one that
escalated 10% of windows at p = 0.002 (calibration.py); at 0.003 it
escalates about 25% (threshold_sweep.py), past the share the chain
sustains, q* of 0.12 to 0.14 (threshold_plots.py). The controller
(RateThreshold, slice 05) starts there too and holds 10%.

Each stream records its escalated share every hundredth window and the
controller's threshold (stream_measures.py); the warmup ends at the
drift, so its latencies, backlog and growth rates are the drifted
half's. drift_plots.py draws them.

Usage
-----

From this folder, with this checkout first on the path (STYLE.md), each
task at five seeds, a process a stream:

```
export PYTHONPATH=$PWD/../..
OUT=$PWD/../../results/basic_case_drift; mkdir -p $OUT
sbatch -N1 -c20 --mem=80G -t 4:00:00 --wrap "for name in \
  drift_d5_static drift_d5_rate drift_d7_static drift_d7_rate; do \
  for seed in 1 2 3 4 5; do echo \$name \$seed; done; done | \
  xargs -P 20 -n 2 sh -c 'python -m decsim stream drift.py --only \$0 \
  --rounds 20000 --warmup 10000 --seed \$1 --out $OUT/\$0_s\$1.json'"
```
"""

import dataclasses

import decsim
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device

config = sweeps.load_beside(__file__, "config.py")
calibration = sweeps.load_beside(__file__, "calibration.py")
PROBABILITY = 0.002
DRIFT_ROUND = 10_000
DRIFT_FACTOR = 1.5
TARGET = 0.1
DISTANCES = (5, 7)


def task(distance: int, control: str) -> decsim.Task:
    """One distance's drifting stream, static or rate-controlled."""
    decibels = calibration.CALIBRATED[(distance, PROBABILITY)][TARGET]
    threshold = threshold_sources.FixedThreshold.Settings(decibels)
    if control == "rate":
        threshold = threshold_sources.RateThreshold.Settings(decibels, TARGET)
    switching = config.machine("switching", distance, PROBABILITY)
    machine = switching.with_settings(machines.SWITCHING, threshold=threshold)
    source = flip_stream_device.FlipStreamDevice.Settings(
        drift_round=DRIFT_ROUND, drift_factor=DRIFT_FACTOR
    )
    qpu = machine.held("qpu")
    drifting = dataclasses.replace(qpu, source=source)
    metadata = {
        "control": control,
        "threshold_db": decibels,
        "target": TARGET,
        "drift_round": DRIFT_ROUND,
        "drift_factor": DRIFT_FACTOR,
        "workload.arguments.physical_error_probability": PROBABILITY,
        "qpu.distance": distance,
    }
    name = f"drift_d{distance}_{control}"
    drifted = machine.with_held("qpu", drifting)
    return decsim.Task(name, drifted, metadata)


tasks = [
    task(distance, control)
    for distance in DISTANCES
    for control in ("static", "rate")
]
# decsim stream runs one shot; an experiment still names its collection
stop = decsim.CollectionSettings(max_shots=1)
experiment = decsim.Experiment("basic_case_drift", tasks, stop)
