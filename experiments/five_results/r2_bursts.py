r"""R2 on streams: more correlated-matching cores on one patch, in a burst.

One patch, correlated matching alone on parallel windows (B5), so its
cores decode side by side, each decode priced by the fitted time (B1
stage 2), so a window in a burst takes longer. At d = 9 parallel windows
decode 6d rounds for every 4d they commit: at 3.35 us a round that is
4.6 cores' work at steady noise, so 5 cores keep up. A burst (B3,
bursts.md: 30 qubits, a rise of 3 rounds, a fade of 550) starts at
round ONSET_ROUND; peak 1 is no burst. The stream's series rows give
the waiting line over time; its record the strong pool's busy share.
Task names: strong_{strong_unit_count}core_peak{peak_factor}_d{distance}
_p{physical_error_probability}.

Usage
-----

```
export PYTHONPATH=$PWD/../..
python -m decsim stream r2_bursts.py --only strong_5core_peak10.0_d9_p0.003 \
  --rounds 8000 --warmup 1000 --seed 1 \
  --set observation.series_every_rounds=100 --out strong.json
```
"""

import dataclasses

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.qpu.flip_stream_device as flip_stream_device
import decsim.topology.graph as graph
import decsim.windows.schemes.parallel as parallel

config = sweeps.load_beside(__file__, "config.py")
CORE_COUNTS = (5, 6, 8, 10)
PEAK_FACTORS = (1.0, 4.0, 10.0)
# after the stream's 1,000 warm-up rounds and 1,000 calm recorded ones
ONSET_ROUND = 2000


def machine(
    strong_unit_count: int,
    peak_factor: float,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """Correlated matching alone on parallel windows, maybe in a burst."""
    alone = config.alone("strong_alone", distance, physical_error_probability)
    scheme = parallel.ParallelWindowScheme.Settings()
    on_parallel = alone.with_settings(machines.WINDOWS, scheme=scheme)
    sized = on_parallel.with_settings(
        machines.STRONG_POOL, unit_count=strong_unit_count
    )
    timed = config.fitted_strong_time(sized, distance, "parallel")
    return with_burst(config.streaming(timed), peak_factor)


def with_burst(machine: graph.Machine, peak_factor: float) -> graph.Machine:
    """The streaming machine with one burst at ONSET_ROUND; peak 1 is none."""
    if peak_factor == 1.0:
        return machine
    burst = flip_stream_device.BurstSettings(
        onset_round=ONSET_ROUND, peak_factor=peak_factor
    )
    qpu = machine.held("qpu")
    source = dataclasses.replace(qpu.source, burst=burst)
    return machine.with_held("qpu", dataclasses.replace(qpu, source=source))


# decsim stream runs one shot; an experiment still names its collection
stop = decsim.CollectionSettings(max_shots=1)
sweep = decsim.Sweep(
    "five_results_r2_bursts",
    machine,
    grid={
        "strong_unit_count": CORE_COUNTS,
        "peak_factor": PEAK_FACTORS,
        "distance": (9,),
        "physical_error_probability": (0.003,),
    },
    task_name=(
        "strong_{strong_unit_count}core_peak{peak_factor}"
        "_d{distance}_p{physical_error_probability}"
    ),
    stop=stop,
    metadata=config.metadata,
)
experiment = sweep.experiment()
