"""Pilot: Ising in sliding windows, alone and as the strong decoder.

Machine.py's 10d-round memories in windows of 2d rounds that commit d,
at d = 9 and 13 and p = 0.003 and 0.006, a few thousand shots a cell,
the same seeds in every cell of one (d, p):

- pymatching_alone and ising_alone, every window by one decoder;
- union_find_alone, the weak decoder by itself;
- switching_ising_redo and switching_ising_double, union-find with the
  cluster gap at 20 dB handing low-gap windows to Ising on the same
  window or on the paper's double window (2510.25222 Sec. III C).

It checks that Ising helps inside windows before the full runs. Usage,
from this folder on Della, in Slurm jobs (run_on_a_slurm_cluster.md):

```
export PYTHONPATH=$PWD:$PYTHONPATH   # ising_row is imported by name
decsim run ising_windows_pilot.py --record --out $OUT
decsim run ising_windows_pilot.py --out $OUT --share $I/$N
decsim run --fold --out $OUT
```
"""

import decsim
import decsim.decoders.settings as decoder_settings
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
ROUNDS_PER_DISTANCE = 10
PILOT_SHOTS = 2_000


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One cell: one decoder alone, or union-find switching to Ising."""
    strong_window = "redo_window"
    if decoding == "switching_ising_double":
        strong_window = "double_window"
    whole = config.machine(
        distance,
        physical_error_probability,
        ROUNDS_PER_DISTANCE,
        strong_window=strong_window,
        strong="ising",
    )
    if decoding == "union_find_alone":
        return _weak_alone(whole)
    if decoding == "ising_alone":
        return _strong_alone(whole)
    if decoding == "pymatching_alone":
        return _pymatching_alone(whole)
    return whole


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    qpu = task.machine.held("qpu")
    return {
        "decoding": cell["decoding"],
        "rounds_per_distance": ROUNDS_PER_DISTANCE,
        "workload.arguments.physical_error_probability": cell[
            "physical_error_probability"
        ],
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": qpu.round_period_microseconds,
    }


def _weak_alone(machine: graph.Machine) -> graph.Machine:
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.STRONG_POOL)
    return without_pool.without_part(machines.STRONG_STORE)


def _strong_alone(machine: graph.Machine) -> graph.Machine:
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.WEAK_POOL)
    return without_pool.without_part(machines.WEAK_STORE)


def _pymatching_alone(machine: graph.Machine) -> graph.Machine:
    """Ising's place and its link, with PyMatching in its pool."""
    alone = _strong_alone(machine)
    settings = constants.ising
    engine = {"type": "Engine", "clock": constants.clocks.chip, **settings.engine}
    pool = config.record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm={"type": "PyMatchingDecoder"},
        engine=engine,
    )
    return alone.with_record(machines.STRONG_POOL, pool)


stop = decsim.CollectionSettings(
    max_shots=PILOT_SHOTS, piece_rounds=constants.piece_rounds
)
sweep = decsim.Sweep(
    "ising_windows_pilot",
    machine,
    grid={
        "decoding": [
            "pymatching_alone",
            "ising_alone",
            "union_find_alone",
            "switching_ising_redo",
            "switching_ising_double",
        ],
        "distance": [9, 13],
        "physical_error_probability": [0.003, 0.006],
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    allow_unread=("tesseract", "belief_matching"),
)
experiment = sweep.experiment()
