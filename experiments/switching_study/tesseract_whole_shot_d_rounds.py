"""Whole shot, d rounds, Tesseract alone: the reference beside both decoders.

Tesseract (2503.10988), on the paper's short-beam profile, decodes the
d-round memories whole_shot_d_rounds.py decodes: the same machine, task
names, cells and seeds, so each shot is the one union-find and belief
matching saw, which every shot's sample_digest shows. Its time is not
studied. The machine is machine.py's, every number in constants.yaml.

Usage
-----

From this folder, on Adroit:

```
RESULTS=/scratch/network/sk2415/decoder/escalation/results/switching_study
decsim run tesseract_whole_shot_d_rounds.py --slurm \
    --out $RESULTS/tesseract_whole_shot_d_rounds/RUN
```
"""

import dataclasses

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph
import decsim.windows.schemes.sliding as sliding_scheme

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
ROUNDS_PER_DISTANCE = 1


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One cell: Tesseract alone, the whole shot in one window."""
    del decoding
    whole = config.machine(
        distance, physical_error_probability, ROUNDS_PER_DISTANCE
    )
    one_window = _in_one_window(whole, distance)
    return _tesseract_alone(one_window)


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


def _in_one_window(machine: graph.Machine, distance: int) -> graph.Machine:
    """The machine committing the whole shot in its one window."""
    rounds = ROUNDS_PER_DISTANCE * distance
    scheme = sliding_scheme.SlidingWindowScheme.Settings(commit_rounds=rounds)
    windows = machine.held(machines.WINDOWS)
    whole_shot = dataclasses.replace(windows, scheme=scheme)
    return machine.with_held(machines.WINDOWS, whole_shot)


def _tesseract_alone(machine: graph.Machine) -> graph.Machine:
    """No switching and no union-find: Tesseract takes every window."""
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.WEAK_POOL)
    without_store = without_pool.without_part(machines.WEAK_STORE)
    tesseract = config.tesseract_pool()
    return without_store.with_held(machines.STRONG_POOL, tesseract)


stop = decsim.CollectionSettings(
    max_shots=constants.shots, piece_rounds=constants.piece_rounds
)
# each piece's statistics summed: the summary needs no shot's own rows
record_options = decsim.RecordOptions(is_statistics_per_shot=False)
sweep = decsim.Sweep(
    "tesseract_whole_shot_d_rounds",
    machine,
    grid={
        "decoding": ["tesseract_alone"],
        "distance": list(constants.distances),
        "physical_error_probability": list(
            constants.physical_error_probabilities
        ),
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    record_options=record_options,
)
experiment = sweep.experiment()
