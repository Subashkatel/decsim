"""Windows with switching: the real-time system, its accuracy and timing.

Union-find on every window first; a window whose cluster gap is under
20 dB is redone by belief matching on the same 2d rounds, committing the
same d (Toshio et al. 2510.25222 Sec. III, on machine.py's redo
window), on 10d-round memories in sliding windows of 2d rounds that
commit d: the windows windows_alone.py decodes. Every hop is timed by
the numbers in constants.yaml, and each shot records its windows'
switching and the decoders' backlog. The machine is machine.py's,
unchanged.

Usage
-----

From this folder, on Della:

```
RESULTS=/scratch/gpfs/MARTONOSI/sk2415/decoder/results/switching_study
decsim run windows_switching.py --slurm --out $RESULTS/windows_switching/RUN
```
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
ROUNDS_PER_DISTANCE = 10


def machine(distance: int, physical_error_probability: float) -> graph.Machine:
    """One cell: the whole switching machine."""
    return config.machine(
        distance, physical_error_probability, ROUNDS_PER_DISTANCE
    )


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    qpu = task.machine.held("qpu")
    return {
        "decoding": "switching",
        "rounds_per_distance": ROUNDS_PER_DISTANCE,
        "workload.arguments.physical_error_probability": cell[
            "physical_error_probability"
        ],
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": qpu.round_period_microseconds,
    }


stop = decsim.CollectionSettings(
    max_shots=constants.shots, piece_rounds=constants.piece_rounds
)
sweep = decsim.Sweep(
    "windows_switching",
    machine,
    grid={
        "distance": list(constants.distances),
        "physical_error_probability": list(
            constants.physical_error_probabilities
        ),
    },
    task_name="switching_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
