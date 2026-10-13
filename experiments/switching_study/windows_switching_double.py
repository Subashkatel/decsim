"""Windows with switching on the paper's double window: its backlog.

windows_switching.py's machine, with belief matching re-decoding the
paper's double window: union-find keeps sliding, and a window whose
cluster gap is under 20 dB hands belief matching r_com + 2 r_buf = 3d
rounds once union-find has fixed both of their faces (Toshio et al.
2510.25222 Sec. III C, Fig. 12). It runs the same cells and seeds as
windows_switching.py, so the two runs' backlog and timing compare on
the same shots; its accuracy is not belief matching alone's, which
decodes 2d windows. The machine is machine.py's.

Usage
-----

From this folder, on Della:

```
RESULTS=/scratch/gpfs/MARTONOSI/sk2415/decoder/results/switching_study
decsim run windows_switching_double.py --slurm \\
    --out $RESULTS/windows_switching_double/RUN
```
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
ROUNDS_PER_DISTANCE = 10


def machine(distance: int, physical_error_probability: float) -> graph.Machine:
    """One cell: the whole switching machine on the double window."""
    return config.machine(
        distance,
        physical_error_probability,
        ROUNDS_PER_DISTANCE,
        strong_window="double_window",
    )


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    qpu = task.machine.held("qpu")
    return {
        "decoding": "switching_double_window",
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
    "windows_switching_double",
    machine,
    grid={
        "distance": list(constants.distances),
        "physical_error_probability": list(
            constants.physical_error_probabilities
        ),
    },
    task_name="switching_double_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    allow_unread=("tesseract",),
)
experiment = sweep.experiment()
