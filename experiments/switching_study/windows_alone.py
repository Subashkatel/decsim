"""Windows, each decoder alone: what windowing costs each decoder.

Union-find alone and belief matching alone on 10d-round memories, in
sliding windows of 2d rounds that commit d (decsim's sliding default,
after Skoric et al. 2209.08552), laid out as windows_switching.py lays
them, so the three runs decode the same windows of the same shots. The
machine is machine.py's, every number in constants.yaml.

Usage
-----

From this folder, on Della:

```
RESULTS=/scratch/gpfs/MARTONOSI/sk2415/decoder/results/switching_study
decsim run windows_alone.py --slurm --out $RESULTS/windows_alone/RUN
```
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
ROUNDS_PER_DISTANCE = 10


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One cell: union-find alone, or belief matching alone."""
    whole = config.machine(
        distance, physical_error_probability, ROUNDS_PER_DISTANCE
    )
    if decoding == "union_find_alone":
        return _union_find_alone(whole)
    return _belief_matching_alone(whole)


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


def _union_find_alone(machine: graph.Machine) -> graph.Machine:
    """No switching and no belief matching: union-find keeps every window."""
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.STRONG_POOL)
    return without_pool.without_part(machines.STRONG_STORE)


def _belief_matching_alone(machine: graph.Machine) -> graph.Machine:
    """No switching and no union-find: belief matching takes every window."""
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.WEAK_POOL)
    return without_pool.without_part(machines.WEAK_STORE)


stop = decsim.CollectionSettings(
    max_shots=constants.shots, piece_rounds=constants.piece_rounds
)
# each piece's statistics summed: the summary needs no shot's own rows
record_options = decsim.RecordOptions(is_statistics_per_shot=False)
sweep = decsim.Sweep(
    "windows_alone",
    machine,
    grid={
        "decoding": ["union_find_alone", "belief_matching_alone"],
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
