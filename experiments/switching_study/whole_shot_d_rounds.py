"""Whole shot, d rounds: each decoder alone, against the paper's Fig. 15b.

Union-find alone, recording each shot's cluster gap, and belief matching
alone, on d-round memories decoded in one window, as Toshio et al.
2510.25222 decode offline (Sec. IV, lines 1407-1412). Switching at any
threshold is worked out afterwards from each shot's gap and the two
decoders' answers on the same shots. The machine is machine.py's, every
number in constants.yaml.

Usage
-----

From this folder, on Della:

```
RESULTS=/scratch/gpfs/MARTONOSI/sk2415/decoder/results/switching_study
decsim run whole_shot_d_rounds.py --slurm --out $RESULTS/whole_shot_d_rounds/RUN
```
"""

import dataclasses

import decsim
import decsim.confidence.cluster as cluster
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
    """One cell: union-find alone with its gap, or belief matching alone."""
    whole = config.machine(
        distance, physical_error_probability, ROUNDS_PER_DISTANCE
    )
    one_window = _in_one_window(whole, distance)
    if decoding == "union_find_alone":
        return _union_find_alone_recording_the_gap(one_window)
    return _belief_matching_alone(one_window)


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


def _union_find_alone_recording_the_gap(
    machine: graph.Machine,
) -> graph.Machine:
    """No switching and no belief matching; each window's gap recorded.

    window_outcomes.csv writes every shot's window, its gap beside its
    answer, where window_confidence.csv keeps the first 100 shots.
    """
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.STRONG_POOL)
    union_find_alone = without_pool.without_part(machines.STRONG_STORE)
    gap = config.record(
        cluster.ClusterGap.Settings, constants.escalation.confidence
    )
    windows = union_find_alone.held(machines.WINDOWS)
    recording = dataclasses.replace(windows, recorded_confidence=gap)
    with_gap = union_find_alone.with_held(machines.WINDOWS, recording)
    observation = dataclasses.replace(
        with_gap.observation, record_window_outcomes=True
    )
    return dataclasses.replace(with_gap, observation=observation)


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
    "whole_shot_d_rounds",
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
