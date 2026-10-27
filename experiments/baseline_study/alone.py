"""Each decoder alone (A1-A5): every window decoded by one decoder.

The windows every switching run decodes, 2d rounds committing d over
10d-round memories, with the same seeds, so each switching run is
compared with its decoders alone on the same shots. The weak decoders
record their gap on every window beside its answer (window_outcomes.csv),
deciding nothing, so which windows switching would escalate, and whether
the weak decoder needed it, is read off this run. Union-find records the
cluster gap, matching the complementary gap.

Usage
-----

From this folder, in Slurm jobs (docs/how-to/run_on_a_slurm_cluster.md):

```
export PYTHONPATH=$PWD:$PYTHONPATH   # the Ising row is imported by name
decsim run alone.py --record --out $OUT                      # once
decsim run alone.py --out $OUT --share $I/$N --processes P   # each job
decsim run --fold --out $OUT                                 # once, at the end
```
"""

import dataclasses

import decsim
import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
# the weak decoders alone: the decoder and the gap each window records
WEAK_CELLS = {
    "union_find": (
        "union_find",
        cluster.ClusterGap.Settings,
        constants.escalation.confidence,
    ),
    "minimum_weight_perfect_matching": (
        "minimum_weight_perfect_matching",
        complementary.ComplementaryGap.Settings,
        {},
    ),
}
STRONG_CELLS = ("belief_matching", "ising", "correlated_matching")


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One cell: one decoder alone on every window."""
    if decoding in WEAK_CELLS:
        return _weak_alone(decoding, distance, physical_error_probability)
    whole = config.machine(
        "union_find",
        decoding,
        "redo_window",
        distance,
        physical_error_probability,
    )
    without_switching = whole.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.WEAK_POOL)
    return without_pool.without_part(machines.WEAK_STORE)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    qpu = task.machine.held("qpu")
    return {
        "decoding": f"{cell['decoding']}_alone",
        "rounds_per_distance": constants.rounds_per_distance,
        "workload.arguments.physical_error_probability": cell[
            "physical_error_probability"
        ],
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": qpu.round_period_microseconds,
    }


def _weak_alone(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """The weak decoder alone, its gap recorded on every window."""
    weak, gap_settings, gap_constants = WEAK_CELLS[decoding]
    alone = config.weak_machine(weak, distance, physical_error_probability)
    gap = config.record(gap_settings, gap_constants)
    windows = alone.held(machines.WINDOWS)
    recording = dataclasses.replace(windows, recorded_confidence=gap)
    with_gap = alone.with_held(machines.WINDOWS, recording)
    observation = dataclasses.replace(
        with_gap.observation, record_window_outcomes=True
    )
    return dataclasses.replace(with_gap, observation=observation)


stop = decsim.CollectionSettings(
    max_shots=constants.shots, piece_rounds=constants.piece_rounds
)
sweep = decsim.Sweep(
    "alone",
    machine,
    grid={
        "decoding": [*WEAK_CELLS, *STRONG_CELLS],
        "distance": list(constants.distances),
        "physical_error_probability": list(
            constants.physical_error_probabilities
        ),
    },
    task_name="{decoding}_alone_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    allow_unread=("tesseract", "tesseract_shots_from_distance_9"),
)
experiment = sweep.experiment()
