"""Tesseract alone (A6), the near-optimal reference, whole shot.

Each 10d-round memory decoded at once, no windows (Tesseract 2503.10988,
its short-beam profile in constants.yaml), the same seeds as alone.py
and switching.py. Every shot at d = 3-7; from d = 9 only the first
tesseract_shots_from_distance_9 seeds of each point, since a full run
takes about 40 s a shot at d = 13, and every other decoder is compared
with it on exactly those shots. Usage as alone.py's.
"""

import dataclasses

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph
import decsim.windows.schemes.sliding as sliding_scheme

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants


def machine(distance: int, physical_error_probability: float) -> graph.Machine:
    """One cell: Tesseract alone, the whole shot in one window."""
    whole = config.machine(
        "union_find",
        "belief_matching",
        "redo_window",
        distance,
        physical_error_probability,
    )
    rounds = constants.rounds_per_distance * distance
    scheme = sliding_scheme.SlidingWindowScheme.Settings(commit_rounds=rounds)
    windows = whole.held(machines.WINDOWS)
    whole_shot = dataclasses.replace(windows, scheme=scheme)
    one_window = whole.with_held(machines.WINDOWS, whole_shot)
    without_switching = one_window.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.WEAK_POOL)
    without_store = without_pool.without_part(machines.WEAK_STORE)
    tesseract = config.tesseract_pool()
    return without_store.with_held(machines.STRONG_POOL, tesseract)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    qpu = task.machine.held("qpu")
    return {
        "decoding": "tesseract_alone",
        "rounds_per_distance": constants.rounds_per_distance,
        "workload.arguments.physical_error_probability": cell[
            "physical_error_probability"
        ],
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": qpu.round_period_microseconds,
    }


def stop(cell: dict) -> decsim.CollectionSettings:
    """Every shot below d = 9; the first seeds only from d = 9."""
    shots = constants.shots
    if cell["distance"] >= 9:
        shots = constants.tesseract_shots_from_distance_9
    return decsim.CollectionSettings(
        max_shots=shots, piece_rounds=constants.piece_rounds
    )


sweep = decsim.Sweep(
    "tesseract",
    machine,
    grid={
        "distance": list(constants.distances),
        "physical_error_probability": list(
            constants.physical_error_probabilities
        ),
    },
    task_name="tesseract_alone_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    allow_unread=(
        "minimum_weight_perfect_matching",
        "belief_matching",
        "ising",
        "correlated_matching",
        "escalation",
        "gpu_link",
    ),
)
experiment = sweep.experiment()
