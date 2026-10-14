"""Switching (B1-B10): a weak decoder, then a strong one on low-gap windows.

Every pair of the list, on 10d-round memories in windows of 2d rounds
committing d, the same seeds as alone.py. Union-find escalates on the
cluster gap and matching on the complementary gap, both at 20 dB
(Toshio et al. 2510.25222 Figs. 15, 16). The strong decoder re-decodes
the escalated window (redo) or the paper's r_com + 2 r_buf rounds
(double, Sec. III C). Each shot records its windows' switching and the
decoders' backlog. Usage as alone.py's.
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
WEAK_SHORT = {"union_find": "uf", "minimum_weight_perfect_matching": "mwpm"}
# the list's ten pairs: (weak, strong, strong window)
PAIRS = {
    "B1": ("union_find", "belief_matching", "redo_window"),
    "B2": ("minimum_weight_perfect_matching", "belief_matching", "redo_window"),
    "B3": ("union_find", "ising", "redo_window"),
    "B4": ("minimum_weight_perfect_matching", "ising", "redo_window"),
    "B5": ("union_find", "belief_matching", "double_window"),
    "B6": (
        "minimum_weight_perfect_matching",
        "belief_matching",
        "double_window",
    ),
    "B7": ("union_find", "ising", "double_window"),
    "B8": ("minimum_weight_perfect_matching", "ising", "double_window"),
    "B9": ("union_find", "correlated_matching", "redo_window"),
    "B10": (
        "minimum_weight_perfect_matching",
        "correlated_matching",
        "redo_window",
    ),
}


def machine(
    pair: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One cell: the pair's whole switching machine."""
    weak, strong, strong_window = PAIRS[pair]
    return config.machine(
        weak, strong, strong_window, distance, physical_error_probability
    )


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    qpu = task.machine.held("qpu")
    weak, strong, strong_window = PAIRS[cell["pair"]]
    return {
        "decoding": f"{cell['pair']}_{WEAK_SHORT[weak]}_to_{strong}_{strong_window}",
        "rounds_per_distance": constants.rounds_per_distance,
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
    "switching",
    machine,
    grid={
        "pair": list(PAIRS),
        "distance": list(constants.distances),
        "physical_error_probability": list(
            constants.physical_error_probabilities
        ),
    },
    task_name="{pair}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    allow_unread=("tesseract", "tesseract_shots_from_distance_9"),
)
experiment = sweep.experiment()
