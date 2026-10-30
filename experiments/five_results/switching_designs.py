r"""The switching designs the grid runs, and how each is built.

switching_grid.py (shots at d = 5 and 7), switching_grid_d9.py (the
30% share at d = 9) and switching_grid_streams.py (streams) share these
machines. Each design is a window rule and a count of strong cores; a
design runs at the threshold calibrated to a share (calibration.py).
"""

import decsim
import decsim.experiments.sweeps as sweeps
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "config.py")
calibration = sweeps.load_beside(__file__, "calibration.py")
# design -> (window rule, strong cores)
DESIGNS = {
    "sliding_1core": ("sliding", 1),
    "speculation_1core": ("speculation", 1),
    "double_1core": ("double", 1),
    "double_2core": ("double", 2),
}
SHARES = (0.05, 0.1, 0.2, 0.3)
PHYSICAL_ERROR_PROBABILITIES = (0.003,)
TASK_NAME = "{design}_{share}_d{distance}_p{physical_error_probability}"


def machine(
    design: str,
    share: float,
    distance: int,
    physical_error_probability: float,
) -> graph.Machine:
    """One design at the threshold calibrated to the share."""
    scheme, strong_unit_count = DESIGNS[design]
    point = (distance, physical_error_probability)
    threshold_decibels = calibration.CALIBRATED[point][share]
    return config.switching(
        scheme,
        distance,
        physical_error_probability,
        threshold_decibels,
        strong_unit_count,
    )


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The point's cells as config.metadata names them, and the design's parts."""
    scheme, strong_unit_count = DESIGNS[cell["design"]]
    named = config.metadata(cell, task)
    return {**named, "scheme": scheme, "strong_unit_count": strong_unit_count}


stop = decsim.CollectionSettings(
    max_shots=config.EVALUATION_SHOTS, first_seed=config.EVALUATION_FIRST_SEED
)
