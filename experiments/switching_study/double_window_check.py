"""Check the double window on a simple machine: union-find, then PyMatching.

The strong decoder is PyMatching on the host, so the only change between
the switching cells is the strong window's shape: redo_window, the
window union-find decoded, or double_window, the paper's r_com + 2 r_buf
rounds (2510.25222 Sec. III C). Two thresholds: the study's 20 dB, and
1000 dB, which escalates every window that has a defect. At 1000 dB the
redo window is PyMatching alone window by window, and the double window
is PyMatching on longer windows, so neither should lose to
pymatching_alone; one that does points at the shape itself. 10d-round
memories in windows of 2d rounds committing d, the same seeds in every
cell of one (d, p). Usage as ising_windows_pilot.py.
"""

import decsim
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.experiments.sweeps as sweeps
import decsim.presets.machines as machines
import decsim.topology.graph as graph

config = sweeps.load_beside(__file__, "machine.py")
constants = config.constants
ROUNDS_PER_DISTANCE = 10
CHECK_SHOTS = 20_000
SWITCHING_CELLS = {
    "redo_20dB": ("redo_window", "20dB"),
    "double_20dB": ("double_window", "20dB"),
    "redo_all": ("redo_window", "1000dB"),
    "double_all": ("double_window", "1000dB"),
}


def machine(
    decoding: str, distance: int, physical_error_probability: float
) -> graph.Machine:
    """One cell: a decoder alone, or union-find switching to PyMatching."""
    weak = config.union_find_machine(
        distance, physical_error_probability, ROUNDS_PER_DISTANCE
    )
    if decoding == "union_find_alone":
        return weak
    strong_window, threshold = SWITCHING_CELLS.get(
        decoding, ("redo_window", "20dB")
    )
    edge = _switching(strong_window, threshold)
    whole = machines.with_strong_tier(weak, _pymatching_pool(), edge)
    if decoding == "pymatching_alone":
        return _strong_alone(whole)
    return whole


def metadata(cell: dict, task: decsim.Task) -> dict:
    """The cells under the names the collected results carry."""
    return {
        "decoding": cell["decoding"],
        "rounds_per_distance": ROUNDS_PER_DISTANCE,
        "workload.arguments.physical_error_probability": cell[
            "physical_error_probability"
        ],
        "qpu.distance": cell["distance"],
    }


def _pymatching_pool() -> decoder_settings.DecoderPoolSettings:
    settings = constants.belief_matching
    engine = {"type": "Engine", "clock": constants.clocks.chip, **settings.engine}
    return config.record(
        decoder_settings.DecoderPoolSettings,
        settings.pool,
        algorithm={"type": "PyMatchingDecoder"},
        engine=engine,
    )


def _switching(strong_window: str, threshold: str):
    settings = constants.escalation
    return config.record(
        escalation_settings.SwitchingSettings,
        settings.edge,
        confidence={"type": "cluster_gap", **settings.confidence},
        threshold={"type": "FixedThreshold", "threshold": threshold},
        strong_window={"type": strong_window},
        clock=constants.clocks.chip,
    )


def _strong_alone(machine: graph.Machine) -> graph.Machine:
    without_switching = machine.without_part(machines.SWITCHING)
    without_pool = without_switching.without_part(machines.WEAK_POOL)
    return without_pool.without_part(machines.WEAK_STORE)


stop = decsim.CollectionSettings(
    max_shots=CHECK_SHOTS, piece_rounds=constants.piece_rounds
)
sweep = decsim.Sweep(
    "double_window_check",
    machine,
    grid={
        "decoding": [
            "union_find_alone",
            "pymatching_alone",
            *SWITCHING_CELLS,
        ],
        "distance": [5, 7, 9],
        "physical_error_probability": [0.002, 0.004],
    },
    task_name="{decoding}_d{distance}_p{physical_error_probability}",
    stop=stop,
    metadata=metadata,
    constants=constants,
    allow_unread=(
        "tesseract",
        "ising",
        "shots",
        "distances",
        "physical_error_probabilities",
        "escalation.threshold",
        "belief_matching.algorithm",
        "belief_matching.timing",
    ),
)
experiment = sweep.experiment()
