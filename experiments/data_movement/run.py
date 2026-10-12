"""The data-movement study: four blocks, d = 3, 5 and 7.

Each task reports, under the names its results carry, every setting
some block changes, read off the machine it runs; a hop goes by its
name on the machine's card.
"""

import decsim
import decsim.config as config_module
import decsim.decoders.settings as decoder_settings
import decsim.experiments.sweeps as sweeps
import decsim.links.settings as link_settings
import decsim.presets.clocks as clocks
import decsim.presets.machines as machines

config = sweeps.load_beside(__file__, "config.py")
constants = config.constants
# The word a results column shows for whether a pool copies its input,
# and its boundary fold.
WORDS = {True: "copy", False: "in_place"}
# Blocks 1 to 3 run no strong tier: the weak base names three
# strong-side hops null and writes no fourth hop and no strong decoder.
NO_STRONG_SIDE_CELLS = {
    "escalation": {"kind": "weak_baseline"},
    "strong_decoder": "",
    "links.controller_to_strong_buffer": None,
    "links.weak_to_strong": None,
    "links.strong_buffer_to_strong": None,
    "links.strong_to_frame": "",
}


# the strong side's hops, in column order
STRONG_SIDE_HOPS = (
    "controller_to_strong_buffer",
    "weak_to_strong",
    "strong_buffer_to_strong",
    "strong_to_frame",
)


def metadata(cell: dict, task: decsim.Task) -> dict:
    """Every cell the study's results show for a task, in column order."""
    machine = task.machine
    card = task.layout.card
    weak_decoder = machine.part(machines.WEAK_POOL).settings
    weak_write = card.path_settings("controller_to_weak_buffer")
    weak_link = link_cell(weak_write, "fridge", clocks.FRIDGE_CLOCK)
    input_word = WORDS[weak_decoder.copies_input]
    fold_word = WORDS[weak_decoder.copies_boundary_fold]
    weak_fetch_cycles = weak_decoder.engine.fetch_cycles_per_round
    strong_side = NO_STRONG_SIDE_CELLS
    if task.layout.edges:
        strong_side = strong_side_cells(task)
    return {
        "links.controller_to_weak_buffer": weak_link,
        "workload.arguments.physical_error_probability": (
            constants.physical_error_probability
        ),
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": constants.round_period_microseconds,
        "weak.input": input_word,
        "weak.boundary_fold": fold_word,
        "weak.engine.fetch_cycles_per_round": weak_fetch_cycles,
        **strong_side,
    }


def strong_side_cells(task: decsim.Task) -> dict:
    """Block 4's escalation, strong pool and strong-side hops, on the host."""
    card = task.layout.card
    room = clocks.ROOM_CLOCK
    threshold = task.layout.entry_switching().threshold
    escalation = {
        "kind": "switching",
        "gap_threshold_db": threshold.threshold_decibels,
        "strong_window": "redo_window",
    }
    strong_pool = task.machine.part(machines.STRONG_POOL)
    strong_decoder = pool_cell(strong_pool.settings, "room")
    cells = {"escalation": escalation, "strong_decoder": strong_decoder}
    for hop in STRONG_SIDE_HOPS:
        path = card.path_settings(hop)
        cells[f"links.{hop}"] = link_cell(path, "room", room)
    return cells


def link_cell(
    path: link_settings.PathSettings,
    clock_name: str,
    clock: config_module.Clock,
) -> dict:
    """A path's card in the cycles of its clock, on an unbounded wire."""
    latency_ticks = path.channel.propagation_latency_ticks
    latency_cycles = latency_ticks // clock.period_ticks
    return {
        "latency_cycles": latency_cycles,
        "clock": clock_name,
        "bits_per_cycle": None,
    }


def pool_cell(
    pool: decoder_settings.DecoderPoolSettings, clock_name: str
) -> dict:
    """A priced matching pool: its card, units, input, memory and engine."""
    engine = pool.engine
    unit_memory = {"bits": pool.unit_memory.bits}
    engine_cell = {
        "clock": clock_name,
        "fetch_cycles_per_round": engine.fetch_cycles_per_round,
        "fetch_cycles_per_job": engine.fetch_cycles_per_job,
        "release_cycles_per_job": engine.release_cycles_per_job,
        "release_cycles_per_round": engine.release_cycles_per_round,
    }
    # a priced matching card's kind is its latency in microseconds
    return {
        "kind": pool.algorithm.preset_latency_microseconds,
        "units": pool.unit_count,
        "input": WORDS[pool.copies_input],
        "unit_memory": unit_memory,
        "engine": engine_cell,
    }


stop = decsim.CollectionSettings(max_shots=constants.stop.max_shots)
sweep = decsim.Sweep(
    "data_movement",
    config.machine,
    grid={"block": list(config.BLOCKS), "distance": [3, 5, 7]},
    task_name="{block}_d{distance}",
    stop_all=stop,
    metadata=metadata,
    constants=constants,
)
experiment = sweep.experiment()
