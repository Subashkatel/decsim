"""The data-movement study in three files: four blocks, d = 3, 5 and 7.

Each task reports, under the names its results carry, every setting
some block changes, read off the machine it runs. The tasks are
experiments/data_movement/run.py's (tests/test_three_files_example.py).
"""

import decsim
import decsim.config as config_module
import decsim.decoders.settings as decoder_settings
import decsim.experiments.sweeps as sweeps
import decsim.links.settings as link_settings
import decsim.presets.clocks as clocks

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
    "links.weak_decoder_to_strong_decoder": None,
    "links.strong_buffer_to_strong_decoder": None,
    "links.strong_decoder_to_frame": "",
}


def metadata(cell: dict, settings: decsim.MachineSettings) -> dict:
    """Every cell the study's results show for a task, in column order."""
    weak_decoder = settings.weak_decoder
    weak_link = link_cell(
        settings.links.controller_to_weak_buffer, "fridge", clocks.FRIDGE_CLOCK
    )
    input_word = WORDS[weak_decoder.copies_input]
    fold_word = WORDS[weak_decoder.copies_boundary_fold]
    weak_fetch_cycles = weak_decoder.engine.fetch_cycles_per_round
    strong_side = NO_STRONG_SIDE_CELLS
    if settings.switching is not None:
        strong_side = strong_side_cells(settings)
    return {
        "links.controller_to_weak_buffer": weak_link,
        "workload.arguments.physical_error_probability": (
            constants.physical_error_probability
        ),
        "qpu.distance": cell["distance"],
        "qpu.round_period_microseconds": constants.round_period_microseconds,
        "weak_decoder.input": input_word,
        "weak_decoder.boundary_fold": fold_word,
        "weak_decoder.engine.fetch_cycles_per_round": weak_fetch_cycles,
        **strong_side,
    }


def strong_side_cells(settings: decsim.MachineSettings) -> dict:
    """Block 4's escalation, strong pool and strong-side hops, on the host."""
    links = settings.links
    room = clocks.ROOM_CLOCK
    threshold = settings.switching.threshold
    escalation = {
        "kind": "switching",
        "gap_threshold_db": threshold.threshold_decibels,
        "strong_window": "redo_window",
    }
    strong_decoder = pool_cell(settings.strong_decoder, "room")
    write = link_cell(links.controller_to_strong_buffer, "room", room)
    escalated = link_cell(links.weak_decoder_to_strong_decoder, "room", room)
    read = link_cell(links.strong_buffer_to_strong_decoder, "room", room)
    result = link_cell(links.strong_decoder_to_frame, "room", room)
    return {
        "escalation": escalation,
        "strong_decoder": strong_decoder,
        "links.controller_to_strong_buffer": write,
        "links.weak_decoder_to_strong_decoder": escalated,
        "links.strong_buffer_to_strong_decoder": read,
        "links.strong_decoder_to_frame": result,
    }


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
