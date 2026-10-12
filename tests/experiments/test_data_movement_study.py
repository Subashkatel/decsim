"""The data-movement study's four blocks: one setting apart, each.

The study varies how a tier's unit gets its rounds: whether the weak
decoder copies its input and whether it copies the boundary fold, both
copy in the control block, and its switching block adds the strong tier
and its priced links. So the blocks of experiments/data_movement/run.py
must differ in exactly those settings and in nothing else, and the pair
that stops a run, both in place, must not be among them.
"""

import pathlib

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.experiments.experiment as experiment
import decsim.presets.machines as machines

THIS_FILE = pathlib.Path(__file__)
STUDY = THIS_FILE.parents[2] / "experiments" / "data_movement" / "run.py"
# the study's blocks, in the order it writes them
CONTROL = "every_hop_copies"
INPUT_IN_PLACE = "weak_input_in_place"
FOLD_IN_PLACE = "boundary_folded_in_place"
SWITCHING = "switching"
STUDY_BLOCKS = (CONTROL, INPUT_IN_PLACE, FOLD_IN_PLACE, SWITCHING)
# Toshio et al.'s linear decoders on the grid's 250 MHz clocks: 0.4 and
# 10 tau_gen a round at tau_gen = 1 us (2510.25222 lines 1109-1114). The
# switching block's complementary gap solves each weak window twice.
_CLOCK_250_MEGAHERTZ = config.Clock(period_ticks=4_000)
WEAK_ONE_SOLVE = decoder_settings.linear_decoder_pool(
    0.4, _CLOCK_250_MEGAHERTZ, solves_per_window=1
)
WEAK_TWO_SOLVES = decoder_settings.linear_decoder_pool(
    0.4, _CLOCK_250_MEGAHERTZ, solves_per_window=2
)
WEAK_TOSHIO_BY_BLOCK = {
    CONTROL: WEAK_ONE_SOLVE,
    INPUT_IN_PLACE: WEAK_ONE_SOLVE,
    FOLD_IN_PLACE: WEAK_ONE_SOLVE,
    SWITCHING: WEAK_TWO_SOLVES,
}
STRONG_TOSHIO = decoder_settings.linear_decoder_pool(
    10.0, _CLOCK_250_MEGAHERTZ, solves_per_window=1
)


def block_tasks(block: str) -> list:
    """One block's tasks, at every distance it sweeps."""
    study = experiment.load(STUDY)
    prefix = f"{block}_d"
    return [task for task in study.tasks if task.name.startswith(prefix)]


def block_task(block: str):
    """One block's first task."""
    tasks = block_tasks(block)
    return tasks[0]


def weak_pool(block: str) -> decoder_settings.DecoderPoolSettings:
    """One block's weak pool, at its first task."""
    task = block_task(block)
    part = task.machine.part(machines.WEAK_POOL)
    return part.settings


def edge_names(block: str) -> list:
    """The names of one block's escalate edges, at its first task."""
    task = block_task(block)
    return [edge.name for edge in task.layout.edges]


def test_the_control_block_copies_at_both_settings():
    """The reference defaults: input copy, boundary_fold copy."""
    weak = weak_pool(CONTROL)

    assert weak.copies_input
    assert weak.copies_boundary_fold


def test_the_input_block_changes_the_input_and_nothing_else():
    base = weak_pool(CONTROL)
    variant = weak_pool(INPUT_IN_PLACE)

    assert not variant.copies_input
    assert variant.copies_boundary_fold == base.copies_boundary_fold
    assert variant.algorithm == base.algorithm
    assert edge_names(INPUT_IN_PLACE) == edge_names(CONTROL)


def test_the_fold_block_changes_the_fold_and_nothing_else():
    base = weak_pool(CONTROL)
    variant = weak_pool(FOLD_IN_PLACE)

    assert not variant.copies_boundary_fold
    assert variant.copies_input == base.copies_input
    assert variant.algorithm == base.algorithm
    assert edge_names(FOLD_IN_PLACE) == edge_names(CONTROL)


def test_the_switching_block_opens_the_last_two_hops():
    """Hops 8 and 9 need a strong tier and its two links priced."""
    task = block_task(SWITCHING)
    strong = task.machine.part(machines.STRONG_POOL)
    paths = task.layout.card.path_names()

    assert edge_names(SWITCHING) == [machines.SWITCHING]
    assert strong.settings == STRONG_TOSHIO
    assert strong.settings.copies_input
    assert "weak_to_strong" in paths
    assert "strong_buffer_to_strong" in paths


@pytest.mark.parametrize("block", STUDY_BLOCKS)
def test_every_study_block_counts_its_data_movement(block):
    """The counters are off by default, so the grid asks for them."""
    task = block_task(block)
    observation = task.machine.observation
    assert observation.data_movement
    assert observation.trace == "chrome"


def swept_distances(block: str) -> list:
    """Every code distance one block names."""
    distances = set()
    for task in block_tasks(block):
        qpu = task.machine.held("qpu")
        distances.add(qpu.distance)
    return sorted(distances)


@pytest.mark.parametrize("block", STUDY_BLOCKS)
def test_every_study_block_sweeps_the_same_distances_on_priced_cards(
    block,
):
    """A priced card decodes on no host clock, so the counts repeat."""
    pool = weak_pool(block)
    distances = swept_distances(block)
    weak = WEAK_TOSHIO_BY_BLOCK[block]
    assert pool.algorithm == weak.algorithm
    assert pool.engine == weak.engine
    assert distances == [3, 5, 7]


def test_the_study_holds_the_four_blocks_and_every_task_builds():
    """No task names both in place, since its decoder pool would refuse it."""
    study = experiment.load(STUDY)

    tasks = study.tasks

    assert len(tasks) == len(STUDY_BLOCKS) * 3
