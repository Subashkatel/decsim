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


def block_points(block: str) -> list:
    """One block's points, at every distance it sweeps."""
    study = experiment.load(STUDY)
    prefix = f"{block}_d"
    return [point for point in study.points if point.name.startswith(prefix)]


def block_settings(block: str):
    """One block's settings, at its first point."""
    points = block_points(block)
    first_point = points[0]
    return first_point.machine


def test_the_control_block_copies_at_both_settings():
    """The reference defaults: input copy, boundary_fold copy."""
    settings = block_settings(CONTROL)

    assert settings.weak_decoder.copies_input
    assert settings.weak_decoder.copies_boundary_fold


def test_the_input_block_changes_the_input_and_nothing_else():
    base = block_settings(CONTROL)
    variant = block_settings(INPUT_IN_PLACE)

    assert not variant.weak_decoder.copies_input
    weak_fold = variant.weak_decoder.copies_boundary_fold
    assert weak_fold == base.weak_decoder.copies_boundary_fold
    assert variant.weak_decoder.algorithm == base.weak_decoder.algorithm
    assert variant.escalation_kind == base.escalation_kind


def test_the_fold_block_changes_the_fold_and_nothing_else():
    base = block_settings(CONTROL)
    variant = block_settings(FOLD_IN_PLACE)

    assert not variant.weak_decoder.copies_boundary_fold
    assert variant.weak_decoder.copies_input == base.weak_decoder.copies_input
    assert variant.weak_decoder.algorithm == base.weak_decoder.algorithm
    assert variant.escalation_kind == base.escalation_kind


def test_the_switching_block_opens_the_last_two_hops():
    """Hops 8 and 9 need a strong tier and its two links priced."""
    variant = block_settings(SWITCHING)
    fabric = variant.links

    assert variant.escalation_kind == "switching"
    assert variant.strong_decoder == STRONG_TOSHIO
    assert variant.strong_decoder.copies_input
    assert fabric.weak_decoder_to_strong_decoder is not None
    assert fabric.strong_buffer_to_strong_decoder is not None


@pytest.mark.parametrize("block", STUDY_BLOCKS)
def test_every_study_block_counts_its_data_movement(block):
    """The counters are off by default, so the grid asks for them."""
    settings = block_settings(block)
    assert settings.observation.data_movement
    assert settings.observation.trace == "chrome"


def swept_distances(block: str) -> list:
    """Every code distance one block names."""
    distances = set()
    for point in block_points(block):
        distances.add(point.machine.qpu.distance)
    return sorted(distances)


@pytest.mark.parametrize("block", STUDY_BLOCKS)
def test_every_study_block_sweeps_the_same_points_on_priced_cards(
    block,
):
    """A priced card decodes on no host clock, so the counts repeat."""
    settings = block_settings(block)
    distances = swept_distances(block)
    weak = WEAK_TOSHIO_BY_BLOCK[block]
    assert settings.weak_decoder.algorithm == weak.algorithm
    assert settings.weak_decoder.engine == weak.engine
    assert distances == [3, 5, 7]


def test_the_study_holds_the_four_blocks_and_every_point_builds_its_task():
    """No point names both in place, since its decoder pool would refuse it."""
    study = experiment.load(STUDY)

    tasks = [experiment.task_of(point) for point in study.points]

    assert len(tasks) == len(STUDY_BLOCKS) * 3
