"""The data-movement study's four blocks: one setting apart, each.

The study varies the two keys configs/reference.yaml gives for how a
tier's unit gets its rounds, weak_decoder.input and
weak_decoder.boundary_fold, both copy by default, and its switching
block adds the strong tier and its priced links. So the blocks of
configs/experiments/data_movement/data_movement.yaml must differ in
exactly those settings and in nothing else, and the pair the load
refuses, both in place, must not be among them.
"""

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.experiments.experiment as experiment
import tests.experiments.yaml_configs as yaml_configs

GRID = (
    yaml_configs.CONFIGS_DIR
    / "experiments"
    / "data_movement"
    / "data_movement.yaml"
)
# the grid's sweep blocks, in the order it writes them
CONTROL = 0
INPUT_IN_PLACE = 1
FOLD_IN_PLACE = 2
SWITCHING = 3
STUDY_BLOCKS = (CONTROL, INPUT_IN_PLACE, FOLD_IN_PLACE, SWITCHING)
# Toshio et al.'s linear decoders on the grid's 250 MHz clocks: 0.4 and
# 10 tau_gen a round at tau_gen = 1 us (2510.25222 lines 1109-1114)
_CLOCK_250_MEGAHERTZ = config.Clock(period_ticks=4_000)
WEAK_TOSHIO = decoder_settings.toshio_decoder_pool(0.4, _CLOCK_250_MEGAHERTZ)
STRONG_TOSHIO = decoder_settings.toshio_decoder_pool(10.0, _CLOCK_250_MEGAHERTZ)


def block_settings(block_index):
    """One block's resolved settings, at its first point."""
    config = experiment.load_experiment(GRID)
    block = config.sweep[block_index]
    points = block.points()
    task = config.point_task(points[0])
    return task.settings


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


@pytest.mark.parametrize("block_index", STUDY_BLOCKS)
def test_every_study_block_counts_its_data_movement(block_index):
    """The counters are off by default, so the grid asks for them."""
    settings = block_settings(block_index)
    assert settings.observation.data_movement
    assert settings.observation.trace == "chrome"


def swept_distances(block_index):
    """Every code distance one block names."""
    config = experiment.load_experiment(GRID)
    block = config.sweep[block_index]
    distances = set()
    for point in block.points():
        distances.add(point["qpu.distance"])
    return sorted(distances)


@pytest.mark.parametrize("block_index", STUDY_BLOCKS)
def test_every_study_block_sweeps_the_same_points_on_priced_cards(
    block_index,
):
    """A priced card decodes on no host clock, so the counts repeat."""
    settings = block_settings(block_index)
    distances = swept_distances(block_index)
    assert settings.weak_decoder.algorithm == WEAK_TOSHIO.algorithm
    assert settings.weak_decoder.engine == WEAK_TOSHIO.engine
    assert distances == [3, 5, 7]


def test_the_grid_holds_the_four_blocks_and_every_point_loads():
    """No point names both in place, since the load would refuse it."""
    config = experiment.load_experiment(GRID)

    tasks = config.tasks()

    assert len(config.sweep) == len(STUDY_BLOCKS)
    assert len(tasks) == len(STUDY_BLOCKS) * 3


def test_reading_the_input_in_place_and_folding_in_place_is_refused(tmp_path):
    """Both in place, refused where the yaml enters.

    Folding into the unit's memory needs the unit's own copy of the
    rounds, and a tier that reads its input in place has none, so no
    block of the study names that pair and the load says why.
    """
    both_path = _both_in_place_config(tmp_path)

    with pytest.raises(ValueError, match="boundary_fold in_place\\) needs"):
        experiment.load_experiment(both_path)


def _both_in_place_config(tmp_path):
    """The study's base yaml with both in-place settings named at once."""
    import yaml

    weak = dict(yaml_configs.MINIMAL_CONFIG["weak_decoder"])
    weak["input"] = "in_place"
    weak["boundary_fold"] = "in_place"
    raw = dict(yaml_configs.MINIMAL_CONFIG)
    raw["weak_decoder"] = weak
    raw["sweep"] = [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.01],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": 1},
        }
    ]
    config_path = tmp_path / "both_in_place.yaml"
    written = yaml.safe_dump(raw)
    config_path.write_text(written)
    return config_path
