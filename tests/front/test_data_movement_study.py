"""The data-movement study's four configs: one setting apart, each.

The study reads the data path against docs/rewrite/notes/
data_movement_research.md, so its configs must differ in exactly the
settings that note's section 5 names and in nothing else, and the pair
that note calls refused must not be among them. The sweep grammar
carries no setting axis, so each combination is its own yaml through
`extends`, and these tests pin what each one changed.
"""

import pytest

import decsim.front.experiment as experiment
from tests.front.yaml_configs import CONFIGS_DIR

BASE = "data_movement.yaml"
INPUT_IN_PLACE = "data_movement_input_in_place.yaml"
FOLD_IN_PLACE = "data_movement_fold_in_place.yaml"
SWITCHING = "data_movement_switching.yaml"
STUDY_CONFIGS = (BASE, INPUT_IN_PLACE, FOLD_IN_PLACE, SWITCHING)


def study_settings(name):
    """One study config's resolved settings."""
    config_path = CONFIGS_DIR / name
    config = experiment.load_experiment(config_path)
    return config.settings


def test_the_base_config_copies_at_both_of_the_notes_settings():
    """Section 5's defaults: input copy, boundary_fold copy."""
    settings = study_settings(BASE)

    assert settings.weak_decoder.input == "copy"
    assert settings.weak_decoder.boundary_fold == "copy"


def test_the_input_config_changes_the_input_and_nothing_else():
    base = study_settings(BASE)
    variant = study_settings(INPUT_IN_PLACE)

    assert variant.weak_decoder.input == "in_place"
    assert variant.weak_decoder.boundary_fold == base.weak_decoder.boundary_fold
    assert variant.weak_decoder.kind == base.weak_decoder.kind
    assert variant.escalation.kind == base.escalation.kind


def test_the_fold_config_changes_the_fold_and_nothing_else():
    base = study_settings(BASE)
    variant = study_settings(FOLD_IN_PLACE)

    assert variant.weak_decoder.boundary_fold == "in_place"
    assert variant.weak_decoder.input == base.weak_decoder.input
    assert variant.weak_decoder.kind == base.weak_decoder.kind
    assert variant.escalation.kind == base.escalation.kind


def test_the_switching_config_opens_the_last_two_hops():
    """Hops 8 and 9 need a strong tier and its two links priced."""
    variant = study_settings(SWITCHING)
    fabric = variant.links

    assert variant.escalation.kind == "switching"
    assert variant.strong_decoder.kind == 10.0
    assert variant.strong_decoder.input == "copy"
    assert fabric.weak_decoder_to_strong_decoder is not None
    assert fabric.strong_buffer_to_strong_decoder is not None


def test_every_study_config_counts_its_data_movement():
    """The counters are off by default, so each config asks for them."""
    for name in STUDY_CONFIGS:
        settings = study_settings(name)
        assert settings.observation.data_movement, name
        assert settings.observation.trace == "chrome", name


def swept_distances(name):
    """Every code distance one config's sweep blocks name."""
    config_path = CONFIGS_DIR / name
    config = experiment.load_experiment(config_path)
    distances = set()
    for block in config.sweep:
        for point in block.points():
            distances.add(point[1])
    return sorted(distances)


def test_every_study_config_sweeps_the_same_points_on_priced_cards():
    """A priced card decodes on no host clock, so the counts repeat."""
    for name in STUDY_CONFIGS:
        settings = study_settings(name)
        distances = swept_distances(name)
        assert settings.weak_decoder.kind == 1.0, name
        assert distances == [3, 5, 7], name


def test_reading_the_input_in_place_and_folding_in_place_is_refused(tmp_path):
    """The pair the note's table cannot name, refused where it is asked.

    Folding into the unit's memory needs the unit's own copy of the
    rounds, and a tier that reads its input in place has none, so no
    study config names that pair and the machine says why.
    """
    import decsim.collect as collect

    both_path = _both_in_place_config(tmp_path)
    config = experiment.load_experiment(both_path)
    task = config.point_task(
        physical_error_probability=0.01,
        distance=3,
        round_period_us=1.0,
        shots=1,
    )
    with pytest.raises(RuntimeError, match="boundary_fold in_place"):
        collect.run_shot(task, 0)


def _both_in_place_config(tmp_path):
    """The study's base yaml with both in-place settings named at once."""
    import yaml

    from tests.front.yaml_configs import MINIMAL_CONFIG

    weak = dict(MINIMAL_CONFIG["weak_decoder"])
    weak["input"] = "in_place"
    weak["boundary_fold"] = "in_place"
    raw = dict(MINIMAL_CONFIG)
    raw["weak_decoder"] = weak
    raw["sweep"] = [
        {
            "physical_error_probability": [0.01],
            "distance": [3],
            "round_period_us": [1.0],
            "shots": 1,
        }
    ]
    config_path = tmp_path / "both_in_place.yaml"
    written = yaml.safe_dump(raw)
    config_path.write_text(written)
    return config_path
