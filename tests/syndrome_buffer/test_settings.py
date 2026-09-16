"""Buffer 0 access cycles, on gem5's named clock and integer-cycle law."""

import pytest

import decsim.config as config
import decsim.experiments.experiment as experiment
import decsim.syndrome_buffer.settings as round_store_settings
import tests.experiments.yaml_configs as yaml_configs


@pytest.mark.parametrize("key", ["write_cycles", "read_cycles"])
@pytest.mark.parametrize("cycles", [True, 0.5, float("nan"), float("inf")])
def test_store_cycle_keys_refuse_noninteger_values_by_name(key, cycles):
    clocks = config.ClockSettings({"storage": 250.0})
    section = {"clock": "storage", key: cycles}
    sentence = f"round_store.{key} must be a nonnegative integer"
    with pytest.raises(ValueError, match=sentence):
        round_store_settings.RoundStoreSettings.from_yaml(section, clocks)


def test_the_store_clock_defaults_to_the_controllers_domain(tmp_path):
    controller = dict(yaml_configs.MINIMAL_CONFIG["controller"])
    controller["clock"] = "room"
    path = yaml_configs.write_config(
        tmp_path,
        {"controller": controller, "round_store": {"write_cycles": 3}},
    )
    loaded = experiment.load_experiment(path)
    assert loaded.settings.round_store.clock == loaded.settings.controller.clock
    assert loaded.settings.round_store.write_cycles == 3
    assert loaded.settings.round_store.read_cycles == 0


def test_a_store_clock_must_name_a_declared_domain():
    clocks = config.ClockSettings({"storage": 250.0})
    section = {"clock": "missing"}
    with pytest.raises(
        ValueError, match="clock 'missing' is not a clocks entry"
    ):
        round_store_settings.RoundStoreSettings.from_yaml(section, clocks)


def test_a_charged_store_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged round_store costs need a clock"
    ):
        round_store_settings.RoundStoreSettings(read_cycles=1)
