"""The decoder settings at the yaml boundary."""

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings


def test_unit_memory_rounds_below_one_is_refused_with_a_sentence():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": 0,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
        },
    }
    with pytest.raises(ValueError, match="at least one round"):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_result_blocking_value_that_is_not_a_boolean_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "result_blocks_unit": "yes",
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
        },
    }
    with pytest.raises(
        ValueError, match="weak_decoder.result_blocks_unit 'yes'"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )
