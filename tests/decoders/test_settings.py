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


def test_the_formation_keys_default_to_yangs_latency_at_one_round_a_clock():
    """5 FPGA clock cycles (2605.04892 lines 1274-1275), one round a clock."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
        },
    }

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.detection_event_latency_cycles == 5
    assert settings.detection_event_cycles_per_round == 1


def test_the_engine_card_reads_both_formation_keys():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
            "detection_event_latency_cycles": 9,
            "detection_event_cycles_per_round": 2,
        },
    }

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.detection_event_latency_cycles == 9
    assert settings.detection_event_cycles_per_round == 2
