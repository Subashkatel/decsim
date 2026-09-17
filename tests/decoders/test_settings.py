"""The decoder settings at the yaml boundary."""

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.records.decoder_evidence as evidence_records


def test_unit_memory_rounds_below_one_is_refused_with_a_sentence():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": 0,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
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
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
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
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
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
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
            "detection_event_latency_cycles": 9,
            "detection_event_cycles_per_round": 2,
        },
    }

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.detection_event_latency_cycles == 9
    assert settings.detection_event_cycles_per_round == 2


def test_the_engine_card_reads_the_per_job_and_per_round_stage_cycles():
    """Each stage is priced once a job and once a round.

    Helios loads a header byte per job and a round's bytes per round,
    and writes three header bytes per job and a round's correction bytes
    per round (2301.08419v2, control_node_single_FPGA.v lines 152-182
    and 217-224 at 2dda998).
    """
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 2,
            "fetch_cycles_per_job": 1,
            "release_cycles_per_job": 3,
            "release_cycles_per_round": 4,
        },
    }

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.fetch_cycles_per_job == 1
    assert settings.release_cycles_per_round == 4


def test_a_negative_stage_cycle_count_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": -1,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    }

    with pytest.raises(ValueError, match="fetch_cycles_per_job"):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )

    section["engine"]["fetch_cycles_per_job"] = 0
    section["engine"]["release_cycles_per_round"] = -2
    with pytest.raises(ValueError, match="release_cycles_per_round"):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_an_engine_card_without_a_stage_key_is_refused_by_name():
    """Every stage key is written out, and a missing one reads as a sentence."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    }

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )
    assert str(refusal.value) == (
        "weak_decoder.engine needs fetch_cycles_per_job, the fetch "
        "stage's cost once a job in cycles of its clock"
    )

    del section["engine"]["release_cycles_per_job"]
    section["engine"]["fetch_cycles_per_job"] = 0
    with pytest.raises(ValueError) as older:
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "strong_decoder"
        )
    assert str(older.value) == (
        "strong_decoder.engine needs release_cycles_per_job, the release "
        "stage's cost once a job in cycles of its clock"
    )


def test_the_weight_step_is_read_and_absent_is_the_shipped_one():
    """The growth resolution is a yaml key of the union_find row."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "union_find",
        "units": 1,
        "unit_memory_rounds": None,
        "weight_step": 0.5,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    }

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )
    without = dict(section)
    del without["weight_step"]
    plain = decoder_settings.DecoderSettings.from_yaml(
        without, clocks, "weak_decoder"
    )

    assert settings.weight_step == 0.5
    assert plain.weight_step == evidence_records.DEFAULT_WEIGHT_STEP


def test_a_weight_step_that_is_not_positive_is_refused_with_a_sentence():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "union_find",
        "units": 1,
        "unit_memory_rounds": None,
        "weight_step": 0.0,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    }

    with pytest.raises(ValueError, match="finite and positive"):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_the_cycle_count_block_is_read_and_absent_is_none():
    clocks = config.ClockSettings({"decoder": 250.0, "helios": 100.0})
    section = {
        "kind": "union_find",
        "units": 1,
        "unit_memory_rounds": None,
        "cycle_count": {"clock": "helios", "setup_cycles": 11},
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    }

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )
    without = dict(section)
    del without["cycle_count"]
    plain = decoder_settings.DecoderSettings.from_yaml(
        without, clocks, "weak_decoder"
    )

    assert settings.cycle_count.setup_cycles == 11
    assert settings.cycle_count.clock == clocks.clock("helios")
    assert plain.cycle_count is None
