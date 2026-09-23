"""The decoder settings at the yaml boundary."""

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.records.decoder_evidence as evidence_records


def _tier_section(unit_memory: dict) -> dict:
    """A weak tier card whose unit memory the test writes."""
    return {
        "kind": "pymatching",
        "units": 1,
        "unit_memory": unit_memory,
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    }


@pytest.mark.parametrize("capacity", [0, -8, True, 8.0])
def test_a_unit_memory_capacity_that_is_not_whole_bits_is_refused_by_name(
    capacity,
):
    """Zero, a negative, a flag and a fraction are none of them a memory."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": capacity})
    with pytest.raises(
        ValueError, match="weak_decoder.unit_memory.bits must be at least"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_an_unknown_key_under_unit_memory_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None, "rounds": 12})
    with pytest.raises(
        ValueError, match=r"weak_decoder.unit_memory does not know \['rounds'\]"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_null_unit_memory_capacity_is_an_unbounded_memory():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.unit_memory.bits is None


def test_a_result_blocking_value_that_is_not_a_boolean_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "pymatching",
        "units": 1,
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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

    assert settings.row_settings.weight_step == 0.5
    assert plain.row_settings.weight_step == (
        evidence_records.DEFAULT_WEIGHT_STEP
    )


def test_a_weight_step_that_is_not_positive_is_refused_with_a_sentence():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {
        "kind": "union_find",
        "units": 1,
        "unit_memory": {"bits": None},
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
        "unit_memory": {"bits": None},
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

    cycle_count = settings.row_settings.cycle_count
    assert cycle_count.setup_cycles == 11
    assert cycle_count.clock == clocks.clock("helios")
    assert plain.row_settings.cycle_count is None


@pytest.mark.parametrize("key", ["kind", "units", "unit_memory", "engine"])
def test_a_tier_section_without_a_required_key_is_refused_by_name(key):
    """A sweep that leaves a key out reads a sentence, not a KeyError."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    del section[key]

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )
    assert str(refusal.value) == (
        f"weak_decoder needs the keys ['{key}']; configs/reference.yaml "
        "holds every key with its unit"
    )


def test_an_engine_card_without_a_clock_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    del section["engine"]["clock"]

    with pytest.raises(
        ValueError, match="weak_decoder.engine needs clock, the domain"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_an_unknown_key_on_the_engine_card_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["engine"]["fetch_cycle_per_round"] = 2

    with pytest.raises(
        ValueError,
        match=r"weak_decoder.engine does not know \['fetch_cycle_per_round'\]",
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


@pytest.mark.parametrize("kind", [-1, True, float("nan"), float("inf"), [1]])
def test_a_kind_that_is_neither_a_row_nor_a_latency_is_refused_by_name(kind):
    """A negative, a flag, an infinity and a list are no core latency."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = kind

    with pytest.raises(
        ValueError, match="weak_decoder.kind .* is neither a row nor a latency"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


@pytest.mark.parametrize("kind", [0, 0.028, 10])
def test_a_finite_nonnegative_kind_is_a_preset_core_latency(kind):
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = kind

    settings = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.kind == kind


@pytest.mark.parametrize("units", [0, -1, True, 1.5, "two"])
def test_a_unit_count_that_is_not_a_whole_count_is_refused_by_name(units):
    """None of these is a number of engines a chip can hold."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["units"] = units

    with pytest.raises(
        ValueError, match="weak_decoder.units must be a whole number"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


@pytest.mark.parametrize("key", ["input", "boundary_fold"])
def test_a_memory_row_that_is_not_a_row_is_refused_at_load_by_name(key):
    """The two copy-or-in-place keys are refused where the yaml enters."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section[key] = "in-place"

    with pytest.raises(
        ValueError, match=f"weak_decoder.{key} 'in-place' is not a row"
    ):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_manager_clock_the_clocks_do_not_have_is_refused_at_no_cost():
    """A named domain is resolved even when dispatch charges nothing."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {"clock": "nowhere", "dispatch_cycles": 0}

    with pytest.raises(ValueError, match="clock 'nowhere' is not a clocks"):
        decoder_settings.DecoderManagerSettings.from_yaml(section, clocks)


@pytest.mark.parametrize("weight_step", [True, "0.1", None])
def test_a_weight_step_that_is_not_a_number_is_refused_with_a_sentence(
    weight_step,
):
    """A ValueError, so the experiments layer names the file it is in."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find"
    section["weight_step"] = weight_step

    with pytest.raises(ValueError, match="weight_step must be a real number"):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_cycle_count_block_without_a_clock_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find"
    section["cycle_count"] = {"delay_cycles": 3}

    with pytest.raises(ValueError, match="cycle_count needs clock"):
        decoder_settings.DecoderSettings.from_yaml(
            section, clocks, "weak_decoder"
        )
