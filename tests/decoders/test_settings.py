"""The decoder settings at the yaml boundary."""

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records


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
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_an_unknown_key_under_unit_memory_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None, "rounds": 12})
    with pytest.raises(
        ValueError, match=r"weak_decoder.unit_memory does not know \['rounds'\]"
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_null_unit_memory_capacity_is_an_unbounded_memory():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})

    settings = decoder_settings.DecoderPoolSettings.from_yaml(
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
        ValueError, match="result_blocks_unit 'yes' is not a flag"
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


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

    settings = decoder_settings.DecoderPoolSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.engine.fetch_cycles_per_job == 1
    assert settings.engine.release_cycles_per_round == 4


@pytest.mark.parametrize("section_name", ["weak_decoder", "strong_decoder"])
@pytest.mark.parametrize(
    "key",
    [
        "fetch_cycles_per_round",
        "fetch_cycles_per_job",
        "release_cycles_per_job",
        "release_cycles_per_round",
    ],
)
def test_an_engine_cycle_count_refusal_names_its_tier_and_card(
    section_name, key
):
    """Weak and strong share the keys, so the sentence says which tier."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["engine"][key] = -1

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, section_name
        )
    assert str(refusal.value) == (
        f"{section_name}.engine.{key} must not be negative: cycles must be "
        "nonnegative"
    )


@pytest.mark.parametrize("section_name", ["weak_decoder", "strong_decoder"])
@pytest.mark.parametrize(
    ("key", "value", "sentence"),
    [
        ("delay_cycles", -1, "must not be negative"),
        ("setup_cycles", True, "must be a nonnegative integer"),
        ("setup_cycles_per_vertex", 1.5, "must be a nonnegative integer"),
        ("setup_cycles_per_edge", -2, "must not be negative"),
        ("cycles_per_edge", -0.5, "must be a finite nonnegative number"),
        ("cycles_per_edge", "4", "must be a number"),
    ],
)
def test_a_cycle_count_refusal_names_its_tier(
    section_name, key, value, sentence
):
    """Both tiers take the union_find row, so the sentence says which."""
    clocks = config.ClockSettings({"decoder": 250.0, "helios": 100.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find"
    section["cycle_count"] = {"clock": "helios", key: value}

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, section_name
        )
    message = str(refusal.value)
    assert message.startswith(f"{section_name}.cycle_count.{key} {sentence}")


@pytest.mark.parametrize("section_name", ["weak_decoder", "strong_decoder"])
def test_a_cycle_count_key_it_does_not_read_is_refused_naming_its_tier(
    section_name,
):
    clocks = config.ClockSettings({"decoder": 250.0, "helios": 100.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find"
    section["cycle_count"] = {"clock": "helios", "cycles_per_edeg": 4}

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, section_name
        )
    message = str(refusal.value)
    assert message.startswith(
        f"{section_name}.cycle_count has no key ['cycles_per_edeg']"
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
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )
    assert str(refusal.value) == (
        "weak_decoder.engine needs fetch_cycles_per_job, the fetch "
        "stage's cost once a job in cycles of its clock"
    )

    del section["engine"]["release_cycles_per_job"]
    section["engine"]["fetch_cycles_per_job"] = 0
    with pytest.raises(ValueError) as older:
        decoder_settings.DecoderPoolSettings.from_yaml(
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

    settings = decoder_settings.DecoderPoolSettings.from_yaml(
        section, clocks, "weak_decoder"
    )
    without = dict(section)
    del without["weight_step"]
    plain = decoder_settings.DecoderPoolSettings.from_yaml(
        without, clocks, "weak_decoder"
    )

    assert settings.algorithm.weight_step == 0.5
    assert plain.algorithm.weight_step == (evidence_records.DEFAULT_WEIGHT_STEP)


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

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )
    message = str(refusal.value)
    assert message == "weak_decoder.weight_step must be finite and positive"


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

    settings = decoder_settings.DecoderPoolSettings.from_yaml(
        section, clocks, "weak_decoder"
    )
    without = dict(section)
    del without["cycle_count"]
    plain = decoder_settings.DecoderPoolSettings.from_yaml(
        without, clocks, "weak_decoder"
    )

    cycle_count = settings.algorithm.cycle_count
    assert cycle_count.setup_cycles == 11
    assert cycle_count.clock == clocks.clock("helios")
    assert plain.algorithm.cycle_count is None


@pytest.mark.parametrize("key", ["kind", "units", "unit_memory", "engine"])
def test_a_tier_section_without_a_required_key_is_refused_by_name(key):
    """A sweep that leaves a key out reads a sentence, not a KeyError."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    del section[key]

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
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
        decoder_settings.DecoderPoolSettings.from_yaml(
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
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_decoder_kind_off_the_table_is_refused_naming_the_rows():
    """A typo is refused by the one sentence every table refuses with."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = "my_decodr"
    rows = sorted(decoder_settings.DECODERS)
    sentence = (
        f"weak_decoder.kind 'my_decodr' is not a row of its table; the rows "
        f"are {rows}"
    )

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )

    assert str(refusal.value) == sentence


@pytest.mark.parametrize("kind", [-1, True, float("nan"), float("inf"), [1]])
def test_a_kind_that_is_neither_a_row_nor_a_latency_is_refused_by_name(kind):
    """A negative, a flag, an infinity and a list are no core latency."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = kind

    with pytest.raises(
        ValueError, match="weak_decoder.kind .* is neither a row nor a latency"
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_the_cluster_gap_is_not_a_decoder_row():
    """The cluster gap is a confidence row, not a decoder row.

    escalation.confidence names it (confidence/signals.py) and it reads
    the growth of whatever weak decoder the tier names.
    """
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find_cluster_gap"

    with pytest.raises(
        ValueError,
        match="weak_decoder.kind 'union_find_cluster_gap' is not a row",
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


@pytest.mark.parametrize("kind", [0, 0.028, 10])
def test_a_finite_nonnegative_kind_is_a_preset_core_latency(kind):
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = kind

    settings = decoder_settings.DecoderPoolSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.algorithm.preset_latency_microseconds == kind


@pytest.mark.parametrize("units", [0, -1, True, 1.5, "two"])
def test_a_unit_count_that_is_not_a_whole_count_is_refused_by_name(units):
    """None of these is a number of engines a chip can hold."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["units"] = units

    with pytest.raises(
        ValueError, match="_decoder.units\\) must be a whole number"
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
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
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_manager_clock_the_clocks_do_not_have_is_refused_at_no_cost():
    """A named domain is resolved even when dispatch charges nothing."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = {"clock": "nowhere", "dispatch_cycles": 0}

    with pytest.raises(ValueError, match="clock 'nowhere' is not a clocks"):
        decoder_settings.DecoderManagerSettings.from_yaml(section, clocks)


@pytest.mark.parametrize("section_name", ["weak_decoder", "strong_decoder"])
@pytest.mark.parametrize("weight_step", [True, "0.1", None])
def test_a_weight_step_that_is_not_a_number_is_refused_naming_its_tier(
    section_name, weight_step
):
    """A ValueError, so the experiments layer names the file it is in."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find"
    section["weight_step"] = weight_step

    with pytest.raises(ValueError) as refusal:
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, section_name
        )
    message = str(refusal.value)
    assert message == f"{section_name}.weight_step must be a real number"


def test_a_cycle_count_block_without_a_clock_is_refused_by_name():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section["kind"] = "union_find"
    section["cycle_count"] = {"delay_cycles": 3}

    with pytest.raises(ValueError, match="weak_decoder.cycle_count needs"):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


@pytest.mark.parametrize("key", ["unit_memory", "engine"])
def test_a_nested_block_written_as_one_value_is_refused_by_name(key):
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None})
    section[key] = 4096

    with pytest.raises(
        ValueError, match=f"weak_decoder.{key} holds 4096; it is a mapping"
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_unit_memory_word_is_read_from_its_block():
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None, "word_bits": 8})

    settings = decoder_settings.DecoderPoolSettings.from_yaml(
        section, clocks, "weak_decoder"
    )

    assert settings.unit_memory.word_bits == 8


@pytest.mark.parametrize("word_bits", [0, True, 8.0])
def test_a_unit_memory_word_that_is_not_whole_bits_is_refused_by_name(
    word_bits,
):
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None, "word_bits": word_bits})
    with pytest.raises(
        ValueError,
        match="weak_decoder.unit_memory.word_bits must be a whole number",
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_a_unit_memory_word_under_an_in_place_input_is_refused():
    """The store prices that read; a width here would price it twice."""
    clocks = config.ClockSettings({"decoder": 250.0})
    section = _tier_section({"bits": None, "word_bits": 8})
    section["input"] = "in_place"
    with pytest.raises(
        ValueError, match="reads the rounds where the store keeps them"
    ):
        decoder_settings.DecoderPoolSettings.from_yaml(
            section, clocks, "weak_decoder"
        )


def test_an_engine_stage_of_negative_cycles_is_refused_by_its_record():
    sentence = "engine.fetch_cycles_per_round must not be negative"

    with pytest.raises(ValueError, match=sentence):
        decoder_settings.EngineSettings(fetch_cycles_per_round=-1)


def test_a_unit_memory_of_no_bits_is_refused_by_its_record():
    sentence = "unit_memory.bits must be at least one bit"

    with pytest.raises(ValueError, match=sentence):
        decoder_settings.UnitMemorySettings(bits=0)


def test_toshios_pool_charges_tau_dec_for_every_round_of_the_job():
    """T_dec(r) = tau_dec r, 2510.25222 lines 968-971: 0.4 us x 10 rounds."""
    clock = config.Clock(period_ticks=4_000)
    pool = decoder_settings.toshio_decoder_pool(0.4, clock, solves_per_window=1)
    engine = pool.engine
    fetch = staged_decoder.MemoryFetchStage(
        "fetch",
        cycles_per_job=engine.fetch_cycles_per_job,
        cycles_per_round=engine.fetch_cycles_per_round,
    )
    job = decoding_records.DecodeJob(
        operation_id=0, window_id=0, round_count=10
    )

    fetch_ticks = fetch.cycles_for(job) * clock.period_ticks
    assert fetch_ticks == config.microseconds_to_ticks(4.0)
    assert engine.release_cycles_per_job == 0
    assert engine.release_cycles_per_round == 0
    assert pool.algorithm.preset_latency_microseconds == 0.0


def test_a_toshio_decode_time_off_the_clock_is_refused():
    """1 ns a round is a quarter of a 4 ns cycle."""
    clock = config.Clock(period_ticks=4_000)
    with pytest.raises(ValueError, match="not a whole number of cycles"):
        decoder_settings.toshio_decoder_pool(0.001, clock, solves_per_window=1)


def test_a_window_solved_twice_pays_half_of_toshios_time_on_each_solve():
    """One unit, one time a round: 0.4 us over two solves is 50 cycles each."""
    clock = config.Clock(period_ticks=4_000)
    pool = decoder_settings.toshio_decoder_pool(0.4, clock, solves_per_window=2)

    assert pool.engine.fetch_cycles_per_round == 50
