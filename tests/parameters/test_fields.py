"""A record's fields as parameters, the unit in the value (fields.py).

The referent is the record itself: a parameter converts a written value
to exactly the number the field holds, through decsim's own rounding
(test_units.py), and writes a field's value as text that converts back
to it (test_tree.py reads every path of two machines back).
"""

import fractions

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.links.framings as framings
import decsim.links.settings as link_settings
import decsim.parameters.fields as fields
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.records.program as program_records

FRIDGE = config.Clock.from_megahertz(250.0)


def test_a_unit_word_leaves_the_name_and_is_written_with_the_value():
    parameters = fields.parameters_of(decoder_settings.EngineSettings)

    fetch = parameters["fetch_per_round"]

    assert fetch.field == "fetch_cycles_per_round"
    assert fetch.kind == "cycles"
    assert fetch.written(3) == "3 cycles"


def test_a_name_keeps_its_unit_word_where_another_field_has_the_rest():
    parameters = fields.parameters_of(escalation_settings.SwitchingSettings)

    assert "threshold" in parameters
    assert parameters["threshold_cycles"].field == "threshold_cycles"
    assert parameters["switch"].field == "switch_cycles"


def test_a_count_of_logical_cycles_stays_a_plain_count():
    settings = magic_state_factories.MultiLevelDistillationFactory.Settings
    parameters = fields.parameters_of(settings)

    preparation = parameters["preparation_logical_cycles"]

    assert preparation.kind == fields.PLAIN
    assert preparation.value_of(3, None, "preparation_logical_cycles") == 3


def test_a_time_written_for_cycles_counts_on_the_parts_clock():
    parameters = fields.parameters_of(decoder_settings.EngineSettings)
    fetch = parameters["fetch_per_round"]

    cycles = fetch.value_of("8ns", FRIDGE, "engine.fetch_per_round")

    assert cycles == FRIDGE.cycles_for(8000)


def test_a_time_for_cycles_with_no_clock_anywhere_is_refused():
    parameters = fields.parameters_of(decoder_settings.EngineSettings)
    fetch = parameters["fetch_per_round"]

    with pytest.raises(fields.ParameterError, match="no clock|has a clock"):
        fetch.value_of("8ns", None, "engine.fetch_per_round")


def test_a_size_in_bytes_is_a_whole_number_of_bytes():
    parameters = fields.parameters_of(framings.RoceV2.Settings)
    mtu = parameters["path_mtu"]

    whole = mtu.value_of("4096 bytes", None, "path_mtu")

    assert whole == 4096
    with pytest.raises(fields.ParameterError, match="whole number of bytes"):
        mtu.value_of("12 bits", None, "path_mtu")


def test_a_rate_is_kept_exact_in_bits_per_microsecond():
    parameters = fields.parameters_of(link_settings.CapacitySettings)
    rate = parameters["input"]

    value = rate.value_of("100Gbps", None, "capacity.input")

    assert value == fractions.Fraction(100_000)
    assert rate.written(value) == "100000 bits/us"


def test_a_clock_is_written_as_its_frequency():
    parameters = fields.parameters_of(decoder_settings.EngineSettings)
    clock = parameters["clock"]

    value = clock.value_of("250MHz", None, "engine.clock")

    assert value == FRIDGE
    assert clock.written(value) == "250MHz"


def test_a_time_is_written_in_the_largest_unit_it_fills():
    parameters = fields.parameters_of(link_settings.PathSettings)
    setup = parameters["setup"]

    assert setup.written(4000) == "4ns"
    assert setup.written(1_100_000) == "1.1us"
    assert setup.written(0) == "0us"


def test_a_flag_is_not_a_count_and_a_count_is_a_number():
    parameters = fields.parameters_of(decoder_settings.DecoderPoolSettings)
    unit_count = parameters["unit_count"]
    copies = parameters["copies_input"]

    with pytest.raises(fields.ParameterError, match="is not of type int"):
        unit_count.value_of(True, None, "unit_count")
    assert copies.value_of(False, None, "copies_input") is False


def test_a_stim_circuit_is_shown_and_set_in_python_only():
    parameters = fields.parameters_of(program_records.Operation)
    circuit = parameters["circuit"]

    assert circuit.kind == fields.OPAQUE
    with pytest.raises(fields.ParameterError, match="sets in Python"):
        circuit.value_of("H 0", None, "operations[0].circuit")


def test_a_field_with_no_default_is_required():
    parameters = fields.parameters_of(framings.RoceV2.Settings)

    assert parameters["path_mtu"].default is fields.REQUIRED


def test_a_rate_no_decimal_holds_is_written_as_a_fraction_read_back_exactly():
    parameters = fields.parameters_of(link_settings.CapacitySettings)
    parameter = parameters["input"]
    rate = fractions.Fraction(1, 3)

    written = parameter.written(rate)

    assert written == "1/3 bits/us"
    assert parameter.value_of(written, None, "input") == rate
