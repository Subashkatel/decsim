"""The observation section is a boundary: every key checked once, loudly.

The section is yaml a user writes by hand, so a stale key, the wrong
word and a shot list that is not a list of shot numbers are refused with
a sentence (STYLE.md rule 4). The words are read through yaml itself
because yaml 1.1 turns a bare `on` into True and a bare `off` into
False, which is exactly what the refusals must catch.
"""

import pytest
import yaml

import decsim.observe.settings as observe_settings
import tests.observe.gate_point as gate_point

STUDY_KNOBS = (
    "record_switching_windows",
    "round_store_occupancy",
    "backlog_trace",
    "decoder_utilization",
    "decoder_memory_occupancy",
    "data_movement",
)


def _section(text: str):
    """The observation section of a yaml a user would write."""
    document = yaml.safe_load(text)
    return document["observation"]


def test_a_stale_observation_key_is_refused_by_name():
    """trace_io was the narrator's key two renames ago."""
    section = _section("observation:\n  trace_io: true\n")

    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings.from_yaml(section)

    assert "observation.trace_io is not an observation key" in str(
        refusal.value
    )
    assert "log_component_io" in str(refusal.value)


def test_the_narrators_word_in_the_trace_key_is_refused():
    """Every config saved before the rename carries `trace: print`."""
    section = _section("observation:\n  trace: print\n")

    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings.from_yaml(section)

    message = str(refusal.value)
    assert "observation.trace no longer names the engine narrator" in message
    assert "write `log: print`" in message


def test_a_bare_on_for_the_trace_is_refused():
    """In yaml 1.1 a bare `on` is True, and True is not a path."""
    section = _section("observation:\n  trace: on\n")

    assert section["trace"] is True
    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings.from_yaml(section)

    assert "observation.trace must be off, chrome, or a path" in str(
        refusal.value
    )


def test_a_bare_off_for_the_trace_and_the_log_is_the_word_off():
    """In yaml 1.1 a bare `off` is False, and both keys mean off."""
    section = _section("observation:\n  trace: off\n  log: off\n")

    settings = observe_settings.ObservationSettings.from_yaml(section)

    assert settings.trace == "off"
    assert settings.log == "off"
    assert settings.writes_trace is False


def test_a_single_shot_number_is_not_a_shot_list():
    """`trace_shots: 0` is the shape a user reaches for first."""
    section = _section("observation:\n  trace_shots: 0\n")

    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings.from_yaml(section)

    assert "observation.trace_shots must be a list of shot numbers" in str(
        refusal.value
    )


def test_a_negative_shot_number_is_refused():
    """Shots are counted from 0, so -1 names nothing."""
    section = _section("observation:\n  trace_shots: [0, -1]\n")

    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings.from_yaml(section)

    assert "non-negative whole numbers" in str(refusal.value)


def test_a_study_knob_that_is_not_true_or_false_is_refused():
    """The knobs are on or off; a word is a user's mistake."""
    section = _section("observation:\n  data_movement: yes please\n")

    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings.from_yaml(section)

    assert "observation.data_movement must be true or false" in str(
        refusal.value
    )


def test_every_study_knob_is_read_from_the_section():
    """The six knobs the Machine builds listeners for are yaml keys."""
    text = "observation:\n"
    for knob in STUDY_KNOBS:
        text += f"  {knob}: true\n"
    section = _section(text)

    settings = observe_settings.ObservationSettings.from_yaml(section)

    assert settings.record_switching_windows is True
    assert settings.round_store_occupancy is True
    assert settings.backlog_trace is True
    assert settings.decoder_utilization is True
    assert settings.decoder_memory_occupancy is True
    assert settings.data_movement is True


@pytest.mark.parametrize("knob", STUDY_KNOBS)
def test_a_study_knob_moves_no_number_the_gate_hashes(knob):
    """A listener only listens: asking for one changes no result."""
    plain_machine, plain_result = gate_point.run()
    change = {knob: True}
    observed_machine, observed_result = gate_point.run(**change)

    plain = gate_point.captured_fields(plain_machine, plain_result)
    observed = gate_point.captured_fields(observed_machine, observed_result)
    assert observed == plain
    assert plain["log_sha256"].startswith(gate_point.POINT_LOG_SHA256)
