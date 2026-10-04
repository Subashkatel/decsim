"""The observation record refuses a value it cannot honour, by name.

A study knob adds a record and moves no number the gate hashes.
"""

import pytest

import decsim.observe.settings as observe_settings
import tests.observe.gate_point as gate_point

STUDY_KNOBS = (
    "record_switching_windows",
    "backlog_trace",
    "data_movement",
)
# a value written in Python, and the sentence that refuses it
PYTHON_REFUSALS = [
    ({"log": "loud"}, "observation.log must be one of"),
    ({"trace": True}, "observation.trace must be off, chrome, or a path"),
    ({"trace": ""}, "observation.trace is empty, which names no file"),
    ({"trace": "print"}, "observation.trace no longer names the engine"),
    ({"trace_shots": [0, 1]}, "observation.trace_shots must be a list of"),
    ({"trace_shots": (0, -1)}, "non-negative whole numbers, got -1"),
    ({"log_component_io": 1}, "observation.log_component_io must be true"),
    (
        {"record_switching_windows": "yes"},
        "observation.record_switching_windows must be true or false",
    ),
    ({"backlog_trace": None}, "observation.backlog_trace must be true"),
    ({"data_movement": 0}, "observation.data_movement must be true or false"),
]


@pytest.mark.parametrize("values, sentence", PYTHON_REFUSALS)
def test_the_record_refuses_a_value_by_name(values, sentence):
    with pytest.raises(ValueError) as refusal:
        observe_settings.ObservationSettings(**values)

    assert sentence in str(refusal.value)


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
