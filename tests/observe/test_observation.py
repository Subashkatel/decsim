"""The listeners of one run, by name.

The front, the gate and the experiments read a run's numbers from this
record and never from a component, which is the rule that keeps
observation observation: a component fires a source and knows no
listener. The record's own promise is which listeners are always there
and which are None when the section did not ask for them.
"""

import dataclasses

import decsim.observe.observation as observation_module

ALWAYS_PRESENT = (
    "log",
    "windows",
    "results",
    "traffic",
    "flight_recorder",
    "frame_corrections",
    "runtime_stamps",
    "queue_depth",
    "controller_counters",
    "command_events",
    "stages",
    "round_events",
    "referee_audit",
    "sampled_shots",
)
ASKED_FOR_BY_A_STUDY = (
    "trace_writer",
    "data_movement",
    "decode_records",
    "decode_backlog",
    "decoder_utilization",
    "decoder_memory_occupancy",
    "round_store_occupancy",
)


def _fields_by_name() -> dict:
    fields = dataclasses.fields(observation_module.Observation)
    by_name = {}
    for field in fields:
        by_name[field.name] = field
    return by_name


def test_every_listener_of_a_run_is_named_on_the_record():
    by_name = _fields_by_name()
    named = sorted(by_name)
    every_listener = ALWAYS_PRESENT + ASKED_FOR_BY_A_STUDY
    expected = sorted(every_listener)

    assert named == expected


def test_the_always_present_listeners_are_not_optional_on_the_record():
    """A reader of a run's numbers never has to test these for None."""
    by_name = _fields_by_name()

    for name in ALWAYS_PRESENT:
        annotation = str(by_name[name].type)

        assert "Optional" not in annotation, name


def test_a_listener_a_study_asks_for_is_optional_on_the_record():
    by_name = _fields_by_name()

    for name in ASKED_FOR_BY_A_STUDY:
        annotation = str(by_name[name].type)

        assert "Optional" in annotation, name


def test_the_record_is_frozen_so_a_component_cannot_reach_back_into_it():
    parameters = observation_module.Observation.__dataclass_params__

    assert parameters.frozen is True
