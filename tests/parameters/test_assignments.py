"""Assignments written as text (parameters/assignments.py).

The grammar is the one the --set flag and DECSIM_SET share; the order is
ns-3's precedence (manual, Attributes): the environment first, then the
command line, so a flag wins.
"""

import pytest

import decsim.parameters.assignments as assignments
import decsim.parameters.fields as fields
import decsim.parameters.tree as tree

VALUES = {
    "3": 3,
    "-1": -1,
    "0.25": 0.25,
    "1e-3": 0.001,
    "true": True,
    "false": False,
    "null": None,
    "union_find": "union_find",
    "10us": "10us",
    "'3'": "3",
    '"none"': "none",
    "[a, 3, true]": ["a", 3, True],
    "[]": [],
}


@pytest.mark.parametrize("text", sorted(VALUES))
def test_a_value_reads_as_a_number_a_flag_a_word_or_a_list(text):
    value = assignments.value_of(text)

    assert value == VALUES[text]
    assert type(value) is type(VALUES[text])


def test_a_part_sets_its_type_first_then_its_parameters():
    capacity = "links.qpu_to_controller.channel.capacity"
    text = f"{capacity}={{input: 1Gbps, type: Capacity}}"

    parsed = assignments.parse(text, assignments.FLAG)

    patterns = [assignment.pattern for assignment in parsed]
    assert patterns == [
        "links.qpu_to_controller.channel.capacity.type",
        "links.qpu_to_controller.channel.capacity.input",
    ]
    assert parsed[1].written == "1Gbps"


def test_the_flags_come_after_the_environment_so_a_flag_wins(monkeypatch):
    monkeypatch.setenv(assignments.ENVIRONMENT_VARIABLE, "a.b=1; a.c=2")

    every_assignment = assignments.every_assignment(["a.b=3"])

    expected = [
        tree.Assignment("a.b", 1, assignments.ENVIRONMENT_VARIABLE),
        tree.Assignment("a.c", 2, assignments.ENVIRONMENT_VARIABLE),
        tree.Assignment("a.b", 3, assignments.FLAG),
    ]
    assert every_assignment == expected


def test_an_unset_environment_assigns_nothing(monkeypatch):
    monkeypatch.delenv(assignments.ENVIRONMENT_VARIABLE, raising=False)

    assert assignments.from_environment() == []


@pytest.mark.parametrize("text", ["weak_decoder.unit_count", "=3", "a.b="])
def test_a_text_that_is_not_path_equals_value_is_refused(text):
    with pytest.raises(
        fields.ParameterError, match="weak_decoder.unit_count=3"
    ):
        assignments.parse(text, assignments.FLAG)
