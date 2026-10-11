"""Constants files: read by attribute, every read recorded, overridden.

A section splats as a mapping; a value no one read is refused, so a
misspelt key fails; `--set constants.<name>` replaces a value when the
file is read, as qecboard's QECBOARD_SET did (notes/qecboard_prototype,
constants.py); a sweep cell sets one for its machine.
"""

import pathlib

import pytest

import decsim.parameters.fields as fields
import decsim.parameters.in_force as in_force
import decsim.parameters.tree as tree
import decsim.topology.constants as constants_module

CONSTANTS = """
union_find:
  units: 2  # two units, an estimate
fiber:
  down: 41 cycles  # 2603.16203 lines 903-904
  up: 40 cycles
"""


def test_yaml_1_2_reads_off_as_a_word_and_5e_4_as_a_number():
    document = constants_module.parsed("a: off\nb: 5e-4\nc: 0123\n")

    assert document == {"a": "off", "b": 0.0005, "c": 123}


def test_a_value_is_read_by_attribute_and_by_key(tmp_path):
    constants = _loaded(tmp_path)

    units = constants.union_find.units
    down = constants["fiber"]["down"]

    assert units == 2
    assert down == "41 cycles"


def test_a_section_splats_into_its_values(tmp_path):
    constants = _loaded(tmp_path)

    fiber = dict(**constants.fiber)

    assert fiber == {"down": "41 cycles", "up": "40 cycles"}


def test_the_values_no_one_read_are_listed_and_refused(tmp_path):
    constants = _loaded(tmp_path)
    held = constants_module.held(constants)
    down = constants.fiber.down

    unread = held.unread()

    assert down == "41 cycles"
    assert unread == ["union_find.units", "fiber.up"]
    assert held.unread(allowed=["union_find", "fiber.up"]) == []
    with pytest.raises(fields.ParameterError, match="union_find.units"):
        held.refuse_the_unread()


def test_a_misspelt_key_is_refused_with_the_near_ones(tmp_path):
    constants = _loaded(tmp_path)

    fiber = constants.fiber

    with pytest.raises(fields.ParameterError, match="Did you mean down"):
        fiber["dwn"]


def test_a_comment_is_its_values_source(tmp_path):
    constants = _loaded(tmp_path)
    held = constants_module.held(constants)

    assert held.sources["fiber.down"] == "2603.16203 lines 903-904"
    assert "fiber.up" not in held.sources


def test_a_value_set_for_a_with_block_is_restored_after_it(tmp_path):
    constants = _loaded(tmp_path)

    with constants_module.overriding(constants, {"fiber.down": "50 cycles"}):
        inside = constants.fiber.down
    after = constants.fiber.down

    assert inside == "50 cycles"
    assert after == "41 cycles"


def test_an_assignment_in_force_to_a_constant_replaces_it_when_read(tmp_path):
    path = _written(tmp_path)
    override = tree.Assignment("constants.fiber.down", "45 cycles", "--set")

    with in_force.in_force([override]):
        constants = constants_module.load(path)

    held = constants_module.held(constants)
    assert constants.fiber.down == "45 cycles"
    assert held.sources["fiber.down"] == "--set constants.fiber.down=45 cycles"


def test_an_assignment_to_a_constant_no_file_has_is_refused(tmp_path):
    path = _written(tmp_path)
    override = tree.Assignment("constants.fiber.dwn", "45 cycles", "--set")

    with pytest.raises(fields.ParameterError, match="reaches no parameter"):
        with in_force.in_force([override]):
            constants_module.load(path)


def _written(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "constants.yaml"
    path.write_text(CONSTANTS)
    return path


def _loaded(tmp_path: pathlib.Path) -> constants_module.Section:
    path = _written(tmp_path)
    return constants_module.load(path)
