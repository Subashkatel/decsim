"""Window positions whose front is let go keep their meaning."""

import pytest

import decsim.windows.window_positions as window_positions


def _positions(values) -> window_positions.WindowPositions:
    positions = window_positions.WindowPositions()
    for value in values:
        positions.append(value)
    return positions


def test_a_kept_position_reads_its_value_after_the_front_goes():
    positions = _positions(["a", "b", "c", "d"])

    positions.let_go_before(2)

    assert len(positions) == 4
    assert positions[2] == "c"
    assert list(positions) == ["c", "d"]
    assert positions.index("d") == 3


def test_a_let_go_position_raises():
    positions = _positions(["a", "b", "c"])
    positions.let_go_before(1)

    with pytest.raises(IndexError, match="let go"):
        positions[0]


def test_letting_go_past_the_end_starts_the_next_value_there():
    positions = _positions(["a"])

    positions.let_go_before(3)
    positions.append("d")

    assert len(positions) == 4
    assert positions[3] == "d"
    positions[3] = "e"
    assert list(positions) == ["e"]
