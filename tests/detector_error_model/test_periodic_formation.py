"""A stream's formation table moved in time (periodic_formation.py).

The referent is the table read off the whole assembled circuit
(detector_formation.build_formation_table), live and finished, for every
length up to three templates: every recipe, observable, packet width and
read a seat makes must be the same.
"""

import pytest

import decsim.detector_error_model.periodic_formation as periodic_formation
import tests.qpu.memory_programs as memory_programs

# the reads a seat makes of a table, each by round
ROUND_READS = (
    "rounds_reading",
    "earlier_rounds_read",
    "observable_slots_of_round",
)
# the live table's reach, and none on a finished one
LIVE_REACH = {False: 1, True: None}


def _reads_by_round(table) -> list:
    """Every read a seat makes, round by round, as (read, round, answer)."""
    after_last_round = table.round_count + 1
    answers = []
    for read in ROUND_READS:
        answer_of = getattr(table, read)
        for round_index in range(1, after_last_round):
            answer = answer_of(round_index)
            answers.append((read, round_index, answer))
    return answers


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("is_final", [False, True])
def test_every_length_reads_as_the_whole_circuits_table(distance, is_final):
    program = memory_programs.memory_program(distance)
    template = periodic_formation.StreamTemplate(program)
    live_reach = LIVE_REACH[is_final]
    after_longest = 3 * template.round_count + 1

    for round_count in range(1, after_longest):
        table = template.table(
            round_count, is_final=is_final, live_reach=live_reach
        )
        expected = periodic_formation.assembled_table(
            program, round_count, is_final, live_reach
        )
        assert table.detectors == expected.detectors
        assert table.observables == expected.observables
        assert dict(table.packet_width_by_round) == (
            expected.packet_width_by_round
        )
        assert table.readout_slot_start == expected.readout_slot_start
        assert _reads_by_round(table) == _reads_by_round(expected)


def test_a_longer_live_table_extends_a_shorter_and_a_changed_one_is_refused():
    program = memory_programs.memory_program(3)
    template = periodic_formation.StreamTemplate(program)
    rounds = template.round_count + 4
    more_rounds = rounds + 3
    shorter = template.table(rounds, is_final=False, live_reach=1)
    longer = template.table(more_rounds, is_final=False, live_reach=1)
    other_program = memory_programs.memory_program(5)
    other = periodic_formation.StreamTemplate(other_program)
    changed = other.table(more_rounds, is_final=False, live_reach=1)

    longer.check_extends(shorter)
    with pytest.raises(RuntimeError, match="changes an existing"):
        changed.check_extends(shorter)
