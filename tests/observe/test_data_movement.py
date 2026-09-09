"""How often a run copied bits, referenced them and moved them.

The vocabulary is gem5's: a copy duplicates bits into a structure the
receiver owns (mem/cache/cache_blk.hh 97-104), a reference is a handle
to bits that stay where they are (mem/packet.hh 1163-1171), a move
crosses a link (dev/dma_device.cc 194-213). The grouping by memory class
is the classical sources' point that the class is the cost and not the
count (Horowitz ISSCC 2014 lines 232-247; Dally CACM 2020 lines
231-234), so a copy into a register and a copy across a cryostat link
are never summed. The tables are a report-side grouping and no component
reads them, which is why an unnamed structure or path is unclassified
rather than guessed: the grouped rows still sum to the run's total.
"""

import decsim.observe.data_movement as data_movement


def test_a_named_structure_takes_the_class_its_row_gives_it():
    on_chip = data_movement.memory_class_of_structure("controller intake")

    assert on_chip is data_movement.MemoryClass.ON_CHIP


def test_a_store_is_on_board_and_a_unit_is_on_chip_by_their_prefix():
    store = data_movement.memory_class_of_structure("Buffer 0")
    unit = data_movement.memory_class_of_structure("unit 0")

    assert store is data_movement.MemoryClass.ON_BOARD
    assert unit is data_movement.MemoryClass.ON_CHIP


def test_a_structure_no_row_and_no_prefix_names_is_unclassified():
    """Unclassified, not guessed, so the grouped rows still sum."""
    unknown = data_movement.memory_class_of_structure("a new cache")

    assert unknown is data_movement.MemoryClass.UNCLASSIFIED


def test_the_readout_leaves_the_cryostat_and_the_seam_stays_on_chip():
    readout = data_movement.memory_class_of_link_path("qpu_to_controller")
    seam = data_movement.memory_class_of_link_path("decoder_to_decoder")

    assert readout is data_movement.MemoryClass.OFF_BOARD
    assert seam is data_movement.MemoryClass.ON_CHIP


def test_a_link_path_no_row_names_is_unclassified():
    unknown = data_movement.memory_class_of_link_path("controller_to_moon")

    assert unknown is data_movement.MemoryClass.UNCLASSIFIED


def test_a_run_that_moved_nothing_counts_nothing():
    movement = data_movement.DataMovement()

    counted = movement.json_value()

    assert movement.rounds == 0
    assert movement.copies_per_round == 0.0
    assert counted["copies"] == 0


def test_one_copy_of_one_round_is_one_event_and_one_round():
    movement = data_movement.DataMovement()

    movement.copy_made(("op", 1), 249, "controller intake", "Buffer 0")
    counted = movement.json_value()

    assert counted["copies"] == 1
    assert counted["copied_rounds"] == 1
    assert counted["copy_bits"] == 249


def test_one_copy_of_a_whole_job_input_is_one_event_and_its_rounds():
    """data_path.md's hop table counts per round, the events price the hop."""
    movement = data_movement.DataMovement()
    six_round_job = _JobKey(6)

    movement.copy_made(six_round_job, 1494, "Buffer 0", "unit 0")
    counted = movement.json_value()

    assert counted["copies"] == 1
    assert counted["copied_rounds"] == 6


def test_copies_are_grouped_by_the_class_they_landed_in():
    movement = data_movement.DataMovement()

    movement.copy_made(("op", 1), 8, "controller intake", "Buffer 0")
    movement.copy_made(("op", 1), 8, "Buffer 0", "unit 0")
    counted = movement.json_value()
    rows = counted["copies_by_memory_class"]

    assert rows["on_chip"]["events"] == 1
    assert rows["on_board"]["events"] == 1


def test_the_classes_are_listed_cheapest_first():
    movement = data_movement.DataMovement()

    movement.copy_made(("op", 1), 8, "Buffer 0", "a new cache")
    movement.copy_made(("op", 1), 8, "Buffer 0", "unit 0")
    movement.copy_made(("op", 1), 8, "controller intake", "Buffer 0")
    counted = movement.json_value()
    rows = counted["copies_by_memory_class"]
    listed = list(rows)

    assert listed == ["on_chip", "on_board", "unclassified"]


def test_a_reference_counts_its_rounds_and_copies_no_bits():
    movement = data_movement.DataMovement()

    movement.hold_registered("a token", [("op", 1), ("op", 2)])
    counted = movement.json_value()

    assert counted["references"] == 1
    assert counted["referenced_rounds"] == 2
    assert counted["copies"] == 0
    assert counted["copy_bits"] == 0


def test_handing_a_live_reference_on_copies_nothing():
    movement = data_movement.DataMovement()

    movement.hold_registered("first", [("op", 1)])
    movement.hold_transferred("first", "second")
    movement.hold_released("second")
    counted = movement.json_value()

    assert counted["hold_events"] == 3
    assert counted["copies"] == 0


def test_copies_per_round_has_the_rounds_the_qpu_emitted_as_denominator():
    movement = data_movement.DataMovement()

    first = _Readout(1, 1)
    second = _Readout(1, 2)

    movement.round_emitted(first)
    movement.round_emitted(second)
    movement.copy_made(("op", 1), 8, "controller intake", "Buffer 0")

    assert movement.rounds == 2
    assert movement.copies_per_round == 0.5


def test_the_same_round_emitted_twice_is_one_round():
    movement = data_movement.DataMovement()

    once = _Readout(1, 1)
    again = _Readout(1, 1)

    movement.round_emitted(once)
    movement.round_emitted(again)

    assert movement.rounds == 1


class _Readout:
    """The two fields the round counter reads off a readout."""

    def __init__(self, operation_id: int, round_index: int) -> None:
        self.operation_id = operation_id
        self.round_index = round_index


class _DecoderInput:
    def __init__(self, round_count: int) -> None:
        self.rounds = tuple(range(round_count))


class _JobKey:
    """A copy key that names a whole decode input rather than one round."""

    def __init__(self, round_count: int) -> None:
        self.decoder_input = _DecoderInput(round_count)
