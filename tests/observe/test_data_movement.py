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
import decsim.records.transfers as transfer_records


def test_a_structure_no_row_and_no_prefix_names_is_unclassified():
    """Unclassified, not guessed, so the grouped rows still sum."""
    unknown = data_movement.memory_class_of_structure("a new cache")

    assert unknown is data_movement.MemoryClass.UNCLASSIFIED


def test_a_link_path_no_row_names_is_unclassified():
    unknown = data_movement.memory_class_of_link_path("controller_to_moon")

    assert unknown is data_movement.MemoryClass.UNCLASSIFIED


def test_a_move_counts_the_header_and_known_payload_bits_the_ledger_charges():
    """The link ledger's known payload plus header bits, per move.

    ns-3's device adds the header before it times the packet
    (src/point-to-point/model/point-to-point-net-device.cc lines 528 and
    243); a transfer of unknown size still carries its header.
    """
    movement = data_movement.DataMovement()
    framed = _Delivered(payload_bits=44, header_bits=16)
    unknown_size = _Delivered(payload_bits=None, header_bits=16)

    movement.transfer_delivered(framed)
    movement.transfer_delivered(unknown_size)
    counted = movement.json_value()

    assert counted["moves"] == 2
    assert counted["move_bits"] == 44 + 16 + 16


def test_a_move_counts_the_rounds_its_attribution_names():
    """A window's range runs past the stream; its round keys do not.

    The last lookahead window of a stream keeps the regular stride, so
    its range reads 4 to 9 of a six-round stream while the move carries
    rounds 4 to 6 (windows/round_retention.py, read_keys_for_bounds).
    """
    movement = data_movement.DataMovement()
    terminal = _Delivered(payload_bits=24, header_bits=0)
    terminal.attribution = transfer_records.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=1,
        first_round=4,
        last_round=9,
        round_keys=((1, 4), (1, 5), (1, 6)),
    )

    movement.transfer_delivered(terminal)
    counted = movement.json_value()

    assert counted["moved_rounds"] == 3


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


class _Transfer:
    """The two sizes a move reads off a transfer."""

    def __init__(self, payload_bits, header_bits: int) -> None:
        self.payload_bits = payload_bits
        self.header_bits = header_bits


class _Attribution:
    """A move that carried one round."""

    round_keys = ((1, 1),)


class _Delivered:
    """The three fields a move reads off a delivered transfer record."""

    def __init__(self, payload_bits, header_bits: int) -> None:
        self.transfer = _Transfer(payload_bits, header_bits)
        self.attribution = _Attribution()
        self.path = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER


def test_the_ledger_keeps_the_most_one_history_held_at_each_seat():
    ledger = data_movement.DataMovement()

    ledger.formation_state_held("weak_decoder", 1, 8)
    ledger.formation_state_held("weak_decoder", 1, 16)
    ledger.formation_state_held("weak_decoder", 2, 8)
    ledger.formation_state_held("controller", 1, 4)

    counted = ledger.json_value()
    assert counted["formation_state_bits_by_seat"] == {
        "controller": 4,
        "weak_decoder": 16,
    }
