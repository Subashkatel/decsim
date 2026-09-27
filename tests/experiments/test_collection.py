"""The collection section read from yaml (decsim/experiments/collection.py).

A block's key replaces the top's, as a file's key replaces its base's
under `extends`. sinter's CollectionOptions.combine takes the smaller of
two instead (sinter/_data/_collection_options.py:68-99), which would let
the top's value cut a block that asks for more.
"""

import pytest

import decsim.experiments.collection as collection
import decsim.experiments.refusal as refusal


def test_no_section_gives_the_default_piece_rounds():
    settings = collection.CollectionSettings.from_yaml(None, None, "block 0")

    assert settings.piece_rounds == collection.DEFAULT_PIECE_ROUNDS


def test_a_blocks_key_wins_over_the_tops():
    top = {"piece_rounds": 100}
    block = {"piece_rounds": 7}

    settings = collection.CollectionSettings.from_yaml(top, block, "block 0")

    assert settings.piece_rounds == 7


def test_the_tops_key_stands_where_the_block_sets_none():
    top = {"piece_rounds": 100}

    settings = collection.CollectionSettings.from_yaml(top, {}, "block 0")

    assert settings.piece_rounds == 100


def test_a_piece_holds_its_rounds_over_a_shots_whole_shots():
    settings = collection.CollectionSettings(piece_rounds=20000)

    shots = settings.piece_shots(15)

    assert shots == 1333


def test_a_shot_longer_than_a_piece_is_a_piece_of_one_shot():
    settings = collection.CollectionSettings(piece_rounds=10)

    shots = settings.piece_shots(15)

    assert shots == 1


def test_an_unknown_key_is_refused_by_name():
    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml({"piece_shots": 5}, None, "b")

    message = str(refused.value)
    assert "['piece_shots']" in message
    assert "piece_rounds" in message


def test_a_section_that_is_no_mapping_is_refused():
    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml([1], None, "block 0")

    assert "is [1]; it is a mapping of piece_rounds" in str(refused.value)


@pytest.mark.parametrize("value", [0, -3, 2.5, True, "20000"])
def test_piece_rounds_that_is_no_count_is_refused(value):
    block = {"piece_rounds": value}

    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml(None, block, "block 2")

    message = str(refused.value)
    assert message.startswith("block 2 collection piece_rounds must be")
