"""The collection section read from yaml (decsim/experiments/collection.py).

A block's key replaces the top's, as a file's key replaces its base's
under `extends`. sinter's CollectionOptions.combine takes the smaller of
two instead (sinter/_data/_collection_options.py:68-99), which would let
the top's value cut a block that asks for more.
"""

import math
import random

import pytest
import sinter._collection._collection_manager as sinter_manager

import decsim.experiments.collection as collection
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.refusal as refusal

CAPPED = {"max_shots": 10}
StopKind = failure_statistics.StopKind


def test_a_capped_section_gives_the_default_piece_rounds():
    settings = collection.CollectionSettings.from_yaml(CAPPED, None, "b")

    assert settings.piece_rounds == collection.DEFAULT_PIECE_ROUNDS
    assert settings.min_shots == 0
    assert settings.max_failures is None


def test_a_blocks_key_wins_over_the_tops():
    top = {"piece_rounds": 100, "max_shots": 10}
    block = {"piece_rounds": 7}

    settings = collection.CollectionSettings.from_yaml(top, block, "block 0")

    assert settings.piece_rounds == 7
    assert settings.max_shots == 10


def test_the_tops_key_stands_where_the_block_sets_none():
    top = {"piece_rounds": 100, "max_shots": 10}

    settings = collection.CollectionSettings.from_yaml(top, {}, "block 0")

    assert settings.piece_rounds == 100


def test_a_point_with_no_cap_is_refused():
    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml(None, None, "block 3")

    message = str(refused.value)
    assert message.startswith("block 3 has no cap")
    assert "max_shots or max_core_seconds" in message


def test_a_time_cap_alone_is_a_cap():
    top = {"max_core_seconds": 60}

    settings = collection.CollectionSettings.from_yaml(top, None, "block 0")

    assert settings.max_core_seconds == 60


def test_a_piece_holds_its_rounds_over_a_shots_whole_shots():
    settings = collection.CollectionSettings(piece_rounds=20000)

    shots = settings.piece_shots(15)

    assert shots == 1333


def test_a_shot_longer_than_a_piece_is_a_piece_of_one_shot():
    settings = collection.CollectionSettings(piece_rounds=10)

    shots = settings.piece_shots(15)

    assert shots == 1


def test_an_unknown_key_is_refused_by_name():
    top = {"piece_shots": 5, "max_shots": 10}

    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml(top, None, "b")

    message = str(refused.value)
    assert "['piece_shots']" in message
    assert "piece_rounds" in message


def test_a_section_that_is_no_mapping_is_refused():
    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml([1], None, "block 0")

    message = str(refused.value)
    assert "is [1]; it is a mapping of max_failures, max_shots" in message


@pytest.mark.parametrize("key", ["piece_rounds", "max_shots", "max_failures"])
@pytest.mark.parametrize("value", [0, -3, 2.5, True, "20000"])
def test_a_count_that_is_no_count_is_refused(key, value):
    block = {"max_shots": 10, key: value}

    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml(None, block, "block 2")

    message = str(refused.value)
    assert message.startswith(f"block 2 collection {key} must be")


@pytest.mark.parametrize("value", [0, -1.5, True, "60", math.inf])
def test_a_time_cap_that_is_no_finite_positive_number_is_refused(value):
    """An infinite time cap never stops a point that has no other cap."""
    block = {"max_core_seconds": value}

    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml(None, block, "block 2")

    message = str(refused.value)
    assert message.startswith("block 2 collection max_core_seconds must be")


@pytest.mark.parametrize("value", [-1, 1.5, False, None])
def test_a_minimum_that_is_no_count_is_refused(value):
    block = {"max_shots": 10, "min_shots": value}

    with pytest.raises(refusal.RefusalError) as refused:
        collection.CollectionSettings.from_yaml(None, block, "block 2")

    message = str(refused.value)
    assert message.startswith("block 2 collection min_shots must be")


def test_a_prefix_short_of_every_stop_runs_on():
    settings = _settings(max_failures=3, max_shots=10, min_shots=4)

    kind = _kind_at(settings, 5, 5, 2)

    assert kind is None


def test_the_target_past_the_minimum_is_a_target_stop():
    settings = _settings(max_failures=3, max_shots=10, min_shots=4)

    kind = _kind_at(settings, 6, 5, 3)

    assert kind is StopKind.TARGET


def test_the_target_on_the_minimums_own_shot_is_a_minimum_stop():
    """The rule stops there because of the minimum, not the target."""
    settings = _settings(max_failures=3, max_shots=10, min_shots=4)

    kind = _kind_at(settings, 4, 4, 3)

    assert kind is StopKind.MINIMUM


def test_failures_past_the_target_wait_for_the_minimum():
    settings = _settings(max_failures=1, max_shots=10, min_shots=4)

    early = _kind_at(settings, 3, 3, 2)
    at_minimum = _kind_at(settings, 4, 4, 2)

    assert early is None
    assert at_minimum is StopKind.MINIMUM


def test_the_minimum_counts_scored_shots_and_the_cap_every_shot():
    settings = _settings(max_failures=1, max_shots=6, min_shots=4)

    with_unscored = _kind_at(settings, 5, 3, 1)
    at_cap = _kind_at(settings, 6, 3, 1)

    assert with_unscored is None
    assert at_cap is StopKind.CAP


def test_the_target_on_the_caps_own_shot_is_a_target_stop():
    settings = _settings(max_failures=2, max_shots=6)

    kind = _kind_at(settings, 6, 6, 2)

    assert kind is StopKind.TARGET


def test_the_time_cap_stops_a_point_at_its_seconds():
    settings = _settings(max_core_seconds=30.0)

    before = _kind_at(settings, 9, 9, 0, 29.5)
    at_cap = _kind_at(settings, 10, 10, 0, 30.0)

    assert before is None
    assert at_cap is StopKind.CAP


def test_the_rule_stops_where_sinters_stops_on_the_same_counts_property():
    """No minimum and no time cap: the rule is sinter's, shot for shot."""
    generator = random.Random(7)
    for _case in range(300):
        max_failures = generator.randint(1, 6)
        max_shots = generator.randint(1, 40)
        rate = generator.random()
        outcomes = []
        for _shot in range(max_shots):
            draw = generator.random()
            is_failed = draw < rate
            failed = int(is_failed)
            outcomes.append(failed)
        settings = _settings(max_failures=max_failures, max_shots=max_shots)

        sinter_stop = _sinter_stop_shot(outcomes, max_failures, max_shots)
        decsim_stop = _decsim_stop_shot(outcomes, settings)

        assert decsim_stop == sinter_stop


def test_a_prefix_ends_at_its_stop_and_later_rows_count_nowhere():
    settings = _settings(max_failures=1, max_shots=10)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0), _shot_row(1, failed=True), _shot_row(2, failed=True)]

    tracker = _tracked(rule, rows)

    assert tracker.state() == "target"
    assert tracker.counts.shots == 2
    assert tracker.counts.failures == 1


def test_a_missing_seed_ends_the_prefix_short_of_its_stop():
    """A gap holds the stop: the rows past it wait for the missing seed."""
    settings = _settings(max_failures=1, max_shots=10)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0), _shot_row(2, failed=True)]

    tracker = _tracked(rule, rows)

    assert tracker.state() == "running"
    assert tracker.counts.shots == 1
    assert tracker.stop_kind_for_limits() is StopKind.CAP


def test_shots_fixed_in_advance_are_a_cap_at_the_shots_run():
    rows = [_shot_row(0), _shot_row(1, failed=True)]

    fixed = collection.PointRule()
    tracker = _tracked(fixed, rows)

    assert tracker.state() == "cap"
    assert tracker.counts.shots == 2


def test_a_point_stopped_by_its_time_cap_says_so():
    """Its interval assumes a shot's time is independent of its failure.

    Every shot row takes half a second, so a one-second cap stops the
    prefix on its second shot, short of the shot cap: the state names
    the time cap, the one stop whose limits rest on that assumption
    (design section 7).
    """
    settings = _settings(max_shots=10, max_core_seconds=1.0)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0), _shot_row(1), _shot_row(2)]

    tracker = _tracked(rule, rows)

    assert tracker.state() == "time cap"
    assert tracker.counts.shots == 2
    assert tracker.stop_kind_for_limits() is StopKind.CAP


def test_a_shot_cap_reached_with_the_time_cap_is_a_shot_cap():
    """A count fixed in advance needs no assumption about time."""
    settings = _settings(max_shots=2, max_core_seconds=1.0)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0), _shot_row(1)]

    tracker = _tracked(rule, rows)

    assert tracker.state() == "cap"


def test_an_adaptive_point_says_so_whatever_its_counts():
    settings = _settings(max_shots=2)
    rule = collection.PointRule(settings, True)
    rows = [_shot_row(0), _shot_row(1)]

    tracker = _tracked(rule, rows)

    assert tracker.state() == "adaptive"


def test_a_point_with_no_shot_has_no_data():
    fixed = collection.PointRule()
    tracker = collection.PrefixTracker(fixed)

    assert tracker.state() == "no data"


def test_an_unscored_shot_counts_toward_the_cap_and_not_the_failures():
    settings = _settings(max_failures=1, max_shots=2)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0, is_scored=False), _shot_row(1, is_scored=False)]

    tracker = _tracked(rule, rows)

    assert tracker.state() == "cap"
    assert tracker.counts.scored_shots == 0


def test_a_prefix_whose_shots_ran_one_shape_is_converted_by_it():
    settings = _settings(max_shots=2)
    rule = collection.PointRule(settings, False)
    rows = [
        _shot_row(0, outputs=4, rounds=8),
        _shot_row(1, outputs=4, rounds=8),
    ]

    tracker = _tracked(rule, rows)

    assert tracker.round_shape() == (4, 8)


def test_a_prefix_whose_shots_ran_different_lengths_has_no_one_shape():
    """A live stream's feedback wait can differ from shot to shot."""
    settings = _settings(max_shots=2)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0, rounds=8), _shot_row(1, rounds=9)]

    tracker = _tracked(rule, rows)

    assert tracker.round_shape() is None


def test_a_prefix_whose_outputs_ran_apart_has_no_one_shape():
    """Its shots record no one length per output, which reads as 0."""
    settings = _settings(max_shots=1)
    rule = collection.PointRule(settings, False)
    rows = [_shot_row(0, outputs=2, rounds=0)]

    tracker = _tracked(rule, rows)

    assert tracker.round_shape() is None


def _settings(**keys) -> collection.CollectionSettings:
    return collection.CollectionSettings.from_yaml(keys, None, "block 1")


def _counts(shots, scored_shots, failures, core_seconds=0.0):
    return collection.PrefixCounts(shots, scored_shots, failures, core_seconds)


def _kind_at(settings, shots, scored_shots, failures, core_seconds=0.0):
    """The stop kind of a prefix with these counts."""
    counts = _counts(shots, scored_shots, failures, core_seconds)
    return settings.stop_kind(counts)


def _sinter_stop_shot(outcomes, max_failures, max_shots) -> int:
    """The shot after which sinter's own task state says it is complete.

    sinter's manager starts errors_left at the smaller of max_errors and
    max_shots and takes each shot and each error off
    (sinter/_collection/_collection_manager.py:228-234, 330-342).
    """
    state = sinter_manager._ManagedTaskState(
        partial_task=None,
        strong_id="point",
        shots_left=max_shots,
        errors_left=min(max_failures, max_shots),
    )
    for shot, failed in enumerate(outcomes, start=1):
        state.shots_left -= 1
        state.errors_left -= failed
        if state.is_completed():
            return shot
    return None


def _decsim_stop_shot(outcomes, settings) -> int:
    """The shot after which the collection's rule first gives a kind."""
    counts = collection.PrefixCounts()
    for failed in outcomes:
        shot_counts = _counts(1, 1, failed)
        counts.add(shot_counts)
        if settings.stop_kind(counts) is not None:
            return counts.shots
    return None


def _shot_row(seed, is_scored=True, failed=False, outputs=1, rounds=15):
    """A shot row as shots.csv holds it, the fields a prefix reads."""
    return {
        "seed": str(seed),
        "is_scored": str(is_scored),
        "logical_failure": str(failed),
        "sim_wall_seconds": "0.5",
        "scored_outputs": str(outputs),
        "rounds_per_output": str(rounds),
    }


def _tracked(rule, rows) -> collection.PrefixTracker:
    tracker = collection.PrefixTracker(rule)
    for row in rows:
        tracker.add(row)
    return tracker
