"""The shots the syndrome source sampled, by the operation that asked.

A listener on the SyndromeSource port's shot_sampled source. A sampled
shot is the run's input, not its output: the detection events the device
drew and the circuit they came from are what a whole-circuit reference
decode reads to check the loop's answer (front/measure.py), so they are
heard once at sampling time instead of read back off the device.
"""

import decsim.observe.sampled_shots as sampled_shots_module


class _Operation:
    """The two fields the listener reads off the operation that asked."""

    def __init__(self, operation_id: int, circuit) -> None:
        self.id = operation_id
        self.circuit = circuit


def test_a_run_that_sampled_nothing_holds_no_shot():
    shots = sampled_shots_module.SampledShots()

    assert shots.shots_by_operation == {}


def test_a_shot_is_kept_under_the_operation_it_was_sampled_for():
    shots = sampled_shots_module.SampledShots()
    circuit = object()
    operation = _Operation(1, circuit)

    shots.shot_sampled(operation, [0, 1, 0])
    shot = shots.shots_by_operation[1]

    assert shot.operation_id == 1
    assert shot.circuit is circuit


def test_the_events_are_frozen_into_a_tuple_of_bits():
    """The listener keeps the run's input, so nothing later can change it."""
    shots = sampled_shots_module.SampledShots()
    drawn = [0, 1, 1]

    operation = _Operation(1, None)

    shots.shot_sampled(operation, drawn)
    drawn.append(1)
    shot = shots.shots_by_operation[1]

    assert shot.detection_events == (False, True, True)


def test_a_second_shot_for_one_operation_replaces_the_first():
    shots = sampled_shots_module.SampledShots()

    first_ask = _Operation(1, None)
    second_ask = _Operation(1, None)

    shots.shot_sampled(first_ask, [0])
    shots.shot_sampled(second_ask, [1])

    assert shots.shots_by_operation[1].detection_events == (True,)
