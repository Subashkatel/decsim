"""The final-answer latency of every original window, absorbed ones included.

On Toshio's double window a switched window's region answers the two
windows after it too. The point times each original window from its own
last round (buffer included, cut at the stream's end) leaving the QPU to
the frame commit that covers its rounds: a kept window's own commit, an
absorbed window's region commit. So it has one sample per original
window, equals the committing decode's own point on a window whose last
round is the decode's, and grows by the rounds between the two on a
switched window.
"""

import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.parameters.assignments as assignments_module
import decsim.parameters.tree as tree
import tests.experiments.run_files as run_files

POINT = "window_last_round_to_final_answer"
ROUNDS = 30


def _shot(verdict):
    machine = run_files.switching_machine(rounds_per_shot=ROUNDS)
    pairs = (
        ("switching.strong_window.type", "double_window"),
        ("windows.boundary_policy.type", "Eager"),
        ("windows.terminal_policy", "lookahead"),
        ("switching.verdict.type", verdict),
        ("switching.confidence.type", "none"),
    )
    texts = [f"{path}={value}" for path, value in pairs]
    written = assignments_module.from_texts(texts, "a test")
    applied = tree.apply(machine, written)
    task = collect.Task("final", applied.settings)
    shot = collect.run_shot(task, 0)
    measurement = measure.measure_shot(shot)
    return shot, measurement


def _round_period_microseconds(shot):
    qpu = shot.task.machine.held("qpu")
    return qpu.round_period_microseconds


def _sorted_windows(shot):
    """The shot's windows, (key, window) in key order."""
    windows = shot.machine.observation.windows.windows
    window_items = windows.items()
    return sorted(window_items)


def test_a_run_that_never_switches_times_each_window_by_its_own_decode():
    shot, measurement = _shot("never")
    windows = shot.machine.observation.windows.windows

    assert len(measurement.samples[POINT]) == len(windows)
    assert (
        measurement.samples[POINT]
        == measurement.samples["qpu_last_round_to_frame"]
    )


def test_every_original_window_counts_when_every_window_switches():
    shot, measurement = _shot("always")
    windows = _sorted_windows(shot)
    absorbed = [window for _key, window in windows if window.is_absorbed]

    assert absorbed
    assert len(measurement.samples[POINT]) == len(windows)


def test_an_absorbed_window_ends_at_its_region_and_starts_at_its_own_round():
    """Two windows of one region end together, one window apart at the start."""
    shot, measurement = _shot("always")
    windows = _sorted_windows(shot)
    keys = [key for key, _window in windows]
    samples = dict(zip(keys, measurement.samples[POINT], strict=True))
    round_period_microseconds = _round_period_microseconds(shot)
    compared = 0
    for index, (key, window) in enumerate(windows[1:], start=1):
        earlier_key, earlier = windows[index - 1]
        if not window.is_absorbed or window.buffer_hi > ROUNDS:
            continue
        rounds_apart = window.buffer_hi - earlier.buffer_hi
        apart_microseconds = samples[earlier_key] - samples[key]
        expected_microseconds = rounds_apart * round_period_microseconds
        deviation = apart_microseconds - expected_microseconds
        # one round period a round, up to the rounding of the ticks
        assert abs(deviation) < 0.01
        compared += 1
    assert compared > 0
