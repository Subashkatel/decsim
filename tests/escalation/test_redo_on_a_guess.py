"""A redo that starts on a guess: the chain of one stream's redos is gone.

Under checked boundaries a window escalates only once the boundary its
decode read is final, so one stream's redos run one after another. A
redo window that starts on a guess escalates at the provisional commit,
pinned on the weak boundary, and is withdrawn when that boundary
changes. The referent is serial switching under held boundaries: on the
same seeds every shot must predict and commit what it does, withdrawn
redos included, while redos of one stream now overlap.
"""

import pytest

import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.presets.machines as machines
import decsim.windows.boundary_policies as boundary_policies
import tests.escalation.test_switching_mode as switching_mode
import tests.windows.test_input_checks as input_checks_tests

CHECKED = boundary_policies.Checked.Settings()
GUESSING_REDO = strong_window_shapes.RedoWindow.Settings(
    boundary_policy=CHECKED, starts_on_a_guess=True
)
# enough strong units that redos of one stream could overlap
STRONG_UNITS = 4


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_guess_commits_what_held_boundaries_do(seed):
    """The final answers are serial switching's; only their ticks move."""
    held = _held_graph()
    guessing = _with_a_guess(held)

    held_machine, held_result = switching_mode.run_shot(held, seed)
    machine, result = switching_mode.run_shot(guessing, seed)

    held_predictions = switching_mode.predictions(held_result)
    assert switching_mode.predictions(result) == held_predictions
    held_writes = input_checks_tests._frame_writes(held_machine)
    assert input_checks_tests._frame_writes(machine) == held_writes


def test_a_changed_face_withdraws_the_redo():
    """At p 0.008 a strong answer changes a face a redo pinned.

    The seeds are test_a_guess_commits_what_held_boundaries_do's, so
    those shots, which answer as held ones do, withdraw a redo.
    """
    held = _held_graph()
    guessing = _with_a_guess(held)

    withdrawn = _count_over(guessing, range(2), "withdrawn")

    assert withdrawn > 0


def test_redos_of_one_stream_overlap_only_on_a_guess():
    """Checked boundaries chain a stream's redos; a guess lets them overlap."""
    held = _held_graph()
    checked = _with_checked_redo(held)
    guessing = _with_a_guess(held)

    chained = _overlaps_over(checked, range(4))
    overlapped = _overlaps_over(guessing, range(4))

    assert chained == 0
    assert overlapped > 0


# ---- helpers


def _held_graph():
    """The gate's switching card, its strong pool wide enough to overlap."""
    graph = switching_mode.switching_graph(20.0)
    return graph.with_settings(machines.STRONG_POOL, unit_count=STRONG_UNITS)


def _with_checked_redo(graph):
    """The run with checked boundaries and a redo that waits for them."""
    redo = strong_window_shapes.RedoWindow.Settings(boundary_policy=CHECKED)
    return _with_redo(graph, redo)


def _with_a_guess(graph):
    """The run with checked boundaries and a redo that starts on a guess."""
    return _with_redo(graph, GUESSING_REDO)


def _with_redo(graph, redo):
    """The run with this redo window, its windows on its boundary row."""
    switched = switching_mode.with_switching(graph, strong_window=redo)
    return switched.with_settings(machines.WINDOWS, boundary_policy=CHECKED)


def _count_over(graph, seeds, statistic: str) -> int:
    """A statistic's value summed over the seeds' shots."""
    total = 0
    for seed in seeds:
        machine, _result = switching_mode.run_shot(graph, seed)
        counts = input_checks_tests._counts(machine.observation.statistics)
        total += counts[statistic]
    return total


def _overlaps_over(graph, seeds) -> int:
    """Strong decodes that started before the one of the window before ended."""
    total = 0
    for seed in seeds:
        machine, _result = switching_mode.run_shot(graph, seed)
        total += _overlapping_redos(machine)
    return total


def _overlapping_redos(machine) -> int:
    """Windows whose strong decode began before the earlier one's ended."""
    spans = {}
    strong_unit = f"{machines.STRONG_POOL}#"
    for record in machine.observation.stages.records:
        if not record.unit_name.startswith(strong_unit):
            continue
        key = (record.stream_id, record.window_id)
        span = spans.setdefault(key, [record.start_ticks, record.end_ticks])
        span[0] = min(span[0], record.start_ticks)
        span[1] = max(span[1], record.end_ticks)
    overlapping = 0
    for (stream_id, window_id), span in spans.items():
        later_key = (stream_id, window_id + 1)
        later = spans.get(later_key)
        if later is not None and later[0] < span[1]:
            overlapping += 1
    return overlapping
