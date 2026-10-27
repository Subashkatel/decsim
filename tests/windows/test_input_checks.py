"""The input checks: a decode on a guessed boundary waits for the final one.

The referent is serial switching itself: a window that read its
neighbour's weak boundary before the strong answer must end with the
answer it gets under held boundaries, where it waits for that answer
before it decodes. SWIPER (Vittal et al. 2412.05115 lines 625-644)
restarts only the window whose predicted boundary was wrong; the whole
runs here check that a checked run predicts each shot and commits each
window from the tier a held run does, on the same seeds, while the
window after an escalated one decodes before the strong answer lands.
"""

import pytest

import decsim.engine as engine_module
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.presets.machines as machines
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import decsim.statistics.registry as statistics_registry
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.input_checks as input_checks
import decsim.windows.window_interactions as window_interactions
import tests.escalation.test_switching_mode as switching_mode

SOURCE = (1, 0)
GUESS = {(0, 4): [1, 0, 1]}
OTHER = {(0, 4): [0, 1, 1]}


def test_a_decode_on_final_boundaries_is_kept_nowhere():
    checks, window, registry = _checks_with_a_window()
    job = _job(window)
    _land(checks, window, GUESS, is_final=True)
    checks.note_read(job)
    state = checks.take_result(job)
    assert checks.read_by_window == {}
    assert state is input_checks.InputState.FINAL


def test_a_result_on_a_guess_waits_and_settles_when_the_same_lands():
    checks, window, registry = _checks_with_a_window()
    job = _job(window)
    _land(checks, window, GUESS, is_final=False)
    checks.note_read(job)
    state = checks.take_result(job)
    steps = []
    checks.wait_for_final_input(
        window.key, lambda: steps.append("final"), lambda: steps.append("stale")
    )
    _land(checks, window, GUESS, is_final=True)
    assert state is input_checks.InputState.PROVISIONAL
    assert steps == ["final"]
    assert _counts(registry) == {
        "provisional_reads": 1,
        "confirmed": 1,
        "decoded_again": 0,
    }


def test_a_result_on_a_guess_is_decoded_again_when_another_lands():
    checks, window, registry = _checks_with_a_window()
    job = _job(window)
    _land(checks, window, GUESS, is_final=False)
    checks.note_read(job)
    checks.take_result(job)
    steps = []
    checks.wait_for_final_input(
        window.key, lambda: steps.append("final"), lambda: steps.append("stale")
    )
    _land(checks, window, OTHER, is_final=True)
    assert steps == ["stale"]
    assert _counts(registry) == {
        "provisional_reads": 1,
        "confirmed": 0,
        "decoded_again": 1,
    }


def test_a_running_decode_whose_guess_changed_is_dropped_when_it_ends():
    checks, window, registry = _checks_with_a_window()
    job = _job(window)
    _land(checks, window, GUESS, is_final=False)
    checks.note_read(job)
    _land(checks, window, OTHER, is_final=False)
    first = checks.take_result(job)
    again = checks.take_result(job)
    assert first is input_checks.InputState.STALE
    assert again is input_checks.InputState.FINAL


def test_a_running_decode_confirmed_before_it_ends_reads_final():
    checks, window, registry = _checks_with_a_window()
    job = _job(window)
    _land(checks, window, GUESS, is_final=False)
    checks.note_read(job)
    _land(checks, window, GUESS, is_final=True)
    state = checks.take_result(job)
    assert state is input_checks.InputState.FINAL


def test_trailing_zeros_change_no_boundary():
    checks, window, registry = _checks_with_a_window()
    job = _job(window)
    padded = {(0, 4): [1, 0, 1, 0, 0]}
    _land(checks, window, GUESS, is_final=False)
    checks.note_read(job)
    _land(checks, window, padded, is_final=True)
    state = checks.take_result(job)
    assert state is input_checks.InputState.FINAL


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_checked_boundaries_commit_what_held_ones_do(seed):
    """The final answers are serial switching's; only their ticks move."""
    held = switching_mode.switching_graph(20.0)
    checked = _with_checked_boundaries(held)

    held_machine, held_result = switching_mode.run_shot(held, seed)
    checked_machine, checked_result = switching_mode.run_shot(checked, seed)

    held_predictions = switching_mode.predictions(held_result)
    assert switching_mode.predictions(checked_result) == held_predictions
    held_writes = _frame_writes(held_machine)
    assert _frame_writes(checked_machine) == held_writes
    assert "strong" in {tier for tier, _observables in held_writes.values()}


def test_the_window_after_an_escalated_one_decodes_before_its_answer():
    """Under held boundaries it waits for the strong answer; here, not."""
    graph = switching_mode.switching_graph(20.0)
    checked = _with_checked_boundaries(graph)

    machine, _result = switching_mode.run_shot(checked, 0)

    early = _decodes_before_the_answer_before_them(machine)
    assert early


def test_a_wrong_guess_is_decoded_again():
    """At p 0.008 a strong answer changes some boundary, so a window redoes.

    The seeds are test_checked_boundaries_commit_what_held_ones_do's, so
    those shots, which answer as held ones do, hold a decode again.
    """
    graph = switching_mode.switching_graph(20.0)
    checked = _with_checked_boundaries(graph)

    decoded_again = _decoded_again_over(checked, range(4))

    assert decoded_again > 0


# ---- helpers


def _checks_with_a_window() -> tuple:
    """Checks on the defect masks, counted, and a window two rounds on."""
    checks = input_checks.InputChecks()
    engine = engine_module.Engine()
    registry = statistics_registry.Registry(engine)
    registry.attach(checks, "input_checks")
    payload = boundary_payloads.DenseSeamMask()
    checks.interaction = window_interactions.DefaultWindowInteraction(
        0, payload
    )
    window = window_records.Window(
        stream_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=8,
        round_count=8,
        deps=[SOURCE],
    )
    return checks, window, registry


def _land(checks, window, masks: dict, is_final: bool) -> None:
    """The source's boundary lands: the window's state is these masks."""
    window.boundary_in = dict(masks)
    checks.boundary_landed(window, SOURCE, is_final)


def _job(window) -> decoding_records.DecodeJob:
    """A decode of the window on its entry pool."""
    return decoding_records.DecodeJob(
        stream_id=window.stream_id,
        window_id=window.window_index,
        round_count=window.round_count,
        window=window,
    )


def _counts(registry) -> dict:
    """The input checks' counts, by statistic."""
    counts = {}
    for row in registry.rows():
        counts[row.statistic] = row.value
    return counts


def _frame_writes(machine) -> dict:
    """Each window's one write at the frame: its tier and its observables.

    A second write of a window would leave fewer keys than records.
    """
    snapshot = machine.control.pauli_frame.snapshot()
    records = snapshot.records
    writes = {}
    for record in records:
        writes[record.window_key] = (record.tier, record.logical_observables)
    assert len(writes) == len(records)
    return writes


def _with_checked_boundaries(graph):
    """The run with the redo window's boundary row checked, windows too."""
    checked = boundary_policies.Checked.Settings()
    redo = strong_window_shapes.RedoWindow.Settings(boundary_policy=checked)
    switched = switching_mode.with_switching(graph, strong_window=redo)
    return switched.with_settings(machines.WINDOWS, boundary_policy=checked)


def _decodes_before_the_answer_before_them(machine) -> list:
    """Weak decodes that started before the strong decode before them ended."""
    strong_ends = {}
    weak_starts = {}
    strong_unit = f"{machines.STRONG_POOL}#"
    for record in machine.observation.stages.records:
        key = (record.stream_id, record.window_id)
        if record.unit_name.startswith(strong_unit):
            last_end = strong_ends.get(key, 0)
            strong_ends[key] = max(last_end, record.end_ticks)
            continue
        first = weak_starts.get(key, record.start_ticks)
        weak_starts[key] = min(first, record.start_ticks)
    early = []
    for (stream_id, window_id), strong_end in strong_ends.items():
        after = (stream_id, window_id + 1)
        later = weak_starts.get(after)
        if later is not None and later < strong_end:
            early.append(after)
    return early


def _decoded_again_over(graph, seeds) -> int:
    """The decodes dropped for a changed boundary, over the seeds' shots."""
    total = 0
    for seed in seeds:
        machine, _result = switching_mode.run_shot(graph, seed)
        counts = _counts(machine.observation.statistics)
        total += counts["decoded_again"]
    return total
