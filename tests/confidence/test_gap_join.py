"""The window's gap as two forced-class jobs, on one unit and on two.

The window side submits both
forced-class jobs at one instant, both queue in the weak pool, each unit
is fed from the weak syndrome buffer, and weak_decoder.units alone decides
whether the pair overlaps. The rule the data-movement study rests on is one
copy per unit that reads the window: one unit moves the window's bits once and
is busy for the sum of the two solves, two units move them twice and
overlap, and the committed corrections are the same either way.
"""

import dataclasses
import json
import math

import pytest

import decsim.confidence.complementary as complementary
import decsim.confidence.gap_join as gap_join_module
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.producers as producers
import decsim.records.decoding as decoding_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
import tests.escalation.test_strong_window_shapes as test_strong_window_shapes

WEAK_INPUT_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
# a window identity the run never requests, so its solve is never joined
UNJOINED_WINDOW_ID = 999


def _switching_machine(
    weak_units: int,
    weak_microseconds: float = 4.0,
    trace_path=None,
    result_blocks_unit: bool = False,
    walk_microseconds=None,
    patch_count: int = 1,
):
    """The gate's switching card at d=3, priced so its ticks are declared."""
    settings = test_strong_window_shapes.gate_switching()
    weak_decoder = test_strong_window_shapes.priced_pool(
        settings.weak_decoder, weak_microseconds, weak_units
    )
    weak_decoder = dataclasses.replace(
        weak_decoder, result_blocks_unit=result_blocks_unit
    )
    strong_decoder = test_strong_window_shapes.priced_pool(
        settings.strong_decoder, 20.0
    )
    confidence = complementary.ComplementaryGap.Settings(
        walk_microseconds=walk_microseconds
    )
    switching = dataclasses.replace(settings.switching, confidence=confidence)
    workload = settings.workload
    if patch_count > 1:
        workload = _memory_patches(patch_count)
    trace = "off"
    if trace_path is not None:
        trace = str(trace_path)
    observation = dataclasses.replace(
        settings.observation, record_switching_windows=True, trace=trace
    )
    settings = dataclasses.replace(
        settings,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
        workload=workload,
        observation=observation,
    )
    return machine_module.Machine.build(settings, 0)


def _memory_patches(patch_count: int):
    """patch_count memory patches of the gate's card, 10 d rounds each."""
    patches = producers.memory_patches(
        "surface_code:rotated_memory_z",
        30,
        patch_count,
        3,
        test_strong_window_shapes.GATE_PHYSICAL_ERROR_PROBABILITY,
    )
    return workload_settings.WorkloadSettings.running(patches)


def _weak_requests(machine) -> list:
    weak = window_records.DecoderTier.WEAK
    records = []
    for record in machine.observation.decode_records.requests:
        if record.request_key.tier is weak:
            records.append(record)
    return records


def _weak_decode_starts(machine) -> list:
    starts = fabric.log_lines_containing(machine, "START DECODE mem")
    weak_starts = []
    for line in starts:
        if "strong(" not in line:
            weak_starts.append(line)
    return weak_starts


def _companions_and_answers(requests) -> tuple:
    """The weak requests that ended as companions, and those answered."""
    outcomes = decoding_records.RequestProcessingOutcome
    companion = outcomes.WEAK_FORCED_CLASS_COMPANION
    companions = []
    answered = []
    for ended in requests.of_tier(window_records.DecoderTier.WEAK):
        if ended.outcome is companion:
            companions.append(ended)
            continue
        if ended.result is None:
            continue
        if ended.result.soft_output is not None:
            answered.append(ended)
    return companions, answered


def _rows_named(document: list, name: str) -> list:
    rows = []
    for row in document:
        if row.get("name") == name:
            rows.append(row)
    return rows


def _weak_input_traffic(machine) -> tuple:
    """(transfers, bits) that moved into the weak units."""
    snapshot = machine.observation.traffic.snapshot()
    for path_snapshot in snapshot.paths:
        if path_snapshot.path is WEAK_INPUT_PATH:
            counters = path_snapshot.counters
            return counters.transfer_count, counters.known_payload_bits
    raise AssertionError("the weak input path is not on the fabric")


def _committed_observables(machine) -> list:
    committed = []
    snapshot = machine.control.pauli_frame.snapshot()
    for record in snapshot.records:
        committed.append((record.window_key, record.logical_observables))
    return committed


def test_one_unit_decodes_each_window_twice_and_moves_its_bits_once():
    """Two requests per window, one transfer per window into the unit."""
    machine = _switching_machine(1)
    machine.run()
    windows = len(machine.observation.windows.windows)
    weak_requests = _weak_requests(machine)
    assert len(weak_requests) == 2 * windows
    transfers, _bits = _weak_input_traffic(machine)
    assert transfers == windows


def test_two_units_move_the_bits_twice_and_commit_the_same_corrections():
    """The second unit buys the fast solve and costs a second copy."""
    one_unit = _switching_machine(1)
    one_unit.run()
    two_units = _switching_machine(2)
    two_units.run()
    one_transfers, one_bits = _weak_input_traffic(one_unit)
    two_transfers, two_bits = _weak_input_traffic(two_units)
    assert two_transfers == 2 * one_transfers
    assert two_bits == 2 * one_bits
    assert _committed_observables(two_units) == _committed_observables(one_unit)


def test_one_unit_holds_the_window_for_the_sum_of_its_two_solves():
    """A priced weak card prices one decode; the pair is two of them."""
    machine = _switching_machine(1, weak_microseconds=4.0)
    machine.run()
    weak_starts = _weak_decode_starts(machine)
    windows = len(machine.observation.windows.windows)
    assert len(weak_starts) == 2 * windows


def test_one_unit_that_blocks_on_its_result_joins_both_solves():
    """The held solve is read into the join, so its unit runs the other."""
    machine = _switching_machine(1, result_blocks_unit=True)
    machine.run()
    windows = len(machine.observation.windows.windows)
    committed = _committed_observables(machine)
    assert len(committed) == windows


def _walks_and_next_starts(machine) -> list:
    """(walk end, next start) on the unit, for each decode that walked.

    One unit, so the unit's decodes start in the order recorded here.
    """
    service = machine.decoders.decoder_manager.service
    outcomes = machine.decoders.decoder_manager.outcomes
    engine = machine.engine
    starts = []
    ends = {}

    def started(job, _unit) -> None:
        starts.append((engine.now, job))

    def answered(job, _result, _outcome, answered_ticks) -> None:
        ends[id(job)] = answered_ticks

    service.trace.job_started.connect(started)
    outcomes.trace.request_ended.connect(answered)
    machine.run()
    pairs = []
    pair_count = len(starts) - 1
    for index in range(pair_count):
        (_tick, job) = starts[index]
        (next_tick, _next_job) = starts[index + 1]
        if job.soft_output_ticks > 0:
            walk_end = ends[id(job)]
            pairs.append((walk_end, next_tick))
    return pairs


def test_the_walk_keeps_its_unit_busy_before_the_next_decode_starts():
    """Decision D8: the walk is the unit's time, so nothing overlaps it.

    Two patches, so the other patch's window is ready while the walk runs.
    """
    machine = _switching_machine(1, walk_microseconds=12.0, patch_count=2)

    pairs = _walks_and_next_starts(machine)

    walk_count = len(pairs)
    assert walk_count > 0
    waits = [next_start - walk_end for (walk_end, next_start) in pairs]
    assert min(waits) >= 0


def test_the_first_solve_is_held_and_the_join_names_the_windows_gap():
    """The held half and the join are named in the run's log."""
    machine = _switching_machine(1)
    machine.run()
    held = fabric.log_lines_containing(machine, "GAP HOLD")
    joined = fabric.log_lines_containing(machine, "GAP JOIN")
    windows = len(machine.observation.windows.windows)
    assert len(held) == windows
    assert len(joined) == windows
    assert "waits for the other class" in held[0]
    assert "gap " in joined[0]


def test_a_windows_two_requests_are_one_attempt():
    """One attempt, two forced-class requests, one of them the answer."""
    machine = _switching_machine(1)
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    machine.run()
    companions, answered = _companions_and_answers(requests)
    windows = len(machine.observation.windows.windows)
    assert len(companions) == windows
    assert len(answered) == windows


def test_a_window_whose_other_solve_never_arrives_refuses_the_run():
    """A solve nobody joins is unsettled state, refused at the run's end."""
    machine = _switching_machine(1)
    join = machine.windows.window_manager.requester.gap_join
    joined_solves = join.accept_result

    def hold_the_first_solve_alone(job, result):
        joined_solves(job, result)
        join.accept_result = joined_solves
        solve = gap_join_module.HeldSolve(job, result)
        join.held_by_window[(job.operation_id, UNJOINED_WINDOW_ID)] = [solve]

    join.accept_result = hold_the_first_solve_alone
    with pytest.raises(RuntimeError, match="unjoined solve"):
        machine.run()


class _CostingSignal:
    """A signal row that reports a gap and what computing it cost."""

    forced_logical_classes = ()

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self.source = decoding_records.SoftOutputSource(method="test_signal")

    def compute(self, solves):
        """One gap, and the ticks this row's own computation took."""
        del solves
        soft_output = decoding_records.SoftOutput(gap=1.0, source=self.source)
        return decoding_records.SoftOutputComputation(soft_output, self.ticks)


class _RecordingVerdict:
    """Keeps the tick at which each window's answer reached it."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.answers = []

    def accept_result(self, job, result):
        """The window's answer, stamped with the tick it arrived."""
        self.answers.append((job, result, self.engine.now))


class _RecordingQueue:
    """The decoder side, as the join sees it: it charges and closes."""

    def __init__(self) -> None:
        self.charges = []
        self.closed = []
        self.read = []

    def charge_soft_output(self, job, ticks):
        """Charge the signal's own computation on the job's unit."""
        job.soft_output_ticks = ticks
        self.charges.append((job, ticks))

    def close_companion_request(self, job, result):
        """The solve the window's answer did not come from."""
        self.closed.append((job, result))

    def read_result(self, job: decoding_records.DecodeJob) -> None:
        """The join has the solve in hand."""
        self.read.append(job)


def _join_with(signal_ticks: int):
    engine = engine_module.Engine()
    signal = _CostingSignal(signal_ticks)
    verdict = _RecordingVerdict(engine)
    queue = _RecordingQueue()
    join = gap_join_module.WindowGapJoin(engine)
    join.signal = signal
    join.verdict = verdict
    join.decode_queue = queue
    return engine, join, verdict, queue


def _forced_solve(label: str, forced_class: int, weight: float):
    """One forced-class solve of window 0, with its answer's weight."""
    job = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, label=label
    )
    job.forced_logical_class = forced_class
    result = decoding_records.DecodeResult(1, 0)
    result.forced_class_weight = weight
    return job, result


def _two_solve_join(signal_ticks: int):
    """A join whose signal forces two classes, as the complementary gap does."""
    engine, join, verdict, queue = _join_with(signal_ticks)
    join.signal.forced_logical_classes = (0, 1)
    return engine, join, verdict, queue


def test_the_walk_is_charged_to_the_solve_that_delivered_last():
    """Two solves, and the ticks go to the unit that is still busy.

    The answering solve is the lightest, which is not in general the last
    to arrive. The decoder manager gives a unit back
    job.soft_output_ticks after it delivers that unit's job, which is
    the last one; the lightest solve's unit may have gone back already.
    The complementary gap's card
    (escalation.confidence_walk_microseconds) is the only way to price
    the walk at all, and it defaults to null, so no shipped config sees
    this.
    """
    engine, join, verdict, queue = _two_solve_join(90)
    first_job, first_result = _forced_solve("first", 0, 4.0)
    second_job, second_result = _forced_solve("second", 1, 9.0)
    join.accept_result(first_job, first_result)
    assert queue.charges == []
    join.accept_result(second_job, second_result)
    assert queue.charges == [(second_job, 90)]
    assert second_job.soft_output_ticks == 90
    assert first_job.soft_output_ticks == 0


def test_the_answer_is_still_the_lightest_solve_and_still_waits():
    """The two halves stay apart: the ticks moved, the answer did not."""
    engine, join, verdict, queue = _two_solve_join(90)
    first_job, first_result = _forced_solve("first", 0, 4.0)
    second_job, second_result = _forced_solve("second", 1, 9.0)
    join.accept_result(first_job, first_result)
    join.accept_result(second_job, second_result)
    assert first_result.soft_output.gap == 1.0
    assert second_result.soft_output is None
    engine.run()
    (answered_job, _answered_result, tick) = verdict.answers[0]
    assert answered_job is first_job
    assert tick == 90


def test_a_two_solve_runs_trace_carries_every_held_solve(tmp_path):
    """The trace listens to solve_held, as the join's docstring says.

    Every window of this run needs two forced-class solves, so every
    window holds exactly one.
    """
    path = tmp_path / "switching.trace.json"
    machine = _switching_machine(1, trace_path=path)
    machine.run()
    machine.observation.trace_writer.write(str(path))
    text = path.read_text()
    document = json.loads(text)
    held = _rows_named(document, "solve held")
    windows = len(machine.observation.windows.windows)
    assert len(held) == windows
    assert held[0]["ph"] == "i"
    assert held[0]["args"]["forced_class"] in (0, 1)
    assert held[0]["cat"] == "window,confidence"


def test_the_lightest_forced_solve_is_the_windows_answer():
    """The unconstrained minimum is the minimum over the classes.

    So whichever class came back lighter is the decoder's own answer,
    whatever order the two solves arrived in.
    """
    engine, join, verdict, queue = _two_solve_join(0)
    heavy_job, heavy_result = _forced_solve("heavy", 0, 9.0)
    light_job, light_result = _forced_solve("light", 1, 4.0)

    join.accept_result(heavy_job, heavy_result)
    join.accept_result(light_job, light_result)
    (answered_job, _answered_result, _tick) = verdict.answers[0]

    assert answered_job is light_job
    assert queue.closed == [(heavy_job, heavy_result)]


def test_the_lightest_solve_answers_whichever_order_it_arrived_in():
    engine, join, verdict, queue = _two_solve_join(0)
    light_job, light_result = _forced_solve("light", 0, 4.0)
    heavy_job, heavy_result = _forced_solve("heavy", 1, 9.0)

    join.accept_result(light_job, light_result)
    join.accept_result(heavy_job, heavy_result)
    (answered_job, _answered_result, _tick) = verdict.answers[0]

    assert answered_job is light_job


def test_a_solve_with_no_weight_leaves_the_first_solve_answering():
    """A signal that forces no class, or a window that pins no observable."""
    engine, join, verdict, queue = _two_solve_join(0)
    first_job, first_result = _forced_solve("first", 0, 4.0)
    weightless_job, weightless_result = _forced_solve("second", 1, 9.0)
    weightless_result.forced_class_weight = None

    join.accept_result(first_job, first_result)
    join.accept_result(weightless_job, weightless_result)
    (answered_job, _answered_result, _tick) = verdict.answers[0]

    assert answered_job is first_job


def test_a_first_solve_with_no_weight_answers_the_window_itself():
    engine, join, verdict, queue = _two_solve_join(0)
    weightless_job, weightless_result = _forced_solve("first", 0, 4.0)
    weightless_result.forced_class_weight = None
    other_job, other_result = _forced_solve("second", 1, 1.0)

    join.accept_result(weightless_job, weightless_result)
    join.accept_result(other_job, other_result)
    (answered_job, _answered_result, _tick) = verdict.answers[0]

    assert answered_job is weightless_job


def test_only_the_answering_solve_carries_the_windows_soft_output():
    """The gap belongs to the window's answer, not to every solve of it."""
    engine, join, verdict, queue = _two_solve_join(0)
    heavy_job, heavy_result = _forced_solve("heavy", 0, 9.0)
    light_job, light_result = _forced_solve("light", 1, 4.0)

    join.accept_result(heavy_job, heavy_result)
    join.accept_result(light_job, light_result)

    assert light_result.soft_output.gap == 1.0
    assert heavy_result.soft_output is None


def test_a_class_no_correction_reaches_never_answers_the_window():
    """The reachable class answers, whichever order the two arrived in."""
    engine, join, verdict, queue = _two_solve_join(0)
    unreachable_job, unreachable_result = _forced_solve(
        "unreachable", 0, math.inf
    )
    reachable_job, reachable_result = _forced_solve("reachable", 1, 4.0)

    join.accept_result(unreachable_job, unreachable_result)
    join.accept_result(reachable_job, reachable_result)
    (answered_job, _answered_result, _tick) = verdict.answers[0]

    assert answered_job is reachable_job
