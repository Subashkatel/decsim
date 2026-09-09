"""The window's gap as two forced-class jobs, on one unit and on two.

Design audit note 12 sections 4.1 to 4.4: the window side submits both
forced-class jobs at one instant, both queue in the weak pool, each unit
is fed from Buffer 0, and weak_decoder.units alone decides whether the
pair overlaps. The rule the data-movement study rests on is one copy per
unit that reads the window: one unit moves the window's bits once and is
busy for the sum of the two solves, two units move them twice and
overlap, and the committed corrections are the same either way.
"""

import copy
import dataclasses
import pathlib

import pytest

import decsim.confidence.gap_join as gap_join_module
import decsim.engine as engine_module
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import tests.escalation.declared_fabric as fabric
from tests.escalation.test_strong_window_shapes import GATE_SWITCHING_CARD

WEAK_INPUT_PATH = transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
# a window identity the run never requests, so its solve is never joined
UNJOINED_WINDOW_ID = 999


def _switching_machine(weak_units: int, weak_microseconds: float = 4.0):
    """The gate's switching card at d=3, priced so its ticks are declared."""
    sections = copy.deepcopy(GATE_SWITCHING_CARD)
    sections["weak_decoder"]["kind"] = weak_microseconds
    sections["weak_decoder"]["units"] = weak_units
    sections["strong_decoder"]["kind"] = 20.0
    base_directory = pathlib.Path(".")
    settings = machine_settings.MachineSettings.from_mapping(
        sections, name="switching_validation", base_directory=base_directory
    )
    qpu = dataclasses.replace(
        settings.qpu, distance=3, round_period_microseconds=1.0
    )
    workload = dataclasses.replace(
        settings.workload, physical_error_probability=0.008
    )
    observation = dataclasses.replace(
        settings.observation, record_switching_windows=True
    )
    settings = dataclasses.replace(
        settings, qpu=qpu, workload=workload, observation=observation
    )
    return machine_module.Machine.build(settings, 0)


def _weak_requests(machine) -> list:
    weak = window_records.DecoderTier.WEAK
    records = []
    for record in machine.observation.decode_records.requests:
        if record.request_key.tier is weak:
            records.append(record)
    return records


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
    snapshot = machine.pauli_frame.snapshot()
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
    starts = fabric.log_lines_containing(machine, "START DECODE mem")
    weak_starts = []
    for line in starts:
        if "strong(" not in line:
            weak_starts.append(line)
    windows = len(machine.observation.windows.windows)
    assert len(weak_starts) == 2 * windows


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


def test_a_windows_two_requests_are_one_attempt_in_the_ledger():
    """One attempt, two forced-class requests, one of them the answer."""
    machine = _switching_machine(1)
    machine.run()
    outcomes = decoding_records.RequestProcessingOutcome
    companion = outcomes.WEAK_FORCED_CLASS_COMPANION
    companions = []
    answered = []
    for record in _weak_requests(machine):
        if record.terminal_processing_outcome is companion:
            companions.append(record)
            continue
        if record.soft_output is not None:
            answered.append(record)
    windows = len(machine.observation.windows.windows)
    assert len(companions) == windows
    assert len(answered) == windows


def test_a_window_whose_other_solve_never_arrives_refuses_the_run():
    """A solve nobody joins is unsettled state, refused at the run's end."""
    machine = _switching_machine(1)
    join = machine.window_manager.requester.gap_join
    joined_solves = join.accept_result
    left_alone = []

    def hold_one_solve_alone(job, result):
        joined_solves(job, result)
        if left_alone:
            return
        left_alone.append(job)
        solve = gap_join_module.HeldSolve(job, result)
        join.held_by_window[(job.operation_id, UNJOINED_WINDOW_ID)] = [solve]

    join.accept_result = hold_one_solve_alone
    with pytest.raises(RuntimeError, match="unjoined solve"):
        machine.run()


class _CostingSignal:
    """A signal row that reports a gap and what computing it cost."""

    forced_logical_classes = ()

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self.source = decoding_records.SoftOutputSource(
            method="test_signal",
            cluster_origin="test",
            growth_schedule="test",
            gap_units="log_likelihood_weight",
            correction="none",
            weight_step_natural_log=None,
            references=(),
        )

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

    def charge_soft_output(self, job, ticks):
        """Charge the signal's own computation on the job's unit."""
        job.soft_output_ticks = ticks
        self.charges.append((job, ticks))

    def close_companion_request(self, job, result):
        """The solve the window's answer did not come from."""
        self.closed.append((job, result))


def _join_with(signal_ticks: int):
    engine = engine_module.Engine()
    signal = _CostingSignal(signal_ticks)
    verdict = _RecordingVerdict(engine)
    queue = _RecordingQueue()
    join = gap_join_module.WindowGapJoin(engine, signal, verdict, queue)
    return engine, join, verdict, queue


def _solve_job():
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, label="w0"
    )


def test_the_signals_own_computation_is_charged_and_the_answer_waits():
    """D8: the walk is the unit's time, and the window waits for it."""
    engine, join, verdict, queue = _join_with(70)
    job = _solve_job()
    result = decoding_records.DecodeResult(1, 0)
    join.accept_result(job, result)
    assert queue.charges == [(job, 70)]
    assert job.soft_output_ticks == 70
    assert verdict.answers == []
    engine.run()
    (answered_job, answered_result, tick) = verdict.answers[0]
    assert answered_job is job
    assert answered_result.soft_output.gap == 1.0
    assert tick == 70


def test_a_signal_that_only_subtracts_charges_nothing_and_answers_at_once():
    """The complementary gap's combine is a subtraction (decision D8)."""
    engine, join, verdict, queue = _join_with(0)
    job = _solve_job()
    result = decoding_records.DecodeResult(1, 0)
    join.accept_result(job, result)
    assert queue.charges == [(job, 0)]
    assert job.soft_output_ticks == 0
    assert len(verdict.answers) == 1
    assert engine.now == 0


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
    """C5 item 1: two solves, and the ticks go where a service still closes.

    The answering solve is the lightest, which is not in general the last
    to arrive. decode_outcomes.deliver_weak ends a service at
    now + job.soft_output_ticks for the job it is delivering, which is
    the last one, so charging the lightest solve wrote the walk onto a
    service record that had already closed. The complementary gap's card
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
