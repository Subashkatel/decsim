"""Round conservation on the QEC cycle clock (QPU-001 / QPU-004).

The property: for every patch and every cycle boundary at which that patch
is live, exactly one round is accounted for, as an operation round, an
idle round, or a cycle of a non-emitting body. The failure this protects
against is the discrete-event engine jumping from tick A to tick B and
silently dropping one or more rounds in between.

Randomized schedules are generated with seeded stdlib random rather than
Hypothesis: the pinned run environment does not carry Hypothesis, and the
declared-tick gate treats new dependencies as a behavior-affecting change.
Each seed is fully deterministic. The expected tick sequence comes from an
independent reference walker below that encodes only the documented
semantics (cycle_clock.py module docstring and tests above), never the
device's own code.

Reference semantics encoded by the walker:
- an operation issued in (S - cycle, S] starts at boundary S and emits its
  round for cycle k at S + k*cycle (silently occupying the cycles instead
  when it does not emit detector data);
- after its last round at C = S + rounds*cycle the patch idles, emitting
  one idle round per boundary from C + cycle through the next operation's
  start boundary inclusive;
- finish() lets every idle patch emit through the first boundary at or
  after the finish tick, then stops.
"""

import random

import pytest

from decsim.engine import Engine
from decsim.message import Operation, RunOperationBody
from decsim.qpu.cycle_clock import QPUDevice
from decsim.qpu.syndrome_devices import TimingOnlyDevice


class _Capture:
    def __init__(self, engine):
        self.engine = engine
        self.rounds = []          # (tick, op_id, patch, round_index)
        self.idle = []            # (tick, patch)
        self.done = []            # (tick, op_id)

    def accept_qpu_readout(self, payload, route):
        self.rounds.append((self.engine.now, payload.operation_id,
                            payload.patch_id, payload.round_index))


def _qpu(cycle):
    engine = Engine(verbose=False)
    capture = _Capture(engine)
    qpu = QPUDevice(
        engine, TimingOnlyDevice(), cycle, readout_receiver=capture,
        completion_receiver=lambda op: capture.done.append((engine.now, op.id)),
        idle_receiver=lambda op_id, patch, k: capture.idle.append(
            (engine.now, patch)))
    return engine, qpu, capture


def _operation(op_id, patch, emits=True):
    return Operation(id=op_id, name=f"op{op_id}", qubits=(patch,),
                     patches=(patch,), emits_detector_data=emits)


def _random_schedule(rng):
    """Per patch: a list of (op_id, start_boundary, rounds, emits)."""
    cycle = rng.choice([1, 2, 7, 10, 1000])
    schedule = {}
    op_id = 0
    for patch in range(rng.randint(1, 4)):
        ops = []
        boundary = rng.randint(0, 6) * cycle
        for _ in range(rng.randint(1, 4)):
            op_id += 1
            rounds = rng.randint(1, 6)
            emits = rng.random() > 0.15
            ops.append((op_id, boundary, rounds, emits))
            gap_cycles = rng.randint(0, 5)
            boundary += (rounds + gap_cycles) * cycle
        schedule[patch] = ops
    return cycle, schedule


def _expected(cycle, schedule, end_boundary):
    """The reference walker: expected per-patch round and idle ticks."""
    expected_rounds = {}
    expected_idle = {}
    for patch, ops in schedule.items():
        round_ticks = []
        idle_ticks = []
        for index, (op_id, start, rounds, emits) in enumerate(ops):
            if emits:
                round_ticks += [(start + k * cycle, op_id, k)
                                for k in range(1, rounds + 1)]
            completion = start + rounds * cycle
            next_start = (ops[index + 1][1] if index + 1 < len(ops)
                          else end_boundary)
            idle_ticks += list(range(completion + cycle, next_start + 1, cycle))
        expected_rounds[patch] = round_ticks
        expected_idle[patch] = idle_ticks
    return expected_rounds, expected_idle


def _run_schedule(cycle, schedule, rng):
    engine, qpu, capture = _qpu(cycle)
    last_completion = 0
    for patch, ops in schedule.items():
        for op_id, start, rounds, emits in ops:
            operation = _operation(op_id, patch, emits=emits)
            body = RunOperationBody(operation, cycle, rounds, rounds,
                                    emits_detector_data=emits)
            issue_tick = rng.randint(max(0, start - cycle + 1), start)
            if issue_tick == 0:
                qpu.issue(body)
            else:
                engine.schedule(issue_tick, lambda b=body: qpu.issue(b))
            last_completion = max(last_completion, start + rounds * cycle)
    # finish after everything; off-boundary when the cycle allows it, so
    # the end rule stays the same whichever event order the tick resolves
    finish_tick = last_completion + rng.randint(0, 3) * cycle \
        + (0 if cycle == 1 else rng.randint(1, cycle - 1))
    engine.schedule(finish_tick, qpu.finish)
    # unrelated same-tick and mid-cycle events: they must not perturb rounds
    for _ in range(rng.randint(0, 5)):
        engine.schedule(rng.randint(0, max(finish_tick - 1, 1)),
                        lambda: None, label="noise")
    engine.run()
    end_boundary = (finish_tick if finish_tick % cycle == 0
                    else (finish_tick // cycle + 1) * cycle)
    return capture, end_boundary


@pytest.mark.parametrize("seed", range(20))
def test_every_live_cycle_is_accounted_for(seed):
    """Rounds + idle ticks per patch match the reference walker exactly:
    contiguous at the cycle stride, no duplicates, no skipped boundary."""
    rng = random.Random(seed)
    cycle, schedule = _random_schedule(rng)
    capture, end_boundary = _run_schedule(cycle, schedule, rng)
    expected_rounds, expected_idle = _expected(cycle, schedule, end_boundary)

    observed_rounds = {patch: [] for patch in schedule}
    for tick, op_id, patch, round_index in capture.rounds:
        observed_rounds[patch].append((tick, op_id, round_index))
    observed_idle = {patch: [] for patch in schedule}
    for tick, patch in capture.idle:
        observed_idle[patch].append(tick)

    for patch in schedule:
        assert observed_rounds[patch] == expected_rounds[patch], \
            f"seed {seed} patch {patch} cycle {cycle}: operation rounds diverge"
        assert observed_idle[patch] == expected_idle[patch], \
            f"seed {seed} patch {patch} cycle {cycle}: idle rounds diverge"
        # the conservation statement itself: one event per live boundary,
        # contiguous at the cycle stride from first activity to the end
        all_ticks = sorted([tick for tick, _, _ in observed_rounds[patch]]
                           + observed_idle[patch])
        silent = {start + k * cycle
                  for _, start, rounds, emits in schedule[patch]
                  if not emits for k in range(1, rounds + 1)}
        covered = sorted(set(all_ticks) | silent)
        assert len(covered) == len(all_ticks) + len(silent), \
            f"seed {seed} patch {patch}: duplicate accounting for one boundary"
        for earlier, later in zip(covered, covered[1:]):
            assert later - earlier == cycle, \
                (f"seed {seed} patch {patch} cycle {cycle}: boundary skipped "
                 f"between {earlier} and {later}")


def test_long_event_jump_cannot_skip_rounds():
    """One patch, cycle 1000, sparse unrelated events far apart: the engine
    jumps thousands of ticks between events and every boundary still lands."""
    cycle = 1000
    engine, qpu, capture = _qpu(cycle)
    operation = _operation(1, 0)
    qpu.issue(RunOperationBody(operation, cycle, 3, 3))
    engine.schedule(7777, lambda: None, label="noise")   # mid-cycle
    engine.schedule(10500, qpu.finish)
    engine.run()

    assert [tick for tick, _, _, _ in capture.rounds] == [1000, 2000, 3000]
    assert [tick for tick, _ in capture.idle] == \
        [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]


def test_non_emitting_body_occupies_cycles_without_rounds():
    """A non-emitting body holds its patch silently; idle extraction and
    detector rounds resume with no skipped or extra boundary."""
    cycle = 10
    engine, qpu, capture = _qpu(cycle)
    qpu.issue(RunOperationBody(_operation(1, 0), cycle, 2, 2))
    silent = _operation(2, 0, emits=False)
    engine.schedule(25, lambda: qpu.issue(
        RunOperationBody(silent, cycle, 3, 3, emits_detector_data=False)))
    emitter = _operation(3, 0)
    engine.schedule(65, lambda: qpu.issue(RunOperationBody(emitter, cycle, 1, 1)))
    engine.schedule(85, qpu.finish)
    engine.run()

    assert [(t, op) for t, op, _, _ in capture.rounds] == \
        [(10, 1), (20, 1), (80, 3)]
    assert [t for t, _ in capture.idle] == [30, 70, 90]
    assert capture.done == [(20, 1), (60, 2), (80, 3)]


def test_multi_patch_operation_attributes_rounds_to_its_first_patch():
    """MODEL CHOICE, characterized: TimingOnlyDevice emits one merged round
    per cycle attributed to the operation's first patch; both patches
    idle-extract individually after the body completes."""
    cycle = 10
    engine, qpu, capture = _qpu(cycle)
    merge = Operation(id=1, name="merge", qubits=(0, 1), patches=(0, 1))
    qpu.issue(RunOperationBody(merge, cycle, 2, 2))
    engine.schedule(45, qpu.finish)
    engine.run()

    assert [(t, patch) for t, _, patch, _ in capture.rounds] == [(10, 0), (20, 0)]
    assert sorted(capture.idle) == [(30, 0), (30, 1), (40, 0), (40, 1),
                                    (50, 0), (50, 1)]
