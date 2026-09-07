"""QPUDevice runs one QEC cycle clock: operations start on cycle boundaries and
every live patch yields one round per cycle, idle or not (SWIPER
device_manager._generate_syndrome_round; Google 2207.06431 readout every cycle)."""

from decsim.engine import Engine
from decsim.message import Operation, RunOperationBody
from decsim.observe.command_events import CommandEvents
from decsim.qpu.cycle_clock import QPUDevice
from decsim.qpu.syndrome_devices import TimingOnlyDevice

CYCLE = 10


class _Capture:
    def __init__(self):
        self.rounds = []  # (tick, op_id, patch, round_index)
        self.idle = []  # (tick, op_id, patch, round_index)
        self.done = []  # (tick, op_id)

    def accept_qpu_readout(self, payload, route):
        self.rounds.append(
            (
                self.engine.now,
                payload.operation_id,
                payload.patch_id,
                payload.round_index,
            )
        )


def _qpu():
    engine = Engine()
    capture = _Capture()
    capture.engine = engine
    qpu = QPUDevice(
        engine,
        TimingOnlyDevice(),
        CYCLE,
        readout_receiver=capture,
        completion_receiver=lambda op: capture.done.append((engine.now, op.id)),
        idle_receiver=lambda op_id, patch, k: capture.idle.append(
            (engine.now, op_id, patch, k)
        ),
    )
    return engine, qpu, capture


def _op(op_id, patch, rounds=2, **changes):
    return Operation(
        id=op_id,
        name=f"op{op_id}",
        qubits=(patch,),
        patches=(patch,),
        **changes,
    ), rounds


def _issue(qpu, op, rounds, **changes):
    qpu.issue(RunOperationBody(op, CYCLE, rounds, rounds, **changes))


def test_operations_start_on_the_next_cycle_boundary_and_share_one_clock():
    engine, qpu, capture = _qpu()
    a, ra = _op(1, "A")
    b, rb = _op(2, "B")
    _issue(qpu, a, ra)
    engine.schedule(
        13, lambda: _issue(qpu, b, rb)
    )  # issued mid-cycle: starts at 20
    engine.schedule(41, qpu.finish)
    engine.run()

    assert capture.rounds == [
        (10, 1, "A", 1),
        (20, 1, "A", 2),
        (30, 2, "B", 1),
        (40, 2, "B", 2),
    ]
    assert capture.done == [(20, 1), (40, 2)]


def test_an_idle_patch_keeps_extracting_every_cycle_until_its_next_operation():
    engine, qpu, capture = _qpu()
    a, ra = _op(1, "A")
    c, rc = _op(3, "A", rounds=1)
    _issue(qpu, a, ra)  # rounds at 10, 20; done at 20
    engine.schedule(
        35, lambda: _issue(qpu, c, rc)
    )  # starts at 40, its round at 50
    engine.schedule(50, qpu.finish)
    engine.run()

    assert capture.idle == [
        (30, 1, "A", 1),
        (40, 1, "A", 2),
    ]  # cycles 20-30 and 30-40 idle
    assert capture.rounds == [(10, 1, "A", 1), (20, 1, "A", 2), (50, 3, "A", 1)]


def test_finish_stops_idle_extraction_after_the_current_cycle():
    engine, qpu, capture = _qpu()
    a, ra = _op(1, "A")
    _issue(qpu, a, ra)
    engine.schedule(25, qpu.finish)  # idle round of cycle 20-30 still lands
    engine.run()

    assert capture.idle == [(30, 1, "A", 1)]
    assert engine.now == 30


def test_a_body_without_detector_data_holds_its_patch_for_whole_cycles():
    engine, qpu, capture = _qpu()
    t, rt = _op(1, "A", rounds=3, emits_detector_data=False)
    _issue(qpu, t, rt, emits_detector_data=False)
    engine.schedule(30, qpu.finish)
    engine.run()

    assert capture.rounds == []
    assert capture.done == [(30, 1)]
    assert capture.idle == []


def test_operation_cadence_must_equal_the_qpu_cycle():
    engine, qpu, capture = _qpu()
    a, ra = _op(1, "A")
    try:
        qpu.issue(RunOperationBody(a, CYCLE + 1, ra, ra))
    except ValueError as error:
        assert "cadence" in str(error)
    else:
        raise AssertionError("mismatched cadence accepted")


def test_a_command_starts_on_the_cycle_boundary_at_or_after_its_arrival():
    """Operations are cycle-locked: a command reaching the QPU starts on the
    first boundary not earlier than its arrival, the boundary itself
    included (QubiC's pulse timestamp is the time after which the pulse
    plays, arXiv 2404.15260 Sec. IV; SWIPER starts instructions on the
    next round)."""
    from decsim.engine import Engine
    from decsim.message import Operation, RunOperationBody
    from decsim.qpu.cycle_clock import QPUDevice, _IdlePatch

    class SilentModel:
        def begin_operation(self, operation, rounds, source_rounds):
            pass

        def round_payloads(self, operation, round_index):
            return []

    cycle = 100

    def start_tick(arrival):
        engine = Engine()
        qpu = QPUDevice(
            engine,
            SilentModel(),
            cycle,
            completion_receiver=lambda operation: None,
            idle_receiver=lambda *_: None,
        )
        operation = Operation(
            id=1,
            name="op",
            qubits=(0,),
            patches=(0,),
            emits_detector_data=False,
        )
        command = RunOperationBody(
            operation=operation,
            round_ticks=cycle,
            round_count=2,
            source_round_count=2,
            emits_detector_data=False,
            finalizes_stream_round=False,
        )
        commands = CommandEvents()
        qpu.command_event.connect(commands.command_event)
        # an idle patch keeps the clock ticking
        qpu._idle_by_patch[9] = _IdlePatch(0, 0)
        engine.schedule(arrival, lambda: qpu.issue(command))
        engine.schedule(arrival + 5 * cycle, qpu.finish)
        engine.run()
        return dict((event.kind, event.tick) for event in commands.events)[
            "STARTED"
        ]

    for arrival in (0, 1, 99, 100, 101, 250, 300):
        assert start_tick(arrival) == qpu_boundary(arrival, cycle)


def qpu_boundary(tick, cycle):
    return tick if tick % cycle == 0 else (tick // cycle + 1) * cycle
