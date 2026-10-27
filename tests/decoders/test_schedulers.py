"""The ready-queue rules: FIFO and VQD's three, on hand-made queues.

VQD (Maurya and Tannu 2406.17995v2 Sec. 4.2) serves the critical
decodes first, then the rest by MLS, RR or MFD; the authors' simulator
(satvikmaurya/decoder-resources, scripts/sim.py, Sim.schedule) puts any
other multi-patch measurement between them. The row tests drain a queue in
the rule's order, as the dispatcher does.

The timeline runs four memories, mem1 to mem4 on patches 1 to 4, on the
declared fabric (tests/declared_run.py) with one unit of two input
slots and no idle-region decodes; a fixup waits on mem4's decision, so
mem4's decodes are critical. Every W0's input is ready at 15 us. mem1
W0 takes the unit and mem2 W0 the second slot; each decode takes 10 us
after a 5 us input crossing, so they decode 20-30 and 30-40 us. At 30
us the slot frees with mem3 W0 and mem4 W0 waiting since 15 us; mem1
W1 still owes its boundary, which lands at 30.5 us.
"""

import pytest

import decsim.config as config
import decsim.controller.policies as idle_policies
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import decsim.parameters.kinds as kinds
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import tests.declared_run as declared_run

# op 4 waits on op 2's decision, so op 2's decode on patch 2 is critical
PROGRAM = (
    program_records.Operation(1, "mem1", (1,), patches=(1,)),
    program_records.Operation(2, "measure", (2,), patches=(2,)),
    program_records.Operation(3, "mem2", (2,), patches=(2,)),
    program_records.Operation(4, "fixup", (2,), patches=(2,), blocked_by=2),
    program_records.Operation(5, "merge", (3, 4), patches=(3, 4)),
    program_records.Operation(6, "mem3", (3,), patches=(3,)),
    program_records.Operation(7, "mem3_again", (3,), patches=(3,)),
)
VQD_ROWS = (
    schedulers.LongestUndecoded,
    schedulers.RoundRobin,
    schedulers.MostCritical,
)


def test_fifo_drains_in_admission_order():
    scheduler = schedulers.FifoScheduler()
    queue = [_job(1, ready=0), _job(3, ready=1), _job(2, ready=2)]

    order = _drained(scheduler, queue)

    assert order == [1, 3, 2]


def test_a_critical_decode_takes_its_patchs_backlog_ahead_of_the_policy():
    """Round robin would serve patch 1 first; patch 2 holds op 2.

    Op 3's older job on patch 2 is part of the backlog the critical
    decode needs, so it goes first, then op 2, then patch 1.
    """
    scheduler = _reading(schedulers.RoundRobin)
    queue = [_job(1, ready=0), _job(3, ready=1), _job(2, ready=2)]

    order = _drained(scheduler, queue)

    assert order == [3, 2, 1]


def test_a_multi_patch_job_goes_after_a_critical_one_and_before_the_rest():
    scheduler = _reading(schedulers.RoundRobin)
    queue = [_job(1, ready=0), _job(5, ready=1), _job(2, ready=2)]

    order = _drained(scheduler, queue)

    assert order == [2, 5, 1]


def test_longest_undecoded_serves_the_patch_with_most_rounds_waiting():
    """Patch 3 waits 6 rounds, patch 1 3: patch 3, then the older job."""
    scheduler = _reading(schedulers.LongestUndecoded)
    queue = [
        _job(1, ready=0),
        _job(6, ready=1),
        _job(7, ready=2),
    ]

    order = _drained(scheduler, queue)

    assert order == [6, 1, 7]


def test_round_robin_sends_a_served_patch_to_the_back_of_the_turns():
    scheduler = _reading(schedulers.RoundRobin)
    served = _job(1)
    scheduler.served(served)
    queue = [_job(1, ready=0), _job(6, ready=1)]

    order = _drained(scheduler, queue)

    assert order == [6, 1]


def test_most_critical_serves_the_patch_with_critical_decodes_ahead():
    """Patch 2 has op 2 ahead; once op 2 is served, the older job goes."""
    scheduler = _reading(schedulers.MostCritical)
    first_queue = [_job(1, ready=0), _job(3, ready=1)]
    ahead = _drained(scheduler, first_queue)
    critical = _job(2)
    scheduler.served(critical)
    second_queue = [_job(1, ready=0), _job(3, ready=1)]
    after = _drained(scheduler, second_queue)

    assert ahead == [3, 1]
    assert after == [1, 3]


def test_a_job_the_program_does_not_list_waits_with_the_policy():
    scheduler = _reading(schedulers.RoundRobin)
    batch = _job(-1, ready=0)
    queue = [batch, _job(2, ready=1)]

    order = _drained(scheduler, queue)

    assert order == [2, -1]


def test_a_vqd_row_refuses_a_stream_for_now():
    segment = program_records.Operation(
        1, "prefix", (1,), patches=(1,), stream_id="live", stream_offset=0
    )

    scheduler = schedulers.LongestUndecoded()
    program = (segment,)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        scheduler.read_program(program)


def test_each_vqd_row_is_written_by_its_name():
    offered = kinds.records_for(decoder_settings.SchedulerSettings)
    names = ["longest_undecoded", "round_robin", "most_critical"]

    found = []
    for name in names:
        matches = kinds.records_named(name, offered)
        found.append(len(matches))

    assert found == [1, 1, 1]


def test_fifo_decodes_the_critical_memory_last_of_the_four():
    """FIFO stages mem3 W0 at 30 us: mem4 W0 decodes 50-60 us.

    Its answer rides weak_to_frame (2 us), lands at 62 us and
    commits at 63 us; mem4 W1 lands at 102 us.
    """
    fifo = schedulers.FifoScheduler.Settings()
    machine = _four_memories(fifo)

    records = _records_by_window(machine)

    assert records[(3, 0)] == (_ticks(52.0), _ticks(53.0))
    assert records[(4, 0)] == (_ticks(62.0), _ticks(63.0))
    assert records[(4, 1)] == (_ticks(102.0), _ticks(103.0))


def test_every_vqd_row_decodes_the_critical_memory_first():
    """A VQD row stages mem4 W0 at 30 us: it decodes 40-50 us.

    It lands at 52 us and commits at 53 us; mem3 W0 waits to 62 us. At
    60 us mem4 W1 is again the critical job waiting: it decodes 70-80
    us and lands at 82 us.
    """
    for row in VQD_ROWS:
        rule = row.Settings()
        machine = _four_memories(rule)
        records = _records_by_window(machine)
        assert records[(4, 0)] == (_ticks(52.0), _ticks(53.0))
        assert records[(3, 0)] == (_ticks(62.0), _ticks(63.0))
        assert records[(4, 1)] == (_ticks(82.0), _ticks(83.0))


def _four_memories(rule) -> machine_module.Machine:
    """The declared weak-only machine over mem1 to mem4 and a fixup, run."""
    operations = []
    for stream_id in range(1, 5):
        memory = declared_run.memory_operation(stream_id)
        operations.append(memory)
    fixup = program_records.Operation(
        id=5,
        name="fixup",
        qubits=(4,),
        patches=(4,),
        predecessors=(4,),
        blocked_by=4,
    )
    operations.append(fixup)
    graph = declared_run.weak_only_graph(rounds=9, operations=operations)
    manager = decoder_settings.DecoderManagerSettings(scheduler=rule)
    ignore_idle = idle_policies.IgnoreSettings()
    graph = graph.with_held("decoder_manager", manager)
    graph = graph.with_held("idle_policy", ignore_idle)
    return declared_run.run_machine(graph)


def _records_by_window(machine: machine_module.Machine) -> dict:
    """Each window's landing and commit ticks."""
    snapshot = machine.control.pauli_frame.snapshot()
    records = {}
    for record in snapshot.records:
        records[record.window_key] = (
            record.accepted_ticks,
            record.committed_ticks,
        )
    return records


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)


def _reading(scheduler_class: type):
    """A fresh scheduler of the class, having read the test's program."""
    scheduler = scheduler_class()
    scheduler.read_program(PROGRAM)
    return scheduler


def _drained(scheduler, queue: list) -> list:
    """Each job's operation id, in the order the rule pops them."""
    order = []
    while queue:
        job = scheduler.pop(queue)
        order.append(job.stream_id)
    return order


def _job(stream_id: int, ready: int = 0) -> decoding_records.DecodeJob:
    """A three-round job of the operation, admitted at tick ready."""
    return decoding_records.DecodeJob(
        stream_id=stream_id,
        window_id=0,
        round_count=3,
        ready_time=ready,
    )
