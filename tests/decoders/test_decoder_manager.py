"""The facade's laws: a job is admitted once, withdrawn whole, and settled.

A withdrawn window leaves the queue and gives its claim back; a tier
that blocks holds its unit until the result is read (Chen 2605.30765);
the copy and in-place rows put the rounds where the unit reads them.
A new row through the pool is tests/machine/test_machine.py::
test_a_new_decoder_is_one_class_and_its_settings_record.

A cancelled member of a merged strong batch leaves the rest their
batch. The batch timelines run four memories on the declared switching
fabric (tests/escalation/declared_fabric.py), a weak unit each, one
strong unit of two input slots decoding each batch member in 30 us,
and a 10 us deadline whose late weak result meets the verdict
(test_deadlines.py): every W0 is rescued at 25 us, and its rounds
reach the strong side at 28 us. The unit takes mem1 W0, its input
landing at 34 us, and stages mem2 W0; mem3 W0 and mem4 W0 wait as one
batch. At 30 us each weak result meets its verdict, mem1's first: a
kept one cancels its rescue, a doubted one selects it.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.decoders.batching as batching
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoder_manager as decoder_manager
import decsim.decoders.decoder_pool as decoder_pool
import decsim.decoders.decoders as decoders
import decsim.decoders.detection_events as detection_events
import decsim.decoders.schedulers as schedulers
import decsim.decoders.strong_requests as strong_requests_module
import decsim.detector_error_model.detection_event_formation as event_formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.escalation.deadlines as deadlines
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.decoding as decoding_records
import decsim.records.formation as formation_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as declared_fabric


class FixedRow(decoder_module.DecoderBase):
    """A row for the plug-in law: two microseconds, one fixed observable."""

    def latency(self, job):
        del job
        return config.microseconds_to_ticks(2.0)

    def decode(self, job):
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=(1,)
        )


class OneRoundSource:
    """Recipes of a one-round operation: the value is not the law."""

    def formation_table(self, operation_id):
        """The one table: one event, the round's first outcome."""
        del operation_id
        recipe = formation_records.DetectorRecipe(
            detector_index=0,
            round_index=1,
            kind=formation_records.LayerKind.PREPARATION,
            records=((1, 0),),
            reference_parity=0,
            coordinates=(),
        )
        return formation_records.FormationTable(
            round_count=1,
            packet_width_by_round={1: 2},
            readout_slot_start=None,
            detectors=(recipe,),
            observables=(),
        )


class TwoRoundSource:
    """Recipes of a two-round operation: round 2 compares against round 1."""

    def formation_table(self, operation_id):
        """One event a round, the second the XOR of both rounds' outcome."""
        del operation_id
        first = formation_records.DetectorRecipe(
            detector_index=0,
            round_index=1,
            kind=formation_records.LayerKind.PREPARATION,
            records=((1, 0),),
            reference_parity=0,
            coordinates=(),
        )
        second = formation_records.DetectorRecipe(
            detector_index=1,
            round_index=2,
            kind=formation_records.LayerKind.BULK,
            records=((1, 0), (2, 0)),
            reference_parity=0,
            coordinates=(),
        )
        return formation_records.FormationTable(
            round_count=2,
            packet_width_by_round={1: 1, 2: 1},
            readout_slot_start=None,
            detectors=(first, second),
            observables=(),
        )


def _manager(engine, row, formation=None, unit_count=1):
    scheduler = schedulers.FifoScheduler()
    pool_settings = decoder_pool.PoolSettings(
        name="weak", unit_count=unit_count, formation=formation
    )
    manager = decoder_manager.DecoderManager(
        engine,
        scheduler=scheduler,
        pool_settings=pool_settings,
        pool_id=window_records.WEAK_POOL_ID,
    )
    manager.strong_requests = strong_requests_module.StrongRequests(
        {window_records.WEAK_POOL_ID: window_records.STRONG_POOL_ID}
    )
    manager.decoder = row
    return manager


def _window_job():
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=("p",),
        round_index=1,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.WEAK_POOL_ID, 0
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=1,
        payloads=[payload],
        label="mem W0",
        request_key=request_key,
    )


def _resolving(manager):
    """An on_decoded that closes the request, as the window side does."""
    keep = decoding_records.Verdict.KEEP

    def on_decoded(job, result):
        manager.resolve_weak_request(job, result, keep)

    return on_decoded


def test_a_decoder_row_that_declares_statistics_records_them():
    """A row of one's own declares its statistics and needs no wiring."""
    engine = engine_module.Engine()
    counting = _CountingDecoder(1.0)
    plain = decoders.PresetLatencyDecoder(1.0)
    counting_manager = _manager(engine, counting)
    plain_manager = _manager(engine, plain)
    run_statistics = statistics_registry.Registry(engine)

    counting_manager.attach_statistics(run_statistics, "weak_decoder")
    plain_manager.attach_statistics(run_statistics, "strong_decoder")

    groups = run_statistics.groups
    assert groups["weak_decoder.algorithm"] is counting.statistics
    assert "strong_decoder.algorithm" not in groups


def test_a_spent_job_is_refused():
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _manager(engine, row)
    job = _window_job()
    manager.enqueue(job, None, lambda _job, _result: None)
    with pytest.raises(RuntimeError, match="has already been admitted"):
        manager.enqueue(job, None, lambda _job, _result: None)


def test_a_withdrawn_window_leaves_the_queue_and_the_ledger():
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _manager(engine, row)
    busy = _window_job()
    busy.label = "busy"
    on_decoded = _resolving(manager)
    manager.enqueue(busy, None, on_decoded)
    waiting = _window_job()
    waiting.window_id = 1
    waiting.label = "waiting"
    waiting.request_key = window_records.DecoderRequestKey(
        1, 1, window_records.WEAK_POOL_ID, 1
    )
    manager.enqueue(waiting, None, on_decoded)
    assert manager.queue.waiting == []  # both took a slot of the one unit
    manager.withdraw_window((1, 1))
    assert waiting.cancelled is True
    assert waiting.unit is None
    engine.run()
    manager.check_decode_work_settled()


def test_a_withdrawn_windows_claim_goes_back_to_its_tier():
    """A decode that never started was never charged for forming them.

    Each job claims its rounds when it is staged. The waiting window
    reads a second operation's round, which no earlier job claimed, so
    its claim is the round. A strong region then absorbs the window and
    the job is withdrawn before its decode starts, so the job that reads
    that round next has to pay for forming it.
    """
    engine = engine_module.Engine()
    row = FixedRow()
    former_clock = config.Clock(4000)
    at_the_weak_decoder = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",),
        clock=former_clock,
        latency_cycles=5,
        cycles_per_round=1,
    )
    source = OneRoundSource()
    placement = event_formation.SeatedFormation(source, at_the_weak_decoder)
    formation = detection_events.TierFormation(placement, "weak_decoder")
    manager = _manager(engine, row, formation)
    formation_stage = detection_events.DetectionEventFormationStage(
        "detection_event_formation", formation=formation
    )
    busy = _window_job()
    busy.label = "busy"
    on_decoded = _resolving(manager)
    manager.enqueue(busy, None, on_decoded)
    waiting = _second_operations_job()
    waiting.window_id = 1
    waiting.label = "waiting"
    waiting.request_key = window_records.DecoderRequestKey(
        2, 1, window_records.WEAK_POOL_ID, 1
    )
    manager.enqueue(waiting, None, on_decoded)
    at_the_prediction = formation_stage.cycles_for(waiting)

    manager.withdraw_window((2, 1))

    strong = _second_operations_job()
    strong.label = "strong"
    after_the_withdrawal = formation_stage.cycles_for(strong)
    assert at_the_prediction == 5
    assert after_the_withdrawal == 5
    engine.run()
    manager.check_decode_work_settled()


def test_the_rounds_before_are_held_by_the_tier_and_not_deposited():
    """The unit's memory takes the job's rounds; the tier's logic the rest."""
    engine = engine_module.Engine()
    at_the_weak_decoder = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",)
    )
    source = TwoRoundSource()
    placement = event_formation.SeatedFormation(source, at_the_weak_decoder)
    formation = detection_events.TierFormation(placement, "weak_decoder")
    row = FixedRow()
    manager = _manager(engine, row, formation)
    deposited = []

    def note_deposit(job, *_) -> None:
        landed = job.decoder_input.fragments()
        deposited.extend(landed)

    copy_sources = manager.copy_sources()
    _connect_each(copy_sources, note_deposit)
    job = _window_job()
    round_one = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=("p",),
        round_index=1,
        bits=(1,),
        size_bits=1,
        fragment_index=0,
    )
    round_two = dataclasses.replace(round_one, round_index=2)
    job.payloads = [round_two]
    job.rounds_before = (round_one,)
    on_decoded = _resolving(manager)

    manager.enqueue(job, None, on_decoded)
    engine.run()

    assert [fragment.round_index for fragment in deposited] == [2]
    assert deposited[0].bits == (0,)
    assert job.rounds_before == ()


def test_every_job_kind_says_how_it_is_settled():
    """The settle table is read at every completion, so it is closed at import.

    A kind with no settle row would raise a KeyError inside the decode
    completion, after the decoder has already run, rather than at the
    table that is missing the row.
    """
    kinds = set(decoding_records.DecodeJobKind)
    settled = set(decoder_manager.SETTLE_BY_JOB_KIND)
    assert settled == kinds


def _blocking_manager(engine, row, *, blocks_unit: bool):
    """A one-unit manager whose default pool blocks, or does not."""
    scheduler = schedulers.FifoScheduler()
    pool_settings = decoder_pool.PoolSettings(
        name="weak", unit_count=1, blocks_unit=blocks_unit
    )
    manager = decoder_manager.DecoderManager(
        engine,
        scheduler=scheduler,
        pool_settings=pool_settings,
        pool_id=window_records.WEAK_POOL_ID,
    )
    manager.strong_requests = strong_requests_module.StrongRequests(
        {window_records.WEAK_POOL_ID: window_records.STRONG_POOL_ID}
    )
    manager.decoder = row
    return manager


def test_a_tier_that_does_not_block_frees_its_unit_at_the_decodes_end():
    """R4, false: Chen 2605.30765 lines 1618-1620, the streaming regime.

    "In the streaming regime, it appends each decode result to the frame
    without blocking the pipeline."
    """
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _blocking_manager(engine, row, blocks_unit=False)
    first = _window_job()
    second = _window_job()
    second.window_id = 1

    resolve = _resolving(manager)

    manager.enqueue(first, None, resolve)
    manager.enqueue(second, None, resolve)
    engine.run()

    assert first.completed is True
    assert second.completed is True


def test_a_tier_that_blocks_holds_its_unit_until_the_result_is_read():
    """R4, true: Caune 2410.05202 lines 1256-1259, the polled register.

    "A write instruction to the decoder initiates decoding, followed by a
    polling of the decoder's status register, which stalls the program
    until decoding completes." The second job cannot start while the
    first result is unread, and starts the moment it is read.
    """
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _blocking_manager(engine, row, blocks_unit=True)
    first = _window_job()
    second = _window_job()
    second.window_id = 1
    decoded = []

    def note(job, result):
        del result
        decoded.append(job)

    manager.enqueue(first, None, note)
    manager.enqueue(second, None, note)
    engine.run()
    held_after_the_first_decode = list(decoded)
    manager.read_result(first)
    engine.run()

    assert held_after_the_first_decode == [first]
    assert decoded == [first, second]


def test_reading_a_result_of_a_tier_that_never_blocked_gives_nothing_back():
    """The unit went back at the decode's end, so there is nothing here."""
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _blocking_manager(engine, row, blocks_unit=False)
    job = _window_job()

    resolve = _resolving(manager)

    manager.enqueue(job, None, resolve)
    engine.run()
    manager.read_result(job)
    engine.run()

    assert job.completed is True


def test_the_copy_row_deposits_the_rounds_in_the_units_own_memory():
    """R9, copy: Collision Clustering 2309.05558 lines 268-271.

    "the Init unit (Fig. 2a) processes the decoder configuration, loads
    the input syndrome data and appropriate data into the stor[age
    elements]".
    """
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _staging_manager(engine, row, copies_input=True)
    copies, references = _count_movements(manager)
    job = _window_job()

    resolve = _resolving(manager)

    manager.enqueue(job, None, resolve)
    engine.run()

    assert copies == [job]
    assert references == []


def test_the_in_place_row_reads_the_rounds_where_the_store_keeps_them():
    """R9, in_place: AFS 2001.06598 lines 528-530.

    "For our specialized hardware, the processing elements can directly
    access the data stored on-chip", so nothing is deposited in the
    unit's memory and nothing crosses the input link.
    """
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _staging_manager(engine, row, copies_input=False)
    copies, references = _count_movements(manager)
    job = _window_job()

    resolve = _resolving(manager)

    manager.enqueue(job, None, resolve)
    engine.run()

    assert copies == []
    assert references == [job]
    assert job.memory is None


def _count_movements(manager):
    """Two lists: the jobs whose input was copied, and those referenced."""
    copies = []
    references = []

    def note_copy(job, bits, source_name, target_name):
        del bits, source_name, target_name
        copies.append(job)

    def note_reference(job, round_keys):
        del round_keys
        references.append(job)

    for source in manager.copy_sources():
        source.connect(note_copy)
    for source in manager.reference_sources():
        source.connect(note_reference)
    return copies, references


def _connect_each(sources, listener) -> None:
    for source in sources:
        source.connect(listener)


def _lines_containing(lines: list, text: str) -> list:
    matching = []
    for line in lines:
        if text in line:
            matching.append(line)
    return matching


def _enqueue_timed_jobs(manager, decode_microseconds, send_input, on_decoded):
    """Job J<n> decodes for decode_microseconds[n]; all ready at tick 0."""
    for operation_id, microseconds in enumerate(decode_microseconds):
        request_key = window_records.DecoderRequestKey(
            operation_id, 0, window_records.WEAK_POOL_ID, 0
        )
        job = decoding_records.DecodeJob(
            operation_id=operation_id,
            window_id=0,
            round_count=microseconds,
            label=f"J{operation_id}",
            ready_time=0,
            request_key=request_key,
        )
        manager.enqueue(job, send_input, on_decoded)


def _staging_manager(engine, row, *, copies_input: bool):
    """A one-unit manager whose default pool copies its input, or does not."""
    scheduler = schedulers.FifoScheduler()
    pool_settings = decoder_pool.PoolSettings(
        name="weak", unit_count=1, copies_input=copies_input
    )
    manager = decoder_manager.DecoderManager(
        engine,
        scheduler=scheduler,
        pool_settings=pool_settings,
        pool_id=window_records.WEAK_POOL_ID,
    )
    manager.strong_requests = strong_requests_module.StrongRequests(
        {window_records.WEAK_POOL_ID: window_records.STRONG_POOL_ID}
    )
    manager.decoder = row
    return manager


class RoundsRow(decoder_module.DecoderBase):
    """A row whose decode takes one microsecond a round."""

    def latency(self, job):
        return config.microseconds_to_ticks(job.round_count)

    def decode(self, job):
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


def test_a_startable_job_with_input_starts_when_a_central_fifo_queue_would():
    """Kiefer and Wolfowitz's (1955) FIFO queue over c identical servers.

    Job n starts at max(its arrival, the earliest tick a server is
    free), and that server is then busy for the job's decode. Two units
    take decodes of 10, 1, 1 and 1 microseconds, all queued at tick 0:
    the fourth starts at 2 on the unit that ran the short ones, not at
    10 on the unit that has room for it at admission.
    """
    engine = engine_module.Engine()
    row = RoundsRow()
    manager = _manager(engine, row, unit_count=2)
    start_by_label = {}

    def note_start(job, unit):
        del unit
        start_by_label[job.label] = engine.now

    manager.service.trace.job_started.connect(note_start)

    def send_input(on_landed):
        engine.schedule(0, on_landed)
        return 0

    def ignore(job, result):
        del job, result

    decode_microseconds = (10, 1, 1, 1)
    _enqueue_timed_jobs(manager, decode_microseconds, send_input, ignore)
    engine.run()

    starts = [start_by_label[f"J{index}"] for index in range(4)]
    expected_microseconds = (0, 0, 1, 2)
    expected = [config.microseconds_to_ticks(t) for t in expected_microseconds]
    assert starts == expected


def _second_operations_job():
    """The window job, reading round 1 of operation 2."""
    job = _window_job()
    job.operation_id = 2
    second = dataclasses.replace(job.payloads[0], operation_id=2)
    job.payloads = [second]
    return job


class _CountingDecoder(decoders.PresetLatencyDecoder):
    """A preset-latency row that declares a statistic of its own."""

    STATISTICS = (
        statistics_registry.Statistic(
            "iterations", statistic_kinds.Count, "solver iterations", "steps"
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)


def test_a_queued_batch_that_loses_a_member_serves_the_rest():
    """mem4 is kept and leaves the waiting batch; mem3 W0 decodes alone.

    mem1 W0 decodes 34-64 us and mem2 W0 64-94 us. At 64 us the freed
    slot opens the batch into mem3 W0 alone, whose input lands at 70 us;
    it decodes 94-124 us, lands at 128 us and commits at 129 us.
    """
    machine = _rescued_memories(doubted={1, 2, 3})

    records = _committed_by_window(machine)

    assert records[(3, 0)] == ("strong", _microseconds(129.0))
    assert records[(4, 0)][0] == "weak"


def test_a_staged_batch_that_loses_a_member_decodes_the_rest():
    """mem1's keep frees the unit, which stages the batch at 30 us.

    mem2's keep frees the other slot, and mem4's leaves the staged batch
    without its member. The inputs land at 36 us and the batch decodes
    mem3 W0 alone, 36-66 us; the answer lands at 70 us and commits at
    71 us.
    """
    machine = _rescued_memories(doubted={3})

    records = _committed_by_window(machine)

    assert records[(3, 0)] == ("strong", _microseconds(71.0))
    assert records[(4, 0)][0] == "weak"


def test_a_staged_batch_whose_members_all_leave_settles_every_window():
    """Every weak result is kept: no input of the batch lands after it.

    Each W0 commits its weak result at 33 us and each W1 at 48.5 us.
    """
    machine = _rescued_memories(doubted=set())

    records = _committed_by_window(machine)

    tiers = set()
    for tier, _ticks in records.values():
        tiers.add(tier)
    assert len(records) == 8
    assert tiers == {"weak"}


def _rescued_memories(doubted: set) -> machine_module.Machine:
    """The four rescued memories, those in doubted escalated, run."""
    deadline = deadlines.Deadline.Settings(
        slack_microseconds=10.0, late_source_result="revision"
    )
    switching = declared_run.declared_switching(deadline=deadline)
    graph = declared_fabric.switching_graph(
        rounds=6, escalated_windows=set(), switching=switching
    )
    operations = []
    for operation_id in range(1, 5):
        memory = declared_run.memory_operation(operation_id)
        operations.append(memory)
    graph_workload = graph.held("workload")
    workload = dataclasses.replace(graph_workload, operations=operations)
    is_escalated = _of_operations(doubted)
    weak_microseconds = declared_run.DECLARED_MICROSECONDS["weak"]
    weak_row = declared_run.DeclaredConfidenceDecoder.Settings(
        weak_microseconds, is_escalated
    )
    decode_each = batching.DecodeEach.Settings()
    graph = graph.with_held("workload", workload)
    graph = graph.with_settings(
        machines.WEAK_POOL, algorithm=weak_row, unit_count=4
    )
    graph = graph.with_settings(machines.STRONG_POOL, batching=decode_each)
    machine = machine_module.Machine.build(graph, 0)
    machine.run()
    return machine


def _of_operations(operation_ids: set):
    """Whether a job's operation is one of these."""

    def is_escalated(job) -> bool:
        return job.operation_id in operation_ids

    return is_escalated


def _committed_by_window(machine: machine_module.Machine) -> dict:
    """Each window's committing tier and commit tick."""
    snapshot = machine.control.pauli_frame.snapshot()
    records = {}
    for record in snapshot.records:
        records[record.window_key] = (record.tier, record.committed_ticks)
    return records


def _microseconds(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)
