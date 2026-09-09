"""The facade's laws: a row plugs in through the pool, a job is admitted once.

sinter's BUILT_IN_DECODERS
(sinter/_decoding/_decoding_all_built_in_decoders.py): a new decoder is
one class on the port and one row; here the row is routed by the pool
and its result reaches on_decoded once, with the manager's log naming
the job it started.
"""

import pytest

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoder_manager as decoder_manager
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.staged_decoder as staged_decoder
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.observe.log_writers as log_writers
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
from decsim.decoders.decoder_manager import DecoderManager


class FixedRow(decoder_module.DecoderBase):
    """A row for the plug-in law: two microseconds, one fixed observable."""

    def latency(self, job):
        del job
        return config.microseconds_to_ticks(2.0)

    def decode(self, job):
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=(1,)
        )


def _manager(engine, row):
    router = decoders.CodeRouter(row)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
    )


def _window_job():
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=1,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=1,
        payloads=[payload],
        label="mem W0",
        request_key=request_key,
    )


def test_a_fake_row_through_the_pool_decodes_the_window_once():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    row = FixedRow()
    manager = _manager(engine, row)
    delivered = []

    def on_decoded(job, result):
        delivered.append((engine.now, result))
        manager.resolve_weak_request(job, result, decoding_records.Verdict.KEEP)

    job = _window_job()
    manager.enqueue(job, None, on_decoded)
    engine.run()
    (delivery,) = delivered
    tick, result = delivery
    assert tick == config.microseconds_to_ticks(2.0)
    assert result.logical_observables == (1,)
    assert "Decoder manager: START DECODE mem W0" in log.lines[-1]
    manager.check_decode_work_settled()


def _resolving(manager):
    """An on_decoded that closes the request, as the window side does."""
    keep = decoding_records.Verdict.KEEP

    def on_decoded(job, result):
        manager.resolve_weak_request(job, result, keep)

    return on_decoded


def test_a_spent_job_is_refused():
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _manager(engine, row)
    job = _window_job()
    manager.enqueue(job, None, lambda _job, _result: None)
    with pytest.raises(RuntimeError, match="submitted once"):
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
        1, 1, window_records.DecoderTier.WEAK, 1
    )
    manager.enqueue(waiting, None, on_decoded)
    assert manager.queue.total() == 0  # both took a slot of the one unit
    manager.withdraw_window((1, 1))
    assert waiting.cancelled is True
    assert waiting.unit is None
    engine.run()
    manager.check_decode_work_settled()


def test_an_escalation_routed_to_a_pipelined_unit_is_refused():
    """The strong escalation tier is not pipelined yet, so it refuses.

    A pipelined route serves plain window and external decodes only; a
    strong re-decode, a gap sibling and a merged batch keep occupancy
    equal to latency until they get their own design pass, and routing
    one to a pipelined unit would silently serialize it instead of
    honoring the declared card (decode_service.py, _pipeline_of). The
    escalation request is the one a switching run submits: it names the
    destination window it re-decodes and asks for the strong pool.
    """
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=1.0)
    algorithm = FixedRow()
    strong = staged_decoder.StagedDecoder(algorithm, timing)
    weak = FixedRow()
    router = decoders.SwitchingRouter(weak=weak, strong=strong)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
    manager = DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        unit_pools={"default": 1, "strong": 1},
        escalation_policy=policy,
    )
    job = _window_job()
    job.kind = decoding_records.DecodeJobKind.STRONG_REDECODE
    job.strong_decode_for = (1, 0)
    job.request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 0
    )
    with pytest.raises(RuntimeError, match="not pipelined yet"):
        manager.enqueue(job, None, lambda _job, _result: None)


class UnpinnableRow(FixedRow):
    """A row that reports, at compile time, that no class can be forced."""

    def __init__(self) -> None:
        base = super()
        base.__init__()
        self.forced_solve_unavailable = trace_source.TraceSource()


class _Model:
    """The part of a window model the narrated line reads."""

    detector_ids = (0, 1, 2, 3, 4)


def test_the_manager_narrates_a_model_that_can_pin_no_logical_class():
    """C8 item 5: the line belongs to the component whose row reports it.

    observe wrote it, so the log the frozen gate hashes was not a pure
    product of the components. The manager owns the rows that report
    it, so it says it, and a run with no observer says it too.
    """
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    row = UnpinnableRow()
    _manager(engine, row)
    model = _Model()
    reason = "one observable, no boundary"
    row.forced_solve_unavailable.fire(model, reason)
    lines = []
    for line in log.lines:
        if "NO FORCED SOLVE" in line:
            lines.append(line)
    assert len(lines) == 1
    assert "5-detector window model" in lines[0]
    assert "one observable, no boundary" in lines[0]


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
    router = decoders.CodeRouter(row)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
        blocks_unit_by_pool={"default": blocks_unit},
    )


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


def _staging_manager(engine, row, *, copies_input: bool):
    """A one-unit manager whose default pool copies its input, or does not."""
    router = decoders.CodeRouter(row)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
        copies_input_by_pool={"default": copies_input},
    )
