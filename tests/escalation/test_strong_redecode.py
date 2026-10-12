"""The strong redecode's laws, on fakes of its shape and its ports.

Toshio et al. 2510.25222 Sec. III A: the selection of the strong
decoder's result is sent when the weak result is not confident (steps
3 and 4), and the strong decoder started beside the weak one (Step 1)
is selected as it is; a strong input landed before its selection waits
for it. Sec. III C, lines 1247 to 1250: the region's rounds are
assigned to the strong decoder at the switch, so a held window whose
rounds the strong side lacks has them carried up once, and the landing
releases it. The submission is the DecodeQueue port's enqueue with the
committer's return path (SimPy's callback on the event).
"""

import dataclasses

import pytest

import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_module
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_window_shapes as shapes
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.routes as route_records
import decsim.records.windows as window_records
import decsim.statistics.registry as statistics_registry
import tests.declared_run as declared_run

WINDOW_KEY = (1, 2)
FAR_BOUNDARY_KEY = (1, 4)
REGION_ROUND_KEYS = ((1, 4), (1, 5), (1, 6))
# the edge from the weak pool to the strong one, each hop named by the
# parts at its ends (topology/graph_layout.py)
EDGE_HOPS = route_records.EdgeHops(
    window_records.WEAK_POOL_ID,
    window_records.STRONG_POOL_ID,
    "weak_to_strong",
    "strong_to_weak",
)


def _weak_job(owed_boundaries: int = 0) -> decoding_records.DecodeJob:
    window = window_records.Window(
        operation_id=1,
        window_index=2,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=9,
        round_count=6,
    )
    window.deps_remaining = owed_boundaries
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=2,
        round_count=6,
        window=window,
    )


def _strong_job(sequence: int) -> decoding_records.DecodeJob:
    request_key = window_records.DecoderRequestKey(
        1, 2, window_records.STRONG_POOL_ID, sequence
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=2,
        round_count=9,
        label="strong(mem1 W2)",
        strong_decode_for=WINDOW_KEY,
        request_key=request_key,
    )


class _Shape:
    """Assigns one strong job, built now or held for the commits it names.

    A held row lacks its rounds on the strong side until the test says
    they are stored; rounds_to_carry names them while they are missing.
    """

    def __init__(
        self,
        job,
        is_held: bool,
        waits_on=(FAR_BOUNDARY_KEY,),
        missing_rounds=(),
    ) -> None:
        self.job = job
        self.is_held = is_held
        self.waits_on = waits_on
        self.missing_rounds = tuple(missing_rounds)
        self.planned = []

    def plan(self, weak_job) -> pending_module.StrongAssignment:
        self.planned.append(weak_job)
        if self.is_held:
            return pending_module.StrongAssignment(self.job.request_key, None)
        return pending_module.StrongAssignment(self.job.request_key, self.job)

    def release_conditions(self, assignment):
        del assignment
        stored_data_of_operation = None
        if self.missing_rounds:
            stored_data_of_operation = 1
        return pending_module.ReleaseConditions(
            committed_windows=self.waits_on,
            stored_data_of_operation=stored_data_of_operation,
            name="far_boundary",
            released_description="far-side weak boundary determined",
        )

    def held_job(self, assignment):
        del assignment
        if self.missing_rounds:
            return None
        self.is_held = False
        return self.job

    def rounds_to_carry(self, assignment):
        del assignment
        return self.missing_rounds


class _DecoderOutput:
    """The weak decoder's end of the escalation hop; the test delivers it."""

    def __init__(self) -> None:
        self.selections = []
        self.regions = []

    def send_selection(self, path, weak_job, strong_request_key, on_delivered):
        del path
        self.selections.append((weak_job, strong_request_key, on_delivered))
        return 30

    def send_region(self, path, store, region, on_delivered):
        del path, store
        self.regions.append((region, on_delivered))
        return 30


class _Retention:
    """The weak syndrome buffer's rounds, one three-bit fragment each."""

    def escalated_rounds(self, round_keys, target=None):
        del target
        packets = []
        for operation_id, round_index in round_keys:
            fragment = round_records.RetainedSyndromeFragment(
                operation_id=operation_id,
                patch_ids=(0,),
                round_index=round_index,
                bits=(1, 0, 1),
                size_bits=3,
                fragment_index=0,
            )
            packet = round_records.SyndromeRoundPacket(
                operation_id, round_index, (fragment,)
            )
            packets.append(packet)
        return tuple(packets)


class _StrongReceiver:
    """The strong syndrome buffer's receiving end, recording the landing."""

    def __init__(self) -> None:
        self.reserved = 0
        self.reserved_bits = 0
        self.landed = []

    def reserve_region(self, region):
        self.reserved += len(region.packets)
        self.reserved_bits += region.wire_bits

    def receive_region(self, region, on_stored):
        self.landed.append(region)
        on_stored()


class _FormingStrongReceiver(_StrongReceiver):
    """A receiving end that forms the rounds, storing them after landing."""

    def __init__(self) -> None:
        _StrongReceiver.__init__(self)
        self.stored_callbacks = []

    def receive_region(self, region, on_stored):
        self.landed.append(region)
        self.stored_callbacks.append(on_stored)


class _StrongOutput:
    """The strong syndrome buffer's end of the strong input hop."""

    def __init__(self) -> None:
        self.inputs = []

    def send_input(self, job, on_landed):
        payload_bits = job.payload_bits()
        self.inputs.append((job, payload_bits, on_landed))
        return 60


class _DecodeQueue:
    """Records the port calls in the order they arrive."""

    def __init__(self) -> None:
        self.calls = []

    def enqueue(self, job, send_input=None, on_decoded=None):
        self.calls.append(("enqueue", job, send_input, on_decoded))

    def await_strong_result(self, window_key, request_key):
        self.calls.append(("await", window_key, request_key))

    def accept_selection(self, window_key, request_key):
        self.calls.append(("accept", window_key, request_key))

    def cancel_strong(self, window_key):
        self.calls.append(("cancel", window_key))

    def abandon_strong(self, window_key):
        self.calls.append(("abandon", window_key))


def _redecode(shape, receiver_row=_StrongReceiver):
    engine = engine_module.Engine()
    decoder_output = _DecoderOutput()
    strong_output = _StrongOutput()
    queue = _DecodeQueue()
    verdict = _Verdict()
    redecode = strong_redecode_module.StrongRedecode(engine, EDGE_HOPS)
    redecode.shape = shape
    redecode.pending = pending_module.PendingStrongWindows()
    redecode.retention = _Retention()
    # the store the carried rounds are read out of; the stand-in
    # decoder output reads nothing from it
    redecode.carried_from = object()
    redecode.decoder_output = decoder_output
    redecode.strong_receiver = receiver_row()
    redecode.strong_output = strong_output
    redecode.decode_queue = queue
    redecode.verdict = verdict
    return redecode, decoder_output, strong_output, queue, verdict


class _Join:
    """The next edge's confidence join, holding what reaches it."""

    signal = None

    def __init__(self) -> None:
        self.results = []

    def accept_result(self, job, result) -> None:
        """One target decode finished."""
        self.results.append((job, result))

    def unresolved_windows(self) -> tuple:
        """Nothing waits here."""
        return ()


class _NextEdge:
    """The edge out of the target pool, hearing what reaches it."""

    def __init__(self) -> None:
        self.calls = []

    def arm_deadline(self, source_job) -> None:
        """A target job arrives at its pool, the source of this edge."""
        self.calls.append(("arm", source_job))

    def start_beside_source(self, source_job) -> None:
        """A target job was queued."""
        self.calls.append(("start", source_job))

    def cancel_strong_request(self, window_key) -> None:
        """A keep upstream cancelled the window."""
        self.calls.append(("cancel", window_key))

    def abandon_strong_request(self, window_key) -> None:
        """A keep upstream let the window's decodes run out."""
        self.calls.append(("abandon", window_key))


class _Builder:
    """Numbers each new request after the ones the stubs made."""

    def __init__(self) -> None:
        self.sequence = 20

    def new_request_key(self, operation_id, window_id, pool):
        """The next request key of the run."""
        self.sequence += 1
        return window_records.DecoderRequestKey(
            operation_id, window_id, pool, self.sequence
        )


class _TwoClasses:
    """A signal that reads one solve per logical class."""

    forced_logical_classes = (0, 1)


class _BothAtOnce:
    """A policy that starts every target decode beside its source."""

    def decodes_both_at_once(self, window) -> bool:
        """Every window."""
        del window
        return True


class _Verdict:
    """The window side a strong result returns to."""

    def __init__(self) -> None:
        self.results = []

    def accept_revision(self, job, result) -> None:
        """One strong decode finished."""
        self.results.append((job, result))


def _call_names(queue) -> list:
    names = []
    for call in queue.calls:
        names.append(call[0])
    return names


def test_a_job_built_now_is_queued_behind_its_selection():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, output, _strong, queue, verdict = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    assert len(output.selections) == 1
    assert output.selections[0][1] == strong_job.request_key
    assert _call_names(queue) == ["enqueue", "await"]
    _kind, queued, _send_input, return_path = queue.calls[0]
    assert queued is strong_job
    assert return_path == redecode.accept_target_result
    assert queue.calls[1] == ("await", WINDOW_KEY, strong_job.request_key)


def test_a_target_result_meets_the_verdict_when_no_edge_leaves_the_target():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, _output, _strong, _queue, verdict = _redecode(shape)
    result = object()

    redecode.accept_target_result(strong_job, result)

    assert verdict.results == [(strong_job, result)]


def test_a_target_result_reaches_the_join_of_the_edge_that_leaves_it():
    """The next edge's signal reads the result before any verdict."""
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, _output, _strong, _queue, verdict = _redecode(shape)
    join = _Join()
    redecode.target_join = join
    result = object()

    redecode.accept_target_result(strong_job, result)

    assert join.results == [(strong_job, result)]
    assert verdict.results == []


def test_a_queued_target_job_starts_the_edge_out_of_its_pool_beside_it():
    """The edge out hears the job's arrival, its deadline, before it queues."""
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, _output, _strong, _queue, _verdict = _redecode(shape)
    next_edge = _NextEdge()
    redecode.next_edge = next_edge
    weak_job = _weak_job()

    redecode.escalate(weak_job)

    assert next_edge.calls == [("arm", strong_job), ("start", strong_job)]


@pytest.mark.parametrize("ending", ["cancel", "abandon"])
def test_a_keep_ends_the_windows_work_on_the_edge_out_of_the_target(ending):
    """What a keep does to the target's decode it does one edge down."""
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, _output, _strong, queue, _verdict = _redecode(shape)
    next_edge = _NextEdge()
    redecode.next_edge = next_edge
    end = getattr(redecode, f"{ending}_strong_request")

    end(WINDOW_KEY)

    assert queue.calls == [(ending, WINDOW_KEY)]
    assert next_edge.calls == [(ending, WINDOW_KEY)]


def test_a_decode_started_beside_a_source_past_the_entry_is_selected():
    """One depth down, the speculative decode is the one selected."""
    deep_job = _strong_job(8)
    shape = _Shape(deep_job, is_held=False)
    redecode, output, _strong, queue, _verdict = _redecode(shape)
    redecode.policy = _BothAtOnce()
    source_job = _strong_job(5)

    redecode.start_beside_source(source_job)
    redecode.escalate(source_job)

    assert shape.planned == [source_job]
    assert len(output.selections) == 1
    assert _call_names(queue) == ["enqueue", "await"]
    assert queue.calls[0][1] is deep_job


def test_a_target_job_leaves_with_a_solve_per_class_the_next_edge_reads():
    """The companion reads the lead's input and answers the same window."""
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    shape.builder = _Builder()
    shape.engine = engine_module.Engine()
    redecode, _output, _strong, queue, _verdict = _redecode(shape)
    join = _Join()
    join.signal = _TwoClasses()
    redecode.target_join = join
    weak_job = _weak_job()

    redecode.escalate(weak_job)

    enqueued = []
    for call in queue.calls:
        if call[0] == "enqueue":
            enqueued.append(call[1])
    lead, companion = enqueued
    assert lead is strong_job
    assert (lead.forced_logical_class, companion.forced_logical_class) == (
        0,
        1,
    )
    assert companion.input_key == lead.request_key
    assert companion.strong_decode_for == lead.strong_decode_for


def test_a_held_job_leaves_when_the_window_it_named_commits():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=True)
    redecode, output, strong_output, queue, _done = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    assert redecode.pending_work() == ((WINDOW_KEY, "waiting_far_boundary", 0),)
    assert _call_names(queue) == ["await"]
    redecode.submit_if_commit_releases((1, 3))
    assert _call_names(queue) == ["await"]
    redecode.submit_if_commit_releases(FAR_BOUNDARY_KEY)
    assert _call_names(queue) == ["await", "enqueue"]
    assert queue.calls[1][1] is strong_job
    assert not shape.is_held
    assert not redecode.has_pending()


def test_a_speculative_decode_started_with_the_weak_job_is_selected():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, output, strong_output, queue, _done = _redecode(shape)
    weak_job = _weak_job()
    submission = redecode.parallel_strong_submission(weak_job)
    assert submission.job is strong_job
    redecode.escalate(weak_job)
    # one plan, one selection, no second job: the speculative decode is selected
    assert len(shape.planned) == 1
    assert len(output.selections) == 1
    assert queue.calls == [("await", WINDOW_KEY, strong_job.request_key)]


def test_each_escalation_is_counted_with_its_hold_or_its_selection():
    held_job = _strong_job(5)
    held_shape = _Shape(held_job, is_held=True)
    held_redecode, *_ = _redecode(held_shape)
    speculative_job = _strong_job(6)
    speculative_shape = _Shape(speculative_job, is_held=False)
    speculative_redecode, *_ = _redecode(speculative_shape)
    run_statistics = statistics_registry.Registry(held_redecode.engine)
    run_statistics.attach(held_redecode, "switching")
    run_statistics.attach(speculative_redecode, "switching_speculative")

    held_weak_job = _weak_job()
    held_redecode.escalate(held_weak_job)
    weak_job = _weak_job()
    speculative_redecode.parallel_strong_submission(weak_job)
    speculative_redecode.escalate(weak_job)

    held = held_redecode.statistics
    speculative = speculative_redecode.statistics
    assert (held.escalated.count, held.held.count) == (1, 1)
    assert held.speculative_selected.count == 0
    assert speculative.escalated.count == 1
    assert speculative.speculative_selected.count == 1
    assert speculative.held.count == 0


def test_a_speculative_decode_is_planned_when_its_weak_job_is_released():
    """Step 1 starts both decoders together (2510.25222 lines 598-601).

    A weak job whose window still owes a boundary waits in the queue, so
    its speculative strong decode is planned at the release and not at
    the submission.
    """
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, _output, _strong, _queue, _done = _redecode(shape)
    weak_job = _weak_job(owed_boundaries=1)
    deferred = redecode.parallel_strong_submission(weak_job)
    planned_while_blocked = len(shape.planned)
    submission = redecode.released_submission(WINDOW_KEY)
    assert deferred is None
    assert planned_while_blocked == 0
    assert submission.job is strong_job
    assert redecode.released_submission(WINDOW_KEY) is None


def test_a_strong_input_landed_before_its_selection_waits_for_it():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, output, strong_output, queue, _done = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    _kind, _job, send_input, _return_path = queue.calls[0]
    landings = []
    expected_delay = send_input(lambda: landings.append("landed"))
    # the pool's estimate is the later of the input (60) and the
    # selection (30 ticks from now)
    assert expected_delay == 60
    _job, _bits, input_delivered = strong_output.inputs[0]
    input_delivered()
    assert landings == []
    _weak, _key, selection_delivered = output.selections[0]
    selection_delivered()
    assert landings == ["landed"]
    assert queue.calls[-1] == ("accept", WINDOW_KEY, strong_job.request_key)


def test_a_row_that_names_two_windows_waits_for_both():
    """The general condition: a row waits on every window it named.

    A seam-pinned window is bounded on both faces,
    and a Skoric layer-B window on its two adjacent layer-A commits
    (2209.08552 lines 419-421); the redecode counts the commits down.
    """
    strong_job = _strong_job(5)
    faces = ((1, 1), FAR_BOUNDARY_KEY)
    shape = _Shape(strong_job, is_held=True, waits_on=faces)
    redecode, _output, _strong, queue, _done = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    redecode.submit_if_commit_releases((1, 1))
    assert _call_names(queue) == ["await"]
    assert redecode.has_pending()
    redecode.submit_if_commit_releases(FAR_BOUNDARY_KEY)
    assert _call_names(queue) == ["await", "enqueue"]
    assert queue.calls[1][1] is strong_job


def test_a_held_windows_missing_rounds_are_carried_up_once_and_land():
    """The region goes up when the row's commits are in, and only once.

    Two wake-ups while the region crosses send nothing more; at the
    landing the strong side's receiving end takes the rounds, and the
    row builds its job on the next wake-up.
    """
    strong_job = _strong_job(5)
    shape = _Shape(
        strong_job, is_held=True, waits_on=(), missing_rounds=REGION_ROUND_KEYS
    )
    redecode, output, _strong, queue, _done = _redecode(shape)
    weak_job = _weak_job()

    redecode.escalate(weak_job)
    redecode.submit_if_stored_data_releases(1)
    region, landed = output.regions[0]
    landed()
    shape.missing_rounds = ()
    redecode.submit_if_stored_data_releases(1)

    assert len(output.regions) == 1
    assert region.request_key == strong_job.request_key
    assert region.round_keys == REGION_ROUND_KEYS
    assert region.wire_bits == 9
    assert redecode.strong_receiver.reserved == 3
    assert redecode.strong_receiver.landed == [region]
    assert redecode.carried_round_keys == set()
    assert _call_names(queue) == ["await", "enqueue"]
    assert not redecode.has_pending()


def test_a_region_still_forming_at_its_landing_is_not_carried_again():
    """A wake-up between the landing and the store sends nothing more.

    A strong seat that forms the rounds stores them only once it has
    formed them, so until then they still count as carried.
    """
    strong_job = _strong_job(5)
    shape = _Shape(
        strong_job, is_held=True, waits_on=(), missing_rounds=REGION_ROUND_KEYS
    )
    redecode, output, _strong, _queue, _done = _redecode(
        shape, _FormingStrongReceiver
    )
    receiver = redecode.strong_receiver
    weak_job = _weak_job()

    redecode.escalate(weak_job)
    redecode.submit_if_stored_data_releases(1)
    _region, landed = output.regions[0]
    landed()
    redecode.submit_if_stored_data_releases(1)
    carried_while_forming = set(redecode.carried_round_keys)
    (stored,) = receiver.stored_callbacks
    stored()

    assert len(output.regions) == 1
    assert carried_while_forming == set(REGION_ROUND_KEYS)
    assert redecode.carried_round_keys == set()


class _NoConditionRedo(shapes.RedoWindow):
    """A user row whose held job names nothing that releases it."""

    def release_conditions(self, assignment):
        del assignment
        return pending_module.ReleaseConditions()


@dataclasses.dataclass(frozen=True)
class _NoConditionRedoSettings(shapes.RedoWindow.Settings):
    def build(self, engine) -> _NoConditionRedo:
        return _NoConditionRedo(engine)


def test_a_row_that_holds_its_job_and_names_nothing_is_refused():
    """A held job with no condition would never leave.

    mem2 waits on mem1's feedback, so mem1's patch keeps measuring and a
    run that kept the job would never reach its end-of-run checks.
    """
    first = declared_run.memory_operation(1)
    second = declared_run.memory_operation(2, predecessors=(1,), blocked_by=1)
    no_condition = _NoConditionRedoSettings()

    with pytest.raises(RuntimeError, match="held with no release condition"):
        declared_run.switching_run(
            escalates=True,
            operations=[first, second],
            strong_window=no_condition,
        )
