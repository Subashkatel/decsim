"""The strong redecode's laws, on fakes of its shape and its ports.

Toshio et al. 2510.25222 Sec. III A: the selection of the strong
decoder's result is sent when the weak result is not confident (steps
3 and 4), and the strong decoder started beside the weak one (Step 1)
is selected as it is; a strong input landed before its selection waits
for it. The submission is the DecodeQueue port's enqueue with the
committer's return path (SimPy's callback on the event).
"""

import pytest

import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_module
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_window_shapes as shapes
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records

WINDOW_KEY = (1, 2)
FAR_BOUNDARY_KEY = (1, 4)


def _weak_job() -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=2,
        round_count=6,
        strong_label="strong(mem1 W2)",
    )


def _strong_job(sequence: int) -> decoding_records.DecodeJob:
    request_key = window_records.DecoderRequestKey(
        1, 2, window_records.DecoderTier.STRONG, sequence
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
    """Assigns one strong job, built now or held for the commits it names."""

    def __init__(
        self, job, is_held: bool, waits_on=(FAR_BOUNDARY_KEY,)
    ) -> None:
        self.job = job
        self.is_held = is_held
        self.waits_on = waits_on
        self.planned = []

    def plan(self, weak_job) -> shapes.StrongAssignment:
        self.planned.append(weak_job)
        if self.is_held:
            return shapes.StrongAssignment(self.job.request_key, None)
        return shapes.StrongAssignment(self.job.request_key, self.job)

    def release_conditions(self, assignment):
        del assignment
        return pending_module.ReleaseConditions(
            committed_windows=self.waits_on,
            name="far_boundary",
            released_description="far-side weak boundary determined",
        )

    def held_job(self, assignment):
        del assignment
        self.is_held = False
        return self.job


class _DecoderOutput:
    """The weak decoder's end of the selection hop; the test delivers it."""

    def __init__(self) -> None:
        self.selections = []

    def send_selection(self, weak_job, strong_request_key, on_delivered):
        self.selections.append((weak_job, strong_request_key, on_delivered))
        return 30


class _StrongOutput:
    """Syndrome buffer 1's end of the strong input hop."""

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


def _redecode(shape):
    engine = engine_module.Engine()
    decoder_output = _DecoderOutput()
    strong_output = _StrongOutput()
    queue = _DecodeQueue()
    on_strong_decoded = object()  # opaque: the committer's return path
    redecode = strong_redecode_module.StrongRedecode(
        engine,
        shape,
        decoder_output,
        strong_output,
        queue,
        on_strong_decoded,
    )
    return redecode, decoder_output, strong_output, queue, on_strong_decoded


def _call_names(queue) -> list:
    names = []
    for call in queue.calls:
        names.append(call[0])
    return names


def test_a_job_built_now_is_queued_behind_its_selection():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, output, _strong, queue, on_strong_decoded = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    assert len(output.selections) == 1
    assert output.selections[0][1] == strong_job.request_key
    assert _call_names(queue) == ["enqueue", "await"]
    _kind, queued, _send_input, return_path = queue.calls[0]
    assert queued is strong_job
    assert return_path is on_strong_decoded
    assert queue.calls[1] == ("await", WINDOW_KEY, strong_job.request_key)


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


def test_a_sibling_started_with_the_weak_job_is_selected_as_it_is():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, output, strong_output, queue, _done = _redecode(shape)
    weak_job = _weak_job()
    submission = redecode.parallel_strong_submission(weak_job)
    assert submission.job is strong_job
    redecode.escalate(weak_job)
    # one plan, one selection, no second job: the sibling is selected
    assert len(shape.planned) == 1
    assert len(output.selections) == 1
    assert queue.calls == [("await", WINDOW_KEY, strong_job.request_key)]


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

    A seam-pinned window is bounded on both faces (note 14 section 1.4),
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


def test_a_row_that_holds_its_job_and_names_nothing_is_refused():
    """A held job with no condition would never leave."""
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=True, waits_on=())
    redecode, _output, _strong, _queue, _done = _redecode(shape)
    weak_job = _weak_job()
    with pytest.raises(RuntimeError, match="no release condition"):
        redecode.escalate(weak_job)
