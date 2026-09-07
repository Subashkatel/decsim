"""The strong redecode's laws, on fakes of its shape and its ports.

Toshio et al. 2510.25222 Sec. III A: the selection of the strong
decoder's result is sent when the weak result is not confident (steps
3 and 4), and the strong decoder started beside the weak one (Step 1)
is selected as it is; a strong input landed before its selection waits
for it. The submission is the DecodeQueue port's enqueue with the
committer's return path (SimPy's callback on the event).
"""

import decsim.engine as engine_module
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_window_shapes as shapes
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records

WINDOW_KEY = (1, 2)


def _weak_job() -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1, window_id=2, n_rounds=6, strong_label="strong(mem1 W2)"
    )


def _strong_job(sequence: int) -> decoding_records.DecodeJob:
    request_key = window_records.DecoderRequestKey(
        1, 2, window_records.DecoderTier.STRONG, sequence
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=2,
        n_rounds=9,
        label="strong(mem1 W2)",
        strong_decode_for=WINDOW_KEY,
        request_key=request_key,
    )


class _Shape:
    """Assigns one strong job, built now or held for its far boundary."""

    def __init__(self, job, is_held: bool) -> None:
        self.job = job
        self.is_held = is_held
        self.planned = []
        self.selection_ticks = []

    def plan(self, weak_job) -> shapes.StrongAssignment:
        self.planned.append(weak_job)
        if self.is_held:
            return shapes.StrongAssignment(self.job.request_key, None)
        return shapes.StrongAssignment(self.job.request_key, self.job)

    def note_selection_sent(self, window_key, selection_arrival_ticks):
        self.selection_ticks.append((window_key, selection_arrival_ticks))

    def take_if_far_boundary_committed(self, window_key):
        del window_key
        if not self.is_held:
            return None
        self.is_held = False
        return shapes.DeferredStrongJob(self.job, self.selection_ticks[0][1])

    def take_if_terminal_data_stored(self, operation_id):
        del operation_id
        return None


class _Transfers:
    """Records every send; a delivery is released by the test."""

    def __init__(self) -> None:
        self.selections = []
        self.inputs = []

    def send_selection(self, weak_job, strong_request_key, on_delivered):
        self.selections.append((weak_job, strong_request_key, on_delivered))
        return 30

    def send_for_job(self, path, job, *, payload_bits, on_delivered):
        self.inputs.append((path, job, payload_bits, on_delivered))
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
    transfers = _Transfers()
    queue = _DecodeQueue()
    on_strong_decoded = object()  # opaque: the committer's return path
    redecode = strong_redecode_module.StrongRedecode(
        engine, shape, transfers, queue, on_strong_decoded
    )
    return redecode, transfers, queue, on_strong_decoded


def _call_names(queue) -> list:
    names = []
    for call in queue.calls:
        names.append(call[0])
    return names


def test_a_job_built_now_is_queued_behind_its_selection():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, transfers, queue, on_strong_decoded = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    assert len(transfers.selections) == 1
    assert transfers.selections[0][1] == strong_job.request_key
    assert _call_names(queue) == ["enqueue", "await"]
    _kind, queued, _send_input, return_path = queue.calls[0]
    assert queued is strong_job
    assert return_path is on_strong_decoded
    assert queue.calls[1] == ("await", WINDOW_KEY, strong_job.request_key)


def test_a_held_job_leaves_at_its_far_boundary_commit():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=True)
    redecode, transfers, queue, _on_strong_decoded = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    assert shape.selection_ticks == [(WINDOW_KEY, 30)]
    assert _call_names(queue) == ["await"]
    redecode.submit_if_far_boundary_committed((1, 4))
    assert _call_names(queue) == ["await", "enqueue"]
    assert queue.calls[1][1] is strong_job
    assert not shape.is_held


def test_a_sibling_started_with_the_weak_job_is_selected_as_it_is():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, transfers, queue, _on_strong_decoded = _redecode(shape)
    weak_job = _weak_job()
    submission = redecode.parallel_strong_submission(weak_job)
    assert submission.job is strong_job
    redecode.escalate(weak_job)
    # one plan, one selection, no second job: the sibling is selected
    assert len(shape.planned) == 1
    assert len(transfers.selections) == 1
    assert queue.calls == [("await", WINDOW_KEY, strong_job.request_key)]


def test_a_strong_input_landed_before_its_selection_waits_for_it():
    strong_job = _strong_job(5)
    shape = _Shape(strong_job, is_held=False)
    redecode, transfers, queue, _on_strong_decoded = _redecode(shape)
    weak_job = _weak_job()
    redecode.escalate(weak_job)
    _kind, _job, send_input, _return_path = queue.calls[0]
    landings = []
    expected_delay = send_input(lambda: landings.append("landed"))
    # the pool's estimate is the later of the input (60) and the
    # selection (30 ticks from now)
    assert expected_delay == 60
    _path, _job, _bits, input_delivered = transfers.inputs[0]
    input_delivered()
    assert landings == []
    _weak, _key, selection_delivered = transfers.selections[0]
    selection_delivered()
    assert landings == ["landed"]
    assert queue.calls[-1] == ("accept", WINDOW_KEY, strong_job.request_key)
