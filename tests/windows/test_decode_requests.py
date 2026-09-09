"""The decode requests' laws through the DecodeQueue port.

A complete window is requested once; a blocked window ships its raw
rounds and is masked when its decode starts (qLDPC's net_error folded
into the next window's syndrome, qldpc/decoders/sinter.py
decode_shots_to_error); a withdrawn window is requested again fresh. A
request is priced for the rounds it reads, which the declared run shows
on its lookahead tail (Skoric et al. 2209.08552, Tan et al. 2209.09219).
"""

import types

import pytest

import decsim.decoders.decoder_memory as decoder_memory
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.links.window_transfers as window_transfers
import decsim.observe.run_views as run_views
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.windows.boundary_payloads as boundary_payloads
import decsim.windows.decode_requests as decode_requests
import decsim.windows.round_retention as round_retention
import decsim.windows.window_interactions as window_interactions
import tests.declared_run as declared_run


class _RecordingQueue:
    def __init__(self) -> None:
        self.enqueued = []
        self.withdrawn = []

    def enqueue(self, job, send_input=None, on_decoded=None) -> None:
        job.submitted = True
        job.on_decoded = on_decoded
        self.enqueued.append((job, send_input))

    def withdraw_window(self, window_key) -> None:
        self.withdrawn.append(window_key)

    def release_parked(self, window_key) -> None:
        del window_key


class _Link:
    def expected_delay_ticks(self, _path, _payload_bits, _now_ticks) -> int:
        return 3

    def send(
        self, _path, _payload_bits, _now_ticks, _attribution, on_delivered
    ):
        on_delivered(None)


def _ignore_result(_job, _result) -> None:
    """The requester's tests read the queue, not the result."""


def _fragment(round_index, bits=None) -> round_records.RetainedSyndromeFragment:
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=bits,
        size_bits=None,
        fragment_index=0,
    )


class _Fixture:
    """One six-round operation with one window reading rounds 1 to 5."""

    def __init__(self) -> None:
        self.engine = engine_module.Engine()
        self.operation = program_records.Operation(
            1, "memory", (0,), patches=(0,)
        )
        self.window = window_records.Window(
            operation_id=1,
            window_index=0,
            commit_lo=1,
            commit_hi=3,
            buffer_hi=5,
            round_count=5,
        )
        settings = round_store_settings.RoundStoreSettings()
        self.store = round_store_module.RoundStore(settings)
        self.arrived = 0
        geometry = types.SimpleNamespace(code_name="surface")
        no_models = types.SimpleNamespace(model_by_window={})
        self.planner = types.SimpleNamespace(
            windows_by_key={(1, 0): self.window},
            successors_by_operation={1: []},
            models=no_models,
            spatial_node_count_of=lambda _operation_id: 17,
            code_geometry_of=lambda _operation_id: geometry,
            is_windowed=lambda _operation_id: True,
        )
        self.tracker = types.SimpleNamespace(
            operation_by_id={1: self.operation},
            effective_round_count_for_window=lambda _operation_id, _window: 6,
            round_count_for_window=lambda _operation_id, _window=None: 6,
            rounds_arrived=lambda _operation_id: self.arrived,
            strong_rounds_arrived=lambda _operation_id: 0,
            has_first_round=self._has_first_round,
            is_data_complete=self._is_data_complete,
            is_buffer_filled_by_memory=lambda _window: False,
        )
        self.retention = round_retention.RoundRetention(
            self.store,
            None,
            self.planner,
            self.tracker,
            is_strong_context_retained=False,
            primary_tier=window_records.DecoderTier.WEAK,
        )
        link = _Link()
        transfers = window_transfers.WindowTransfers(self.engine, link)
        store_output = round_output.RoundStoreOutput(
            transfers,
            transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
            "Buffer 0",
            self.store,
        )
        boundary_payload = boundary_payloads.DenseSeamMask()
        interaction = window_interactions.DefaultWindowInteraction(
            0, boundary_payload
        )
        gate = decode_requests.WindowInputGate(self.planner, interaction)
        self.gate = gate
        self.builder = decode_requests.DecodeRequestBuilder(
            self.engine, self.planner, self.tracker, interaction, gate
        )
        self.queue = _RecordingQueue()
        policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
        verdict = types.SimpleNamespace(
            accept_result=_ignore_result, accept_strong_result=_ignore_result
        )
        self.requester = decode_requests.DecodeRequester(
            self.tracker,
            self.retention,
            self.builder,
            self.queue,
            policy,
            verdict,
            store_output,
        )
        self.retention.register_window((1, 0), self.window)

    def _has_first_round(self, _window) -> bool:
        return self.arrived >= 1

    def _is_data_complete(self, _window) -> bool:
        return self.arrived >= 5

    def arrive(self, round_index: int) -> None:
        fragment = _fragment(round_index)
        packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
        self.store.accept_packed_round(packet, publication_tick=round_index)
        self.arrived = round_index
        self.requester.request_if_ready(self.window, None)


def test_a_complete_window_is_requested_once():
    fixture = _Fixture()
    for round_index in (1, 2, 3, 4):
        fixture.arrive(round_index)
    assert fixture.queue.enqueued == []
    fixture.arrive(5)
    fixture.requester.request_if_ready(fixture.window, None)
    assert len(fixture.queue.enqueued) == 1
    (job, send_input) = fixture.queue.enqueued[0]
    assert job.window is fixture.window
    assert job.gate is fixture.gate
    assert job.on_decoded is _ignore_result
    assert [payload.round_index for payload in job.payloads] == [1, 2, 3, 4, 5]
    assert fixture.window.queued
    assert fixture.window.t_queued == 0
    assert send_input(lambda: None) == 3


def test_a_blocked_window_ships_raw_rounds_and_is_masked_at_start():
    fixture = _Fixture()
    fixture.window.deps = [(1, 9)]
    fixture.window.deps_remaining = 1
    fixture.window.boundary_in = {2: [1, 0, 1]}
    for round_index in (1, 2, 3, 4, 5):
        fixture.arrive(round_index)
    (job, _send_input) = fixture.queue.enqueued[0]
    assert fixture.gate.may_start(job) is False
    raw = job.payloads[1]
    assert raw.bits is None
    landed = decoder_memory.MaterializedSyndromeRound(1, 2, (raw,))
    job.decoder_input = decoder_memory.DecoderInput(
        1, 0, job.request_key, (landed,)
    )
    job.memory = decoder_memory.DecoderMemory("default", 0, None)
    fixture.gate.mask_input(job)
    (masked,) = job.decoder_input.rounds[0].fragments
    assert masked.bits == (1, 0, 1)
    fixture.window.deps_remaining = 0
    assert fixture.gate.may_start(job) is True


def test_a_withdrawn_window_is_requested_again_fresh():
    fixture = _Fixture()
    for round_index in (1, 2, 3, 4, 5):
        fixture.arrive(round_index)
    (first_job, _send_input) = fixture.queue.enqueued[0]
    fixture.requester.withdraw(fixture.window)
    assert fixture.queue.withdrawn == [(1, 0)]
    assert not fixture.window.queued
    assert fixture.window.t_queued is None
    fixture.requester.request_if_ready(fixture.window, None)
    assert len(fixture.queue.enqueued) == 2
    (second_job, _send_input) = fixture.queue.enqueued[1]
    assert second_job is not first_job
    assert second_job.request_key.run_sequence == 1


def test_a_decode_job_is_priced_for_the_rounds_it_reads():
    """A window's decode reads the rounds that exist, not its plan.

    Decoder work scales with the rounds actually fed (Skoric et al.
    2209.08552, tau_W over n_W), the last window of an operation may be
    smaller than a regular one (Tan et al. 2209.09219), and no
    sliding-window implementation feeds rounds past the data (Gong et
    al.'s sliding-window decoder, cudaq-qec sliding_window). On the
    nine-round declared run the regular window 1 reads its planned
    rounds 4 to 9, while the lookahead tail plans rounds 7 to 12 and
    reads only the three rounds the operation ever emitted.
    """
    machine = declared_run.switching_run(rounds=9, record=True)
    view = run_views.switching_records_view(
        machine.observation.windows, machine.observation.decode_records
    )
    by_window = {}
    for record in view.requests:
        window_id = record.request_key.window_id
        by_window[window_id] = record
    regular = by_window[1]
    tail = by_window[2]
    assert regular.request_key.tier is window_records.DecoderTier.WEAK
    assert (regular.input_round_lo, regular.input_round_hi) == (4, 9)
    assert regular.input_round_count == 6
    assert tail.request_key.tier is window_records.DecoderTier.WEAK
    assert (tail.input_round_lo, tail.input_round_hi) == (7, 12)
    assert tail.input_round_count == 3


def _landed_job(fixture, folds_in_place: bool):
    """One blocked window's job, landed in a real unit memory."""
    fixture.window.deps = [(1, 9)]
    fixture.window.deps_remaining = 1
    fixture.window.boundary_in = {2: [1, 0, 1]}
    for round_index in (1, 2, 3, 4, 5):
        fixture.arrive(round_index)
    (job, _send_input) = fixture.queue.enqueued[0]
    memory = decoder_memory.DecoderMemory("default", 0, None)
    job.decoder_input = memory.deposit(job)
    job.memory = memory
    if folds_in_place:
        fixture.gate.copies_the_fold = False
    return job, memory


def test_the_copy_fold_leaves_the_units_rounds_raw():
    """Today's row: the job reads a masked duplicate (cuda-q QEC's shape)."""
    fixture = _Fixture()
    job, memory = _landed_job(fixture, folds_in_place=False)
    fixture.gate.mask_input(job)
    resident = memory.input_of(job)
    (masked,) = job.decoder_input.rounds[1].fragments
    assert masked.bits == (1, 0, 1)
    (raw,) = resident.rounds[1].fragments
    assert raw.bits is None
    assert job.decoder_input is not resident


def test_the_in_place_fold_writes_the_units_own_memory():
    """AFS 2001.06598 lines 528-531: the unit's memory is written directly."""
    fixture = _Fixture()
    job, memory = _landed_job(fixture, folds_in_place=True)
    fixture.gate.mask_input(job)
    resident = memory.input_of(job)
    assert job.decoder_input is resident
    (masked,) = resident.rounds[1].fragments
    assert masked.bits == (1, 0, 1)


def test_an_in_place_fold_of_an_input_two_jobs_read_is_refused():
    """One input has one writer (Helios 2301.08419 lines 632-640)."""
    fixture = _Fixture()
    job, memory = _landed_job(fixture, folds_in_place=True)
    sibling = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=5,
        request_key=job.request_key,
    )
    memory.add_reader(sibling)
    with pytest.raises(RuntimeError, match="2 jobs read"):
        fixture.gate.mask_input(job)
