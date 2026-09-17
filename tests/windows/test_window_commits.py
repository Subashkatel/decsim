"""The window commits' laws through a job's on_decoded.

The boundary leaves at decode done, before the frame commit (Skoric et
al. 2209.08552 lines 275-278; qLDPC's net_error,
qldpc/decoders/sinter.py); a strong result replaces the weak
prediction; a window awaiting strong is not final.
"""

import types

import pytest

import decsim.config as config
import decsim.decoders.decoder_output as decoder_output_module
import decsim.decoders.decoders as decoders
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.windows.window_commits as window_commits
import tests.declared_run as declared_run


class _RecordingCourier:
    def __init__(self) -> None:
        self.handed = []
        self.shipped = []

    def hand_on(self, window, _operation, _result, _request_key, is_final):
        self.handed.append((window.key, is_final))

    def ship_held(self, window, _result, _request_key):
        self.shipped.append(window.key)


class _RecordingResults:
    def __init__(self) -> None:
        self.contributions = []
        self.replaced = []
        self.committed = []
        self.delivered = []
        self.released = []

    def install_window_contribution(self, window, logical_observables):
        self.contributions.append((window.key, logical_observables))

    def replace_prediction(self, key, logical_observables):
        self.replaced.append((key, logical_observables))

    def note_window_committed(self, window, is_final):
        self.committed.append((window.key, is_final))

    def deliver_if_final(self, operation):
        self.delivered.append(operation.id)

    def release_committed_segments(self, stream_id):
        self.released.append(stream_id)


class _Frame:
    def __init__(self, engine, commit_ticks: int) -> None:
        self.engine = engine
        self.commit_ticks = commit_ticks
        self.commits = []

    def commit_correction(
        self, *, window_key, logical_observables, request_key, on_committed
    ):
        del request_key
        self.commits.append((self.engine.now, window_key, logical_observables))
        self.engine.schedule(self.commit_ticks, on_committed)


class _Transfers:
    """A link that delivers after a fixed delay."""

    def __init__(self, engine, delay_ticks: int) -> None:
        self.engine = engine
        self.delay_ticks = delay_ticks
        self.sent = []

    def send_for_window(
        self, path, window, _operation, request_key, payload_bits, on_delivered
    ):
        self.sent.append((path, window.key, request_key.tier, payload_bits))
        self.engine.schedule(self.delay_ticks, on_delivered)


class _Fixture:
    def __init__(
        self,
        frame_ticks: int = 3,
        clock=None,
        threshold_cycles=0,
        switch_cycles=0,
    ) -> None:
        self.engine = engine_module.Engine()
        self.window = window_records.Window(
            operation_id=4,
            window_index=1,
            commit_lo=4,
            commit_hi=6,
            buffer_hi=8,
            round_count=5,
        )
        operation = program_records.Operation(4, "logical", (0,), patches=(0,))
        planner = types.SimpleNamespace(windows_by_key={(4, 1): self.window})
        tracker = types.SimpleNamespace(operation_by_id={4: operation})
        self.courier = _RecordingCourier()
        self.results = _RecordingResults()
        self.escalation = types.SimpleNamespace(
            escalate=lambda job: self.escalated.append(job),
            submit_if_commit_releases=lambda key: self.after_weak.append(key),
            cancel_strong_request=lambda key: self.kept.append(key),
        )
        self.escalated = []
        self.after_weak = []
        self.kept = []
        self.resolved = []
        self.verdicts = {}
        self.verdict_ticks = []
        self.policy = types.SimpleNamespace(
            verdict_for_weak_result=self._verdict_for,
            decides_on_a_confidence=True,
        )
        self.reads = []
        self.decode_queue = types.SimpleNamespace(
            resolve_weak_request=self._resolve, read_result=self._read
        )
        self.transfers = _Transfers(self.engine, 4)
        self.frame = _Frame(self.engine, frame_ticks)
        decoder_output = decoder_output_module.DecoderOutput()
        decoder_output.transfers = self.transfers
        decoder_output.frame = self.frame
        committer = window_commits.WindowCommitter(self.engine)
        committer.courier = self.courier
        committer.decoder_output = decoder_output
        committer.results = self.results
        committer.strong_redecode = self.escalation
        self.committer = committer
        self.verdict = window_commits.WindowVerdict(
            self.engine,
            clock=clock,
            threshold_cycles=threshold_cycles,
            switch_cycles=switch_cycles,
        )
        self.verdict.planner = planner
        self.verdict.tracker = tracker
        self.verdict.escalation_policy = self.policy
        self.verdict.strong_redecode = self.escalation
        self.verdict.decode_queue = self.decode_queue
        self.verdict.committer = committer

    def job(self, tier, sequence, awaiting=False) -> decoding_records.DecodeJob:
        request_key = window_records.DecoderRequestKey(4, 1, tier, sequence)
        job = decoding_records.DecodeJob(
            operation_id=4, window_id=1, round_count=5, request_key=request_key
        )
        self.verdicts[request_key] = awaiting
        return job

    def _verdict_for(self, job, _result) -> decoding_records.Verdict:
        self.verdict_ticks.append(self.engine.now)
        if self.verdicts[job.request_key]:
            return decoding_records.Verdict.ESCALATE
        return decoding_records.Verdict.KEEP

    def _resolve(self, job, _result, verdict) -> None:
        self.resolved.append((job.request_key, verdict))

    def _read(self, job) -> None:
        self.reads.append((self.engine.now, job.request_key))


def test_the_boundary_leaves_at_decode_done_before_the_frame_commit():
    fixture = _Fixture()
    job = fixture.job(window_records.DecoderTier.WEAK, 0)
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1, 0))
    fixture.engine.schedule(
        10, lambda: fixture.verdict.accept_result(job, result)
    )
    fixture.engine.run()
    assert fixture.window.t_done == 10
    assert fixture.courier.handed == [((4, 1), True)]
    (path, key, tier, payload_bits) = fixture.transfers.sent[0]
    assert path is transfer_records.LinkPath.WEAK_DECODER_TO_FRAME
    assert (key, tier, payload_bits) == (
        (4, 1),
        window_records.DecoderTier.WEAK,
        2,
    )
    assert fixture.frame.commits == [(14, (4, 1), (1, 0))]
    assert fixture.window.committed
    assert fixture.window.published_request_key is job.request_key
    assert fixture.results.contributions == [((4, 1), (1, 0))]
    assert fixture.results.committed == [((4, 1), True)]
    assert fixture.after_weak == [(4, 1)]
    assert fixture.results.delivered == [4]
    # the result is read at the commit, after the hop and the frame's
    # write, not at the decode's end at 10
    assert fixture.reads == [(17, job.request_key)]


def test_a_window_awaiting_strong_commits_provisionally_and_is_not_final():
    fixture = _Fixture()
    job = fixture.job(window_records.DecoderTier.WEAK, 0, awaiting=True)
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1, 0))
    fixture.verdict.accept_result(job, result)
    assert fixture.transfers.sent == []
    assert fixture.frame.commits == []
    assert fixture.window.committed
    assert fixture.window.published_request_key is None
    assert fixture.courier.handed == [((4, 1), False)]
    assert fixture.results.committed == [((4, 1), False)]
    # a provisional commit sends nothing, so its read is the same instant
    assert fixture.reads == [(0, job.request_key)]


def test_a_strong_result_replaces_the_weak_prediction_and_ships_the_held():
    fixture = _Fixture()
    weak = fixture.job(window_records.DecoderTier.WEAK, 0, awaiting=True)
    weak_result = decoding_records.DecodeResult(
        4, 1, logical_observables=(1, 0)
    )
    fixture.verdict.accept_result(weak, weak_result)
    strong = fixture.job(window_records.DecoderTier.STRONG, 1)
    result = decoding_records.DecodeResult(4, 1, logical_observables=(0, 1))
    fixture.engine.schedule(
        20, lambda: fixture.verdict.accept_strong_result(strong, result)
    )
    fixture.engine.run()
    (path, _key, tier, _bits) = fixture.transfers.sent[0]
    assert path is transfer_records.LinkPath.STRONG_DECODER_TO_FRAME
    assert tier is window_records.DecoderTier.STRONG
    assert fixture.frame.commits == [(24, (4, 1), (0, 1))]
    assert fixture.results.replaced == [((4, 1), (0, 1))]
    assert fixture.window.published_request_key is strong.request_key
    assert fixture.courier.shipped == [(4, 1)]
    assert fixture.results.released == [4]
    # once at the provisional commit, once when the strong result lands
    assert fixture.results.delivered == [4, 4]


def test_a_frameless_run_commits_at_the_delivery():
    fixture = _Fixture()
    decoder_output = decoder_output_module.DecoderOutput()
    decoder_output.transfers = fixture.transfers
    committed = []
    window = fixture.window
    operation = program_records.Operation(4, "logical", (0,), patches=(0,))
    request_key = window_records.DecoderRequestKey(
        4, 1, window_records.DecoderTier.WEAK, 0
    )
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1,))
    decoder_output.publish(
        window,
        operation,
        result,
        request_key,
        lambda: committed.append(fixture.engine.now),
    )
    fixture.engine.run()
    assert committed == [4]


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_threshold_cycles_delay_kept_and_escalated_frame_points(probability):
    clocks = config.ClockSettings.from_yaml({"decisions": 1.0})
    clock = clocks.clock("decisions")
    free = declared_run.switching_run(
        rounds=3, escalation_probability=probability
    )
    charged = declared_run.switching_run(
        rounds=3,
        escalation_probability=probability,
        clock=clock,
        threshold_cycles=3,
    )
    free_ticks = declared_run.reaction_ticks(free)
    charged_ticks = declared_run.reaction_ticks(charged)
    paired = zip(charged_ticks, free_ticks)
    shifts = [charged_tick - free_tick for charged_tick, free_tick in paired]
    expected = 3 * clock.period_ticks
    assert shifts == [0, 0, 0, 0, expected, expected]


def test_switch_cycles_delay_the_strong_request_and_frame_points():
    clocks = config.ClockSettings.from_yaml({"decisions": 1.0})
    clock = clocks.clock("decisions")
    free = declared_run.switching_run(
        rounds=3, escalation_probability=1.0, record=True
    )
    charged = declared_run.switching_run(
        rounds=3,
        escalation_probability=1.0,
        clock=clock,
        switch_cycles=3,
        record=True,
    )
    free_ticks = declared_run.reaction_ticks(free)
    charged_ticks = declared_run.reaction_ticks(charged)
    paired = zip(charged_ticks, free_ticks)
    shifts = [charged_tick - free_tick for charged_tick, free_tick in paired]
    expected = 3 * clock.period_ticks
    assert shifts == [0, 0, 0, 0, expected, expected]
    free_requests = free.observation.decode_records.requests
    charged_requests = charged.observation.decode_records.requests
    strong = window_records.DecoderTier.STRONG
    free_admissions = [
        row.admitted_ticks
        for row in free_requests
        if row.request_key.tier is strong
    ]
    charged_admissions = [
        row.admitted_ticks
        for row in charged_requests
        if row.request_key.tier is strong
    ]
    assert len(free_admissions) == 1
    assert charged_admissions == [free_admissions[0] + expected]


def test_a_kept_verdict_pays_no_switch_cycles():
    clock = config.Clock(1_000_000)
    free = declared_run.switching_run(rounds=3, escalation_probability=0.0)
    charged = declared_run.switching_run(
        rounds=3,
        escalation_probability=0.0,
        clock=clock,
        switch_cycles=3,
    )
    assert declared_run.reaction_ticks(charged) == declared_run.reaction_ticks(
        free
    )


def test_a_verdict_without_confidence_pays_no_threshold_cycles():
    clock = config.Clock(10)
    fixture = _Fixture(clock=clock, threshold_cycles=3)
    fixture.engine.now = 1
    job = fixture.job(window_records.DecoderTier.WEAK, 0, awaiting=True)
    result = decoding_records.DecodeResult(4, 1)

    fixture.verdict.accept_result(job, result)

    assert fixture.engine.idle
    assert fixture.reads == [(1, job.request_key)]


def test_threshold_and_switch_cycles_are_charged_in_sequence():
    clock = config.Clock(10)
    fixture = _Fixture(clock=clock, threshold_cycles=2, switch_cycles=3)
    fixture.engine.now = 1
    job = fixture.job(window_records.DecoderTier.WEAK, 0, awaiting=True)
    source = decoders.SAMPLED_CONFIDENCE_SOURCE
    soft_output = decoding_records.SoftOutput(0.0, source)
    result = decoding_records.DecodeResult(4, 1, soft_output=soft_output)

    fixture.verdict.accept_result(job, result)
    fixture.engine.run()

    assert fixture.verdict_ticks == [30]
    assert fixture.reads == [(60, job.request_key)]
