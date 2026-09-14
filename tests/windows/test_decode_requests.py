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

import decsim.collect as collect
import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoder_memory_transfer as decoder_memory_transfer
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.front.experiment as experiment
import decsim.front.measure as measure
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
import tests.front.yaml_configs as yaml_configs


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
        input_fold = decoder_memory_transfer.DecoderInputStaging(
            None, self.engine
        )
        gate = decode_requests.WindowInputGate(
            self.planner, interaction, input_fold
        )
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
        self.interaction = interaction

    def deliver_boundary(self, defects: dict) -> None:
        """The window's one predecessor hands it this seam mask.

        Through the interaction, because the state a fold reads is the
        interaction's own and carries which sources contributed to it.
        """
        info = window_records.WindowInfo.from_window(self.window)
        delivery = window_records.BoundaryDelivery(
            source_key=(1, 9),
            destination_key=(1, 0),
            source_revision=1,
            delivery_revision=0,
            latest_source_revision=1,
            latest_delivery_revision=0,
            source_operation_round_count=6,
            dependency_released=True,
            payload=defects,
        )
        empty = self.interaction.initial_boundary_state(info)
        update = self.interaction.merge_boundary(delivery, info, empty)
        self.window.boundary_in = update.state

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
    fixture.deliver_boundary({5: [1, 0, 1]})
    for round_index in (1, 2, 3, 4, 5):
        fixture.arrive(round_index)
    (job, _send_input) = fixture.queue.enqueued[0]
    assert fixture.gate.may_start(job) is False
    raw = job.payloads[4]
    assert raw.bits is None
    landed = decoder_memory.MaterializedSyndromeRound(1, 5, (raw,))
    job.decoder_input = decoder_memory.DecoderInput(
        1, 0, job.request_key, (landed,)
    )
    job.memory = decoder_memory.DecoderMemory("default", 0, None)
    fixture.gate.mask_input(job)
    (masked,) = job.decoder_input.rounds[0].fragments
    assert masked.bits == (1, 0, 1)
    fixture.window.deps_remaining = 0
    assert fixture.gate.may_start(job) is True


def test_a_boundary_that_flipped_nothing_is_still_folded():
    """The fold follows the arrival, not the bits the arrival carried.

    cuda-q QEC applies the accumulated syndrome mods to every window
    past the first whatever they hold (sliding_window.cpp:283-292), so
    an all-zero seam still costs the window its masked view; the rounds
    it reads come out of the fold unchanged.
    """
    fixture = _Fixture()
    fixture.window.deps = [(1, 9)]
    fixture.window.deps_remaining = 1
    fixture.deliver_boundary({5: [0, 0, 0]})
    for round_index in (1, 2, 3, 4, 5):
        fixture.arrive(round_index)
    (job, _send_input) = fixture.queue.enqueued[0]
    raw = _fragment(5, bits=(1, 0, 1))
    landed = decoder_memory.MaterializedSyndromeRound(1, 5, (raw,))
    job.decoder_input = decoder_memory.DecoderInput(
        1, 0, job.request_key, (landed,)
    )
    job.memory = decoder_memory.DecoderMemory("default", 0, None)
    folded = job.decoder_input

    fixture.gate.mask_input(job)

    (masked,) = job.decoder_input.rounds[0].fragments
    assert job.decoder_input is not folded
    assert masked.bits == (1, 0, 1)


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
    fixture.deliver_boundary({5: [1, 0, 1]})
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
    (masked,) = job.decoder_input.rounds[4].fragments
    assert masked.bits == (1, 0, 1)
    (raw,) = resident.rounds[4].fragments
    assert raw.bits is None
    assert job.decoder_input is not resident


def test_the_in_place_fold_writes_the_units_own_memory():
    """AFS 2001.06598 lines 528-531: the unit's memory is written directly."""
    fixture = _Fixture()
    job, memory = _landed_job(fixture, folds_in_place=True)
    fixture.gate.mask_input(job)
    resident = memory.input_of(job)
    assert job.decoder_input is resident
    (masked,) = resident.rounds[4].fragments
    assert masked.bits == (1, 0, 1)


def test_a_shared_input_is_written_once_and_the_other_solve_reads_it():
    """One input has one writer (Helios 2301.08419 lines 632-640).

    The jobs that share one landed input are the two forced-class solves
    of one window's request (decision D2), so the boundary they fold is
    that window's one boundary: the solve that starts first writes the
    mask into the unit's memory and the other reads exactly those
    rounds, which is what the copy fold gives each of them too.
    """
    fixture = _Fixture()
    job, memory = _landed_job(fixture, folds_in_place=True)
    sibling = _sibling_solve(job, memory)

    fixture.gate.mask_input(job)
    fixture.gate.mask_input(sibling)

    resident = memory.input_of(job)
    assert sibling.decoder_input is resident
    (masked,) = resident.rounds[4].fragments
    assert masked.bits == (1, 0, 1)


def test_an_input_that_carries_its_mask_is_not_masked_again():
    """A second write would be a second mask over the first.

    The memory that holds the input refuses it, so a caller that folds
    twice is told rather than decoding rounds the boundary has been
    XORed into twice (Helios 2301.08419 lines 632-640).
    """
    fixture = _Fixture()
    job, memory = _landed_job(fixture, folds_in_place=True)
    fixture.gate.mask_input(job)

    with pytest.raises(RuntimeError, match="already written"):
        memory.rewrite(job, job.decoder_input)


def _sibling_solve(job, memory):
    """The window's other forced-class solve, reading the same input."""
    sibling = decoding_records.DecodeJob(
        operation_id=job.operation_id,
        window_id=job.window_id,
        round_count=job.round_count,
        request_key=job.request_key,
        window=job.window,
        memory=memory,
    )
    sibling.decoder_input = memory.add_reader(sibling)
    return sibling


# one sweep point per noise level, counting the movement each shot made
FOLD_COUNTING_SWEEP = {
    "observation": {"data_movement": True},
    "sweep": [
        {
            "physical_error_probability": [0.001, 0.01],
            "distance": [3],
            "round_period_us": [1.0],
            "shots": 1,
        }
    ],
}


FOLD_SEEDS = (0, 1, 2, 3, 4)


@pytest.mark.parametrize("probability", (0.001, 0.01))
def test_every_window_with_a_predecessor_folds_its_boundary(
    tmp_path, probability
):
    """The fold's count is the geometry's, not the noise's.

    Fifteen rounds at distance 3 are four sliding windows, so three of
    them have a predecessor and three masked views are made, at a noise
    level where almost no seam carries a defect and at one where many
    do. The rounds come out of a fold that changed nothing unchanged, so
    every one of five seeds still agrees with whole-circuit PyMatching.
    """
    config_path = yaml_configs.write_config(tmp_path, FOLD_COUNTING_SWEEP)
    config = experiment.load_experiment(config_path)
    for seed in FOLD_SEEDS:
        measurement = yaml_configs.measure_point_shot(
            config,
            physical_error_probability=probability,
            distance=3,
            round_period_us=1.0,
            seed=seed,
        )
        by_path = measurement.data_movement["copies_by_path"]
        folds = by_path["unit default#0 memory -> masked view"]

        assert measurement.windows == 4
        assert folds["events"] == measurement.windows - 1
        assert folds["rounds"] == 18
        assert measurement.direct_mismatch is False


# switching, so every window's request runs the two forced-class solves
# of a complementary gap and they read one landed input (decision D2)
SWITCHING_FOLD_SWEEP = {
    "escalation": {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "two_sided_context",
        "run_both_at_once": False,
    },
    "strong_decoder": {
        "kind": 1.0,
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 1,
        },
    },
    "sweep": [
        {
            "physical_error_probability": [0.0001, 0.008],
            "distance": [3],
            "round_period_us": [1.0],
            "shots": len(FOLD_SEEDS),
        }
    ],
}


@pytest.mark.parametrize("probability", (0.0001, 0.008))
def test_two_solves_of_one_request_fold_one_mask_into_one_input(
    tmp_path, probability
):
    """The in-place fold under switching answers what the copy fold does.

    The two forced-class solves of one window read one landed input
    (decision D2) and fold one window's boundary, so the mask is written
    into the unit's memory once and both solves read those rounds: one
    input, one writer (Helios 2301.08419 lines 632-640). Five seeds at
    each of two noise levels, one where almost no seam carries a defect
    and one where many do: every window's committed logical observables,
    the shot's logical outcome and its agreement with whole-circuit
    PyMatching are what the copy fold reaches on the same seed.
    """
    in_place = _fold_run(tmp_path, "in_place", probability)
    copied = _fold_run(tmp_path, "copy", probability)

    assert in_place == copied


def _fold_run(tmp_path, boundary_fold: str, probability: float) -> list:
    """Each seed's committed corrections and verdicts under that fold row."""
    overrides = dict(SWITCHING_FOLD_SWEEP)
    weak_decoder = dict(yaml_configs.MINIMAL_CONFIG["weak_decoder"])
    weak_decoder["input"] = "copy"
    weak_decoder["boundary_fold"] = boundary_fold
    overrides["weak_decoder"] = weak_decoder
    directory = tmp_path / boundary_fold
    directory.mkdir()
    config_path = yaml_configs.write_config(directory, overrides)
    config = experiment.load_experiment(config_path)
    outcomes = []
    for seed in FOLD_SEEDS:
        outcome = _fold_outcome(config, probability, seed)
        outcomes.append(outcome)
    return outcomes


def _fold_outcome(config, probability: float, seed: int) -> tuple:
    """One shot's committed corrections, its outcome and its agreement."""
    task = config.point_task(
        physical_error_probability=probability,
        distance=3,
        round_period_us=1.0,
        shots=len(FOLD_SEEDS),
    )
    shot = collect.run_shot(task, seed)
    measurement = measure.measure_shot(shot)
    corrections = []
    for record in shot.machine.observation.frame_corrections.committed:
        corrections.append((record.window_key, record.logical_observables))
    return (
        sorted(corrections),
        measurement.logical_failure,
        measurement.direct_mismatch,
    )
