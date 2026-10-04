"""One collected shot -> one shot's numbers. Nothing is built here.

measure_shot reads the run's listeners, its RunResult, and the code card
and cadence its QPU ran, and no other component. Transfers are named by
role, not by wire; the tier that decodes the plan's windows decides
which wire carries each role (MachineSettings.window_tier).
"""

import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from typing import Optional, Union

import decsim.collect as collect
import decsim.config as config_module
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_output as decoder_output
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.observe.observation as observation_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.results as result_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings

# Latency points, in path order, in microseconds per window unless noted.
# Every point is one span, and its comment names the two ticks it runs
# between.
POINTS = (
    # per round: the controller's send -> the round readable in the weak
    # syndrome buffer
    "cwb_per_round",
    # per round: the round found the weak syndrome buffer full -> the slot that
    # freed admitted it, zero for a round that found room
    "cwb_stall_per_round",
    # per round: the same wait in front of the strong syndrome buffer
    "csb_stall_per_round",
    # the window's first round readable -> its last (waiting on the QPU)
    "buffer_fill",
    # that last round readable -> the job entered the decode queue: the
    # window side's decision (windows.decision_cycles) and any earlier
    # request of the window that was withdrawn before it, as a restart
    # window's is when a forward strong window re-slices it. Ciw keeps
    # a customer's pre-service wait as a field of its own and writes a
    # visit that left the queue unserved as its own record (ciw
    # data_record.py lines 3-21, node.py write_reneging_record)
    "admission_wait",
    # the committing decode's input read in its tier's store: the
    # dispatch that asked for it -> the read's end, its port waits
    # included, before the input hop
    "store_read",
    # the dependency wait around the committing decode's own hop: from
    # where its path started, the dispatch or the verdict that escalated
    # the window, to the first tick it may compute, less the input hop
    # and the store's read. It is the wait for the escalated rounds to
    # land in the strong store before the input hop, the predecessor's
    # boundary and the escalation's selection after it, and zero when
    # nothing was owed
    "dep_block",
    # that first startable tick -> the compute started: the wait for the
    # unit's own compute, busy with another decode (gem5's fuBusy)
    "compute_wait",
    # the job entered the decode queue -> a unit took the window's first
    # decode
    "queue_wait",
    # the committing decode's own input hop, from its tier's store into
    # the unit's memory: zero when that decode read an input already
    # there, which is a tier reading in place and the second forced
    # solve reading what the first one brought
    "input_link_per_window",
    # the compute start -> the fetch stage's end (the unit reads the
    # window out of its own memory)
    "fetch",
    "algorithm",  # the fetch's end -> the decoding algorithm's end
    # inside the algorithm: the ticks a strong backend's decode waited
    # for its dispatcher or a worker (decoders/strong_backend.py), summed
    # as Ciw writes a customer's wait (ciw node.py lines 856-862). The
    # algorithm less this is the device's own steps: launch, copies and
    # kernel. Zero for a decoder with no queue of its own
    "backend_queue_wait",
    "release",  # the algorithm's end -> the correction written out
    # the compute start -> the decode's end: every stage the unit ran,
    # and nothing the decode waited for outside the unit; a strong
    # backend's queue holds the unit, so backend_queue_wait is inside
    "service",
    # the committing decode's end -> the verdict on the window's answer:
    # the confidence signal's own computation, which is the walk under
    # cluster_gap and the other forced-class solve's remaining time under
    # complementary_gap. Zero for an escalated window whose strong decode
    # began after the verdict, since its weak_attempt already runs to it
    "confidence",
    # the verdict, or the committing decode's end when later -> its answer
    # leaving for the frame: a finished strong answer kept in its unit's
    # output slot (decoders/decoder_unit.py hold_output) until the
    # verdict's selection has crossed weak_decoder_to_strong_decoder, the
    # wait a strong decode started with the weak one under
    # run_both_at_once has; zero for every other window
    "selection_wait",
    # a unit took the window's first decode -> the verdict that
    # escalated it: the weak attempt whose result did not commit, zero
    # for a window the first decode committed
    "weak_attempt",
    # weak decoder -> strong decoder, the escalation hop: from its first
    # transfer's request to its last transfer's delivery, the selection and
    # then the rounds the strong store lacked, which the strong input
    # hop waits for; zero for a window that did not escalate. Under
    # run_both_at_once the rounds cross at the weak dispatch and the
    # selection at the verdict, and the span covers both
    "escalation_link_per_window",
    # the decode's end -> the boundary readable in the next window
    "dd_per_window",
    # the decode's end -> the correction at the Pauli frame
    "output_link_per_window",
    "frame_commit",  # the frame accepted the correction -> committed
    # Totals. The buffer0 pair starts the clock at the publication in the
    # syndrome buffer the first decode reads; the qpu pair starts it when
    # the round leaves the QPU (the QC send), so it includes QC,
    # controller processing, packing and CWB.
    "buffer0_ready_to_frame",  # window complete in the buffer -> frame
    "buffer0_first_round_to_frame",  # first round in the buffer -> frame
    "qpu_last_round_to_frame",  # last required round off QPU -> frame
    "qpu_first_round_to_frame",  # first required round off QPU -> frame
)

# the points sampled once a round, not once a window, so no committing
# tier names their samples
ROUND_POINTS = ("cwb_per_round", "cwb_stall_per_round", "csb_stall_per_round")

# One count column per status a committed window's decode may carry
# besides success (records/decoding.py BackendDecodeStatus), so a status
# the enum gains is counted with no change here; sinter keeps its
# custom counts the same way, one named counter per kind
# (sinter/_data/_task_stats.py:71)
WINDOW_STATUS_COLUMNS = tuple(
    f"{status.value}_windows"
    for status in decoding_records.BackendDecodeStatus
    if status is not decoding_records.BackendDecodeStatus.SUCCEEDED
)

# which store's output link carries a tier's window in; the way home is
# the decoders' own FRAME_PATH_BY_TIER, and both are keyed by the tier
# whose decode the frame committed rather than by the row's name
INPUT_LINK_BY_TIER = {
    window_records.DecoderTier.WEAK: (
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER
    ),
    window_records.DecoderTier.STRONG: (
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER
    ),
}


@dataclasses.dataclass(frozen=True)
class ShotConfidence:
    """A shot's windows' confidence gaps, when a signal decided escalation.

    is_sampled says whether the shot is among the first
    observation.confidence_shot_count, whose windows window_confidence.csv
    lists, while the histogram counts every scored shot. piece.json records
    sampled_shot_count so a fold can tell pieces that sampled different
    shots apart.
    """

    signal: str
    windows: tuple
    is_sampled: bool
    sampled_shot_count: Optional[int]


@dataclasses.dataclass(frozen=True)
class ShotMeasurement:
    """One shot's numbers; the field names are the csv columns."""

    # the point's strong id, sinter's strong_id column
    # (sinter/_data/_csv_out.py:69-77); the report adds its swept values
    point_id: str
    algorithm: object  # the active unit's card: a name or a latency in us
    seed: int
    decoded_windows: int
    logical_failure: bool
    # the patch-rounds the scored owners read out, added up; the owners
    # scored, each an independent output the shot fails on when it is
    # wrong; and the patch-rounds each ran, 0 when they ran apart. A
    # per-round rate reads the last two (failure_statistics
    # per_output_round_rate)
    executed_rounds: int
    scored_outputs: int
    rounds_per_output: int
    samples: dict  # point -> us list, one per window (per round for cwb)
    # the tier whose decode the frame committed, one per window in the
    # samples' order: weak for a kept window, strong for an escalated one
    # on a run the weak tier decodes, as Ciw writes each record's
    # customer class (ciw node.py lines 856-862)
    window_tiers: tuple
    means: dict  # point -> mean us over this shot's windows
    maxes: dict  # point -> max us
    load: float  # service per window / window inter-arrival
    throughput_windows_per_us: float
    throughput_rounds_per_us: float
    max_queued_windows: int
    # the deepest each tier's own ready queue got, in jobs: the max
    # backlog per decoder instance DART-Q reports (2605.09142 lines
    # 1101-1109); zero for a tier the run does not have
    weak_queue_max: int
    strong_queue_max: int
    # each tier's time-weighted fraction of busy units, Triage's
    # utilization rate (2605.04459 lines 1024-1031)
    weak_busy_fraction: float
    strong_busy_fraction: float
    # the windows the strong tier committed, the rounds its decodes read
    # and their summed service, and r_com, the rounds a window commits.
    # The report divides the summed service by the summed windows per
    # sweep point, a ratio of two sums as gem5 forms avgMissLatency =
    # missLatency / misses (src/mem/cache/base.cc:2188), and forms
    # Toshio's Theorem 1 bound on that mean from the windows and r_com
    # (2510.25222 eq. (6))
    escalated_windows: int
    strong_decoded_rounds: int
    strong_service_sum_us: float
    commit_rounds: int
    # tau_gen r_com: a window's inter-arrival, commit rounds times the
    # round period, which the chain's load divides by
    window_period_us: float
    # Skoric's least count of parallel decoding processes, ceil(2 tau_W
    # / ((n_com + n_W) tau_rd)) (2209.08552 lines 429-438)
    parallel_processes_needed: int
    # what tells harder windows from an overloaded strong side in a
    # paired run: the set bits of each weak decode's input, its
    # detection events when they are formed ahead of the decoder; each
    # weak decode's compute, its stages' span;
    # each strong decode's wait from its enqueue to its compute start,
    # the queueing delay a queue splits from service; the most strong
    # decodes held in the units' memory at once, landed and free to
    # compute but waiting on a unit's compute, which the strong ready
    # queue's peak does not see because a unit takes the next decode
    # into its memory while it computes; and the most undecoded rounds
    # the machine held at once. The first six need
    # observation.record_switching_windows and the last
    # observation.backlog_trace; a shot that kept neither holds None,
    # and its row no column, since zeros would say nothing waited
    weak_syndrome_weight_mean: Optional[float]
    weak_syndrome_weight_max: Optional[int]
    weak_service_mean_us: Optional[float]
    strong_wait_mean_us: Optional[float]
    strong_wait_max_us: Optional[float]
    strong_held_in_units_max: Optional[int]
    backlog_peak_rounds: Optional[int]
    referee_windows_checked: int  # referee re-decodes (0 = referee off)
    # referee reached a different owned observable contribution
    referee_window_disagreements: int
    # path -> the run's own ledger counters plus rounds/windows context,
    # for shot_links.csv; the totals come straight off the ledger's counters
    link_totals: dict
    sim_wall_seconds: float
    # the shot's copies, references and moves as the RunResult carries
    # them, None when observation.data_movement is off, so a run that
    # counted none writes no data-movement row rather than a row of zeros
    data_movement: Optional[dict]
    # WINDOW_STATUS_COLUMNS -> the committed windows whose decode carried
    # that status; shots.csv holds one column per entry
    window_statuses: dict
    # whether every decode a window committed, provisional or final, got
    # a correction from its backend; an unscored shot is sinter's
    # discard, never a logical failure (sinter/_decoding/_decoding.py:
    # 123-125), and unscored_reason names the backends' reasons, empty on
    # a scored shot. provisional_no_correction_windows counts the windows
    # whose replaced provisional decode had none, which the status
    # columns, counting final decodes, do not show
    is_scored: bool
    unscored_reason: str
    provisional_no_correction_windows: int
    # sha256 of every scored owner's sampled detection events and
    # observable truth, so two points' shots of one seed are checked to
    # be one draw
    sample_digest: str
    # the windows' confidence gaps, None when no confidence signal
    # decided the escalation; its own files hold it, not shots.csv
    confidence: Optional[ShotConfidence]
    # every operation's predicted observables, so two decoders' shots of
    # one seed compare answer by answer and not only failure by failure
    predictions: str


def measure_shot(
    shot: collect.Shot,
    run_dir: Optional[pathlib.Path] = None,
    *,
    only_traced_shot: bool = False,
) -> ShotMeasurement:
    """Read one collected shot's numbers off its machine and result.

    run_dir receives the log file and the Chrome trace when asked; None
    writes nothing. only_traced_shot writes a named trace path as it stands.
    """
    settings = shot.task.settings
    point_id = shot.task.strong_id()
    label = shot_label(point_id, shot.seed)
    observation = shot.machine.observation
    if run_dir is not None and settings.observation.writes_log:
        run_folder.write_log(observation, run_dir, label)
    if run_dir is not None:
        _write_trace(shot, run_dir, label, only_traced_shot)
    device = shot.machine.qpu.device
    round_period_microseconds = config_module.ticks_to_microseconds(
        device.clock.period_ticks
    )
    return _measurement(
        settings,
        observation,
        shot.result,
        record_options=shot.task.record_options,
        code=device.code,
        round_period_microseconds=round_period_microseconds,
        point_id=point_id,
        seed=shot.seed,
        wall_seconds=shot.wall_seconds,
    )


def link_totals(traffic: dict) -> dict:
    """The ledger's own counters for this shot by path, in microseconds."""
    counters_by_path = {}
    for edge in traffic["semantic_edges"]:
        empty_counters = collections.Counter()
        counters = counters_by_path.setdefault(edge["path"], empty_counters)
        counters.update(edge["counters"])
    totals = {}
    for path, counters in counters_by_path.items():
        totals[path] = {
            "transfers": counters["transfer_count"],
            "payload_bits": counters["known_payload_bits"],
            "unknown_payload_transfers": counters[
                "unknown_payload_transfer_count"
            ],
            "queue_wait_us": config_module.ticks_to_microseconds(
                counters["queue_wait_ticks"]
            ),
            "serialization_us": config_module.ticks_to_microseconds(
                counters["serialization_ticks"]
            ),
            "propagation_us": config_module.ticks_to_microseconds(
                counters["propagation_ticks"]
            ),
        }
    return totals


def link_delay_by_window(transfers: list) -> dict:
    """Ticks from the first request to the last delivery by (path, window key).

    A hop's transfers for one window can overlap (the selection and its
    rounds), so the span is what the window waited and a sum would count
    the overlap twice. A window is keyed by (operation_id, window_index),
    since several streams each have a window 3, as gem5's reorder buffer
    finds an instruction by thread and sequence number
    (src/cpu/o3/rob.hh:131-134).
    """
    first_request = {}
    last_delivery = {}
    for row in transfers:
        attribution = row["attribution"]
        recorded_operation = attribution["operation_id"]
        operation_id = identity_records.stable_identity_from_json(
            recorded_operation
        )
        key = (row["path"], operation_id, attribution["window_id"])
        request = _hop_start_ticks(row)
        earliest_request = first_request.get(key, request)
        first_request[key] = min(earliest_request, request)
        delivery = row["delivery_ticks"]
        latest_delivery = last_delivery.get(key, delivery)
        last_delivery[key] = max(latest_delivery, delivery)
    delay = {}
    for key, request in first_request.items():
        delay[key] = last_delivery[key] - request
    return delay


def input_hop_by_request(transfers: list) -> dict:
    """(path, window id, run ordinal) -> (delay ticks, last delivery tick).

    A window decoded more than once has one entry per decode, so a point
    reads the hop of the decode it describes. The landing is the latest
    delivery, since a decode starts once every input transfer landed.
    """
    hops = {}
    for row in transfers:
        run_sequence = _request_run_sequence(row)
        if run_sequence is None:
            continue
        key = (row["path"], row["attribution"]["window_id"], run_sequence)
        delay, landing = hops.get(key, (0, 0))
        delay += row["total_delay_ticks"]
        landing = max(landing, row["delivery_ticks"])
        hops[key] = (delay, landing)
    return hops


def controller_to_weak_buffer_delays_us(transfers: list) -> list:
    """Every round's controller_to_weak_buffer delay, in microseconds."""
    delays = []
    for row in transfers:
        if row["path"] != "controller_to_weak_buffer":
            continue
        delay = config_module.ticks_to_microseconds(row["total_delay_ticks"])
        delays.append(delay)
    return delays


def round_stall_ticks(round_events: list) -> dict:
    """Ticks each held round waited for store room, by (operation, round).

    The wait is the waiting line's own two events: the refusal that held
    the round and the freed slot that admitted it
    (controller/syndrome_round_sender.py, HeldRounds). A round that found
    room is not in here at all.
    """
    held_at = {}
    waits = {}
    for event in round_events:
        key = (event.operation_id, event.round_index)
        if event.kind == "STALLED":
            held_at[key] = event.tick
        if event.kind == "RELEASED":
            held_tick = held_at[key]
            waits[key] = event.tick - held_tick
    return waits


def round_stall_delays_us(round_keys: list, waits: dict) -> list:
    """The wait for store room of every round that reached one store.

    One sample per round, in the order the rounds reached it, zero for a
    round that found room, so the point reads beside that store's own
    per-round hop.
    """
    delays = []
    for key in round_keys:
        ticks = waits.get(key, 0)
        delay = config_module.ticks_to_microseconds(ticks)
        delays.append(delay)
    return delays


def weak_store_round_keys(round_events: list) -> list:
    """Every round that left for the weak syndrome buffer, in leaving order."""
    keys = []
    for event in round_events:
        if event.kind != "CWB_SENT":
            continue
        key = (event.operation_id, event.round_index)
        keys.append(key)
    return keys


def strong_store_round_keys(stored_rounds: list) -> list:
    """Every round landed in the strong syndrome buffer, in landing order."""
    keys = []
    for _tick, operation_id, round_index in stored_rounds:
        key = (operation_id, round_index)
        keys.append(key)
    return keys


def qpu_send_ticks(transfers: list) -> dict:
    """The tick each round left the QPU, by (operation, round).

    A round index counts within its operation's stream, so a round is found
    by both, as gem5 finds an instruction (src/cpu/o3/rob.hh:131-134); its
    send is its earliest QC request.
    """
    send = {}
    for row in transfers:
        if row["path"] != "qpu_to_controller":
            continue
        (rounds,) = row["attribution"]["rounds_by_operation"]
        operation_id = identity_records.stable_identity_from_json(
            rounds["operation_id"]
        )
        round_key = (operation_id, rounds["round_lo"])
        start = _hop_start_ticks(row)
        earlier = send.get(round_key, start)
        send[round_key] = min(earlier, start)
    return send


def window_points_us(
    window: window_records.Window,
    frame_record: decoding_records.PauliFrameCommitRecord,
    decode: "_CommittedDecode",
    first_dispatch: int,
    stage_us: dict,
    link_delay: dict,
    input_hop: dict,
    qpu_send: dict,
    input_path: str,
    output_path: str,
) -> dict:
    """The per-window latency points, in us, for one decoded window.

    Every tick belongs to one decode. The window waits in the ready queue
    until a unit takes its first decode; a window the strong tier recovered
    then spends its weak attempt, ending at the verdict (Toshio et al.
    2510.25222 Sec. III A steps 2 to 4); the input hop, the park, the compute
    and the way home are the committing decode's own. A window decoded more
    than once (two forced-class solves) has several dispatches, and the
    window record keeps only the last, so they must not be mixed.

    The decode's time and its wait are separate points, and the wait is two
    by cause. dep_block runs from the verdict to the input hop's request
    (escalated rounds landing in the strong store) and from the input
    landing to the first tick the decode may start (the predecessor's
    boundary). compute_wait is the rest, the unit busy with another decode.
    Skoric et al. 2209.08552 keep tau_W and tau_0 apart (lines 429-435,
    1004-1008); gem5 counts a ready instruction with no free unit apart
    (inst_queue.cc:1009-1014); Ciw keeps the pre-service wait in named parts
    (ciw/data_record.py lines 3-21).
    """
    operation_id, window_id = window.key
    last_emitted_round = _last_emitted_round(qpu_send, operation_id)
    last_required_round = min(window.buffer_hi, last_emitted_round)
    committed = frame_record.committed_ticks
    handoff_ticks = link_delay.get(
        ("decoder_to_decoder", operation_id, window_id), 0
    )
    escalation_ticks = link_delay.get(
        ("weak_decoder_to_strong_decoder", operation_id, window_id), 0
    )
    output_ticks = link_delay.get((output_path, operation_id, window_id), 0)
    attempt_end = _attempt_end_ticks(window, decode, first_dispatch)
    input_key = (input_path, window_id, decode.run_sequence)
    input_ticks, input_landed = input_hop.get(input_key, (0, attempt_end))
    input_sent = input_landed - input_ticks
    startable = _startable_ticks(decode, input_landed)
    park = _span_microseconds(startable, input_landed)
    rounds_wait = _span_microseconds(input_sent, attempt_end)
    store_read = config_module.ticks_to_microseconds(decode.store_read_ticks)
    dep_block_with_read = rounds_wait + park
    confidence_ticks = _confidence_ticks(window, decode)
    output_sent = frame_record.accepted_ticks - output_ticks
    answered = max(window.t_done, decode.done_ticks)
    last_required_send = qpu_send[(operation_id, last_required_round)]
    first_required_send = qpu_send[(operation_id, window.start_round)]
    return {
        "buffer_fill": _span_microseconds(
            window.t_data_complete, window.t_first_round
        ),
        "admission_wait": _span_microseconds(
            window.t_queued, window.t_data_complete
        ),
        "store_read": store_read,
        "dep_block": dep_block_with_read - store_read,
        "compute_wait": _span_microseconds(
            decode.compute_start_ticks, startable
        ),
        "queue_wait": _span_microseconds(first_dispatch, window.t_queued),
        "input_link_per_window": config_module.ticks_to_microseconds(
            input_ticks
        ),
        "fetch": stage_us["fetch"],
        "algorithm": stage_us["algorithm"],
        "backend_queue_wait": config_module.ticks_to_microseconds(
            decode.backend_queue_wait_ticks
        ),
        "release": stage_us["release"],
        "service": _span_microseconds(
            decode.done_ticks, decode.compute_start_ticks
        ),
        "confidence": config_module.ticks_to_microseconds(confidence_ticks),
        "selection_wait": _span_microseconds(output_sent, answered),
        "weak_attempt": _span_microseconds(attempt_end, first_dispatch),
        "escalation_link_per_window": config_module.ticks_to_microseconds(
            escalation_ticks
        ),
        "dd_per_window": config_module.ticks_to_microseconds(handoff_ticks),
        "output_link_per_window": config_module.ticks_to_microseconds(
            output_ticks
        ),
        "frame_commit": _span_microseconds(
            committed, frame_record.accepted_ticks
        ),
        "buffer0_ready_to_frame": _span_microseconds(
            committed, window.t_data_complete
        ),
        "buffer0_first_round_to_frame": _span_microseconds(
            committed, window.t_first_round
        ),
        "qpu_last_round_to_frame": _span_microseconds(
            committed, last_required_send
        ),
        "qpu_first_round_to_frame": _span_microseconds(
            committed, first_required_send
        ),
    }


def frame_records_by_window(
    observation: observation_module.Observation,
) -> dict:
    """The frame's committed corrections, by (operation_id, window_index).

    Keyed by the index alone, several streams would keep one stream's
    correction per index and measure the others against the wrong decode.
    """
    records = {}
    for record in observation.frame_corrections.committed:
        records[record.window_key] = record
    return records


def collect_samples(
    observation: observation_module.Observation,
    result: result_records.RunResult,
) -> tuple:
    """Every point's microsecond samples, and each window's committing tier.

    Every point describes the decode the frame committed, named by the
    frame record's tier and run ordinal. It stays whole past the size
    prompt: one walk over the windows in key order, read top to bottom.
    """
    transfers = result.link_traffic["transfers"]
    link_delay = link_delay_by_window(transfers)
    input_hop = input_hop_by_request(transfers)
    qpu_send = qpu_send_ticks(transfers)
    frame_by_window = frame_records_by_window(observation)
    samples = {}
    for point in POINTS:
        samples[point] = []
    window_tiers = []
    samples["cwb_per_round"] = controller_to_weak_buffer_delays_us(transfers)
    round_events = observation.round_events
    waits = round_stall_ticks(round_events.events)
    weak_keys = weak_store_round_keys(round_events.events)
    strong_keys = strong_store_round_keys(round_events.stored_rounds)
    samples["cwb_stall_per_round"] = round_stall_delays_us(weak_keys, waits)
    samples["csb_stall_per_round"] = round_stall_delays_us(strong_keys, waits)
    window_items = observation.windows.windows.items()
    all_windows = sorted(window_items, key=_window_order)
    stages = observation.stages
    for window_key, window in all_windows:
        frame_record = frame_by_window.get(window_key)
        if frame_record is None or window.t_done is None:
            continue
        decode = _committed_decode(stages, frame_record)
        first_dispatch = _first_dispatch_ticks(stages, window, frame_record)
        input_link = INPUT_LINK_BY_TIER[decode.tier]
        output_link = decoder_output.FRAME_PATH_BY_TIER[decode.tier]
        input_path = input_link.value
        output_path = output_link.value
        stage_us = _stage_microseconds(stages, frame_record)
        points = window_points_us(
            window,
            frame_record,
            decode,
            first_dispatch,
            stage_us,
            link_delay,
            input_hop,
            qpu_send,
            input_path,
            output_path,
        )
        for point, value in points.items():
            samples[point].append(value)
        window_tiers.append(decode.tier.value)
    return samples, tuple(window_tiers)


def chain_load(samples: dict, window_period_us: float) -> float:
    """rho: the serial chain's service per window over the window period.

    Service is the unit's occupancy per window plus the boundary handoff
    that serialises the chain; the period is commit rounds times the round
    period. Above 1 the chain cannot keep up: Skoric et al. 2209.08552's
    backlog condition as a ratio (lines 429-435). The occupancy includes
    the confidence step, charged on the same weak unit, and under
    complementary_gap the second forced-class solve.
    """
    service_us = _mean_or_zero(samples["service"])
    confidence_us = _mean_or_zero(samples["confidence"])
    handoff_us = _mean_or_zero(samples["dd_per_window"])
    chain_us = service_us + confidence_us + handoff_us
    return chain_us / window_period_us


def active_decoder_kind(
    settings: machine_settings.MachineSettings,
) -> Optional[Union[str, float]]:
    """The kind of the tier that decodes the plan's windows."""
    tier = settings.window_tier.value
    tier_settings = settings.decoder_settings_for(tier)
    if tier_settings is None:
        return None
    return tier_settings.algorithm.name


def trace_path_for_shot(path: Union[str, pathlib.Path], label: str) -> str:
    """The path a swept shot writes to: the label joins the given path.

    run.trace.json becomes run_<label>.trace.json, the label holding the
    point id and seed, so two points that differ only in a setting no file
    name spells never write one file (gem5's multisim names each output
    folder by its id).
    """
    name = pathlib.Path(path)
    directory = name.parent
    stem = name.name
    first_dot = stem.find(".")
    if first_dot < 0:
        labelled = directory / f"{stem}_{label}"
        return str(labelled)
    head = stem[:first_dot]
    suffixes = stem[first_dot:]
    labelled = directory / f"{head}_{label}{suffixes}"
    return str(labelled)


def shot_label(point_id: str, seed: int) -> str:
    """The name a shot's log and trace files carry: its point id and seed."""
    return f"{point_id}_seed{seed}"


def parallel_processes_needed(
    samples: dict, code: ports.CodeModel, round_period_microseconds: float
) -> int:
    """Skoric's least count of parallel decoding processes for no backlog.

    N_par >= 2 tau_W / ((n_com + n_W) tau_rd) (2209.08552 lines 429-438;
    NVQLink equation 2, 2510.25213), with n_W = n_com + 2 n_buf ("nW = 3w",
    lines 388-390), tau_W this shot's mean service and the sizes the code
    card's. The serial chain's own condition is chain_load.
    """
    service_us = _mean_or_zero(samples["service"])
    commit_rounds = code.commit_rounds()
    buffer_rounds = code.buffer_rounds()
    both_buffers_round_count = 2 * buffer_rounds
    window_round_count = commit_rounds + both_buffers_round_count
    committed_round_count = commit_rounds + window_round_count
    committed_rounds_us = committed_round_count * round_period_microseconds
    both_layers_service_us = 2 * service_us
    processes = both_layers_service_us / committed_rounds_us
    return math.ceil(processes)


def _request_run_sequence(row: dict):
    """The run ordinal of the request a transfer serves, None when it has none.

    A boundary hand-off names its source request instead, and a round's
    transfer names no request at all.
    """
    relation = row["attribution"].get("relation")
    if not relation:
        return None
    request_key = relation.get("request_key")
    if request_key is None:
        return None
    return request_key["run_sequence"]


def _measurement(
    settings: machine_settings.MachineSettings,
    observation: observation_module.Observation,
    result: result_records.RunResult,
    *,
    record_options: collect.RecordOptions,
    code: ports.CodeModel,
    round_period_microseconds: float,
    point_id: str,
    seed: int,
    wall_seconds: float,
) -> ShotMeasurement:
    """Read every number of one completed shot off its records.

    It stays whole past the size prompt: each number is read once, named
    and placed, and a split would put reading and placing in two places.
    """
    owners = _scored_owners(result)
    sample_digest = _sample_digest(observation, owners)
    rounds_by_owner = _rounds_by_owner(observation, owners)
    owner_rounds = rounds_by_owner.values()
    executed_rounds = sum(owner_rounds)
    rounds_per_output = _one_length(owner_rounds)
    samples, window_tiers = collect_samples(observation, result)
    logical_failure = _logical_failure(owners)
    predictions = _predictions(result)
    throughput = _throughput_per_microsecond(observation, samples)
    referee = _referee_counts(observation)
    decoded_windows = len(samples["service"])
    commit_rounds = code.commit_rounds()
    window_period_us = commit_rounds * round_period_microseconds
    load = chain_load(samples, window_period_us)
    algorithm = active_decoder_kind(settings)
    queued = observation.queue_depth.peak
    primary_tier = settings.window_tier.value
    pools = _pool_measures(observation, primary_tier)
    strong = _strong_decodes(observation)
    tiers = _tier_records(observation)
    backlog_peak = _backlog_peak_rounds(observation)
    processes = parallel_processes_needed(
        samples, code, round_period_microseconds
    )
    totals = link_totals(result.link_traffic)
    means = _means(samples)
    maxes = _maxes(samples)
    window_statuses = _window_statuses(observation)
    unscored_reason = _unscored_reason(observation)
    is_scored = unscored_reason == ""
    provisional_windows = _provisional_no_correction_windows(observation)
    is_scored_failure = logical_failure and is_scored
    confidence = _shot_confidence(settings, observation, seed, record_options)
    return ShotMeasurement(
        point_id=point_id,
        algorithm=algorithm,
        seed=seed,
        decoded_windows=decoded_windows,
        logical_failure=is_scored_failure,
        executed_rounds=executed_rounds,
        scored_outputs=len(owners),
        rounds_per_output=rounds_per_output,
        samples=samples,
        window_tiers=window_tiers,
        means=means,
        maxes=maxes,
        load=load,
        throughput_windows_per_us=throughput.windows_per_microsecond,
        throughput_rounds_per_us=throughput.rounds_per_microsecond,
        max_queued_windows=queued,
        weak_queue_max=pools.weak_queue_max,
        strong_queue_max=pools.strong_queue_max,
        weak_busy_fraction=pools.weak_busy_fraction,
        strong_busy_fraction=pools.strong_busy_fraction,
        escalated_windows=strong.windows,
        strong_decoded_rounds=strong.rounds,
        strong_service_sum_us=strong.service_sum_microseconds,
        commit_rounds=commit_rounds,
        window_period_us=window_period_us,
        parallel_processes_needed=processes,
        weak_syndrome_weight_mean=tiers.weak_syndrome_weight_mean,
        weak_syndrome_weight_max=tiers.weak_syndrome_weight_max,
        weak_service_mean_us=tiers.weak_service_mean_us,
        strong_wait_mean_us=tiers.strong_wait_mean_us,
        strong_wait_max_us=tiers.strong_wait_max_us,
        strong_held_in_units_max=tiers.strong_held_in_units_max,
        backlog_peak_rounds=backlog_peak,
        referee_windows_checked=referee.windows_checked,
        referee_window_disagreements=referee.window_disagreements,
        link_totals=totals,
        sim_wall_seconds=wall_seconds,
        data_movement=result.data_movement,
        window_statuses=window_statuses,
        is_scored=is_scored,
        unscored_reason=unscored_reason,
        provisional_no_correction_windows=provisional_windows,
        sample_digest=sample_digest,
        confidence=confidence,
        predictions=predictions,
    )


def _shot_confidence(
    settings: machine_settings.MachineSettings,
    observation: observation_module.Observation,
    seed: int,
    record_options: collect.RecordOptions,
) -> Optional[ShotConfidence]:
    """The confidence ledger's windows, None when no signal ran."""
    ledger = observation.confidence
    if ledger is None:
        return None
    windows = ledger.windows()
    is_sampled = record_options.samples_confidence_of(seed)
    signal = settings.switching.confidence.name
    sampled_shot_count = record_options.confidence_shot_count
    return ShotConfidence(signal, windows, is_sampled, sampled_shot_count)


def _window_statuses(observation: observation_module.Observation) -> dict:
    """How many committed windows carried each status besides success.

    A window keeps the status of its final decode (windows/
    window_commits.py, commit and finish_strong), so an escalated window
    counts the strong answer's status and not the weak one it replaced.
    """
    counts = dict.fromkeys(WINDOW_STATUS_COLUMNS, 0)
    for window in observation.windows.windows.values():
        if window.decode_status is None:
            continue
        column = f"{window.decode_status}_windows"
        counts[column] += 1
    return counts


def _unscored_reason(observation: observation_module.Observation) -> str:
    """The backends' reasons for every decode committed with no correction.

    A replaced provisional decode counts too: the strong answer replaces its
    correction but not what it already fed forward (a shipped boundary is
    never revised; its crossing commit is kept), and decsim does not trace
    which a commit reached, so any unscores the shot. Distinct reasons,
    sorted, joined by ';'; empty when every commit got a correction.
    """
    reasons = set()
    for window in observation.windows.windows.values():
        reasons.add(window.no_correction_reason)
        reasons.add(window.provisional_no_correction_reason)
    reasons.discard(None)
    ordered = sorted(reasons)
    return ";".join(ordered)


def _provisional_no_correction_windows(
    observation: observation_module.Observation,
) -> int:
    """The windows whose replaced provisional decode had no correction."""
    count = 0
    for window in observation.windows.windows.values():
        if window.provisional_no_correction_reason is not None:
            count += 1
    return count


def _sample_digest(
    observation: observation_module.Observation, owners: tuple
) -> str:
    """sha256 of each scored owner's sampled detection events and truth.

    Two points that differ only in their decoder draw the same shot at the
    same seed, so pairing their shots can be checked rather than assumed.
    """
    shots_by_operation = observation.sampled_shots.shots_by_operation
    digest = hashlib.sha256()
    for owner in owners:
        shot = shots_by_operation[owner.operation_id]
        event_bytes = bytes(shot.detection_events)
        truth_bytes = bytes(owner.observable_truth)
        digest.update(event_bytes)
        digest.update(truth_bytes)
    return digest.hexdigest()


def _scored_owners(result: result_records.RunResult) -> tuple:
    """The operations whose logical output the source sampled a truth for.

    A live stream's segments and patch holders have no truth of their own:
    the owner's one circuit is scored once, as sinter scores a complete
    circuit (sinter/_decoding/_stim_then_decode_sampler.py:76). A source
    that draws no shot leaves no owner, and the shot is refused.
    """
    owners = []
    for operation_result in result.operation_results:
        if operation_result.observable_truth is not None:
            owners.append(operation_result)
    if not owners:
        raise refusal.RefusalError(
            "decsim run scores every shot against the logical "
            "observables its syndrome source sampled, and the source "
            "sampled none; name a qpu.kind that samples the circuit, such "
            "as stim_device"
        )
    return tuple(owners)


def _rounds_by_owner(
    observation: observation_module.Observation, owners: tuple
) -> dict:
    """Each scored owner's patch-rounds read out this shot.

    A round of a patch counts once, whenever it ran: a live stream's
    rounds include those it idled through while its feedback waited,
    which only the run knows, and all of them reach the owner's circuit.
    """
    rounds_by_owner = {}
    for owner in owners:
        rounds_by_owner[owner.operation_id] = set()
    for event in observation.round_events.events:
        _add_the_emitted_patch_rounds(rounds_by_owner, event)
    counts = {}
    for owner_id, patch_rounds in rounds_by_owner.items():
        counts[owner_id] = len(patch_rounds)
    return counts


def _add_the_emitted_patch_rounds(rounds_by_owner: dict, event) -> None:
    """An emitted round's patch-rounds, added to its scored owner's set."""
    patch_rounds = rounds_by_owner.get(event.operation_id)
    if event.kind != "EMITTED" or patch_rounds is None:
        return
    for patch in event.patch_ids:
        patch_rounds.add((event.round_index, patch))


def _one_length(owner_rounds) -> int:
    """The patch-rounds every scored owner ran; 0 when they ran apart."""
    lengths = set(owner_rounds)
    if len(lengths) != 1:
        return 0
    (length,) = lengths
    return length


@dataclasses.dataclass(frozen=True)
class _CommittedDecode:
    """The decode whose result the frame committed, as the points read it.

    One window can be decoded several times (forced-class solves, a strong
    re-decode, a cancelled speculative one), all under one window key, so
    the frame record's tier and run ordinal name the one that committed.
    """

    tier: window_records.DecoderTier
    compute_start_ticks: int
    done_ticks: int
    ready_ticks: Optional[int]  # it first may compute, whatever the unit did
    run_sequence: int  # the run ordinal of the request it committed
    round_count: int  # the rounds it read, its job's own count
    backend_queue_wait_ticks: int  # its waits inside a strong backend
    store_read_ticks: int  # its input's read in its tier's store


@dataclasses.dataclass(frozen=True)
class _Throughput:
    """The shot's throughput, per microsecond of its span."""

    windows_per_microsecond: float
    rounds_per_microsecond: float


@dataclasses.dataclass(frozen=True)
class _RefereeCounts:
    """The window referee's tally over the shot."""

    windows_checked: int
    window_disagreements: int


@dataclasses.dataclass(frozen=True)
class _PoolMeasures:
    """Each tier's pool load over the shot."""

    weak_queue_max: int
    strong_queue_max: int
    weak_busy_fraction: float
    strong_busy_fraction: float


@dataclasses.dataclass(frozen=True)
class _StrongDecodes:
    """What the strong tier committed over the shot."""

    windows: int
    rounds: int
    service_sum_microseconds: float


@dataclasses.dataclass(frozen=True)
class _DecodeLife:
    """One decode's ticks: taken by a unit, input in, compute, done."""

    dispatch: int
    ready: int
    compute_start: int
    end: int


@dataclasses.dataclass(frozen=True)
class _TierRecords:
    """Each tier's summary from the switching records."""

    weak_syndrome_weight_mean: Optional[float] = None
    weak_syndrome_weight_max: Optional[int] = None
    weak_service_mean_us: Optional[float] = None
    strong_wait_mean_us: Optional[float] = None
    strong_wait_max_us: Optional[float] = None
    strong_held_in_units_max: Optional[int] = None


def _logical_failure(owners: tuple) -> bool:
    """Whether the loop's observables missed the truth.

    A shot fails when any of its owners reads a wrong observable, as a
    computation fails when any of its logical qubits does; a workload
    of several patches is judged over all of them.
    """
    is_logical_failure = False
    for owner in owners:
        truth = tuple(owner.observable_truth)
        loop_prediction = tuple(owner.logical_observables)
        is_logical_failure |= loop_prediction != truth
    return is_logical_failure


def _predictions(result: result_records.RunResult) -> str:
    """Each operation's predicted observables, keyed by operation id.

    One character a bit, Stim's 01 format, null when unanswered; compact
    sorted json, as sinter writes a json cell (sinter/_data/_csv_out.py:35-37),
    since a csv reader that types numbers would read 01 as 1.
    """
    predicted = {}
    for operation_result in result.operation_results:
        key = str(operation_result.operation_id)
        observables = operation_result.logical_observables
        predicted[key] = _bit_text(observables)
    return json.dumps(predicted, sort_keys=True, separators=(",", ":"))


def _bit_text(bits: Optional[tuple]) -> Optional[str]:
    """Bits as the characters 0 and 1, or None for no bits."""
    if bits is None:
        return None
    characters = [str(int(bit)) for bit in bits]
    return "".join(characters)


def _throughput_per_microsecond(
    observation: observation_module.Observation, samples: dict
) -> _Throughput:
    """Windows and rounds over the span from first round to last commit.

    Rounds are those the QPU read out, each once, so it holds for any
    workload. Concurrent operations add up: the rate is the load the shared
    decoders serve, weighed against their service rate (Holmes 2004.04794
    section III). A shot that commits nothing measures zero.
    """
    if not observation.frame_corrections.committed:
        return _Throughput(
            windows_per_microsecond=0.0, rounds_per_microsecond=0.0
        )
    decoded_windows = len(samples["service"])
    rounds_this_shot = _emitted_round_count(observation)
    span_us = _decoded_span_microseconds(observation)
    windows_per_us = decoded_windows / span_us
    rounds_per_us = rounds_this_shot / span_us
    return _Throughput(
        windows_per_microsecond=windows_per_us,
        rounds_per_microsecond=rounds_per_us,
    )


def _emitted_round_count(observation: observation_module.Observation) -> int:
    """The rounds the QPU read out this shot, each (operation, round) once."""
    emitted = set()
    for event in observation.round_events.events:
        if event.kind != "EMITTED":
            continue
        emitted.add((event.operation_id, event.round_index))
    return len(emitted)


def _referee_counts(
    observation: observation_module.Observation,
) -> _RefereeCounts:
    """The referee's counters; zero when no referee wraps the decoder."""
    audit = observation.referee_audit
    return _RefereeCounts(
        windows_checked=audit.windows_checked,
        window_disagreements=audit.window_disagreements,
    )


def _pool_measures(
    observation: observation_module.Observation, primary_tier: str
) -> _PoolMeasures:
    """Each pool's own queue peak and busy fraction, by the tier's name.

    The plan's windows queue in the default pool, so its numbers are the
    primary tier's; a run without a pool reads zero for it.
    """
    peaks = observation.queue_depth.peak_by_pool
    utilization = observation.decoder_utilization.result()
    busy = utilization["per_pool_busy_fraction"]
    default_queue_max = peaks.get(decode_queue.DEFAULT_POOL, 0)
    default_busy_fraction = busy.get(decode_queue.DEFAULT_POOL, 0.0)
    if primary_tier == window_records.DecoderTier.STRONG.value:
        return _PoolMeasures(
            weak_queue_max=0,
            strong_queue_max=default_queue_max,
            weak_busy_fraction=0.0,
            strong_busy_fraction=default_busy_fraction,
        )
    weak_queue_max = default_queue_max
    strong_queue_max = peaks.get(decode_queue.STRONG_POOL, 0)
    weak_busy_fraction = default_busy_fraction
    strong_busy_fraction = busy.get(decode_queue.STRONG_POOL, 0.0)
    return _PoolMeasures(
        weak_queue_max=weak_queue_max,
        strong_queue_max=strong_queue_max,
        weak_busy_fraction=weak_busy_fraction,
        strong_busy_fraction=strong_busy_fraction,
    )


def _strong_decodes(
    observation: observation_module.Observation,
) -> _StrongDecodes:
    """The windows the strong tier committed, their rounds, their service.

    The report forms Toshio's Theorem 1 bound (2510.25222 eq. (6)) per point
    from these; the summed service is taken in ticks, so it is exact.
    """
    stages = observation.stages
    windows = 0
    rounds = 0
    service_ticks = 0
    for frame_record in observation.frame_corrections.committed:
        tier = window_records.DecoderTier(frame_record.tier)
        if tier is not window_records.DecoderTier.STRONG:
            continue
        decode = _committed_decode(stages, frame_record)
        windows += 1
        rounds += decode.round_count
        service_ticks += decode.done_ticks - decode.compute_start_ticks
    service_sum_microseconds = config_module.ticks_to_microseconds(
        service_ticks
    )
    return _StrongDecodes(windows, rounds, service_sum_microseconds)


def _tier_records(
    observation: observation_module.Observation,
) -> _TierRecords:
    """Each tier's inputs, computes and waits off the switching records.

    A request key's run ordinal names the stages it served, so a decode's
    compute is its own stages' span. A strong decode's wait is queue_wait
    plus compute_wait (gem5's fuBusy), not the input hop between them, which
    a free unit pays too.
    """
    records = observation.decode_records
    if records is None:
        return _TierRecords()
    lives = _decode_lives(observation.stages)
    weights = _weak_syndrome_weights(records.requests)
    weak_computes = _weak_compute_microseconds(records.requests, lives)
    strong_waits = _strong_wait_microseconds(records.requests, lives)
    held_peak = _strong_held_in_units_peak(records.requests, lives)
    weight_mean = _mean_or_zero(weights)
    weight_max = max(weights, default=0)
    compute_mean = _mean_or_zero(weak_computes)
    wait_mean = _mean_or_zero(strong_waits)
    wait_max = _max_or_zero(strong_waits)
    return _TierRecords(
        weak_syndrome_weight_mean=weight_mean,
        weak_syndrome_weight_max=weight_max,
        weak_service_mean_us=compute_mean,
        strong_wait_mean_us=wait_mean,
        strong_wait_max_us=wait_max,
        strong_held_in_units_max=held_peak,
    )


def _decode_lives(stages) -> dict:
    """Each decode's dispatch, ready, compute start and end, by ordinal."""
    records_by_run_sequence = collections.defaultdict(list)
    for record in stages.records:
        for run_sequence in record.run_sequences:
            records_by_run_sequence[run_sequence].append(record)
    lives = {}
    for run_sequence, records in records_by_run_sequence.items():
        lives[run_sequence] = _decode_life(records)
    return lives


def _decode_life(records: list) -> _DecodeLife:
    """One decode's ticks off its own stage records.

    A decoder that records no dispatch or ready tick leaves them at the
    compute start, so none of its time counts as an input hop.
    """
    starts = [record.start_ticks for record in records]
    ends = [record.end_ticks for record in records]
    compute_start = min(starts)
    dispatch = _dispatch_ticks(records, compute_start)
    ready = _ready_ticks(records)
    if ready is None:
        ready = dispatch
    return _DecodeLife(dispatch, ready, compute_start, max(ends))


def _weak_syndrome_weights(requests: list) -> list:
    """The set bits of every weak request's landed input that has bits."""
    weights = []
    for request in requests:
        is_weak = request.request_key.tier is window_records.DecoderTier.WEAK
        if is_weak and request.syndrome_weight is not None:
            weights.append(request.syndrome_weight)
    return weights


def _weak_compute_microseconds(requests: list, lives: dict) -> list:
    """Each weak decode's compute, first stage start to last stage end."""
    durations = []
    for request in _requests_of_tier(requests, lives, "weak"):
        life = lives[request.request_key.run_sequence]
        duration = _span_microseconds(life.end, life.compute_start)
        durations.append(duration)
    return durations


def _strong_wait_microseconds(requests: list, lives: dict) -> list:
    """Each strong decode's queue wait plus its compute wait."""
    waits = []
    for request in _requests_of_tier(requests, lives, "strong"):
        life = lives[request.request_key.run_sequence]
        queue_wait = life.dispatch - request.ready_ticks
        compute_wait = life.compute_start - life.ready
        wait_ticks = queue_wait + compute_wait
        wait = config_module.ticks_to_microseconds(wait_ticks)
        waits.append(wait)
    return waits


def _strong_held_in_units_peak(requests: list, lives: dict) -> int:
    """The most strong decodes held in unit memory at once, on compute.

    Held from the tick it may compute to its compute start, gem5's fuBusy
    wait (src/cpu/o3/inst_queue.cc:1009-1014). A merged batch counts once,
    and a depth counts only when time passes at it (observe/queue_depth.py).
    """
    change_by_tick = collections.defaultdict(int)
    held_decodes = set()
    for request in _requests_of_tier(requests, lives, "strong"):
        run_sequence = request.request_key.run_sequence
        if run_sequence in held_decodes:
            continue
        held_decodes.add(run_sequence)
        life = lives[run_sequence]
        change_by_tick[life.ready] += 1
        change_by_tick[life.compute_start] -= 1
    depth = 0
    peak = 0
    for tick in sorted(change_by_tick):
        depth += change_by_tick[tick]
        peak = max(peak, depth)
    return peak


def _requests_of_tier(requests: list, lives: dict, tier: str) -> list:
    """The tier's requests whose decode ran at least one stage."""
    chosen = []
    for request in requests:
        key = request.request_key
        if key.tier.value == tier and key.run_sequence in lives:
            chosen.append(request)
    return chosen


def _backlog_peak_rounds(
    observation: observation_module.Observation,
) -> Optional[int]:
    """The most undecoded rounds held at once, when the sampler ran."""
    backlog = observation.decode_backlog
    if backlog is None:
        return None
    return backlog.peak


def _window_order(window_item: tuple) -> bytes:
    """A (window key, window) pair's place: the key's canonical bytes.

    An operation id may be an int or a str, which Python does not order
    against each other (records/identity.py).
    """
    window_key, _window = window_item
    return identity_records.stable_identity_bytes(window_key)


def _span_microseconds(end_ticks: int, start_ticks: int) -> float:
    span_ticks = end_ticks - start_ticks
    return config_module.ticks_to_microseconds(span_ticks)


def _last_emitted_round(qpu_send: dict, operation_id) -> int:
    last_round = 0
    for sent_operation_id, round_index in qpu_send:
        if sent_operation_id == operation_id:
            last_round = max(last_round, round_index)
    return last_round


def _hop_start_ticks(row: dict) -> int:
    """The tick a transfer was asked for, which is where its hop starts.

    The request is the delivery less total_delay_ticks. The setup is the
    hop's own cost, as gem5 adds a DMA's fixed delay to the completion
    (src/dev/dma_device.cc:116-118).
    """
    return row["delivery_ticks"] - row["total_delay_ticks"]


def _committed_decode(stages, frame_record) -> _CommittedDecode:
    """The committing decode's tier and the ticks of its own life.

    Its stage records are those whose request ordinals hold the frame
    record's, so a merged batch answers for every window it served and a
    window decoded more than once never mixes two decodes.
    """
    tier = window_records.DecoderTier(frame_record.tier)
    operation_id, window_id = frame_record.window_key
    records = _committing_stage_records(
        stages, operation_id, window_id, frame_record.run_sequence
    )
    assert records, (
        f"window ({operation_id}, {window_id}) committed the result of "
        f"request {frame_record.run_sequence}, which recorded no stage"
    )
    starts = []
    ends = []
    for record in records:
        starts.append(record.start_ticks)
        ends.append(record.end_ticks)
    first = min(starts)
    last = max(ends)
    ready = _ready_ticks(records)
    run_sequence = frame_record.run_sequence
    rounds = _rounds_read(records)
    waited = _backend_queue_wait_ticks(records)
    read = _store_read_ticks(records)
    return _CommittedDecode(
        tier, first, last, ready, run_sequence, rounds, waited, read
    )


def _backend_queue_wait_ticks(records: list) -> int:
    """The decode's waits inside a strong backend, whole at its last stage."""
    waits = []
    for record in records:
        waits.append(record.backend_queue_wait_ticks)
    return max(waits, default=0)


def _store_read_ticks(records: list) -> int:
    """The decode's input read in its store, carried on every stage."""
    reads = []
    for record in records:
        reads.append(record.store_read_ticks)
    return max(reads, default=0)


def _rounds_read(records: list) -> int:
    """The rounds this decode read, as every one of its stages carries."""
    counts = []
    for record in records:
        counts.append(record.round_count)
    return max(counts, default=0)


def _dispatch_ticks(records: list, fallback: int) -> int:
    """The tick a unit took this decode, off its own stage records.

    Every stage of one decode carries it; a decoder that records none (a
    backend firing its own stages without it) leaves the points to the
    fallback rather than to a missing tick.
    """
    ticks = []
    for record in records:
        if record.dispatch_ticks is not None:
            ticks.append(record.dispatch_ticks)
    if not ticks:
        return fallback
    return min(ticks)


def _ready_ticks(records: list):
    """The tick this decode first may compute, off its own stage records.

    The dependency it waited for was met then, and what it waited for
    after it is the unit's compute. A decoder that records none (a
    backend firing its own stages without it) leaves it unknown and the
    points read the whole park as the dependency wait.
    """
    ticks = []
    for record in records:
        if record.ready_ticks is not None:
            ticks.append(record.ready_ticks)
    if not ticks:
        return None
    return min(ticks)


def _first_dispatch_ticks(stages, window, frame_record) -> int:
    """The tick a unit took the first decode of this window.

    The ready-queue wait ends there and an escalated window's weak attempt
    starts there (Toshio et al. 2510.25222 Sec. III A steps 2 to 4). A
    cancelled decode is not on the window's path, so it is left out.
    """
    operation_id, window_id = frame_record.window_key
    records = stages.records_for(operation_id, window_id)
    cancelled = _cancelled_run_sequences(records)
    ticks = []
    for record in records:
        if cancelled.intersection(record.run_sequences):
            continue
        if record.dispatch_ticks is None:
            continue
        ticks.append(record.dispatch_ticks)
    if not ticks:
        return window.t_dispatch
    return min(ticks)


def _cancelled_run_sequences(records) -> set:
    """The run ordinals of the decodes a cancel closed."""
    ordinals = set()
    for record in records:
        if not record.cancelled:
            continue
        ordinals.update(record.run_sequences)
    return ordinals


def _committing_stage_records(
    stages, operation_id, window_id, run_sequence: int
) -> list:
    """One window's stage records that belong to one request ordinal."""
    records = []
    for record in stages.records_for(operation_id, window_id):
        if run_sequence in record.run_sequences:
            records.append(record)
    return records


def _confidence_ticks(window, decode: _CommittedDecode) -> int:
    """The confidence step: the committing decode's end to the verdict.

    The signal's computation is on the window's path, since the weak
    decoder "simultaneously generates a soft output g for each decoding
    window" (Toshio et al. 2510.25222 lines 657-663): the walk under
    cluster_gap, the other forced-class solve under complementary_gap. An
    escalated window has none: weak_attempt already runs to its verdict.
    """
    if window.t_done <= decode.done_ticks:
        return 0
    return window.t_done - decode.done_ticks


def _startable_ticks(decode: _CommittedDecode, input_landed: int) -> int:
    """Where the dependency wait ends and the wait for the compute begins.

    The decode's own stamp, held inside the park the two points divide:
    never before its input was readable in the unit's memory, never
    after its compute began. A decode that recorded no such tick reads
    the whole park as its dependency wait.
    """
    ready = decode.ready_ticks
    if ready is None:
        return decode.compute_start_ticks
    ready = max(ready, input_landed)
    return min(ready, decode.compute_start_ticks)


def _attempt_end_ticks(window, decode: _CommittedDecode, first_dispatch) -> int:
    """Where the window's weak attempt ended and the committing decode began.

    An escalated window's strong decode starts from the verdict (Toshio et
    al. 2510.25222 Sec. III A). A committing decode already computing then
    (a kept weak result, a speculative strong decode) cost the window no
    attempt, and both start at the dispatch.
    """
    if window.t_done >= decode.compute_start_ticks:
        return first_dispatch
    return window.t_done


def _stage_microseconds(stages, frame_record) -> dict:
    """The committing decode's stages, in microseconds, by stage name.

    Every decode of a window records under its key, so the stages are the
    committing request's, as for every other point.
    """
    operation_id, window_id = frame_record.window_key
    records = _committing_stage_records(
        stages, operation_id, window_id, frame_record.run_sequence
    )
    stage_us = {}
    for record in records:
        stage_us[record.stage] = _span_microseconds(
            record.end_ticks, record.start_ticks
        )
    return stage_us


def _decoded_span_microseconds(
    observation: observation_module.Observation,
) -> float:
    """First round into any window to the last frame commit, in us."""
    first_round_ticks = []
    for window in observation.windows.windows.values():
        if window.t_first_round is not None:
            first_round_ticks.append(window.t_first_round)
    first_round_tick = min(first_round_ticks)
    commit_ticks = []
    for record in observation.frame_corrections.committed:
        commit_ticks.append(record.committed_ticks)
    last_commit_tick = max(commit_ticks)
    return _span_microseconds(last_commit_tick, first_round_tick)


def _mean_or_zero(values: list) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def _max_or_zero(values: list) -> float:
    if not values:
        return 0.0
    return max(values)


def _means(samples: dict) -> dict:
    means = {}
    for point, values in samples.items():
        means[point] = _mean_or_zero(values)
    return means


def _maxes(samples: dict) -> dict:
    maxes = {}
    for point, values in samples.items():
        maxes[point] = _max_or_zero(values)
    return maxes


def _write_trace(shot, run_dir, label: str, only_traced_shot: bool) -> None:
    """The shot's Chrome trace, when the section asked and named it.

    Each file carries the shot's label, so no shot overwrites another's;
    only trace_shots are written.
    """
    observation = shot.task.settings.observation
    if not observation.writes_trace:
        return
    if shot.seed not in observation.trace_shots:
        return
    writer = shot.machine.observation.trace_writer
    path = run_folder.trace_path_of(observation, run_dir, label)
    if observation.trace_path is not None and not only_traced_shot:
        path = trace_path_for_shot(path, label)
    writer.write(str(path))
