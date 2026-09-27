"""One collected shot -> one shot's numbers. Nothing is built here.

A shot is one circuit through the whole reaction path. `measure_shot`
reads the run's listeners (`machine.observation`) and its RunResult, and
no component: the window ledger, the stage ledger, the frame's
corrections, the queue depth log, the referee's audit, the sampled shot
and the link traffic of the result. The input and output transfers are
named by role, not by wire, and which wire carries each role follows
from the escalation row's declared primary_tier (ports.py
EscalationPolicy), so a row written outside decsim is measured like any
other.
"""

import collections
import dataclasses
import math
import pathlib
import statistics
from typing import Optional

import numpy
import pymatching
import stim

import decsim.build.escalation as escalation_build
import decsim.collect as collect
import decsim.config as config_module
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoder_output as decoder_output
import decsim.experiments.refusal as refusal
import decsim.observe.observation as observation_module
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
    # the dependency wait around the committing decode's own hop: from
    # where its path started, the dispatch or the verdict that escalated
    # the window, to the first tick it may compute, less the input hop
    # itself. It is the wait for the escalated rounds to land in the
    # strong store before the input hop, the predecessor's boundary and
    # the escalation's selection after it, and zero when nothing was
    # owed
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
    "release",  # the algorithm's end -> the correction written out
    # the compute start -> the decode's end: every stage the unit ran,
    # and nothing the decode waited for
    "service",
    # the committing decode's end -> the verdict on the window's answer:
    # the confidence signal's own computation, which is the walk under
    # cluster_gap and the sibling forced solve's remaining time under
    # complementary_gap. Zero for an escalated window, whose committing
    # decode is the strong one and whose weak_attempt already runs to
    # the verdict
    "confidence",
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
    # Totals. The buffer0 pair starts the clock at the weak syndrome buffer
    # publication; the qpu pair starts it when the round leaves the QPU (the QC
    # send), so it includes QC, controller processing, packing and CWB.
    "buffer0_ready_to_frame",  # window complete in the weak buffer -> frame
    "buffer0_first_round_to_frame",  # first round in the weak buffer -> frame
    "qpu_last_round_to_frame",  # last required round off QPU -> frame
    "qpu_first_round_to_frame",  # first required round off QPU -> frame
)

# One count column per status a committed window's decode may carry
# besides success (decoders/decoder.py BackendDecodeStatus), so a status
# the enum gains is counted with no change here; sinter keeps its
# custom counts the same way, one named counter per kind
# (sinter/_data/_task_stats.py:71)
WINDOW_STATUS_COLUMNS = tuple(
    f"{status.value}_windows"
    for status in decoder_module.BackendDecodeStatus
    if status is not decoder_module.BackendDecodeStatus.SUCCEEDED
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
class ShotMeasurement:
    """One shot's numbers; the field names are the csv columns."""

    # the point's strong id, sinter's strong_id column
    # (sinter/_data/_csv_out.py:69-77); the report adds its swept values
    point_id: str
    algorithm: object  # the active unit's card: a name or a latency in us
    seed: int
    windows: int
    logical_failure: bool
    samples: dict  # point -> us list, one per window (per round for cwb)
    means: dict  # point -> mean us over this shot's windows
    maxes: dict  # point -> max us
    load: float  # service per window / window inter-arrival
    direct_failure: bool  # whole-circuit PyMatching on the same events failed
    direct_mismatch: bool  # loop prediction differs from direct PyMatching
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
    # and their mean service, and r_com, the rounds a window commits: the
    # report forms Toshio's Theorem 1 bound on that service per sweep
    # point from the first and the last (2510.25222 eq. (6))
    escalated_windows: int
    strong_decoded_rounds: int
    strong_service_mean_us: float
    commit_rounds: int
    # tau_gen r_com: a window's inter-arrival, commit rounds times the
    # round period, which the chain's load divides by
    window_period_us: float
    # Skoric's least count of parallel decoding processes, ceil(2 tau_W
    # / ((n_com + n_W) tau_rd)) (2209.08552 lines 429-438)
    parallel_processes_needed: int
    # what tells harder windows from an overloaded strong side (the
    # paired run of docs/how-to/compare_two_runs.md): the set bits of each
    # weak decode's input, its detection events when they are formed
    # ahead of the decoder; each weak decode's compute, its stages' span;
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
    tesseract_windows_checked: int  # referee re-decodes (0 = referee off)
    # referee reached a different owned observable contribution
    tesseract_window_disagreements: int
    # path -> the run's own ledger counters plus rounds/windows context,
    # for links.csv; the totals come straight off the ledger's counters
    link_totals: dict
    sim_wall_seconds: float
    # the shot's copies, references and moves as the RunResult carries
    # them, None when observation.data_movement is off, so a run that
    # counted none writes no data-movement row rather than a row of zeros
    data_movement: Optional[dict]
    # where this shot's Chrome trace was written, None when the shot was
    # not one of observation.trace_shots; the residence table reads it
    trace_path: Optional[str]
    # the burst detector's first flagged round at or after the burst's
    # onset, from round 1 on a shot with no burst, 0 when it flagged none;
    # and whether that flag came within burst_detector.catch_deadline_rounds
    # of the onset. None, and no column, without a detector; the second
    # is None on a shot with no burst
    burst_first_flag_round: Optional[int]
    burst_caught_in_time: Optional[bool]
    # WINDOW_STATUS_COLUMNS -> the committed windows whose decode carried
    # that status; shots.csv holds one column per entry
    window_statuses: dict


def measure_shot(
    shot: collect.Shot, run_dir=None, *, only_traced_shot: bool = False
) -> ShotMeasurement:
    """Read one collected shot's numbers off its machine and result.

    run_dir receives the log file when log: file|both is on and the
    Chrome trace when trace names one and the shot is in trace_shots;
    None writes nothing beyond the returned measurement.
    only_traced_shot says the run traces this shot alone, so a trace
    path the yaml names is written as it stands.
    """
    settings = shot.task.settings
    point_id = shot.task.strong_id()
    label = shot_label(point_id, shot.seed)
    observation = shot.machine.observation
    if run_dir is not None and settings.observation.writes_log:
        _write_log(observation, run_dir, label)
    trace_path = None
    if run_dir is not None:
        trace_path = _write_trace(shot, run_dir, label, only_traced_shot)
    return _measurement(
        settings,
        observation,
        shot.result,
        point_id=point_id,
        seed=shot.seed,
        wall_seconds=shot.wall_seconds,
        trace_path=trace_path,
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

    A hop's transfers for one window can overlap, as the escalation's
    selection and its rounds do, so the span is what the window waited
    on the hop and a sum would count the overlap twice. A window is
    named by its operation and its index, never by its index alone: the
    window record's own key is (operation_id, window_index)
    (records/windows.py:79-82 and 144-146), so a workload of several
    streams holds one window 3 per stream and a key without the
    operation would sum every stream's window 3 into one entry. gem5
    names a per-instruction fact the same way: the reorder buffer finds
    an instruction by its thread and its sequence number, findInst(
    ThreadID tid, InstSeqNum squash_inst) walking instList[tid]
    (gem5 src/cpu/o3/rob.hh:131-134 and rob.cc:515-523), and a retired
    instruction's counters land in that thread's own bucket,
    commitStats[tid] and thread[tid]->threadStats (gem5
    src/cpu/o3/cpu.cc:1156-1174).
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

    Every transfer that serves a decoder request names it
    (records/transfers.py, RequestTransferRelation), so a window decoded
    more than once has one entry per decode and a point reads the hop of
    the decode it describes rather than the window's last. The delay is
    summed and the landing is the latest delivery, because a decode
    starts once every transfer of its input has landed.
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
    """The tick each round left the QPU (its earliest QC request), by round."""
    send = {}
    for row in transfers:
        if row["path"] != "qpu_to_controller":
            continue
        # a readout carries one round of one operation
        (rounds,) = row["attribution"]["rounds_by_operation"]
        round_index = rounds["round_lo"]
        earlier = send.get(round_index)
        start = _hop_start_ticks(row)
        if earlier is None or start < earlier:
            send[round_index] = start
    return send


def window_points_us(
    window,
    frame_record,
    decode,
    first_dispatch: int,
    stage_us: dict,
    link_delay: dict,
    input_hop: dict,
    qpu_send: dict,
    input_path: str,
    output_path: str,
) -> dict:
    """The per-window latency points, in us, for one decoded window.

    Every tick here belongs to one decode. The window waits in the ready
    queue until a unit takes its first decode; a window the strong tier
    recovered then spends its weak attempt, which ends at the verdict
    (Toshio et al. 2510.25222 Sec. III A steps 2 to 4, lines 602-617);
    and the input hop, the park, the compute and the way home are the
    committing decode's own, named by the frame record's tier and run
    ordinal. Mixing them is what a window decoded more than once
    punishes: its two forced-class solves (decision D2) are two decodes
    with two dispatches, and the window record keeps only the last.

    The decode's own time and the time it waited are two points, not
    one: a window's service is what its compute took on the unit, and
    the park before that compute is two points by cause. dep_block is
    the dependency wait: from the verdict to the input hop's request, which
    is a strong decode waiting for its escalated rounds to land in the
    strong store, and from the input landing in the unit's memory to
    the first tick the decode may start, which is where the
    predecessor's boundary arrived and is the landing itself when
    nothing was owed. compute_wait is the rest, from that tick to the
    compute starting, which is the unit's own compute busy with another
    decode. Skoric et al. 2209.08552 keep decode and wait apart, tau_W
    the window's decoding time against which the backlog condition is
    read (2209.08552.txt lines 429-435) and tau_0 the time to send a
    window to a worker and start it (lines 1004-1008); gem5 keeps the
    two waits apart, an instruction reaching the ready list only when
    its operands are there (inst_queue.cc addIfReady 1536-1562,
    wakeDependents 1074) and a ready instruction that finds no
    functional unit counted on its own (NoFreeFU and statFuBusy,
    inst_queue.cc:1009-1014, the stats at 306-316); and Ciw's per
    customer record keeps the whole pre-service wait as named parts
    rather than one number (Ciw ciw/data_record.py lines 3-21).
    """
    operation_id, window_id = window.key
    last_emitted_round = max(qpu_send)
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
    confidence_ticks = _confidence_ticks(window, decode)
    last_required_send = qpu_send[last_required_round]
    first_required_send = qpu_send[window.start_round]
    return {
        "buffer_fill": _span_microseconds(
            window.t_data_complete, window.t_first_round
        ),
        "dep_block": rounds_wait + park,
        "compute_wait": _span_microseconds(
            decode.compute_start_ticks, startable
        ),
        "queue_wait": _span_microseconds(first_dispatch, window.t_queued),
        "input_link_per_window": config_module.ticks_to_microseconds(
            input_ticks
        ),
        "fetch": stage_us["fetch"],
        "algorithm": stage_us["algorithm"],
        "release": stage_us["release"],
        "service": _span_microseconds(
            decode.done_ticks, decode.compute_start_ticks
        ),
        "confidence": config_module.ticks_to_microseconds(confidence_ticks),
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
    """The frame's committed corrections, by the window each one wrote.

    A window is named by its operation and its index, never by its index
    alone: the window record's own key is (operation_id, window_index)
    (records/windows.py:79-82 and 144-146), and a frame record carries
    that whole key. Keyed by the index alone, a workload of several
    streams would keep one stream's correction per index and drop the
    rest, and the windows that lost theirs would be measured against
    another stream's decode. gem5 asks the same way: an instruction in
    the reorder buffer is found by its thread and its sequence number,
    findInst(ThreadID tid, InstSeqNum squash_inst) (gem5
    src/cpu/o3/rob.hh:131-134).
    """
    records = {}
    for record in observation.frame_corrections.committed:
        records[record.window_key] = record
    return records


def collect_samples(
    observation: observation_module.Observation,
    result: result_records.RunResult,
) -> dict:
    """Every point's microsecond samples over the shot's decoded windows.

    Every point of a window describes the decode whose result the frame
    committed: the two links it rode, in from its tier's store and home
    to the frame, its own stages, and the wait in front of it. The
    frame's record names that decode, by the tier it ran on and by the
    run ordinal of its request. The function stays whole past the size
    prompt: it is one walk over the windows in key order, each window's
    points appended to the same lists, read top to bottom.
    """
    transfers = result.link_traffic["transfers"]
    link_delay = link_delay_by_window(transfers)
    input_hop = input_hop_by_request(transfers)
    qpu_send = qpu_send_ticks(transfers)
    frame_by_window = frame_records_by_window(observation)
    samples = {}
    for point in POINTS:
        samples[point] = []
    samples["cwb_per_round"] = controller_to_weak_buffer_delays_us(transfers)
    round_events = observation.round_events
    waits = round_stall_ticks(round_events.events)
    weak_keys = weak_store_round_keys(round_events.events)
    strong_keys = strong_store_round_keys(round_events.stored_rounds)
    samples["cwb_stall_per_round"] = round_stall_delays_us(weak_keys, waits)
    samples["csb_stall_per_round"] = round_stall_delays_us(strong_keys, waits)
    window_items = observation.windows.windows.items()
    all_windows = sorted(window_items)
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
    return samples


def direct_prediction(
    observation: observation_module.Observation, operation_id
) -> tuple:
    """Whole-circuit PyMatching on the detection events the device sampled.

    The reference the loop must agree with. The shot is the one the
    source drew for this operation, heard at sampling time. A source that
    draws no shot (qpu.kind timing_only and syndrome_bits) leaves the loop
    nothing to be judged against, so the shot is refused rather than
    counted as right or wrong.
    """
    shots_by_operation = observation.sampled_shots.shots_by_operation
    if operation_id not in shots_by_operation:
        raise refusal.RefusalError(
            "decsim collect judges every shot against the logical "
            "observables its syndrome source sampled, and the source "
            f"sampled none for operation {operation_id}; name a qpu.kind "
            "that samples the circuit, such as stim_device"
        )
    shot = shots_by_operation[operation_id]
    circuit: stim.Circuit = shot.circuit
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(
        detector_error_model
    )
    events = numpy.asarray(shot.detection_events, dtype=bool)
    predicted = matching.decode(events)
    bits = []
    for bit in predicted:
        bits.append(int(bit))
    return tuple(bits)


def chain_load(samples: dict, window_period_us: float) -> float:
    """rho: the serial chain's service per window over the window period.

    Service is the unit's occupancy per window plus the DD boundary
    handoff that serialises the chain; the window inter-arrival time is
    commit rounds times the round period. Above 1 the chain cannot keep
    up, which is Skoric et al. 2209.08552's backlog condition read as a
    ratio (2209.08552.txt lines 429-435).

    The occupancy is the committing decode's compute and the confidence
    step after it, because that step is the same unit's time: the walk
    is charged on the weak unit that produced the evidence (decision
    D8), and under complementary_gap the window's second forced-class
    solve is its own job's service on that unit (decision D2). A window
    whose signal costs nothing adds nothing, so a run without a gap
    reads as it always did.
    """
    service_us = _mean_or_zero(samples["service"])
    confidence_us = _mean_or_zero(samples["confidence"])
    handoff_us = _mean_or_zero(samples["dd_per_window"])
    chain_us = service_us + confidence_us + handoff_us
    return chain_us / window_period_us


def commit_round_count(
    settings: machine_settings.MachineSettings, distance: int
) -> int:
    """r_com: the rounds a window commits, the code distance when null."""
    commit_rounds = settings.windows.commit_rounds
    if commit_rounds is None:
        return distance
    return commit_rounds


def active_decoder_kind(settings: machine_settings.MachineSettings):
    """The kind of the tier that decodes the plan's windows."""
    tier = escalation_build.primary_tier(settings.escalation)
    tier_settings = settings.decoder_settings_for(tier)
    return tier_settings.kind


def trace_path_for_shot(path: str, label: str) -> str:
    """The path a swept shot writes to: the label joins the yaml's path.

    A sweep traces the shots trace_shots names at every point, so each
    file takes the shot's label, its point id and seed, before its
    suffixes: run.trace.json becomes run_<label>.trace.json, and
    run.trace.json.gz becomes run_<label>.trace.json.gz. Two points
    that differ only in a setting no file name spells, a basis or a
    window size, then never write one file (gem5's multisim names each
    simulation's output folder by its id,
    src/python/gem5/utils/multisim/multisim.py).
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


def _measurement(
    settings: machine_settings.MachineSettings,
    observation: observation_module.Observation,
    result: result_records.RunResult,
    *,
    point_id: str,
    seed: int,
    wall_seconds: float,
    trace_path: Optional[str],
) -> ShotMeasurement:
    """Read every number of one completed shot off its records.

    It stays whole past the size prompt: each number is read once, named,
    and placed in the record, top to bottom, and a split would put the
    reading and the placing of one number in two places.
    """
    samples = collect_samples(observation, result)
    verdicts = _logical_verdicts(observation, result)
    throughput = _throughput_per_microsecond(observation, samples)
    referee = _referee_counts(observation)
    decoded_windows = len(samples["service"])
    distance = settings.qpu.distance
    round_period_microseconds = settings.qpu.round_period_microseconds
    commit_rounds = commit_round_count(settings, distance)
    window_period_us = commit_rounds * round_period_microseconds
    load = chain_load(samples, window_period_us)
    algorithm = active_decoder_kind(settings)
    queued = observation.queue_depth.peak
    primary_tier = escalation_build.primary_tier(settings.escalation)
    pools = _pool_measures(observation, primary_tier)
    strong = _strong_decodes(observation)
    tiers = _tier_records(observation)
    backlog_peak = _backlog_peak_rounds(observation)
    processes = parallel_processes_needed(
        samples, settings, distance, round_period_microseconds
    )
    totals = link_totals(result.link_traffic)
    means = _means(samples)
    maxes = _maxes(samples)
    first_flag_round, is_caught = _burst_catch(settings, observation)
    window_statuses = _window_statuses(observation)
    return ShotMeasurement(
        point_id=point_id,
        algorithm=algorithm,
        seed=seed,
        windows=decoded_windows,
        logical_failure=verdicts.logical_failure,
        samples=samples,
        means=means,
        maxes=maxes,
        load=load,
        direct_failure=verdicts.direct_failure,
        direct_mismatch=verdicts.direct_mismatch,
        throughput_windows_per_us=throughput.windows_per_microsecond,
        throughput_rounds_per_us=throughput.rounds_per_microsecond,
        max_queued_windows=queued,
        weak_queue_max=pools.weak_queue_max,
        strong_queue_max=pools.strong_queue_max,
        weak_busy_fraction=pools.weak_busy_fraction,
        strong_busy_fraction=pools.strong_busy_fraction,
        escalated_windows=strong.windows,
        strong_decoded_rounds=strong.rounds,
        strong_service_mean_us=strong.service_mean_us,
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
        tesseract_windows_checked=referee.windows_checked,
        tesseract_window_disagreements=referee.window_disagreements,
        link_totals=totals,
        sim_wall_seconds=wall_seconds,
        data_movement=result.data_movement,
        trace_path=trace_path,
        burst_first_flag_round=first_flag_round,
        burst_caught_in_time=is_caught,
        window_statuses=window_statuses,
    )


def _burst_catch(
    settings: machine_settings.MachineSettings,
    observation: observation_module.Observation,
) -> tuple:
    """The first flag at or after the burst's onset, and whether in time.

    A detector's delay is its first alarm at or after the onset less the
    onset, and it catches the burst within k rounds when that delay is
    at most k, as detection delay is scored for change-point detectors
    (Xie et al. 2104.04186 lines 161-171). A shot with no burst counts
    from round 1, so any flag on it is a false alarm. (None, None)
    without a detector.
    """
    flags = observation.burst_flags
    if flags is None:
        return None, None
    onset_round = _burst_onset_round(settings.qpu.row_settings)
    if onset_round is None:
        first_flag_round = flags.first_flag_from(1)
        return first_flag_round, None
    first_flag_round = flags.first_flag_from(onset_round)
    delay = first_flag_round - onset_round
    deadline = settings.burst_detector.catch_deadline_rounds
    is_caught = first_flag_round > 0 and delay <= deadline
    return first_flag_round, is_caught


def _window_statuses(observation: observation_module.Observation) -> dict:
    """How many committed windows carried each status besides success.

    A window keeps the status of the decode it committed last
    (windows/window_commits.py commit), so a provisional weak answer
    the strong tier replaced is counted as the strong one.
    """
    counts = dict.fromkeys(WINDOW_STATUS_COLUMNS, 0)
    for window in observation.windows.windows.values():
        if window.decode_status is None:
            continue
        column = f"{window.decode_status}_windows"
        counts[column] += 1
    return counts


def _burst_onset_round(qpu_row_settings) -> Optional[int]:
    """The burst's first round; None when the shot draws no burst.

    Read by the keys' names rather than the row's class, so any qpu row
    whose settings carry a burst probability and onset, as burst_stim's
    do, is measured alike; a probability of 0 is no burst.
    """
    probability = getattr(qpu_row_settings, "burst_error_probability", 0)
    if probability == 0:
        return None
    return qpu_row_settings.burst_onset_round


@dataclasses.dataclass(frozen=True)
class _CommittedDecode:
    """The decode whose result the frame committed, as the points read it.

    One window can be decoded several times: the two forced-class solves
    of a complementary gap (decision D2), a strong re-decode after an
    escalation (Toshio et al. 2510.25222 Sec. III A), and under
    run_both_at_once a sibling that is cancelled. Every stage record of
    all of them carries the same window key, so the decode that
    committed is named by the frame's own record: the tier it ran on and
    the run ordinal of its request.
    """

    tier: window_records.DecoderTier
    compute_start_ticks: int
    done_ticks: int
    ready_ticks: Optional[int]  # it first may compute, whatever the unit did
    run_sequence: int  # the run ordinal of the request it committed
    round_count: int  # the rounds it read, its job's own count


@dataclasses.dataclass(frozen=True)
class _LogicalVerdicts:
    """Whether the loop, and whole-circuit PyMatching, reached the truth."""

    logical_failure: bool
    direct_failure: bool
    direct_mismatch: bool


@dataclasses.dataclass(frozen=True)
class _Throughput:
    """Decoded windows and QEC rounds per microsecond of the shot's span."""

    windows_per_microsecond: float
    rounds_per_microsecond: float


@dataclasses.dataclass(frozen=True)
class _RefereeCounts:
    """What the window referee re-decoded and where it disagreed."""

    windows_checked: int
    window_disagreements: int


@dataclasses.dataclass(frozen=True)
class _PoolMeasures:
    """Each tier's deepest ready queue and busy fraction over the shot."""

    weak_queue_max: int
    strong_queue_max: int
    weak_busy_fraction: float
    strong_busy_fraction: float


@dataclasses.dataclass(frozen=True)
class _StrongDecodes:
    """What the strong tier committed over the shot."""

    windows: int
    rounds: int
    service_mean_us: float


@dataclasses.dataclass(frozen=True)
class _DecodeLife:
    """One decode's ticks: taken by a unit, input in, compute, done."""

    dispatch: int
    ready: int
    compute_start: int
    end: int


@dataclasses.dataclass(frozen=True)
class _TierRecords:
    """The switching records' weak inputs and services and strong waits."""

    weak_syndrome_weight_mean: Optional[float] = None
    weak_syndrome_weight_max: Optional[int] = None
    weak_service_mean_us: Optional[float] = None
    strong_wait_mean_us: Optional[float] = None
    strong_wait_max_us: Optional[float] = None
    strong_held_in_units_max: Optional[int] = None


def _logical_verdicts(
    observation: observation_module.Observation,
    result: result_records.RunResult,
) -> _LogicalVerdicts:
    """The loop's observables beside the truth and beside the reference.

    A shot fails when any of its operations reads a wrong observable, as
    a computation fails when any of its logical qubits does; a workload
    of several patches is judged over all of them.
    """
    is_logical_failure = False
    is_direct_failure = False
    is_direct_mismatch = False
    for operation_result in result.operation_results:
        operation_id = operation_result.operation_id
        reference_prediction = direct_prediction(observation, operation_id)
        truth = tuple(operation_result.observable_truth)
        loop_prediction = tuple(operation_result.logical_observables)
        is_logical_failure |= loop_prediction != truth
        is_direct_failure |= reference_prediction != truth
        is_direct_mismatch |= loop_prediction != reference_prediction
    return _LogicalVerdicts(
        logical_failure=is_logical_failure,
        direct_failure=is_direct_failure,
        direct_mismatch=is_direct_mismatch,
    )


def _throughput_per_microsecond(
    observation: observation_module.Observation, samples: dict
) -> _Throughput:
    """Windows and rounds over the span from first round to last commit.

    The rounds are the ones the QPU read out, each (operation, round)
    once (the EMITTED rows, qpu/cycle_clock.py), so the number holds for
    any workload, not only the maker that declares rounds_per_shot.
    Operations that run at once add their rounds: the rate is the load
    the shared decoders serve, which the backlog condition weighs against
    their service rate (Holmes 2004.04794 section III), not the QEC
    cycle rate, which is one over the round period by construction.
    """
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

    The plan's windows queue in the default pool and a strong re-decode
    in the strong pool (decode_queue.POOL_BY_JOB_KIND), so the default
    pool's numbers are the primary tier's: the weak tier's on a
    weak-primary run, the strong tier's under strong_only, where the
    weak columns read zero; a run without a pool reads zero for it.
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

    Toshio's Theorem 1 bounds one strong decode's time by the round
    time times d over the switching rate (2510.25222 eq. (6)); the
    report forms that bound per sweep point from the escalated windows,
    and the mean service here is the time it bounds.
    """
    stages = observation.stages
    windows = 0
    rounds = 0
    services = []
    for frame_record in observation.frame_corrections.committed:
        tier = window_records.DecoderTier(frame_record.tier)
        if tier is not window_records.DecoderTier.STRONG:
            continue
        decode = _committed_decode(stages, frame_record)
        windows += 1
        rounds += decode.round_count
        service = _span_microseconds(
            decode.done_ticks, decode.compute_start_ticks
        )
        services.append(service)
    service_mean_us = _mean_or_zero(services)
    return _StrongDecodes(windows, rounds, service_mean_us)


def _tier_records(
    observation: observation_module.Observation,
) -> _TierRecords:
    """Each tier's inputs, computes and waits off the switching records.

    Every request record names its tier and run ordinal in its request
    key (records/windows.py DecoderRequestKey), and the stage ledger
    names the decode each stage served by the same ordinal, so a
    decode's compute is its own stages, first start to last end, the
    service point's span. A strong decode's wait is the two waits the
    latency points split, queue_wait (enqueue to a unit taking it) and
    compute_wait (its input landed to its compute start, the unit busy
    with another decode, gem5's fuBusy), and not the input hop between
    them, which a free unit pays too.
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

    A strong decode is held from the tick it may compute, its input
    landed and no boundary owed, to its compute start: the compute_wait
    of the latency points, the wait gem5 counts apart as fuBusy when a
    ready instruction finds no free functional unit (gem5
    src/cpu/o3/inst_queue.cc:1009-1014). A merged batch is one decode
    in one unit and counts once. A depth counts only when time passes
    at it, the rule of the ready queue's peak (observe/queue_depth.py),
    so a decode that starts on the tick it may start was never held.
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


def parallel_processes_needed(
    samples: dict,
    settings: machine_settings.MachineSettings,
    distance: int,
    round_period_microseconds: float,
) -> int:
    """Skoric's least count of parallel decoding processes for no backlog.

    N_par >= 2 tau_W / ((n_com + n_W) tau_rd): a window's decoding time
    twice, over the rounds its two layers commit (2209.08552 lines
    429-438; NVQLink states the same as its equation 2, 2510.25213
    lines 1625-1631). Layer A commits n_com rounds and layer B its
    whole window, and n_W is the window with a buffer on each side,
    n_com + 2 n_buf, "nW = 3w" at the paper's sizes (lines 388-390).
    tau_W is this shot's mean service and the sizes are the window
    scheme's, d when null. It is the count the parallel window scheme
    needs; the serial chain's own condition is chain_load.
    """
    service_us = _mean_or_zero(samples["service"])
    commit_rounds = commit_round_count(settings, distance)
    buffer_rounds = settings.windows.buffer_rounds
    if buffer_rounds is None:
        buffer_rounds = distance
    both_buffers_round_count = 2 * buffer_rounds
    window_round_count = commit_rounds + both_buffers_round_count
    committed_round_count = commit_rounds + window_round_count
    committed_rounds_us = committed_round_count * round_period_microseconds
    both_layers_service_us = 2 * service_us
    processes = both_layers_service_us / committed_rounds_us
    return math.ceil(processes)


def _span_microseconds(end_ticks: int, start_ticks: int) -> float:
    span_ticks = end_ticks - start_ticks
    return config_module.ticks_to_microseconds(span_ticks)


def _hop_start_ticks(row: dict) -> int:
    """The tick a transfer was asked for, which is where its hop starts.

    A transfer's send_ticks follows its wait for the channel's setup
    engine and its own setup, and total_delay_ticks counts from the
    request (records/transfers.py, Transfer), so the request is the
    delivery less the total. The setup is the hop's own cost: gem5 adds
    a DMA's fixed delay to the completion its requester sees
    (src/dev/dma_device.cc:116-118), and a point that started at the
    send would charge it to whatever wait comes before the hop.
    """
    return row["delivery_ticks"] - row["total_delay_ticks"]


def _committed_decode(stages, frame_record) -> _CommittedDecode:
    """The committing decode's tier and the ticks of its own life.

    Its stage records are the ones whose request ordinals hold the
    frame record's, so a merged batch answers for every window it
    served. The compute began at its first stage and ended at its last,
    and the records carry the two ticks before that compute: the tick a
    unit took this decode and the tick its input was readable in that
    unit's memory. Every one of them belongs to this decode, so a window
    decoded more than once never mixes one decode's dispatch with
    another's compute.
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
    return _CommittedDecode(tier, first, last, ready, run_sequence, rounds)


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

    The window's queue wait ends there, and the weak attempt of an
    escalated window starts there (Toshio et al. 2510.25222 Sec. III A
    steps 2 to 4, lines 602-617: the weak decoder answers, the verdict
    is read, and only then does the strong decode that commits begin).
    A window decoded once has one dispatch and this is it; a window that
    ran its two forced-class solves (decision D2) or escalated has
    several, and the first is the one its wait in the ready queue ended
    at. A decode the cancel closed is not on the window's path at all
    (Toshio Sec. III A step 1), so it is left out.
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

    The weak tier reports a soft output for each window and the window
    is answered on it, so the signal's own computation is on the
    window's path (Toshio et al. 2510.25222 lines 657-663, the weak
    decoder "simultaneously generates a soft output g for each decoding
    window", and lines 360-363, the response time running to the
    correction). What that computation is depends on the row: the walk
    under cluster_gap, charged on the weak unit (decision D8), and the
    sibling forced-class solve's remaining time under complementary_gap,
    which is that solve's own service on the same unit (decision D2).
    An escalated window has none: its committing decode is the strong
    one, which answers after the verdict, and weak_attempt already runs
    from the window's first dispatch to that verdict.
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

    An escalated window answers once on the weak tier and the strong
    decode starts from that verdict (Toshio et al. 2510.25222 Sec. III A
    steps 2 to 4, lines 602-617), so the span from the unit assignment
    to the verdict is the weak_attempt point and the committing decode's
    own hop starts there. When the committing decode was already
    computing at that answer, which is a kept weak result and which is
    run_both_at_once's parallel sibling, the attempt cost the window
    nothing and both start at the dispatch.
    """
    if window.t_done >= decode.compute_start_ticks:
        return first_dispatch
    return window.t_done


def _stage_microseconds(stages, frame_record) -> dict:
    """The committing decode's stages, in microseconds, by stage name.

    Every decode of a window records its stages under the window's key:
    the two forced-class solves of a complementary gap, which are two
    jobs of one window and are charged one card each (decisions D2 and
    D7), the strong re-decode of an escalated window, and a sibling that
    was cancelled. Keeping the last record of each name mixes them, so
    the stages are the committing request's, which is the rule every
    other point of the window follows.
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


def _write_trace(
    shot, run_dir, label: str, only_traced_shot: bool
) -> Optional[str]:
    """The shot's Chrome trace, when the section asked and named it.

    trace: chrome names the file by the shot's label, its point id and
    seed, under trace/; a path of its own is written where it says with
    that label in the name, so no shot overwrites another's file, unless
    the run traces this shot alone and no other can take the path. Only
    the shots trace_shots names are written, so a sweep point of two
    thousand shots writes one file. The file it wrote
    comes back, so the shot's measurement can say where its trace is; a
    shot that was not traced returns None.
    """
    observation = shot.task.settings.observation
    if not observation.writes_trace:
        return None
    if shot.seed not in observation.trace_shots:
        return None
    writer = shot.machine.observation.trace_writer
    path = observation.trace_path
    if path is None:
        trace_dir = run_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{label}.trace.json"
    elif not only_traced_shot:
        path = trace_path_for_shot(path, label)
    written = str(path)
    writer.write(written)
    return written


def _write_log(
    observation: observation_module.Observation, run_dir, label: str
) -> None:
    """One file per shot with the engine narrator's full line record.

    The same lines log: print shows live. The Chrome trace of the data
    path is a separate file under trace/, written for the shots
    observation.trace_shots names.
    """
    log_dir = run_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(observation.log.lines)
    log_path = log_dir / f"{label}.log"
    contents = text + "\n"
    log_path.write_text(contents)
