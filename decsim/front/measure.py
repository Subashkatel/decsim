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

import dataclasses
import pathlib
import statistics
from typing import Optional

import numpy
import pymatching
import stim

import decsim.build.escalation as escalation_build
import decsim.collect as collect
import decsim.config as config_module
import decsim.decoders.decoder_output as decoder_output
import decsim.observe.observation as observation_module
import decsim.records.results as result_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings

# Latency points, in path order, in microseconds per window unless noted.
# Every point is one span, and its comment names the two ticks it runs
# between.
POINTS = (
    # per round: the controller's send -> the round readable in Buffer 0
    "cwb_per_round",
    # per round: the round found Buffer 0 full -> the slot that freed
    # admitted it, zero for a round that found room
    "cwb_stall_per_round",
    # per round: the same wait in front of Buffer 1
    "csb_stall_per_round",
    # the window's first round readable -> its last (waiting on the QPU)
    "buffer_fill",
    # the committing decode's input landed in its unit's memory -> the
    # first tick it may compute: the dependency wait, for the
    # predecessor's boundary and for the escalation message beside the
    # input hop, and zero when nothing was owed at the landing. An input
    # the decode found already there starts this at the tick a unit took
    # the decode instead
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
    # weak decoder -> strong decoder, the escalation hop, zero for a
    # window that did not escalate
    "escalation_link_per_window",
    # the decode's end -> the boundary readable in the next window
    "dd_per_window",
    # the decode's end -> the correction at the Pauli frame
    "output_link_per_window",
    "frame_commit",  # the frame accepted the correction -> committed
    # Totals. The buffer0 pair starts the clock at Buffer 0 publication;
    # the qpu pair starts it when the round leaves the QPU (the QC send),
    # so it includes QC, controller processing, packing and CWB.
    "buffer0_ready_to_frame",  # window complete in Buffer 0 -> frame
    "buffer0_first_round_to_frame",  # first round in Buffer 0 -> frame
    "qpu_last_round_to_frame",  # last required round off QPU -> frame
    "qpu_first_round_to_frame",  # first required round off QPU -> frame
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

    physical_error_probability: float
    distance: int
    round_period_us: float
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


def measure_shot(shot: collect.Shot, run_dir=None) -> ShotMeasurement:
    """Read one collected shot's numbers off its machine and result.

    run_dir receives the log file when log: file|both is on and the
    Chrome trace when trace names one and the shot is in trace_shots;
    None writes nothing beyond the returned measurement.
    """
    settings = shot.task.settings
    physical_error_probability = settings.workload.physical_error_probability
    distance = settings.qpu.distance
    round_period_us = settings.qpu.round_period_microseconds
    label = shot_label(settings, shot.seed)
    observation = shot.machine.observation
    if run_dir is not None and settings.observation.writes_log:
        _write_log(observation, run_dir, label)
    trace_path = None
    if run_dir is not None:
        trace_path = _write_trace(shot, run_dir, label)
    return _measurement(
        settings,
        observation,
        shot.result,
        physical_error_probability=physical_error_probability,
        distance=distance,
        round_period_us=round_period_us,
        seed=shot.seed,
        wall_seconds=shot.wall_seconds,
        trace_path=trace_path,
    )


def ticks_to_microseconds(ticks: int) -> float:
    """A tick count as a float of microseconds, for the csv columns."""
    return ticks / config_module.TICKS_PER_MICROSECOND


def link_totals(traffic: dict) -> dict:
    """The ledger's own counters for this shot by path, in microseconds."""
    totals = {}
    for edge in traffic["semantic_edges"]:
        counters = edge["counters"]
        totals[edge["path"]] = {
            "transfers": counters["transfer_count"],
            "payload_bits": counters["known_payload_bits"],
            "unknown_payload_transfers": counters[
                "unknown_payload_transfer_count"
            ],
            "queue_wait_us": ticks_to_microseconds(
                counters["queue_wait_ticks"]
            ),
            "serialization_us": ticks_to_microseconds(
                counters["serialization_ticks"]
            ),
            "propagation_us": ticks_to_microseconds(
                counters["propagation_ticks"]
            ),
        }
    return totals


def link_delay_by_window(transfers: list) -> dict:
    """Ticks from send to delivery by (path, window id), summed per key."""
    delay = {}
    for row in transfers:
        key = (row["path"], row["attribution"]["window_id"])
        transfer_ticks = row["delivery_ticks"] - row["send_ticks"]
        delay[key] = delay.get(key, 0) + transfer_ticks
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
        delay += row["delivery_ticks"] - row["send_ticks"]
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
    """Every round's controller-to-Buffer-0 delay, in microseconds."""
    delays = []
    for row in transfers:
        if row["path"] != "controller_to_weak_buffer":
            continue
        delay = _span_microseconds(row["delivery_ticks"], row["send_ticks"])
        delays.append(delay)
    return delays


def round_stall_ticks(round_events: list) -> dict:
    """Ticks each held round waited for store room, by (operation, round).

    The wait is the waiting line's own two events: the refusal that held
    the round and the freed slot that admitted it
    (controller/round_writes.py, HeldRounds). A round that found room is
    not in here at all.
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
        delay = ticks_to_microseconds(ticks)
        delays.append(delay)
    return delays


def weak_store_round_keys(round_events: list) -> list:
    """Every round that left for Buffer 0, in the order they left."""
    keys = []
    for event in round_events:
        if event.kind != "CWB_SENT":
            continue
        key = (event.operation_id, event.round_index)
        keys.append(key)
    return keys


def strong_store_round_keys(stored_rounds: list) -> list:
    """Every round that landed in Buffer 1, in the order they landed."""
    keys = []
    for _tick, operation_id, round_index in stored_rounds:
        key = (operation_id, round_index)
        keys.append(key)
    return keys


def qpu_send_ticks(transfers: list) -> dict:
    """The tick each round left the QPU (its earliest QC send), by round."""
    send = {}
    for row in transfers:
        if row["path"] != "qpu_to_controller":
            continue
        round_index = row["attribution"]["round_lo"]
        earlier = send.get(round_index)
        if earlier is None or row["send_ticks"] < earlier:
            send[round_index] = row["send_ticks"]
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
    the dependency wait, from the input landing in the unit's memory to
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
    rather than one number (tmp/resources/l5_buffers/Ciw/
    ciw/data_record.py lines 3-21).
    """
    window_id = window.key[1]
    last_emitted_round = max(qpu_send)
    last_required_round = min(window.buffer_hi, last_emitted_round)
    committed = frame_record.committed_ticks
    handoff_ticks = link_delay.get(("decoder_to_decoder", window_id), 0)
    escalation_ticks = link_delay.get(
        ("weak_decoder_to_strong_decoder", window_id), 0
    )
    output_ticks = link_delay.get((output_path, window_id), 0)
    attempt_end = _attempt_end_ticks(window, decode, first_dispatch)
    input_key = (input_path, window_id, decode.run_sequence)
    input_ticks, input_landed = input_hop.get(input_key, (0, attempt_end))
    startable = _startable_ticks(decode, input_landed)
    confidence_ticks = _confidence_ticks(window, decode)
    last_required_send = qpu_send[last_required_round]
    first_required_send = qpu_send[window.start_round]
    return {
        "buffer_fill": _span_microseconds(
            window.t_data_complete, window.t_first_round
        ),
        "dep_block": _span_microseconds(startable, input_landed),
        "compute_wait": _span_microseconds(
            decode.compute_start_ticks, startable
        ),
        "queue_wait": _span_microseconds(first_dispatch, window.t_queued),
        "input_link_per_window": ticks_to_microseconds(input_ticks),
        "fetch": stage_us["fetch"],
        "algorithm": stage_us["algorithm"],
        "release": stage_us["release"],
        "service": _span_microseconds(
            decode.done_ticks, decode.compute_start_ticks
        ),
        "confidence": ticks_to_microseconds(confidence_ticks),
        "weak_attempt": _span_microseconds(attempt_end, first_dispatch),
        "escalation_link_per_window": ticks_to_microseconds(escalation_ticks),
        "dd_per_window": ticks_to_microseconds(handoff_ticks),
        "output_link_per_window": ticks_to_microseconds(output_ticks),
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
    """The frame's committed corrections, by the window each one wrote."""
    records = {}
    for record in observation.frame_corrections.committed:
        window_id = record.window_key[1]
        records[window_id] = record
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
    for (_operation_id, window_id), window in all_windows:
        frame_record = frame_by_window.get(window_id)
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
    source drew for this operation, heard at sampling time.
    """
    shot = observation.sampled_shots.shots_by_operation[operation_id]
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


def chain_load(
    samples: dict,
    settings: machine_settings.MachineSettings,
    distance: int,
    round_period_us: float,
) -> float:
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
    commit_rounds = settings.windows.commit_rounds
    if commit_rounds is None:
        commit_rounds = distance
    inter_arrival_us = commit_rounds * round_period_us
    chain_us = service_us + confidence_us + handoff_us
    return chain_us / inter_arrival_us


def active_decoder_kind(settings: machine_settings.MachineSettings):
    """The kind of the tier that decodes the plan's windows."""
    tier = escalation_build.primary_tier(settings.escalation)
    tier_settings = settings.decoder_settings_for(tier)
    return tier_settings.kind


def trace_path_for_shot(path: str, seed: int, trace_shots) -> str:
    """The path one shot writes to; the seed joins it when several trace.

    One traced shot keeps the path the yaml gave. Several would all
    write the same file, so each takes the seed before its suffixes:
    run.trace.json becomes run_seed3.trace.json, and run.trace.json.gz
    becomes run_seed3.trace.json.gz.
    """
    if len(trace_shots) < 2:
        return path
    name = pathlib.Path(path)
    directory = name.parent
    stem = name.name
    first_dot = stem.find(".")
    if first_dot < 0:
        seeded = directory / f"{stem}_seed{seed}"
        return str(seeded)
    head = stem[:first_dot]
    suffixes = stem[first_dot:]
    seeded = directory / f"{head}_seed{seed}{suffixes}"
    return str(seeded)


def shot_label(settings: machine_settings.MachineSettings, seed: int) -> str:
    """The name a shot's log and trace files carry: its point and seed."""
    physical_error_probability = settings.workload.physical_error_probability
    distance = settings.qpu.distance
    round_period_us = settings.qpu.round_period_microseconds
    algorithm = active_decoder_kind(settings)
    return (
        f"p{physical_error_probability:g}_d{distance}_algo{algorithm}"
        f"_round{round_period_us:g}us_seed{seed}"
    )


def _measurement(
    settings: machine_settings.MachineSettings,
    observation: observation_module.Observation,
    result: result_records.RunResult,
    *,
    physical_error_probability: float,
    distance: int,
    round_period_us: float,
    seed: int,
    wall_seconds: float,
    trace_path: Optional[str],
) -> ShotMeasurement:
    """Read every number of one completed shot off its records."""
    samples = collect_samples(observation, result)
    verdicts = _logical_verdicts(observation, result)
    throughput = _throughput_per_microsecond(
        observation, settings, samples, distance
    )
    referee = _referee_counts(observation)
    decoded_windows = len(samples["service"])
    load = chain_load(samples, settings, distance, round_period_us)
    algorithm = active_decoder_kind(settings)
    queued = _max_queued_windows(observation)
    totals = link_totals(result.link_traffic)
    means = _means(samples)
    maxes = _maxes(samples)
    return ShotMeasurement(
        physical_error_probability=physical_error_probability,
        distance=distance,
        round_period_us=round_period_us,
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
        tesseract_windows_checked=referee.windows_checked,
        tesseract_window_disagreements=referee.window_disagreements,
        link_totals=totals,
        sim_wall_seconds=wall_seconds,
        data_movement=result.data_movement,
        trace_path=trace_path,
    )


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
    dispatch_ticks: int  # a unit took this decode
    ready_ticks: Optional[int]  # it first may compute, whatever the unit did
    run_sequence: int  # the run ordinal of the request it committed


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


def _logical_verdicts(
    observation: observation_module.Observation,
    result: result_records.RunResult,
) -> _LogicalVerdicts:
    """The loop's observables beside the truth and beside the reference."""
    operation_result = result.operation_results[0]
    operation_id = operation_result.operation_id
    reference_prediction = direct_prediction(observation, operation_id)
    truth = tuple(operation_result.observable_truth)
    loop_prediction = tuple(operation_result.logical_observables)
    is_logical_failure = loop_prediction != truth
    is_direct_failure = reference_prediction != truth
    is_direct_mismatch = loop_prediction != reference_prediction
    return _LogicalVerdicts(
        logical_failure=is_logical_failure,
        direct_failure=is_direct_failure,
        direct_mismatch=is_direct_mismatch,
    )


def _throughput_per_microsecond(
    observation: observation_module.Observation,
    settings: machine_settings.MachineSettings,
    samples: dict,
    distance: int,
) -> _Throughput:
    """Windows and rounds over the span from first round to last commit."""
    decoded_windows = len(samples["service"])
    rounds_per_shot = settings.workload.rounds_per_shot
    rounds_this_shot = rounds_per_shot.rounds_for(distance)
    span_us = _decoded_span_microseconds(observation)
    windows_per_us = decoded_windows / span_us
    rounds_per_us = rounds_this_shot / span_us
    return _Throughput(
        windows_per_microsecond=windows_per_us,
        rounds_per_microsecond=rounds_per_us,
    )


def _referee_counts(
    observation: observation_module.Observation,
) -> _RefereeCounts:
    """The referee's counters; zero when no referee wraps the decoder."""
    audit = observation.referee_audit
    return _RefereeCounts(
        windows_checked=audit.windows_checked,
        window_disagreements=audit.window_disagreements,
    )


def _max_queued_windows(
    observation: observation_module.Observation,
) -> int:
    """The deepest the decode ready queue got over the shot."""
    depths = []
    for _tick, depth in observation.queue_depth.samples:
        depths.append(depth)
    return max(depths, default=0)


def _span_microseconds(end_ticks: int, start_ticks: int) -> float:
    span_ticks = end_ticks - start_ticks
    return ticks_to_microseconds(span_ticks)


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
    dispatch = _dispatch_ticks(records, first)
    ready = _ready_ticks(records)
    run_sequence = frame_record.run_sequence
    return _CommittedDecode(tier, first, last, dispatch, ready, run_sequence)


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


def _write_trace(shot, run_dir, label: str) -> Optional[str]:
    """The shot's Chrome trace, when the section asked and named it.

    trace: chrome names the file after the point, under trace/; a path
    of its own is written where it says, with the shot's seed in the
    name when trace_shots asks for more than one, so no shot overwrites
    another's file. Only the shots trace_shots names are written, so a
    sweep point of two thousand shots writes one file
    (trace_and_viewer.md section 10, ruling 1). The file it wrote comes
    back, so the shot's measurement can say where its trace is; a shot
    that was not traced returns None.
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
    else:
        path = trace_path_for_shot(path, shot.seed, observation.trace_shots)
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
