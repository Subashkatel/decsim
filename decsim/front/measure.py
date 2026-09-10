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
POINTS = (
    # controller -> Buffer 0, one round (latency + serialization + queue)
    "cwb_per_round",
    # first round in window arrives -> last arrives (waiting on the QPU)
    "buffer_fill",
    "dep_block",  # window complete -> job queued (dependencies)
    "queue_wait",  # queued -> unit assigned (ready-queue wait)
    # unit assigned -> input in decoder memory (the tier's input link)
    "input_link_per_window",
    "fetch",  # engine: read the window out of decoder memory
    "algorithm",  # engine: the decoding algorithm
    "release",  # engine: correction write-out
    # unit assigned -> decode done (transfer + fetch + algorithm + release)
    "service",
    "dd_per_window",  # decoder -> next decoder, the boundary handoff
    "output_link_per_window",  # decoder -> Pauli frame (the tier's link)
    "frame_commit",  # Pauli frame accepted -> committed
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
# the escalation row declares rather than by the row's name
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


def controller_to_weak_buffer_delays_us(transfers: list) -> list:
    """Every round's controller-to-Buffer-0 delay, in microseconds."""
    delays = []
    for row in transfers:
        if row["path"] != "controller_to_weak_buffer":
            continue
        delay = _span_microseconds(row["delivery_ticks"], row["send_ticks"])
        delays.append(delay)
    return delays


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
    stage_us: dict,
    link_delay: dict,
    qpu_send: dict,
    input_path: str,
    output_path: str,
) -> dict:
    """The per-window latency points, in us, for one decoded window."""
    window_id = window.key[1]
    last_emitted_round = max(qpu_send)
    last_required_round = min(window.buffer_hi, last_emitted_round)
    committed = frame_record.committed_ticks
    input_ticks = link_delay.get((input_path, window_id), 0)
    handoff_ticks = link_delay.get(("decoder_to_decoder", window_id), 0)
    output_ticks = link_delay.get((output_path, window_id), 0)
    last_required_send = qpu_send[last_required_round]
    first_required_send = qpu_send[window.start_round]
    return {
        "buffer_fill": _span_microseconds(
            window.t_data_complete, window.t_first_round
        ),
        "dep_block": _span_microseconds(
            window.t_queued, window.t_data_complete
        ),
        "queue_wait": _span_microseconds(window.t_dispatch, window.t_queued),
        "input_link_per_window": ticks_to_microseconds(input_ticks),
        "fetch": stage_us["fetch"],
        "algorithm": stage_us["algorithm"],
        "release": stage_us["release"],
        "service": _span_microseconds(window.t_done, window.t_dispatch),
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
    escalation_row,
) -> dict:
    """Every point's microsecond samples over the shot's decoded windows.

    The escalation row's primary tier says which two links a window
    rides: in from that tier's store, and home to the frame. The
    function stays whole past the size prompt: it is one walk over the
    windows in key order, each window's points appended to the same
    lists, read top to bottom.
    """
    transfers = result.link_traffic["transfers"]
    link_delay = link_delay_by_window(transfers)
    qpu_send = qpu_send_ticks(transfers)
    frame_by_window = frame_records_by_window(observation)
    samples = {}
    for point in POINTS:
        samples[point] = []
    samples["cwb_per_round"] = controller_to_weak_buffer_delays_us(transfers)
    primary_tier = escalation_row.primary_tier
    input_link = INPUT_LINK_BY_TIER[primary_tier]
    output_link = decoder_output.FRAME_PATH_BY_TIER[primary_tier]
    input_path = input_link.value
    output_path = output_link.value
    window_items = observation.windows.windows.items()
    all_windows = sorted(window_items)
    stages = observation.stages
    for (operation_id, window_id), window in all_windows:
        frame_record = frame_by_window.get(window_id)
        if frame_record is None or window.t_done is None:
            continue
        stage_us = _stage_microseconds(stages, operation_id, window_id)
        points = window_points_us(
            window,
            frame_record,
            stage_us,
            link_delay,
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

    Service is unit assigned to decode done plus the DD boundary handoff;
    the window inter-arrival time is commit rounds times the round
    period. Above 1 the chain cannot keep up.
    """
    service_us = _mean_or_zero(samples["service"])
    handoff_us = _mean_or_zero(samples["dd_per_window"])
    commit_rounds = settings.windows.commit_rounds
    if commit_rounds is None:
        commit_rounds = distance
    inter_arrival_us = commit_rounds * round_period_us
    chain_us = service_us + handoff_us
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
    escalation_row = escalation_build.escalation_row(settings.escalation)
    samples = collect_samples(observation, result, escalation_row)
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


def _stage_microseconds(stages, operation_id, window_id) -> dict:
    """The recorded duration of each of a window's stages, in microseconds."""
    stage_us = {}
    for record in stages.records_for(operation_id, window_id):
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
