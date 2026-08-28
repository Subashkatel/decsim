"""One completed run -> one shot's numbers. Nothing is built here.

A shot is one circuit through the whole reaction path. `measure_shot` runs
it and reads every latency point in POINTS off the run's own records: the
link ledger, the window stamps, the engine's stage records and the Pauli
frame. The input and output transfers are named by role, not by wire,
because the mode picks the wire: weak_baseline moves windows on wbd and
results on wdo, strong_only on sbd and do.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np
import pymatching
import stim

from decsim.config import TICKS_PER_US

from experiments.build_run import build_run
from experiments.experiment_config import ExperimentConfig

# Latency points, in path order, in microseconds per window unless noted.
POINTS = (
    # controller -> Buffer 0, one round (latency + serialization + queue)
    "cwb_per_round",
    # first round in window arrives -> last arrives (waiting on the QPU)
    "buffer_fill",
    "dep_block",            # window complete -> job queued (dependencies)
    "queue_wait",           # queued -> unit assigned (ready-queue wait)
    # unit assigned -> input in decoder memory (wbd weak, sbd strong)
    "input_link_per_window",
    "fetch",                # engine: read the window out of decoder memory
    "algorithm",            # engine: the decoding algorithm
    "release",              # engine: correction write-out
    # unit assigned -> decode done (transfer + fetch + algorithm + release)
    "service",
    "dd_per_window",        # decoder -> next decoder, the boundary handoff
    "output_link_per_window",  # decoder -> Pauli frame (wdo weak, do strong)
    "frame_commit",         # Pauli frame accepted -> committed
    # Totals. The buffer0 pair starts the clock at Buffer 0 publication;
    # the qpu pair starts it when the round leaves the QPU (the QC send),
    # so it includes QC, controller processing, packing and CWB.
    "buffer0_ready_to_frame",        # window complete in Buffer 0 -> frame
    "buffer0_first_round_to_frame",  # first round in Buffer 0 -> frame
    "qpu_last_round_to_frame",       # last required round off QPU -> frame
    "qpu_first_round_to_frame",      # first required round off QPU -> frame
)

INPUT_LINK = {"weak_baseline": "wbd", "strong_only": "sbd",
              "switching": "wbd"}
OUTPUT_LINK = {"weak_baseline": "wdo", "strong_only": "do",
               "switching": "wdo"}


@dataclass(frozen=True)
class ShotMeasurement:
    physical_error_probability: float
    distance: int
    round_period_us: float
    algorithm: object       # the active unit's card: a name or a latency in us
    seed: int
    windows: int
    logical_failure: bool
    samples: dict          # point -> us list, one per window (per round
                           # for cwb)
    means: dict            # point -> mean us over this shot's windows
    maxes: dict            # point -> max us
    load: float            # service per window / window inter-arrival
    direct_failure: bool   # whole-circuit PyMatching on the same events
                           # failed
    direct_mismatch: bool  # loop prediction differs from direct PyMatching
    throughput_windows_per_us: float
    throughput_rounds_per_us: float
    max_queued_windows: int
    tesseract_windows_checked: int       # referee re-decodes (0 = referee off)
    tesseract_window_disagreements: int  # referee reached a different owned
                                         # observable contribution
    link_totals: dict      # path -> the run's own ledger counters plus
                           # rounds/windows context, for links.csv; totals
                           # come straight off TrafficCounters, no manual
                           # counting (the ledger refuses counts that do
                           # not reconcile per channel)
    sim_wall_seconds: float


def us(ticks: int) -> float:
    return ticks / TICKS_PER_US


def link_totals(traffic: dict) -> dict:
    """path -> the ledger's own counters for this shot, in microseconds."""
    totals = {}
    for edge in traffic["semantic_edges"]:
        counters = edge["counters"]
        totals[edge["path"]] = {
            "transfers": counters["transfer_count"],
            "payload_bits": counters["known_payload_bits"],
            "unknown_payload_transfers":
                counters["unknown_payload_transfer_count"],
            "queue_wait_us": us(counters["queue_wait_ticks"]),
            "serialization_us": us(counters["serialization_ticks"]),
            "propagation_us": us(counters["propagation_ticks"]),
        }
    return totals


def link_delay_by_window(transfers: list) -> dict:
    """(path, window_id) -> total ticks from send to delivery over that path."""
    delay = {}
    for row in transfers:
        key = (row["path"], row["attribution"]["window_id"])
        transfer_ticks = row["delivery_ticks"] - row["send_ticks"]
        delay[key] = delay.get(key, 0) + transfer_ticks
    return delay


def cwb_delays_us(transfers: list) -> list:
    """Every round's controller-to-Buffer-0 delay, in microseconds."""
    delays = []
    for row in transfers:
        if row["path"] != "cwb":
            continue
        delays.append(us(row["delivery_ticks"] - row["send_ticks"]))
    return delays


def qc_send_ticks(transfers: list) -> dict:
    """round -> tick that round left the QPU (its earliest QC send)."""
    send = {}
    for row in transfers:
        if row["path"] != "qc":
            continue
        round_index = row["attribution"]["round_lo"]
        earlier = send.get(round_index)
        if earlier is None or row["send_ticks"] < earlier:
            send[round_index] = row["send_ticks"]
    return send


def window_points_us(window, frame_record, stage_us: dict, link_delay: dict,
                     qc_send: dict, input_path: str, output_path: str) -> dict:
    """The per-window latency points, in us, for one decoded window."""
    window_id = window.key[1]
    last_emitted_round = max(qc_send)
    last_required_round = min(window.buffer_hi, last_emitted_round)
    return {
        "buffer_fill": us(window.t_data_complete - window.t_first_round),
        "dep_block": us(window.t_queued - window.t_data_complete),
        "queue_wait": us(window.t_dispatch - window.t_queued),
        "input_link_per_window": us(link_delay.get((input_path, window_id), 0)),
        "fetch": stage_us["fetch"],
        "algorithm": stage_us["algorithm"],
        "release": stage_us["release"],
        "service": us(window.t_done - window.t_dispatch),
        "dd_per_window": us(link_delay.get(("dd", window_id), 0)),
        "output_link_per_window":
            us(link_delay.get((output_path, window_id), 0)),
        "frame_commit":
            us(frame_record.committed_ticks - frame_record.accepted_ticks),
        "buffer0_ready_to_frame":
            us(frame_record.committed_ticks - window.t_data_complete),
        "buffer0_first_round_to_frame":
            us(frame_record.committed_ticks - window.t_first_round),
        "qpu_last_round_to_frame":
            us(frame_record.committed_ticks - qc_send[last_required_round]),
        "qpu_first_round_to_frame":
            us(frame_record.committed_ticks - qc_send[window.start_round]),
    }


def collect_samples(completed, engine, mode: str) -> dict:
    """point -> list of microsecond samples over this shot's decoded windows."""
    transfers = completed.result.link_traffic["transfers"]
    link_delay = link_delay_by_window(transfers)
    qc_send = qc_send_ticks(transfers)
    frame_by_window = {record.window_key[1]: record
                       for record in completed.pauli_frame.snapshot().records}
    samples = {point: [] for point in POINTS}
    samples["cwb_per_round"] = cwb_delays_us(transfers)
    all_windows = sorted(completed.window_manager.windows.items())
    for (op_id, window_id), window in all_windows:
        frame_record = frame_by_window.get(window_id)
        decoded = frame_record is not None and window.t_done is not None
        if not decoded:
            continue
        stage_records = engine.stage_records_for(op_id, window_id)
        stage_us = {record.stage: us(record.end_ticks - record.start_ticks)
                    for record in stage_records}
        points = window_points_us(window, frame_record, stage_us, link_delay,
                                  qc_send, INPUT_LINK[mode], OUTPUT_LINK[mode])
        for point, value in points.items():
            samples[point].append(value)
    return samples


def direct_prediction(completed, circuit: stim.Circuit) -> tuple:
    """Whole-circuit PyMatching on the detection events the device sampled:
    the reference the loop must agree with."""
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    operation_id = completed.result.operation_results[0].operation_id
    events = np.asarray(
        completed.qpu.model.sampled_detection_events(operation_id), dtype=bool)
    predicted = matching.decode(events)
    return tuple(int(bit) for bit in predicted)


def chain_load(samples: dict, config: ExperimentConfig, distance: int,
               round_period_us: float) -> float:
    """rho: the serial chain's service per window (unit assigned -> decode
    done, plus the DD boundary handoff) over the window inter-arrival time
    (commit rounds x round period). Above 1 the chain cannot keep up."""
    service_samples = samples["service"]
    handoff_samples = samples["dd_per_window"]
    service_us = statistics.fmean(service_samples) if service_samples else 0.0
    handoff_us = statistics.fmean(handoff_samples) if handoff_samples else 0.0
    commit_rounds = config.windowing.commit_rounds or distance
    inter_arrival_us = commit_rounds * round_period_us
    return (service_us + handoff_us) / inter_arrival_us


def _shot_label(config: ExperimentConfig, physical_error_probability: float,
                distance: int, round_period_us: float, seed: int) -> str:
    return (f"p{physical_error_probability:g}_d{distance}"
            f"_algo{config.active_decoder.algorithm}"
            f"_round{round_period_us:g}us_seed{seed}")


def _write_trace(completed, run_dir, label: str) -> None:
    """One file per shot with the engine narrator's full line record: the
    same lines trace: print shows live."""
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    name = f"{label}.log"
    (trace_dir / name).write_text(
        "\n".join(completed.engine.log_lines) + "\n")


def measure_shot(config: ExperimentConfig, *, physical_error_probability: float,
                 distance: int, round_period_us: float, seed: int,
                 run_dir=None, threshold_calibrator=None) -> ShotMeasurement:
    """run_dir receives the trace file when trace: file|both is on;
    None writes nothing beyond the returned measurement.
    threshold_calibrator is the sweep point's online controller
    (threshold_source online): one instance across the point's shots."""
    spec, engine = build_run(
        config, physical_error_probability=physical_error_probability,
        distance=distance, round_period_us=round_period_us, seed=seed,
        threshold_calibrator=threshold_calibrator)
    wall_start = time.perf_counter()
    completed = spec.build(verbose=config.trace in ("print", "both"),
                           io_trace=config.trace_io)
    wall_seconds = time.perf_counter() - wall_start
    if completed.result.terminal_status != "complete":
        raise RuntimeError(
            f"run did not complete: {completed.result.terminal_status}")
    if run_dir is not None and config.trace in ("file", "both"):
        _write_trace(completed, run_dir,
                     _shot_label(config, physical_error_probability, distance,
                                 round_period_us, seed))

    samples = collect_samples(completed, engine, config.mode)
    decoded_windows = len(samples["service"])
    rounds_this_shot = config.rounds_per_shot.rounds_for(distance)
    load = chain_load(samples, config, distance, round_period_us)
    operation_result = completed.result.operation_results[0]
    truth = tuple(operation_result.observable_truth)
    loop_prediction = tuple(operation_result.logical_observables)
    reference_prediction = direct_prediction(completed, spec.ops[0].circuit)
    windows = completed.window_manager.windows.values()
    first_round_tick = min(window.t_first_round for window in windows
                           if window.t_first_round is not None)
    frame_records = completed.pauli_frame.snapshot().records
    last_commit_tick = max(record.committed_ticks for record in frame_records)
    span_us = us(last_commit_tick - first_round_tick)
    queue_depths = [depth for _, depth in completed.decoder_manager.queue_log]
    return ShotMeasurement(
        physical_error_probability=physical_error_probability,
        distance=distance,
        round_period_us=round_period_us,
        algorithm=config.active_decoder.algorithm,
        seed=seed, windows=decoded_windows,
        logical_failure=loop_prediction != truth,
        samples=samples,
        means={point: (statistics.fmean(values) if values else 0.0)
               for point, values in samples.items()},
        maxes={point: (max(values) if values else 0.0)
               for point, values in samples.items()},
        load=load,
        direct_failure=reference_prediction != truth,
        direct_mismatch=loop_prediction != reference_prediction,
        throughput_windows_per_us=decoded_windows / span_us,
        throughput_rounds_per_us=rounds_this_shot / span_us,
        max_queued_windows=max(queue_depths, default=0),
        tesseract_windows_checked=getattr(
            engine.decoder, "windows_checked", 0),
        tesseract_window_disagreements=getattr(
            engine.decoder, "window_disagreements", 0),
        link_totals=link_totals(completed.result.link_traffic),
        sim_wall_seconds=wall_seconds)
