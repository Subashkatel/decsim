"""Baseline closed loop on real Stim data, measured at every point.

One shot = one rotated memory circuit run through the whole reaction path:
QPU rounds -> controller (pulses to binary) -> packing -> link C2B -> Buffer 0
-> window manager -> link CWD -> decoder memory -> decoder engine (fetch,
PyMatching algorithm, release) -> link WDO -> Pauli frame commit. Nothing is
skipped and every hop charges its configured cost; the report shows the
simulated latency at each point and the throughput, per sweep point.

Usage: python -m experiments.baseline_closed_loop [config.yaml]
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import stim
import yaml

from decsim.config import TICKS_PER_US, TimingConfig
from decsim.decoders.decoder_engine import DecoderEngine, DecoderStage, DecoderTiming
from decsim.decoders.decoder_memory import DecoderMemoryConfig
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.links import link_profiles
from decsim.links.link_profiles import with_controller_to_buffer_edge
from decsim.message import Operation
from decsim.pauli_frame.pauli_frame import PauliFrameConfig
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice
from decsim.run_spec import RunSpec

DEFAULT_CONFIG = Path(__file__).with_suffix(".yaml")
MEASURED = "measured"   # the algorithm-latency card that charges the real PyMatching wall clock

# Latency points, in path order, all in microseconds per window unless noted.
POINTS = (
    "c2b_per_round",        # controller -> Buffer 0, one round (link latency + serialization + queue)
    "buffer_fill",          # first round in window arrives -> last round arrives (waiting on the QPU)
    "dep_block",            # window complete -> job queued (window dependencies)
    "queue_wait",           # queued -> unit assigned (ready-queue wait only)
    "cwd_per_window",       # unit assigned -> input in its decoder memory, link CWD
    "fetch",                # decoder engine: read the window out of decoder memory
    "algorithm",            # decoder engine: the decoding algorithm
    "release",              # decoder engine: correction write-out
    "service",              # unit assigned -> decode done (CWD transfer into its memory + fetch+algorithm+release)
    "wdo_per_window",       # decoder -> Pauli frame, link WDO
    "frame_commit",         # Pauli frame accepted -> committed
    "last_round_to_frame",  # last round of the window arrives -> its correction is in the frame
    "reaction_first_round", # first round of the window arrives -> correction in the frame
)

# The points stacked in latency_stack.png: the first-round-to-frame path, hop by hop.
STACKED_POINTS = ("buffer_fill", "dep_block", "cwd_per_window", "fetch", "algorithm", "release",
                  "wdo_per_window", "frame_commit")


@dataclass(frozen=True)
class ShotMeasurement:
    round_period_us: float
    algorithm_latency_us: float
    seed: int
    windows: int
    logical_failure: bool
    means: dict            # point -> mean us over windows of this shot
    maxes: dict            # point -> max us
    throughput_windows_per_us: float
    throughput_rounds_per_us: float
    decoder_utilization: float
    max_queued_windows: int
    sim_wall_seconds: float


def us(ticks: int) -> float:
    return ticks / TICKS_PER_US


def load_config(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


# ---- one run ---------------------------------------------------------------

def memory_circuit(config: dict) -> stim.Circuit:
    """The real data: Stim's rotated memory with all four noise channels at p."""
    p = config["noise_probability"]
    return stim.Circuit.generated(
        config["code_task"], rounds=config["rounds_per_shot"], distance=config["distance"],
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def weak_decoder(config: dict, algorithm_latency_us) -> DecoderEngine:
    """PyMatching inside the decoder engine: fetch cycles before it, release
    cycles after it, at the engine clock. The algorithm charges either its real
    wall clock (MEASURED) or the stated ASIC latency."""
    if algorithm_latency_us == MEASURED:
        algorithm = PyMatchingDecoder(latency_model=None)
    else:
        algorithm = PyMatchingDecoder(PresetLatencyDecoder(algorithm_latency_us))
    engine = config["decoder"]["engine"]
    fetch = DecoderStage("fetch", cycles_per_round=engine["fetch_cycles_per_round"])
    release = DecoderStage("release", cycles_per_job=engine["release_cycles_per_job"])
    timing = DecoderTiming(before=(fetch,), after=(release,), frequency_mhz=engine["frequency_mhz"])
    return DecoderEngine(algorithm, timing)


def link_cards(config: dict):
    """The named link profile plus the priced controller-to-Buffer-0 hop."""
    profile = getattr(link_profiles, config["links_profile"])()
    c2b = config["controller_to_buffer"]
    return with_controller_to_buffer_edge(
        profile, latency_us=c2b["latency_us"], aggregate_bits_per_us=c2b["aggregate_bits_per_us"],
        source="experiments/baseline_closed_loop.yaml controller_to_buffer")


def decoder_memory(config: dict):
    """Per-unit input memory in rounds; None is unbounded (the baseline)."""
    memory_rounds = config["decoder_memory_rounds"]
    if memory_rounds is None:
        return None
    return DecoderMemoryConfig({"default": memory_rounds})


def build_run(config: dict, *, round_period_us: float, algorithm_latency_us, seed: int):
    operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                          circuit=memory_circuit(config))
    decoder_engine = weak_decoder(config, algorithm_latency_us)
    controller = config["controller"]
    timing = TimingConfig(round_us=round_period_us,
                          t_binary_availability_us=controller["t_binary_availability_us"],
                          t_pack_us=controller["t_pack_us"])
    spec = RunSpec(
        ops=[operation], d=config["distance"],
        rounds_policy=FixedRounds(config["rounds_per_shot"]),
        device=StimDevice(), decoder=decoder_engine, num_units=config["decoder"]["units"],
        timing=timing, links=link_cards(config), decoder_memory=decoder_memory(config),
        pauli_frame=PauliFrameConfig(commit_us=config["pauli_frame"]["commit_us"]),
        seed=seed)
    return spec, decoder_engine


# ---- measuring one shot ----------------------------------------------------

def link_delay_by_window(transfers: list) -> dict:
    """(path, window_id) -> total ticks from send to delivery over that path."""
    delay = {}
    for row in transfers:
        key = (row["path"], row["attribution"]["window_id"])
        delay[key] = delay.get(key, 0) + row["delivery_ticks"] - row["send_ticks"]
    return delay


def c2b_delays_us(transfers: list) -> list:
    """Every round's controller-to-Buffer-0 delay, in microseconds."""
    return [us(row["delivery_ticks"] - row["send_ticks"])
            for row in transfers if row["path"] == "c2b"]


def window_points_us(window, frame_record, stage_us: dict, link_delay: dict) -> dict:
    """The per-window latency points, in microseconds, for one decoded window."""
    window_id = window.key[1]
    return {
        "buffer_fill": us(window.t_data_complete - window.t_first_round),
        "dep_block": us(window.t_queued - window.t_data_complete),
        "queue_wait": us(window.t_dispatch - window.t_queued),
        "cwd_per_window": us(link_delay.get(("cwd", window_id), 0)),
        "fetch": stage_us["fetch"],
        "algorithm": stage_us["algorithm"],
        "release": stage_us["release"],
        "service": us(window.t_done - window.t_dispatch),
        "wdo_per_window": us(link_delay.get(("wdo", window_id), 0)),
        "frame_commit": us(frame_record.committed_ticks - frame_record.accepted_ticks),
        "last_round_to_frame": us(frame_record.committed_ticks - window.t_data_complete),
        "reaction_first_round": us(frame_record.committed_ticks - window.t_first_round),
    }


def collect_samples(completed, decoder_engine) -> dict:
    """point -> list of microsecond samples over this shot's decoded windows."""
    transfers = completed.result.link_traffic["transfers"]
    link_delay = link_delay_by_window(transfers)
    frame_by_window = {record.window_key[1]: record
                       for record in completed.pauli_frame.snapshot().records}
    samples = {point: [] for point in POINTS}
    samples["c2b_per_round"] = c2b_delays_us(transfers)
    for (op_id, window_id), window in sorted(completed.window_manager.windows.items()):
        frame_record = frame_by_window.get(window_id)
        decoded = frame_record is not None and window.t_done is not None
        if not decoded:
            continue
        stage_us = {record.stage: us(record.end_ticks - record.start_ticks)
                    for record in decoder_engine.stage_records_for(op_id, window_id)}
        for point, value in window_points_us(window, frame_record, stage_us, link_delay).items():
            samples[point].append(value)
    return samples


def measure_shot(config: dict, *, round_period_us: float, algorithm_latency_us,
                 seed: int) -> ShotMeasurement:
    spec, decoder_engine = build_run(config, round_period_us=round_period_us,
                                     algorithm_latency_us=algorithm_latency_us, seed=seed)
    wall_start = time.perf_counter()
    completed = spec.build()
    wall_seconds = time.perf_counter() - wall_start
    if completed.result.terminal_status != "complete":
        raise RuntimeError(f"run did not complete: {completed.result.terminal_status}")

    samples = collect_samples(completed, decoder_engine)
    decoded_windows = len(samples["service"])
    windows = completed.window_manager.windows.values()
    first_round_tick = min(w.t_first_round for w in windows if w.t_first_round is not None)
    last_commit_tick = max(r.committed_ticks for r in completed.pauli_frame.snapshot().records)
    span_us = us(last_commit_tick - first_round_tick)
    operation_result = completed.result.operation_results[0]
    queue_depths = [depth for _, depth in completed.decoder_manager.queue_log]
    return ShotMeasurement(
        round_period_us=round_period_us, algorithm_latency_us=algorithm_latency_us,
        seed=seed, windows=decoded_windows,
        logical_failure=operation_result.logical_observables != operation_result.observable_truth,
        means={point: (statistics.fmean(values) if values else 0.0)
               for point, values in samples.items()},
        maxes={point: (max(values) if values else 0.0) for point, values in samples.items()},
        throughput_windows_per_us=decoded_windows / span_us,
        throughput_rounds_per_us=config["rounds_per_shot"] / span_us,
        decoder_utilization=sum(samples["service"]) / span_us,
        max_queued_windows=max(queue_depths, default=0),
        sim_wall_seconds=wall_seconds)


# ---- the sweep -------------------------------------------------------------

def run_sweep(config: dict) -> list:
    measurements = []
    for algorithm_latency_us in config["algorithm_latency_us"]:
        for round_period_us in config["round_period_us"]:
            for seed in config["seeds"]:
                measurements.append(measure_shot(
                    config, round_period_us=round_period_us,
                    algorithm_latency_us=algorithm_latency_us, seed=seed))
            print(f"algorithm {algorithm_latency_us} us, round period {round_period_us} us: done",
                  file=sys.stderr)
    return measurements


def sweep_point_of(measurement: ShotMeasurement) -> tuple:
    return (measurement.algorithm_latency_us, measurement.round_period_us)


def summarize_point(group: list) -> dict:
    """One sweep point: means over seeds of the per-shot means, maxes of maxes."""
    algorithm_latency_us, round_period_us = sweep_point_of(group[0])
    row = {"algorithm_latency_us": algorithm_latency_us,
           "round_period_us": round_period_us,
           "shots": len(group),
           "windows_per_shot": statistics.fmean(m.windows for m in group),
           "logical_error_rate": sum(m.logical_failure for m in group) / len(group),
           "throughput_windows_per_us": statistics.fmean(m.throughput_windows_per_us for m in group),
           "throughput_rounds_per_us": statistics.fmean(m.throughput_rounds_per_us for m in group),
           "decoder_utilization": statistics.fmean(m.decoder_utilization for m in group),
           "max_queued_windows": max(m.max_queued_windows for m in group),
           "sim_wall_seconds_per_shot": statistics.fmean(m.sim_wall_seconds for m in group)}
    for point in POINTS:
        row[f"{point}_mean_us"] = statistics.fmean(m.means[point] for m in group)
        row[f"{point}_max_us"] = max(m.maxes[point] for m in group)
    return row


def summarize(measurements: list) -> list:
    """One row per sweep point, in a stable order."""
    sweep_points = sorted({sweep_point_of(m) for m in measurements}, key=str)
    rows = []
    for sweep_point in sweep_points:
        group = [m for m in measurements if sweep_point_of(m) == sweep_point]
        rows.append(summarize_point(group))
    return rows


# ---- the report ------------------------------------------------------------

def algorithm_label(algorithm_latency_us) -> str:
    if isinstance(algorithm_latency_us, str):
        return algorithm_latency_us
    return f"{algorithm_latency_us:g} us"


def write_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_lines(rows: list) -> list:
    head = ["algo us", "round us", "LER", "win/us", "rounds/us", "util", "max q"] + list(POINTS)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        algorithm = row["algorithm_latency_us"]
        cells = [algorithm if isinstance(algorithm, str) else f"{algorithm:g}",
                 f"{row['round_period_us']:g}",
                 f"{row['logical_error_rate']:.2f}",
                 f"{row['throughput_windows_per_us']:.4f}",
                 f"{row['throughput_rounds_per_us']:.3f}",
                 f"{row['decoder_utilization']:.3f}",
                 f"{row['max_queued_windows']}"]
        cells += [f"{row[f'{point}_mean_us']:.3f}" for point in POINTS]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def host_cpu_name() -> str:
    import platform
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def serial_chain_notes(rows: list, commit_rounds: int) -> list:
    """Per algorithm at the fastest input: the measured serial chain per window
    and the round period at which the loop stops keeping up."""
    fastest_period = min(row["round_period_us"] for row in rows)
    notes = []
    for row in rows:
        if row["round_period_us"] != fastest_period:
            continue
        chain_us = 1 / row["throughput_windows_per_us"]
        notes.append(f"{algorithm_label(row['algorithm_latency_us'])}: {chain_us:.2f} us per window, "
                     f"{commit_rounds / chain_us:.2f} rounds/us, knee near a "
                     f"{chain_us / commit_rounds:.2f} us round period")
    return notes


def write_report(config: dict, rows: list, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, report_dir / "sweep.csv")
    lines = ["# Baseline closed loop, per-point latency and throughput", "",
             "Plots: latency_stack.png (per-point stack vs input round period), throughput.png "
             "(decoded vs input rounds/us).", "",
             f"Circuit: {config['code_task']} d={config['distance']}, "
             f"{config['rounds_per_shot']} rounds per shot, p={config['noise_probability']}, "
             f"{len(config['seeds'])} shots per point. All latencies simulated, in microseconds, "
             "mean over decoded windows (max in the CSV).", ""]
    lines += table_lines(rows)
    chains = serial_chain_notes(rows, commit_rounds=config["distance"])
    wall_clocks = ", ".join(f"{row['sim_wall_seconds_per_shot']:.2f}s" for row in rows[:6])
    lines += [
        "",
        f"algorithm = 'measured' rows charge the wall clock of each real PyMatching call "
        f"(software decoder on this host: {host_cpu_name()}, one thread, graph cached); numeric rows charge "
        "the stated modeled latency (an ASIC card).",
        "",
        "Reading the table. Windows are sliding (commit d, buffer d) and serial: window k+1 "
        "starts only after window k's boundary arrives, so the loop's capacity is one window "
        "per serial chain = unit assigned, CWD transfer into its memory, decoder service, boundary "
        "handoff over DD at decode done (the WDO delivery and frame commit run downstream, off the "
        "chain). Measured chain at the fastest input, per algorithm: " + "; ".join(chains) + ". "
        "Faster input only grows dep_block (the wait for the previous window). A unit is held from "
        "assignment through its input transfer to the end of its decode, so utilization counts the "
        "CWD transfer; the decode itself is the fetch+algorithm+release columns. With these link "
        "cards the ASIC rows are link-bound (CWD 2 us + DD 0.5 us per window), not decoder-bound; "
        "the measured software row is decoder-bound.",
        "",
        f"Simulator wall clock per shot (host CPU, not a modeled latency): {wall_clocks} ...",
        "",
        "Anchor comparison against published numbers: anchor.md (experiments/baseline_anchor.py).",
    ]
    (report_dir / "sweep.md").write_text("\n".join(lines) + "\n")


# ---- the plots -------------------------------------------------------------

def rows_of_algorithm(rows: list, algorithm) -> list:
    """This algorithm's rows, fastest input last."""
    return sorted((row for row in rows if row["algorithm_latency_us"] == algorithm),
                  key=lambda row: row["round_period_us"])


def latency_stack_plot(rows: list, algorithms: list, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(algorithms), figsize=(4.2 * len(algorithms), 3.8), sharey=True)
    axes = list(axes) if len(algorithms) > 1 else [axes]
    for axis, algorithm in zip(axes, algorithms):
        group = rows_of_algorithm(rows, algorithm)
        x_labels = [str(row["round_period_us"]) for row in group]
        bottom = [0.0] * len(group)
        for point in STACKED_POINTS:
            values = [row[f"{point}_mean_us"] for row in group]
            axis.bar(x_labels, values, bottom=bottom, label=point)
            bottom = [below + value for below, value in zip(bottom, values)]
        axis.set_title(f"algorithm {algorithm_label(algorithm)}")
        axis.set_xlabel("input round period (us)")
        axis.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("first round -> frame, mean us per window")
    axes[-1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def throughput_plot(rows: list, algorithms: list, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5, 3.6))
    for algorithm in algorithms:
        group = rows_of_algorithm(rows, algorithm)
        input_rounds_per_us = [1 / row["round_period_us"] for row in group]
        decoded_rounds_per_us = [row["throughput_rounds_per_us"] for row in group]
        axis.plot(input_rounds_per_us, decoded_rounds_per_us, "o-",
                  label=f"algorithm {algorithm_label(algorithm)}")
    fastest_input = max(1 / row["round_period_us"] for row in rows)
    axis.plot([0, fastest_input], [0, fastest_input], "k--", lw=0.8, label="keeps up (out = in)")
    axis.set_xlabel("input rounds/us")
    axis.set_ylabel("decoded rounds/us")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def plots(rows: list, report_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    algorithms = sorted({row["algorithm_latency_us"] for row in rows}, key=str)
    latency_stack_plot(rows, algorithms, report_dir / "latency_stack.png")
    throughput_plot(rows, algorithms, report_dir / "throughput.png")


def main(argv) -> None:
    config_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)
    rows = summarize(run_sweep(config))
    report_dir = Path(config["report_dir"])
    write_report(config, rows, report_dir)
    plots(rows, report_dir)
    print((report_dir / "sweep.md").read_text())


if __name__ == "__main__":
    main(sys.argv)
