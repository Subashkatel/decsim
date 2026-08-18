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

from decsim.adapters.stim_device import StimDevice
from decsim.config import TICKS_PER_US, TimingConfig
from decsim.decoder_engine import DecoderEngine, DecoderStage, DecoderTiming
from decsim.decoder_memory import DecoderMemoryConfig
from decsim.decoders import PresetLatencyDecoder
from decsim import link_profiles
from decsim.link_profiles import with_controller_to_buffer_edge
from decsim.message import Operation
from decsim.mwpm_decoder.decoder import PyMatchingDecoder
from decsim.pauli_frame import PauliFrameConfig
from decsim.rounds import FixedRounds
from decsim.run_spec import RunSpec

DEFAULT_CONFIG = Path(__file__).with_suffix(".yaml")

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
    "wdo_per_window",       # decoder -> orchestrator, link WDO
    "frame_commit",         # Pauli frame accepted -> committed
    "last_round_to_frame",  # last round of the window arrives -> its correction is in the frame
    "reaction_first_round", # first round of the window arrives -> correction in the frame
)


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


def load_config(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def build_run(config: dict, *, round_period_us: float, algorithm_latency_us: float,
              seed: int):
    p = config["noise_probability"]
    circuit = stim.Circuit.generated(
        config["code_task"], rounds=config["rounds_per_shot"],
        distance=config["distance"], after_clifford_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p,
        before_round_data_depolarization=p)
    operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                          circuit=circuit)
    engine_config = config["decoder"]["engine"]
    algorithm = (PyMatchingDecoder(latency_model=None) if algorithm_latency_us == "measured"
                 else PyMatchingDecoder(PresetLatencyDecoder(algorithm_latency_us)))
    decoder_engine = DecoderEngine(
        algorithm,
        DecoderTiming(
            before=(DecoderStage("fetch",
                                 cycles_per_round=engine_config["fetch_cycles_per_round"]),),
            after=(DecoderStage("release",
                                cycles_per_job=engine_config["release_cycles_per_job"]),),
            frequency_mhz=engine_config["frequency_mhz"]))
    c2b = config["controller_to_buffer"]
    links = with_controller_to_buffer_edge(
        getattr(link_profiles, config["links_profile"])(),
        latency_us=c2b["latency_us"],
        aggregate_bits_per_us=c2b["aggregate_bits_per_us"],
        source="experiments/baseline_closed_loop.yaml controller_to_buffer")
    memory_rounds = config["decoder_memory_rounds"]
    spec = RunSpec(
        ops=[operation], d=config["distance"],
        rounds_policy=FixedRounds(config["rounds_per_shot"]),
        device=StimDevice(), decoder=decoder_engine,
        num_units=config["decoder"]["units"],
        timing=TimingConfig(
            round_us=round_period_us,
            t_binary_availability_us=config["controller"]["t_binary_availability_us"],
            t_pack_us=config["controller"]["t_pack_us"]),
        links=links,
        decoder_memory=(None if memory_rounds is None
                        else DecoderMemoryConfig({"default": memory_rounds})),
        pauli_frame=PauliFrameConfig(commit_us=config["pauli_frame"]["commit_us"]),
        seed=seed)
    return spec, decoder_engine


def measure_shot(config: dict, *, round_period_us: float, algorithm_latency_us: float,
                 seed: int) -> ShotMeasurement:
    spec, decoder_engine = build_run(
        config, round_period_us=round_period_us,
        algorithm_latency_us=algorithm_latency_us, seed=seed)
    wall = time.perf_counter()
    completed = spec.build()
    wall = time.perf_counter() - wall
    if completed.result.terminal_status != "complete":
        raise RuntimeError(f"run did not complete: {completed.result.terminal_status}")

    us = lambda ticks: ticks / TICKS_PER_US
    windows = completed.window_manager.windows
    transfers = completed.result.link_traffic["transfers"]
    per_window_link = {}
    for row in transfers:
        key = (row["path"], row["attribution"]["window_id"])
        per_window_link.setdefault(key, 0)
        per_window_link[key] += row["delivery_ticks"] - row["send_ticks"]
    c2b_delays = [us(row["delivery_ticks"] - row["send_ticks"])
                  for row in transfers if row["path"] == "c2b"]
    frame_by_window = {record.window_key[1]: record
                       for record in completed.pauli_frame.snapshot().records}

    samples = {point: [] for point in POINTS}
    samples["c2b_per_round"] = c2b_delays
    for (op_id, window_id), window in sorted(windows.items()):
        frame = frame_by_window.get(window_id)
        if frame is None or window.t_done is None:
            continue
        stages = {r.stage: us(r.end_ticks - r.start_ticks)
                  for r in decoder_engine.stage_records_for(op_id, window_id)}
        samples["buffer_fill"].append(us(window.t_data_complete - window.t_first_round))
        samples["dep_block"].append(us(window.t_queued - window.t_data_complete))
        samples["queue_wait"].append(us(window.t_dispatch - window.t_queued))
        samples["cwd_per_window"].append(us(per_window_link.get(("cwd", window_id), 0)))
        samples["fetch"].append(stages["fetch"])
        samples["algorithm"].append(stages["algorithm"])
        samples["release"].append(stages["release"])
        samples["service"].append(us(window.t_done - window.t_dispatch))
        samples["wdo_per_window"].append(us(per_window_link.get(("wdo", window_id), 0)))
        samples["frame_commit"].append(us(frame.committed_ticks - frame.accepted_ticks))
        samples["last_round_to_frame"].append(us(frame.committed_ticks - window.t_data_complete))
        samples["reaction_first_round"].append(us(frame.committed_ticks - window.t_first_round))

    decoded = len(samples["service"])
    first_round = min(w.t_first_round for w in windows.values() if w.t_first_round is not None)
    last_commit = max(f.committed_ticks for f in frame_by_window.values())
    span_us = us(last_commit - first_round)
    result = completed.result.operation_results[0]
    return ShotMeasurement(
        round_period_us=round_period_us, algorithm_latency_us=algorithm_latency_us,
        seed=seed, windows=decoded,
        logical_failure=result.logical_observables != result.observable_truth,
        means={p: (statistics.fmean(v) if v else 0.0) for p, v in samples.items()},
        maxes={p: (max(v) if v else 0.0) for p, v in samples.items()},
        throughput_windows_per_us=decoded / span_us,
        throughput_rounds_per_us=config["rounds_per_shot"] / span_us,
        decoder_utilization=sum(samples["service"]) / span_us,
        max_queued_windows=max((n for _, n in completed.decoder_manager.queue_log), default=0),
        sim_wall_seconds=wall)


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


def summarize(measurements: list) -> list:
    """One row per sweep point: means over seeds of the per-shot means, maxes of maxes."""
    rows = []
    points = sorted({(m.algorithm_latency_us, m.round_period_us) for m in measurements}, key=str)
    for algorithm_latency_us, round_period_us in points:
        group = [m for m in measurements
                 if (m.algorithm_latency_us, m.round_period_us) == (algorithm_latency_us, round_period_us)]
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
        rows.append(row)
    return rows


def write_report(config: dict, rows: list, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "sweep.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Baseline closed loop, per-point latency and throughput", "",
             "Plots: latency_stack.png (per-point stack vs input round period), throughput.png "
             "(decoded vs input rounds/us).", "",
             f"Circuit: {config['code_task']} d={config['distance']}, "
             f"{config['rounds_per_shot']} rounds per shot, p={config['noise_probability']}, "
             f"{len(config['seeds'])} shots per point. All latencies simulated, in microseconds, "
             "mean over decoded windows (max in the CSV).", ""]
    head = ["algo us", "round us", "LER", "win/us", "rounds/us", "util", "max q"] + list(POINTS)
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for row in rows:
        algo = row['algorithm_latency_us']
        cells = [algo if isinstance(algo, str) else f"{algo:g}", f"{row['round_period_us']:g}",
                 f"{row['logical_error_rate']:.2f}",
                 f"{row['throughput_windows_per_us']:.4f}", f"{row['throughput_rounds_per_us']:.3f}",
                 f"{row['decoder_utilization']:.3f}", f"{row['max_queued_windows']}"]
        cells += [f"{row[f'{p}_mean_us']:.3f}" for p in POINTS]
        lines.append("| " + " | ".join(cells) + " |")
    import platform
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    lines += ["", f"algorithm = 'measured' rows charge the wall clock of each real PyMatching call "
              f"(software decoder on this host: {cpu}, one thread, graph cached); numeric rows charge "
              "the stated modeled latency (an ASIC card)."]
    commit_rounds = config["distance"]
    fastest_period = min(r["round_period_us"] for r in rows)
    chains = []
    for r in rows:
        if r["round_period_us"] == fastest_period:
            chain_us = 1 / r["throughput_windows_per_us"]
            algo = r["algorithm_latency_us"]
            chains.append(f"{algo if isinstance(algo, str) else f'{algo:g} us'}: {chain_us:.2f} us per window, "
                          f"{commit_rounds / chain_us:.2f} rounds/us, knee near a {chain_us / commit_rounds:.2f} us round period")
    lines += ["", "Reading the table. Windows are sliding (commit d, buffer d) and serial: window k+1 "
              "starts only after window k's boundary arrives, so the loop's capacity is one window "
              "per serial chain = unit assigned, CWD transfer into its memory, decoder service, boundary "
              "handoff over DD at decode done (the WDO delivery and frame commit run downstream, off the "
              "chain). Measured chain at the fastest input, per algorithm: " + "; ".join(chains) + ". "
              "Faster input only grows dep_block (the wait for the previous window). A unit is held from "
              "assignment through its input transfer to the end of its decode, so utilization counts the "
              "CWD transfer; the decode itself is the fetch+algorithm+release columns. With these link "
              "cards the ASIC rows are link-bound (CWD 2 us + DD 0.5 us per window), not decoder-bound; "
              "the measured software row is decoder-bound.",
              "", "Simulator wall clock per shot (host CPU, not a modeled latency): "
              + ", ".join(f"{r['sim_wall_seconds_per_shot']:.2f}s" for r in rows[:6]) + " ...",
              "", "Anchor comparison against published numbers: anchor.md (experiments/baseline_anchor.py)."]
    (report_dir / "sweep.md").write_text("\n".join(lines) + "\n")


def plots(rows: list, report_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = sorted({r["algorithm_latency_us"] for r in rows}, key=str)
    stack = ("buffer_fill", "dep_block", "cwd_per_window", "fetch", "algorithm", "release",
             "wdo_per_window", "frame_commit")
    fig, axes = plt.subplots(1, len(algorithms), figsize=(4.2 * len(algorithms), 3.8), sharey=True)
    for ax, algorithm in zip(axes if len(algorithms) > 1 else [axes], algorithms):
        group = sorted((r for r in rows if r["algorithm_latency_us"] == algorithm),
                       key=lambda r: r["round_period_us"])
        x = [str(r["round_period_us"]) for r in group]
        bottom = [0.0] * len(group)
        for point in stack:
            values = [r[f"{point}_mean_us"] for r in group]
            ax.bar(x, values, bottom=bottom, label=point)
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_title(f"algorithm {algorithm if isinstance(algorithm, str) else f'{algorithm:g} us'}")
        ax.set_xlabel("input round period (us)")
        ax.grid(alpha=0.3, axis="y")
    (axes[0] if len(algorithms) > 1 else axes).set_ylabel("first round -> frame, mean us per window")
    (axes[-1] if len(algorithms) > 1 else axes).legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(report_dir / "latency_stack.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for algorithm in algorithms:
        group = sorted((r for r in rows if r["algorithm_latency_us"] == algorithm),
                       key=lambda r: r["round_period_us"])
        ax.plot([1 / r["round_period_us"] for r in group],
                [r["throughput_rounds_per_us"] for r in group], "o-",
                label=f"algorithm {algorithm if isinstance(algorithm, str) else f'{algorithm:g} us'}")
    limit = max(1 / r["round_period_us"] for r in rows)
    ax.plot([0, limit], [0, limit], "k--", lw=0.8, label="keeps up (out = in)")
    ax.set_xlabel("input rounds/us")
    ax.set_ylabel("decoded rounds/us")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(report_dir / "throughput.png", dpi=150)


def main(argv) -> None:
    config_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)
    rows = summarize(run_sweep(config))
    report_dir = Path(config["report_dir"])
    write_report(config, rows, report_dir)
    plots(rows, report_dir)
    print(open(report_dir / "sweep.md").read())


if __name__ == "__main__":
    main(sys.argv)
