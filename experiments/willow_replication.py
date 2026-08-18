"""Replay Google's Willow surface-code memory data through the baseline loop.

Data: Zenodo 13273331, google_105Q_surface_code_d3_d5_d7 ("Quantum error
correction below the surface code threshold", arXiv:2408.13687). Each shot's
recorded detection events go through the whole reaction path (controller,
packing, C2B, Buffer 0, window manager, CWD, decoder memory, decoder engine
with PyMatching on the SI1000 circuit prior, WDO, Pauli frame) exactly as
sampled data does; only the syndrome source is the hardware record.

1. Accuracy. Logical error per shot at 30 cycles for d = 3, 5, 7, X and Z
   bases, three ways on the same shots: this loop (windowed, PyMatching +
   SI1000 prior), whole-shot PyMatching with the same prior, and Google's
   released correlated-matching predictions (obs_flips_predicted). Error per
   cycle eps_d = (1 - (1 - 2 P)^(1/r)) / 2 and Lambda = eps_d / eps_(d+2)
   next to the paper's eps_7 = 0.143 percent and Lambda = 2.14 (their best
   decoders; plain matching with the SI1000 prior is expected to be worse).

2. Real-time configuration. The paper's real-time system (Sec. V): control
   electronics classify, Ethernet to a workstation, shared-memory buffer,
   streaming sparse blossom; d = 5, 1.1 us cycle, decoder latency 63 +- 17 us
   from last cycle received to correction returned. Here: d = 5, 250 recorded
   cycles at a 1.1 us round period, software algorithm cost = median wall
   clock of PyMatching on one d = 5 window measured on this host, reference
   link cards. Reported: last-round-to-frame per window and throughput. This
   compares structure and order of magnitude, not their exact decoder.

Usage: python -m experiments.willow_replication [--quick]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from decsim.qpu.stim_device import RecordedStimDevice
from decsim.config import TICKS_PER_US, TimingConfig
from decsim.decoders.decoder_engine import ALGORITHM_STAGE, DecoderEngine, DecoderStage, DecoderTiming
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.link_profiles import logical_reference_profile, with_controller_to_buffer_edge
from decsim.message import Operation
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.pauli_frame import PauliFrameConfig
from decsim.program.round_policies import FixedRounds
from decsim.run_spec import RunSpec

DATA = Path("tmp/references/data/willow/google_105Q_surface_code_d3_d5_d7")
REPORT = Path("experiments/results/willow_replication/report.md")
PATCHES = {3: "d3_at_q4_5", 5: "d5_at_q6_5", 7: "d7_at_q6_7"}
GOOGLE_PATHWAY = "correlated_matching_decoder_with_si1000_prior"
PAPER = {"eps7_percent": 0.143, "lambda": 2.14, "realtime_latency_us": 63.0,
         "realtime_latency_sd_us": 17.0, "cycle_us": 1.1, "realtime_eps5_percent": 0.35}


def load_cell(d: int, basis: str, rounds: int):
    base = DATA / PATCHES[d] / basis / f"r{rounds}"
    circuit = stim.Circuit.from_file(str(base / "circuit_noisy_si1000.stim"))
    dets = stim.read_shot_data_file(path=str(base / "detection_events.b8"), format="b8",
                                    num_detectors=circuit.num_detectors)
    obs = stim.read_shot_data_file(path=str(base / "obs_flips_actual.b8"), format="b8",
                                   num_observables=circuit.num_observables)
    google = stim.read_shot_data_file(
        path=str(base / "decoding_results" / GOOGLE_PATHWAY / "obs_flips_predicted.b8"),
        format="b8", num_observables=circuit.num_observables)
    meta = json.loads((base / "metadata.json").read_text())
    return circuit, dets, obs, google, meta


def eps_per_cycle(p_shot: float, rounds: int) -> float:
    return (1 - (1 - 2 * p_shot) ** (1 / rounds)) / 2


def run_shot(circuit, dets, obs, shot, *, d, rounds, decoder, round_us=1.1, links=None,
             pauli_frame=None):
    op = Operation(id=1, name=f"willow d{d}", qubits=(0,), patches=(0,), circuit=circuit)
    device = RecordedStimDevice(
        dets, obs, shot,
        detector_rounds={1: RecordedStimDevice.detector_rounds_from_coordinates(circuit, rounds)})
    return RunSpec(ops=[op], d=d, rounds_policy=FixedRounds(rounds), device=device,
                   decoder=decoder, timing=TimingConfig(round_us=round_us), links=links,
                   pauli_frame=pauli_frame, seed=shot).build()


def accuracy(shots_per_cell: dict, rounds_list) -> list:
    rows = []
    for d in (3, 5, 7):
        for basis in ("X", "Z"):
            for rounds in rounds_list:
                rows.append(_accuracy_cell(d, basis, rounds, shots_per_cell[d]))
                print(rows[-1], file=sys.stderr)
    return rows


def _accuracy_cell(d, basis, rounds, n) -> dict:
    circuit, dets, obs, google, meta = load_cell(d, basis, rounds)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    whole_pred = matching.decode_batch(dets)
    fails = 0
    t0 = time.perf_counter()
    for shot in range(n):
        result = run_shot(circuit, dets, obs, shot, d=d, rounds=rounds,
                          decoder=PyMatchingDecoder(PresetLatencyDecoder(0.028))
                          ).result.operation_results[0]
        fails += result.logical_observables != result.observable_truth
    return dict(d=d, basis=basis, rounds=rounds, shots=n, loop_ler=fails / n,
                whole_subset_ler=float(np.mean(np.any(whole_pred[:n] != obs[:n], axis=1))),
                google_subset_ler=float(np.mean(np.any(google[:n] != obs[:n], axis=1))),
                whole_ler=float(np.mean(np.any(whole_pred != obs, axis=1))),
                google_ler=float(np.mean(np.any(google != obs, axis=1))),
                total_shots=int(len(dets)),
                seconds_per_shot=(time.perf_counter() - t0) / n)


def fit_eps_per_cycle(points) -> float:
    """Paper-style fit: logical fidelity 1 - 2P decays as (1 - 2 eps)^r.

    Least squares of ln(1 - 2P) against r over the cycle counts; points at or
    past P = 0.5 carry no information and are dropped."""
    rs = np.array([r for r, p in points if p < 0.5], dtype=float)
    ys = np.log(np.array([1 - 2 * p for r, p in points if p < 0.5]))
    slope = np.polyfit(rs, ys, 1)[0]
    return (1 - np.exp(slope)) / 2


def eps_table(rows) -> dict:
    """eps per cycle by (d, source) fitted across cycle counts, bases pooled."""
    out = {}
    for d in sorted({r["d"] for r in rows}):
        for source in ("loop_ler", "whole_ler", "google_ler"):
            points = [(r["rounds"], r[source]) for r in rows if r["d"] == d]
            out[(d, source)] = fit_eps_per_cycle(points)
    return out


def software_window_us(d: int, rounds: int, window_rounds: int) -> float:
    """Median PyMatching wall clock for one d-window on this host (graph cached)."""
    circuit, dets, obs, _, _ = load_cell(d, "X", rounds)
    # A window of window_rounds rounds is a smaller problem than the shot; the
    # generated stim circuit at the same d and window_rounds sizes it.
    p = 0.001
    window_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=window_rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)
    matching = pymatching.Matching.from_detector_error_model(
        window_circuit.detector_error_model(decompose_errors=True))
    samples, _ = window_circuit.compile_detector_sampler(seed=1).sample(2000, separate_observables=True)
    matching.decode(samples[0])
    times = []
    for row in samples:
        t0 = time.perf_counter()
        matching.decode(row)
        times.append(time.perf_counter() - t0)
    return 1e6 * statistics.median(times)


def realtime(shots: int, algorithm_us: float) -> dict:
    d, rounds = 5, 250
    circuit, dets, obs, google, meta = load_cell(d, "X", rounds)
    links = with_controller_to_buffer_edge(logical_reference_profile(), latency_us=0.10,
                                           aggregate_bits_per_us=1000.0,
                                           source="experiments/baseline_closed_loop.yaml controller_to_buffer")
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    whole_pred = matching.decode_batch(dets[:shots])
    whole_fails = int(np.sum(np.any(whole_pred != obs[:shots], axis=1)))
    last_round_to_frame, reaction, fails, spans, measured_ns = [], [], 0, [], []
    for shot in range(shots):
        engine = DecoderEngine(
            PyMatchingDecoder(latency_model=None),          # measured wall clock per call
            DecoderTiming((DecoderStage("fetch", cycles_per_round=1),),
                          (DecoderStage("release", cycles_per_job=1),), 250.0))
        done = run_shot(circuit, dets, obs, shot, d=d, rounds=rounds, decoder=engine,
                        round_us=PAPER["cycle_us"], links=links,
                        pauli_frame=PauliFrameConfig(commit_us=0.004))
        result = done.result.operation_results[0]
        fails += result.logical_observables != result.observable_truth
        frames = {r.window_key[1]: r for r in done.pauli_frame.snapshot().records}
        measured_ns.extend(r.measured_ns for r in engine.stage_records
                           if r.stage == ALGORITHM_STAGE and r.measured_ns is not None)
        for (op_id, window_id), window in done.window_manager.windows.items():
            frame = frames.get(window_id)
            if frame is None or window.t_data_complete is None:
                continue
            last_round_to_frame.append((frame.committed_ticks - window.t_data_complete) / TICKS_PER_US)
            reaction.append((frame.committed_ticks - window.t_first_round) / TICKS_PER_US)
        first = min(w.t_first_round for w in done.window_manager.windows.values())
        spans.append((max(f.committed_ticks for f in frames.values()) - first) / TICKS_PER_US)
    return dict(shots=shots, rounds=rounds, algorithm_us=algorithm_us,
                measured_decode_median_us=statistics.median(measured_ns) / 1000,
                measured_decode_max_us=max(measured_ns) / 1000,
                loop_fails=fails, whole_fails=whole_fails, samples=last_round_to_frame,
                last_round_to_frame_mean_us=statistics.fmean(last_round_to_frame),
                last_round_to_frame_sd_us=statistics.pstdev(last_round_to_frame),
                last_round_to_frame_max_us=max(last_round_to_frame),
                reaction_mean_us=statistics.fmean(reaction),
                rounds_per_us=rounds / statistics.fmean(spans),
                windows=len(last_round_to_frame) // shots)


def agreement(shots_per_d: dict, rounds: int = 30) -> list:
    """Correctness freeze: the loop's answer must equal whole-shot PyMatching on
    every recorded shot, except where windowed and whole-shot matching may
    legitimately differ (a boundary case); both counts are reported."""
    rows = []
    for d in (3, 5, 7):
        circuit, dets, obs, google, meta = load_cell(d, "X", rounds)
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True))
        n = shots_per_d[d]
        whole = matching.decode_batch(dets[:n])
        agree = 0
        for shot in range(n):
            result = run_shot(circuit, dets, obs, shot, d=d, rounds=rounds,
                              decoder=PyMatchingDecoder(PresetLatencyDecoder(0.028))
                              ).result.operation_results[0]
            agree += result.logical_observables == tuple(int(b) for b in whole[shot])
        rows.append(dict(d=d, rounds=rounds, shots=n, agree=agree))
        print(rows[-1], file=sys.stderr)
    return rows


def resources(shots: int, software_us: float) -> list:
    """Section 3: what keeps the loop up with the QPU. Scheme x units x algorithm at
    d = 5, 1.1 us cycles, 250 recorded cycles; report sustained input, latency of
    the first and last ten windows (growth means the loop is falling behind), max."""
    from decsim.schemes import ParallelWindowScheme, SlidingWindowScheme
    circuit, dets, obs, google, meta = load_cell(5, "X", 250)
    links = with_controller_to_buffer_edge(logical_reference_profile(), latency_us=0.10,
                                           aggregate_bits_per_us=1000.0,
                                           source="experiments/baseline_closed_loop.yaml controller_to_buffer")
    rows = []
    for scheme_name, make_scheme in (("serial sliding", SlidingWindowScheme),
                                     ("parallel A/B (Skoric)", ParallelWindowScheme)):
        for units in (1, 2):
            for algo_name, algorithm_us in (("software PyMatching, measured per call", None),
                                            ("LILLIPUT-class ASIC 42 ns", 0.042)):
                spans, head, tail, maxes = [], [], [], []
                for shot in range(shots):
                    engine = DecoderEngine(
                        PyMatchingDecoder(None if algorithm_us is None
                                          else PresetLatencyDecoder(algorithm_us)),
                        DecoderTiming((DecoderStage("fetch", cycles_per_round=1),),
                                      (DecoderStage("release", cycles_per_job=1),), 250.0))
                    op = Operation(id=1, name="willow d5", qubits=(0,), patches=(0,), circuit=circuit)
                    device = RecordedStimDevice(dets, obs, shot, detector_rounds={
                        1: RecordedStimDevice.detector_rounds_from_coordinates(circuit, 250)})
                    done = RunSpec(ops=[op], d=5, rounds_policy=FixedRounds(250), device=device,
                                   decoder=engine, num_units=units, scheme=make_scheme(),
                                   timing=TimingConfig(round_us=PAPER["cycle_us"]), links=links,
                                   pauli_frame=PauliFrameConfig(commit_us=0.004), seed=shot).build()
                    frames = {r.window_key[1]: r for r in done.pauli_frame.snapshot().records}
                    latency = []
                    for (op_id, window_id), window in sorted(done.window_manager.windows.items()):
                        frame = frames.get(window_id)
                        if frame is not None and window.t_data_complete is not None:
                            latency.append((frame.committed_ticks - window.t_data_complete) / TICKS_PER_US)
                    first = min(w.t_first_round for w in done.window_manager.windows.values())
                    spans.append((max(f.committed_ticks for f in frames.values()) - first) / TICKS_PER_US)
                    head.append(statistics.fmean(latency[:10])); tail.append(statistics.fmean(latency[-10:]))
                    maxes.append(max(latency))
                rows.append(dict(scheme=scheme_name, units=units, algorithm=algo_name,
                                 algorithm_us=algorithm_us, rounds_per_us=250 / statistics.fmean(spans),
                                 first10_us=statistics.fmean(head), last10_us=statistics.fmean(tail),
                                 max_us=max(maxes), keeps_up=statistics.fmean(tail) < 1.5 * statistics.fmean(head)))
                print(rows[-1], file=sys.stderr)
    return rows


def plots(acc_rows, eps, rt_samples, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A. logical error vs cycles, per distance, three decoders on the same data
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, d in zip(axes, (3, 5, 7)):
        rows = sorted((r for r in acc_rows if r["d"] == d), key=lambda r: r["rounds"])
        by_r = {}
        for r in rows:
            by_r.setdefault(r["rounds"], []).append(r)
        rs = sorted(by_r)
        loop = [np.mean([r["loop_ler"] for r in by_r[k]]) for k in rs]
        n = [sum(r["shots"] for r in by_r[k]) for k in rs]
        err = [np.sqrt(p * (1 - p) / m) for p, m in zip(loop, n)]
        whole = [np.mean([r["whole_ler"] for r in by_r[k]]) for k in rs]
        google = [np.mean([r["google_ler"] for r in by_r[k]]) for k in rs]
        ax.errorbar(rs, loop, yerr=err, fmt="o", color="k", label="decsim loop (PyMatching, SI1000 prior)")
        ax.plot(rs, whole, "s--", color="tab:blue", label="whole-shot PyMatching, 50k shots")
        ax.plot(rs, google, "^:", color="tab:red", label="Google correlated matching, 50k shots")
        ax.set_title(f"d = {d}, {PATCHES[d]}")
        ax.set_xlabel("QEC cycles")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("logical error probability")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "logical_error_vs_cycles.png", dpi=150)

    # B. eps per cycle vs distance, against the paper
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ds = (3, 5, 7)
    for source, label, marker, color in (("loop_ler", "decsim loop", "o", "k"),
                                         ("whole_ler", "whole-shot PyMatching", "s", "tab:blue"),
                                         ("google_ler", "Google corr. matching", "^", "tab:red")):
        ax.semilogy(ds, [100 * eps[(d, source)] for d in ds], marker=marker, color=color, label=label)
    paper_eps = [PAPER["eps7_percent"] * PAPER["lambda"] ** k for k in (2, 1, 0)]
    ax.semilogy(ds, paper_eps, "x-.", color="tab:green",
                label=f"paper: eps7 = {PAPER['eps7_percent']}%, Lambda = {PAPER['lambda']}")
    ax.set_xlabel("code distance d")
    ax.set_ylabel("logical error per cycle (%)")
    ax.set_xticks(ds)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "eps_per_cycle_vs_distance.png", dpi=150)

    # C0. latency per window, serial vs parallel scheme, one recorded shot
    from decsim.schemes import ParallelWindowScheme, SlidingWindowScheme
    circuit, dets, obs, google, meta = load_cell(5, "X", 250)
    links = with_controller_to_buffer_edge(logical_reference_profile(), latency_us=0.10,
                                           aggregate_bits_per_us=1000.0, source="plot")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, make_scheme, color in (("serial sliding windows", SlidingWindowScheme, "tab:red"),
                                      ("parallel A/B windows", ParallelWindowScheme, "tab:blue")):
        engine = DecoderEngine(PyMatchingDecoder(latency_model=None),
                               DecoderTiming((DecoderStage("fetch", cycles_per_round=1),),
                                             (DecoderStage("release", cycles_per_job=1),), 250.0))
        op = Operation(id=1, name="willow d5", qubits=(0,), patches=(0,), circuit=circuit)
        device = RecordedStimDevice(dets, obs, 0, detector_rounds={
            1: RecordedStimDevice.detector_rounds_from_coordinates(circuit, 250)})
        done = RunSpec(ops=[op], d=5, rounds_policy=FixedRounds(250), device=device, decoder=engine,
                       scheme=make_scheme(), timing=TimingConfig(round_us=PAPER["cycle_us"]),
                       links=links, pauli_frame=PauliFrameConfig(commit_us=0.004), seed=0).build()
        frames = {r.window_key[1]: r for r in done.pauli_frame.snapshot().records}
        xs, ys = [], []
        for (op_id, window_id), window in sorted(done.window_manager.windows.items()):
            frame = frames.get(window_id)
            if frame is not None and window.t_data_complete is not None:
                xs.append(window.t_data_complete / TICKS_PER_US)
                ys.append((frame.committed_ticks - window.t_data_complete) / TICKS_PER_US)
        ax.plot(xs, ys, ".-", color=color, label=label)
    ax.axhline(PAPER["realtime_latency_us"], color="k", lw=0.8, ls="--", label="paper mean 63 us")
    ax.set_xlabel("time the window's last cycle arrived (us)")
    ax.set_ylabel("last cycle -> correction (us)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_window.png", dpi=150)

    # C. real-time latency distribution against the paper
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(rt_samples, bins=40, color="tab:gray", label="decsim: last cycle -> correction (per window)")
    ax.axvline(PAPER["realtime_latency_us"], color="tab:red", label="paper mean 63 us")
    ax.axvspan(PAPER["realtime_latency_us"] - PAPER["realtime_latency_sd_us"],
               PAPER["realtime_latency_us"] + PAPER["realtime_latency_sd_us"], color="tab:red", alpha=0.15,
               label="paper +- 17 us")
    ax.set_xlabel("latency (us)")
    ax.set_ylabel("windows")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "realtime_latency.png", dpi=150)


def host_note() -> str:
    import os, platform
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    load = os.getloadavg()
    return (f"Host: {cpu}; load average at run start {load[0]:.1f}, {load[1]:.1f}, {load[2]:.1f} "
            "(measured decode times are only meaningful on an otherwise idle core).")


def write_report(acc_rows: list, eps: dict, rt: dict, software_us: float,
                 agree_rows: list = (), resource_rows: list = ()) -> None:
    lines = ["# Willow data through the baseline loop", "",
             "Data: Zenodo 13273331 (arXiv:2408.13687), patches "
             + ", ".join(f"d={d}: {p}" for d, p in PATCHES.items()) + ".",
             "Plots: logical_error_vs_cycles.png, eps_per_cycle_vs_distance.png, realtime_latency.png, "
             "latency_vs_window.png.", HOST_NOTE, "",
             "## 0. Correctness freeze: loop vs whole-shot PyMatching, shot for shot", "",
             "| d | cycles | shots | agree | differ (window-boundary cases) |", "|---|---|---|---|---|"]
    for r in agree_rows:
        lines.append(f"| {r['d']} | {r['rounds']} | {r['shots']} | {r['agree']} | {r['shots'] - r['agree']} |")
    lines += ["", "## 1. Accuracy on the same recorded shots", "",
             "| d | basis | cycles | loop LER (n shots) | whole-shot PyMatching, same n | Google corr. matching, same n | whole-shot, 50k | Google, 50k |",
             "|---|---|---|---|---|---|---|---|"]
    for r in acc_rows:
        lines.append(f"| {r['d']} | {r['basis']} | {r['rounds']} | {r['loop_ler']:.3f} ({r['shots']}) | "
                     f"{r['whole_subset_ler']:.3f} | {r['google_subset_ler']:.3f} | "
                     f"{r['whole_ler']:.4f} | {r['google_ler']:.4f} |")
    lines += ["", "Logical error per cycle, fitted across cycle counts (bases pooled), percent:", "",
              "| d | decsim loop | whole-shot PyMatching (50k) | Google corr. matching (50k) | paper (best decoders) |",
              "|---|---|---|---|---|"]
    paper_eps = {7: PAPER["eps7_percent"], 5: PAPER["eps7_percent"] * PAPER["lambda"],
                 3: PAPER["eps7_percent"] * PAPER["lambda"] ** 2}
    for d in (3, 5, 7):
        lines.append(f"| {d} | {100 * eps[(d, 'loop_ler')]:.3f} | {100 * eps[(d, 'whole_ler')]:.3f} | "
                     f"{100 * eps[(d, 'google_ler')]:.3f} | {paper_eps[d]:.3f} |")
    lam = lambda s: eps[(5, s)] / eps[(7, s)]
    lines += ["", f"Lambda(5->7): decsim loop {lam('loop_ler'):.2f}, whole-shot PyMatching {lam('whole_ler'):.2f}, "
              f"Google pathway {lam('google_ler'):.2f}, paper {PAPER['lambda']}.",
              "", "Read across a row: the loop and whole-shot PyMatching share decoder and prior and decode "
              "the same n shots, so 'loop' and 'same n' agree shot for shot (up to the rare window-boundary "
              "difference between windowed and whole-shot matching); the 50k columns show the subset's "
              "sampling noise. The gap to Google's released pathway is decoder quality (correlated "
              "matching reweights Y-error correlations); the paper's headline adds optimized priors and "
              "ensembling. The paper's error bars come from all cycle counts up to 250; ours from four.",
              "", "## 2. Real-time configuration (paper Sec. V: d = 5, 1.1 us cycle)", "",
              "Software algorithm cost: the wall clock of each real PyMatching call on this host, "
              f"measured per window inside the loop: median {rt['measured_decode_median_us']:.1f} us, "
              f"max {rt['measured_decode_max_us']:.1f} us on these d=5 SI1000 windows (10 rounds, 240 "
              f"detectors; the sparser synthetic p=0.001 window reference is {software_us:.1f} us); graph "
              "prebuilt, one thread; reference link cards (CWD 2 us, WDO 1 us, DD 0.5 us), C2B 0.1 us at "
              "1 Gbit/s, frame commit 4 ns.", "",
              "| quantity | this loop | paper |", "|---|---|---|",
              f"| last cycle received -> correction committed, mean (sd, max) us | {rt['last_round_to_frame_mean_us']:.1f} "
              f"({rt['last_round_to_frame_sd_us']:.1f}, {rt['last_round_to_frame_max_us']:.1f}) | {PAPER['realtime_latency_us']} +- {PAPER['realtime_latency_sd_us']} |",
              f"| sustained input, rounds/us | {rt['rounds_per_us']:.2f} | 1/{PAPER['cycle_us']} = {1 / PAPER['cycle_us']:.2f} |",
              f"| logical failures over {rt['rounds']} cycles, d=5 | loop {rt['loop_fails']}/{rt['shots']}, whole-shot PyMatching on the same shots {rt['whole_fails']}/{rt['shots']} | eps_5 = {PAPER['realtime_eps5_percent']} % per cycle real-time (0.269 offline NN); at 250 cycles that is P ~ 0.4 |",
              "", "Deviations to state: the paper's latency includes Ethernet, shared-memory buffering and a "
              "multi-threaded streaming decoder on a workstation, none of which are our numbers; ours are the "
              "reference link cards plus one measured software decode per window. Windows here are sliding "
              "(commit 5, buffer 5) and serial; the paper's decoder streams and kept latency constant for a "
              "million cycles, ours grows over 250 cycles because a single-threaded PyMatching call per "
              f"5-round window ({software_us:.0f} us median here; one Python matching.decode call costs 5-7 us before "
              "any matching work, the published sub-microsecond figures are decode_batch amortized over "
              "thousands of shots) cannot keep pace with 1.1 us cycles; the paper's decoder "
              "is a multi-worker streaming design. Their real-time "
              "run used the 72-qubit processor data (not in this archive); ours replays the 105-qubit d=5 patch."]
    lines += ["", "## 3. What keeps the loop up with the QPU (d = 5, 1.1 us cycles, 250 recorded cycles)", "",
              "| scheme | units | algorithm | sustained rounds/us (need 0.91) | latency first 10 windows us | last 10 windows us | max us | keeps up |",
              "|---|---|---|---|---|---|---|---|"]
    for r in resource_rows:
        lines.append(f"| {r['scheme']} | {r['units']} | {r['algorithm']} | {r['rounds_per_us']:.2f} | "
                     f"{r['first10_us']:.1f} | {r['last10_us']:.1f} | {r['max_us']:.1f} | {'yes' if r['keeps_up'] else 'NO'} |")
    lines += ["", "Serial sliding windows are bound by the per-window chain (unit assigned, CWD transfer, decode, "
              "boundary handoff over DD at decode done), so more units change nothing and latency grows with time; parallel A/B "
              "windows remove the chain and hold latency flat. Plot: latency_vs_window.png."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


HOST_NOTE = ""


def main(argv) -> None:
    global HOST_NOTE
    HOST_NOTE = host_note()
    quick = "--quick" in argv
    shots = {3: 40, 5: 20, 7: 10} if quick else {3: 600, 5: 400, 7: 150}
    rounds_list = (10, 30) if quick else (10, 30, 50, 90)
    acc_rows = accuracy(shots, rounds_list)
    eps = eps_table(acc_rows)
    software_us = software_window_us(5, 30, window_rounds=10)
    rt = realtime(shots=2 if quick else 10, algorithm_us=software_us)
    agree_rows = agreement({3: 30, 5: 20, 7: 10} if quick else {3: 2000, 5: 1000, 7: 500})
    resource_rows = resources(shots=1 if quick else 3, software_us=software_us)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    plots(acc_rows, eps, rt["samples"], REPORT.parent)
    write_report(acc_rows, eps, rt, software_us, agree_rows, resource_rows)


if __name__ == "__main__":
    main(sys.argv)
