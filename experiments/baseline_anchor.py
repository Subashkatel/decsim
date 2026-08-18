"""Anchor the baseline loop against published numbers (Q-062 part f).

Two checks, both on real Stim data:

1. Software decoder time. PyMatching v2 publishes microseconds per shot for
   rotated memory_x at p = 0.001, d in {5, 7, 9, 13, 17}, measured with
   ``decode_batch`` on one M1 Max core (tmp/references/code/pymatching/
   benchmarks/surface_codes, arXiv:2303.15933). We repeat the published
   procedure on this host, then time the SAME decode as decsim's decoder sees
   it: one call per window through ``PyMatchingDecoder.decode`` on the window
   model built by the pipeline. The ratio host/M1 is the CPU difference; the
   ratio decsim/host is decsim's per-call overhead over the batch path.

2. Logical error rate. The windowed loop (sliding commit d, buffer d) must
   reproduce the whole-circuit PyMatching error rate on the same circuit
   within statistics; a windowed decoder may be slightly worse, never better.

Usage: python -m experiments.baseline_anchor
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from decsim.qpu.stim_device import StimDevice
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.message import Operation
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.program.round_policies import FixedRounds
from decsim.run_spec import RunSpec
from decsim.windows.windowing_schemes import NaiveOnlineScheme

BENCH = Path("tmp/references/code/pymatching/benchmarks/surface_codes/"
             "surface_code_rotated_memory_x_p_0.001_d_5_7_9_13_17_23_29_39_50_both_bases")
REPORT = Path("experiments/results/baseline_closed_loop/anchor.md")


def circuit_for(task, d, rounds, p):
    return stim.Circuit.generated(
        task, distance=d, rounds=rounds, after_clifford_depolarization=p,
        before_round_data_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p)


def published_us_per_shot() -> dict:
    with open(BENCH / "pymatching_v2.csv") as handle:
        return {int(row["d"]): float(row["microseconds"]) for row in csv.DictReader(handle)}


def host_us_per_shot(d: int, p: float, num_shots: int = 10000) -> float:
    """The published README procedure, verbatim in method."""
    circuit = circuit_for("surface_code:rotated_memory_x", d, d, p)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    shots, _ = circuit.compile_detector_sampler().sample(shots=num_shots, separate_observables=True)
    matching.decode_batch(shots[0:1, :])
    t0 = time.time()
    matching.decode_batch(shots)
    return 1e6 * (time.time() - t0) / num_shots


class _TimedDecoder(PyMatchingDecoder):
    """PyMatchingDecoder that records the wall clock of each decode() call."""

    def __init__(self, latency_model):
        super().__init__(latency_model)
        self.seconds = []

    def decode(self, job):
        super().decode(job)                 # first call builds and caches the graph
        t0 = time.perf_counter()
        result = super().decode(job)        # timed: graph cached, as in the README
        self.seconds.append(time.perf_counter() - t0)
        return result


def decsim_us_per_shot(d: int, p: float, shots: int) -> float:
    """One decode() per shot inside the pipeline: whole shot as one window."""
    circuit = circuit_for("surface_code:rotated_memory_x", d, d, p)
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    decoder = _TimedDecoder(PresetLatencyDecoder(1.0))
    for seed in range(shots):
        RunSpec(ops=[op], d=d, rounds_policy=FixedRounds(d), device=StimDevice(),
                decoder=decoder, scheme=NaiveOnlineScheme(), seed=seed).build()
    return 1e6 * statistics.median(decoder.seconds)


def logical_error_rates(d: int, rounds: int, p: float, pipeline_shots: int,
                        reference_shots: int = 100_000) -> tuple:
    circuit = circuit_for("surface_code:rotated_memory_z", d, rounds, p)
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    failures = 0
    for seed in range(pipeline_shots):
        result = RunSpec(ops=[op], d=d, rounds_policy=FixedRounds(rounds), device=StimDevice(),
                         decoder=PyMatchingDecoder(PresetLatencyDecoder(0.028)),
                         seed=seed).build().result.operation_results[0]
        failures += result.logical_observables != result.observable_truth
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(
        reference_shots, separate_observables=True)
    reference = float(np.mean(np.any(matching.decode_batch(dets) != obs, axis=1)))
    return failures / pipeline_shots, reference


def main(argv) -> None:
    published = published_us_per_shot()
    lines = ["# Anchor comparison against published numbers", "",
             "## 1. Software decoder time, microseconds per shot (memory_x, p = 0.001, rounds = d)", "",
             "| d | published (M1 Max, decode_batch) | this host (decode_batch, same method) | host/M1 | decsim decode() per window (median) | decsim/host |",
             "|---|---|---|---|---|---|"]
    for d in (5, 7, 9, 13, 17):
        host = host_us_per_shot(d, 0.001)
        inside = decsim_us_per_shot(d, 0.001, shots=6 if d >= 13 else 12)
        lines.append(f"| {d} | {published[d]:.3f} | {host:.3f} | {host / published[d]:.2f} | "
                     f"{inside:.1f} | {inside / host:.1f} |")
        print(lines[-1], file=sys.stderr)
    lines += ["",
              "host/M1 is the CPU-speed ratio and should be roughly constant across d; "
              "decsim/host is the cost of one Python decode() call per window over the "
              "batched C++ path, which is why the simulator charges a MODELED algorithm "
              "latency (LILLIPUT card or a measured software figure), never its own wall clock.",
              "", "## 2. Logical error rate, windowed loop vs whole-circuit PyMatching (memory_z)", "",
              "| d | rounds | p | pipeline LER (shots) | reference LER (100k shots) |",
              "|---|---|---|---|---|"]
    for d, rounds, p, shots in ((3, 9, 0.01, 400), (5, 15, 0.01, 200)):
        pipeline, reference = logical_error_rates(d, rounds, p, shots)
        lines.append(f"| {d} | {rounds} | {p} | {pipeline:.3f} ({shots}) | {reference:.4f} |")
        print(lines[-1], file=sys.stderr)
    lines += ["", "The pipeline decodes sliding windows (commit d, buffer d) serially with real "
              "boundary handoff, so its rate may sit slightly above the whole-circuit reference; "
              "a rate significantly below it, or outside the binomial interval of the pipeline "
              "shot count, would indicate a data-path defect."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
