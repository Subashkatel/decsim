"""Gate 1: decsim's sliding windows against qLDPC's SlidingWindowDecoder.

One saved problem (Stim circuit, its detector error model, sampled detection
events, decsim's detector-to-round map) is handed unchanged to two independent
implementations: qLDPC's SlidingWindowDecoder(window_size=2d, stride=d) running
in its own environment, and decsim's loop replaying the same shots. This file
holds no window logic and no decoder logic; it writes inputs, invokes both, and
compares named fields: window count, per-window detector sets, per-window
commit sets, and the final logical prediction of every shot.

Usage: python -m experiments.validate_windows_qldpc [shots]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import stim

from decsim.qpu.stim_device import RecordedStimDevice
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
from decsim.message import Operation
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec

EXTERNAL = Path("tmp/validation/external")
QLDPC_PYTHON = EXTERNAL / "qldpc-venv" / "bin" / "python"
REPORT = Path("experiments/results/validation/qldpc_windows.md")


def problem(d: int, rounds: int, p: float, shots: int, seed: int):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    rounds_of = resolve_detector_rounds(circuit, None, rounds)     # decsim's map, one-based
    return circuit, dets, obs, rounds_of


def run_qldpc(circuit, dets, rounds_of, d: int, workdir: Path) -> dict:
    inputs = workdir / "qldpc_inputs.npz"
    outputs = workdir / "qldpc_outputs.npz"
    np.savez(inputs, dem_text=str(circuit.detector_error_model(decompose_errors=True)),
             detection_events=dets.astype(np.uint8), window_size=2 * d, stride=d,
             detector_time=np.array([(det, r - 1) for det, r in rounds_of.items()]))
    subprocess.run([str(QLDPC_PYTHON), str(EXTERNAL / "qldpc_windows.py"), str(inputs), str(outputs)],
                   check=True)
    out = np.load(outputs, allow_pickle=True)
    return dict(predictions=out["predictions"], window_count=int(out["window_count"]),
                window_detectors=[sorted(w) for w in out["window_detectors"]],
                window_commits=[sorted(w) for w in out["window_commits"]])


def run_decsim(circuit, dets, obs, rounds_of, d: int, rounds: int) -> dict:
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    predictions, windows = [], None
    for shot in range(len(dets)):
        done = RunSpec(ops=[op], d=d, rounds_policy=FixedRounds(rounds),
                       device=RecordedStimDevice(dets, obs, shot),
                       decoder=PyMatchingDecoder(PresetLatencyDecoder(0.028)), seed=shot).build()
        predictions.append(done.result.operation_results[0].logical_observables)
        if windows is None:
            windows = [w for (op_id, k), w in sorted(done.window_manager.windows.items())]
    dets_of_round = {}
    for det, r in rounds_of.items():
        dets_of_round.setdefault(r, []).append(det)
    window_detectors = [sorted(det for r in range(w.start_round, w.buffer_hi + 1)
                               for det in dets_of_round.get(r, ())) for w in windows]
    window_commits = [sorted(det for r in range(w.commit_lo, w.commit_hi + 1)
                             for det in dets_of_round.get(r, ())) for w in windows]
    return dict(predictions=np.array(predictions, dtype=np.uint8), window_count=len(windows),
                window_detectors=window_detectors, window_commits=window_commits)


def main(argv) -> None:
    shots = int(argv[1]) if len(argv) > 1 else 300
    d, rounds, p = 3, 60, 0.005
    circuit, dets, obs, rounds_of = problem(d, rounds, p, shots, seed=11)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    q = run_qldpc(circuit, dets, rounds_of, d, REPORT.parent)
    s = run_decsim(circuit, dets, obs, rounds_of, d, rounds)
    same_windows = q["window_count"] == s["window_count"]
    det_match = sum(a == b for a, b in zip(q["window_detectors"], s["window_detectors"]))
    commit_match = sum(a == b for a, b in zip(q["window_commits"], s["window_commits"]))
    agree = int(np.sum(np.all(q["predictions"] == s["predictions"], axis=1)))
    q_fail = int(np.sum(np.any(q["predictions"] != obs, axis=1)))
    s_fail = int(np.sum(np.any(s["predictions"] != obs, axis=1)))
    lines = ["# Gate 1: decsim sliding windows vs qLDPC SlidingWindowDecoder", "",
             f"Problem: rotated_memory_z d={d}, {rounds} rounds, p={p}, {shots} shots (Stim seed 11), "
             "same DEM (decompose_errors=True), same shots, decsim's detector-to-round map given to both; "
             f"qLDPC pinned at tmp/references/code/qldpc (04d35a7), window_size={2*d}, stride={d}, with_MWPM.", "",
             "| quantity | qLDPC | decsim | match |", "|---|---|---|---|",
             f"| windows | {q['window_count']} | {s['window_count']} | {'yes' if same_windows else 'NO'} |",
             f"| per-window detector sets equal | | | {det_match}/{max(q['window_count'], s['window_count'])} |",
             f"| per-window commit sets equal | | | {commit_match}/{max(q['window_count'], s['window_count'])} |",
             f"| final logical prediction equal, shot for shot | | | {agree}/{shots} |",
             f"| logical failures vs truth | {q_fail}/{shots} | {s_fail}/{shots} | |", ""]
    if agree != shots:
        differing = [i for i in range(shots) if not np.all(q["predictions"][i] == s["predictions"][i])]
        lines.append(f"Differing shots: {differing[:20]}{' ...' if len(differing) > 20 else ''}")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
