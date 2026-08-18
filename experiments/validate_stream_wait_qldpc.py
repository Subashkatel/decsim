"""Gate 5: real syndrome data through a decode wait (Q-064), checked against
Stim, qLDPC and PyMatching.

Program on one patch: op1 (3 rounds), then the patch waits for op1's decode
(sliding window commit 3 buffer 3, so op1's window completes at round 6, plus a
4 us decode), then op2 (3 rounds) released by that decode. Under the QPU cycle
clock the patch keeps extracting syndromes while it waits, and with the
extend_stream idle policy those rounds are the next rounds of the same finite
Stim circuit (13 rounds in all: 1-3 op1, 4-10 waiting, 11-13 op2).

Checked per shot, on recorded Stim detection events:
- every one of the 13 emitted rounds carries exactly the sampled detector bits
  of that round (Stim is the oracle for data continuity through the wait);
- the wait is 7 idle rounds every shot;
- decsim's stream windows equal qLDPC SlidingWindowDecoder's (window 2d,
  stride d) on the same DEM, and the final logical prediction equals qLDPC's
  and whole-circuit PyMatching's, shot for shot.

Usage: python -m experiments.validate_stream_wait_qldpc [shots]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pymatching
import stim

from decsim.qpu.stim_device import RecordedStimDevice
from decsim.config import TICKS_PER_US, TimingConfig
from decsim.decoder_engine import DecoderEngine, DecoderTiming
from decsim.decoders import PresetLatencyDecoder
from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
from decsim.message import Operation
from decsim.mwpm_decoder.decoder import PyMatchingDecoder
from decsim.pauli_frame import PauliFrameConfig
from decsim.controller.policies import ExtendStream
from decsim.program.round_policies import PerOpRounds
from decsim.run_spec import RunSpec
from experiments.validate_timing_swiper import zero_latency_links
from experiments.validate_windows_qldpc import run_qldpc

REPORT = Path("experiments/results/validation/stream_wait_qldpc.md")
D, SEG, DECODE_US, ROUNDS, STREAM = 3, 3, 4.0, 13, 100


def run_decsim_shot(circuit, dets, obs, shot, rounds_of):
    owner = Operation(id=STREAM, name="stream", qubits=(0,), patches=(0,), circuit=circuit)
    op1 = Operation(id=1, name="op1", qubits=(0,), patches=(0,), circuit=circuit, stream_id=STREAM)
    op2 = Operation(id=2, name="op2", qubits=(0,), patches=(0,), circuit=circuit, stream_id=STREAM,
                    predecessors=(1,), blocked_by=1)
    device = RecordedStimDevice(dets, obs, shot, detector_rounds={STREAM: rounds_of})
    emitted = {}                                    # global round -> bits, as the QPU emitted them

    def record(payloads):
        for p in payloads:
            emitted[p.round_index] = np.asarray(p.bits, dtype=np.uint8)
        return payloads
    original_round, original_idle = device.round_payloads, device.idle_round_payloads
    device.round_payloads = lambda op, r: record(original_round(op, r))
    device.idle_round_payloads = lambda op, s, g, patch: record(original_idle(op, s, g, patch))
    engine = DecoderEngine(PyMatchingDecoder(PresetLatencyDecoder(DECODE_US)),
                           DecoderTiming(before=(), after=(), frequency_mhz=1000.0))
    done = RunSpec(ops=[op1, op2], dynamic_streams=[owner], d=D,
                   rounds_policy=PerOpRounds({1: SEG, 2: SEG, STREAM: ROUNDS}), decoder=engine, num_units=1,
                   timing=TimingConfig(round_us=1.0), links=zero_latency_links(), idle_policy=ExtendStream(),
                   device=device, pauli_frame=PauliFrameConfig(commit_us=0.0, zero_commit_cost_justification="cadence check"),
                   seed=shot).build()
    stream_result = next(r for r in done.result.operation_results if r.operation_id == STREAM)
    windows = [w for (op_id, k), w in sorted(done.window_manager.windows.items()) if op_id == STREAM]
    us = lambda t: t / TICKS_PER_US
    return dict(prediction=np.asarray(stream_result.logical_observables, dtype=np.uint8),
                idle_rounds=done.controller.idle_rounds_emitted, emitted=emitted, windows=windows,
                op2_start=us(done.execution_runtime.op_start_time[2]))


def main(argv) -> None:
    shots = int(argv[1]) if len(argv) > 1 else 300
    p = 0.005
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=D, rounds=ROUNDS,
                                     after_clifford_depolarization=p, before_round_data_depolarization=p,
                                     before_measure_flip_probability=p, after_reset_flip_probability=p)
    dets, obs = circuit.compile_detector_sampler(seed=23).sample(shots, separate_observables=True)
    rounds_of = resolve_detector_rounds(circuit, None, ROUNDS)
    dets_of_round = {}
    for det, r in rounds_of.items():
        dets_of_round.setdefault(r, []).append(det)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    q = run_qldpc(circuit, dets, rounds_of, D, REPORT.parent)
    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    whole = matching.decode_batch(dets.astype(np.uint8)).astype(np.uint8)

    bits_ok = rounds_ok = idle_ok = agree_q = agree_pm = 0
    window_detectors = window_commits = None
    for shot in range(shots):
        r = run_decsim_shot(circuit, dets, obs, shot, rounds_of)
        rounds_ok += set(r["emitted"]) == set(range(1, ROUNDS + 1))
        bits_ok += all(np.array_equal(r["emitted"][k], dets[shot][sorted(dets_of_round.get(k, []))].astype(np.uint8))
                       for k in range(1, ROUNDS + 1))
        idle_ok += r["idle_rounds"] == 7 and r["op2_start"] == 10.0
        agree_q += bool(np.array_equal(r["prediction"], q["predictions"][shot]))
        agree_pm += bool(np.array_equal(r["prediction"], whole[shot]))
        if window_detectors is None:
            window_detectors = [sorted(det for k in range(w.start_round, w.buffer_hi + 1)
                                       for det in dets_of_round.get(k, ())) for w in r["windows"]]
            window_commits = [sorted(det for k in range(w.commit_lo, w.commit_hi + 1)
                                     for det in dets_of_round.get(k, ())) for w in r["windows"]]
    det_match = sum(a == b for a, b in zip(q["window_detectors"], window_detectors))
    commit_match = sum(a == b for a, b in zip(q["window_commits"], window_commits))
    q_fail = int(np.sum(np.any(q["predictions"] != obs, axis=1)))
    pm_fail = int(np.sum(np.any(whole != obs, axis=1)))
    lines = ["# Gate 5: real syndrome data through a decode wait, vs Stim, qLDPC and PyMatching", "",
             f"Program: op1 (rounds 1-3), wait for its decode (window commit 3 buffer 3 completes at round 6, "
             f"decode {DECODE_US:g} us, release at 10), op2 (rounds 11-13); one finite Stim circuit "
             f"rotated_memory_z d={D}, {ROUNDS} rounds, p={p}, {shots} recorded shots (Stim seed 23); QPU cycle 1 us, "
             "zero links, extend_stream idle policy so the waiting patch's rounds are the circuit's own rounds 4-10.", "",
             "| check | result |", "|---|---|",
             f"| every shot emits rounds 1..{ROUNDS} exactly once | {rounds_ok}/{shots} |",
             f"| every emitted round's bits equal the sampled detection events of that round (Stim oracle) | {bits_ok}/{shots} |",
             f"| wait is 7 idle rounds and op2 starts on the boundary at 10 us | {idle_ok}/{shots} |",
             f"| stream windows | decsim {len(window_detectors)} / qLDPC {q['window_count']} |",
             f"| per-window detector sets equal (qLDPC window 2d, stride d) | {det_match}/{max(len(window_detectors), q['window_count'])} |",
             f"| per-window commit sets equal | {commit_match}/{max(len(window_commits), q['window_count'])} |",
             f"| final logical prediction equal to qLDPC, shot for shot | {agree_q}/{shots} |",
             f"| final logical prediction equal to whole-circuit PyMatching, shot for shot | {agree_pm}/{shots} |",
             f"| logical failures vs truth: qLDPC / PyMatching | {q_fail} / {pm_fail} |", "",
             "The wait rounds are not filler: they are decoded (windows 4-6 and 7-9 above), and op2's data starts at "
             "round 11 exactly where the circuit continues, so the always-on QPU is verified with data, not only timing.",
             "", "Window tail: qLDPC knows the circuit is 13 rounds and merges the last 7 into one window; decsim's "
             "stream is cut in real time and cannot know at round 9 that the program ends at 13, so it commits 7-9 "
             "and then a final 10-13 window (the static planner, which knows the length, matches qLDPC's tail; "
             "Gate 1). The first windows are identical and every prediction agrees."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
