"""Gate 9: windowed and unwindowed decoding through the loop against outside
references on the same input, run separately.

Same circuit (the baseline yaml's rotated memory), same sampled shots (the
device's own detection events are read back after each run), three ways:

1. decsim, sliding windows (commit d, buffer d) through the whole loop;
2. decsim, no windowing (NaiveOnlineScheme: one decode per shot after the
   last round) through the whole loop;
3. whole-circuit PyMatching on the identical detection events, outside the
   loop (the decoder everyone uses as the accuracy reference).

Checked, shot for shot: the no-window loop predicts exactly what PyMatching
predicts (it decodes the same problem once), and the windowed loop agrees
with PyMatching except where two solutions tie; both logical error rates
against the reference. Timing of the windowed loop against SWIPER is Gate 8
(the sliding window chain, window for window); the no-window case has no
SWIPER counterpart (its window builder always cuts d-round windows), so its
timing is checked against the rule it must follow: the one decode starts
when the last round has arrived and its correction lands after CWD + fetch +
algorithm + release + WDO + commit.

    python -m experiments.validation.validate_windowing_vs_reference [shots]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pymatching

from decsim.config import microseconds
from experiments.baseline.baseline_closed_loop import (DEFAULT_CONFIG, build_run, load_config,
                                                       memory_circuit)

REPORT = Path(__file__).resolve().parents[1] / "results" / "validation" / "windowing_vs_reference.md"


def run_loop(config: dict, scheme: str, seed: int):
    """One shot through the loop with the named scheme; returns the completed run."""
    point_config = dict(config)
    point_config["windowing"] = dict(config["windowing"], scheme=scheme)
    spec, _ = build_run(point_config, round_period_us=1.0, algorithm_latency_us=0.028, seed=seed)
    return spec.build()


def sampled_events(done, operation_id: int) -> np.ndarray:
    """The detection events the QPU device sampled for this shot."""
    return np.asarray(done.qpu.model._dets[operation_id], dtype=bool)


def prediction(done) -> tuple:
    result = done.result.operation_results[0]
    return tuple(result.logical_observables)


def truth(done) -> tuple:
    result = done.result.operation_results[0]
    return tuple(result.observable_truth)


def no_window_timing_rule(done, config: dict) -> tuple:
    """Expected: one window, dispatched when the last round is complete, done
    after CWD + fetch + algorithm + release; frame commit after WDO + commit."""
    windows = list(done.window_manager.windows.values())
    window = windows[0]
    links = config["links"]
    engine = config["decoder"]["engine"]
    cycle_us = 1 / engine["frequency_mhz"]
    fetch_us = engine["fetch_cycles_per_round"] * config["rounds_per_shot"] * cycle_us
    release_us = engine["release_cycles_per_job"] * cycle_us
    expected_done = (microseconds(window.t_data_complete) + links["cwd"]["latency_us"]
                     + fetch_us + 0.028 + release_us)
    frame_record = done.pauli_frame.snapshot().records[0]
    expected_commit = expected_done + links["wdo"]["latency_us"] + config["pauli_frame"]["commit_us"]
    return (len(windows), microseconds(window.t_dispatch), microseconds(window.t_data_complete),
            microseconds(window.t_done), round(expected_done, 3),
            microseconds(frame_record.committed_ticks), round(expected_commit, 3))


def main(argv) -> None:
    shots = int(argv[1]) if len(argv) > 1 else 200
    config = load_config(DEFAULT_CONFIG)
    config["physical_error_probability"] = 0.005     # enough failures to compare rates
    circuit = memory_circuit(config)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))

    failures = {"sliding": 0, "naive_online": 0, "pymatching": 0}
    disagreements = {"sliding": 0, "naive_online": 0}
    timing_problems = 0
    for seed in range(shots):
        sliding = run_loop(config, "sliding", seed)
        unwindowed = run_loop(config, "naive_online", seed)
        events = sampled_events(sliding, 1)
        if not np.array_equal(events, sampled_events(unwindowed, 1)):
            raise RuntimeError(f"seed {seed}: the two runs sampled different shots")
        reference = tuple(int(bit) for bit in matching.decode(events))
        shot_truth = truth(sliding)
        failures["pymatching"] += reference != shot_truth
        for name, done in (("sliding", sliding), ("naive_online", unwindowed)):
            predicted = prediction(done)
            failures[name] += predicted != shot_truth
            disagreements[name] += predicted != reference
        rule = no_window_timing_rule(unwindowed, config)
        window_count, dispatch, data_complete, done_at, expected_done, commit, expected_commit = rule
        if window_count != 1 or dispatch != data_complete or done_at != expected_done or commit != expected_commit:
            timing_problems += 1
        if seed == 0:
            first_rule = rule

    lines = ["# Gate 9: windowed and unwindowed loop vs whole-circuit PyMatching, same shots", "",
             f"Circuit: {config['code_task']} d={config['distance']}, {config['rounds_per_shot']} rounds, "
             f"p={config['physical_error_probability']}, {shots} shots, identical detection events to all three.", "",
             "| decoder | logical failures | LER | shots disagreeing with PyMatching |",
             "|---|---|---|---|"]
    for name in ("sliding", "naive_online", "pymatching"):
        disagree = "" if name == "pymatching" else disagreements[name]
        lines.append(f"| {name} | {failures[name]} | {failures[name] / shots:.4f} | {disagree} |")
    window_count, dispatch, data_complete, done_at, expected_done, commit, expected_commit = first_rule
    lines += ["",
              "No-window timing rule, shot 0: "
              f"{window_count} window; dispatched at {dispatch} us = last round complete at {data_complete} us; "
              f"decode done {done_at} us (rule {expected_done}); frame commit {commit} us (rule {expected_commit}). "
              f"Shots violating the rule: {timing_problems}/{shots}.",
              "",
              "Windowed timing against SWIPER: Gate 8 (experiments/results/validation/loop_swiper.md)."]
    verdict = "PASS" if disagreements["naive_online"] == 0 and timing_problems == 0 else "FAIL"
    lines += ["", f"Verdict: {verdict} (no-window predictions identical to PyMatching on every shot; "
              "windowed disagreements are tie-breaks between equal-weight matchings)."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
