"""Gate 10: timing knobs must not change what the loop decodes.

Same circuit, same physical error probability, same seed, run through the
whole loop once per timing point (round period x fixed algorithm-latency
card, deterministic preset cards only, never the measured wall clock).
Across all timing points of one seed these must be identical:

1. the detection events the QPU device sampled;
2. the sampled observable truth;
3. the windowed loop's logical prediction;
4. whole-circuit PyMatching's prediction on those events;
5. the logical-failure result.

Only timestamps may move: the last frame commit tick must differ between the
slowest and the fastest round period, or the timing knobs are dead.

    python -m experiments.validation.validate_timing_invariance [seeds]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pymatching

from experiments.baseline.baseline_closed_loop import (DEFAULT_CONFIG, build_run, load_config,
                                                       memory_circuit)

REPORT = Path(__file__).resolve().parents[1] / "results" / "validation" / "timing_invariance.md"

PHYSICAL_ERROR_PROBABILITY = 0.005          # enough logical failures to make check 5 meaningful
ROUND_PERIODS_US = (1.0, 0.9, 0.5, 0.1)     # safe, near the knee, and overloaded
ALGORITHM_CARDS_US = (0.028, 0.28)          # deterministic preset cards only


def run_point(config: dict, round_period_us: float, algorithm_latency_us: float, seed: int):
    spec, _ = build_run(config, physical_error_probability=PHYSICAL_ERROR_PROBABILITY,
                        round_period_us=round_period_us,
                        algorithm_latency_us=algorithm_latency_us, seed=seed)
    return spec.build()


def functional_outcome(done, matching) -> tuple:
    """(events, truth, loop prediction, direct prediction, failure) of one run."""
    operation_result = done.result.operation_results[0]
    events = done.qpu.model.sampled_detection_events(operation_result.operation_id)
    truth = tuple(operation_result.observable_truth)
    loop_prediction = tuple(operation_result.logical_observables)
    direct_prediction = tuple(int(bit) for bit in matching.decode(np.asarray(events, dtype=bool)))
    failure = loop_prediction != truth
    return (events, truth, loop_prediction, direct_prediction, failure)


def last_commit_tick(done) -> int:
    return max(record.committed_ticks for record in done.pauli_frame.snapshot().records)


def main(argv) -> None:
    seeds = int(argv[1]) if len(argv) > 1 else 50
    config = load_config(DEFAULT_CONFIG)
    circuit = memory_circuit(config, PHYSICAL_ERROR_PROBABILITY)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))

    checks = 0
    functional_violations = 0
    frozen_timestamp_seeds = 0
    failures_seen = 0
    for seed in range(seeds):
        outcomes = {}
        commit_ticks = {}
        for algorithm_latency_us in ALGORITHM_CARDS_US:
            for round_period_us in ROUND_PERIODS_US:
                done = run_point(config, round_period_us, algorithm_latency_us, seed)
                outcomes[(algorithm_latency_us, round_period_us)] = functional_outcome(done, matching)
                commit_ticks[(algorithm_latency_us, round_period_us)] = last_commit_tick(done)
        reference_point = (ALGORITHM_CARDS_US[0], ROUND_PERIODS_US[0])
        reference = outcomes[reference_point]
        failures_seen += reference[4]
        for point, outcome in outcomes.items():
            checks += 1
            if outcome != reference:
                functional_violations += 1
                print(f"seed {seed}, point {point}: functional outcome differs", file=sys.stderr)
        slow = commit_ticks[(ALGORITHM_CARDS_US[0], ROUND_PERIODS_US[0])]
        fast = commit_ticks[(ALGORITHM_CARDS_US[0], ROUND_PERIODS_US[-1])]
        if slow == fast:
            frozen_timestamp_seeds += 1

    points = len(ALGORITHM_CARDS_US) * len(ROUND_PERIODS_US)
    verdict = "PASS" if functional_violations == 0 and frozen_timestamp_seeds == 0 else "FAIL"
    lines = ["# Gate 10: timing invariance of the functional outcome", "",
             f"Circuit: {config['code_task']} d={config['distance']}, {config['rounds_per_shot']} rounds, "
             f"p={PHYSICAL_ERROR_PROBABILITY}, {seeds} seeds x {points} timing points "
             f"(round periods {ROUND_PERIODS_US} us x algorithm cards {ALGORITHM_CARDS_US} us).", "",
             "Compared per seed across all timing points: sampled detection events, observable "
             "truth, windowed prediction, whole-circuit PyMatching prediction, logical failure.", "",
             f"- functional comparisons: {checks}, violations: {functional_violations}",
             f"- seeds with a logical failure (check 5 exercised): {failures_seen}/{seeds}",
             f"- seeds whose last frame commit tick did not move between "
             f"{ROUND_PERIODS_US[0]} and {ROUND_PERIODS_US[-1]} us rounds: {frozen_timestamp_seeds}", "",
             f"Verdict: {verdict} (timing knobs move timestamps only, never the decoded answer)."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv)
