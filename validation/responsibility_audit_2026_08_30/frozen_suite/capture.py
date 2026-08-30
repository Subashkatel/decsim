#!/usr/bin/env python
"""Frozen reference suite for the decsim responsibility audit.

Captures, for a fixed set of sweep points, everything the refactor gate
must preserve: scientific results, event ordering, timestamps, link
traffic, buffer occupancy (via the I/O trace), the selected decoder tier
per window, and the Pauli-frame records.

Usage (from the decsim repo root, with its .venv):
    .venv/bin/python <this file> capture   # writes golden.json next to it
    .venv/bin/python <this file> check     # re-runs and diffs against golden

The suite reads core decsim strictly read-only. Configs are local copies
in this directory (weak_decoder_baseline.yaml, strong_decoder_baseline.yaml,
switching_validation.yaml with trace_io on where noted).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = SUITE_DIR.parent / "golden" / "golden.json"

# (config file, physical error probability, distance, round period us, seed,
#  comparison tier). "strict" pins every field including the full log hash;
# it holds only when every decode engine prices latency from a deterministic
# card. The strong tier and the switching weak tier price latency from the
# MEASURED wall clock of the real decoder (decsim design: belief_matching /
# mwpm decoder.py, time.perf_counter_ns), so their tick-bearing fields vary
# run to run BY DESIGN; those points compare the "semantic" projection:
# logical results, window structure and decode statuses, the frame-record
# sequence (keys, tiers, observables, order), tick-free link traffic, max
# queue depth, idle rounds, and strong needed/cancelled counters.
SUITE_POINTS = [
    ("weak_decoder_baseline.yaml", 0.003, 3, 1.0, 0, "strict"),
    ("weak_decoder_baseline.yaml", 0.003, 3, 1.0, 1, "strict"),
    ("weak_decoder_baseline.yaml", 0.005, 5, 1.0, 0, "strict"),
    ("weak_decoder_baseline.yaml", 0.001, 3, 0.5, 0, "strict"),
    ("strong_decoder_baseline.yaml", 0.003, 3, 1.0, 0, "semantic"),
    ("strong_decoder_baseline.yaml", 0.005, 5, 1.0, 0, "semantic"),
    ("switching_validation.yaml", 0.008, 3, 1.0, 0, "semantic"),
    ("switching_validation.yaml", 0.008, 3, 1.0, 1, "semantic"),
    ("switching_validation.yaml", 0.008, 5, 1.0, 0, "semantic"),
]

SEMANTIC_FIELDS = ("point", "tier", "operation_results", "windows_semantic",
                   "frame_records_semantic", "link_traffic_semantic",
                   "max_queue_depth", "controller_idle_rounds", "packing",
                   "strong_counters", "frame_duplicate_drops")


def strip_ticks(value):
    """Drop tick/time-bearing keys from a nested JSON value."""
    if isinstance(value, dict):
        return {key: strip_ticks(item) for key, item in sorted(value.items())
                if "tick" not in key.lower() and "time" not in key.lower()}
    if isinstance(value, list):
        return [strip_ticks(item) for item in value]
    return value


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value):
    """JSON-stable projection of the runtime values we compare."""
    return json.loads(json.dumps(value, sort_keys=True, default=repr))


def capture_point(config_name, probability, distance, period_us, seed, tier):
    from experiments.build_run import build_run
    from experiments.experiment_config import load_experiment

    config = load_experiment(SUITE_DIR / config_name)
    spec, _engine = build_run(
        config, physical_error_probability=probability, distance=distance,
        round_period_us=period_us, seed=seed)
    completed = spec.build(verbose=False, io_trace=config.trace_io)

    runtime = completed.execution_runtime
    window_manager = completed.window_manager
    frame = completed.pauli_frame.snapshot()
    log_text = "\n".join(completed.engine.log_lines)

    windows = {
        repr(key): {
            "commit": [window.commit_lo, window.commit_hi],
            "buffer_hi": window.buffer_hi,
            "t_first_round": window.t_first_round,
            "t_data_complete": window.t_data_complete,
            "t_queued": window.t_queued,
            "t_dispatch": window.t_dispatch,
            "t_done": window.t_done,
            "committed": window.committed,
            "decode_status": window.decode_status,
        }
        for key, window in sorted(window_manager.windows.items())
    }
    frame_records = [
        {
            "window_key": repr(record.window_key),
            "tier": record.tier,
            "run_sequence": record.run_sequence,
            "accepted_ticks": record.accepted_ticks,
            "committed_ticks": record.committed_ticks,
            "logical_observables": record.logical_observables,
        }
        for record in frame.records
    ]
    operation_results = {
        str(row.operation_id): {
            "status": row.result_status,
            "observables": row.logical_observables,
            "truth": row.observable_truth,
        }
        for row in completed.result.operation_results
    }
    link_traffic = canonical(completed.result.link_traffic)
    return {
        "point": {
            "config": config_name, "p": probability, "d": distance,
            "round_period_us": period_us, "seed": seed,
        },
        "tier": tier,
        "windows_semantic": canonical({
            key: {field: row[field] for field in
                  ("commit", "buffer_hi", "committed", "decode_status")}
            for key, row in windows.items()}),
        "frame_records_semantic": canonical([
            {field: record[field] for field in
             ("window_key", "tier", "run_sequence", "logical_observables")}
            for record in frame_records]),
        "link_traffic_semantic": strip_ticks(link_traffic),
        "max_queue_depth": max(
            (depth for _, depth in completed.decoder_manager.queue_log),
            default=0),
        "operation_results": canonical(operation_results),
        # full log: event ordering, timestamps, and (with trace_io on)
        # buffer-occupancy narration, pinned as one hash
        "log_lines": len(completed.engine.log_lines),
        "log_sha256": sha(log_text),
        "runtime_timestamps": canonical({
            "op_start": runtime.op_start_time,
            "body_done": runtime.body_done_time,
            "decode_release": runtime.decode_release_time,
            "result_return": runtime.result_return_time_by_operation,
            "last_finish": runtime.last_finish_time,
        }),
        "windows": canonical(windows),
        "frame_records": canonical(frame_records),
        "frame_duplicate_drops": frame.duplicate_drop_count,
        "link_traffic_sha256": sha(json.dumps(link_traffic, sort_keys=True)),
        "link_traffic": link_traffic,
        "queue_log": canonical(completed.decoder_manager.queue_log),
        "controller_idle_rounds": completed.controller.idle_rounds_emitted,
        "packing": canonical({
            "reassembly_timeouts": completed.syndrome_packing.reassembly_timeouts,
            "packing_drops": completed.syndrome_packing.packing_drops,
        }),
        "strong_counters": {
            "needed": completed.decoder_manager.strong_needed,
            "cancelled": completed.decoder_manager.strong_cancelled,
        },
    }


def run_suite():
    rows = []
    for point in SUITE_POINTS:
        row = capture_point(*point)
        print(f"captured {row['point']}  log={row['log_lines']} lines "
              f"sha={row['log_sha256'][:12]}", flush=True)
        rows.append(row)
    return rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    rows = run_suite()
    if mode == "capture":
        GOLDEN_PATH.write_text(json.dumps(rows, indent=1, sort_keys=True))
        print(f"golden written: {GOLDEN_PATH} ({len(rows)} points)")
        return 0
    golden = json.loads(GOLDEN_PATH.read_text())
    failures = 0
    for fresh, frozen in zip(rows, golden):
        label = fresh["point"]
        fields = (sorted(frozen) if frozen.get("tier") == "strict"
                  else SEMANTIC_FIELDS)
        for field in fields:
            if fresh.get(field) != frozen.get(field):
                failures += 1
                print(f"MISMATCH {label} field {field}")
    if len(rows) != len(golden):
        failures += 1
        print(f"MISMATCH point count {len(rows)} vs {len(golden)}")
    print("PASS: behavior preserved" if failures == 0
          else f"FAIL: {failures} field mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
