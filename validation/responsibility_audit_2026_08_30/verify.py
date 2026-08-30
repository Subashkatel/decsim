#!/usr/bin/env python
"""Independent verifier for the frozen responsibility-audit suite.

This verifier does not reuse capture.py's own check logic. It
1. checks the golden file's recorded sha256 (corruption or silent edit),
2. re-runs the nine suite points through capture.py's capture routine,
3. compares fresh rows to golden with its own comparison code:
   strict points must match on every field; wall-clock points must match
   on the semantic projection only (the strong tier and the switching
   weak tier price decode latency from measured wall clock by design, so
   their tick-bearing fields legitimately vary between runs).

Run from the decsim repo root:
    .venv/bin/python validation/responsibility_audit_2026_08_30/verify.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden" / "golden.json"

# Recorded when the golden was frozen at decsim 65660a0. A hash mismatch
# means the golden file changed; that requires an approved design note,
# never a silent regeneration.
GOLDEN_SHA256 = "84e8f5bb4299dbf9b85f8c7c236925f1ce8a008a6d6466c752973b60d9a66f32"

# The verifier's own list of what a wall-clock point must preserve.
SEMANTIC_PROJECTION = (
    "point", "tier", "operation_results", "windows_semantic",
    "frame_records_semantic", "link_traffic_semantic", "max_queue_depth",
    "controller_idle_rounds", "packing", "strong_counters",
    "frame_duplicate_drops",
)


def load_capture_module():
    spec = importlib.util.spec_from_file_location(
        "frozen_capture", HERE / "frozen_suite" / "capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diff_point(fresh: dict, frozen: dict) -> list[str]:
    strict = frozen.get("tier") == "strict"
    fields = sorted(set(frozen) | set(fresh)) if strict else SEMANTIC_PROJECTION
    return [field for field in fields if fresh.get(field) != frozen.get(field)]


def main() -> int:
    golden_bytes = GOLDEN.read_bytes()
    digest = hashlib.sha256(golden_bytes).hexdigest()
    if digest != GOLDEN_SHA256:
        print(f"FAIL: golden.json sha256 {digest} does not match the frozen "
              f"record {GOLDEN_SHA256}")
        return 1
    print("golden integrity: ok")

    golden_rows = json.loads(golden_bytes)
    capture = load_capture_module()
    fresh_rows = capture.run_suite()

    if len(fresh_rows) != len(golden_rows):
        print(f"FAIL: {len(fresh_rows)} fresh points vs "
              f"{len(golden_rows)} golden points")
        return 1

    failing_points = 0
    for fresh, frozen in zip(fresh_rows, golden_rows):
        mismatched_fields = diff_point(fresh, frozen)
        label = frozen["point"]
        tier = frozen.get("tier")
        if mismatched_fields:
            failing_points += 1
            print(f"FAIL {label} [{tier}]: {', '.join(mismatched_fields)}")
        else:
            print(f"pass {label} [{tier}]")

    if failing_points:
        print(f"FAIL: {failing_points} of {len(golden_rows)} points differ")
        return 1
    print("PASS: behavior preserved (independent verifier)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
