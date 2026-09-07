"""Gate point 1 for the observation tests: how to run it, what it holds.

weak_decoder_baseline d 3 p 0.003 seed 0, the first strict point of the
frozen suite (validation/responsibility_audit_2026_08_30/frozen_suite in
the sandbox beside this checkout). captured_fields returns the row the
gate's capture.py hashes, so a test can say that a knob moved nothing.
A checkout without the sandbox skips these tests rather than erroring.
"""

import dataclasses
import hashlib
import json
import pathlib

import pytest

import decsim.machine as machine_module

SUITE = pathlib.Path(
    "/scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/validation/"
    "responsibility_audit_2026_08_30/frozen_suite"
)
POINT = {
    "physical_error_probability": 0.003,
    "distance": 3,
    "round_period_us": 1.0,
}
SEED = 0
# the golden log hash of the point, from the frozen suite
POINT_LOG_SHA256 = "74c2e7aee37a"

_suite_is_missing = not SUITE.is_dir()
_REASON = (
    f"the frozen suite lives in the sandbox at {SUITE}, beside the "
    "repository; this checkout does not have it"
)
needs_the_frozen_suite = pytest.mark.skipif(_suite_is_missing, reason=_REASON)


def settings(**observation_changes):
    """Gate point 1's settings, with its observation section changed."""
    from experiments.experiment_config import load_experiment

    config_path = SUITE / "weak_decoder_baseline.yaml"
    config = load_experiment(config_path)
    point = config.point_settings(**POINT)
    if not observation_changes:
        return point
    observation = dataclasses.replace(point.observation, **observation_changes)
    return dataclasses.replace(point, observation=observation)


def run(**observation_changes):
    """One shot of gate point 1: its machine and its result."""
    point = settings(**observation_changes)
    machine = machine_module.Machine.build(point, SEED)
    result = machine.run()
    return machine, result


def log_sha256(machine) -> str:
    """The hash the gate pins over the whole narration."""
    text = "\n".join(machine.observation.log.lines)
    encoded = text.encode()
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def captured_fields(machine, result) -> dict:
    """Every field the gate's capture.py hashes, from one finished run."""
    observation = machine.observation
    frame = machine.pauli_frame.snapshot()
    stamps = observation.runtime_stamps
    strong_requests = machine.decoder_manager.strong_requests
    traffic_text = json.dumps(result.link_traffic, sort_keys=True)
    traffic_bytes = traffic_text.encode()
    traffic_digest = hashlib.sha256(traffic_bytes)
    depths = [depth for _tick, depth in observation.queue_depth.samples]
    fields = {
        "log_sha256": log_sha256(machine),
        "log_lines": len(observation.log.lines),
        "windows": _window_rows(observation.windows.windows),
        "frame_records": _frame_rows(frame.records),
        "frame_commit_count": frame.commit_count,
        "operation_results": _result_rows(result.operation_results),
        "link_traffic": result.link_traffic,
        "link_traffic_sha256": traffic_digest.hexdigest(),
        "queue_log": list(observation.queue_depth.samples),
        "max_queue_depth": max(depths, default=0),
        "runtime_timestamps": _stamp_rows(stamps),
        "controller_idle_rounds": observation.controller_counters.idle_rounds,
        "packing_drops": machine.observation.round_events.packing_drops,
        "strong_needed": strong_requests.counts.needed,
        "strong_cancelled": strong_requests.counts.cancelled,
    }
    return fields


def _window_rows(windows_by_key) -> dict:
    """The window fields the gate reads, keyed by the window's text."""
    rows = {}
    for key, window in windows_by_key.items():
        rows[repr(key)] = (
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
            window.t_first_round,
            window.t_data_complete,
            window.t_queued,
            window.t_dispatch,
            window.t_done,
            window.committed,
            window.decode_status,
        )
    return rows


def _frame_rows(records) -> list:
    """The frame record fields the gate reads, in commit order."""
    rows = []
    for record in records:
        rows.append(
            (
                repr(record.window_key),
                record.tier,
                record.run_sequence,
                record.accepted_ticks,
                record.committed_ticks,
                record.logical_observables,
            )
        )
    return rows


def _result_rows(operation_results) -> dict:
    """The per-operation results the gate reads, keyed by operation."""
    rows = {}
    for row in operation_results:
        rows[row.operation_id] = (
            row.result_status,
            row.logical_observables,
            row.observable_truth,
        )
    return rows


def _stamp_rows(stamps) -> tuple:
    """The five runtime timestamp maps the gate reads."""
    return (
        dict(stamps.op_start),
        dict(stamps.body_done),
        dict(stamps.decode_release),
        dict(stamps.result_return),
        stamps.last_finish,
    )
