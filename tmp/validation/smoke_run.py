"""Deterministic integration smoke run for the evidence-first cleanup."""

import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decsim.decoders import PerRoundDecoder
from decsim.message import OpKind, Operation
from decsim.metrics import DecoderUtilization
from decsim.run_spec import RunSpec


def make_metrics(engine, window_manager, decoder_manager, execution_runtime, factory):
    return [DecoderUtilization(decoder_manager)]


def operation_snapshot(operation):
    return {
        "id": operation.id,
        "name": operation.name,
        "qubits": list(operation.qubits),
        "patches": list(operation.patches),
        "predecessors": list(operation.predecessors),
        "kind": operation.kind.name,
    }


def load_baseline():
    baseline_path = Path(__file__).with_name("SMOKE_BASELINE.yaml")
    with baseline_path.open("r", encoding="utf-8") as baseline_file:
        baseline = yaml.safe_load(baseline_file)
    if not isinstance(baseline, Mapping) or baseline.get("schema_version") != 1:
        raise ValueError("smoke baseline must be a schema-version-1 mapping")
    if not isinstance(baseline.get("snapshot"), Mapping):
        raise ValueError("smoke baseline must contain a snapshot mapping")
    return baseline["snapshot"]


operations = (
    Operation(
        id=0,
        name="prepare patch 0",
        qubits=(0,),
        patches=(0,),
        kind=OpKind.MEMORY,
    ),
    Operation(
        id=1,
        name="merge patches 0 and 1",
        qubits=(0, 1),
        patches=(0, 1),
        predecessors=(0,),
        kind=OpKind.MERGE,
    ),
    Operation(
        id=2,
        name="measure patch 1",
        qubits=(1,),
        patches=(1,),
        predecessors=(1,),
        kind=OpKind.MEASURE,
    ),
)

completed = RunSpec(
    ops=operations,
    decoder=PerRoundDecoder(tau_us=0.1),
    make_metrics=make_metrics,
).build()
result = completed.result
metric_values = result.metric_values()

assert result.terminal_status == "complete"
assert result.event_queue_empty
assert result.decode_work_settled
assert result.execution_workload_complete
assert result.metric_results
assert "decoder_utilization" in metric_values

actual_snapshot = {
    "workload": {
        "operations": [operation_snapshot(operation) for operation in operations],
    },
    "result": {
        "terminal_status": result.terminal_status,
        "event_queue_empty": result.event_queue_empty,
        "decode_work_settled": result.decode_work_settled,
        "execution_workload_complete": result.execution_workload_complete,
        "execution_done_ticks": result.execution_done_ticks,
        "fully_done_ticks": result.fully_done_ticks,
        "operation_results": [
            {
                "operation_id": operation_result.operation_id,
                "result_status": operation_result.result_status,
            }
            for operation_result in result.operation_results
        ],
    },
    "metric_values": metric_values,
}
assert actual_snapshot == load_baseline()
