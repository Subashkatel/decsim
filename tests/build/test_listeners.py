"""Registering the workload with every component, and naming the seed roots.

Every stochastic owner is bound under a path in the run's seed tree
(decsim/seeding.py), and this module is where the root names the path of
each one, so a component draws the same numbers whatever else the run
holds. The load order is the other rule here: streams first, then every
operation with the windows, then the idle accounting, then the runtime.
"""

from collections.abc import Callable
from typing import Optional

import pytest

import decsim.build.escalation as escalation_build
import decsim.build.listeners as listener_build
import decsim.build.plan as plan_build
import decsim.frontends.settings as workload_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.program as program_records
import decsim.records.seeds as seed_records
import decsim.settings as machine_settings


def test_each_stochastic_owner_is_named_by_the_field_it_arrived_under():
    owners = {"qpu": "the device", "weak_decoder": "the weak unit"}

    roots = listener_build.build_seed_roots(**owners)

    assert roots == (
        (
            (seed_records.RunSeedPathSegment("field", "qpu"),),
            "the device",
        ),
        (
            (seed_records.RunSeedPathSegment("field", "weak_decoder"),),
            "the weak unit",
        ),
    )


def test_a_run_with_no_stochastic_owner_names_no_root():
    roots = listener_build.build_seed_roots()

    assert roots == ()


@pytest.mark.parametrize("source_round_limit", [None, 11])
def test_the_workload_reaches_every_component_in_the_stated_order(
    source_round_limit: Optional[int],
) -> None:
    """Physical declaration precedes stream models and execution consumers."""
    order = []
    device = _PhysicalDevice(order, source_round_limit)
    planned_round_count = 9
    plan = _plan(device, planned_round_count)
    release = _Recorder("conditional_release", order)
    windows = _Recorder("window_manager", order)
    streams = _Recorder("streams", order)
    idle = _Recorder("idle_rounds", order)
    runtime = _Recorder("execution_runtime", order)

    listener_build.load_program(plan, release, windows, streams, idle, runtime)

    assert order == [
        "conditional_release.register_blocked_operation",
        "window_manager.register_operation",
        "window_manager.install_planned_holds",
        "physical_device.declare_stream",
        "window_manager.register_stream",
        "streams.load",
        "window_manager.register_operation",
        "idle_rounds.load",
        "execution_runtime.load_program",
    ]
    stream = plan.dynamic_streams[0]
    assert device.declared_streams == [(stream, planned_round_count)]
    assert windows.calls[2] == (
        "register_stream",
        (stream, source_round_limit),
        {},
    )


def _plan(device: "_PhysicalDevice", round_count: int) -> plan_build.Plan:
    operation = program_records.Operation(
        1, "blocked", (0,), blocked_by=2, emits_detector_data=False
    )
    blocker = program_records.Operation(2, "blocker", (0,))
    stream = program_records.Operation(3, "memory", (0,), patches=(0,))
    rounds_policy = round_policies.FixedRounds(round_count)
    workload = workload_settings.WorkloadSettings(
        operations=(operation,),
        decode_operations=(blocker,),
        dynamic_streams=(stream,),
        rounds_policy=rounds_policy,
    )
    qpu = qpu_settings.QpuSettings(distance=3, device=device)
    settings = machine_settings.MachineSettings(workload=workload, qpu=qpu)
    policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    return plan_build.build_plan(settings, policy)


class _PhysicalDevice:
    """Declare physical history with a limit independent of planning rounds."""

    operation_circuit_scope = "none"

    def __init__(self, order: list[str], round_limit: Optional[int]) -> None:
        self.order = order
        self.round_limit = round_limit
        self.declared_streams: list[tuple[program_records.Operation, int]] = []

    def declare_stream(
        self, stream: program_records.Operation, round_count: int
    ) -> Optional[int]:
        self.order.append("physical_device.declare_stream")
        self.declared_streams.append((stream, round_count))
        return self.round_limit

    def window_model_source(self) -> syndrome_devices.NoWindowModels:
        return syndrome_devices.NO_WINDOW_MODELS


class _Recorder:
    """Records every call made on it, by component and method name."""

    def __init__(self, name: str, order: list[str]) -> None:
        self.name = name
        self.order = order
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, method_name: str) -> Callable[..., None]:
        def record(*arguments: object, **keywords: object) -> None:
            self.calls.append((method_name, arguments, keywords))
            self.order.append(f"{self.name}.{method_name}")

        return record
