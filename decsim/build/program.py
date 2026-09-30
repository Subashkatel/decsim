"""Register the workload with every component that reads it."""

import decsim.build.control as control_part
import decsim.build.plan as plan_build
import decsim.build.windows as windows_part
import decsim.records.program as program_records


def load_program(
    plan: plan_build.Plan,
    control: control_part.Control,
    windows: windows_part.Windows,
) -> None:
    """Register the workload with every component that reads it.

    The blocked operations with the release first, then every operation
    with the windows, then the streams, the idle accounting, and last
    the runtime, which starts the roots.
    """
    conditional_release = control.conditional_release
    window_manager = windows.window_manager
    for operation in plan.operations:
        if operation.blocked_by is not None:
            conditional_release.register_blocked_operation(
                operation.id, operation.blocked_by
            )
    for operation in plan.planned_operations:
        window_manager.register_operation(operation)
    run_plan = plan.run_plan
    window_manager.install_planned_holds(run_plan.buffering)
    source_round_limit_by_stream = _register_streams(plan, window_manager)
    program = program_records.ExecutionProgram(
        plan.operations,
        plan.decode_operations,
        plan.dynamic_streams,
        plan.protected_regions,
    )
    control.streams.load(program, source_round_limit_by_stream)
    for operation in program.operations:
        window_manager.register_operation(operation)
    control.idle_rounds.load(program)
    control.execution_runtime.load_program(program)


def _register_streams(plan, window_manager) -> dict:
    """Declare physical state independently of decoder model selection.

    gem5's AbstractMemory owns its backing state independently of timing
    policy; selecting a model likewise cannot create or reset this history.
    Returns each stream's source round limit, None for an open-ended one.
    """
    resolved_by_id = {
        resolved.operation_id: resolved
        for resolved in plan.run_plan.resolved_operations
    }
    source_round_limit_by_stream = {}
    for stream in plan.dynamic_streams:
        resolved = resolved_by_id[stream.id]
        source_round_limit = plan.device.declare_stream(
            stream, resolved.round_count
        )
        window_manager.register_stream(stream, source_round_limit)
        source_round_limit_by_stream[stream.id] = source_round_limit
    return source_round_limit_by_stream
