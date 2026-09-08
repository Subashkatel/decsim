"""Register the workload with every component, and name every seed root."""

import decsim.build.plan as plan_build
import decsim.records.program as program_records
import decsim.records.seeds as seed_records


def build_seed_roots(**parts) -> tuple:
    """The seed path of every stochastic owner; the segments are results."""
    roots = []
    for name, value in parts.items():
        path = (seed_records.RunSeedPathSegment("field", name),)
        roots.append((path, value))
    return tuple(roots)


def load_program(
    plan: plan_build.Plan,
    conditional_release,
    window_manager,
    streams,
    idle_rounds,
    execution_runtime,
) -> None:
    """Register the workload with every component that reads it.

    The streams first, then every operation with the windows, then the
    idle accounting, then the runtime starts the roots.
    """
    for operation in plan.operations:
        if operation.blocked_by is not None:
            conditional_release.register_blocked_operation(
                operation.id, operation.blocked_by
            )
    for operation in plan.planned_operations:
        window_manager.register_operation(operation)
    run_plan = plan.run_plan
    window_manager.install_planned_holds(run_plan.buffering)
    for stream in plan.dynamic_streams:
        window_manager.register_stream(stream)
    program = program_records.ExecutionProgram(
        plan.operations,
        plan.decode_operations,
        plan.dynamic_streams,
        plan.protected_regions,
    )
    streams.load(program)
    for operation in program.operations:
        window_manager.register_operation(operation)
    idle_rounds.load(program)
    execution_runtime.load_program(program)
