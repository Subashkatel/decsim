"""Build the run's plan: the code, the workload's operations, the windows.

Everything the wiring reads that the planner and the workload fix, laid
out once before any component is built. gem5's configuration scripts do
the same: configs/deprecated/example/se.py resolves the workload and the
system before it wires a single port.
"""

import copy
import dataclasses
from typing import Any

import stim

import decsim.build.escalation as escalation_build
import decsim.controller.settings as controller_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.planner as planner
import decsim.frontends.settings as workload_settings
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.tables as tables
import decsim.windows.settings as window_settings
import decsim.windows.window_interactions as window_interactions

# What every windowing scheme row declares, so the escalation policy and
# the plan read a fact rather than the row's class (ports.py,
# WindowingScheme).
_SCHEME_DECLARATIONS = (
    "has_trailing_tail_context",
    "commits_in_one_serial_chain",
    "supports_dynamic_streams",
)


@dataclasses.dataclass(frozen=True)
class Plan:
    """Everything the wiring reads that the planner and the workload fix."""

    code: Any
    layout: Any
    scheme: Any
    boundary_policy: Any
    window_interaction: Any
    idle_policy: Any
    operations: tuple
    decode_operations: tuple
    dynamic_streams: tuple
    protected_regions: tuple
    all_operations: tuple
    planned_operations: tuple
    view_by_id: dict
    run_plan: planner.RunPlan
    resource_claims: dict
    device: Any
    error_model_provider: Any

    @property
    def round_ticks(self) -> int:
        """The ticks one syndrome round takes, as the run plan sized it."""
        return self.run_plan.round_ticks


def build_plan(
    settings: machine_settings.MachineSettings, escalation_policy
) -> Plan:
    """The code, the workload's operations and the window plan."""
    code, layout = settings.qpu.build_code(
        commit_rounds_override=settings.windows.commit_rounds,
        buffer_rounds_override=settings.windows.buffer_rounds,
    )
    operations, decode_operations, dynamic_streams, rounds_policy = _operations(
        settings.workload, code
    )
    every_operation = operations + decode_operations + dynamic_streams
    all_operations = _unique_operations(every_operation)
    views = []
    for operation in all_operations:
        view = program_records.OperationPlanningView.from_operation(operation)
        views.append(view)
    views = tuple(views)
    view_by_id = {}
    for view in views:
        view_by_id[view.id] = view
    external_blocker_ids = []
    for operation in decode_operations + dynamic_streams:
        external_blocker_ids.append(operation.id)
    planner.check_operation_graph(
        list(operations),
        validate_blockers=True,
        external_blocker_ids=external_blocker_ids,
    )
    scheme = _scheme(settings.windows, escalation_policy)
    boundary_policy = _boundary_policy(
        settings.windows, settings.escalation, escalation_policy
    )
    absorbs_weak_windows = escalation_build.absorbs_weak_windows(
        settings.escalation
    )
    reread_regions = settings.escalation.restart_reread_buffer_regions
    window_interaction = settings.windows.window_interaction
    if window_interaction is None:
        payload_row = tables.row(
            window_settings.BOUNDARY_PAYLOADS,
            "windows.boundary_payload",
            settings.windows.boundary_payload,
        )
        boundary_payload = payload_row()
        window_interaction = window_interactions.DefaultWindowInteraction(
            reread_regions, boundary_payload
        )
    if dynamic_streams and not scheme.supports_dynamic_streams:
        raise ValueError(
            "dynamic streams require a windowing scheme that supports them"
        )
    has_static_decode_plan = settings.workload.decode_operations is not None
    workload_row = tables.row(
        workload_settings.WORKLOADS, "workload.kind", settings.workload.kind
    )
    has_frontend = workload_row.has_frontend
    commit_round_count = code.commit_rounds()
    buffer_round_count = code.buffer_rounds()
    run_shape = decoding_records.RunShape(
        scheme=scheme,
        boundary_policy=boundary_policy,
        operations=views,
        commit_round_count=commit_round_count,
        buffer_round_count=buffer_round_count,
        strong_window=settings.escalation.strong_window,
        is_absorbing_strong_window=absorbs_weak_windows,
        is_bulk_strong=settings.decoder_manager.bulk_strong,
        has_dynamic_streams=bool(dynamic_streams),
        has_static_decode_plan=has_static_decode_plan,
        has_frontend=has_frontend,
    )
    escalation_policy.check_plan(run_shape)
    planned_operations = _decode_plan_operations(
        operations,
        decode_operations,
        dynamic_streams,
        static_decode_selected=has_static_decode_plan,
    )
    planned_ids = []
    for operation in planned_operations:
        planned_ids.append(operation.id)
    run_plan = planner.plan_execution(
        operations=views,
        planned_operation_ids=tuple(planned_ids),
        code=code,
        layout=layout,
        scheme=scheme,
        rounds_policy=rounds_policy,
        fallback_round_microseconds=settings.qpu.round_period_microseconds,
        retain_strong_context=escalation_policy.requires_strong_context,
        absorbs_weak_windows=absorbs_weak_windows,
        restart_reread_buffer_regions=reread_regions,
        has_open_ended_dynamic_streams=bool(dynamic_streams),
    )
    resource_claims = {}
    for operation in operations:
        view = view_by_id[operation.id]
        claims = layout.resources_for(view)
        resource_claims[operation.id] = tuple(claims)
    device = _syndrome_source(settings.qpu)
    _install_device_circuits(device, all_operations)
    error_model_provider = settings.qpu.error_model_provider
    if error_model_provider is None:
        error_model_provider = device
    elif hasattr(error_model_provider, "operation_circuit_scope"):
        _install_device_circuits(error_model_provider, all_operations)
    idle_policy = _idle_policy(settings.idle_policy)
    return Plan(
        code=code,
        layout=layout,
        scheme=scheme,
        boundary_policy=boundary_policy,
        window_interaction=window_interaction,
        idle_policy=idle_policy,
        operations=operations,
        decode_operations=decode_operations,
        dynamic_streams=dynamic_streams,
        protected_regions=tuple(settings.workload.protected_regions),
        all_operations=all_operations,
        planned_operations=tuple(planned_operations),
        view_by_id=view_by_id,
        run_plan=run_plan,
        resource_claims=resource_claims,
        device=device,
        error_model_provider=error_model_provider,
    )


def _operations(settings: workload_settings.WorkloadSettings, code) -> tuple:
    """The workload's private operation copies and its rounds policy.

    The run never mutates the caller's operations; an operation without
    its own feedback boundary mode takes the workload's.
    """
    row = tables.row(
        workload_settings.WORKLOADS, "workload.kind", settings.kind
    )
    source_operations, fixed_rounds_policy = row.operations(settings, code)
    rounds_policy = settings.rounds_policy
    if rounds_policy is None:
        rounds_policy = fixed_rounds_policy
    if rounds_policy is None:
        rounds_policy = round_policies.GateRounds()
    decode_operations = settings.decode_operations or ()
    copies = {}
    operations = _copies(source_operations, copies, settings)
    decode_copies = _copies(decode_operations, copies, settings)
    stream_copies = _copies(settings.dynamic_streams, copies, settings)
    planner.check_workload_identity(operations, decode_copies, stream_copies)
    return operations, decode_copies, stream_copies, rounds_policy


def _copies(
    operations, copies: dict, settings: workload_settings.WorkloadSettings
) -> tuple:
    """Private copies of the operations, one per object identity."""
    copied = []
    for operation in operations:
        identity = id(operation)
        if identity not in copies:
            copies[identity] = _private_copy(operation, settings)
        copied.append(copies[identity])
    return tuple(copied)


def _private_copy(operation, settings: workload_settings.WorkloadSettings):
    private = copy.copy(operation)
    if private.feedback_boundary_mode is None:
        private.feedback_boundary_mode = settings.feedback_boundary_mode
    return private


def _unique_operations(operations) -> tuple:
    unique = {}
    for operation in operations:
        unique.setdefault(operation.id, operation)
    unique_operations = unique.values()
    return tuple(unique_operations)


def _decode_plan_operations(
    operations, decode_operations, dynamic_streams, *, static_decode_selected
) -> tuple:
    """The operations whose windows the plan decodes."""
    if static_decode_selected:
        return decode_operations
    dynamic_ids = set()
    for stream in dynamic_streams:
        dynamic_ids.add(stream.id)
    planned = []
    for operation in operations:
        if not operation.emits_detector_data:
            continue
        if operation.stream_id in dynamic_ids:
            continue
        planned.append(operation)
    return tuple(planned)


def _scheme(windows: window_settings.WindowSettings, escalation_policy):
    """The windowing scheme of the kind, or the Python-built one.

    A policy that may escalate needs the lookahead terminal policy on
    sliding windows: the literature-exact flush has no trailing tail
    context, which is the fact the policy's own refusal reads.
    """
    scheme = _chosen_scheme(windows, escalation_policy)
    _refuse_undeclared_scheme(scheme)
    return scheme


def _chosen_scheme(windows: window_settings.WindowSettings, escalation_policy):
    """The Python-built scheme, or the kind's row on the section's card."""
    if windows.scheme is not None:
        return windows.scheme
    row = tables.row(
        window_settings.WINDOWING_SCHEMES, "windows.kind", windows.kind
    )
    terminal_policy = _terminal_policy(windows, escalation_policy)
    card = window_records.WindowingSchemeCard(terminal_policy=terminal_policy)
    return row(card)


def _terminal_policy(
    windows: window_settings.WindowSettings, escalation_policy
) -> str:
    """The section's terminal policy, or the one the escalation needs.

    A policy that may escalate reads context past the last window's
    commit, so a silent section gets the lookahead tail; the policy's own
    refusal (escalation/policies.py) is what stops a scheme whose last
    window carries no trailing tail.
    """
    if windows.terminal_policy is not None:
        return windows.terminal_policy
    if escalation_policy.requires_strong_context:
        return "lookahead"
    return "flush"


def _refuse_undeclared_scheme(scheme) -> None:
    """A windowing scheme row declares the facts its callers read."""
    row = type(scheme)
    row_name = row.__name__
    declarations = ", ".join(_SCHEME_DECLARATIONS)
    for declaration in _SCHEME_DECLARATIONS:
        if hasattr(scheme, declaration):
            continue
        raise ValueError(
            f"windowing scheme {row_name} does not declare "
            f"{declaration}; every row of windows.kind declares "
            f"{declarations}"
        )


def _refuse_undeclared_boundary_policy(boundary_policy) -> None:
    """A boundary policy row declares whether it ships provisionally."""
    if hasattr(boundary_policy, "ships_provisional_boundaries"):
        return
    row = type(boundary_policy)
    row_name = row.__name__
    raise ValueError(
        f"boundary policy {row_name} does not declare "
        "ships_provisional_boundaries; every boundary policy row declares "
        "whether it ships a boundary before the result is final"
    )


def _boundary_policy(
    windows: window_settings.WindowSettings,
    escalation: escalation_settings.EscalationSettings,
    escalation_policy,
):
    """The row windows.boundaries names, or the one the escalation needs.

    A policy that may escalate holds boundaries until results are final
    (descendants wait out an escalation); a strong window that absorbs
    the weak windows it covers keeps the weak chain committing eagerly.
    The boundary policy's own check_plan refuses the wrong pairing when a
    yaml names it against the escalation.
    """
    if windows.boundary_policy is not None:
        _refuse_undeclared_boundary_policy(windows.boundary_policy)
        return windows.boundary_policy
    boundaries = _boundaries_name(windows, escalation, escalation_policy)
    row = tables.row(
        window_settings.BOUNDARY_POLICIES, "windows.boundaries", boundaries
    )
    return row()


def _boundaries_name(
    windows: window_settings.WindowSettings,
    escalation: escalation_settings.EscalationSettings,
    escalation_policy,
) -> str:
    """The section's boundaries row, or the one the escalation needs."""
    if windows.boundaries is not None:
        return windows.boundaries
    absorbs_weak_windows = escalation_build.absorbs_weak_windows(escalation)
    may_escalate = escalation_policy.requires_strong_context
    is_serial_escalation = may_escalate and not absorbs_weak_windows
    if is_serial_escalation:
        return "held"
    return "eager"


def _idle_policy(settings: controller_settings.IdlePolicySettings):
    if settings.policy is not None:
        return settings.policy
    row = tables.row(
        controller_settings.IDLE_POLICIES, "idle_policy", settings.kind
    )
    return row()


def _syndrome_source(settings: qpu_settings.QpuSettings):
    """The device of the qpu kind, or the Python-built one."""
    if settings.device is not None:
        return settings.device
    row = tables.row(qpu_settings.SYNDROME_SOURCES, "qpu.kind", settings.kind)
    return row(**settings.arguments)


def _install_device_circuits(device, operations) -> None:
    """Give a per-operation device its own copy of every circuit."""
    scope = getattr(device, "operation_circuit_scope", None)
    if scope == "none":
        for operation in operations:
            operation.circuit = None
        return
    if scope != "per_operation":
        raise ValueError(
            "device operation_circuit_scope must be none or per_operation"
        )
    for operation in operations:
        if operation.circuit is None:
            continue
        circuit_text = str(operation.circuit)
        operation.circuit = stim.Circuit(circuit_text)
