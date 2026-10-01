"""Build the run's plan: the code, the workload's operations, the windows.

Everything the wiring reads that the planner and the workload fix, laid
out once before any component is built. gem5's configuration scripts do
the same: configs/deprecated/example/se.py resolves the workload and the
system before it wires a single port.
"""

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any

import stim

import decsim.build.escalation as escalation_build
import decsim.controller.settings as controller_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.escalation.settings as escalation_settings
import decsim.frontends.planner as planner
import decsim.frontends.settings as workload_settings
import decsim.ports as ports
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.records.workload as workload_records
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
    run_plan: planner.RunPlan
    resource_claims: dict
    device: Any
    error_model_provider: Any
    formation_reads: window_records.FormationReads

    @property
    def round_ticks(self) -> int:
        """The ticks one syndrome round takes, as the run plan sized it."""
        return self.run_plan.round_ticks


def build_plan(
    settings: machine_settings.MachineSettings, escalation_policy
) -> Plan:
    """The code, the workload's operations and the window plan.

    Keep the root's run-shape and plan records together so their shared
    inputs remain visible; named helpers resolve individual collaborators.
    """
    code, layout = settings.qpu.build_code(
        commit_rounds_override=settings.windows.commit_rounds,
        buffer_rounds_override=settings.windows.buffer_rounds,
    )
    operations, decode_operations, dynamic_streams, rounds_policy = _operations(
        settings.workload
    )
    external_decode_operations = decode_operations + dynamic_streams
    every_operation = operations + external_decode_operations
    all_operations = _unique_operations(every_operation)
    views, view_by_id = _planning_views(all_operations)
    external_blocker_ids = []
    for operation in external_decode_operations:
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
    window_interaction = _window_interaction(settings.windows, reread_regions)
    if dynamic_streams and not scheme.supports_dynamic_streams:
        raise ValueError(
            "dynamic streams require a windowing scheme that supports them"
        )
    has_static_decode_plan = settings.workload.decode_operations is not None
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
    physical_circuits = settings.workload.physical_circuits
    physical_tables = _physical_formation_tables(physical_circuits)
    device = _syndrome_source(
        settings.qpu, code, physical_circuits, physical_tables
    )
    formation_tables = _formation_tables(
        device, planned_operations, physical_tables, rounds_policy, code
    )
    formation_reads = _formation_reads(
        settings, escalation_policy, formation_tables
    )
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
        formation_reads=formation_reads,
    )
    resource_claims = _resource_claims(operations, view_by_id, layout)
    error_model_provider = settings.qpu.error_model_provider
    if error_model_provider is None:
        error_model_provider = device.window_model_source()
    _refuse_bulk_strong_without_a_merge(
        settings, escalation_policy, device, error_model_provider
    )
    _install_operation_circuits(device, error_model_provider, all_operations)
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
        run_plan=run_plan,
        resource_claims=resource_claims,
        device=device,
        error_model_provider=error_model_provider,
        formation_reads=formation_reads,
    )


def _planning_views(operations):
    views = []
    view_by_id = {}
    for operation in operations:
        view = program_records.OperationPlanningView.from_operation(operation)
        views.append(view)
        view_by_id[view.id] = view
    return tuple(views), view_by_id


def _resource_claims(operations, view_by_id, layout):
    claims_by_operation = {}
    for operation in operations:
        view = view_by_id[operation.id]
        claims = layout.resources_for(view)
        claims_by_operation[operation.id] = tuple(claims)
    return claims_by_operation


def _window_interaction(settings, reread_regions):
    payload_row = tables.row(
        window_settings.BOUNDARY_PAYLOADS,
        "windows.boundary_payload",
        settings.boundary_payload,
    )
    boundary_payload = payload_row()
    return window_interactions.DefaultWindowInteraction(
        reread_regions, boundary_payload
    )


def _operations(settings: workload_settings.WorkloadSettings) -> tuple:
    """The workload's private operation copies and its rounds policy.

    The run never mutates the caller's operations; an operation without
    its own feedback boundary mode takes the workload's.
    """
    rounds_policy = settings.rounds_policy
    if rounds_policy is None:
        rounds_policy = round_policies.GateRounds()
    decode_operations = settings.decode_operations or ()
    copies = {}
    operations = _copies(settings.operations, copies, settings)
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
    """The Python-built scheme, or the kind's row on the section's card.

    A row with keys of its own is built with its Settings record too.
    """
    if windows.scheme is not None:
        return windows.scheme
    row = tables.row(
        window_settings.WINDOWING_SCHEMES, "windows.kind", windows.kind
    )
    terminal_policy = _terminal_policy(windows, escalation_policy)
    card = window_records.WindowingSchemeCard(terminal_policy=terminal_policy)
    if windows.row_settings is None:
        return row(card)
    return row(card, settings=windows.row_settings)


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
    """The section's boundaries row, or the one the rows declare.

    A default lives on the class that owns the parameter, gem5's rule for
    a SimObject's params (gem5 src/python/m5/SimObject.py
    :313-318, _new_param setting the ParamDesc's default on the class it
    is declared in, inherited through the _values parent chain set at
    :240-254). The escalation policy row owns this one, and a row that
    may escalate hands it to its strong window shape, whose absorption is
    what decides.
    """
    if windows.boundaries is not None:
        return windows.boundaries
    if not escalation_policy.requires_strong_context:
        return escalation_policy.default_boundary_policy
    shape = escalation_build.strong_window_row(escalation)
    return shape.default_boundary_policy


def _idle_policy(settings: controller_settings.IdlePolicySettings):
    """The Python-built policy, or the kind's row.

    A row with keys of its own is built with its Settings record.
    """
    if settings.policy is not None:
        return settings.policy
    row = tables.row(
        controller_settings.IDLE_POLICIES, "idle_policy.kind", settings.kind
    )
    if settings.row_settings is None:
        return row()
    return row(settings=settings.row_settings)


def _syndrome_source(
    settings: qpu_settings.QpuSettings,
    code,
    physical_circuits: Mapping,
    physical_tables: Mapping,
):
    """The device of the qpu kind, or the Python-built one.

    A row that shapes its payloads by the code card is built with the
    run's card, so a yaml that names it needs no argument of its own; a
    row that reads its widths off a circuit is built with the
    workload's physical circuits instead; a row with keys of its own is
    built with its Settings record too.
    """
    if settings.device is not None:
        return settings.device
    row = tables.row(qpu_settings.SYNDROME_SOURCES, "qpu.kind", settings.kind)
    arguments = {}
    if row.takes_code_card:
        arguments["code"] = code
    else:
        circuit_arguments = _circuit_arguments(
            physical_circuits, physical_tables
        )
        arguments.update(circuit_arguments)
    if settings.row_settings is not None:
        arguments["settings"] = settings.row_settings
    return row(**arguments)


def _circuit_arguments(
    physical_circuits: Mapping, physical_tables: Mapping
) -> dict:
    """The source's constructor arguments for the workload's circuits.

    A finite circuit is its measurement schedule and the round each
    detector completes in, keyed by the stream that runs it, which is
    StimDevice's declaration (qpu/stim_device.py); live fragments are
    the programs StreamingStimDevice executes.
    """
    measurement_rounds = {}
    detector_rounds = {}
    programs = {}
    for key, physical in physical_circuits.items():
        if isinstance(physical, workload_records.FiniteCircuit):
            measurement_rounds[key] = dict(physical.measurement_rounds)
            table = physical_tables[key]
            detector_rounds[key] = table.detector_rounds()
            continue
        programs[key] = physical
    arguments = {}
    if measurement_rounds:
        arguments["measurement_rounds"] = measurement_rounds
        arguments["detector_rounds"] = detector_rounds
    if programs:
        arguments["programs"] = programs
    return arguments


def _physical_formation_tables(physical_circuits: Mapping) -> dict:
    """Each finite circuit's recipes, by the stream key that runs it.

    The source is declared each detector's round from this table, and
    the plan reads from it how many rounds before a read its first
    round's recipes reach.
    """
    tables = {}
    for key, physical in physical_circuits.items():
        if not isinstance(physical, workload_records.FiniteCircuit):
            continue
        tables[key] = _physical_formation_table(physical)
    return tables


def _physical_formation_table(
    physical: workload_records.FiniteCircuit,
) -> detector_formation.FormationTable:
    """The recipes, formed off the declared measurement schedule."""
    schedule = physical.measurement_rounds
    rounds = schedule.values()
    round_count = max(rounds)
    return detector_formation.build_formation_table(
        physical.circuit, round_count, measurement_rounds=schedule
    )


def _formation_tables(
    device, planned_operations, physical_tables: Mapping, rounds_policy, code
) -> dict:
    """The recipes of every planned operation the source forms from.

    The source reads an operation's recipes off its circuit when the
    operation begins (qpu/stim_device.py, _sample_shot), after the plan
    has placed the holds, so the plan reads the same circuit here. A
    source that answers no recipes forms nothing, and a stream's
    segments are planned while their stream runs, so neither has one.
    """
    if not isinstance(device, ports.DetectionEventFormer):
        return {}
    tables = {}
    for operation in planned_operations:
        table = physical_tables.get(operation.id)
        if table is None:
            table = _operation_formation_table(operation, rounds_policy, code)
        if table is None:
            continue
        tables[operation.id] = table
    return tables


def _operation_formation_table(operation, rounds_policy, code):
    """An operation's own circuit's recipes; None when it has none."""
    if operation.circuit is None or operation.stream_id is not None:
        return None
    round_count = rounds_policy.rounds_for(operation, code)
    return detector_formation.build_formation_table(
        operation.circuit, round_count
    )


def _formation_reads(
    settings: machine_settings.MachineSettings,
    escalation_policy,
    tables: Mapping,
) -> window_records.FormationReads:
    """Which reads also hold the raw rounds their first round reads.

    The primary store feeds the primary tier's decoder, whose seat
    forms when detection_events.formed_at names it.
    """
    detection_events = settings.detection_events
    strong_side_seat = detection_events.strong_side_seat()
    primary_seat = "weak_decoder"
    if escalation_policy.primary_tier is window_records.DecoderTier.STRONG:
        primary_seat = "strong_decoder"
    primary_reader_forms = primary_seat in detection_events.formed_at
    return window_records.FormationReads(
        strong_side_seat=strong_side_seat,
        primary_reader_forms=primary_reader_forms,
        tables=tables,
    )


def _refuse_bulk_strong_without_a_merge(
    settings: machine_settings.MachineSettings,
    escalation_policy,
    device,
    error_model_provider,
) -> None:
    """bulk_strong merges strong re-decodes that carry timing alone.

    Only the strong pool of a switching run merges, so the key beside
    any other escalation is read by nothing. The merged decode reads no
    bits and returns no correction (decoder_manager.bulk_strong in
    configs/reference.yaml), so rounds that carry values, or windows
    whose models come from a provider other than the source's own,
    would be lost.
    """
    if not settings.decoder_manager.bulk_strong:
        return
    if not escalation_policy.requires_strong_context:
        raise ValueError(
            "decoder_manager.bulk_strong merges the strong pool's queued "
            f"re-decodes, and escalation.kind {settings.escalation.kind} "
            "has no strong pool; remove the key or run switching"
        )
    own_models = device.window_model_source()
    builds_models = error_model_provider is not own_models
    if not device.emits_bit_values and not builds_models:
        return
    raise ValueError(
        "decoder_manager.bulk_strong merges timing-only strong re-decodes, "
        f"and qpu.kind {settings.qpu.kind} gives the decoders bits and "
        "models the merged decode would drop; set bulk_strong false or "
        "qpu.kind timing_only"
    )


def _install_operation_circuits(device, model_provider, operations) -> None:
    """Copy the root-owned circuit once when either consumer requires it."""
    source_scope = _operation_circuit_scope(device, "syndrome source")
    model_scope = _operation_circuit_scope(model_provider, "model provider")
    needs_circuit = "per_operation" in (source_scope, model_scope)
    for operation in operations:
        if not needs_circuit:
            operation.circuit = None
            continue
        if operation.circuit is None:
            continue
        circuit_text = str(operation.circuit)
        operation.circuit = stim.Circuit(circuit_text)


def _operation_circuit_scope(component, role):
    scope = component.operation_circuit_scope
    if scope not in ("none", "per_operation"):
        raise ValueError(
            f"{role} operation_circuit_scope must be none or per_operation"
        )
    return scope
