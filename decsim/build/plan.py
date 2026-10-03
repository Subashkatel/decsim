"""Build the run's plan: the code, the workload's operations, the windows.

Everything the wiring reads that the planner and the workload fix, laid
out once before any component is built. gem5's configuration scripts do
the same: configs/deprecated/example/se.py resolves the workload and the
system before it wires a single port.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional, Union

import stim

import decsim.build.escalation as escalation_build
import decsim.controller.policies as idle_policies
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as detection_event_settings
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
import decsim.windows.settings as window_settings
import decsim.windows.window_interactions as window_interactions


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
    qpu: qpu_settings.QpuSettings,
    workload: workload_settings.WorkloadSettings,
    windows: window_settings.WindowSettings,
    idle_policy_settings: Union[
        idle_policies.IgnoreSettings, idle_policies.SeparateDecodeJobsSettings
    ],
    detection_events: detection_event_settings.DetectionEventSettings,
    switching_settings: Optional[escalation_settings.SwitchingSettings],
    is_bulk_strong: bool,
    switching: Optional[escalation_build.Switching],
) -> Plan:
    """The code, the workload's operations and the window plan.

    Keep the root's run-shape and plan records together so their shared
    inputs remain visible; named helpers resolve individual collaborators.
    A switching run's policy refuses a run shape it cannot serve.
    is_bulk_strong is the decoder manager's bulk_strong.
    """
    is_switching = switching_settings is not None
    window_sizes = windows.scheme
    code, layout = qpu.build_code(
        commit_rounds_override=window_sizes.commit_rounds,
        buffer_rounds_override=window_sizes.buffer_rounds,
    )
    operations, decode_operations, dynamic_streams, rounds_policy = _operations(
        workload
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
    scheme = _scheme(windows)
    boundary_policy = _boundary_policy(windows)
    # the refusals print the strong window's row name, and only a window
    # that absorbs the weak windows restarts any, so a run with no
    # switching carries the default record, which absorbs none, unread
    strong_window = _strong_window_settings(switching_settings)
    absorbs_weak_windows = strong_window.absorbs_weak_windows
    strong_window_name = strong_window.name
    reread_regions = strong_window.restart_reread_buffer_regions
    window_interaction = _window_interaction(windows, reread_regions)
    if dynamic_streams and not scheme.supports_dynamic_streams:
        raise ValueError(
            "dynamic streams require a windowing scheme that supports them"
        )
    has_static_decode_plan = workload.decode_operations is not None
    commit_round_count = code.commit_rounds()
    buffer_round_count = code.buffer_rounds()
    run_shape = decoding_records.RunShape(
        scheme=scheme,
        boundary_policy=boundary_policy,
        operations=views,
        commit_round_count=commit_round_count,
        buffer_round_count=buffer_round_count,
        strong_window=strong_window_name,
        is_absorbing_strong_window=absorbs_weak_windows,
        is_bulk_strong=is_bulk_strong,
        has_dynamic_streams=bool(dynamic_streams),
        has_static_decode_plan=has_static_decode_plan,
    )
    if switching is not None:
        switching.policy.check_plan(run_shape)
    planned_operations = _decode_plan_operations(
        operations,
        decode_operations,
        dynamic_streams,
        static_decode_selected=has_static_decode_plan,
    )
    planned_ids = []
    for operation in planned_operations:
        planned_ids.append(operation.id)
    physical_circuits = workload.physical_circuits
    physical_tables = _physical_formation_tables(physical_circuits)
    circuit_arguments = _circuit_arguments(physical_circuits, physical_tables)
    device = qpu.source.build(code, circuit_arguments)
    formation_tables = _formation_tables(
        device, planned_operations, physical_tables, rounds_policy, code
    )
    # a strong read also holds the raw rounds its first round reads when
    # a seat past the weak syndrome buffer forms the events
    strong_side_seat = detection_events.strong_side_seat()
    formation_reads = window_records.FormationReads(
        strong_side_seat=strong_side_seat, tables=formation_tables
    )
    run_plan = planner.plan_execution(
        operations=views,
        planned_operation_ids=tuple(planned_ids),
        code=code,
        layout=layout,
        scheme=scheme,
        rounds_policy=rounds_policy,
        fallback_round_microseconds=qpu.round_period_microseconds,
        retain_strong_context=is_switching,
        absorbs_weak_windows=absorbs_weak_windows,
        restart_reread_buffer_regions=reread_regions,
        has_open_ended_dynamic_streams=bool(dynamic_streams),
        formation_reads=formation_reads,
    )
    resource_claims = _resource_claims(operations, view_by_id, layout)
    error_model_provider = _error_model_provider(
        qpu, device, code, circuit_arguments
    )
    _refuse_bulk_strong_without_a_merge(
        is_bulk_strong, is_switching, qpu.source, device, error_model_provider
    )
    installed_by_id = _installed_circuits(
        device, error_model_provider, all_operations
    )
    operations = _installed(operations, installed_by_id)
    decode_operations = _installed(decode_operations, installed_by_id)
    dynamic_streams = _installed(dynamic_streams, installed_by_id)
    all_operations = _installed(all_operations, installed_by_id)
    planned_operations = _installed(planned_operations, installed_by_id)
    idle_policy = idle_policy_settings.build()
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
        protected_regions=tuple(workload.protected_regions),
        all_operations=all_operations,
        planned_operations=planned_operations,
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
    boundary_payload = settings.boundary_payload.build()
    return window_interactions.DefaultWindowInteraction(
        reread_regions, boundary_payload
    )


def _operations(settings: workload_settings.WorkloadSettings) -> tuple:
    """The workload's operations, each with its mode, and its rounds policy.

    An operation without its own feedback boundary mode takes the
    workload's; one operation named in two roles stays one record.
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
    """The operations with their modes, one record per object identity."""
    copied = []
    for operation in operations:
        identity = id(operation)
        if identity not in copies:
            copies[identity] = _with_boundary_mode(operation, settings)
        copied.append(copies[identity])
    return tuple(copied)


def _with_boundary_mode(
    operation: program_records.Operation,
    settings: workload_settings.WorkloadSettings,
) -> program_records.Operation:
    mode = operation.feedback_boundary_mode
    if mode is None:
        mode = settings.feedback_boundary_mode
    return dataclasses.replace(operation, feedback_boundary_mode=mode)


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


def _strong_window_settings(
    switching: Optional[escalation_settings.SwitchingSettings],
):
    """The strong window's record; the default on a run with no switching."""
    if switching is None:
        return escalation_settings.SwitchingSettings.strong_window
    return switching.strong_window


def _scheme(windows: window_settings.WindowSettings):
    """The windowing scheme the section's record builds, with its tail.

    A switching run's policy refuses a scheme whose last window has no
    trailing tail context, the fact the terminal tail sets.
    """
    return windows.scheme.build(windows.terminal_policy)


def _boundary_policy(windows: window_settings.WindowSettings):
    """The section's boundary policy.

    A switching run's policy refuses the row its strong window cannot
    serve (escalation/policies.py check_plan).
    """
    return windows.boundary_policy.build()


def _error_model_provider(
    settings: qpu_settings.QpuSettings,
    device: ports.SyndromeSource,
    code: ports.CodeModel,
    circuit_arguments: Mapping,
) -> ports.WindowModelSource:
    """The window models' source: the run's own, or the one the qpu names.

    A named one is built over the same card and circuits as the run's
    source, and the decoders read its window models, not its rounds.
    """
    model_record = settings.error_model_provider
    if model_record is None:
        return device.window_model_source()
    return model_record.build(code, circuit_arguments)


def _circuit_arguments(
    physical_circuits: tuple, physical_tables: Mapping
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
    for key, physical in physical_circuits:
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


def _physical_formation_tables(physical_circuits: tuple) -> dict:
    """Each finite circuit's recipes, by the stream key that runs it.

    The source is declared each detector's round from this table, and
    the plan reads from it how many rounds before a read its first
    round's recipes reach.
    """
    tables = {}
    for key, physical in physical_circuits:
        if not isinstance(physical, workload_records.FiniteCircuit):
            continue
        tables[key] = _physical_formation_table(physical)
    return tables


def _physical_formation_table(
    physical: workload_records.FiniteCircuit,
) -> detector_formation.FormationTable:
    """The recipes, formed off the declared measurement schedule."""
    schedule = dict(physical.measurement_rounds)
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


def _refuse_bulk_strong_without_a_merge(
    is_bulk_strong: bool,
    is_switching: bool,
    source_settings: qpu_settings.SourceSettings,
    device: ports.SyndromeSource,
    error_model_provider: ports.WindowModelSource,
) -> None:
    """bulk_strong merges strong re-decodes that carry timing alone.

    Only the strong pool of a switching run merges, so the key beside
    any other escalation is read by nothing. The merged decode reads no
    bits and returns no correction (decoder_manager.bulk_strong in
    configs/reference.yaml), so rounds that carry values, or windows
    whose models come from a provider other than the source's own,
    would be lost.
    """
    if not is_bulk_strong:
        return
    if not is_switching:
        raise ValueError(
            "decoder_manager.bulk_strong merges the strong pool's queued "
            "re-decodes, and a run with no switching has no strong pool; "
            "remove the key or run switching"
        )
    own_models = device.window_model_source()
    builds_models = error_model_provider is not own_models
    if not device.emits_bit_values and not builds_models:
        return
    source_name = source_settings.name
    raise ValueError(
        "decoder_manager.bulk_strong merges timing-only strong re-decodes, "
        f"and qpu.kind {source_name} gives the decoders bits and "
        "models the merged decode would drop; set bulk_strong false or "
        "qpu.kind timing_only"
    )


def _installed_circuits(device, model_provider, operations) -> dict:
    """Each operation by id, its circuit a copy when either consumer reads it.

    A run whose source and model provider read no operation circuit
    carries none; otherwise each circuit is a copy of the settings' own,
    so the shot holds no circuit another shot shares.
    """
    source_scope = _operation_circuit_scope(device, "syndrome source")
    model_scope = _operation_circuit_scope(model_provider, "model provider")
    needs_circuit = "per_operation" in (source_scope, model_scope)
    installed_by_id = {}
    for operation in operations:
        circuit = None
        if needs_circuit and operation.circuit is not None:
            circuit_text = str(operation.circuit)
            circuit = stim.Circuit(circuit_text)
        installed = dataclasses.replace(operation, circuit=circuit)
        installed_by_id[operation.id] = installed
    return installed_by_id


def _installed(operations: tuple, installed_by_id: dict) -> tuple:
    installed = []
    for operation in operations:
        installed_operation = installed_by_id[operation.id]
        installed.append(installed_operation)
    return tuple(installed)


def _operation_circuit_scope(component, role):
    scope = component.operation_circuit_scope
    if scope not in ("none", "per_operation"):
        raise ValueError(
            f"{role} operation_circuit_scope must be none or per_operation"
        )
    return scope
