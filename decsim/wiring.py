"""Build the standard simulator pipeline and run it."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from .chip import Chip
from .cluster import DecoderCluster
from .codes import SurfaceCodeModel
from .config import FEEDBACK_BOUNDARY_MODES, IDLE_ROUND_MODES, SimConfig, fmt, us
from .devices import TimingOnlyDevice
from .engine import Engine
from .factories import InfiniteFactory
from .metrics import DecodeBacklog
from .orchestrators import ExecutionOrchestrator
from .planner import WindowPlanner
from .schedulers import FifoScheduler

if TYPE_CHECKING:
    from .message import Operation
    from .protocols import (CodeModel, Controller, DeadlinePolicy, Decoder,
                            DecoderRouter, DecodingScheme,
                            ExecutionPlanner, InputFrontend, LayoutModel,
                            MagicStateFactory, Orchestrator, RoundsPolicy,
                            Scheduler, SyndromeSource)
    from .switching import Switching

def _print_title(title: str) -> None:
    """Print the optional run title."""
    if not title:
        return
    print("=" * 78)
    print(title)
    print("=" * 78)


def _apply_config_defaults(config: Optional[SimConfig], num_units: Optional[int],
                           rounds_per_op: Optional[int], round_us: Optional[float],
                           scheme, switching):
    """Resolve scalar defaults from SimConfig while preserving explicit arguments."""
    sim_config = config or SimConfig()
    if num_units is None:
        num_units = sim_config.num_units
    if rounds_per_op is None:
        rounds_per_op = sim_config.rounds_per_op
    if round_us is None:
        round_us = sim_config.round_us
    if scheme is None:
        scheme = sim_config.make_scheme()
    if switching is None:
        switching = sim_config.make_switching()
    return sim_config, num_units, rounds_per_op, round_us, scheme, switching


def _operation_list(ops, frontend):
    """Return operations from a frontend or from the explicit operation list."""
    if frontend is not None:
        return frontend.build()
    if ops is None:
        raise ValueError("provide either ops=<list[Operation]> or frontend=<InputFrontend>")
    return ops


def _code_model(code, layout, distance: int):
    """Return the code model used to size default components."""
    if code is not None:
        return code
    if layout is not None:
        return layout.codes()[0]
    return SurfaceCodeModel(d=distance)


def _build_controller(engine: Engine, sim_config: SimConfig, controller, make_controller):
    """Build or reuse the controller."""
    if make_controller is not None:
        return make_controller(engine)
    if controller is not None:
        return controller
    return sim_config.make_controller(engine)


def _build_orchestrator(engine: Engine, orchestrator, make_orchestrator):
    """Build or reuse the feedback orchestrator."""
    if make_orchestrator is not None:
        return make_orchestrator(engine)
    if orchestrator is not None:
        return orchestrator
    return ExecutionOrchestrator(engine)


def _link_model(controller, sim_config: SimConfig):
    """Use the controller's link model, or the config default."""
    links = getattr(controller, "links", None)
    return links if links is not None else sim_config.make_links()


def _build_cluster(*, engine: Engine, decoder, scheduler, controller, orchestrator,
                   make_cluster, num_units: int, rounds_per_op: int, code,
                   scheme, layout, decoders, rounds_policy, router,
                   deadline_policy, links, unit_pools, switching,
                   syndrome_source):
    """Build the decoder workload manager."""
    if make_cluster is not None:
        return make_cluster(engine, decoder, scheduler, controller, orchestrator)

    return DecoderCluster(
        engine, decoder, scheduler, controller, orchestrator,
        num_units=num_units, rounds_per_op=rounds_per_op, code=code,
        scheme=scheme, layout=layout, decoders=decoders,
        rounds_policy=rounds_policy, router=router,
        deadline_policy=deadline_policy, links=links,
        unit_pools=unit_pools, switching=switching,
        syndrome_source=syndrome_source)


def _build_factory(engine: Engine, cluster, factory, make_factory):
    """Build or reuse the magic-state factory."""
    if factory is not None:
        return factory
    if make_factory is not None:
        return make_factory(engine, cluster)
    return InfiniteFactory(engine)


def _build_chip(*, engine: Engine, device, controller, cluster, factory,
                round_us: float, code, make_chip, idle_round_mode: str,
                max_idle_rounds: Optional[int],
                gates_start_on_round_boundaries: bool):
    """Build the quantum processor."""
    round_ticks = us(round_us)
    if make_chip is not None:
        return make_chip(engine, device, controller, cluster, factory, round_ticks, code)

    return Chip(
        engine, device, controller, cluster, factory,
        round_ticks=round_ticks,
        code_distance=code.distance,
        idle_round_mode=idle_round_mode,
        max_idle_rounds=max_idle_rounds,
        gates_start_on_round_boundaries=gates_start_on_round_boundaries)


def _add_metrics(engine: Engine, cluster, chip, factory, make_metrics, verbose: bool):
    """Add optional metrics and return the verbose backlog metric."""
    if make_metrics is not None:
        for metric in make_metrics(engine, cluster, chip, factory):
            engine.add_metric(metric)

    backlog_metric = DecodeBacklog(cluster) if verbose else None
    if backlog_metric is not None:
        engine.add_metric(backlog_metric)
    return backlog_metric


def _register_feedback_blocks(ops, orchestrator) -> None:
    """Tell the orchestrator which operations wait for earlier decode results."""
    for op in ops:
        if op.blocked_by is None:
            continue
        orchestrator.register_blocked_operation(
            blocked_op_id=op.id,
            blocking_op_id=op.blocked_by)


def _execution_planner(planner, cluster):
    """Return the supplied planner or the default planner for this cluster."""
    if planner is not None:
        return planner
    return WindowPlanner(cluster.scheme, cluster.layout, cluster.rounds_policy)


def _register_dynamic_streams(cluster, dynamic_streams, code) -> None:
    """Register streams whose windows are built at runtime."""
    for stream in (dynamic_streams or []):
        cluster.register_dynamic_stream(stream, code)


def _resolve_idle_round_mode(sim_config: SimConfig,
                             idle_round_mode: Optional[str]) -> str:
    """Return the explicit idle-round mode used by the chip."""
    if idle_round_mode is None:
        return sim_config.idle_round_mode
    if idle_round_mode not in IDLE_ROUND_MODES:
        raise ValueError(
            f"idle_round_mode must be one of {IDLE_ROUND_MODES} "
            f"(got {idle_round_mode!r})")
    return idle_round_mode


def _resolve_feedback_boundary_mode(sim_config: SimConfig,
                                    feedback_boundary_mode: Optional[str]) -> str:
    """Return the default feedback-boundary mode for operations without an override."""
    if feedback_boundary_mode is None:
        return sim_config.feedback_boundary_mode
    if feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
        raise ValueError(
            f"feedback_boundary_mode must be one of {FEEDBACK_BOUNDARY_MODES} "
            f"(got {feedback_boundary_mode!r})")
    return feedback_boundary_mode


def _apply_feedback_boundary_default(ops, decode_ops, dynamic_streams,
                                     feedback_boundary_mode: str) -> None:
    """Fill the boundary mode on operations that did not choose one explicitly."""
    operations = list(ops) + list(decode_ops or []) + list(dynamic_streams or [])
    for operation in operations:
        if operation.feedback_boundary_mode is None:
            operation.feedback_boundary_mode = feedback_boundary_mode


def _decode_plan_operations(ops, decode_ops, dynamic_streams):
    """Return operations that should receive decode windows before runtime."""
    if decode_ops is not None:
        return decode_ops
    if not dynamic_streams:
        return None

    dynamic_stream_ids = {stream.id for stream in dynamic_streams}
    return [
        operation
        for operation in ops
        if operation.stream_id not in dynamic_stream_ids
    ]


def _print_summary(*, num_units: int, chip_done: int, last_event: int,
                   cluster, factory, backlog_metric) -> None:
    """Print the verbose run summary."""
    print("-" * 78)
    print(f"SUMMARY ({num_units} decoder unit(s)):")
    print(f"  chip finished all physical work : {fmt(chip_done)}")
    print(f"  decoder fully finished          : {fmt(last_event)}")
    print(f"  reaction tail (chip->fully done): {fmt(last_event - chip_done)}")

    peak_queue = max((q for _, q in getattr(cluster, "queue_log", [])), default=0)
    print(f"  peak ready-queue (contention)   : {peak_queue}")

    if backlog_metric is not None:
        backlog = backlog_metric.result()
        print(f"  peak decode backlog (rounds)    : {backlog['peak_rounds']}")
        print(f"  decode backlog, mean (rounds)   : {backlog['time_avg_rounds']:.1f}")

    print(f"  peak syndrome RAM (payloads)    : {getattr(cluster, 'peak_payloads', 0)}")

    if isinstance(getattr(factory, "produced", None), int):
        print(f"  magic states produced           : {factory.produced}")
        print(f"  peak magic states in storage    : {factory.peak_in_flight}")
        print(f"  total magic-state supply stall  : {fmt(factory.total_stall)}")
    print()


def build_and_run(ops: Optional[list[Operation]] = None, num_units: Optional[int] = None,
                  d: int = 3, rounds_per_op: Optional[int] = None,
                  rounds_policy: Optional["RoundsPolicy"] = None,
                  round_us: Optional[float] = None,
                  factory: Optional[MagicStateFactory] = None,
                  make_factory: Optional[Callable[[Engine, "DecoderCluster"],
                                                   MagicStateFactory]] = None,
                  decoder: Optional[Decoder] = None,
                  decoders: Optional[dict] = None,
                  controller: Optional[Controller] = None,
                  make_controller: Optional[Callable[[Engine], "Controller"]] = None,
                  orchestrator: Optional[Orchestrator] = None,
                  make_orchestrator: Optional[Callable[[Engine], "Orchestrator"]] = None,
                  scheduler: Optional[Scheduler] = None,
                  router: Optional["DecoderRouter"] = None,
                  deadline_policy: Optional["DeadlinePolicy"] = None,
                  idle_round_mode: Optional[str] = None,
                  max_idle_rounds: Optional[int] = None,
                  gates_start_on_round_boundaries: bool = False,
                  unit_pools: Optional[dict] = None,
                  device: Optional["SyndromeSource"] = None,
                  make_cluster: Optional[Callable] = None,
                  planner: Optional["ExecutionPlanner"] = None,
                  make_chip: Optional[Callable] = None,
                  make_metrics: Optional[Callable] = None,
                  code: Optional[CodeModel] = None,
                  scheme: Optional["DecodingScheme"] = None,
                  switching: Optional["Switching"] = None,
                  layout: Optional["LayoutModel"] = None,
                  frontend: Optional["InputFrontend"] = None,
                  config: Optional[SimConfig] = None,
                  feedback_boundary_mode: Optional[str] = None,
                  decode_ops: Optional[list] = None,
                  dynamic_streams: Optional[list] = None,
                  verbose: bool = True, title: str = "") -> dict:
    """Assemble the standard simulator pipeline, run it, and return the live objects."""
    engine = Engine(verbose=verbose)
    _print_title(title)

    sim_config, num_units, rounds_per_op, round_us, scheme, switching = \
        _apply_config_defaults(
            config, num_units, rounds_per_op, round_us, scheme, switching)
    idle_round_mode = _resolve_idle_round_mode(sim_config, idle_round_mode)
    feedback_boundary_mode = _resolve_feedback_boundary_mode(
        sim_config, feedback_boundary_mode)

    ops = _operation_list(ops, frontend)
    _apply_feedback_boundary_default(
        ops, decode_ops, dynamic_streams, feedback_boundary_mode)
    code = _code_model(code, layout, d)

    device = device if device is not None else TimingOnlyDevice()
    decoder = decoder if decoder is not None else sim_config.make_decoder(code)
    controller = _build_controller(engine, sim_config, controller, make_controller)
    orchestrator = _build_orchestrator(engine, orchestrator, make_orchestrator)
    scheduler = scheduler if scheduler is not None else FifoScheduler()
    links = _link_model(controller, sim_config)

    cluster = _build_cluster(
        engine=engine, decoder=decoder, scheduler=scheduler,
        controller=controller, orchestrator=orchestrator,
        make_cluster=make_cluster, num_units=num_units,
        rounds_per_op=rounds_per_op, code=code, scheme=scheme,
        layout=layout, decoders=decoders, rounds_policy=rounds_policy,
        router=router, deadline_policy=deadline_policy, links=links,
        unit_pools=unit_pools, switching=switching,
        syndrome_source=device)

    factory = _build_factory(engine, cluster, factory, make_factory)
    chip = _build_chip(
        engine=engine, device=device, controller=controller, cluster=cluster,
        factory=factory, round_us=round_us, code=code, make_chip=make_chip,
        idle_round_mode=idle_round_mode, max_idle_rounds=max_idle_rounds,
        gates_start_on_round_boundaries=gates_start_on_round_boundaries)

    orchestrator.connect(controller, chip.on_decision)
    # One software Pauli frame across the loop: the orchestrator accumulates decoded
    # Clifford corrections into it and the chip folds feed-forward byproducts into
    # the same object when it consumes a decision (no-op for custom orchestrators
    # without a real frame).
    orchestrator_frame = getattr(orchestrator, "frame", None)
    if orchestrator_frame is not None and hasattr(chip, "frame"):
        chip.frame = orchestrator_frame
    cluster.on_workload_complete = factory.shutdown

    backlog_metric = _add_metrics(
        engine, cluster, chip, factory, make_metrics, verbose)
    _register_feedback_blocks(ops, orchestrator)
    planner = _execution_planner(planner, cluster)
    decode_plan_operations = _decode_plan_operations(
        ops, decode_ops, dynamic_streams)
    orchestrator.prepare_execution(
        operations=ops, cluster=cluster, planner=planner,
        decode_operations=decode_plan_operations)
    _register_dynamic_streams(cluster, dynamic_streams, code)

    chip.load(ops)
    engine.run()

    chip_done = chip.last_finish_time
    last_event = engine.now
    if verbose:
        _print_summary(num_units=num_units, chip_done=chip_done,
                       last_event=last_event, cluster=cluster,
                       factory=factory, backlog_metric=backlog_metric)
    return {"engine": engine, "cluster": cluster, "factory": factory, "chip": chip,
            "orchestrator": orchestrator, "controller": controller,
            "chip_done": chip_done, "fully_done": last_event,
            "metrics": engine.metric_results()}
