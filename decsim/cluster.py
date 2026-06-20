"""Public decoder cluster coordinator.

The coordinator exposes one entry point while window state stays in
WindowManager and decoder queues stay in DecoderManager.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from .codes import SurfaceCodeModel
from .decoder_manager import DecoderManager
from .engine import Engine
from .layouts import UniformLayout
from .links import LinkModel
from .message import Operation, SyndromePayload, WindowPlan
from .planner import FixedRounds
from .schemes import SlidingWindowScheme
from .schedulers import EnqueueTimeDeadline
from .window_manager import WindowManager

if TYPE_CHECKING:
    from .protocols import (CodeModel, Controller, DeadlinePolicy, Decoder,
                            DecoderRouter, DecodingScheme, LayoutModel,
                            Orchestrator, RoundsPolicy, Scheduler)
    from .switching import Switching


class DecoderCluster:
    """Stable public entry point for chip, controller, metrics, and tests."""

    def __init__(self, engine: Engine, decoder: "Decoder", scheduler: "Scheduler",
                 controller: "Controller", orchestrator: "Orchestrator",
                 num_units: int, code_distance: Optional[int] = None,
                 rounds_per_op: int = 11, *, code: Optional["CodeModel"] = None,
                 scheme: Optional["DecodingScheme"] = None,
                 layout: Optional["LayoutModel"] = None,
                 decoders: Optional[dict] = None,
                 rounds_policy: Optional["RoundsPolicy"] = None,
                 router: Optional["DecoderRouter"] = None,
                 deadline_policy: Optional["DeadlinePolicy"] = None,
                 links: Optional[LinkModel] = None,
                 unit_pools: Optional[dict] = None,
                 switching: Optional["Switching"] = None):
        self.engine = engine
        self.decoder = decoder
        self.decoders = dict(decoders) if decoders else {}
        self.scheduler = scheduler
        self.orchestrator = orchestrator
        self.links = links if links is not None else LinkModel()

        self._configure_code_context(code, layout, code_distance, scheme)
        self.rounds_policy = rounds_policy if rounds_policy is not None \
            else FixedRounds(rounds_per_op)
        self.deadline_policy = deadline_policy if deadline_policy is not None \
            else EnqueueTimeDeadline()
        self.switching = switching

        self.decoder_manager = self._build_decoder_manager(
            num_units, unit_pools, router, switching)
        self.window_manager = self._build_window_manager(orchestrator)
        self.on_workload_complete = None

    def _configure_code_context(self, code, layout, code_distance, scheme) -> None:
        """Set the code, layout, scheme, and derived window sizes."""
        if code is None and layout is None:
            if code_distance is None:
                raise ValueError("provide code=<CodeModel>, layout=<LayoutModel>, "
                                 "or code_distance=<int>")
            code = SurfaceCodeModel(d=code_distance)

        self.layout = UniformLayout(code) if layout is None else layout
        self.code = code if code is not None else self.layout.codes()[0]
        self.scheme = scheme if scheme is not None else SlidingWindowScheme()
        self.d = self.code.distance
        self.commit = self.code.commit_rounds()
        self.buffer = self.code.buffer_rounds()

    def _build_decoder_manager(self, num_units, unit_pools, router, switching):
        """Create the decoder queue and unit-pool manager."""
        return DecoderManager(
            self.engine, self.decoder, self.scheduler, decoders=self.decoders,
            router=router, links=self.links, num_units=num_units,
            unit_pools=unit_pools, switching=switching,
            on_window_decoded=lambda job, result:
            self.window_manager.on_decode_done(job, result))

    def _build_window_manager(self, orchestrator):
        """Create the syndrome-window runtime manager."""
        return WindowManager(
            self.engine, scheme=self.scheme, layout=self.layout,
            rounds_policy=self.rounds_policy, code=self.code,
            decoder_manager=self.decoder_manager,
            deadline_policy=self.deadline_policy,
            links=self.links, orchestrator=orchestrator)

    @property
    def on_workload_complete(self):
        return self.window_manager.on_workload_complete

    @on_workload_complete.setter
    def on_workload_complete(self, callback) -> None:
        self.window_manager.on_workload_complete = callback

    @property
    def ops(self) -> dict:
        return self.window_manager.ops

    @property
    def rounds_arrived(self) -> dict:
        return self.window_manager.rounds_arrived

    @property
    def memory_rounds(self) -> dict:
        return self.window_manager.memory_rounds

    @property
    def payload_store(self) -> dict:
        return self.window_manager.payload_store

    @property
    def payloads_held(self) -> int:
        return self.window_manager.payloads_held

    @property
    def peak_payloads(self) -> int:
        return self.window_manager.peak_payloads

    @property
    def windows(self) -> dict:
        return self.window_manager.windows

    @property
    def op_windows(self) -> dict:
        return self.window_manager.op_windows

    @property
    def window_count(self) -> dict:
        return self.window_manager.window_count

    @property
    def successors(self) -> dict:
        return self.window_manager.successors

    @property
    def committed_windows(self) -> set:
        return self.window_manager.committed_windows

    @property
    def _committed_per_op(self) -> dict:
        return self.window_manager._committed_per_op

    @property
    def op_results(self) -> dict:
        return self.window_manager.op_results

    @property
    def window_models(self) -> dict:
        return self.window_manager.window_models

    @property
    def total_windows(self) -> int:
        return self.window_manager.total_windows

    @property
    def _windows_built(self) -> bool:
        return self.window_manager._windows_built

    @property
    def _round_refs(self) -> dict:
        return self.window_manager._round_refs

    @property
    def _read_sets(self) -> dict:
        return self.window_manager._read_sets

    def register_op(self, op: Operation) -> None:
        self.window_manager.register_op(op)

    def register_dynamic_stream(self, stream_op: Operation, code) -> None:
        self.window_manager.register_dynamic_stream(stream_op, code)

    def grow_stream(self, stream_id) -> None:
        self.window_manager.grow_stream(stream_id)

    def seal_stream(self, stream_id, total_rounds: int) -> None:
        self.window_manager.seal_stream(stream_id, total_rounds)

    def rounds_for(self, op: Operation) -> int:
        return self.window_manager.rounds_for(op)

    def load_execution_plan(self, plan: WindowPlan) -> None:
        self.window_manager.load_execution_plan(plan)

    def build_windows(self) -> None:
        self.window_manager.build_windows()

    def on_syndrome_arrival(self, payload: SyndromePayload) -> None:
        self.window_manager.on_syndrome_arrival(payload)

    def prepend_idle_rounds(self, op_id: int, round_count: int) -> None:
        self.window_manager.prepend_idle_rounds(op_id, round_count)

    def on_memory_round(self, op_id: int) -> None:
        self.window_manager.on_memory_round(op_id)

    @property
    def free_units(self) -> int:
        return self.decoder_manager.free_units

    @property
    def unit_totals(self) -> dict:
        return self.decoder_manager.unit_totals

    @property
    def pool_free(self) -> dict:
        return self.decoder_manager.pool_free

    @property
    def num_units(self) -> int:
        return self.decoder_manager.num_units

    @property
    def ready(self) -> list:
        return self.decoder_manager.ready

    @property
    def pool_ready(self) -> dict:
        return self.decoder_manager.pool_ready

    @property
    def queue_log(self) -> list:
        return self.decoder_manager.queue_log

    @property
    def strong_needed(self) -> int:
        return self.decoder_manager.strong_needed

    @property
    def strong_cancelled(self) -> int:
        return self.decoder_manager.strong_cancelled

    @property
    def strong_running_rounds(self) -> int:
        return self.decoder_manager.strong_running_rounds

    def _pool_tag(self, pool: str) -> str:
        return self.decoder_manager.pool_tag(pool)

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "external", deadline: Optional[int] = None,
                      code: Optional[str] = None,
                      spatial_nodes: Optional[int] = None,
                      hint: Optional[str] = None) -> None:
        """Submit an external decode job that competes for the same decoder units."""
        self.decoder_manager.submit_decode(
            round_count, on_done, label=label, deadline=deadline, code=code,
            spatial_nodes=spatial_nodes, hint=hint)
