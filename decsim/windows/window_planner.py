"""The window planner: which windows exist, planned or grown.

The static plan (frontends/planner.py) lays out every window of an
operation of known length up front; a dynamic stream's windows are laid
out here as its rounds arrive, the runtime twin of that plan. The
commit and buffer regions are the planner's only geometry: a window
reads [start_round, buffer_hi] and commits [commit_lo, commit_hi]
(Skoric et al. 2209.08552, the overlapping recovery method; Tan et al.
2209.09219, core and buffer regions). qLDPC separates the window plan
from the decode loop the same way (qldpc/decoders/sinter.py,
SlidingWindowDecoder.compile_decoder_for_dem and
CompiledSequentialWindowDecoder.decode_shots_to_error); readiness is
the RoundTracker's.
"""

import types
from typing import Optional

import decsim.message as message


class WindowModels:
    """The decoder-facing models of windows and streams, from the source.

    A run whose source gives no models (a timing-only device) has none;
    every question answers None or nothing.
    """

    def __init__(self, error_model_provider, fault_model_requirement_for):
        self.provider = error_model_provider
        self.fault_model_requirement_for = fault_model_requirement_for

    def has_provider(self) -> bool:
        """Whether the source builds models at all."""
        return self.provider is not None

    def requirement_for(self, resolved_operation):
        """The decoder views for the operation's frozen code."""
        code_name = resolved_operation.code_geometry.code_name
        return self.fault_model_requirement_for(code_name)

    def models_for_operation(
        self, operation, resolved_operation, windows: list, protocol
    ) -> list:
        """One model per window of a planned operation, or none."""
        if self.provider is None:
            return []
        requirement = self.requirement_for(resolved_operation)
        return self.provider.window_models_for_operation(
            operation,
            windows,
            resolved_operation.round_count,
            fault_model_requirement=requirement,
            fault_exclusion_ranges=(),
            window_protocol=protocol,
        )

    def model_for_stream(self, stream_id, window: message.Window):
        """The model of one window of a dynamic stream, or None."""
        if self.provider is None:
            return None
        return self.provider.window_model_for_stream(stream_id, window)

    def register_stream(self, stream_operation, resolved_operation):
        """Note a dynamic stream; the rounds its source can supply, or None."""
        if self.provider is None:
            return None
        requirement = self.requirement_for(resolved_operation)
        return self.provider.register_dynamic_stream(
            stream_operation,
            resolved_operation.round_count,
            fault_model_requirement=requirement,
        )

    def check_stream_length(
        self, stream_operation, stream_round_count: int
    ) -> None:
        """Refuse a stream longer than its source can supply."""
        if self.provider is None:
            return
        self.provider.validate_stream_length(
            stream_operation, stream_round_count
        )

    def strong_model_for_operation(
        self,
        operation,
        resolved_operation,
        window: message.Window,
        round_count: int,
        fault_exclusion_ranges: tuple,
    ):
        """The model of one strong window, with the listed faults excluded."""
        if self.provider is None:
            return None
        requirement = self.requirement_for(resolved_operation)
        if len(fault_exclusion_ranges) <= 1:
            exclusion = None
            if fault_exclusion_ranges:
                exclusion = fault_exclusion_ranges[0]
            return self.provider.strong_window_model_for_operation(
                operation,
                window,
                round_count,
                fault_model_requirement=requirement,
                exclude_faults_touching=exclusion,
            )
        return self.provider.strong_window_model_for_operation_with_exclusions(
            operation,
            window,
            round_count,
            fault_model_requirement=requirement,
            fault_exclusion_ranges=fault_exclusion_ranges,
        )


class WindowPlanner:
    """Which windows exist: the plan's, and a stream's as it grows."""

    def __init__(
        self,
        scheme,
        resolved_operations,
        plan: message.WindowPlan,
        models: WindowModels,
        planned_operations,
    ) -> None:
        self.scheme = scheme
        resolved_by_id = {
            resolved.operation_id: resolved for resolved in resolved_operations
        }
        self.resolved_operation_by_id = types.MappingProxyType(resolved_by_id)
        self.plan = plan
        self.models = models
        self.model_by_window: dict = {}
        self.growth_by_stream: dict = {}
        for operation in planned_operations:
            self._build_operation_models(operation)

    # ---- the plan's tables

    @property
    def windows_by_key(self) -> dict:
        """Every window by (operation id, index)."""
        return self.plan.windows

    @property
    def successors_by_operation(self) -> dict:
        """The operations whose rounds a window's overflow may read."""
        return self.plan.successors

    @property
    def total_windows(self) -> int:
        """Every window planned so far, streams included."""
        return self.plan.total_windows

    def window_indices_of(self, operation_id) -> list:
        """The operation's window indices in order; none when unplanned."""
        return self.plan.op_windows.get(operation_id, [])

    def window_count_of(self, operation_id) -> int:
        """How many windows the operation has."""
        return self.plan.window_count[operation_id]

    def windows_of(self, operation_id) -> list:
        """The operation's windows, in index order."""
        windows = []
        for window_index in self.window_indices_of(operation_id):
            windows.append(self.plan.windows[(operation_id, window_index)])
        return windows

    def is_windowed(self, operation_id) -> bool:
        """False for an operation decoded as one whole batch."""
        return self.plan.windowed_by_operation[operation_id]

    def batches_idle_rounds(self, operation_id) -> bool:
        """Whether idle rounds before the operation fold into its batch."""
        return self.plan.batch_preceding_idle_rounds_by_operation.get(
            operation_id, False
        )

    def prepend_idle_rounds(self, operation_id, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation."""
        if round_count <= 0:
            return
        if not self.batches_idle_rounds(operation_id):
            return
        window = self.plan.windows[(operation_id, 0)]
        window.batched_preceding_idle_round_count += round_count

    # ---- the resolved operations

    def round_count_of(self, operation_id) -> int:
        """The root-resolved round count of an operation."""
        return self.resolved_operation_by_id[operation_id].round_count

    def code_geometry_of(self, operation_id):
        """The resolved code geometry of an operation."""
        return self.resolved_operation_by_id[operation_id].code_geometry

    def spatial_node_count_of(self, operation_id) -> int:
        """The decoding-graph nodes per round of an operation."""
        return self.resolved_operation_by_id[operation_id].spatial_node_count

    # ---- a dynamic stream: registered, grown, clipped

    def register_stream(self, stream_operation: message.Operation):
        """Open a stream's tables; returns the source's round limit or None."""
        stream_id = stream_operation.id
        self.plan.window_count[stream_id] = 0
        self.plan.op_windows[stream_id] = []
        self.plan.successors.setdefault(stream_id, [])
        self.plan.windowed_by_operation[stream_id] = True
        self.plan.batch_preceding_idle_rounds_by_operation[stream_id] = False
        resolved = self.resolved_operation_by_id[stream_id]
        source_round_limit = self.models.register_stream(
            stream_operation, resolved
        )
        geometry = resolved.code_geometry
        finite_geometries = None
        if source_round_limit is not None:
            finite_plan = self.scheme.plan_operation(
                stream_id,
                source_round_limit,
                commit_round_count=geometry.commit_round_count,
                buffer_round_count=geometry.buffer_round_count,
            )
            finite_geometries = finite_plan.windows
        self.growth_by_stream[stream_id] = _StreamGrowth(
            geometry.commit_round_count,
            geometry.buffer_round_count,
            finite_geometries,
        )
        return source_round_limit

    def has_stream(self, stream_id) -> bool:
        """True for a stream whose windows are planned at runtime."""
        return stream_id in self.growth_by_stream

    def is_finite_stream(self, stream_id) -> bool:
        """True when the source fixed the stream's window plan up front."""
        return self.growth_by_stream[stream_id].finite_geometries is not None

    def grow_stream(
        self,
        stream_id,
        highest_known_round: int,
        round_cap: Optional[int],
    ) -> list:
        """Create every window whose commit region has begun.

        round_cap clips an open stream's commit ends once its length is
        known. The new windows are returned in index order, in the plan's
        tables and without their model (attach_stream_model), so the
        caller can link each one's boundary before the model exists.
        """
        growth = self.growth_by_stream[stream_id]
        if growth.finite_geometries is not None:
            geometries = growth.finite_geometries_begun(highest_known_round)
        else:
            geometries = growth.arithmetic_geometries_begun(
                highest_known_round, round_cap
            )
        created = []
        for geometry in geometries:
            window = self._create_stream_window(stream_id, growth, geometry)
            created.append(window)
        return created

    def attach_stream_model(self, window: message.Window) -> None:
        """Give a new stream window its model, when the source has one."""
        model = self.models.model_for_stream(window.op_id, window)
        if model is not None:
            self.model_by_window[window.key] = model

    def trim_stream_tail(
        self, stream_id, stream_round_count: int
    ) -> Optional[message.Window]:
        """Clip the one window whose commit region holds the sealed length.

        Its buffer follows the new commit end; the window is returned so
        its holds can shrink with it. None when no window is clipped.
        """
        growth = self.growth_by_stream[stream_id]
        for window in self.windows_of(stream_id):
            if window.commit_lo <= stream_round_count <= window.commit_hi:
                window.commit_hi = stream_round_count
                window.buffer_hi = stream_round_count + growth.buffer_rounds
                window.n_rounds = window.buffer_hi - window.start_round + 1
                return window
        return None

    # ---- private

    def _build_operation_models(self, operation: message.Operation) -> None:
        windows = self.windows_of(operation.id)
        if not windows:
            return
        resolved = self.resolved_operation_by_id[operation.id]
        protocol = self.plan.protocol_by_operation.get(
            operation.id, message.WindowProtocol.GENERIC
        )
        models = self.models.models_for_operation(
            operation, resolved, windows, protocol
        )
        if not models:
            return
        for window, model in zip(windows, models):
            self.model_by_window[window.key] = model

    def _create_stream_window(
        self, stream_id, growth: "_StreamGrowth", geometry
    ) -> message.Window:
        window_index = growth.next_window_index
        buffer_lo = geometry.commit_lo
        round_count = geometry.buffer_hi - buffer_lo + 1
        window = message.Window(
            op_id=stream_id,
            k=window_index,
            commit_lo=geometry.commit_lo,
            commit_hi=geometry.commit_hi,
            buffer_hi=geometry.buffer_hi,
            n_rounds=round_count,
            buffer_lo=buffer_lo,
        )
        self.plan.windows[window.key] = window
        self.plan.op_windows[stream_id].append(window_index)
        self.plan.window_count[stream_id] += 1
        self.plan.total_windows += 1
        growth.next_window_index += 1
        return window


class _StreamGrowth:
    """One stream's growth: its region sizes and the next window to lay."""

    def __init__(
        self, commit_rounds: int, buffer_rounds: int, finite_geometries
    ) -> None:
        self.commit_rounds = commit_rounds
        self.buffer_rounds = buffer_rounds
        self.finite_geometries = finite_geometries
        self.next_window_index = 0

    def finite_geometries_begun(self, highest_known_round: int) -> list:
        """The source's next geometries whose commit region has begun."""
        begun = []
        geometry_count = len(self.finite_geometries)
        index = self.next_window_index
        while index < geometry_count:
            geometry = self.finite_geometries[index]
            if geometry.commit_lo > highest_known_round:
                break
            begun.append(geometry)
            index += 1
        return begun

    def arithmetic_geometries_begun(
        self, highest_known_round: int, round_cap: Optional[int]
    ) -> list:
        """Window k commits [k*ncom+1, (k+1)*ncom], clipped at the cap."""
        begun = []
        index = self.next_window_index
        while True:
            commit_lo = index * self.commit_rounds + 1
            if commit_lo > highest_known_round:
                return begun
            commit_hi = (index + 1) * self.commit_rounds
            if round_cap is not None:
                commit_hi = min(commit_hi, round_cap)
            buffer_hi = commit_hi + self.buffer_rounds
            geometry = message.WindowGeometry(
                buffer_lo=commit_lo,
                commit_lo=commit_lo,
                commit_hi=commit_hi,
                buffer_hi=buffer_hi,
            )
            begun.append(geometry)
            index += 1
