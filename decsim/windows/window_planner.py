"""The window planner: which windows exist, planned or grown.

The static plan (frontends/planner.py) lays every window of an operation
of known length up front; a dynamic stream's windows are laid here as
its rounds arrive. A window reads [start_round, buffer_hi] and commits
[commit_lo, commit_hi] (Skoric et al. 2209.08552, the overlapping
recovery method; Tan et al. 2209.09219, core and buffer regions). As in
qLDPC, the plan is apart from the decode loop (qldpc/decoders/sinter.py,
SlidingWindowDecoder); readiness is the RoundTracker's.
"""

import dataclasses
import types
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.fault_model_contracts as fault_models
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
import decsim.windows.built_window_models as built_window_models
import decsim.windows.window_positions as window_positions


class WindowModels:
    """The decoder-facing window models, from the source.

    They cover the planned windows and those a dynamic stream lays. A run
    whose source gives no models (a timing-only device) has none; every
    question answers None or nothing.
    """

    # a circuit-less source names one that answers every model with None
    provider = ports.Port(ports.WindowModelSource, optional=True)
    # the decoder of the pool that decodes the plan's windows; a model
    # is also built for each pool a window may escalate to
    # (escalates_to)
    decoder = ports.Port(ports.Decoder)

    def __init__(
        self, built_models: built_window_models.BuiltWindowModels
    ) -> None:
        self.built_models = built_models
        # the model each planned or grown window was laid with; the
        # models object builds them, so it is where they live
        self.model_by_window: dict = {}
        # the decoder of each pool an escalate edge leads to, in graph
        # order
        self.target_decoders: list = []

    def escalates_to(self, decoder: ports.Decoder) -> None:
        """A window may be decoded again by this decoder: models serve it."""
        self.target_decoders.append(decoder)

    def requirement(self) -> fault_models.DecoderFaultModelRequirement:
        """What a model must offer every decoder that may decode it."""
        requirement = self.decoder.fault_model_requirement
        for target_decoder in self.target_decoders:
            target_needs = target_decoder.fault_model_requirement
            requirement = requirement.joined(target_needs)
        return requirement

    def models_for_operation(
        self,
        operation: program_records.Operation,
        resolved_operation: program_records.ResolvedOperationPlanning,
        windows: list,
        protocol: window_records.WindowProtocol,
    ) -> list:
        """One model per window of a planned operation, or none.

        Built once per task and shared by its shots
        (built_window_models).
        """
        if self.provider is None:
            return []
        requirement = self.requirement()
        key = _model_key(
            operation, resolved_operation, windows, requirement, protocol
        )
        held = self.built_models.models_of(key)
        if held:
            return held
        models = self.provider.window_models_for_operation(
            operation,
            windows,
            resolved_operation.round_count,
            fault_model_requirement=requirement,
            fault_exclusion_ranges=(),
            window_protocol=protocol,
        )
        self.built_models.remember(key, models)
        return models

    def model_for_stream(
        self,
        stream_id: Any,  # an opaque identity
        window: window_records.Window,
    ) -> Optional[fault_models.WindowErrorModel]:
        """The model of one window of a dynamic stream, or None."""
        if self.provider is None:
            return None
        return self.provider.window_model_for_stream(stream_id, window)

    def register_stream(
        self,
        stream_operation: program_records.Operation,
        resolved_operation: program_records.ResolvedOperationPlanning,
    ) -> Optional[int]:
        """Note a dynamic stream; the rounds its source can supply, or None."""
        if self.provider is None:
            return None
        requirement = self.requirement()
        return self.provider.register_dynamic_stream(
            stream_operation,
            resolved_operation.round_count,
            fault_model_requirement=requirement,
        )

    def finalize_stream_models(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        """Fix the model boundary after physical completion is validated."""
        if self.provider is None:
            return False
        return self.provider.finalize_stream_models(
            stream_operation, stream_round_count
        )

    def strong_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict],
    ) -> Optional[fault_models.WindowErrorModel]:
        """The model of one strong window, with the listed faults excluded.

        prior_faults is what the neighbour of a pinned face has already
        committed; those faults are no columns of this model.
        """
        if self.provider is None:
            return None
        requirement = self.requirement()
        return self.provider.strong_window_model_for_operation(
            operation,
            window,
            round_count,
            fault_model_requirement=requirement,
            fault_exclusion_ranges=fault_exclusion_ranges,
            prior_faults=prior_faults,
        )


class WindowPlanner:
    """Which windows exist.

    The plan lays some up front, and a stream lays its own as it grows,
    where its scheme's stream layout says (ports.StreamLayout); the
    planner keeps every window's tables. Trace source:
    window_planned(window) for every window a stream lays after build;
    the plan's own windows exist before anyone listens.
    """

    scheme = ports.Port(ports.WindowingScheme)
    models = ports.Port(WindowModels)
    # the scheme's stream layouts; unbound when it grows no stream
    layouts = ports.Port(ports.StreamLayouts, optional=True)

    def __init__(
        self,
        resolved_operations: tuple[
            program_records.ResolvedOperationPlanning, ...
        ],
        plan: window_records.WindowPlan,
        planned_operations: tuple,
    ) -> None:
        resolved_by_id = {
            resolved.operation_id: resolved for resolved in resolved_operations
        }
        self.resolved_operation_by_id = types.MappingProxyType(resolved_by_id)
        self.plan = plan
        self.planned_operations = planned_operations
        self.layout_by_stream: dict = {}
        self.trace = _TraceSources()
        # stream id -> per window position, the least start round of
        # that window and every later one; see start_floors_of
        self.start_floors_by_stream: dict = {}

    def start(self) -> None:
        """Build the model of every window the plan laid.

        It runs once the ports are bound, as gem5's startup does
        (src/sim/sim_object.hh lines 194 and 280).
        """
        for operation in self.planned_operations:
            self._build_operation_models(operation)

    # ---- the plan's tables

    @property
    def windows_by_key(self) -> dict:
        """Every window by (stream id, index)."""
        return self.plan.windows

    @property
    def successors_by_stream(self) -> dict:
        """The streams whose rounds a window's overflow may read."""
        return self.plan.successors_by_stream

    @property
    def total_windows(self) -> int:
        """Every window planned so far, streams included."""
        return self.plan.total_windows

    def window_at(self, key: tuple) -> window_records.Window:
        """The window at (stream id, index)."""
        window = self.plan.windows.get(key)
        if window is None:
            raise KeyError(
                f"window {key} is not planned, or a stream let it go once "
                f"it was final (window_release.py)"
            )
        return window

    def later_windows(
        self,
        stream_id: Any,  # an opaque identity
        window_index: int,
    ) -> list:
        """The stream's windows past that index, in index order."""
        later = []
        for index in self.window_indices_of(stream_id):
            if index > window_index:
                later.append(self.plan.windows[(stream_id, index)])
        return later

    def absorb_window(
        self, key: tuple, restart_key: Optional[tuple]
    ) -> window_records.Window:
        """A window a strong region covers is never weak-decoded.

        It counts queued, so the requester leaves it alone, and
        committed with no logical contribution; the window that restarts
        the weak chain no longer waits for it.
        """
        window = self.plan.windows[key]
        assert not window.queued, f"absorbing queued {key}"
        assert not window.committed, f"absorbing committed {key}"
        window.queued = True
        window.committed = True
        window.is_absorbed = True
        if restart_key is not None:
            self._unhook_dependency(key, restart_key, window)
        return window

    def reslice_window(
        self,
        key: tuple,
        buffer_lo: int,
        model: Optional[fault_models.WindowErrorModel],
    ) -> window_records.Window:
        """Move a window's read start and install the model it reads with."""
        window = self.plan.windows[key]
        window.buffer_lo = buffer_lo
        window.round_count = window.buffer_hi - window.buffer_lo + 1
        self._lower_the_floors_to(key, window.start_round)
        if model is not None:
            self.models.model_by_window[key] = model
        return window

    def set_aside_window(self, key: tuple) -> None:
        """A strong region laid out later covers the window: skip it now.

        A stream escalates a window before it has laid every window the
        region covers, so the region is laid out later
        (escalation/strong_window_shapes.py DoubleWindow). Until then the
        requester and the courier leave the window alone, as they leave
        an absorbed one, and the region absorbs it when it lands.
        """
        window = self.plan.windows[key]
        window.is_absorbed = True

    def check_absorbable(self, window_keys: tuple) -> None:
        """Every listed window is still undecoded and uncommitted."""
        for key in window_keys:
            window = self.plan.windows[key]
            assert not window.committed, f"absorbing committed {key}"
            assert window.t_done is None, f"absorbing decoded {key}"

    def _unhook_dependency(
        self, key: tuple, dependent_key: tuple, window: window_records.Window
    ) -> None:
        """The dependent no longer waits on this window's commit."""
        dependent = self.plan.windows[dependent_key]
        if key in dependent.deps:
            dependent.deps.remove(key)
            dependent.deps_remaining -= 1
        if dependent_key in window.dependents:
            window.dependents.remove(dependent_key)

    def strong_window_model(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict],
    ) -> Optional[fault_models.WindowErrorModel]:
        """The error model of one strong window of that operation."""
        return self.models.strong_model_for_operation(
            operation,
            window,
            round_count,
            fault_exclusion_ranges,
            prior_faults,
        )

    def owned_faults_of(self, key: tuple) -> Optional[dict]:
        """The faults the window at that key commits, per representation."""
        model = self.models.model_by_window.get(key)
        if model is None:
            return None
        return model.owned_fault_ids()

    def crossing_faults_of(self, key: tuple) -> Optional[dict]:
        """The faults the window at that key commits behind its first round."""
        model = self.models.model_by_window.get(key)
        if model is None:
            return None
        return model.crossing_fault_ids()

    def window_indices_of(
        self,
        stream_id: Any,  # an opaque identity
    ) -> list:
        """The stream's window indices in order; none when unplanned."""
        return self.plan.windows_by_stream.get(stream_id, [])

    def windowed_stream_ids(self) -> list:
        """Every stream with windows planned."""
        return list(self.plan.windows_by_stream)

    def start_floors_of(self, stream_id: Any) -> list:  # an opaque identity
        """Per window position, the least start round from it to the last.

        Once a position's floor passes the rounds arrived, no window from
        it on has begun. Windows need not start in index order (a Tan
        seam starts before the core ahead of it). The floors grow as a
        stream plans windows, and a reslice that moves a start earlier
        lowers them; one that moves it later leaves a floor low, which
        only lets a reader look further.
        """
        floors = self._floors(stream_id)
        indices = self.window_indices_of(stream_id)
        while len(floors) < len(indices):
            window_index = indices[len(floors)]
            window = self.plan.windows[(stream_id, window_index)]
            before = len(floors) - 1
            floors.append(window.start_round)
            _lower_the_floors(floors, before, window.start_round)
        return floors

    def window_count_of(self, stream_id: Any) -> int:  # an opaque identity
        """How many windows the stream has."""
        return self.plan.window_count_by_stream[stream_id]

    def windows_of(self, stream_id: Any) -> list:  # an opaque identity
        """The stream's windows still kept, in index order.

        A stream that lets its final windows go keeps the later ones
        only (window_release.py).
        """
        windows = []
        for window_index in self.window_indices_of(stream_id):
            windows.append(self.plan.windows[(stream_id, window_index)])
        return windows

    def commit_extent_of(
        self,
        stream_id: Any,  # an opaque identity
    ) -> tuple:
        """The first and the last round the stream's windows commit.

        A stream's windows tile it from round 1 (its layout), so one that
        let its first windows go still commits from round 1.
        """
        windows = self.windows_of(stream_id)
        commit_los = [window.commit_lo for window in windows]
        commit_his = [window.commit_hi for window in windows]
        commit_lo = min(commit_los)
        commit_hi = max(commit_his)
        indices = self.window_indices_of(stream_id)
        if len(windows) < len(indices):
            commit_lo = 1
        return commit_lo, commit_hi

    def cut_rounds_of(
        self,
        stream_id: Any,  # an opaque identity
    ) -> tuple:
        """The rounds a segment of the stream ends on, a window with it."""
        layout = self.layout_by_stream[stream_id]
        return layout.cut_rounds()

    def let_go_before(self, stream_id: Any, window_index: int) -> None:
        """Let the stream's windows before the index go, with their models.

        They are final and measured (window_release.py). A stream lays
        its windows in index order, so a window's index is its position;
        the count of windows it has had stays.
        """
        indices = self.window_indices_of(stream_id)
        for position in range(indices.first_kept, window_index):
            index = indices[position]
            key = (stream_id, index)
            del self.plan.windows[key]
            self.models.model_by_window.pop(key, None)
        indices.let_go_before(window_index)
        floors = self._floors(stream_id)
        floors.let_go_before(window_index)
        layout = self.layout_by_stream.get(stream_id)
        if layout is not None:
            layout.let_go_before(window_index)

    def is_windowed(self, stream_id: Any) -> bool:  # an opaque identity
        """False for a stream decoded as one whole batch."""
        return self.plan.windowed_by_stream[stream_id]

    def batches_idle_rounds(
        self,
        stream_id: Any,  # an opaque identity
    ) -> bool:
        """Whether idle rounds before the stream fold into its batch."""
        return self.plan.batch_preceding_idle_rounds_by_stream.get(
            stream_id, False
        )

    def prepend_idle_rounds(self, stream_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style stream."""
        if round_count <= 0:
            return
        if not self.batches_idle_rounds(stream_id):
            return
        window = self.plan.windows[(stream_id, 0)]
        window.batched_preceding_idle_round_count += round_count

    # ---- the resolved operations

    def round_count_of(self, operation_id: Any) -> int:  # an opaque identity
        """The root-resolved round count of an operation."""
        return self.resolved_operation_by_id[operation_id].round_count

    def syndrome_bits_per_round_of(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """The checks one round of an operation reads, by its code card."""
        resolved = self.resolved_operation_by_id[operation_id]
        return resolved.syndrome_bits_per_round

    # ---- a dynamic stream: registered, grown, cut, closed

    def register_stream(
        self,
        stream_operation: program_records.Operation,
        physical_round_limit: Optional[int],
    ) -> Optional[int]:
        """Open a stream with agreeing physical and model limits, if finite."""
        stream_id = stream_operation.id
        self.plan.window_count_by_stream[stream_id] = 0
        self.plan.windows_by_stream[stream_id] = (
            window_positions.WindowPositions()
        )
        self.plan.successors_by_stream.setdefault(stream_id, [])
        self.plan.windowed_by_stream[stream_id] = True
        self.plan.batch_preceding_idle_rounds_by_stream[stream_id] = False
        resolved = self.resolved_operation_by_id[stream_id]
        model_round_limit = self.models.register_stream(
            stream_operation, resolved
        )
        source_round_limit = _stream_round_limit(
            physical_round_limit, model_round_limit
        )
        geometry = resolved.code_geometry
        layout = self.layouts.open_stream(
            stream_id,
            source_round_limit,
            commit_round_count=geometry.commit_round_count,
            buffer_round_count=geometry.buffer_round_count,
        )
        self.layout_by_stream[stream_id] = layout
        return source_round_limit

    def has_stream(self, stream_id: Any) -> bool:  # an opaque identity
        """True for a stream whose windows are planned at runtime."""
        return stream_id in self.layout_by_stream

    def grow_stream(
        self,
        stream_id: Any,  # an opaque identity
        highest_known_round: int,
        round_cap: Optional[int],
    ) -> list:
        """Create every window whose commit region has begun.

        round_cap is the stream's length once it is known. The new
        windows are returned in index order, in the plan's tables and
        without their model (attach_stream_model), so the caller can link
        each one's boundary before the model exists.
        """
        layout = self.layout_by_stream[stream_id]
        geometries = layout.begun(highest_known_round, round_cap)
        created = []
        for geometry in geometries:
            window = self._create_stream_window(stream_id, geometry)
            created.append(window)
        return created

    def attach_stream_model(self, window: window_records.Window) -> None:
        """Give a new stream window its model, when the source has one."""
        model = self.models.model_for_stream(window.stream_id, window)
        if model is not None:
            self.models.model_by_window[window.key] = model

    def refresh_stream_models(
        self,
        stream_id: Any,  # an opaque identity
    ) -> None:
        """Install the terminal models before the remaining windows queue.

        A readout chosen at run time can change every unqueued tail
        buffer's model; published ones stay.
        """
        for window in self.windows_of(stream_id):
            if window.queued or window.committed:
                continue
            self.attach_stream_model(window)

    def cut_stream_after(
        self,
        stream_id: Any,  # an opaque identity
        last_round: int,
    ) -> list:
        """A segment ends on the round; returns the windows that moved.

        A segment's result is the sum of the windows committed over its
        rounds (Skoric et al. 2209.08552; operation_results.py), so a
        layout that reads segment results cuts there; one that lines its
        regions up across couplings cuts at couplings only.
        """
        layout = self.layout_by_stream[stream_id]
        windows = self.windows_of(stream_id)
        return layout.cut_after(windows, last_round)

    def close_stream_tail(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> window_records.TailChange:
        """The stream's rounds end on the round; the windows that moved.

        The tail is the sealed length or a measurement-closed boundary,
        and a last window may be smaller than the others (Tan et al.
        2209.09219 lines 1052-1056; Skoric et al. 2209.08552 lines
        692-700).
        """
        layout = self.layout_by_stream[stream_id]
        windows = self.windows_of(stream_id)
        return layout.close_tail(windows, stream_round_count)

    def couple_segments(self, spans_by_block: dict, couplings: tuple) -> tuple:
        """A joint measurement's segments begin and end regions, read across.

        Returns each (reader span, read span) pair read across.
        """
        return self.layouts.couple_segments(spans_by_block, couplings)

    def relay_stream(
        self,
        stream_id: Any,  # an opaque identity
    ) -> list:
        """Lay the stream's open windows again; returns those that moved."""
        layout = self.layout_by_stream[stream_id]
        windows = self.windows_of(stream_id)
        return layout.relay(windows)

    def reader_keys_of(self, window: window_records.Window) -> tuple:
        """The window of each region reading this one; None if not begun."""
        layout = self.layout_by_stream[window.stream_id]
        return layout.readers_of(window)

    def begun_read_keys_of(self, window: window_records.Window) -> tuple:
        """The window of each region this one reads that has begun."""
        layout = self.layout_by_stream[window.stream_id]
        return layout.reads_of(window)

    def reader_keys_at(
        self,
        stream_id: Any,  # an opaque identity
        round_index: int,
    ) -> tuple:
        """The windows reading the stream's region that holds the round."""
        layout = self.layout_by_stream.get(stream_id)
        if layout is None:
            return ()
        return layout.readers_at(round_index)

    # ---- private

    def _build_operation_models(
        self, operation: program_records.Operation
    ) -> None:
        windows = self.windows_of(operation.id)
        if not windows:
            return
        resolved = self.resolved_operation_by_id[operation.id]
        protocol = self.plan.protocol_by_stream.get(
            operation.id, window_records.WindowProtocol.GENERIC
        )
        models = self.models.models_for_operation(
            operation, resolved, windows, protocol
        )
        if not models:
            return
        for window, model in zip(windows, models, strict=True):
            self.models.model_by_window[window.key] = model

    def _create_stream_window(
        self, stream_id, geometry: window_records.WindowGeometry
    ) -> window_records.Window:
        window_index = self.plan.window_count_by_stream[stream_id]
        window = window_records.Window(
            stream_id=stream_id,
            window_index=window_index,
            commit_lo=geometry.commit_lo,
            commit_hi=geometry.commit_hi,
            buffer_hi=geometry.buffer_hi,
            round_count=geometry.round_count,
            buffer_lo=geometry.buffer_lo,
            closed_temporal_boundaries=geometry.closed_temporal_boundaries,
            spatial_reads=geometry.spatial_reads,
        )
        self.plan.windows[window.key] = window
        self.plan.windows_by_stream[stream_id].append(window_index)
        self.plan.window_count_by_stream[stream_id] += 1
        self.plan.total_windows += 1
        self.trace.window_planned.fire(window)
        return window

    def _floors(self, stream_id) -> window_positions.WindowPositions:
        """The stream's start floors, by window position."""
        floors = self.start_floors_by_stream.get(stream_id)
        if floors is None:
            floors = window_positions.WindowPositions()
            self.start_floors_by_stream[stream_id] = floors
        return floors

    def _lower_the_floors_to(self, key: tuple, start_round: int) -> None:
        """A window's start moved: the floors up to it lowered to it."""
        stream_id, window_index = key
        floors = self.start_floors_by_stream.get(stream_id)
        if not floors:
            return
        indices = self.window_indices_of(stream_id)
        position = indices.index(window_index)
        if position < len(floors):
            _lower_the_floors(floors, position, start_round)


def _lower_the_floors(
    floors: window_positions.WindowPositions, position: int, start_round: int
) -> None:
    """The kept floors at the position and before it, lowered to the start."""
    while position >= floors.first_kept and floors[position] > start_round:
        floors[position] = start_round
        position -= 1


def _model_key(
    operation, resolved_operation, windows: list, requirement, protocol
) -> tuple:
    """What one operation's window models are a function of.

    The circuit and its rounds, every window's span, dependencies and
    closed boundaries, the decoder's fault-model requirement and the
    window protocol: exactly the arguments the builder is given
    (qpu/stim_device.py window_models_for_operation), and nothing a seed
    touches.
    """
    circuit_text = str(operation.circuit)
    plan = []
    for window in windows:
        deps = tuple(window.deps)
        span = (
            window.key,
            window.start_round,
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
            deps,
            window.closed_temporal_boundaries,
        )
        plan.append(span)
    return (
        operation.id,
        circuit_text,
        resolved_operation.round_count,
        tuple(plan),
        requirement,
        protocol,
    )


def _stream_round_limit(physical_round_limit, model_round_limit):
    if physical_round_limit is None:
        return model_round_limit
    if model_round_limit is None:
        return physical_round_limit
    if physical_round_limit != model_round_limit:
        raise ValueError("physical and model stream round limits must agree")
    return physical_round_limit


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the window planner reports, as one member."""

    window_planned: trace_source.TraceSource = trace_source.new_source()
