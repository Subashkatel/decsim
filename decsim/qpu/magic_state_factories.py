"""The magic-state factories: where a non-Clifford operation gets its state.

A factory fills the MagicStateFactory port (decsim/ports.py): an
operation that needs a magic state calls request and is called back when
one is ready; an empty store stalls the requester, and that supply stall
is the quantity the factories exist to measure. Trace source:
state_delivered(operation_id, waited_ticks) at every delivery, which the
run result sums.

InfiniteFactory is the idealized supply with no stall. DistillationFactory
is one 15-to-1 distillation stage: every unit attempts a distillation
every attempt_ticks, a success submits one correction decode per
multi-qubit pi/8 rotation of the 15-to-1 circuit, 11 of them (rotations
5 to 15 of Fig. 3; the first four are single-qubit rotations whose
Clifford corrections are Pauli corrections and need no ancilla: Litinski,
Magic state distillation: not as costly as you think, arXiv 1905.06903,
Sec. 4, text lines 1038-1043; Silva 2411.04270 line 381 counts the same
11 logical cycles for the T gates) to the run's decoder pool, where they
compete with the core's windows, and the state reaches the store one
return trip after the last decode. MultiLevelDistillationFactory
is the pull-driven supply chain of Silva et al., Optimizing multi-level
magic state factories for fault-tolerant quantum architectures (arXiv
2411.04270, Sec. II B): level 0 prepares physical states, each level
above consumes inputs_per_round states from the buffer below and, after
logical_cycles_per_round * distance rounds, yields outputs_per_round
states with its success probability; a failure discards the inputs. A
request at the top propagates demand down the chain, so no level
free-runs unless production_mode is "continuous".
"""

import collections
import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from typing import Any, Optional

import decsim.config as config
import decsim.engine
import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.seeding as seeding
import decsim.trace_source as trace_source


@dataclasses.dataclass
class StateTrace:
    """When one magic state was distilled, corrected, released, delivered."""

    state_id: int
    distill_start_tick: int
    physical_done_tick: int
    correction_submit_tick: int
    correction_done_tick: Optional[int] = None
    released_tick: Optional[int] = None
    delivered_tick: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class FactoryCollaborators:
    """What the root supplies every magic state factory row.

    One record so every row of MAGIC_STATE_FACTORIES has one constructor
    signature and the root builds a row without asking which supply
    model it is; a row reads the collaborators its own model needs,
    ignores the rest, and reads its own card out of settings, its own
    Settings record of the magic_state_factory section's keys (decsim/
    tables.py row_settings), None for a row with no keys. This is gem5's params
    object, where a SimObject's collaborators arrive as one structure
    rather than as a signature per subclass
    (gem5 src/python/m5/SimObject.py:204-205). The decode
    queue a row submits its correction decodes to is not here: it is a
    port, decode_queue, which the root binds once every seat exists.
    """

    engine: decsim.engine.Engine
    round_ticks: int
    # the row's own Settings record, opaque to the root
    settings: Optional[Any] = None


class InfiniteFactory:
    """The idealized factory: a magic state is always in stock."""

    # bound by the root as on every row; this row corrects nothing
    decode_queue = ports.Port(ports.DecodeQueue)

    def __init__(self, collaborators: FactoryCollaborators):
        self.engine = collaborators.engine
        # a state is always in stock, so no request waits and none fires
        self.trace = _TraceSources()

    def start(self) -> None:
        """Nothing is made ahead of a request, so nothing is queued."""

    def request(self, operation_id: int, callback: Callable[[], None]) -> None:
        """Deliver at once."""
        del operation_id
        callback()

    def shutdown(self) -> None:
        """Nothing runs, so nothing stops."""


class DistillationFactory(seeding._RandomSeedConsumer):
    """One 15-to-1 distillation stage with optional continuous production.

    Demand mode starts an attempt only for an unmet request; continuous
    mode also keeps buffer_capacity states in the pipeline. A state with
    no correction decodes to wait on is released one return trip after
    the physical attempt. initial_store warm-starts the store with states
    that carry no trace.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row's own magic_state_factory keys, checked where they enter.

        attempt_ticks and return_ticks are engine ticks. The eleven
        correction decodes are Litinski's multi-qubit rotations (module
        docstring).
        """

        unit_count: int
        attempt_ticks: int
        correction_round_count: int
        correction_decode_count: int = 11
        return_ticks: int = 0
        success_probability: float = 1.0
        seed: Optional[int] = None
        initial_store: int = 0
        production_mode: str = "demand"
        buffer_capacity: Optional[int] = None

        def __post_init__(self) -> None:
            _check_production_mode(self.production_mode, self.buffer_capacity)
            _check_count(
                "correction_decode_count",
                self.correction_decode_count,
                minimum=0,
            )
            _check_count("unit_count", self.unit_count, minimum=1)
            _check_count("attempt_ticks", self.attempt_ticks, minimum=0)
            _check_count(
                "correction_round_count",
                self.correction_round_count,
                minimum=0,
            )
            _check_count("return_ticks", self.return_ticks, minimum=0)
            _check_count("initial_store", self.initial_store, minimum=0)
            _check_probability("success_probability", self.success_probability)

        @classmethod
        def from_yaml(cls, section: Mapping) -> "DistillationFactory.Settings":
            """The keys the yaml wrote; the three with no default must be."""
            required = ("unit_count", "attempt_ticks", "correction_round_count")
            _check_required_keys("distillation", section, required)
            fields = _with_numbers(section, ("success_probability",))
            return cls(**fields)

    # the run's decoder manager, where the correction decodes compete
    # with the core's windows
    decode_queue = ports.Port(ports.DecodeQueue)

    def __init__(self, collaborators: FactoryCollaborators):
        self.engine = collaborators.engine
        self.card = collaborators.settings
        self.trace = _TraceSources()
        self._initialize_run_seed_state(self.card.seed)
        self._reset_state(self.card.initial_store)

    def start(self) -> None:
        """Continuous production fills the pipeline before any request."""
        if self.card.production_mode != "continuous":
            return
        self.engine.schedule(0, self._start_attempts, label="factory_start")

    def request(self, operation_id: int, callback: Callable[[], None]) -> None:
        """Deliver a state now if one is in stock, else when one is ready."""
        request = (operation_id, callback)
        self.waiting.append(request)
        self._stall_start_by_operation_id[operation_id] = self.engine.now
        waiting_count = len(self.waiting)
        self.engine.log(
            log_sources.MAGIC_STATE_FACTORY,
            f"op#{operation_id} requests a magic state "
            f"(store {self.stored_state_count}, waiting {waiting_count})",
        )
        self._deliver_to_waiting()
        self._start_attempts()

    def shutdown(self) -> None:
        """Stop launching attempts; the program is complete."""
        self._is_shut_down = True

    def latency_aggregate_snapshot(self) -> dict:
        """Exact totals over every delivery, whatever traces were evicted."""
        snapshot = {}
        totals = self._latency_totals
        for stage, total in totals.sum_by_stage.items():
            snapshot[stage] = {
                "sum": total,
                "max": totals.max_by_stage[stage],
                "n": totals.delivered_count,
            }
        return snapshot

    def _reset_state(self, initial_store: int) -> None:
        self.stored_state_count = initial_store
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.produced_count = 0
        self.in_flight_count = 0
        self.busy_unit_count = 0
        self.peak_in_flight_count = 0
        self.total_stall_ticks = 0
        self._stall_start_by_operation_id: dict[int, int] = {}
        self._is_shut_down = False
        self.traces: collections.deque = collections.deque(maxlen=4096)
        sum_by_stage = dict.fromkeys(_LATENCY_STAGES, 0)
        max_by_stage = dict.fromkeys(_LATENCY_STAGES, 0)
        self._latency_totals = _LatencyTotals(0, sum_by_stage, max_by_stage)
        self._ready_traces: list[StateTrace] = []
        self._next_state_id = 0

    def _start_attempts(self) -> None:
        """Launch attempts while demand is unmet or the pipeline is short."""
        while self._can_start_attempt():
            self.busy_unit_count += 1
            self.engine.schedule(
                self.card.attempt_ticks,
                self._finish_attempt,
                label="distill_attempt",
            )

    def _can_start_attempt(self) -> bool:
        if self._is_shut_down:
            return False
        if self.busy_unit_count >= self.card.unit_count:
            return False
        committed_count = self.busy_unit_count + self.in_flight_count
        has_unmet_demand = len(self.waiting) > committed_count
        if has_unmet_demand:
            return True
        if self.card.production_mode != "continuous":
            return False
        pipeline_count = self.stored_state_count + committed_count
        wanted_count = self.card.buffer_capacity + len(self.waiting)
        return pipeline_count < wanted_count

    def _finish_attempt(self) -> None:
        """A success waits for its corrections; a failure retries."""
        self.busy_unit_count -= 1
        self._mark_stochastic_use()
        is_success = self._rng.random() < self.card.success_probability
        if is_success:
            self._submit_corrections()
        else:
            self.engine.log(
                log_sources.MAGIC_STATE_FACTORY,
                "a unit's distillation DISCARDED, retrying",
            )
        self._start_attempts()

    def _submit_corrections(self) -> None:
        self.in_flight_count += 1
        self.peak_in_flight_count = max(
            self.peak_in_flight_count, self.in_flight_count
        )
        now = self.engine.now
        distill_start_tick = now - self.card.attempt_ticks
        trace = StateTrace(
            state_id=self._next_state_id,
            distill_start_tick=distill_start_tick,
            physical_done_tick=now,
            correction_submit_tick=now,
        )
        self._next_state_id += 1
        if not self.card.correction_decode_count:
            # Nothing to wait on: the state returns after the physical
            # attempt, as it does in the multi-level factory.
            trace.correction_done_tick = now
            self.engine.schedule(
                self.card.return_ticks,
                lambda: self._release(trace),
                label="distill_release",
            )
            return
        self.engine.log(
            log_sources.MAGIC_STATE_FACTORY,
            f"a unit distilled a state; submitting "
            f"{self.card.correction_decode_count} correction-qubit decode jobs "
            f"to the cluster (parallel)",
        )
        batch = _CorrectionBatch(self.card.correction_decode_count, trace)
        on_done = functools.partial(self._finish_correction_decode, batch)
        for _ in range(self.card.correction_decode_count):
            self.decode_queue.enqueue_without_input(
                self.card.correction_round_count,
                on_done=on_done,
                label="MSF-corr",
            )

    def _finish_correction_decode(self, batch: "_CorrectionBatch") -> None:
        batch.remaining_count -= 1
        if batch.remaining_count != 0:
            return
        trace = batch.trace
        trace.correction_done_tick = self.engine.now
        self.engine.schedule(
            self.card.return_ticks,
            lambda: self._release(trace),
            label="distill_release",
        )

    def _release(self, trace: StateTrace) -> None:
        """A corrected state reaches the store and serves the oldest request."""
        self.in_flight_count -= 1
        self.stored_state_count += 1
        self.produced_count += 1
        trace.released_tick = self.engine.now
        self._ready_traces.append(trace)
        self.engine.log(
            log_sources.MAGIC_STATE_FACTORY,
            f"magic state ready (store now {self.stored_state_count})",
        )
        self._deliver_to_waiting()
        self._start_attempts()

    def _deliver_to_waiting(self) -> None:
        while self.stored_state_count > 0 and self.waiting:
            self.stored_state_count -= 1
            self._stamp_delivered_trace()
            operation_id, callback = self.waiting.pop(0)
            waited_ticks = self._stall_ticks_of(operation_id)
            self.total_stall_ticks += waited_ticks
            self.trace.state_delivered.fire(operation_id, waited_ticks)
            tag = _stall_tag(waited_ticks)
            self.engine.log(
                log_sources.MAGIC_STATE_FACTORY,
                f"  -> delivered to op#{operation_id} "
                f"(store now {self.stored_state_count}){tag}",
            )
            callback()

    def _stamp_delivered_trace(self) -> None:
        # A warm-start state has no trace.
        if not self._ready_traces:
            return
        trace = self._ready_traces.pop(0)
        trace.delivered_tick = self.engine.now
        self._record_delivered_trace(trace)
        self.traces.append(trace)

    def _stall_ticks_of(self, operation_id: int) -> int:
        now = self.engine.now
        started = self._stall_start_by_operation_id.pop(operation_id, now)
        return now - started

    def _record_delivered_trace(self, trace: StateTrace) -> None:
        ticks_by_stage = {
            "distill": trace.physical_done_tick - trace.distill_start_tick,
            "corr_decode": trace.correction_done_tick
            - trace.physical_done_tick,
            "deliver": trace.delivered_tick - trace.correction_done_tick,
            "total": trace.delivered_tick - trace.distill_start_tick,
        }
        totals = self._latency_totals
        totals.delivered_count += 1
        for stage, ticks in ticks_by_stage.items():
            totals.sum_by_stage[stage] += ticks
            totals.max_by_stage[stage] = max(totals.max_by_stage[stage], ticks)


@dataclasses.dataclass
class DistillLevel:
    """One level of the multi-level factory: its units and its protocol."""

    unit_count: int
    distance: int
    # None takes the paper's count for the level's place in the chain: 13
    # logical cycles at the first level, 15 above it, where two more
    # cycles load the inputs (Silva et al. 2411.04270, Fig. 2 caption,
    # text lines 223-224 and 249-251).
    logical_cycles_per_round: Optional[int] = None
    success_probability: float = 1.0


class MultiLevelDistillationFactory(seeding._RandomSeedConsumer):
    """A pull-driven chain of distillation levels feeding one store.

    Level 0 holds preparation_unit_count units that each inject a physical
    state into a distance preparation_distance patch, one prepared state
    every preparation_logical_cycles * preparation_distance rounds. Level l
    consumes inputs_per_round states of level l - 1 per round and yields
    outputs_per_round. A request at the top level pulls demand down the
    chain; continuous mode also keeps buffer_capacity states at the top.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The row's own magic_state_factory keys, checked where they enter.

        levels runs from the first level up. The preparation_logical_cycles
        of two and preparation_distance of three are project coefficients:
        Silva et al. 2411.04270 Sec. II B prices a level in logical cycles
        of its own distance but fixes neither number for the physical
        injection stage. No correction decode by default: Silva line 249
        counts the correction inside a level's 13 logical cycles.
        """

        levels: tuple
        inputs_per_round: int = 15
        outputs_per_round: int = 1
        preparation_unit_count: int = 1
        preparation_logical_cycles: int = 2
        preparation_distance: int = 3
        preparation_success_probability: float = 1.0
        correction_round_count: int = 0
        correction_decode_count: int = 0
        seed: Optional[int] = None
        production_mode: str = "demand"
        buffer_capacity: Optional[int] = None

        def __post_init__(self) -> None:
            _check_levels(self.levels)
            _check_production_mode(self.production_mode, self.buffer_capacity)
            _check_count(
                "correction_decode_count",
                self.correction_decode_count,
                minimum=0,
            )
            _check_count("inputs_per_round", self.inputs_per_round, minimum=1)
            _check_count("outputs_per_round", self.outputs_per_round, minimum=1)
            _check_count(
                "preparation_unit_count",
                self.preparation_unit_count,
                minimum=1,
            )
            _check_count(
                "correction_round_count",
                self.correction_round_count,
                minimum=0,
            )
            config.check_cycles(
                "preparation_logical_cycles", self.preparation_logical_cycles
            )
            _check_count(
                "preparation_distance", self.preparation_distance, minimum=1
            )
            _check_probability(
                "preparation_success_probability",
                self.preparation_success_probability,
            )

        @classmethod
        def from_yaml(
            cls, section: Mapping
        ) -> "MultiLevelDistillationFactory.Settings":
            """The keys the yaml wrote; levels is a list of level mappings."""
            _check_required_keys("multi_level", section, ("levels",))
            levels = _levels_from_yaml(section["levels"])
            probability_keys = ("preparation_success_probability",)
            fields = _with_numbers(section, probability_keys)
            fields["levels"] = levels
            return cls(**fields)

    # the run's decoder manager, where a card's correction decodes go
    decode_queue = ports.Port(ports.DecodeQueue)

    def __init__(self, collaborators: FactoryCollaborators):
        self.engine = collaborators.engine
        self.card = collaborators.settings
        self.levels = _with_the_papers_cycles(self.card.levels)
        self.preparation_ticks = _preparation_ticks(
            self.card, collaborators.round_ticks
        )
        self.trace = _TraceSources()
        self._initialize_run_seed_state(self.card.seed)
        self._reset_state(collaborators.round_ticks)

    def start(self) -> None:
        """Continuous production fills the top buffer before any request."""
        if self.card.production_mode != "continuous":
            return
        self.engine.schedule(0, self._start_work, label="factory_start")

    def request(self, operation_id: int, callback: Callable[[], None]) -> None:
        """Record the demand for a final state and pull the chain."""
        request = (operation_id, callback)
        self.waiting.append(request)
        self._stall_start_by_operation_id[operation_id] = self.engine.now
        top_counters = self.counters_by_level[len(self.levels)]
        top_store = top_counters.stored_state_count
        waiting_count = len(self.waiting)
        self.engine.log(
            log_sources.MAGIC_STATE_FACTORY,
            f"op#{operation_id} requests a magic state "
            f"(top-level store {top_store}, waiting {waiting_count})",
        )
        self._start_work()

    def shutdown(self) -> None:
        """Stop the production loop; the program is complete."""
        self._is_shut_down = True

    def _reset_state(self, round_ticks: int) -> None:
        self.round_ticks_by_level = {}
        above_top_level = len(self.levels) + 1
        for level in range(1, above_top_level):
            settings = self.levels[level - 1]
            self.round_ticks_by_level[level] = _round_ticks(
                settings.logical_cycles_per_round,
                settings.distance,
                round_ticks,
            )
        # Level 0 is preparation; level l is self.levels[l - 1].
        self.counters_by_level = {}
        for level in range(0, above_top_level):
            self.counters_by_level[level] = _LevelCounters()
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.total_stall_ticks = 0
        self._stall_start_by_operation_id: dict[int, int] = {}
        self.peak_in_flight_count = 0
        self._is_shut_down = False

    def _deliver_to_waiting(self) -> None:
        top_counters = self.counters_by_level[len(self.levels)]
        while top_counters.stored_state_count > 0 and self.waiting:
            top_counters.stored_state_count -= 1
            operation_id, callback = self.waiting.pop(0)
            waited_ticks = self._stall_ticks_of(operation_id)
            self.total_stall_ticks += waited_ticks
            self.trace.state_delivered.fire(operation_id, waited_ticks)
            tag = _stall_tag(waited_ticks)
            self.engine.log(
                log_sources.MAGIC_STATE_FACTORY,
                f"  -> delivered final state to op#{operation_id}{tag}",
            )
            callback()

    def _stall_ticks_of(self, operation_id: int) -> int:
        now = self.engine.now
        started = self._stall_start_by_operation_id.pop(operation_id, now)
        return now - started

    def _start_work(self) -> None:
        """Deliver what is ready, then start every round the demand allows."""
        if self._is_shut_down:
            return
        self._deliver_to_waiting()
        has_progress = True
        while has_progress:
            demand_by_level = self._demand_by_level()
            has_started_preparation = self._start_preparation(demand_by_level)
            has_started_distillation = self._start_distillation(demand_by_level)
            has_progress = has_started_preparation or has_started_distillation
        total_busy_unit_count = 0
        for counters in self.counters_by_level.values():
            total_busy_unit_count += counters.busy_unit_count
        self.peak_in_flight_count = max(
            self.peak_in_flight_count, total_busy_unit_count
        )

    def _demand_by_level(self) -> dict:
        """How many states each level must supply, top down."""
        top_level = len(self.levels)
        demand_by_level = {top_level: len(self.waiting)}
        if self.card.production_mode == "continuous":
            demand_by_level[top_level] += self.card.buffer_capacity
        for level in range(top_level, 0, -1):
            wanted_round_count = self._wanted_round_count(
                level, demand_by_level[level]
            )
            input_level = level - 1
            demand_by_level[input_level] = (
                self.card.inputs_per_round * wanted_round_count
            )
        return demand_by_level

    def _wanted_round_count(self, level: int, demand: int) -> int:
        """Rounds a level must run to cover a demand for its outputs."""
        counters = self.counters_by_level[level]
        in_progress_count = (
            counters.busy_unit_count * self.card.outputs_per_round
        )
        committed_count = counters.stored_state_count + in_progress_count
        deficit_count = demand - committed_count
        if deficit_count <= 0:
            return 0
        fractional_round_count = deficit_count / self.card.outputs_per_round
        return math.ceil(fractional_round_count)

    def _start_preparation(self, demand_by_level: dict) -> bool:
        has_progress = False
        counters = self.counters_by_level[0]
        idle_unit_count = (
            self.card.preparation_unit_count - counters.busy_unit_count
        )
        shortfall_count = demand_by_level[0] - counters.stored_state_count
        shortfall_count -= counters.busy_unit_count
        deficit_count = max(0, shortfall_count)
        while deficit_count > 0 and idle_unit_count > 0:
            counters.busy_unit_count += 1
            self.engine.schedule(
                self.preparation_ticks,
                self._finish_preparation,
                label="prep",
            )
            idle_unit_count -= 1
            deficit_count -= 1
            has_progress = True
        return has_progress

    def _start_distillation(self, demand_by_level: dict) -> bool:
        has_progress = False
        above_top_level = len(self.levels) + 1
        for level in range(1, above_top_level):
            has_started_rounds = self._start_level_rounds(
                level, demand_by_level[level]
            )
            if has_started_rounds:
                has_progress = True
        return has_progress

    def _start_level_rounds(self, level: int, demand: int) -> bool:
        has_progress = False
        wanted_round_count = self._wanted_round_count(level, demand)
        settings = self.levels[level - 1]
        counters = self.counters_by_level[level]
        idle_unit_count = settings.unit_count - counters.busy_unit_count
        input_counters = self.counters_by_level[level - 1]
        while wanted_round_count > 0 and idle_unit_count > 0:
            if input_counters.stored_state_count < self.card.inputs_per_round:
                break
            input_counters.stored_state_count -= self.card.inputs_per_round
            counters.busy_unit_count += 1
            self._start_round(level)
            idle_unit_count -= 1
            wanted_round_count -= 1
            has_progress = True
        return has_progress

    def _start_round(self, level: int) -> None:
        distillation_round = _DistillationRound(level)
        self.engine.schedule(
            self.round_ticks_by_level[level],
            lambda: self._finish_physical_round(distillation_round),
            label=f"distill_L{level}",
        )

    def _finish_physical_round(
        self, distillation_round: "_DistillationRound"
    ) -> None:
        """Submit the correction decodes; measurement data exists only now."""
        distillation_round.is_physically_done = True
        if self.card.correction_decode_count:
            distillation_round.remaining_decode_count = (
                self.card.correction_decode_count
            )
            level = distillation_round.level
            on_done = functools.partial(
                self._finish_correction_decode, distillation_round
            )
            for _ in range(self.card.correction_decode_count):
                self.decode_queue.enqueue_without_input(
                    self.card.correction_round_count,
                    on_done=on_done,
                    label=f"MSF-corr-L{level}",
                )
        self._finish_round(distillation_round)

    def _finish_correction_decode(
        self, distillation_round: "_DistillationRound"
    ) -> None:
        distillation_round.remaining_decode_count -= 1
        self._finish_round(distillation_round)

    def _finish_round(self, distillation_round: "_DistillationRound") -> None:
        """Yield the round's states once physical time and decodes are done."""
        if distillation_round.is_done:
            return
        if not distillation_round.is_physically_done:
            return
        if distillation_round.remaining_decode_count > 0:
            return
        distillation_round.is_done = True
        level = distillation_round.level
        counters = self.counters_by_level[level]
        counters.busy_unit_count -= 1
        self._mark_stochastic_use()
        settings = self.levels[level - 1]
        is_success = self._rng.random() < settings.success_probability
        if is_success:
            self._add_outputs(level)
        else:
            counters.failure_count += 1
            self.engine.log(
                log_sources.MAGIC_STATE_FACTORY,
                f"level {level} distillation failed (inputs discarded), "
                "retrying",
            )
        self._start_work()

    def _add_outputs(self, level: int) -> None:
        counters = self.counters_by_level[level]
        counters.stored_state_count += self.card.outputs_per_round
        counters.produced_count += self.card.outputs_per_round
        destination = f"level-{level} state to buffer"
        if level == len(self.levels):
            destination = "final state to core buffer"
        input_level = level - 1
        consumed = self.card.inputs_per_round
        self.engine.log(
            log_sources.MAGIC_STATE_FACTORY,
            f"level {level} distilled a state ({destination}; "
            f"consumed {consumed} level-{input_level} states)",
        )

    def _finish_preparation(self) -> None:
        counters = self.counters_by_level[0]
        counters.busy_unit_count -= 1
        self._mark_stochastic_use()
        is_success = (
            self._rng.random() < self.card.preparation_success_probability
        )
        if is_success:
            counters.stored_state_count += 1
            counters.produced_count += 1
        else:
            counters.failure_count += 1
        self._start_work()


_LATENCY_STAGES = ("distill", "corr_decode", "deliver", "total")


@dataclasses.dataclass
class _LatencyTotals:
    """Exact sums and maxima per latency stage over every delivery."""

    delivered_count: int
    sum_by_stage: dict
    max_by_stage: dict


@dataclasses.dataclass
class _LevelCounters:
    """The states in store, units busy, states made and rounds failed."""

    stored_state_count: int = 0
    busy_unit_count: int = 0
    produced_count: int = 0
    failure_count: int = 0


@dataclasses.dataclass
class _CorrectionBatch:
    """The correction decodes one distilled state is waiting on."""

    remaining_count: int
    trace: StateTrace


@dataclasses.dataclass
class _DistillationRound:
    """One distillation round in flight at one level."""

    level: int
    is_physically_done: bool = False
    remaining_decode_count: int = 0
    is_done: bool = False


def _check_production_mode(
    production_mode: str, buffer_capacity: Optional[int]
) -> None:
    if production_mode not in ("demand", "continuous"):
        raise ValueError(
            "production_mode must be 'demand' or 'continuous' "
            f"(got {production_mode!r})"
        )
    if production_mode != "continuous":
        return
    if buffer_capacity is None or buffer_capacity < 1:
        raise ValueError("continuous production needs buffer_capacity >= 1")


def _check_count(name: str, value, *, minimum: int) -> None:
    if value < minimum:
        relation = "nonnegative"
        if minimum == 1:
            relation = "positive"
        raise ValueError(f"{name} must be {relation}")


def _check_probability(name: str, value) -> None:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _with_numbers(section: Mapping, keys: tuple) -> dict:
    """The section's keys, each named one read as the number it writes.

    YAML 1.1 loads `1e-3` as text, so a probability the yaml writes that
    way arrives as a string, as escalation's online card reads it too
    (decsim/escalation/settings.py _online_float).
    """
    fields = dict(section)
    for key in keys:
        if key in fields:
            fields[key] = float(fields[key])
    return fields


def _check_required_keys(kind: str, section: Mapping, required: tuple) -> None:
    """A row's keys with no default are written, or named as missing."""
    missing = []
    for key in required:
        if key not in section:
            missing.append(key)
    if missing:
        raise ValueError(
            f"magic_state_factory kind {kind} needs {missing}; "
            "configs/reference.yaml lists every key"
        )


def _levels_from_yaml(level_sections) -> tuple:
    """The yaml's list of level mappings as DistillLevel records."""
    level_keys = _distill_level_keys()
    levels = []
    for index, level_section in enumerate(level_sections):
        unknown = set(level_section) - set(level_keys)
        if unknown:
            listed = sorted(unknown)
            raise ValueError(
                f"magic_state_factory.levels[{index}] does not know "
                f"{listed}; its keys are {list(level_keys)}"
            )
        level_fields = _with_numbers(level_section, ("success_probability",))
        level = DistillLevel(**level_fields)
        levels.append(level)
    return tuple(levels)


def _distill_level_keys() -> tuple:
    names = []
    for field in dataclasses.fields(DistillLevel):
        names.append(field.name)
    return tuple(names)


def _check_levels(levels: tuple) -> None:
    for index, level in enumerate(levels):
        _check_count(f"levels[{index}].unit_count", level.unit_count, minimum=1)
        _check_count(f"levels[{index}].distance", level.distance, minimum=1)
        _check_probability(
            f"levels[{index}].success_probability", level.success_probability
        )
        logical_cycles = level.logical_cycles_per_round
        if logical_cycles is not None:
            config.check_cycles(
                f"levels[{index}].logical_cycles_per_round", logical_cycles
            )


def _with_the_papers_cycles(levels: tuple) -> tuple:
    """The levels, each with its logical cycles; None takes the paper's."""
    completed = []
    for index, level in enumerate(levels):
        logical_cycles = level.logical_cycles_per_round
        if logical_cycles is None:
            logical_cycles = _paper_logical_cycles(index)
        completed_level = dataclasses.replace(
            level, logical_cycles_per_round=logical_cycles
        )
        completed.append(completed_level)
    return tuple(completed)


def _paper_logical_cycles(level_index: int) -> int:
    """Silva 2411.04270: 13 logical cycles at the first level, 15 above it."""
    if level_index == 0:
        return 13
    return 15


def _preparation_ticks(settings, round_ticks: int) -> int:
    """The ticks one physical preparation lasts, on the run's round."""
    _check_count("round_ticks", round_ticks, minimum=0)
    return _round_ticks(
        settings.preparation_logical_cycles,
        settings.preparation_distance,
        round_ticks,
    )


def _round_ticks(logical_cycles: int, distance: int, round_ticks: int) -> int:
    """The ticks one distillation round lasts: cycles times distance rounds."""
    round_count = logical_cycles * distance
    return round_count * round_ticks


def _stall_tag(waited_ticks: int) -> str:
    if waited_ticks == 0:
        return ""
    stall_text = config.format_ticks(waited_ticks)
    stall_text = stall_text.strip()
    return f"  (supply stall {stall_text})"


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event a factory reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one member per
    counter; a component's events are the same shape, so a listener
    reaches all of them through one name.
    """

    state_delivered: trace_source.TraceSource = trace_source.new_source()
