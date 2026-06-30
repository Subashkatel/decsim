"""Magic-state supply models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from .config import fmt
from .engine import Engine

if TYPE_CHECKING:
    from .protocols import DecodeService


def _validate_production_mode(production: str, buffer_capacity: Optional[int]) -> None:
    """Validate the factory production mode and buffer setting."""
    if production not in ("demand", "continuous"):
        raise ValueError(
            f"production must be 'demand' or 'continuous' (got {production!r})")
    if production == "continuous" and (buffer_capacity is None or buffer_capacity < 1):
        raise ValueError("continuous production needs buffer_capacity >= 1")


@dataclass
class StateTrace:
    """Per-magic-state production and delivery timestamps."""

    state_id: int
    t_distill_start: int            # distillation began
    t_phys_done: int                # physical distillation done
    t_corr_submit: int              # correction decodes submitted
    t_corr_done: Optional[int] = None   # last correction decode returned
    t_released: Optional[int] = None    # entered store (after return trip)
    t_delivered: Optional[int] = None   # handed to consumer


class InfiniteFactory:
    """Idealized factory with unlimited magic states."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def request(self, op_id: int, callback: Callable[[], None]) -> None:
        """Deliver instantly."""
        callback()

    def shutdown(self) -> None:
        """Nothing to stop."""
        return None


class DistillationFactory:
    """Single-level magic-state factory with optional continuous production."""

    def __init__(self, engine: Engine, num_units: int, cycle_ticks: int,
                 decode_service: "DecodeService", corr_rounds: int, n_corr: int = 11,
                 return_ticks: int = 0, p_success: float = 1.0, seed: int = 0,
                 initial_store: int = 0, production: str = "demand",
                 buffer_capacity: Optional[int] = None):
        import random

        _validate_production_mode(production, buffer_capacity)
        self.engine = engine
        self.num_units = num_units
        self.cycle_ticks = cycle_ticks
        self.decode_service = decode_service
        self.corr_rounds = corr_rounds
        self.n_corr = n_corr
        self.return_ticks = return_ticks
        self.p_success = p_success
        self.rng = random.Random(seed)
        self.production = production
        self.buffer_capacity = buffer_capacity
        self._init_runtime_state(initial_store)

        if production == "continuous":
            self.engine.schedule(0, self._maybe_start, label="factory_start")

    def _init_runtime_state(self, initial_store: int) -> None:
        """Initialize queues, counters, and bounded trace storage."""
        self.store = initial_store
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.produced = 0
        self.in_flight = 0
        self.busy_units = 0
        self.peak_in_flight = 0
        self.total_stall = 0
        self._stall_start: dict[int, int] = {}
        self._shutdown = False
        self.traces: deque = deque(maxlen=4096)
        self._ready_traces: list[StateTrace] = []
        self._next_state_id = 0

    def shutdown(self) -> None:
        """Stop launching new attempts (called when the circuit is complete)."""
        self._shutdown = True

    def _maybe_start(self) -> None:
        """Launch attempts while demand is unmet, or (continuous) while the pipeline is below buffer_capacity."""
        while not self._shutdown and self.busy_units < self.num_units:
            demand = len(self.waiting) > self.busy_units + self.in_flight
            stocking = (self.production == "continuous"
                        and self.store + self.in_flight + self.busy_units
                        < self.buffer_capacity + len(self.waiting))
            if not (demand or stocking):
                break
            self.busy_units += 1
            self.engine.schedule(self.cycle_ticks, self._attempt_done,
                                 label="distill_attempt")

    def _attempt_done(self) -> None:
        """A distillation attempt finished; on success, queue its correction decode."""
        self.busy_units -= 1
        if self.rng.random() < self.p_success:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            trace = StateTrace(state_id=self._next_state_id,
                               t_distill_start=self.engine.now - self.cycle_ticks,
                               t_phys_done=self.engine.now,
                               t_corr_submit=self.engine.now)
            self._next_state_id += 1
            remaining = {"n": self.n_corr, "trace": trace}
            self.engine.log("Factory",
                            f"a unit distilled a state; submitting {self.n_corr} "
                            f"correction-qubit decode jobs to the cluster (parallel)")
            for _ in range(self.n_corr):
                self.decode_service.submit_decode(
                    self.corr_rounds,
                    on_done=lambda rem=remaining: self._corr_done(rem),
                    label="MSF-corr")
        else:
            self.engine.log("Factory", "a unit's distillation DISCARDED, retrying")
        self._maybe_start()                    # keep going only if demand remains

    def _corr_done(self, remaining: dict) -> None:
        """A correction decode finished; the state is now ready in the store."""
        remaining["n"] -= 1
        if remaining["n"] == 0:
            remaining["trace"].t_corr_done = self.engine.now
            self.engine.schedule(self.return_ticks,
                                 lambda trace=remaining["trace"]: self._release(trace),
                                 label="distill_release")

    def _release(self, trace: StateTrace) -> None:
        """Hand a finished state to the oldest waiting request."""
        self.in_flight -= 1
        self.store += 1
        self.produced += 1
        trace.t_released = self.engine.now
        self._ready_traces.append(trace)
        self.engine.log("Factory", f"magic state ready (store now {self.store})")
        self._fulfil()
        self._maybe_start()

    def request(self, op_id: int, callback: Callable[[], None]) -> None:
        """A gate asks for a state: deliver now if in stock, else deliver when ready."""
        self.waiting.append((op_id, callback))
        self._stall_start[op_id] = self.engine.now
        self.engine.log("Factory",
                        f"op#{op_id} requests a magic state "
                        f"(store {self.store}, waiting {len(self.waiting)})")
        self._fulfil()
        self._maybe_start()

    def _fulfil(self) -> None:
        """Deliver a state to a waiting request and log it."""
        while self.store > 0 and self.waiting:
            self.store -= 1
            if self._ready_traces:             # warm-start states have no trace
                trace = self._ready_traces.pop(0)
                trace.t_delivered = self.engine.now
                self.traces.append(trace)
            op_id, callback = self.waiting.pop(0)
            waited = self.engine.now - self._stall_start.pop(op_id, self.engine.now)
            self.total_stall += waited
            tag = "" if waited == 0 else f"  (supply stall {fmt(waited).strip()})"
            self.engine.log("Factory",
                            f"  -> delivered to op#{op_id} (store now {self.store}){tag}")
            callback()
            self._maybe_start()                # continuous mode: refill the slot just taken

@dataclass
class DistillLevel:
    """One level of a multi-level magic-state factory."""

    units: int
    d: int
    O: int = 13
    P: float = 1.0


class MultiLevelDistillationFactory:
    """Multi-level pull-driven magic-state factory."""

    def __init__(self, engine: Engine, levels: list[DistillLevel], *,
                 W_ticks: int, M: int = 15, N: int = 1,
                 prep_units: int = 1, prep_O: int = 2, prep_d: int = 3, prep_P: float = 1.0,
                 decode_service: Optional["DecodeService"] = None,
                 corr_rounds: int = 0, n_corr: int = 0, seed: int = 0,
                 production: str = "demand", buffer_capacity: Optional[int] = None):
        import random

        _validate_production_mode(production, buffer_capacity)
        self.production = production
        self.buffer_capacity = buffer_capacity
        self.engine = engine
        self.levels = levels
        self.L = len(levels)
        self.M = M
        self.N = N
        self.prep_units = prep_units
        self.prep_time = prep_O * prep_d * W_ticks
        self.prep_P = prep_P
        self.decode_service = decode_service
        self.corr_rounds = corr_rounds
        self.n_corr = n_corr
        self.rng = random.Random(seed)
        self._init_multilevel_state(W_ticks)

        if production == "continuous":
            self.engine.schedule(0, self._drive, label="factory_start")

    def _init_multilevel_state(self, W_ticks: int) -> None:
        """Initialize buffers, busy counts, counters, and round times."""
        self.round_time = {
            level: self.levels[level - 1].O * self.levels[level - 1].d * W_ticks
            for level in range(1, self.L + 1)
        }
        self.buffer = {level: 0 for level in range(0, self.L + 1)}
        self.busy = {level: 0 for level in range(0, self.L + 1)}
        self.produced = {level: 0 for level in range(0, self.L + 1)}
        self.failures = {level: 0 for level in range(0, self.L + 1)}
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.total_stall = 0
        self._stall_start: dict[int, int] = {}
        self.peak_in_flight = 0
        self._shutdown = False

    def shutdown(self) -> None:
        """Stop the production loop."""
        self._shutdown = True

    def request(self, op_id: int, callback: Callable[[], None]) -> None:
        """A gate asks for a final state; record demand and start producing."""
        self.waiting.append((op_id, callback))
        self._stall_start[op_id] = self.engine.now
        self.engine.log("Factory",
                        f"op#{op_id} requests a magic state "
                        f"(top-level store {self.buffer[self.L]}, waiting {len(self.waiting)})")
        self._drive()

    def _fulfil_core(self) -> None:
        """Deliver a finished final state to a waiting request."""
        while self.buffer[self.L] > 0 and self.waiting:
            self.buffer[self.L] -= 1
            op_id, callback = self.waiting.pop(0)
            waited = self.engine.now - self._stall_start.pop(op_id, self.engine.now)
            self.total_stall += waited
            tag = "" if waited == 0 else f"  (supply stall {fmt(waited).strip()})"
            self.engine.log("Factory", f"  -> delivered final state to op#{op_id}{tag}")
            callback()

    def _drive(self) -> None:
        """Pull engine: recompute demand top-down and start work each level can do."""
        if self._shutdown:
            return

        self._fulfil_core()
        progress = True
        while progress:
            need = self._pull_demand()
            prep_started = self._start_preparation_work(need)
            distillation_started = self._start_distillation_work(need)
            progress = prep_started or distillation_started
        self.peak_in_flight = max(self.peak_in_flight, sum(self.busy.values()))

    def _pull_demand(self) -> dict:
        """Compute how many input states each level must supply."""
        import math

        need = {self.L: len(self.waiting)}
        if self.production == "continuous":
            need[self.L] += self.buffer_capacity

        for level in range(self.L, 0, -1):
            deficit = max(
                0,
                need[level] - self.buffer[level] - self.busy[level] * self.N)
            rounds = math.ceil(deficit / self.N) if deficit > 0 else 0
            need[level - 1] = self.M * rounds
        return need

    def _start_preparation_work(self, need: dict) -> bool:
        """Start level-0 preparation jobs if inputs are needed."""
        progress = False
        idle_units = self.prep_units - self.busy[0]
        deficit = max(0, need[0] - self.buffer[0] - self.busy[0])

        while deficit > 0 and idle_units > 0:
            self.busy[0] += 1
            self.engine.schedule(self.prep_time, self._prep_done, label="prep")
            idle_units -= 1
            deficit -= 1
            progress = True
        return progress

    def _start_distillation_work(self, need: dict) -> bool:
        """Start every distillation level that has inputs and idle units."""
        progress = False
        for level in range(1, self.L + 1):
            if self._start_level_work(level, need):
                progress = True
        return progress

    def _start_level_work(self, level: int, need: dict) -> bool:
        """Start distillation rounds for one level."""
        import math

        progress = False
        deficit = max(
            0,
            need[level] - self.buffer[level] - self.busy[level] * self.N)
        rounds_wanted = math.ceil(deficit / self.N) if deficit > 0 else 0
        idle_units = self.levels[level - 1].units - self.busy[level]

        while rounds_wanted > 0 and idle_units > 0 and self.buffer[level - 1] >= self.M:
            self.buffer[level - 1] -= self.M
            self.busy[level] += 1
            self._start_round(level)
            idle_units -= 1
            rounds_wanted -= 1
            progress = True
        return progress

    def _start_round(self, level: int) -> None:
        """Begin one distillation round at a level."""
        round_state = {"level": level, "phys": False, "decodes_left": 0, "done": False}
        if self.decode_service is not None and self.n_corr:
            round_state["decodes_left"] = self.n_corr
            for _ in range(self.n_corr):
                self.decode_service.submit_decode(
                    self.corr_rounds,
                    on_done=lambda state=round_state: self._corr_done(state),
                    label=f"MSF-corr-L{level}")
        self.engine.schedule(
            self.round_time[level],
            lambda state=round_state: self._phys_done(state),
            label=f"distill_L{level}",
        )

    def _phys_done(self, round_state: dict) -> None:
        """The physical distillation time elapsed; finish the round if decoding is also done."""
        round_state["phys"] = True
        self._finish_round(round_state)

    def _corr_done(self, round_state: dict) -> None:
        """One correction decode returned."""
        round_state["decodes_left"] -= 1
        self._finish_round(round_state)

    def _finish_round(self, round_state: dict) -> None:
        """Produce the state once physical time and correction decodes are complete."""
        if round_state["done"] or not round_state["phys"] or round_state["decodes_left"] > 0:
            return
        round_state["done"] = True
        level = round_state["level"]
        self.busy[level] -= 1
        if self.rng.random() < self.levels[level - 1].P:
            self.buffer[level] += self.N
            self.produced[level] += self.N
            destination = "final state to core buffer" if level == self.L \
                else f"level-{level} state to buffer"
            self.engine.log("Factory",
                            f"level {level} distilled a state ({destination}; "
                            f"consumed {self.M} level-{level - 1} states)")
        else:
            self.failures[level] += 1
            self.engine.log(
                "Factory",
                f"level {level} distillation failed (inputs discarded), retrying",
            )
        self._drive()

    def _prep_done(self) -> None:
        """A level-0 prepared state is ready; add it to the buffer."""
        self.busy[0] -= 1
        if self.rng.random() < self.prep_P:
            self.buffer[0] += 1
            self.produced[0] += 1
        else:
            self.failures[0] += 1
        self._drive()

