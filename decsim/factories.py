"""Magic-state supply models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
import threading
from typing import Callable, Optional, TYPE_CHECKING

from .config import fmt
from .engine import Engine
from .message import RunSeedChild, RunSeedPathSegment, RunSeedReservation

if TYPE_CHECKING:
    from .protocols import ResourcePool as DecodeService


class _RandomSeedConsumer:
    """Leaf-owned atomic run-seed state for random.Random factory models."""

    def _initialize_run_seed_state(self, seed: Optional[int]) -> None:
        self._explicit_seed = seed
        self._rng = random.Random(seed)
        self._run_seed_lock = threading.Lock()
        self._pending_run_seed = None
        self._run_seed_claimed = False
        self._stochastic_use_started = False

    def reserve_run_seed(self, seed: Optional[int]) -> RunSeedReservation:
        component_name = type(self).__name__
        if seed is not None and (
            type(seed) is not int or not 0 <= seed < (1 << 64)
        ):
            raise TypeError(
                f"{component_name} run root must be an unsigned 64-bit "
                f"built-in integer or None; got {seed!r}"
            )
        with self._run_seed_lock:
            if self._stochastic_use_started:
                raise ValueError(
                    f"{component_name} was already used and cannot be rebound"
                )
            if self._run_seed_claimed:
                raise ValueError(
                    f"{component_name} is already claimed by a built run"
                )
            if self._pending_run_seed is not None:
                raise ValueError(
                    f"{component_name} already has a pending run-seed "
                    "reservation"
                )
            if seed is not None and self._explicit_seed is not None:
                raise ValueError(
                    f"{component_name} has an explicit seed that conflicts "
                    f"with numeric run root {seed}"
                )
            if seed is not None:
                seed_source = "derived"
                effective_seed = seed
            elif self._explicit_seed is not None:
                if type(self._explicit_seed) is not int:
                    raise TypeError(
                        f"{component_name} explicit seed must be a built-in "
                        "integer for run provenance"
                    )
                seed_source = "explicit_local"
                effective_seed = self._explicit_seed
            else:
                seed_source = "entropy"
                effective_seed = None
            reservation = RunSeedReservation(
                proposed_seed_source=seed_source,
                proposed_seed=effective_seed,
                prepared_state=random.Random(effective_seed),
            )
            self._pending_run_seed = reservation
            return reservation

    def cancel_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._run_seed_lock:
            if self._pending_run_seed is reservation:
                self._pending_run_seed = None

    def commit_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._run_seed_lock:
            if self._pending_run_seed is not reservation:
                raise ValueError(
                    f"{type(self).__name__} can commit only its exact pending "
                    "run-seed reservation"
                )
            self._rng = reservation.prepared_state
            self._pending_run_seed = None
            self._run_seed_claimed = True

    def _mark_stochastic_use(self) -> None:
        with self._run_seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    f"{type(self).__name__} cannot draw while a run-seed "
                    "reservation is pending"
                )
            self._stochastic_use_started = True


def _validate_production_mode(production: str, buffer_capacity: Optional[int]) -> None:
    """Validate the factory production mode and buffer setting."""
    if type(production) is not str:
        raise TypeError("production must be a built-in str")
    if production not in ("demand", "continuous"):
        raise ValueError(
            f"production must be 'demand' or 'continuous' (got {production!r})")
    if buffer_capacity is not None and (
        type(buffer_capacity) is not int or buffer_capacity < 1
    ):
        raise TypeError("buffer_capacity must be a positive built-in int or None")
    if production == "continuous" and buffer_capacity is None:
        raise ValueError("continuous production needs buffer_capacity >= 1")


def _validate_exact_integer(name: str, value, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        relation = "positive" if minimum == 1 else "nonnegative"
        raise TypeError(f"{name} must be a {relation} built-in int")
    return value


def _validate_probability(name: str, value) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a built-in int or float")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _validate_correction_decode_service(decode_service, n_corr: int) -> None:
    """Require one unambiguous correction-service disposition."""
    if type(n_corr) is not int or n_corr < 0:
        raise TypeError("n_corr must be a nonnegative built-in int")
    if n_corr == 0 and decode_service is not None:
        raise ValueError("decode_service must be None when n_corr is zero")
    if n_corr > 0 and decode_service is None:
        raise ValueError("decode_service is required when n_corr is positive")


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



@dataclass
class Ticket:
    """Cancellable handle for one factory request (spec §5.21 port 19)."""

    op_id: int
    entry: tuple
    factory: object

    def cancel(self) -> bool:
        return self.factory.cancel(self)


class InfiniteFactory:
    """Idealized factory with unlimited magic states."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def request(self, op_id: int, callback: Callable[[], None]) -> "Ticket":
        """Deliver instantly."""
        callback()
        return Ticket(op_id, (), self)

    def cancel(self, ticket: "Ticket") -> bool:
        """Nothing pending to cancel (delivery was instant)."""
        return False

    def shutdown(self) -> None:
        """Nothing to stop."""
        return None


class DistillationFactory(_RandomSeedConsumer):
    """Single-level magic-state factory with optional continuous production."""

    def __init__(self, engine: Engine, num_units: int, cycle_ticks: int,
                 decode_service: "DecodeService", corr_rounds: int, n_corr: int = 11,
                 return_ticks: int = 0, p_success: float = 1.0,
                 seed: Optional[int] = None,
                 initial_store: int = 0, production: str = "demand",
                 buffer_capacity: Optional[int] = None):
        _validate_production_mode(production, buffer_capacity)
        _validate_correction_decode_service(decode_service, n_corr)
        num_units = _validate_exact_integer(
            "num_units", num_units, minimum=1
        )
        cycle_ticks = _validate_exact_integer(
            "cycle_ticks", cycle_ticks, minimum=0
        )
        corr_rounds = _validate_exact_integer(
            "corr_rounds", corr_rounds, minimum=0
        )
        return_ticks = _validate_exact_integer(
            "return_ticks", return_ticks, minimum=0
        )
        initial_store = _validate_exact_integer(
            "initial_store", initial_store, minimum=0
        )
        p_success = _validate_probability("p_success", p_success)
        self.engine = engine
        self.num_units = num_units
        self.cycle_ticks = cycle_ticks
        self.decode_service = decode_service
        self.corr_rounds = corr_rounds
        self.n_corr = n_corr
        self.return_ticks = return_ticks
        self.p_success = p_success
        self._initialize_run_seed_state(seed)
        self.production = production
        self.buffer_capacity = buffer_capacity
        self._initial_store = initial_store
        self._init_runtime_state(initial_store)

        if production == "continuous":
            self.engine.schedule(0, self._maybe_start, label="factory_start")

    def run_seed_children(self):
        """Expose the active correction service that affects completion."""
        if self.decode_service is None:
            return ()
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "decode_service"),),
                self.decode_service,
            ),
        )

    def _init_runtime_state(self, initial_store: int) -> None:
        """Initialize queues, counters, and delivered-state traces."""
        self.store = initial_store
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.produced = 0
        self.in_flight = 0
        self.busy_units = 0
        self.peak_in_flight = 0
        self.total_stall = 0
        self._stall_start: dict[int, int] = {}
        self._shutdown = False
        self.traces: deque = deque()
        self._ready_traces: list[StateTrace] = []
        self._next_state_id = 0

    def shutdown(self) -> None:
        """Stop launching new attempts (called when the circuit is complete)."""
        self._shutdown = True

    def _maybe_start(self) -> None:
        """Launch attempts while demand is unmet, or (continuous) while
        the pipeline is below buffer_capacity."""
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
        self._mark_stochastic_use()
        if self._rng.random() < self.p_success:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            trace = StateTrace(state_id=self._next_state_id,
                               t_distill_start=self.engine.now - self.cycle_ticks,
                               t_phys_done=self.engine.now,
                               t_corr_submit=self.engine.now)
            self._next_state_id += 1
            remaining = {"n": self.n_corr, "trace": trace}
            if self.n_corr:
                self.engine.log("Factory",
                                f"a unit distilled a state; submitting {self.n_corr} "
                                f"correction-qubit decode jobs to the cluster (parallel)")
                for _ in range(self.n_corr):
                    self.decode_service.submit_decode(
                        self.corr_rounds,
                        on_done=lambda rem=remaining: self._corr_done(rem),
                        label="MSF-corr")
            else:
                # No correction decodes to wait on: release after physical
                # distillation instead of hanging forever (parity with the
                # multi-level factory's `and self.n_corr` guard).
                trace.t_corr_done = self.engine.now
                self.engine.schedule(
                    self.return_ticks,
                    lambda t=trace: self._release(t),
                    label="distill_release")
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

    def request(self, op_id: int, callback: Callable[[], None]) -> "Ticket":
        """A gate asks for a state: deliver now if in stock, else deliver when ready."""
        entry = (op_id, callback)
        self.waiting.append(entry)
        self._stall_start[op_id] = self.engine.now
        self.engine.log("Factory",
                        f"op#{op_id} requests a magic state "
                        f"(store {self.store}, waiting {len(self.waiting)})")
        self._fulfil()
        self._maybe_start()
        return Ticket(op_id, entry, self)

    def cancel(self, ticket: "Ticket") -> bool:
        """Withdraw an undelivered request; FIFO order of the rest is preserved."""
        if ticket.entry in self.waiting:
            self.waiting.remove(ticket.entry)
            self._stall_start.pop(ticket.op_id, None)
            return True
        return False

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


class MultiLevelDistillationFactory(_RandomSeedConsumer):
    """Multi-level pull-driven magic-state factory."""

    def __init__(self, engine: Engine, levels: list[DistillLevel], *,
                 W_ticks: int, M: int = 15, N: int = 1,
                 prep_units: int = 1, prep_O: int = 2, prep_d: int = 3, prep_P: float = 1.0,
                 decode_service: Optional["DecodeService"] = None,
                 corr_rounds: int = 0, n_corr: int = 0,
                 seed: Optional[int] = None,
                 production: str = "demand", buffer_capacity: Optional[int] = None):
        _validate_production_mode(production, buffer_capacity)
        _validate_correction_decode_service(decode_service, n_corr)
        if type(levels) is not list or not levels:
            raise TypeError("levels must be a nonempty built-in list")
        validated_levels = []
        for index, level in enumerate(levels):
            if type(level) is not DistillLevel:
                raise TypeError(
                    f"levels[{index}] must be an exact DistillLevel"
                )
            validated_levels.append(
                DistillLevel(
                    units=_validate_exact_integer(
                        f"levels[{index}].units", level.units, minimum=1
                    ),
                    d=_validate_exact_integer(
                        f"levels[{index}].d", level.d, minimum=1
                    ),
                    O=_validate_exact_integer(
                        f"levels[{index}].O", level.O, minimum=0
                    ),
                    P=_validate_probability(
                        f"levels[{index}].P", level.P
                    ),
                )
            )
        W_ticks = _validate_exact_integer("W_ticks", W_ticks, minimum=0)
        M = _validate_exact_integer("M", M, minimum=1)
        N = _validate_exact_integer("N", N, minimum=1)
        prep_units = _validate_exact_integer(
            "prep_units", prep_units, minimum=1
        )
        prep_O = _validate_exact_integer("prep_O", prep_O, minimum=0)
        prep_d = _validate_exact_integer("prep_d", prep_d, minimum=1)
        prep_P = _validate_probability("prep_P", prep_P)
        corr_rounds = _validate_exact_integer(
            "corr_rounds", corr_rounds, minimum=0
        )
        self.production = production
        self.buffer_capacity = buffer_capacity
        self.engine = engine
        self.levels = tuple(validated_levels)
        self.L = len(self.levels)
        self.M = M
        self.N = N
        self.prep_units = prep_units
        self.prep_time = prep_O * prep_d * W_ticks
        self.prep_P = prep_P
        self._window_ticks = W_ticks
        self._prep_O = prep_O
        self._prep_d = prep_d
        self.decode_service = decode_service
        self.corr_rounds = corr_rounds
        self.n_corr = n_corr
        self._initialize_run_seed_state(seed)
        self._init_multilevel_state(W_ticks)

        if production == "continuous":
            self.engine.schedule(0, self._drive, label="factory_start")

    def run_seed_children(self):
        """Expose the active correction service that affects completion."""
        if self.decode_service is None:
            return ()
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "decode_service"),),
                self.decode_service,
            ),
        )

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

    def request(self, op_id: int, callback: Callable[[], None]) -> "Ticket":
        """A gate asks for a final state; record demand and start producing."""
        entry = (op_id, callback)
        self.waiting.append(entry)
        self._stall_start[op_id] = self.engine.now
        self.engine.log("Factory",
                        f"op#{op_id} requests a magic state "
                        f"(top-level store {self.buffer[self.L]}, waiting {len(self.waiting)})")
        self._drive()
        return Ticket(op_id, entry, self)

    def cancel(self, ticket: "Ticket") -> bool:
        """Withdraw an undelivered request; FIFO order of the rest is preserved."""
        if ticket.entry in self.waiting:
            self.waiting.remove(ticket.entry)
            self._stall_start.pop(ticket.op_id, None)
            return True
        return False

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
        self.engine.schedule(
            self.round_time[level],
            lambda state=round_state: self._phys_done(state),
            label=f"distill_L{level}",
        )

    def _phys_done(self, round_state: dict) -> None:
        """Physical time elapsed; submit correction decodes NOW (one overlap
        rule, spec §5.21: corrections start at the end of the physical
        attempt in every factory — measurement data exists only then)."""
        round_state["phys"] = True
        if self.decode_service is not None and self.n_corr:
            round_state["decodes_left"] = self.n_corr
            level = round_state["level"]
            for _ in range(self.n_corr):
                self.decode_service.submit_decode(
                    self.corr_rounds,
                    on_done=lambda state=round_state: self._corr_done(state),
                    label=f"MSF-corr-L{level}")
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
        self._mark_stochastic_use()
        if self._rng.random() < self.levels[level - 1].P:
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
        self._mark_stochastic_use()
        if self._rng.random() < self.prep_P:
            self.buffer[0] += 1
            self.produced[0] += 1
        else:
            self.failures[0] += 1
        self._drive()
