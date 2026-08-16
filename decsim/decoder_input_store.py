"""Own decoder-side input storage and the admission boundary in front of it.

A ``DecoderInputStore`` owns one pool's round credits and its live slots: a
request reserves the rounds it will store, deposits one immutable input, and
returns its credits when the decoder is finished with it or it is cancelled.

``DecoderInputStoreStager`` is the storage-admission boundary that every
decoder-input transport ends at. It counts a request's round demand, admits it
when the pool's credits fit, materializes the input once, releases the upstream
hold, and hands the job to the manager continuation. A request that does not
fit waits in strict FIFO order within its pool (``STALL``) or fails loudly
(``FAIL_STOP``). With no configuration the stager owns one shared unbounded
store, so nothing waits and nothing is recorded.

This module does not queue or schedule decoder work, and it never calls back
into the decoder manager when credits are returned: release and cancel only
mark a pool drainable, and the manager drains it at its own settled boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from .message import (
    DecodeJob,
    DecoderRequestKey,
    RetainedSyndromeFragment,
    same_stable_identity,
    stable_identity_order_key,
)


SHARED_UNBOUNDED_STORE_POOL = "*"
"""Label of the one unbounded store used when no finite store is configured.

The label is reserved: with no configuration there are no per-pool stores, so
it never stands for a decoder unit pool.
"""


class DecoderInputStoreCapacityExhaustion(RuntimeError):
    """A request's rounds did not fit in its pool's free round credits."""

    status = "decoder_input_store_capacity_exhaustion"

    def __init__(self, *, pool: str, requested_rounds: int,
                 capacity_rounds: Optional[int], snapshot) -> None:
        self.pool = pool
        self.requested_rounds = requested_rounds
        self.capacity_rounds = capacity_rounds
        self.snapshot = snapshot
        super().__init__(
            f"decoder-input store pool {pool!r} holds "
            f"{snapshot.occupied_rounds} of {capacity_rounds} rounds and "
            f"cannot admit {requested_rounds} more"
        )


class DecoderInputStoreUnsatisfiableDemand(RuntimeError):
    """One request needs more rounds than its pool's whole budget.

    Distinct from capacity exhaustion because no release can ever make this
    request fit, so waiting for credits would never end.
    """

    status = "decoder_input_store_unsatisfiable_demand"

    def __init__(self, *, pool: str, requested_rounds: int,
                 capacity_rounds: Optional[int], snapshot) -> None:
        self.pool = pool
        self.requested_rounds = requested_rounds
        self.capacity_rounds = capacity_rounds
        self.snapshot = snapshot
        super().__init__(
            f"decoder-input store pool {pool!r} has a {capacity_rounds}-round "
            f"budget and can never admit a {requested_rounds}-round request"
        )


class DecoderInputStoreOverflowPolicy(Enum):
    """What a configured finite store does with a request that does not fit."""

    STALL = "stall"
    FAIL_STOP = "fail_stop"


@dataclass(frozen=True)
class DecoderInputStoreConfig:
    """Finite decoder-input storage, in syndrome rounds, per decoder unit pool.

    ``round_capacity_by_pool`` is copied and wrapped read-only, so a later edit
    of the caller's mapping cannot change a live run. Its pool names must equal
    the decoder manager's normalized unit pools; there is no conversion from
    decoder-request slots and no default budget. Leaving
    ``RunSpec.decoder_input_store`` unset disables the finite model entirely.
    """

    round_capacity_by_pool: Mapping[str, int]
    overflow_policy: DecoderInputStoreOverflowPolicy = (
        DecoderInputStoreOverflowPolicy.STALL)

    def __post_init__(self) -> None:
        if type(self.overflow_policy) is not DecoderInputStoreOverflowPolicy:
            raise TypeError(
                "overflow_policy must be a DecoderInputStoreOverflowPolicy")
        source = self.round_capacity_by_pool
        entries = source.items() if hasattr(source, "items") else source
        copied_capacity_by_pool: dict[str, int] = {}
        for pool_name, capacity_rounds in entries:
            if type(pool_name) is not str:
                raise TypeError("decoder-input store pool names must be str")
            if pool_name in copied_capacity_by_pool:
                raise ValueError(
                    f"duplicate decoder-input store pool {pool_name!r}")
            if type(capacity_rounds) is not int or capacity_rounds < 1:
                raise TypeError(
                    f"pool {pool_name!r} needs a positive built-in int round "
                    f"capacity")
            copied_capacity_by_pool[pool_name] = capacity_rounds
        object.__setattr__(self, "round_capacity_by_pool",
                           MappingProxyType(copied_capacity_by_pool))


@dataclass(frozen=True)
class MaterializedSyndromeRound:
    """One immutable syndrome round owned by the decoder side."""

    operation_id: Any
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]

    def __post_init__(self) -> None:
        if type(self.round_index) is not int:
            raise TypeError("round_index must be an exact built-in int")
        for fragment in self.fragments:
            if not same_stable_identity(
                    fragment.operation_id, self.operation_id):
                raise ValueError(
                    "materialized fragments must share operation identity")


@dataclass(frozen=True)
class DecoderInput:
    """Immutable local input for one decoder request.

    Rounds are ordered by operation identity and round index.
    """

    op_id: int
    window_id: int
    request_key: Optional[DecoderRequestKey]
    rounds: tuple[MaterializedSyndromeRound, ...]


def _check_detector_row_layout(
    job: DecodeJob,
    rounds: tuple[MaterializedSyndromeRound, ...],
) -> None:
    """Check model-backed operation/round order and dense row positions."""
    model = getattr(job, "dem", None)
    if model is None:
        return
    missing_layout_member = object()
    detector_ids = getattr(model, "detector_ids", missing_layout_member)
    defect_positions = getattr(
        model, "defect_positions", missing_layout_member
    )
    if (
        detector_ids is missing_layout_member
        or defect_positions is missing_layout_member
    ):
        return

    input_row_identities = []
    for round_input in rounds:
        if not same_stable_identity(round_input.operation_id, job.op_id):
            raise ValueError(
                f"{getattr(job, 'label', '')}: model-backed decoder-input "
                f"round operation {round_input.operation_id!r} does not match "
                f"job operation {job.op_id!r}"
            )
        position_in_round = 0
        for fragment in round_input.fragments:
            if fragment.bits is None:
                continue
            input_row_identities.extend(
                (
                    round_input.operation_id,
                    round_input.round_index,
                    position_in_round + bit_offset,
                )
                for bit_offset in range(len(fragment.bits))
            )
            position_in_round += len(fragment.bits)
    input_row_identities = tuple(input_row_identities)

    model_row_identities = []
    for detector_id in detector_ids:
        round_index, position_in_round = defect_positions[detector_id]
        model_row_identities.append(
            (job.op_id, round_index, position_in_round)
        )
    model_row_identities = tuple(model_row_identities)
    if input_row_identities != model_row_identities:
        raise ValueError(
            f"{getattr(job, 'label', '')}: canonical decoder-input row layout "
            f"{input_row_identities!r} does not match the window error model's "
            f"row layout {model_row_identities!r}"
        )

def materialize_decoder_input(job: DecodeJob) -> DecoderInput:
    """Build one immutable decoder-input store input from a job's fragments."""
    fragments_by_round: dict[tuple, list[RetainedSyndromeFragment]] = {}
    for payload in job.payloads:
        if type(payload) is not RetainedSyndromeFragment:
            raise TypeError(
                "every job payload must be a RetainedSyndromeFragment")
        identity = (payload.operation_id, payload.round_index)
        fragments_by_round.setdefault(identity, []).append(payload)
    ordered = sorted(
        fragments_by_round.items(),
        key=lambda item: (
            stable_identity_order_key(item[0][0]), item[0][1]
        ),
    )
    rounds = tuple(
        MaterializedSyndromeRound(
            operation_id=identity[0],
            round_index=identity[1],
            fragments=tuple(fragments),
        )
        for identity, fragments in ordered
    )
    _check_detector_row_layout(job, rounds)
    return DecoderInput(
        op_id=job.op_id,
        window_id=job.window_id,
        request_key=job.request_key,
        rounds=rounds,
    )


@dataclass(frozen=True)
class DecoderInputStoreSnapshot:
    """Immutable observation of one pool's round credits and live slots."""

    pool: str
    capacity_rounds: Optional[int]
    occupied_rounds: int
    peak_occupied_rounds: int
    slots_in_use: int
    admissions: int


def count_decoder_input_round_demand(job: DecodeJob) -> int:
    """Count the distinct syndrome rounds one job will store.

    The demand is the number of distinct ``(operation_id, round_index)``
    identities in the job's payloads, which is exactly the number of rounds
    ``materialize_decoder_input`` groups them into. It is counted here so that
    capacity is charged in actual stored rounds without building a second
    immutable input while a request waits, and without reading ``n_rounds``,
    which is service extent rather than stored data. Payload type admission
    stays at materialization, after credits fit.
    """
    round_identities = {
        (payload.operation_id, payload.round_index) for payload in job.payloads
    }
    return len(round_identities)


class _SlotState(Enum):
    RESERVED = "reserved"
    DEPOSITED = "deposited"


class DecoderInputStore:
    """Own one pool's decoder-input round credits and its live slots.

    One key owns one slot holding the rounds it reserved.
    ``round_capacity=None`` means unbounded. Credits are charged on reserve and
    returned exactly once by ``take`` or ``discard``. Invalid state changes
    raise ``RuntimeError`` without changing the store.
    """

    def __init__(self, *, pool: str, round_capacity: Optional[int]) -> None:
        if type(pool) is not str:
            raise TypeError("decoder-input store pool must be a str label")
        if round_capacity is not None and (
                type(round_capacity) is not int or round_capacity < 1):
            raise TypeError(
                "round_capacity must be None (unbounded) or a positive "
                "built-in int")
        self.pool = pool
        self.round_capacity = round_capacity
        self._slots: dict[
            Any, tuple[_SlotState, int, Optional[DecoderInput]]] = {}
        self._occupied_rounds = 0
        self._peak_occupied_rounds = 0
        self._admissions = 0

    @property
    def slots_in_use(self) -> int:
        return len(self._slots)

    @property
    def rounds_in_use(self) -> int:
        return self._occupied_rounds

    def snapshot(self) -> DecoderInputStoreSnapshot:
        """Take one immutable observation of this store."""
        return DecoderInputStoreSnapshot(
            pool=self.pool,
            capacity_rounds=self.round_capacity,
            occupied_rounds=self._occupied_rounds,
            peak_occupied_rounds=self._peak_occupied_rounds,
            slots_in_use=len(self._slots),
            admissions=self._admissions,
        )

    def exceeds_total_capacity(self, round_demand: int) -> bool:
        """Whether this demand could not fit even in an empty store."""
        return (self.round_capacity is not None
                and round_demand > self.round_capacity)

    def has_free_credits(self, round_demand: int) -> bool:
        """Whether this demand fits in the round credits free right now."""
        if self.round_capacity is None:
            return True
        return self._occupied_rounds + round_demand <= self.round_capacity

    def unsatisfiable_demand_error(
        self, round_demand: int,
    ) -> DecoderInputStoreUnsatisfiableDemand:
        """Build the error for a request larger than the whole budget."""
        return DecoderInputStoreUnsatisfiableDemand(
            pool=self.pool, requested_rounds=round_demand,
            capacity_rounds=self.round_capacity, snapshot=self.snapshot())

    def capacity_exhaustion_error(
        self, round_demand: int,
    ) -> DecoderInputStoreCapacityExhaustion:
        """Build the error for a request that does not fit right now."""
        return DecoderInputStoreCapacityExhaustion(
            pool=self.pool, requested_rounds=round_demand,
            capacity_rounds=self.round_capacity, snapshot=self.snapshot())

    def reserve(self, key: Any, round_demand: int) -> None:
        """Claim one slot and its round credits for a future deposit."""
        if type(round_demand) is not int:
            raise TypeError("round_demand must be an exact built-in int")
        if round_demand < 0:
            raise ValueError("round_demand must be nonnegative")
        if key in self._slots:
            raise RuntimeError(
                f"decoder-input store slot {key!r} is already "
                f"{self._slots[key][0].value}")
        if self.exceeds_total_capacity(round_demand):
            raise self.unsatisfiable_demand_error(round_demand)
        if not self.has_free_credits(round_demand):
            raise self.capacity_exhaustion_error(round_demand)
        self._slots[key] = (_SlotState.RESERVED, round_demand, None)
        self._occupied_rounds += round_demand
        self._peak_occupied_rounds = max(
            self._peak_occupied_rounds, self._occupied_rounds)
        self._admissions += 1

    def deposit(self, key: Any, job: DecodeJob) -> DecoderInput:
        """Materialize one job's input into a reserved slot and return it.

        A materialized round count other than the reserved demand is an
        internal accounting failure: the slot stays reserved and the caller's
        error unwind returns its credits exactly once.
        """
        state, reserved_rounds, _ = self._slots[key]
        if state is not _SlotState.RESERVED:
            raise RuntimeError(
                f"decoder-input store slot {key!r} already holds a deposit")
        decoder_input = materialize_decoder_input(job)
        if len(decoder_input.rounds) != reserved_rounds:
            raise RuntimeError(
                f"decoder-input store slot {key!r} reserved {reserved_rounds} "
                f"rounds but materialized {len(decoder_input.rounds)}")
        self._slots[key] = (_SlotState.DEPOSITED, reserved_rounds,
                            decoder_input)
        return decoder_input

    def take(self, key: Any) -> DecoderInput:
        """Consume one deposited input and return its round credits."""
        state, reserved_rounds, decoder_input = self._slots[key]
        if state is not _SlotState.DEPOSITED:
            raise RuntimeError(
                f"take from decoder-input store slot {key!r} before any deposit")
        del self._slots[key]
        self._occupied_rounds -= reserved_rounds
        return decoder_input

    def discard(self, key: Any) -> None:
        """Free one live slot (reserved or deposited) and return its credits."""
        if key not in self._slots:
            raise RuntimeError(
                f"discard of unknown decoder-input store slot {key!r}")
        _, reserved_rounds, _ = self._slots.pop(key)
        self._occupied_rounds -= reserved_rounds


@dataclass(frozen=True)
class DecoderInputStoreStallRecord:
    """One request's wait for round credits, from arrival to admission.

    Recorded only for a configured finite store, so a run without one keeps
    exactly the per-stage latency surface it had before.
    """

    request_key: Optional[DecoderRequestKey]
    pool: str
    arrival_tick: int
    admitted_tick: int
    round_demand: int


@dataclass(frozen=True)
class _PendingDecoderInputStoreRequest:
    """One request waiting in a pool's FIFO for round credits.

    It holds the original job with its frozen upstream fragments and the
    manager's admission continuation. No second immutable input exists while a
    request waits, and the transport receiver is already finished with it.
    """

    job: DecodeJob
    round_demand: int
    pool: str
    on_admitted: Callable[[DecodeJob], None]
    on_materialized: Optional[Callable[[DecodeJob], None]]
    arrival_tick: int


@dataclass(frozen=True)
class DecoderInputStoreStagerSnapshot:
    """Immutable observation of every store plus the waiting and stall facts.

    The per-store rows are exact whether or not a finite store is configured;
    the waiting, stall, and record fields exist only for a configured one.
    """

    enabled: bool
    per_pool_store: tuple[DecoderInputStoreSnapshot, ...]
    waiting_jobs_by_pool: tuple[tuple[str, int], ...]
    waiting_rounds_by_pool: tuple[tuple[str, int], ...]
    stall_events_by_pool: tuple[tuple[str, int], ...]
    stall_ticks_by_pool: tuple[tuple[str, int], ...]
    stall_records: tuple[DecoderInputStoreStallRecord, ...]


class DecoderInputStoreStager:
    """Admit decoder requests into storage after every decoder-input transport.

    The decoder manager owns one stager and every transport, built-in or
    supplied by a user, ends at it, so no transport can bypass the configured
    storage axis. Without a configuration it owns one shared unbounded store
    and admits every request at its arrival tick. With one it owns an exact
    store per decoder unit pool, a strict FIFO of requests waiting for credits
    per pool, and the stall accounting for them. Strict means arrival order:
    once a pool has a waiting request, every later arrival for that pool queues
    behind it whether or not its own rounds would fit.

    Credit returns never call back into the manager: ``release`` and ``cancel``
    only mark a pool drainable, and the manager drains admissible waiting heads
    from its own non-reentrant dispatch loop at a settled tick boundary.
    """

    def __init__(self, engine, *, config: Optional[DecoderInputStoreConfig],
                 pool_names: tuple[str, ...]) -> None:
        if config is not None and type(config) is not DecoderInputStoreConfig:
            raise TypeError(
                "decoder_input_store must be a DecoderInputStoreConfig or None")
        self.engine = engine
        self.enabled = config is not None
        self.overflow_policy = None if config is None else config.overflow_policy
        if config is None:
            self._stores = {
                SHARED_UNBOUNDED_STORE_POOL: DecoderInputStore(
                    pool=SHARED_UNBOUNDED_STORE_POOL, round_capacity=None),
            }
        else:
            configured_pools = set(config.round_capacity_by_pool)
            if configured_pools != set(pool_names):
                raise ValueError(
                    f"decoder-input store pools {sorted(configured_pools)} must "
                    f"be exactly the decoder unit pools {sorted(pool_names)}")
            self._stores = {
                pool_name: DecoderInputStore(
                    pool=pool_name,
                    round_capacity=config.round_capacity_by_pool[pool_name])
                for pool_name in pool_names
            }
        self._waiting_by_pool: dict[
            str, list[_PendingDecoderInputStoreRequest]] = {
            pool_name: [] for pool_name in self._stores}
        self._waiting_pool_by_key: dict[Any, str] = {}
        self._admitted_pool_by_key: dict[Any, str] = {}
        self._stall_events_by_pool = {pool_name: 0 for pool_name in self._stores}
        self._stall_ticks_by_pool = {pool_name: 0 for pool_name in self._stores}
        self._stall_records: list[DecoderInputStoreStallRecord] = []
        self._drainable_pools: set[str] = set()

    @staticmethod
    def _key(job: DecodeJob):
        """One request's storage identity, stable for its whole lifetime."""
        return job.request_key if job.request_key is not None else id(job)

    def admit(self, job: DecodeJob, *, pool: Optional[str],
              on_admitted: Callable[[DecodeJob], None],
              on_materialized: Optional[Callable[[DecodeJob], None]] = None,
    ) -> None:
        """Take one arrived request into storage, or make it wait for credits.

        ``pool`` is the decoder unit pool the manager resolved for this request
        before its transport; it is ignored without a configured store, where
        one shared store holds everything and pool selection stays late.

        Admission is in strict arrival order per pool: a request joins the tail
        whenever that pool already has one waiting, even if its own rounds
        would fit right now.
        """
        store = self._store_for(pool)
        round_demand = count_decoder_input_round_demand(job)
        if store.exceeds_total_capacity(round_demand):
            raise store.unsatisfiable_demand_error(round_demand)
        # Strict FIFO is order, not only fit: a small arrival must never
        # overtake a request already waiting in the same pool, or a stream of
        # small requests could starve the head indefinitely.
        pool_has_waiting_requests = bool(self._waiting_by_pool[store.pool])
        credits_fit_now = store.has_free_credits(round_demand)
        if not pool_has_waiting_requests and credits_fit_now:
            self._admit_request(store, job, round_demand, on_admitted,
                                on_materialized)
            return
        if self.overflow_policy is DecoderInputStoreOverflowPolicy.FAIL_STOP:
            # FAIL_STOP never queues, so its pools have no waiting requests and
            # this is always the genuine aggregate-overflow case.
            raise store.capacity_exhaustion_error(round_demand)
        self._append_waiting_request(store, job, round_demand, on_admitted,
                                     on_materialized)

    def release(self, job: DecodeJob) -> None:
        """Return one admitted request's credits; do nothing if it holds none."""
        self._return_credits(job)

    def cancel(self, job: DecodeJob) -> None:
        """Drop one request from storage wherever it is, exactly once.

        A waiting request loses its FIFO place with no callback, no credit
        return and no stall record, because it never occupied storage. An
        admitted request returns its credits and loses its input. A request
        that is unknown or already cleared is left alone.
        """
        key = self._key(job)
        waiting_pool = self._waiting_pool_by_key.pop(key, None)
        if waiting_pool is not None:
            waiting = self._waiting_by_pool[waiting_pool]
            for position, pending in enumerate(waiting):
                if pending.job is job:
                    del waiting[position]
                    return
            raise RuntimeError(
                f"decoder-input request {key!r} is recorded as waiting in pool "
                f"{waiting_pool!r} but is not in its queue")
        self._return_credits(job)

    def has_drainable_work(self) -> bool:
        """Whether some pool returned credits since its last drain."""
        return bool(self._drainable_pools)

    def drain_admissible_requests(self) -> None:
        """Admit every waiting head whose rounds now fit, in strict FIFO order.

        Head-of-line blocking is deliberate: a pool stops at its first waiting
        request that does not fit, so a stream of small requests cannot starve
        a large one. Only the manager's dispatch loop calls this, so admission
        continuations never re-enter a drain.
        """
        self._drainable_pools.clear()
        for pool_name, waiting in self._waiting_by_pool.items():
            store = self._stores[pool_name]
            while waiting and store.has_free_credits(waiting[0].round_demand):
                pending = waiting.pop(0)
                del self._waiting_pool_by_key[self._key(pending.job)]
                self._record_stall(pending)
                self._admit_request(
                    store, pending.job, pending.round_demand,
                    pending.on_admitted, pending.on_materialized)

    def snapshot(self) -> DecoderInputStoreStagerSnapshot:
        """Take one immutable observation of storage, waiting, and stalls."""
        pool_names = sorted(self._stores)
        return DecoderInputStoreStagerSnapshot(
            enabled=self.enabled,
            per_pool_store=tuple(
                self._stores[pool_name].snapshot() for pool_name in pool_names),
            waiting_jobs_by_pool=tuple(
                (pool_name, len(self._waiting_by_pool[pool_name]))
                for pool_name in pool_names),
            waiting_rounds_by_pool=tuple(
                (pool_name, sum(pending.round_demand
                                for pending in self._waiting_by_pool[pool_name]))
                for pool_name in pool_names),
            stall_events_by_pool=tuple(
                (pool_name, self._stall_events_by_pool[pool_name])
                for pool_name in pool_names),
            stall_ticks_by_pool=tuple(
                (pool_name, self._stall_ticks_by_pool[pool_name])
                for pool_name in pool_names),
            stall_records=tuple(self._stall_records),
        )

    def unsettled_storage(self) -> tuple[tuple[str, list], ...]:
        """Storage state that must be empty once the simulation is quiescent."""
        occupied_pools = sorted(
            pool_name for pool_name, store in self._stores.items()
            if store.rounds_in_use or store.slots_in_use)
        waiting_pools = sorted(
            pool_name for pool_name, waiting in self._waiting_by_pool.items()
            if waiting)
        unsettled = []
        if occupied_pools:
            unsettled.append(
                ("still holding decoder input", occupied_pools))
        if waiting_pools:
            unsettled.append(
                ("waiting for decoder-input round credits", waiting_pools))
        if self._drainable_pools:
            unsettled.append(
                ("left with undrained decoder-input credits",
                 sorted(self._drainable_pools)))
        return tuple(unsettled)

    def _store_for(self, pool: Optional[str]) -> DecoderInputStore:
        """The store that owns this request's credits."""
        if not self.enabled:
            return self._stores[SHARED_UNBOUNDED_STORE_POOL]
        if pool is None:
            raise RuntimeError(
                "a configured decoder-input store needs the request's pool")
        store = self._stores.get(pool)
        if store is None:
            raise RuntimeError(f"unknown decoder-input store pool {pool!r}")
        return store

    def _admit_request(self, store: DecoderInputStore, job: DecodeJob,
                       round_demand: int,
                       on_admitted: Callable[[DecodeJob], None],
                       on_materialized: Optional[Callable[[DecodeJob], None]],
    ) -> None:
        """Reserve credits, materialize once, then hand the job on.

        This is the only materialization boundary, so the stored input is built
        exactly once, after its credits fit. Any failure between reserving and
        the manager continuation returns those credits exactly once; the
        upstream hold is released only after storage has accepted the request.
        """
        key = self._key(job)
        store.reserve(key, round_demand)
        self._admitted_pool_by_key[key] = store.pool
        try:
            job.decoder_input = store.deposit(key, job)
            job.payloads = []
            if on_materialized is not None:
                on_materialized(job)
            on_admitted(job)
        except BaseException:
            self._return_credits(job)
            raise

    def _append_waiting_request(self, store: DecoderInputStore, job: DecodeJob,
                                round_demand: int,
                                on_admitted: Callable[[DecodeJob], None],
                                on_materialized: Optional[
                                    Callable[[DecodeJob], None]],
    ) -> None:
        """Put one request at the back of its pool's FIFO, upstream hold kept."""
        key = self._key(job)
        if key in self._waiting_pool_by_key:
            raise RuntimeError(
                f"decoder-input request {key!r} is already waiting for credits")
        self._waiting_by_pool[store.pool].append(
            _PendingDecoderInputStoreRequest(
                job=job, round_demand=round_demand, pool=store.pool,
                on_admitted=on_admitted, on_materialized=on_materialized,
                arrival_tick=self.engine.now))
        self._waiting_pool_by_key[key] = store.pool
        self._stall_events_by_pool[store.pool] += 1

    def _record_stall(self,
                      pending: _PendingDecoderInputStoreRequest) -> None:
        """Record one wait that ended in admission, for stage decomposition."""
        admitted_tick = self.engine.now
        self._stall_ticks_by_pool[pending.pool] += (
            admitted_tick - pending.arrival_tick)
        self._stall_records.append(DecoderInputStoreStallRecord(
            request_key=pending.job.request_key,
            pool=pending.pool,
            arrival_tick=pending.arrival_tick,
            admitted_tick=admitted_tick,
            round_demand=pending.round_demand,
        ))

    def _return_credits(self, job: DecodeJob) -> None:
        """Free one admitted request's slot exactly once and clear its input.

        A request that reserved credits but never deposited is only reachable
        through the admission error unwind, so its slot is discarded unread; a
        stored input is taken back and checked to be the one the job still
        points at, because a swapped input would decode the wrong syndrome.
        """
        key = self._key(job)
        pool = self._admitted_pool_by_key.pop(key, None)
        if pool is None:
            return
        store = self._stores[pool]
        if job.decoder_input is None:
            store.discard(key)
        else:
            taken = store.take(key)
            if taken is not job.decoder_input:
                raise RuntimeError("decoder-input store input identity changed")
            job.decoder_input = None
        self._drainable_pools.add(pool)
