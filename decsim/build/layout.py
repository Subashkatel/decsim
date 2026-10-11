"""The layout a machine is built from: its pools, stores and escalate edges.

The build loops over these, entry first, then each edge's target in
graph order, so a machine of any number of pools builds through one
code path. Each record carries the names its components go by: the
word its results write, its statistics path, its seed names, its log
labels and its hops. A legacy machine's records keep the names its
slots have always given (legacy_layout), so its results, seeds and
statistics are unchanged; a graph's other parts go by their own names.

The order is part of a run's results: the entry store is built first,
since a store's tick-end listener runs in the order the stores were
built (syndrome_buffer/syndrome_buffer.py), and the entry pool's jobs
are admitted before any target's (windows/decode_requests.py).
"""

import dataclasses
from typing import Any, Optional

import decsim.build.routes as routes_build
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.records.log_sources as log_sources
import decsim.records.routes as route_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module


@dataclasses.dataclass(frozen=True)
class PoolNames:
    """The names one pool's components go by.

    seat is its units' seat, which detection_events.formed_at names;
    queue the name its manager's pool and units log under;
    statistics_path where its manager's statistics sit; unit_seed and
    manager_seed the roots its decoder's and its manager's seeds hash.
    """

    seat: str
    queue: str
    statistics_path: str
    unit_seed: str
    manager_seed: str


@dataclasses.dataclass(frozen=True)
class PoolPlan:
    """One decoder pool: its settings, the store it reads, its hops.

    settings None is a run that decodes nothing, whose one pool holds
    an idle unit. store names the StorePlan it reads.
    """

    hops: route_records.PoolHops
    settings: Optional[decoder_settings.DecoderPoolSettings]
    names: PoolNames
    store: str

    @property
    def pool(self) -> window_records.PoolId:
        """The word this pool's results write."""
        return self.hops.pool


@dataclasses.dataclass(frozen=True)
class StorePlan:
    """One syndrome buffer: its settings, how it takes rounds, its read.

    name is its statistics path; label the name its outgoing end and
    the trace use; read_hop the hop its reader's input rides; reader
    the seat of the pool that reads it, and reads_in_place whether that
    pool's units read the rounds where the store keeps them.
    takes_machine_clock is a store whose writes and reads are priced
    on the machine's clock when it names none of its own.
    """

    name: str
    settings: Any
    intake: receiver_module.StoreIntake
    label: str
    read_hop: transfer_records.LinkPath
    reader: str
    reads_in_place: bool
    takes_machine_clock: bool


@dataclasses.dataclass(frozen=True)
class EdgePlan:
    """One escalate edge: its switching settings and its hops.

    name is the path its statistics sit at; carried_from the store an
    escalated window's missing rounds are carried from; policy_seed the
    root its verdict's seed hashes.
    """

    name: str
    settings: escalation_settings.SwitchingSettings
    hops: route_records.EdgeHops
    carried_from: str
    policy_seed: str


@dataclasses.dataclass(frozen=True)
class Layout:
    """A machine's pools, stores and edges, each entry first.

    The entry pool decodes the planner's windows and reads the entry
    store, the one the controller writes.
    """

    pools: tuple
    stores: tuple
    edges: tuple

    @property
    def entry_pool(self) -> PoolPlan:
        """The pool that decodes the planner's windows."""
        return self.pools[0]

    def pool(self, pool: window_records.PoolId) -> PoolPlan:
        """The plan of one pool."""
        for plan in self.pools:
            if plan.pool == pool:
                return plan
        raise KeyError(pool)

    def store(self, name: str) -> StorePlan:
        """The plan of one store, by name."""
        for plan in self.stores:
            if plan.name == name:
                return plan
        raise KeyError(name)

    def target_by_source(self) -> dict:
        """The pool each source pool escalates to."""
        targets = {}
        for edge in self.edges:
            hops = edge.hops
            targets[hops.source] = hops.target
        return targets

    def routes(self) -> route_records.Routes:
        """Every answer and boundary route these pools and edges give."""
        pool_hops = []
        for plan in self.pools:
            pool_hops.append(plan.hops)
        edge_hops = []
        for edge in self.edges:
            edge_hops.append(edge.hops)
        entry = self.entry_pool
        return routes_build.generated(
            tuple(pool_hops), tuple(edge_hops), entry.pool
        )

    def formation_paths(self) -> tuple:
        """The seats a round passes from the controller to each pool.

        The entry pool reads the store the controller writes; an edge's
        target reads its own store, which takes an escalated window's
        rounds from the store the edge carries them from, so a round
        reaches a pool down a chain through every store before it.
        """
        entry = self.entry_pool
        entry_stores = (CONTROLLER_SEAT, entry.store)
        stores_by_pool = {entry.pool: entry_stores}
        entry_path = (*entry_stores, entry.names.seat)
        paths = [entry_path]
        for edge in self.edges:
            hops = edge.hops
            target = self.pool(hops.target)
            carried = stores_by_pool[hops.source]
            stores = (*carried, target.store)
            stores_by_pool[hops.target] = stores
            path = (*stores, target.names.seat)
            paths.append(path)
        return tuple(paths)


# A legacy machine's names, by slot. The pool that decodes the windows
# is always the primary one, its manager's pool "default", even when
# the strong tier is that pool (a strong-only run): its seeds and
# statistics stay where a legacy run has them.
WEAK_STORE = "weak_syndrome_buffer"
STRONG_STORE = "strong_syndrome_buffer"
PRIMARY_UNIT_SEED = "primary_decoder"
PRIMARY_MANAGER_SEED = "decoder_manager"
STRONG_SEAT = "strong_decoder"
STRONG_NAMES = PoolNames(
    seat=STRONG_SEAT,
    queue=decode_queue.STRONG_POOL,
    statistics_path=STRONG_SEAT,
    unit_seed=STRONG_SEAT,
    manager_seed="strong_decoder_manager",
)
# the edge's statistics path, the switching settings' own, and the
# root its verdict's seed hashes
LEGACY_EDGE_NAME = "switching"
LEGACY_POLICY_SEED = "escalation_policy"
# where every round's path to a decoder starts
CONTROLLER_SEAT = "controller"
# how each legacy store takes its rounds, by the names its log and
# copies use: the weak store written by the controller, the strong one
# landed, from the controller or carried up with an escalation
CONTROLLER_TO_WEAK_STORE = receiver_module.Hop(
    transfer_records.CONTROLLER_TO_WEAK_BUFFER, "controller assembler"
)
CONTROLLER_TO_STRONG_STORE = receiver_module.Hop(
    transfer_records.CONTROLLER_TO_STRONG_BUFFER, "controller assembler"
)
ESCALATION_TO_STRONG_STORE = receiver_module.Hop(
    transfer_records.WEAK_DECODER_TO_STRONG_DECODER, "weak syndrome buffer"
)
WEAK_STORE_INTAKE = receiver_module.StoreIntake(
    name=log_sources.WEAK_BUFFER,
    seat=WEAK_STORE,
    shape=receiver_module.WRITTEN,
    controller_write=CONTROLLER_TO_WEAK_STORE,
)
STRONG_STORE_INTAKE = receiver_module.StoreIntake(
    name=log_sources.STRONG_BUFFER,
    seat=STRONG_STORE,
    shape=receiver_module.LANDED,
    controller_write=CONTROLLER_TO_STRONG_STORE,
    escalation=ESCALATION_TO_STRONG_STORE,
)


def legacy_layout(settings: Any) -> Layout:
    """The layout of a MachineSettings record, every name its slot's.

    settings is Any: build sits below the settings package, which
    imports it.

    The weak pool decodes the windows when it is set, else the strong
    one; a switching run adds the strong pool and the edge to it. A run
    with no decoder has one idle pool and no store.
    """
    window_tier = settings.window_tier
    window_decoder = settings.decoder_settings_for(window_tier.word)
    entry = _legacy_entry_pool(window_tier, window_decoder)
    pools = [entry]
    stores = []
    if window_decoder is not None:
        entry_store = _legacy_entry_store(settings, entry)
        stores.append(entry_store)
    edges = []
    switching = settings.switching
    if switching is not None:
        strong = _legacy_strong_pool(settings.strong_decoder)
        pools.append(strong)
        strong_store = _legacy_store(
            settings.strong_syndrome_buffer, STRONG_STORE, strong
        )
        stores.append(strong_store)
        hops = routes_build.legacy_edge(switching.strong_answer_route)
        edge = EdgePlan(
            LEGACY_EDGE_NAME, switching, hops, WEAK_STORE, LEGACY_POLICY_SEED
        )
        edges.append(edge)
    return Layout(tuple(pools), tuple(stores), tuple(edges))


def _legacy_entry_pool(
    window_tier: window_records.PoolId,
    window_decoder: Optional[decoder_settings.DecoderPoolSettings],
) -> PoolPlan:
    """The pool of the window tier, under the primary names."""
    seat = f"{window_tier.word}_decoder"
    names = PoolNames(
        seat=seat,
        queue=decode_queue.DEFAULT_POOL,
        statistics_path=seat,
        unit_seed=PRIMARY_UNIT_SEED,
        manager_seed=PRIMARY_MANAGER_SEED,
    )
    hops = routes_build.LEGACY_WEAK_POOL_HOPS
    store = WEAK_STORE
    if window_tier == window_records.STRONG_POOL_ID:
        hops = routes_build.LEGACY_STRONG_POOL_HOPS
        store = STRONG_STORE
    return PoolPlan(hops, window_decoder, names, store)


def _legacy_strong_pool(
    strong_decoder: Optional[decoder_settings.DecoderPoolSettings],
) -> PoolPlan:
    hops = routes_build.LEGACY_STRONG_POOL_HOPS
    return PoolPlan(hops, strong_decoder, STRONG_NAMES, STRONG_STORE)


def _legacy_entry_store(settings, entry: PoolPlan) -> StorePlan:
    """The store the window tier reads: the weak one, or the strong one."""
    if entry.store == STRONG_STORE:
        return _legacy_store(
            settings.strong_syndrome_buffer, STRONG_STORE, entry
        )
    return _legacy_store(settings.weak_syndrome_buffer, WEAK_STORE, entry)


def _legacy_store(store_settings, name: str, reader: PoolPlan) -> StorePlan:
    """A legacy store, read by one pool, under its slot's names."""
    reads_in_place = _reads_in_place(reader.settings)
    if name == WEAK_STORE:
        return StorePlan(
            name=name,
            settings=store_settings,
            intake=WEAK_STORE_INTAKE,
            label="weak syndrome buffer",
            read_hop=transfer_records.WEAK_BUFFER_TO_WEAK_DECODER,
            reader=reader.names.seat,
            reads_in_place=reads_in_place,
            takes_machine_clock=True,
        )
    return StorePlan(
        name=name,
        settings=store_settings,
        intake=STRONG_STORE_INTAKE,
        label="strong syndrome buffer",
        read_hop=transfer_records.STRONG_BUFFER_TO_STRONG_DECODER,
        reader=reader.names.seat,
        reads_in_place=reads_in_place,
        takes_machine_clock=False,
    )


def _reads_in_place(
    pool_settings: Optional[decoder_settings.DecoderPoolSettings],
) -> bool:
    """Whether a pool's units read their rounds where the store keeps them."""
    if pool_settings is None:
        return False
    return not pool_settings.copies_input
