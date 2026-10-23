"""The layout a machine is built from: its pools, stores and escalate edges.

The build loops over these, entry first, then each edge's target in
graph order, so a machine of any number of pools builds through one
code path. Each record carries the names its components go by: the
word its results write, its statistics path, its seed names, its log
labels and its hops (topology/graph_layout.py names them by part).

The order is part of a run's results: the entry store is built first,
since a store's tick-end listener runs in the order the stores were
built (syndrome_buffer/syndrome_buffer.py), and the entry pool's jobs
are admitted before any target's (windows/decode_requests.py).
"""

import dataclasses
from typing import Any, Optional

import decsim.build.routes as routes_build
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.links.settings as link_settings
import decsim.records.routes as route_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module

# where every round's path to a decoder starts
CONTROLLER_SEAT = "controller"


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
    on the machine's clock when it names none of its own. other_readers
    are the other entry pools a router after this store feeds from it,
    each a StoreReader.
    """

    name: str
    settings: Any
    intake: receiver_module.StoreIntake
    label: str
    read_hop: transfer_records.LinkPath
    reader: str
    reads_in_place: bool
    takes_machine_clock: bool
    other_readers: tuple = ()


@dataclasses.dataclass(frozen=True)
class StoreReader:
    """One more pool reading a store: its pool, hop, seat and read.

    A router after a store has its entry pools share it, each reading it
    over a hop of its own, as VQD's decoders each take their syndromes from
    the Arbiter (Maurya and Tannu 2406.17995v2 Sec. 5).
    """

    pool: window_records.PoolId
    read_hop: transfer_records.LinkPath
    reader: str
    reads_in_place: bool


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

    The entry pools decode the planner's windows, each from the store
    it reads: one store they share, or one each the controller writes.
    router, None for most machines, sends each window to one of
    entry_count entry pools, the first pools laid out
    (windows/routers.py). card prices every hop the
    machine rides. frame_site is where the Pauli frame sits, which a
    message to it crosses the wall to reach from any other site.
    """

    pools: tuple
    stores: tuple
    edges: tuple
    card: link_settings.FabricSettings
    router: Optional[Any] = None
    entry_count: int = 1
    frame_site: str = route_records.FRIDGE

    @property
    def entry_pool(self) -> PoolPlan:
        """The first pool that decodes the planner's windows."""
        return self.pools[0]

    @property
    def entry_pools(self) -> tuple:
        """Every pool that decodes the planner's windows, first first."""
        return self.pools[: self.entry_count]

    def is_entry(self, pool: window_records.PoolId) -> bool:
        """Whether the pool decodes the planner's windows first."""
        entry_ids = [plan.pool for plan in self.entry_pools]
        return pool in entry_ids

    def edge_leaving(self, pool: window_records.PoolId) -> Optional[EdgePlan]:
        """The edge a pool escalates on; None for a pool that keeps all."""
        for edge in self.edges:
            if edge.hops.source == pool:
                return edge
        return None

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

    def tier_pools(self) -> tuple:
        """The weak tier's pool and the strong tier's, either None if absent.

        The weak tier is the entry pool and the strong one its entry
        edge's target, whatever their parts are named. An entry pool whose
        rounds land in its store, as a host pool's do, is a strong tier
        with no weak one. A run with no store has neither. The retention
        and the reports read the two tiers off this one rule.
        """
        if not self.stores:
            return (None, None)
        entry = self.entry_pool
        entry_store = self.store(entry.store)
        if entry_store.intake.shape == receiver_module.LANDED:
            return (None, entry.pool)
        if not self.edges:
            return (entry.pool, None)
        entry_edge = self.edges[0]
        return (entry.pool, entry_edge.hops.target)

    def input_hop_by_pool(self) -> dict:
        """The hop each pool's input rides: its store's read hop.

        A pool with no store, a run that decodes nothing, has none.
        """
        read_hop_by_store = {}
        for store in self.stores:
            read_hop_by_store[store.name] = store.read_hop
        hops = {}
        for plan in self.pools:
            if plan.store in read_hop_by_store:
                hops[plan.pool] = read_hop_by_store[plan.store]
        for store in self.stores:
            for reader in store.other_readers:
                hops[reader.pool] = reader.read_hop
        return hops

    def landed_seats(self) -> tuple:
        """Each store a round lands in, and each pool that reads one."""
        seats = []
        for store in self.stores:
            if store.intake.shape == receiver_module.LANDED:
                seats.append(store.name)
        landed = set(seats)
        for plan in self.pools:
            if plan.store in landed:
                seats.append(plan.names.seat)
        return tuple(seats)

    def controller_write_hops(self) -> tuple:
        """The hop the controller writes each written store over."""
        hops = []
        for store in self.stores:
            intake = store.intake
            if intake.shape == receiver_module.WRITTEN:
                hops.append(intake.controller_write.path_name)
        return tuple(hops)

    def entry_stores(self) -> tuple:
        """Each store an entry pool reads, once, the first entry's first."""
        names = []
        for entry in self.entry_pools:
            if entry.store not in names:
                names.append(entry.store)
        return tuple(names)

    def entry_switching(
        self,
    ) -> Optional[escalation_settings.SwitchingSettings]:
        """The first entry edge's switching record; None for none.

        Every edge out of an entry pool agrees with it on what the plan
        and a shot's record read (build/escalation.py).
        """
        if not self.edges:
            return None
        entry_edge = self.edges[0]
        return entry_edge.settings

    def factory_pool(self) -> window_records.PoolId:
        """The pool the magic-state factory's decodes go to.

        The router's factory_pool, or the one entry pool of a machine
        with one; a machine of several entry pools must name it.
        """
        name = None
        if self.router is not None:
            name = self.router.factory_pool
        if name is None:
            return self._only_entry_pool("router.factory_pool")
        pool = window_records.PoolId(name)
        _refuse_a_pool_not_held(self, pool)
        return pool

    def _only_entry_pool(self, setting: str) -> window_records.PoolId:
        """The one entry pool; refused when there are several."""
        if self.entry_count == 1:
            return self.entry_pool.pool
        names = [entry.pool.word for entry in self.entry_pools]
        raise ValueError(
            f"{setting} names no pool, and this machine has "
            f"{self.entry_count} entry pools {names}; name the one that "
            "serves the magic-state factory's decodes"
        )

    def entry_escalation_hops(self) -> tuple:
        """The hop each edge out of an entry pool escalates a window over."""
        hops = []
        for edge in self.edges:
            if self.is_entry(edge.hops.source):
                hops.append(edge.hops.forward)
        return tuple(hops)

    def statistics_path_by_pool(self) -> dict:
        """The path each pool's statistics sit at."""
        paths = {}
        for plan in self.pools:
            paths[plan.pool] = plan.names.statistics_path
        return paths

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
        return routes_build.generated(
            tuple(pool_hops), tuple(edge_hops), self.frame_site
        )

    def formation_paths(self) -> tuple:
        """The seats a round passes from the controller to each pool.

        Each entry pool reads a store the controller writes; an edge's
        target reads its own store, which takes an escalated window's
        rounds from the store the edge carries them from, so a round
        reaches a pool down its entry pool's chain through every store
        before it. A target several chains share is reached once down
        each, and two entry pools that read one store give one path.
        """
        paths = []
        for entry in self.entry_pools:
            stores = (CONTROLLER_SEAT, entry.store)
            _add_once(paths, (*stores, entry.names.seat))
            edge = self.edge_leaving(entry.pool)
            while edge is not None:
                target = self.pool(edge.hops.target)
                stores = (*stores, target.store)
                _add_once(paths, (*stores, target.names.seat))
                edge = self.edge_leaving(target.pool)
        return tuple(paths)


def _add_once(paths: list, path: tuple) -> None:
    """Append the path unless it is already listed."""
    if path in paths:
        return
    paths.append(path)


def _refuse_a_pool_not_held(
    layout: Layout, pool: window_records.PoolId
) -> None:
    for plan in layout.pools:
        if plan.pool == pool:
            return
    names = [plan.pool.word for plan in layout.pools]
    raise ValueError(
        f"router.factory_pool names {pool.word!r}, which is no pool of "
        f"this machine; its pools are {names}"
    )
