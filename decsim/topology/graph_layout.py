"""A graph laid out as pools, stores and escalate edges.

Every machine is built from its graph's own parts. Each pool, store
and edge goes by its part's name, and each hop by the parts at its
ends, <source part>_to_<target part>, with its role
(records/routes.py HopRole): the wire's own link prices it, else the
machine's fabric rule, on the reference card's payload rule for its
role. The pools form a tree: one pool decodes
the planner's windows, the entry, and every other pool is the target of
one switching part, an escalate edge, whose results come from another
pool. They are laid out entry first, then each edge's target in the
order the switching parts are written.

The entry pool reads the store the controller writes; an edge's target
reads its own store, which takes an escalated window's rounds from the
store its source pool reads. A router between the entry store and
several pools makes each of them an entry pool: they share that store,
each over a hop of its own, and the first wired is the entry
(windows/routers.py). Two pools no edge joins trade boundaries on a hop
of their own. A store no pool reads is not built, and a machine with no
pool decodes nothing on one idle pool.

The entry pool, its edge and that edge's target draw their seeds at
roots kept by role, and every other part at its own name: a pool added
past the entry moves no other part's seed, so a chain whose deep pool
is never asked runs as its two-pool machine, shot for shot.
"""

import dataclasses
from typing import Any, Optional

import decsim.build.layout as layout_module
import decsim.records.routes as route_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.refusals as refusals
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph

# the hops every machine rides between its parts that are held once,
# by the kinds at their ends, each with its role
SHARED_HOPS = {
    ("qpu", "controller"): (
        transfer_records.QPU_TO_CONTROLLER,
        route_records.HopRole(route_records.READOUT),
    ),
    ("controller", "qpu"): (
        transfer_records.CONTROLLER_TO_QPU,
        route_records.HopRole(route_records.COMMAND),
    ),
    ("pauli_frame", "controller"): (
        transfer_records.FRAME_TO_CONTROLLER,
        route_records.HopRole(route_records.DECISION),
    ),
}
# The seed roots of the entry pool, its edge and that edge's target, by
# role. The strings are the ones every recorded shot hashed, so a shot
# draws as it always has; every other part draws at its own name.
ENTRY_UNIT_SEED = "primary_decoder"
ENTRY_MANAGER_SEED = "decoder_manager"
ENTRY_TARGET_UNIT_SEED = "strong_decoder"
ENTRY_TARGET_MANAGER_SEED = "strong_decoder_manager"
ENTRY_EDGE_POLICY_SEED = "escalation_policy"
_ENTRY_SEED_ROOTS = (ENTRY_UNIT_SEED, ENTRY_MANAGER_SEED)
_ENTRY_TARGET_SEED_ROOTS = (ENTRY_TARGET_UNIT_SEED, ENTRY_TARGET_MANAGER_SEED)
# the pool and the store a machine that decodes nothing names, though
# neither decodes nor holds a round
IDLE_POOL = "idle"
IDLE_STORE = "idle_buffer"
# the part that writes the stores, as a hop into a store names it
CONTROLLER_LABEL = "controller assembler"
_FORWARD = route_records.HopRole(route_records.FORWARD)
_REVERSE = route_records.HopRole(route_records.REVERSE)
_POOL_KIND = "decoder_pool"
_STORE_KIND = "syndrome_buffer"
_SWITCHING_KIND = "switching"
_ROUTER_KIND = "router"


def layout_of(machine: graph.Machine) -> layout_module.Layout:
    """The graph's pools, stores and edges, each under its part's name.

    machine is at its task (graph.Machine.at_task), each edge's
    threshold read; the card prices every hop.
    """
    shape = _shape(machine)
    card = _card(shape)
    frame_name = _frame_name(machine)
    if not shape.pools:
        idle = _idle_pool(frame_name)
        return layout_module.Layout((idle,), (), (), card)
    pools = []
    stores = []
    for pool in shape.pools:
        pool_plan = _pool_plan(pool, shape, frame_name)
        pools.append(pool_plan)
        if pool in shape.entries[1:]:
            continue
        store_plan = _store_plan(shape, pool)
        stores.append(store_plan)
    stores[0] = _with_other_readers(stores[0], shape)
    edges = []
    for edge in shape.edges:
        carried_from = shape.store_by_pool[edge.source.name]
        policy_seed = _policy_seed(shape, edge)
        edge_plan = _edge_plan(edge, carried_from.name, policy_seed)
        edges.append(edge_plan)
    router = shape.router_settings()
    entry_count = len(shape.entries)
    return layout_module.Layout(
        tuple(pools),
        tuple(stores),
        tuple(edges),
        card,
        router=router,
        entry_count=entry_count,
    )


@dataclasses.dataclass(frozen=True)
class _Edge:
    """One switching part, the pool its results come from, its target."""

    part: graph.Part
    source: graph.Part
    target: graph.Part


@dataclasses.dataclass(frozen=True)
class _Shape:
    """A graph's pools in build order, its edges, each pool's store.

    store_by_pool maps a pool's name to the store part it reads; entries
    are the pools that decode the planner's windows, first first, and
    router the part that feeds them, None for one entry pool.
    """

    machine: graph.Machine
    pools: tuple
    edges: tuple
    store_by_pool: dict
    entries: tuple
    router: Optional[graph.Part] = None

    @property
    def entry(self) -> graph.Part:
        """The first pool that decodes the planner's windows."""
        return self.pools[0]

    def router_settings(self) -> Optional[Any]:
        """The router's record; None for a machine with no router."""
        if self.router is None:
            return None
        return self.router.settings

    def entry_store(self) -> graph.Part:
        """The store the entry pool reads."""
        entry = self.entry
        return self.store_by_pool[entry.name]


def _shape(machine: graph.Machine) -> _Shape:
    """The graph's pools from its entry, its edges, each pool's store."""
    _refuse_unbound_ports(machine)
    pools = machine.parts_of(_POOL_KIND)
    switching_parts = machine.parts_of(_SWITCHING_KIND)
    edges = []
    for part in switching_parts:
        edge = _edge(machine, part)
        edges.append(edge)
    router = _router(machine)
    if not pools:
        return _Shape(machine, (), (), {}, ())
    entries = _entry_pools(machine, pools, edges, router)
    ordered_pools, ordered_edges = _from_the_entries(entries, pools, edges)
    _refuse_an_edge_from_a_later_entry(entries, ordered_edges)
    store_by_pool = {}
    for pool in ordered_pools:
        store_by_pool[pool.name] = _store_read_by(machine, pool)
    return _Shape(
        machine, ordered_pools, ordered_edges, store_by_pool, entries, router
    )


def _refuse_unbound_ports(machine: graph.Machine) -> None:
    unbound = machine.unbound_ports()
    if unbound:
        raise graph.GraphError(f"required ports with no wire: {unbound}")


def _edge(machine: graph.Machine, part: graph.Part) -> _Edge:
    """A switching part, from the pool its results port hears."""
    results = f"{part.name}.results"
    source = None
    for wire in machine.wires:
        if wire.target == results:
            source = _part_at(machine, wire.source)
    if source is None:
        raise graph.GraphError(
            f"{results} hears no pool: an escalate edge decides on the "
            "results of the pool whose escalate port is wired to it"
        )
    target = _redecode_target(machine, part.name)
    return _Edge(part, source, target)


def _redecode_target(machine: graph.Machine, switching_name: str) -> graph.Part:
    """The pool a switching part's re-decodes reach; refused when none."""
    written = f"{switching_name}.redecode"
    for wire in machine.wires:
        if wire.source == written:
            return _part_at(machine, wire.target)
    raise graph.GraphError(
        f"{written} reaches no pool: an escalate edge re-decodes on the "
        "pool its redecode port is wired to"
    )


def _router(machine: graph.Machine) -> Optional[graph.Part]:
    """The machine's router; None for none, and at most one."""
    routers = machine.parts_of(_ROUTER_KIND)
    if not routers:
        return None
    if len(routers) > 1:
        names = [part.name for part in routers]
        raise graph.GraphError(
            f"the routers {names}: a machine of more than one router is "
            f"not implemented yet; {refusals.REFUSALS}"
        )
    return routers[0]


def _entry_pools(
    machine: graph.Machine,
    pools: list,
    edges: list,
    router: Optional[graph.Part],
) -> tuple:
    """The pools no edge reaches: one, or each a router feeds, in its order."""
    entries = _unreached(pools, edges)
    if router is not None:
        return _routed_entries(machine, router, entries)
    if len(entries) == 1:
        return tuple(entries)
    names = [pool.name for pool in entries]
    raise graph.GraphError(
        f"the pools {names} are reached by no escalate edge: a machine "
        "decodes the planner's windows on one pool, or on each pool a "
        "router feeds, and every other pool is the target of one edge"
    )


def _unreached(pools: list, edges: list) -> list:
    """The pools no edge reaches, in build order."""
    targets = {edge.target.name for edge in edges}
    return [pool for pool in pools if pool.name not in targets]


def _routed_entries(
    machine: graph.Machine, router: graph.Part, entries: list
) -> tuple:
    """The pools the router feeds, once each pool no edge reaches is one."""
    routed = _routed_pools(machine, router)
    _refuse_entries_the_router_misses(entries, routed)
    return routed


def _routed_pools(machine: graph.Machine, router: graph.Part) -> tuple:
    """The pools the router's windows port reaches, in wire order."""
    windows = f"{router.name}.windows"
    routed = []
    for wire in machine.wires:
        if wire.source == windows:
            pool = _part_at(machine, wire.target)
            routed.append(pool)
    return tuple(routed)


def _refuse_entries_the_router_misses(entries: list, routed: tuple) -> None:
    routed_names = {pool.name for pool in routed}
    missed = [pool.name for pool in entries if pool.name not in routed_names]
    if not missed:
        return
    raise graph.GraphError(
        f"the pools {missed} are reached by no escalate edge and no "
        "router: with a router, the planner's windows are decoded on the "
        "pools it feeds"
    )


def _from_the_entries(entries: tuple, pools: list, edges: list) -> tuple:
    """The pools and edges from the entries, each edge in written order."""
    ordered_pools = list(entries)
    ordered_edges = []
    for pool in ordered_pools:
        leaving = _edges_leaving(pool, edges)
        ordered_edges.extend(leaving)
        for edge in leaving:
            ordered_pools.append(edge.target)
    _refuse_the_unreached(entries[0], pools, ordered_pools)
    return tuple(ordered_pools), tuple(ordered_edges)


def _refuse_an_edge_from_a_later_entry(entries: tuple, edges: tuple) -> None:
    """Only the first entry pool escalates yet."""
    later = {pool.name for pool in entries[1:]}
    for edge in edges:
        if edge.source.name not in later:
            continue
        raise graph.GraphError(
            f"{edge.part.name} escalates from {edge.source.name}, an entry "
            "pool past the first: an escalate edge from such a pool is not "
            "implemented yet, its boundaries being the first's; "
            f"{refusals.REFUSALS}"
        )


def _edges_leaving(pool: graph.Part, edges: list) -> list:
    return [edge for edge in edges if edge.source.name == pool.name]


def _refuse_the_unreached(
    entry: graph.Part, pools: list, reached_pools: list
) -> None:
    """Every pool is reached from the entry; one that is not is a circle."""
    reached = {pool.name for pool in reached_pools}
    cut_off = [pool.name for pool in pools if pool.name not in reached]
    if not cut_off:
        return
    raise graph.GraphError(
        f"the pools {cut_off} escalate in a circle the entry pool "
        f"{entry.name} never reaches"
    )


def _store_read_by(machine: graph.Machine, pool: graph.Part) -> graph.Part:
    """The store whose reads port reaches the pool's rounds.

    A router's pool reads the store whose reads reach the router.
    """
    rounds = f"{pool.name}.rounds"
    for wire in machine.wires:
        if wire.target != rounds:
            continue
        source = _part_at(machine, wire.source)
        kind = source.kind
        if kind.name == _STORE_KIND:
            return source
        if kind.name == _ROUTER_KIND:
            return _store_read_by(machine, source)
    raise graph.GraphError(
        f"{rounds} is wired to no store: a pool reads its rounds from "
        "a syndrome buffer"
    )


def _part_at(machine: graph.Machine, written: str) -> graph.Part:
    part_name, _, _ = written.partition(".")
    return machine.part(part_name)


def _frame_name(machine: graph.Machine) -> str:
    """The frame's part name; "frame" for a machine with none."""
    frames = machine.parts_of("pauli_frame")
    if not frames:
        return "frame"
    frame = frames[0]
    return frame.name


def _site(part: graph.Part) -> str:
    """Where a part of a new-shape graph sits; the fridge when unsaid."""
    if part.placed is None:
        return fabrics.FRIDGE
    return part.placed


def _hop(source_name: str, target_name: str) -> str:
    return route_records.hop_between(source_name, target_name)


def _intake_shape(shape: _Shape, pool: graph.Part) -> str:
    """How a pool's store takes its rounds.

    The controller writes the entry store, unless the first entry pool
    sits on the host, which reads a store the rounds land in; an edge's
    target store takes its rounds as they land.
    """
    if pool not in shape.entries:
        return receiver_module.LANDED
    pool = shape.entry
    if _site(pool) == fabrics.HOST:
        return receiver_module.LANDED
    return receiver_module.WRITTEN


def _controller_name(machine: graph.Machine) -> str:
    controllers = machine.parts_of("controller")
    controller = controllers[0]
    return controller.name


def _pool_plan(
    pool: graph.Part, shape: _Shape, frame_name: str
) -> layout_module.PoolPlan:
    """A pool under its part's name: its seat, queue, seeds and hops."""
    name = pool.name
    unit_seed, manager_seed = _seed_roots(shape, pool)
    names = layout_module.PoolNames(
        seat=name,
        queue=name,
        statistics_path=name,
        unit_seed=unit_seed,
        manager_seed=manager_seed,
    )
    pool_id = window_records.PoolId(name)
    site = _site(pool)
    answer = _hop(name, frame_name)
    same_pool = _hop(name, name)
    hops = route_records.PoolHops(
        pool=pool_id, site=site, answer=answer, same_pool=same_pool
    )
    store = shape.store_by_pool[name]
    return layout_module.PoolPlan(hops, pool.settings, names, store.name)


def _idle_pool(frame_name: str) -> layout_module.PoolPlan:
    """The one idle pool of a machine that decodes nothing, and no store."""
    names = layout_module.PoolNames(
        seat=IDLE_POOL,
        queue=IDLE_POOL,
        statistics_path=IDLE_POOL,
        unit_seed=ENTRY_UNIT_SEED,
        manager_seed=ENTRY_MANAGER_SEED,
    )
    pool_id = window_records.PoolId(IDLE_POOL)
    answer = _hop(IDLE_POOL, frame_name)
    same_pool = _hop(IDLE_POOL, IDLE_POOL)
    hops = route_records.PoolHops(
        pool=pool_id, site=fabrics.FRIDGE, answer=answer, same_pool=same_pool
    )
    return layout_module.PoolPlan(hops, None, names, IDLE_STORE)


def _seed_roots(shape: _Shape, pool: graph.Part) -> tuple:
    """A pool's unit and manager seed roots: its slot's, else its own."""
    if pool.name == shape.entry.name:
        return _ENTRY_SEED_ROOTS
    if shape.edges and pool.name == shape.edges[0].target.name:
        return _ENTRY_TARGET_SEED_ROOTS
    return (pool.name, f"{pool.name}_manager")


def _policy_seed(shape: _Shape, edge: _Edge) -> str:
    """An edge's policy seed root: the entry edge's by role, else its own."""
    if edge is shape.edges[0]:
        return ENTRY_EDGE_POLICY_SEED
    return f"{edge.part.name}_policy"


def _store_plan(shape: _Shape, pool: graph.Part) -> layout_module.StorePlan:
    """The store a pool reads, under its part's name."""
    store = shape.store_by_pool[pool.name]
    intake_shape = _intake_shape(shape, pool)
    controller_name = _controller_name(shape.machine)
    write_hop = _hop(controller_name, store.name)
    controller_write = receiver_module.Hop(write_hop, CONTROLLER_LABEL)
    escalation = _escalation_hop(shape, pool)
    intake = receiver_module.StoreIntake(
        name=store.name,
        seat=store.name,
        shape=intake_shape,
        controller_write=controller_write,
        escalation=escalation,
    )
    pool_settings = pool.settings
    read_hop = _hop(store.name, pool.name)
    reads_in_place = not pool_settings.copies_input
    is_written = intake_shape == receiver_module.WRITTEN
    return layout_module.StorePlan(
        name=store.name,
        settings=store.settings,
        intake=intake,
        label=store.name,
        read_hop=read_hop,
        reader=pool.name,
        reads_in_place=reads_in_place,
        takes_machine_clock=is_written,
    )


def _with_other_readers(
    entry_store: layout_module.StorePlan, shape: _Shape
) -> layout_module.StorePlan:
    """The entry store, read by each entry pool past the first too."""
    readers = []
    for pool in shape.entries[1:]:
        pool_id = window_records.PoolId(pool.name)
        read_hop = _hop(entry_store.name, pool.name)
        reader = layout_module.StoreReader(
            pool=pool_id,
            read_hop=read_hop,
            reader=pool.name,
            reads_in_place=not pool.settings.copies_input,
        )
        readers.append(reader)
    return dataclasses.replace(entry_store, other_readers=tuple(readers))


def _escalation_hop(
    shape: _Shape, pool: graph.Part
) -> Optional[receiver_module.Hop]:
    """The hop an escalated region lands in a target's store over.

    The region is carried from the store its source pool reads.
    """
    entering = [edge for edge in shape.edges if edge.target.name == pool.name]
    if not entering:
        return None
    edge = entering[0]
    carried_from = shape.store_by_pool[edge.source.name]
    forward = _hop(edge.source.name, pool.name)
    return receiver_module.Hop(forward, carried_from.name)


def _edge_plan(
    edge: _Edge, carried_from: str, policy_seed: str
) -> layout_module.EdgePlan:
    """A switching part's edge."""
    settings = edge.part.settings
    source = window_records.PoolId(edge.source.name)
    target = window_records.PoolId(edge.target.name)
    forward = _hop(edge.source.name, edge.target.name)
    reverse = _hop(edge.target.name, edge.source.name)
    hops = route_records.EdgeHops(
        source=source,
        target=target,
        forward=forward,
        reverse=reverse,
        answer_route=settings.strong_answer_route,
    )
    name = edge.part.name
    return layout_module.EdgePlan(
        name, settings, hops, carried_from, policy_seed
    )


def _card(shape: _Shape) -> Any:
    """Every hop the graph rides, priced: the shared hops, then its own."""
    machine = shape.machine
    shared_routes = _shared_routes(machine)
    own_routes = _own_routes(shape)
    routes = (*shared_routes, *own_routes)
    wire_links = _wire_links(shape)
    return fabrics.card(
        machine.fabric,
        routes,
        wire_links,
        machine.clock,
        machine.readout_routes,
    )


def _shared_routes(machine: graph.Machine) -> tuple:
    """The readout, the command and the decision, at their parts' sites.

    A part held once sits in the fridge unless it says otherwise; a
    machine with no frame still decides on the controller's chip.
    """
    site_by_kind = {}
    for part in machine.parts:
        kind = part.kind
        if kind is None:
            continue
        site_by_kind[kind.name] = _site(part)
    routes = []
    for (source_kind, target_kind), (path, role) in SHARED_HOPS.items():
        source_site = site_by_kind.get(source_kind, fabrics.FRIDGE)
        target_site = site_by_kind.get(target_kind, fabrics.FRIDGE)
        route = fabrics.Route(path, source_site, target_site, role)
        routes.append(route)
    return tuple(routes)


def _own_routes(shape: _Shape) -> tuple:
    """The graph's own hops, each with its ends' sites and its role."""
    machine = shape.machine
    controller_name = _controller_name(machine)
    frame_name = _frame_name(machine)
    routes = []
    for pool in shape.pools:
        pool_routes = _pool_routes(shape, pool, controller_name, frame_name)
        routes.extend(pool_routes)
    for edge in shape.edges:
        edge_routes = _edge_routes(edge)
        routes.extend(edge_routes)
    unjoined_routes = _unjoined_routes(shape)
    routes.extend(unjoined_routes)
    return tuple(routes)


def _pool_routes(
    shape: _Shape, pool: graph.Part, controller_name: str, frame_name: str
) -> tuple:
    """A pool's write into its store, its read, its answer, its hand-on."""
    store = shape.store_by_pool[pool.name]
    intake_shape = _intake_shape(shape, pool)
    pool_site = _site(pool)
    store_site = _site(store)
    read_role = route_records.HopRole(route_records.READ, intake_shape)
    answer_role = route_records.HopRole(route_records.ANSWER, pool_site)
    same_pool_role = route_records.HopRole(route_records.SAME_POOL, pool_site)
    read = _route((store.name, pool.name), (store_site, pool_site), read_role)
    answer = _route(
        (pool.name, frame_name), (pool_site, fabrics.FRIDGE), answer_role
    )
    same_pool = _route(
        (pool.name, pool.name), (pool_site, pool_site), same_pool_role
    )
    if pool in shape.entries[1:]:
        # the entry store is written once, on the first entry's route
        return (read, answer, same_pool)
    write_role = route_records.HopRole(route_records.WRITE, intake_shape)
    write = _route(
        (controller_name, store.name), (fabrics.FRIDGE, store_site), write_role
    )
    return (write, read, answer, same_pool)


def _edge_routes(edge: _Edge) -> tuple:
    """An edge's forward and reverse hops between its pools' sites."""
    source_site = _site(edge.source)
    target_site = _site(edge.target)
    source_name = edge.source.name
    target_name = edge.target.name
    forward = _route(
        (source_name, target_name),
        (source_site, target_site),
        _FORWARD,
    )
    reverse = _route(
        (target_name, source_name),
        (target_site, source_site),
        _REVERSE,
    )
    return (forward, reverse)


def _unjoined_routes(shape: _Shape) -> list:
    """A boundary hop between each two pools no edge joins.

    It is priced as an edge's: forward to a pool laid out later, reverse
    to one laid out earlier.
    """
    joined = set()
    for edge in shape.edges:
        joined.add((edge.source.name, edge.target.name))
        joined.add((edge.target.name, edge.source.name))
    routes = []
    for sender_index, sender in enumerate(shape.pools):
        sender_site = _site(sender)
        receivers = _unjoined_receivers(shape, sender, joined)
        for receiver_index, receiver in receivers:
            role = _boundary_role(sender_index, receiver_index)
            ends = (sender.name, receiver.name)
            receiver_site = _site(receiver)
            sites = (sender_site, receiver_site)
            route = _route(ends, sites, role)
            routes.append(route)
    return routes


def _unjoined_receivers(shape: _Shape, sender: graph.Part, joined: set) -> list:
    """The other pools no edge joins to sender, with their build index."""
    receivers = []
    for index, receiver in enumerate(shape.pools):
        pair = (sender.name, receiver.name)
        if receiver.name == sender.name or pair in joined:
            continue
        receivers.append((index, receiver))
    return receivers


def _boundary_role(
    sender_index: int, receiver_index: int
) -> route_records.HopRole:
    if receiver_index > sender_index:
        return _FORWARD
    return _REVERSE


def _route(
    ends: tuple, sites: tuple, role: route_records.HopRole
) -> fabrics.Route:
    """The hop between two parts, at their sites, with its role."""
    source_name, target_name = ends
    source_site, target_site = sites
    path = _hop(source_name, target_name)
    return fabrics.Route(path, source_site, target_site, role)


def _wire_links(shape: _Shape) -> dict:
    """Each wire's own link by the hop it prices.

    A wire between parts held once prices its shared hop; a wire to or
    from a switching part prices its edge's forward hop, and one from a
    router the read of the store that feeds it.
    """
    machine = shape.machine
    target_by_switching = {}
    source_by_switching = {}
    for edge in shape.edges:
        target_by_switching[edge.part.name] = edge.target.name
        source_by_switching[edge.part.name] = edge.source.name
    if shape.router is not None:
        # a wire from the router prices the entry store's read
        entry_store = shape.entry_store()
        source_by_switching[shape.router.name] = entry_store.name
    links = {}
    for wire in machine.wires:
        if wire.link is None:
            continue
        source = _part_at(machine, wire.source)
        target = _part_at(machine, wire.target)
        source_name = source_by_switching.get(source.name, source.name)
        target_name = target_by_switching.get(target.name, target.name)
        hop = _wire_hop(source, target, source_name, target_name)
        links[hop] = wire.link
    return links


def _wire_hop(
    source: graph.Part, target: graph.Part, source_name: str, target_name: str
) -> str:
    """The hop a wire prices: a shared hop by its kinds, else by its parts."""
    ends = (source.kind.name, target.kind.name)
    shared = SHARED_HOPS.get(ends)
    if shared is not None:
        path, _role = shared
        return path
    return _hop(source_name, target_name)
