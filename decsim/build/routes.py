"""Generate a run's answer and boundary routes from its pools and edges.

An answer goes home on its pool's own hop, or, for an edge whose
answer route is through_weak_chip, down the edge's reverse hop to the
source chip, which joins it and sends it home on the source's hop. A
boundary between two decodes of one pool rides the pool's same-pool
hop; across an edge it rides the edge's forward or reverse hop. Every
message that crosses the wall between two sites names the request it
belongs to, as every message across the wall does while other requests
are open (records/windows.py REQUEST_KEY_WIRE_BITS); one that stays on
its site carries no name.
"""

from typing import Optional

import decsim.records.routes as route_records
import decsim.records.windows as window_records

# the frame sits on the chip with the controller
FRAME_SITE = route_records.FRIDGE


def generated(
    pools: tuple,
    edges: tuple,
    window_pool: window_records.PoolId,
) -> route_records.Routes:
    """Every answer route and boundary route of pools joined by edges."""
    pool_by_id = {}
    for pool in pools:
        pool_by_id[pool.pool] = pool
    answer_by_pool = {}
    name_bits_by_hop = {}
    for pool in pools:
        route = _answer_route(pool, pool_by_id, edges, name_bits_by_hop)
        answer_by_pool[pool.pool] = route
    boundary_by_pair = _boundary_routes(pools, pool_by_id, edges)
    return route_records.Routes(
        window_pool=window_pool,
        answer_by_pool=answer_by_pool,
        boundary_by_pair=boundary_by_pair,
        name_bits_by_answer_hop=name_bits_by_hop,
    )


def _answer_route(
    pool: route_records.PoolHops,
    pool_by_id: dict,
    edges: tuple,
    name_bits_by_hop: dict,
) -> route_records.AnswerRoute:
    """The pool's answer route; records the name each of its hops carries."""
    edge = _incoming_edge(pool.pool, edges)
    if edge is None or edge.answer_route == route_records.DIRECT:
        name_bits_by_hop[pool.answer] = _name_bits(pool.site, FRAME_SITE)
        return route_records.AnswerRoute(None, pool.answer, 0)
    source = pool_by_id[edge.source]
    name_bits_by_hop[edge.reverse] = _name_bits(pool.site, source.site)
    name_bits_by_hop[source.answer] = _name_bits(source.site, FRAME_SITE)
    return route_records.AnswerRoute(
        down=edge.reverse,
        home=source.answer,
        commit_cycles=route_records.CHIP_COMMIT_CYCLES,
    )


def _boundary_routes(pools: tuple, pool_by_id: dict, edges: tuple) -> dict:
    """A boundary's route within each pool and both ways across each edge.

    Two units of a pool meet on their own site, a link per graph edge
    on a chip (Helios 2301.08419 lines 764-769). A boundary between two
    sites crosses on the hop the escalation or the answer takes: no
    paper prices this message (Toshio 2510.25222 lines 1965-1966 count
    the syndrome data and the decoding output alone), so the cable leg
    is decsim's choice.
    """
    routes = {}
    for pool in pools:
        routes[pool.pool, pool.pool] = route_records.BoundaryRoute(
            pool.same_pool, 0
        )
    for edge in edges:
        source = pool_by_id[edge.source]
        target = pool_by_id[edge.target]
        name_bits = _name_bits(source.site, target.site)
        routes[edge.source, edge.target] = route_records.BoundaryRoute(
            edge.forward, name_bits
        )
        routes[edge.target, edge.source] = route_records.BoundaryRoute(
            edge.reverse, name_bits
        )
    _join_the_unjoined(pools, routes)
    return routes


def _join_the_unjoined(pools: tuple, routes: dict) -> None:
    """A boundary between two pools no edge joins takes a hop of its own.

    In a chain the deep pool's boundary reaches the entry pool's windows
    directly, named by the pools' parts (route_records.hop_between).
    """
    for sender in pools:
        _join_from(sender, pools, routes)


def _join_from(
    sender: route_records.PoolHops, pools: tuple, routes: dict
) -> None:
    """The sender's own hop to each pool it has no route to yet."""
    for receiver in pools:
        pair = (sender.pool, receiver.pool)
        if pair in routes:
            continue
        path = route_records.hop_between(sender.pool.word, receiver.pool.word)
        name_bits = _name_bits(sender.site, receiver.site)
        routes[pair] = route_records.BoundaryRoute(path, name_bits)


def _incoming_edge(
    pool: window_records.PoolId, edges: tuple
) -> Optional[route_records.EdgeHops]:
    for edge in edges:
        if edge.target == pool:
            return edge
    return None


def _name_bits(sender_site: str, receiver_site: str) -> int:
    """The request's name on a hop across the wall, none on one site."""
    if sender_site == receiver_site:
        return 0
    return window_records.REQUEST_KEY_WIRE_BITS
