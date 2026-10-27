"""Generate a run's answer and boundary routes from its pools and edges.

An answer goes home on its pool's own hop, or, for an edge whose
answer route is through_weak_chip, down the edge's reverse hop to the
source chip, which joins it and sends it home on the source's hop. A
pool several edges target answers on each edge's route in turn. A
boundary between two decodes of one pool rides the pool's same-pool
hop; across an edge it rides the edge's forward or reverse hop. Every
message that crosses the wall between two sites names the request it
belongs to, as every message across the wall does while other requests
are open (records/windows.py REQUEST_KEY_WIRE_BITS); one that stays on
its site carries no name.
"""

import decsim.records.routes as route_records
import decsim.records.windows as window_records


def generated(
    pools: tuple, edges: tuple, frame_site: str = route_records.FRIDGE
) -> route_records.Routes:
    """Every answer route and boundary route of pools joined by edges.

    Each pool answers its own windows home on its own hop, and the
    windows an edge asks of it on that edge's route. frame_site is the
    frame's, which an answer from any other site crosses the wall to.
    """
    pool_by_id = {}
    for pool in pools:
        pool_by_id[pool.pool] = pool
    answer_by_asker = {}
    name_bits_by_hop = {}
    for pool in pools:
        name_bits_by_hop[pool.answer] = _name_bits(pool.site, frame_site)
        own_route = route_records.AnswerRoute(None, pool.answer, 0)
        answer_by_asker[pool.pool, None] = own_route
    for edge in edges:
        route = _edge_answer_route(edge, pool_by_id, name_bits_by_hop)
        answer_by_asker[edge.target, edge.source] = route
    boundary_by_pair = _boundary_routes(pools, pool_by_id, edges)
    return route_records.Routes(
        answer_by_asker=answer_by_asker,
        boundary_by_pair=boundary_by_pair,
        name_bits_by_answer_hop=name_bits_by_hop,
    )


def _edge_answer_route(
    edge: route_records.EdgeHops,
    pool_by_id: dict,
    name_bits_by_hop: dict,
) -> route_records.AnswerRoute:
    """The route of the target's answer to the edge; records its down hop."""
    target = pool_by_id[edge.target]
    if edge.answer_route == route_records.DIRECT:
        return route_records.AnswerRoute(None, target.answer, 0)
    source = pool_by_id[edge.source]
    name_bits_by_hop[edge.reverse] = _name_bits(target.site, source.site)
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


def _name_bits(sender_site: str, receiver_site: str) -> int:
    """The request's name on a hop across the wall, none on one site."""
    if sender_site == receiver_site:
        return 0
    return window_records.REQUEST_KEY_WIRE_BITS
