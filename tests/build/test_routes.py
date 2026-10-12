"""The routes build/routes.py generates from pools and escalate edges.

A weak pool on the chip escalating to a strong pool on the host answers
and trades boundaries on the hops named by the parts at their ends. A
hop across the wall between two sites carries the request's name; a hop
on one site carries none.
"""

import decsim.build.routes as routes_build
import decsim.records.routes as route_records
import decsim.records.windows as window_records

NAME_BITS = window_records.REQUEST_KEY_WIRE_BITS
WEAK = window_records.WEAK_POOL_ID
STRONG = window_records.STRONG_POOL_ID
WEAK_HOPS = route_records.PoolHops(
    WEAK, "fridge", "weak_to_frame", "weak_to_weak"
)
STRONG_HOPS = route_records.PoolHops(
    STRONG, "host", "strong_to_frame", "strong_to_strong"
)


def _with_edge(answer_route: str) -> route_records.Routes:
    edge = route_records.EdgeHops(
        WEAK,
        STRONG,
        "weak_to_strong",
        "strong_to_weak",
        answer_route=answer_route,
    )
    pools = (WEAK_HOPS, STRONG_HOPS)
    return routes_build.generated(pools, (edge,), WEAK)


def test_a_run_with_no_edge_answers_each_pool_on_its_own_hop():
    """The weak answer stays on the chip; the strong one names its request."""
    pools = (WEAK_HOPS, STRONG_HOPS)
    routes = routes_build.generated(pools, (), WEAK)
    assert routes.answer(WEAK) == route_records.AnswerRoute(
        None, "weak_to_frame", 0
    )
    assert routes.answer(STRONG) == route_records.AnswerRoute(
        None, "strong_to_frame", 0
    )
    assert routes.name_bits_by_answer_hop == {
        "weak_to_frame": 0,
        "strong_to_frame": NAME_BITS,
    }


def test_an_edges_boundaries_are_the_four_routes_of_its_two_pools():
    """Within a pool no name; across the wall, either way, the name."""
    routes = _with_edge(route_records.DIRECT)
    assert routes.boundary_by_pair == {
        (WEAK, WEAK): route_records.BoundaryRoute("weak_to_weak", 0),
        (STRONG, STRONG): route_records.BoundaryRoute("strong_to_strong", 0),
        (WEAK, STRONG): route_records.BoundaryRoute(
            "weak_to_strong", NAME_BITS
        ),
        (STRONG, WEAK): route_records.BoundaryRoute(
            "strong_to_weak", NAME_BITS
        ),
    }


def test_an_answer_through_the_weak_chip_goes_down_then_home_on_the_chip():
    """Down the reverse hop with its name, home on the weak hop without."""
    routes = _with_edge(route_records.THROUGH_WEAK_CHIP)
    assert routes.answer(STRONG) == route_records.AnswerRoute(
        down="strong_to_weak",
        home="weak_to_frame",
        commit_cycles=route_records.CHIP_COMMIT_CYCLES,
    )
    assert routes.name_bits_by_answer_hop == {
        "weak_to_frame": 0,
        "strong_to_weak": NAME_BITS,
    }


def test_a_chain_names_only_the_hops_that_cross_the_wall():
    """Chip, host, host: the two host pools meet without a name.

    The third pool answers through the second, which sits on the host,
    so its answer goes down to the second pool unnamed and home named.
    """
    chip = window_records.PoolId("chip")
    relay = window_records.PoolId("relay")
    osd = window_records.PoolId("osd")
    pools = (
        route_records.PoolHops(chip, "fridge", "chip_to_frame", "chip_to_chip"),
        route_records.PoolHops(
            relay, "host", "relay_to_frame", "relay_to_relay"
        ),
        route_records.PoolHops(osd, "host", "osd_to_frame", "osd_to_osd"),
    )
    edges = (
        route_records.EdgeHops(chip, relay, "chip_to_relay", "relay_to_chip"),
        route_records.EdgeHops(
            relay,
            osd,
            "relay_to_osd",
            "osd_to_relay",
            answer_route=route_records.THROUGH_WEAK_CHIP,
        ),
    )
    routes = routes_build.generated(pools, edges, chip)
    assert routes.answer(osd) == route_records.AnswerRoute(
        "osd_to_relay", "relay_to_frame", 1
    )
    assert routes.boundary(relay, osd) == route_records.BoundaryRoute(
        "relay_to_osd", 0
    )
    assert routes.boundary(relay, chip) == route_records.BoundaryRoute(
        "relay_to_chip", NAME_BITS
    )
    assert routes.name_bits_by_answer_hop == {
        "chip_to_frame": 0,
        "relay_to_frame": NAME_BITS,
        "osd_to_relay": 0,
    }
