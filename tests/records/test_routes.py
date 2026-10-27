"""The route records of decsim/records/routes.py.

A run's routes are looked up by pool, by pair of pools and by hop.
"""

import pytest

import decsim.records.routes as route_records
import decsim.records.windows as window_records

WEAK = window_records.WEAK_POOL_ID
STRONG = window_records.STRONG_POOL_ID


def _routes() -> route_records.Routes:
    answer = route_records.AnswerRoute(None, "weak_to_frame", 0)
    boundary = route_records.BoundaryRoute("weak_to_weak", 0)
    return route_records.Routes(
        answer_by_asker={(WEAK, None): answer},
        boundary_by_pair={(WEAK, WEAK): boundary},
        name_bits_by_answer_hop={"weak_to_frame": 0},
    )


def test_each_lookup_reads_its_own_table():
    routes = _routes()
    answer = routes.answer(WEAK)
    boundary = routes.boundary(WEAK, WEAK)
    assert answer.home == "weak_to_frame"
    assert boundary.path == "weak_to_weak"
    assert routes.answer_name_bits("weak_to_frame") == 0


def test_a_pair_no_edge_joins_has_no_boundary_route():
    routes = _routes()
    with pytest.raises(KeyError):
        routes.boundary(WEAK, STRONG)
