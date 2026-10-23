"""A strong result teaches the policy of the edge that asked for it.

Several weak pools may share one strong pool, Delfosse's lazy decoders
over shared decoding units (2001.11427 Sec. V) and Toshio's N_window
weak decoders over one strong decoder (2510.25222 Sec. III C); an
online threshold audits its own escalations, so the strong pool's
result goes back to the edge on the window's own chain.
"""

import types

import decsim.records.windows as window_records
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.routers as routers
import tests.declared_run as declared_run

WEAK = window_records.PoolId("weak")
FPGA_B = window_records.PoolId("fpga_b")
STRONG = window_records.PoolId("strong")
DEEP = window_records.PoolId("deep")
# mem1 runs on patch 1, mem2 on patch 2
BY_PATCH = routers.ByPatch.Settings(pool_by_patch=((1, "weak"), (2, "fpga_b")))


class _Policy:
    """Remembers each strong result it is taught, under its window."""

    def __init__(self) -> None:
        self.learned = []

    def learn_from_strong_result(self, window_key, result) -> None:
        self.learned.append((window_key, result))


def _edge(source, target):
    """An edge's source, target and policy: all the learner reads."""
    policy = _Policy()
    return types.SimpleNamespace(source=source, target=target, policy=policy)


def _router():
    """The router sending mem1's windows to weak and mem2's to fpga_b."""
    router = routers.Router(BY_PATCH)
    mem1 = declared_run.memory_operation(1)
    mem2 = declared_run.memory_operation(2)
    operation_by_id = {1: mem1, 2: mem2}
    router.tracker = types.SimpleNamespace(operation_by_id=operation_by_id)
    return router


def test_a_shared_strong_result_teaches_the_edge_of_the_windows_entry_pool():
    weak_edge = _edge(WEAK, STRONG)
    fpga_b_edge = _edge(FPGA_B, STRONG)
    edges = escalate_edges.EscalateEdges((weak_edge, fpga_b_edge))
    router = _router()
    learner = escalate_edges.StrongResultLearner(edges, router, STRONG)
    # the policy keeps the result as given; any identity will do
    mem1_result = object()
    mem2_result = object()

    learner.learn_from_strong_result((1, 0), mem1_result)
    learner.learn_from_strong_result((2, 0), mem2_result)

    assert weak_edge.policy.learned == [((1, 0), mem1_result)]
    assert fpga_b_edge.policy.learned == [((2, 0), mem2_result)]


def test_a_deep_result_teaches_the_edge_into_deep_past_the_shared_target():
    """mem2 climbs fpga_b, strong, deep: the strong-to-deep edge learns."""
    weak_edge = _edge(WEAK, STRONG)
    fpga_b_edge = _edge(FPGA_B, STRONG)
    deep_edge = _edge(STRONG, DEEP)
    edges = escalate_edges.EscalateEdges((weak_edge, fpga_b_edge, deep_edge))
    router = _router()
    learner = escalate_edges.StrongResultLearner(edges, router, DEEP)
    # the policy keeps the result as given; any identity will do
    mem2_result = object()

    learner.learn_from_strong_result((2, 3), mem2_result)

    assert deep_edge.policy.learned == [((2, 3), mem2_result)]
    assert fpga_b_edge.policy.learned == []
