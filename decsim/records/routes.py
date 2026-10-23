"""The routes of a run: the hops each answer and each boundary takes.

A route is generated from the pools and the escalate edges between
them (build/routes.py), never written by hand, so a machine of any
number of pools has one for every answer and every pair of pools a
boundary joins.
"""

import dataclasses
from typing import Optional

import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

# The two sites a part sits at: the chip in the fridge, the host in the
# room. A message between them crosses the wall.
FRIDGE = "fridge"
HOST = "host"

# The ways an edge's target answer reaches the frame, by
# SwitchingSettings.strong_answer_route. direct: the target joins the
# source's crossing commit inside its own decode, as CUDA-Q's host worker
# adds the predecoder's logical prediction to its own (cudaqx
# test_realtime_predecoder_w_pymatching.cpp line 535) with 0.1 us of
# worker overhead around its whole decode
# (hybrid_ai_predecoder_pipeline.md lines 706-712), so the join costs
# nothing, and the target sends the answer to the frame (Toshio et al.
# 2510.25222 Fig. 1). through_weak_chip: the answer goes down to the
# source's chip, which joins it in CHIP_COMMIT_CYCLES and sends it home
# on its own hop.
DIRECT = "direct"
THROUGH_WEAK_CHIP = "through_weak_chip"
ANSWER_ROUTE_NAMES = (DIRECT, THROUGH_WEAK_CHIP)
# the chip's join: the Pauli frame update of Yang et al. 2605.04892
# Table I, one cycle (4 ns at 250 MHz)
CHIP_COMMIT_CYCLES = 1


# What a hop carries. The reference card prices a hop by its role
# (links/link_profiles.py reference_path), a fabric may pick its link by
# it (topology/fabrics.py RoleLinks), and the data-movement report
# classes the memory it crosses by it (observe/data_movement.py).
READOUT = "readout"  # the QPU's measurements to the controller
COMMAND = "command"  # the controller's instruction to the QPU
DECISION = "decision"  # the frame's conditional release to the controller
WRITE = "write"  # the controller's round into a store
READ = "read"  # a store's rounds into the pool that reads it
ANSWER = "answer"  # a pool's correction to the frame
SAME_POOL = "same_pool"  # a boundary between two decodes of one pool
FORWARD = "forward"  # an escalation, or a boundary to a later pool
REVERSE = "reverse"  # an answer down an edge, or a boundary back
HOP_KINDS = (
    READOUT,
    COMMAND,
    DECISION,
    WRITE,
    READ,
    ANSWER,
    SAME_POOL,
    FORWARD,
    REVERSE,
)


@dataclasses.dataclass(frozen=True)
class HopRole:
    """What a hop carries, and the variant its price splits on.

    kind is one of HOP_KINDS. variant is how a store takes its rounds,
    written or landed, for a write or a read; the site of the pool, the
    fridge or the host, for an answer or a same-pool hop; None for the
    rest.
    """

    kind: str
    variant: Optional[str] = None

    @property
    def key(self) -> str:
        """The role as one word: its kind, then its variant if it has one."""
        if self.variant is None:
            return self.kind
        return f"{self.kind}_{self.variant}"


def hop_between(sender: str, receiver: str) -> transfer_records.LinkPath:
    """The hop of a graph's own from one part to another, by their names."""
    return f"{sender}_to_{receiver}"


@dataclasses.dataclass(frozen=True)
class AnswerRoute:
    """The hops an answer takes to the frame, and where it is joined.

    down is the hop to the source chip's commit step; None when the
    sender joins the answer itself. home is the hop to the frame.
    commit_cycles is the chip's join, on the switching clock.
    """

    down: Optional[transfer_records.LinkPath]
    home: transfer_records.LinkPath
    commit_cycles: int


@dataclasses.dataclass(frozen=True)
class BoundaryRoute:
    """The hop a boundary takes, and the bits of the name it carries."""

    path: transfer_records.LinkPath
    name_bits: int


@dataclasses.dataclass(frozen=True)
class PoolHops:
    """One pool's own hops, and the site it sits at.

    answer is its hop to the frame. same_pool carries a boundary
    between two of its own decodes.
    """

    pool: window_records.PoolId
    site: str
    answer: transfer_records.LinkPath
    same_pool: transfer_records.LinkPath


@dataclasses.dataclass(frozen=True)
class EdgeHops:
    """One escalate edge's hops, and the route its target answers on.

    forward carries the escalation and a source boundary the target
    reads; reverse carries a target answer down and a target boundary
    the source reads.
    """

    source: window_records.PoolId
    target: window_records.PoolId
    forward: transfer_records.LinkPath
    reverse: transfer_records.LinkPath
    answer_route: str = DIRECT


@dataclasses.dataclass(frozen=True)
class Routes:
    """Every answer's route by who asked, every boundary's by pair of pools.

    answer_by_asker keys a route by (pool, asked_by): asked_by is the
    source of the edge that asked the pool, None for a window the pool
    decodes first. name_bits_by_answer_hop is each hop an answer rides,
    with the bits of the name it carries beside its flips.
    """

    answer_by_asker: dict
    boundary_by_pair: dict
    name_bits_by_answer_hop: dict

    def answer(
        self,
        pool: window_records.PoolId,
        asked_by: Optional[window_records.PoolId] = None,
    ) -> AnswerRoute:
        """The route a pool's answer takes to the frame.

        A pool several edges target answers each down its own edge, as
        each of Toshio's weak decoders joins the one strong decoder's
        answer for its own window (2510.25222 Sec. III C).
        """
        return self.answer_by_asker[pool, asked_by]

    def boundary(
        self,
        source: window_records.PoolId,
        destination: window_records.PoolId,
    ) -> BoundaryRoute:
        """The route a boundary takes from one pool's decode to another's."""
        return self.boundary_by_pair[source, destination]

    def answer_name_bits(self, path: transfer_records.LinkPath) -> int:
        """The name an answer carries on one of its hops."""
        return self.name_bits_by_answer_hop[path]
