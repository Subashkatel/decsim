"""The router: which entry pool decodes each window.

A machine of several entry pools sends every window to one of them. VQD
(Maurya and Tannu 2406.17995v2) gives each logical qubit its own decoder
as its baseline (Sec. 4.1, Fig. 5), as parallel-window decoders assume
(its Sec. 8), and routes each qubit's tagged syndromes to its decoder by
a schedule fixed at compile time (Sec. 5, the Arbiter). A routing rule
here is such a fixed schedule: it reads only the window's operation, so
a window's pool is known when the window is planned, and a boundary
that leaves for it rides to that pool.

    by_patch  each patch's windows to the pool its table names
    one_pool  every window to the one entry pool, the rule of a machine
              with no router part

Every machine has a router, so every part asks one owner which entry
pool decodes a window, and the line of edges a window may climb starts
there (windows/escalate_edges.py).

A rule that reads the run's state (the least loaded pool, a backlog, a
round) has no paper behind it and is not built; the rule slot takes a
researcher's own row. Choosing which waiting window a shared pool
serves next (VQD's MFD, RR and MLS) is the pool's scheduler, not a
router.
"""

import dataclasses
from typing import Any, Optional, Protocol

import decsim.ports as ports
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.round_tracker as round_tracker_module


class RuleSettings(Protocol):
    """A routing rule's settings record (this module)."""

    name: str

    def pool_for(self, operation: program_records.Operation) -> str:
        """The name of the entry pool that decodes the operation's windows."""

    def check_operations(self, operations: tuple, pool_names: tuple) -> None:
        """Refuse an operation the rule sends to no entry pool."""


class ByPatch:
    """Each patch's windows go to the pool its table names.

    VQD's decoder per logical qubit (2406.17995v2 Sec. 4.1, Fig. 5),
    or per group of patches when one pool is named for several. A window
    over several patches, a merge, goes to the pool of its operation's
    first patch: VQD decodes per qubit and says nothing of a decode that
    spans two, so this is decsim's reading.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The table: (patch, pool name) pairs, one pair per patch."""

        pool_by_patch: tuple = ()
        name = "by_patch"

        def __post_init__(self) -> None:
            # a pair written in YAML, a constants file's, is a list
            pairs = []
            for patch, pool_name in self.pool_by_patch:
                pairs.append((patch, pool_name))
            object.__setattr__(self, "pool_by_patch", tuple(pairs))
            patches = [pair[0] for pair in pairs]
            if len(set(patches)) != len(patches):
                raise ValueError(
                    "router.rule.pool_by_patch names a patch twice: "
                    f"{list(self.pool_by_patch)}"
                )

        def pool_for(self, operation: program_records.Operation) -> str:
            """The pool of the operation's first patch."""
            table = dict(self.pool_by_patch)
            first_patch = operation.patches[0]
            return table[first_patch]

        def check_operations(
            self, operations: tuple, pool_names: tuple
        ) -> None:
            """Every operation's first patch has an entry pool."""
            table = dict(self.pool_by_patch)
            for pool_name in table.values():
                _refuse_a_pool_not_routed_to(pool_name, pool_names)
            for operation in operations:
                _refuse_an_unrouted_operation(operation, table)


class OnePool:
    """Every window to the one entry pool: a machine with no router part."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The one entry pool, by its part's name."""

        pool: str
        name = "one_pool"

        def pool_for(self, operation: program_records.Operation) -> str:
            """The one pool."""
            del operation
            return self.pool

        def check_operations(
            self, operations: tuple, pool_names: tuple
        ) -> None:
            """Nothing to refuse: the build names the entry pool itself."""
            del operations, pool_names


@dataclasses.dataclass(frozen=True)
class EntrySide:
    """What one entry pool's windows are asked through.

    decode_queue is the pool's manager; store the store it reads, where
    its windows' reads are held, and store_output that store's end that
    sends to it; is_store_written whether the controller writes the
    store, so its arrivals are the rounds published to the windows;
    gap_join the join of its solves, None when it computes no
    confidence. A run with no decoder reads no store.
    """

    pool: window_records.PoolId
    decode_queue: Any
    store_output: Any
    gap_join: Optional[Any] = None
    store: Optional[Any] = None
    is_store_written: bool = True


class Router:
    """Sends each window to the entry pool its rule names.

    It holds every entry pool's side, the first's as any other's, so no
    part keeps a first entry pool of its own: a part asks the router.
    """

    tracker = ports.Port(round_tracker_module.RoundTracker)

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The router's rule (this module's rows), and the factory's pool.

        factory_pool names the pool the magic-state factory's correction
        decodes go to; None names the one entry pool of a machine with
        one. No paper sets decoders aside for a factory: DecoNet decodes
        15-to-1 distillation on its per-qubit decoders (2504.11805 Sec.
        VIII-F) and VQD decodes the factory's storage patches in its
        shared pool (2406.17995 Sec. 5.3), so a run names which pool
        serves them (build/layout.py factory_pool).
        """

        rule: RuleSettings = ByPatch.Settings()
        factory_pool: Optional[str] = None
        name = "router"

        def build(self) -> "Router":
            """A fresh router over its rule."""
            return Router(self.rule)

    def __init__(self, rule: RuleSettings) -> None:
        self.rule = rule
        self.side_by_pool: dict = {}

    def add_side(self, side: EntrySide) -> None:
        """One entry pool, reached through this router."""
        self.side_by_pool[side.pool] = side

    def pool_for(self, window: window_records.Window) -> window_records.PoolId:
        """The entry pool that decodes this window."""
        return self.pool_of_stream(window.stream_id)

    def pool_of_stream(
        self,
        stream_id: Any,  # an opaque identity
    ) -> window_records.PoolId:
        """The entry pool that decodes every window of the stream."""
        operation = self.tracker.operation_by_id[stream_id]
        name = self.rule.pool_for(operation)
        return window_records.PoolId(name)

    def is_entry(self, pool: window_records.PoolId) -> bool:
        """Whether the pool decodes windows the router sends it."""
        return pool in self.side_by_pool

    def queue_for(
        self, operation: program_records.Operation
    ) -> ports.DecodeQueue:
        """The manager of the entry pool that decodes the operation."""
        name = self.rule.pool_for(operation)
        side = self.side(window_records.PoolId(name))
        return side.decode_queue

    def side(self, pool: window_records.PoolId) -> EntrySide:
        """The side of an entry pool.

        A part asks only for the pool a window was routed to, so a pool
        with no side is a broken contract.
        """
        found = self.side_by_pool.get(pool)
        if found is None:
            raise RuntimeError(
                f"{pool.word} is no entry pool of this router; its entry "
                f"pools are {[entry.word for entry in self.side_by_pool]}"
            )
        return found

    def sides(self) -> tuple:
        """Every entry pool's side, in build order."""
        sides = self.side_by_pool.values()
        return tuple(sides)


def _refuse_a_pool_not_routed_to(pool_name: str, pool_names: tuple) -> None:
    if pool_name in pool_names:
        return
    raise ValueError(
        f"router.rule.pool_by_patch names the pool {pool_name!r}, which "
        f"the router's windows port does not reach; it reaches "
        f"{list(pool_names)}"
    )


def _refuse_an_unrouted_operation(
    operation: program_records.Operation, table: dict
) -> None:
    patches = operation.patches
    if patches and patches[0] in table:
        return
    raise ValueError(
        f"router.rule.pool_by_patch routes no pool for operation "
        f"{operation.name!r} on patches {patches}: it routes the patches "
        f"{sorted(table)}"
    )
