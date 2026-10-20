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


@dataclasses.dataclass(frozen=True)
class EntrySide:
    """What one entry pool's windows are asked through.

    decode_queue is the pool's manager, store_output the entry store's
    end that sends to it, and gap_join the join of its solves, None when
    no edge leaves it.
    """

    pool: window_records.PoolId
    decode_queue: Any
    store_output: Any
    gap_join: Optional[Any] = None


class Router:
    """Sends each window to the entry pool its rule names.

    sides holds every entry pool's side but the first, whose objects the
    window side holds itself (build/windows.py): a machine of one entry
    pool has no router.
    """

    tracker = ports.Port(round_tracker_module.RoundTracker)

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The router's rule (this module's rows)."""

        rule: RuleSettings = ByPatch.Settings()
        name = "router"

        def build(self) -> "Router":
            """A fresh router over its rule."""
            return Router(self.rule)

    def __init__(self, rule: RuleSettings) -> None:
        self.rule = rule
        self.side_by_pool: dict = {}

    def add_side(self, side: EntrySide) -> None:
        """One entry pool past the first, reached through this router."""
        self.side_by_pool[side.pool] = side

    def pool_for(self, window: window_records.Window) -> window_records.PoolId:
        """The entry pool that decodes this window."""
        operation = self.tracker.operation_by_id[window.stream_id]
        name = self.rule.pool_for(operation)
        return window_records.PoolId(name)

    def side_of(self, pool: window_records.PoolId) -> Optional[EntrySide]:
        """The side of an entry pool past the first; None for any other."""
        return self.side_by_pool.get(pool)

    def sides(self) -> tuple:
        """Every entry pool's side past the first, in build order."""
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
