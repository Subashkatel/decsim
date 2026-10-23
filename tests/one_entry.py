"""The router of a machine with one entry pool, for a part tested alone.

Every machine has a router (windows/routers.py), so a window side part
tested without a machine asks this one, which sends every stream to the
weak pool.
"""

import collections
import types

import decsim.records.windows as window_records
import decsim.windows.routers as routers


def router(
    pool: window_records.PoolId = window_records.WEAK_POOL_ID,
) -> routers.Router:
    """A one_pool router whose tracker knows every stream."""
    rule = routers.OnePool.Settings(pool.word)
    one_entry_router = routers.Router(rule)
    operations = collections.defaultdict(lambda: None)
    one_entry_router.tracker = types.SimpleNamespace(operation_by_id=operations)
    return one_entry_router
