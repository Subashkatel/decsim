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
    decode_queue=None,
    store_output=None,
    gap_join=None,
    store=None,
) -> routers.Router:
    """A one_pool router whose tracker knows every stream.

    Its one side holds the pool's manager, store, store end and join,
    the parts a test hands it; None for those the test never reaches.
    """
    rule = routers.OnePool.Settings(pool.word)
    one_entry_router = routers.Router(rule)
    operations = collections.defaultdict(lambda: None)
    one_entry_router.tracker = types.SimpleNamespace(operation_by_id=operations)
    side = routers.EntrySide(
        pool, decode_queue, store_output, gap_join, store=store
    )
    one_entry_router.add_side(side)
    return one_entry_router


def stores_router(store_by_pool: dict, entry_of) -> types.SimpleNamespace:
    """A router whose sides name each entry pool's store, for a retention.

    entry_of names each stream's entry pool, the router's answer; the
    sides carry only the store, the one thing a retention asks of them.
    """
    side_by_pool = {}
    for pool, store in store_by_pool.items():
        side = routers.EntrySide(pool, None, None, store=store)
        side_by_pool[pool] = side
    sides = tuple(side_by_pool.values())
    return types.SimpleNamespace(
        pool_of_stream=entry_of,
        side=side_by_pool.__getitem__,
        sides=lambda: sides,
    )
