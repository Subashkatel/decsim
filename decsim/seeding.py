"""Deterministic, atomic run-seed binding for a runtime component graph."""

import hashlib
from .message import RunSeedChild, RunSeedPathSegment, RunSeedReservation
from .protocols import RunSeedComposite, RunSeedConsumer
_NAMESPACE = b"decsim.run-seed.v1"

def derive_component_seed(root_seed: int, path) -> int:
    encoded_path = b"".join(segment.canonical_bytes() for segment in path)
    digest = hashlib.blake2b(
        _NAMESPACE + root_seed.to_bytes(8, "big") + encoded_path,
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")
def bind_run_seed(root_seed, roots) -> None:
    """Bind each stochastic leaf once, cancelling every claim on failure."""
    if root_seed is not None and (
        type(root_seed) is not int or not 0 <= root_seed < 2**64
    ):
        raise ValueError("root seed must be an unsigned 64-bit integer or None")
    paths_by_identity = {}
    active = set()
    paths = set()
    plan = []
    def encoded(path):
        return b"".join(segment.canonical_bytes() for segment in path)

    def walk(path, component):
        path_bytes = encoded(path)
        if path_bytes in paths:
            raise ValueError(f"duplicate seed path {_render(path)}")
        paths.add(path_bytes)
        identity = id(component)
        if identity in active:
            raise ValueError(
                f"seed cycle from {_render(path)} to "
                f"{_render(paths_by_identity[identity])}"
            )
        if identity in paths_by_identity:
            return
        paths_by_identity[identity] = path
        active.add(identity)
        try:
            if isinstance(component, RunSeedConsumer):
                seed = (
                    None
                    if root_seed is None
                    else derive_component_seed(root_seed, path)
                )
                plan.append((path, component, seed))
            if not isinstance(component, RunSeedComposite):
                return
            children = []
            for child in component.run_seed_children():
                if type(child) is not RunSeedChild:
                    raise TypeError("run_seed_children() must yield RunSeedChild")
                children.append((encoded(child.relative_path), child))
            for _, child in sorted(children, key=lambda item: item[0]):
                walk(path + child.relative_path, child.child)
        finally:
            active.remove(identity)

    for _, path, component in sorted(
        (encoded(path), path, component) for path, component in roots
    ):
        walk(path, component)

    acquired = []
    try:
        for path, component, seed in plan:
            reservation = component.reserve_run_seed(seed)
            if type(reservation) is not RunSeedReservation:
                raise TypeError("reserve_run_seed() must return RunSeedReservation")
            if seed is not None and (
                reservation.proposed_seed_source != "derived"
                or reservation.proposed_seed != seed
            ):
                raise ValueError(
                    f"{type(component).__name__} disagrees with the run seed at "
                    f"{_render(path)}"
                )
            if seed is None and reservation.proposed_seed_source not in (
                "explicit_local", "entropy"
            ):
                raise ValueError("unseeded components must report their seed source")
            acquired.append((component, reservation))
    except BaseException:
        for component, reservation in reversed(acquired):
            component.cancel_run_seed(reservation)
        raise
    for component, reservation in acquired:
        component.commit_run_seed(reservation)
def _render(path: tuple[RunSeedPathSegment, ...]) -> str:
    return ".".join(
        segment.value
        if segment.kind == "field"
        else f"[{segment.value!r}]"
        for segment in path
    )
