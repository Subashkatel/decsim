"""Deterministic, atomic run-seed binding for a runtime component graph."""

import hashlib
import random
import threading

from .message import RunSeedPathSegment, RunSeedReservation
from .protocols import RunSeedComposite, RunSeedConsumer

_NAMESPACE = b"decsim.run-seed.v1"


class _AtomicRunSeedConsumer:
    """Shared two-phase seed binding for stochastic runtime leaves."""

    _explicit_seed_label = "seed"

    def _initialize_run_seed_binding(self, explicit_seed) -> None:
        self._explicit_seed = explicit_seed
        self._run_seed_lock = threading.Lock()
        self._pending_run_seed = None
        self._run_seed_claimed = False
        self._stochastic_use_started = False

    def _explicit_run_seed(self):
        return self._explicit_seed

    def _entropy_seed(self):
        return None

    def _prepare_run_seed_state(self, effective_seed):
        return effective_seed

    def _install_run_seed_state(self, prepared_state) -> None:
        raise NotImplementedError

    def reserve_run_seed(self, seed) -> RunSeedReservation:
        component_name = type(self).__name__
        with self._run_seed_lock:
            if self._stochastic_use_started:
                raise ValueError(
                    f"{component_name} was already used and cannot be rebound"
                )
            if self._run_seed_claimed:
                raise ValueError(
                    f"{component_name} is already claimed by a built run"
                )
            if self._pending_run_seed is not None:
                raise ValueError(
                    f"{component_name} already has a pending run-seed reservation"
                )
            if seed is not None and self._explicit_seed is not None:
                raise ValueError(
                    f"{component_name} has an explicit "
                    f"{self._explicit_seed_label} that conflicts with "
                    f"numeric run root {seed}"
                )
            if seed is not None:
                source = "derived"
                effective_seed = seed
            elif self._explicit_seed is not None:
                source = "explicit_local"
                effective_seed = self._explicit_run_seed()
            else:
                source = "entropy"
                effective_seed = self._entropy_seed()
            reservation = RunSeedReservation(
                proposed_seed_source=source,
                proposed_seed=None if source == "entropy" else effective_seed,
                prepared_state=self._prepare_run_seed_state(effective_seed),
            )
            self._pending_run_seed = reservation
            return reservation

    def cancel_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._run_seed_lock:
            if self._pending_run_seed is reservation:
                self._pending_run_seed = None

    def commit_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._run_seed_lock:
            self._install_run_seed_state(reservation.prepared_state)
            self._pending_run_seed = None
            self._run_seed_claimed = True

    def _mark_stochastic_use(self) -> None:
        with self._run_seed_lock:
            self._stochastic_use_started = True


class _RandomSeedConsumer(_AtomicRunSeedConsumer):
    """Atomic run-seed ownership for random.Random components."""

    def _initialize_run_seed_state(self, seed) -> None:
        self._initialize_run_seed_binding(seed)
        self._rng = random.Random(seed)

    def _prepare_run_seed_state(self, effective_seed):
        return random.Random(effective_seed)

    def _install_run_seed_state(self, prepared_state) -> None:
        self._rng = prepared_state


def derive_component_seed(root_seed: int, path) -> int:
    """Derive one unsigned 64-bit seed from a framed semantic path."""
    encoded_path = b"".join(segment.canonical_bytes() for segment in path)
    digest = hashlib.blake2b(
        _NAMESPACE + root_seed.to_bytes(8, "big") + encoded_path,
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def bind_run_seed(root_seed, roots) -> None:
    """Bind each stochastic leaf once, cancelling every claim on failure."""
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
                children.append((encoded(child.relative_path), child))
            for _, child in sorted(children, key=lambda item: item[0]):
                walk(path + child.relative_path, child.child)
        finally:
            active.remove(identity)

    ordered_roots = (
        (encoded(path), path, component) for path, component in roots
    )
    for _, path, component in sorted(ordered_roots, key=lambda item: item[0]):
        walk(path, component)

    acquired = []
    try:
        for path, component, seed in plan:
            reservation = component.reserve_run_seed(seed)
            acquired.append((component, reservation))
            if seed is None and reservation.proposed_seed_source not in (
                "explicit_local", "entropy"
            ):
                raise ValueError("unseeded components must report their seed source")
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
