"""Deterministic seeds for every stochastic component of one run.

A component's seed is derived from the run's root seed and the component's
path in the machine (blake2b over a namespace, the root and the framed
path), so a component draws the same numbers whatever else the run holds.
Binding is a two-phase transaction: every consumer reserves before any
commits, and a failed reservation cancels the ones before it.
"""

import hashlib
import random
import threading
from collections.abc import Iterable
from typing import Any, Optional, Protocol, runtime_checkable

import decsim.records.seeds as seed_records

_NAMESPACE = b"decsim.run-seed.v1"
_UNSEEDED_SOURCES = ("explicit_local", "entropy")


def derive_component_seed(root_seed: int, path) -> int:
    """One unsigned 64-bit seed from the root seed and a framed path."""
    encoded_path = _encode_path(path)
    root_bytes = root_seed.to_bytes(8, "big")
    preimage = _NAMESPACE + root_bytes + encoded_path
    digest = hashlib.blake2b(preimage, digest_size=8)
    digest_bytes = digest.digest()
    return int.from_bytes(digest_bytes, "big")


def bind_run_seed(root_seed: Optional[int], roots) -> None:
    """Bind each stochastic leaf once, cancelling every claim on failure.

    Roots are (path, component) pairs; a composite's children are walked
    under its path. Every leaf is reserved in walk order before any is
    committed, so a reservation that fails leaves no leaf half bound.
    """
    walk = _SeedWalk(root_seed)
    for path, component in _sorted_by_path(roots):
        walk.visit(path, component)
    acquired = []
    try:
        for _path, component, seed in walk.leaves:
            reservation = component.reserve_run_seed(seed)
            acquired.append((component, reservation))
            _check_seed_source(seed, reservation)
    except BaseException:
        for component, reservation in reversed(acquired):
            component.cancel_run_seed(reservation)
        raise
    for component, reservation in acquired:
        component.commit_run_seed(reservation)


@runtime_checkable
class RunSeedConsumer(Protocol):
    """A stochastic component bound in two phases: reserve, then commit.

    Every consumer is reserved before any is committed, so preparation
    that can fail belongs in reserve_run_seed; commit_run_seed installs
    the prepared state and never fails.
    """

    def reserve_run_seed(
        self, seed: Optional[int]
    ) -> seed_records.RunSeedReservation:
        """Prepare the state a commit installs, leaving the active one."""

    def commit_run_seed(
        self, reservation: seed_records.RunSeedReservation
    ) -> None:
        """Install the prepared state and close the binding."""

    def cancel_run_seed(
        self, reservation: seed_records.RunSeedReservation
    ) -> None:
        """Drop the pending reservation without touching the active state."""


@runtime_checkable
class RunSeedComposite(Protocol):
    """A component whose stochastic children are bound under its path."""

    def run_seed_children(self) -> Iterable[seed_records.RunSeedChild]:
        """The children, each with its path relative to this component."""


class _AtomicRunSeedConsumer:
    """Shared two-phase seed binding for stochastic runtime leaves.

    A subclass installs its random state in _install_run_seed_state and,
    when it has one, names its explicit seed in _explicit_seed_label.
    """

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

    def reserve_run_seed(self, seed) -> seed_records.RunSeedReservation:
        """Prepare the state a commit installs, leaving the active one."""
        with self._run_seed_lock:
            self._refuse_second_binding(seed)
            source, effective_seed = self._seed_source(seed)
            proposed_seed = effective_seed
            if source == "entropy":
                proposed_seed = None
            prepared_state = self._prepare_run_seed_state(effective_seed)
            reservation = seed_records.RunSeedReservation(
                proposed_seed_source=source,
                proposed_seed=proposed_seed,
                prepared_state=prepared_state,
            )
            self._pending_run_seed = reservation
            return reservation

    def cancel_run_seed(
        self, reservation: seed_records.RunSeedReservation
    ) -> None:
        """Drop the pending reservation; a stale one is ignored."""
        with self._run_seed_lock:
            if self._pending_run_seed is reservation:
                self._pending_run_seed = None

    def commit_run_seed(
        self, reservation: seed_records.RunSeedReservation
    ) -> None:
        """Install the prepared state and close the binding."""
        with self._run_seed_lock:
            self._install_run_seed_state(reservation.prepared_state)
            self._pending_run_seed = None
            self._run_seed_claimed = True

    def _mark_stochastic_use(self) -> None:
        with self._run_seed_lock:
            self._stochastic_use_started = True

    def _refuse_second_binding(self, seed) -> None:
        component_type = type(self)
        component_name = component_type.__name__
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

    def _seed_source(self, seed) -> tuple:
        """The seed source and value: the run, the component, or entropy."""
        if seed is not None:
            return "derived", seed
        if self._explicit_seed is not None:
            explicit_seed = self._explicit_run_seed()
            return "explicit_local", explicit_seed
        entropy_seed = self._entropy_seed()
        return "entropy", entropy_seed


class _RandomSeedConsumer(_AtomicRunSeedConsumer):
    """Atomic run-seed ownership for a component drawing from random.Random."""

    def _initialize_run_seed_state(self, seed) -> None:
        self._initialize_run_seed_binding(seed)
        self._rng = random.Random(seed)

    def _prepare_run_seed_state(self, effective_seed):
        return random.Random(effective_seed)

    def _install_run_seed_state(self, prepared_state) -> None:
        self._rng = prepared_state


class _SeedWalk:
    """The leaves of a component graph in binding order, with their seeds."""

    def __init__(self, root_seed: Optional[int]):
        self._root_seed = root_seed
        self._path_by_identity: dict[int, tuple] = {}
        self._active_identities: set[int] = set()
        self._encoded_paths: set[bytes] = set()
        # (path, component, seed) in preorder; a shared component is
        # planned once, at its first sorted path.
        self.leaves: list[tuple] = []

    def visit(self, path, component) -> None:
        """Plan the component and, when it is a composite, its children."""
        self._note_path(path)
        identity = id(component)
        if identity in self._active_identities:
            cycle_start = self._path_by_identity[identity]
            path_text = _render(path)
            cycle_text = _render(cycle_start)
            raise ValueError(f"seed cycle from {path_text} to {cycle_text}")
        if identity in self._path_by_identity:
            return
        self._path_by_identity[identity] = path
        self._active_identities.add(identity)
        try:
            self._plan_leaf(path, component)
            self._visit_children(path, component)
        finally:
            self._active_identities.remove(identity)

    def _note_path(self, path) -> None:
        encoded = _encode_path(path)
        if encoded in self._encoded_paths:
            path_text = _render(path)
            raise ValueError(f"duplicate seed path {path_text}")
        self._encoded_paths.add(encoded)

    def _plan_leaf(self, path, component) -> None:
        if not isinstance(component, RunSeedConsumer):
            return
        seed = None
        if self._root_seed is not None:
            seed = derive_component_seed(self._root_seed, path)
        self.leaves.append((path, component, seed))

    def _visit_children(self, path, component) -> None:
        if not isinstance(component, RunSeedComposite):
            return
        children = component.run_seed_children()
        pairs = []
        for child in children:
            pairs.append((child.relative_path, child))
        for relative_path, child in _sorted_by_path(pairs):
            child_path = path + relative_path
            self.visit(child_path, child.child)


def _sorted_by_path(pairs) -> list:
    """(path, item) pairs in the order of their encoded paths."""
    keyed = []
    for path, item in pairs:
        encoded = _encode_path(path)
        keyed.append((encoded, path, item))
    keyed.sort(key=lambda entry: entry[0])
    ordered = []
    for _, path, item in keyed:
        ordered.append((path, item))
    return ordered


def _encode_path(path) -> bytes:
    pieces = []
    for segment in path:
        segment_bytes = segment.canonical_bytes()
        pieces.append(segment_bytes)
    return b"".join(pieces)


def _check_seed_source(
    seed: Optional[int], reservation: seed_records.RunSeedReservation
) -> None:
    """A leaf bound without a run seed must say where its seed came from."""
    if seed is not None:
        return
    if reservation.proposed_seed_source not in _UNSEEDED_SOURCES:
        raise ValueError("unseeded components must report their seed source")


def _render(path: tuple[Any, ...]) -> str:
    """The path as text for a refusal: fields dotted, keys in brackets."""
    words = []
    for segment in path:
        if segment.kind == "field":
            words.append(segment.value)
        else:
            words.append(f"[{segment.value!r}]")
    return ".".join(words)
