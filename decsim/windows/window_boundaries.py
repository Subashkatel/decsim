"""A boundary is the residual defects at a window's commit edge, produced by
the decoder at decode done and delivered to dependent windows over DD; a
held boundary waits for a final result. Versions make late deliveries
harmless: every send bumps the source's version and each delivery's version,
and a receiver only accepts the latest. The courier works on the window
manager's tables (windows, models, links, the interaction) and is built by
the manager.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Optional

from ..links.links import BoundaryTransferRelation, LinkPath
from ..message import (BoundaryDelivery, BoundaryUpdate, DecoderRequestKey, Operation,
                       Window, WindowInfo)


@dataclass(frozen=True)
class HeldBoundary:
    """A boundary kept back until the window's result is final (Held policy)."""

    source_request_key: DecoderRequestKey
    operation_id: object
    boundary: object


class BoundaryCourier:
    def __init__(self, window_manager):
        self.wm = window_manager
        self._committed_boundaries: dict[tuple, object] = {}
        self._boundary_versions: dict[tuple, int] = {}
        self._boundary_delivery_versions: dict[tuple, int] = {}
        self._released_boundary_dependencies: set[tuple] = set()
        self._held_boundary: dict[tuple, HeldBoundary] = {}

    # ---- what the manager and the recovery ask

    def hold(self, key: tuple, held: HeldBoundary) -> None:
        self._held_boundary[key] = held

    def take_held(self, key: tuple) -> Optional[HeldBoundary]:
        return self._held_boundary.pop(key, None)

    def has_committed(self, key: tuple) -> bool:
        return key in self._committed_boundaries

    def committed(self, key: tuple):
        return self._committed_boundaries.get(key)

    def set_committed(self, key: tuple, boundary) -> None:
        """A strong result corrected the boundary a window had already shipped."""
        self._committed_boundaries[key] = boundary

    def invalidate(self, window: Window) -> None:
        """A window is about to be decoded again: a boundary already in transit
        belongs to the invalidated decode, so advancing the versions makes its
        scheduled delivery a no-op; its own shipped and held boundaries are gone."""
        key = window.key
        self._boundary_versions[key] = self._boundary_versions.get(key, 0) + 1
        for dependency in window.deps:
            delivery_key = (dependency, key)
            self._boundary_delivery_versions[delivery_key] = \
                self._boundary_delivery_versions.get(delivery_key, 0) + 1
            self._released_boundary_dependencies.discard(delivery_key)
        self._committed_boundaries.pop(key, None)
        self._held_boundary.pop(key, None)

    def restore_dependency(self, source_key: tuple, destination: Window) -> None:
        """A replayed window takes an already-committed predecessor's boundary again."""
        self.merge_available(source_key, destination, self._committed_boundaries.get(source_key))
        self._released_boundary_dependencies.add((source_key, destination.key))

    def is_published(self, key: tuple) -> bool:
        """The window has held or shipped a boundary, or been versioned."""
        return (key in self._held_boundary or key in self._committed_boundaries
                or key in self._boundary_versions)

    def touches(self, keys) -> bool:
        """Any delivery or released dependency involves one of these windows."""
        return any(source in keys or destination in keys
                   for source, destination in self._boundary_delivery_versions) or any(
            source in keys or destination in keys
            for source, destination in self._released_boundary_dependencies)

    # ---- delivery

    def send(self, window: Window, op: Operation, boundary, *,
             source_request_key: DecoderRequestKey) -> None:
        """Ship a committed boundary over DD to the dependent windows the
        interaction selects; a new version supersedes any delivery in flight."""
        source_key = (window.op_id, window.k)
        selected_targets = tuple(self.wm.window_interaction.boundary_targets(
            WindowInfo.from_window(window), self.wm._window_infos()))
        if len(set(selected_targets)) != len(selected_targets):
            raise RuntimeError(
                f"window interaction selected duplicate boundary targets "
                f"for source {source_key}: {selected_targets}")
        targets = tuple(
            key for key in selected_targets
            if key not in self.wm.absorbed_windows
        )
        for dep_key in targets:
            if dep_key not in self.wm.windows:
                raise RuntimeError(
                    f"window interaction selected unknown boundary target "
                    f"{dep_key} for source {source_key}")
            target = self.wm.windows[dep_key]
            if source_key not in target.deps:
                raise RuntimeError(
                    f"window interaction selected boundary target {dep_key} "
                    f"for source {source_key}, but it is not a live "
                    f"dependency declared by the window scheme")
            if target.queued or target.committed:
                raise RuntimeError(
                    f"window interaction selected boundary target {dep_key} "
                    f"for source {source_key} after its decode lifecycle "
                    f"started")

        version = self._boundary_versions.get(source_key, 0) + 1
        deliveries = []
        for dep_key in targets:
            delivery_key = (source_key, dep_key)
            delivery_version = \
                self._boundary_delivery_versions.get(delivery_key, 0) + 1
            deliveries.append((dep_key, delivery_key, delivery_version))

        self._boundary_versions[source_key] = version
        self._committed_boundaries[source_key] = boundary
        for dep_key, delivery_key, delivery_version in deliveries:
            self._boundary_delivery_versions[delivery_key] = delivery_version
            reservation = self.wm.links.reserve(
                LinkPath.DD,
                payload_bits=None,
                now_ticks=self.wm.engine.now,
                attribution=replace(
                    self.wm._window_attribution(window, op, source_request_key),
                    relation=BoundaryTransferRelation(
                        source_request_key, source_key, dep_key,
                        version, delivery_version)),
            )
            self.wm.engine.schedule(
                reservation.total_delay_ticks,
                lambda dk=dep_key, so=op.id, bd=boundary,
                       sk=source_key, v=version, dv=delivery_version:
                    self._receive_boundary(dk, so, bd, sk, v, dv),
                label=f"boundary {op.name}W{window.k}->W{dep_key}")
    def merge_available(
        self, source_key: tuple, destination: Window, boundary,
    ) -> None:
        """Merge an already-delivered predecessor into a newly built window."""
        delivery_key = (source_key, destination.key)
        delivery = BoundaryDelivery(
            source_key=source_key,
            destination_key=destination.key,
            source_revision=self._boundary_versions.get(source_key, 0),
            delivery_revision=self._boundary_delivery_versions.get(
                delivery_key, 0),
            latest_source_revision=self._boundary_versions.get(source_key, 0),
            latest_delivery_revision=self._boundary_delivery_versions.get(
                delivery_key, 0),
            source_operation_round_count=self.wm.rounds_for(
                self.wm._ops[source_key[0]]),
            dependency_released=True,
            payload=boundary,
        )
        update = self._propose_boundary_update(delivery, destination)
        if update.release_dependency:
            raise RuntimeError(
                f"window interaction released boundary dependency "
                f"{delivery_key} more than once")
        if update.accepted:
            destination.boundary_in = update.state
    def _receive_boundary(self, key: tuple, src_op_id: int,
                          defects: Optional[dict] = None,
                          source_key: Optional[tuple] = None,
                          version: Optional[int] = None,
                          delivery_version: Optional[int] = None) -> None:
        if source_key is None or version is None or delivery_version is None:
            raise RuntimeError("boundary delivery is missing source provenance")
        delivery_key = (source_key, key)
        w = self.wm.windows[key]
        dependency_released = \
            delivery_key in self._released_boundary_dependencies
        delivery = BoundaryDelivery(
            source_key=source_key,
            destination_key=key,
            source_revision=version,
            delivery_revision=delivery_version,
            latest_source_revision=self._boundary_versions.get(source_key, 0),
            latest_delivery_revision=self._boundary_delivery_versions.get(
                delivery_key, 0),
            source_operation_round_count=self.wm.rounds_for(self.wm._ops[src_op_id]),
            dependency_released=dependency_released,
            payload=defects,
        )
        update = self._propose_boundary_update(delivery, w)
        if update.accepted and (w.queued or w.committed):
            raise RuntimeError(
                f"accepted boundary delivery {delivery_key} reached window "
                f"{key} after its decode lifecycle started")
        if update.release_dependency:
            if dependency_released:
                raise RuntimeError(
                    f"window interaction released boundary dependency "
                    f"{delivery_key} more than once")
            if source_key not in w.deps or w.deps_remaining <= 0:
                raise RuntimeError(
                    f"window interaction released unresolved edge "
                    f"{delivery_key}, but it is not a live dependency")
        if update.accepted:
            w.boundary_in = update.state
            if update.release_dependency:
                self._released_boundary_dependencies.add(delivery_key)
                w.deps_remaining -= 1
        self.wm.check_window(key)
    def _propose_boundary_update(
        self, delivery: BoundaryDelivery, destination: Window,
    ) -> BoundaryUpdate:
        """Let the interaction modify an isolated candidate boundary state."""
        try:
            candidate_state = deepcopy(destination.boundary_in)
        except Exception as error:
            raise TypeError(
                f"boundary state for {delivery.destination_key} must support "
                "deep copying before merge_boundary"
            ) from error
        destination_model = self.wm.window_models.get(destination.key)
        update = self.wm.window_interaction.merge_boundary(
            delivery,
            WindowInfo.from_window(
                destination,
                detector_positions=(
                    None if destination_model is None
                    else destination_model.defect_positions
                ),
            ),
            candidate_state,
        )
        self._validate_boundary_update(delivery, update)
        return update
    @staticmethod
    def _validate_boundary_update(
        delivery: BoundaryDelivery, update,
    ) -> None:
        if not update.accepted and update.release_dependency:
            raise RuntimeError(
                f"rejected boundary {delivery.source_key}->"
                f"{delivery.destination_key} cannot release a dependency")
