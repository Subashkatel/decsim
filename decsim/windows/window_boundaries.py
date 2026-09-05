"""The boundary courier: residual defects travel to dependent windows.

A boundary is the residual defects at a window's commit edge, produced by
the decoder at decode done and delivered to dependent windows over
decoder_to_decoder; a held boundary waits for a final result. Versions
make late deliveries harmless: every send bumps the source's version and
each delivery's version, and a receiver only accepts the latest. The
courier works on the window manager's tables (windows, models, links,
the interaction) and is built by the manager; slice 5's structural D
gives it its own record per source window and its own collaborators.
"""

import copy
import dataclasses
import functools
from typing import Optional

import decsim.message as message


@dataclasses.dataclass(frozen=True)
class HeldBoundary:
    """A boundary kept back until the window's result is final (Held)."""

    source_request_key: message.DecoderRequestKey
    operation_id: object
    boundary: object


class BoundaryCourier:
    """Delivers committed boundaries, versioned, held until final."""

    def __init__(self, window_manager):
        self.window_manager = window_manager
        self._committed_boundaries: dict[tuple, object] = {}
        self._boundary_versions: dict[tuple, int] = {}
        self._boundary_delivery_versions: dict[tuple, int] = {}
        self._released_boundary_dependencies: set[tuple] = set()
        self._held_boundary: dict[tuple, HeldBoundary] = {}

    # ---- what the manager asks

    def hold(self, key: tuple, held: HeldBoundary) -> None:
        """Keep a boundary back until the window's result is final."""
        self._held_boundary[key] = held

    def take_held(self, key: tuple) -> Optional[HeldBoundary]:
        """The boundary held for the window, taken out; None when none."""
        return self._held_boundary.pop(key, None)

    def has_committed(self, key: tuple) -> bool:
        """True once the window has shipped a boundary."""
        return key in self._committed_boundaries

    def committed(self, key: tuple):
        """The boundary the window shipped, or None."""
        return self._committed_boundaries.get(key)

    def invalidate(self, window: message.Window) -> None:
        """A window is about to be decoded again.

        A boundary already in transit belongs to the invalidated decode,
        so advancing the versions makes its scheduled delivery a no-op;
        its own shipped and held boundaries are gone.
        """
        key = window.key
        version = self._boundary_versions.get(key, 0)
        self._boundary_versions[key] = version + 1
        for dependency in window.deps:
            delivery_key = (dependency, key)
            delivery_version = self._boundary_delivery_versions.get(
                delivery_key, 0
            )
            self._boundary_delivery_versions[delivery_key] = (
                delivery_version + 1
            )
            self._released_boundary_dependencies.discard(delivery_key)
        self._committed_boundaries.pop(key, None)
        self._held_boundary.pop(key, None)

    # ---- delivery

    def send(
        self,
        window: message.Window,
        operation: message.Operation,
        boundary,
        *,
        source_request_key: message.DecoderRequestKey,
    ) -> None:
        """Ship a committed boundary to the dependents the interaction selects.

        A new version supersedes any delivery in flight.
        """
        source_key = (window.op_id, window.k)
        targets = self._targets(window)
        version = self._boundary_versions.get(source_key, 0)
        version += 1
        deliveries = []
        for dependent_key in targets:
            delivery_key = (source_key, dependent_key)
            delivery_version = self._boundary_delivery_versions.get(
                delivery_key, 0
            )
            delivery_version += 1
            deliveries.append((dependent_key, delivery_key, delivery_version))
        self._boundary_versions[source_key] = version
        self._committed_boundaries[source_key] = boundary
        for dependent_key, delivery_key, delivery_version in deliveries:
            self._boundary_delivery_versions[delivery_key] = delivery_version
            self._send_one(
                window,
                operation,
                boundary,
                source_request_key,
                source_key,
                dependent_key,
                version,
                delivery_version,
            )

    def _targets(self, window: message.Window) -> tuple:
        """The dependents the interaction selects, absorbed ones left out."""
        window_info = message.WindowInfo.from_window(window)
        window_infos = self.window_manager._window_infos()
        interaction = self.window_manager.window_interaction
        selected = interaction.boundary_targets(window_info, window_infos)
        targets = []
        for dependent_key in selected:
            dependent = self.window_manager.planner.windows_by_key[
                dependent_key
            ]
            if dependent.is_absorbed:
                continue
            targets.append(dependent_key)
        return tuple(targets)

    def _send_one(
        self,
        window: message.Window,
        operation: message.Operation,
        boundary,
        source_request_key: message.DecoderRequestKey,
        source_key: tuple,
        dependent_key: tuple,
        version: int,
        delivery_version: int,
    ) -> None:
        """One delivery over decoder_to_decoder, received at its landing."""
        window_attribution = message.TransferAttribution.for_window(
            window, operation, source_request_key
        )
        relation = message.BoundaryTransferRelation(
            source_request_key,
            source_key,
            dependent_key,
            version,
            delivery_version,
        )
        attribution = dataclasses.replace(window_attribution, relation=relation)
        receive = functools.partial(
            self._receive_boundary,
            dependent_key,
            operation.id,
            boundary,
            source_key,
            version,
            delivery_version,
        )
        self.window_manager.transfers.send_boundary(attribution, receive)

    def merge_available(
        self, source_key: tuple, destination: message.Window, boundary
    ) -> None:
        """Merge an already-delivered predecessor into a newly built window."""
        delivery_key = (source_key, destination.key)
        source_revision = self._boundary_versions.get(source_key, 0)
        delivery_revision = self._boundary_delivery_versions.get(
            delivery_key, 0
        )
        source_round_count = self._operation_round_count(source_key[0])
        delivery = message.BoundaryDelivery(
            source_key=source_key,
            destination_key=destination.key,
            source_revision=source_revision,
            delivery_revision=delivery_revision,
            latest_source_revision=source_revision,
            latest_delivery_revision=delivery_revision,
            source_operation_round_count=source_round_count,
            dependency_released=True,
            payload=boundary,
        )
        update = self._propose_boundary_update(delivery, destination)
        if update.release_dependency:
            raise RuntimeError(
                f"window interaction released boundary dependency "
                f"{delivery_key} more than once"
            )
        if update.accepted:
            destination.boundary_in = update.state

    def _operation_round_count(self, operation_id) -> int:
        return self.window_manager.planner.round_count_of(operation_id)

    def _receive_boundary(
        self,
        key: tuple,
        source_operation_id,
        defects,
        source_key: tuple,
        version: int,
        delivery_version: int,
        _transfer=None,
    ) -> None:
        """A delivery landed: merge it if current, release the edge once."""
        delivery_key = (source_key, key)
        window = self.window_manager.planner.windows_by_key[key]
        dependency_released = (
            delivery_key in self._released_boundary_dependencies
        )
        latest_source_revision = self._boundary_versions.get(source_key, 0)
        latest_delivery_revision = self._boundary_delivery_versions.get(
            delivery_key, 0
        )
        source_round_count = self._operation_round_count(source_operation_id)
        delivery = message.BoundaryDelivery(
            source_key=source_key,
            destination_key=key,
            source_revision=version,
            delivery_revision=delivery_version,
            latest_source_revision=latest_source_revision,
            latest_delivery_revision=latest_delivery_revision,
            source_operation_round_count=source_round_count,
            dependency_released=dependency_released,
            payload=defects,
        )
        update = self._propose_boundary_update(delivery, window)
        self._check_update(update, window, delivery_key, dependency_released)
        if update.accepted:
            window.boundary_in = update.state
            if update.release_dependency:
                self._released_boundary_dependencies.add(delivery_key)
                window.deps_remaining -= 1
        self._wake_parked_decode(key, window)
        self.window_manager.check_window(key)

    def _check_update(
        self,
        update: message.BoundaryUpdate,
        window: message.Window,
        delivery_key: tuple,
        dependency_released: bool,
    ) -> None:
        """Refuse an interaction's update that breaks the edge's contract."""
        if update.accepted and window.committed:
            raise RuntimeError(
                f"accepted boundary delivery {delivery_key} reached window "
                f"{window.key} after its decode lifecycle started"
            )
        if not update.release_dependency:
            return
        if dependency_released:
            raise RuntimeError(
                f"window interaction released boundary dependency "
                f"{delivery_key} more than once"
            )
        source_key = delivery_key[0]
        if source_key not in window.deps or window.deps_remaining <= 0:
            raise RuntimeError(
                f"window interaction released unresolved edge "
                f"{delivery_key}, but it is not a live dependency"
            )

    def _wake_parked_decode(self, key: tuple, window: message.Window) -> None:
        """The last boundary arrived for a shipped window: wake its decode."""
        if not window.queued or window.committed:
            return
        if window.deps_remaining != 0:
            return
        self.window_manager.requester.release_parked(key)

    def _propose_boundary_update(
        self, delivery: message.BoundaryDelivery, destination: message.Window
    ) -> message.BoundaryUpdate:
        """Let the interaction modify an isolated candidate boundary state."""
        try:
            candidate_state = copy.deepcopy(destination.boundary_in)
        except Exception as error:
            raise TypeError(
                f"boundary state for {delivery.destination_key} must support "
                "deep copying before merge_boundary"
            ) from error
        model = self.window_manager.planner.model_by_window.get(destination.key)
        detector_positions = None
        if model is not None:
            detector_positions = model.defect_positions
        destination_info = message.WindowInfo.from_window(
            destination, detector_positions=detector_positions
        )
        update = self.window_manager.window_interaction.merge_boundary(
            delivery, destination_info, candidate_state
        )
        if not update.accepted and update.release_dependency:
            raise RuntimeError(
                f"rejected boundary {delivery.source_key}->"
                f"{delivery.destination_key} cannot release a dependency"
            )
        return update
