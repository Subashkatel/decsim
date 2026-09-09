"""The boundary courier: residual defects travel to dependent windows.

A boundary is the residual defects at a window's commit edge, produced by
the decoder at decode done and delivered to dependent windows over
decoder_to_decoder; a held boundary waits for a final result. Versions
make late deliveries harmless: every send bumps the source's version and
each delivery's version, and a receiver only accepts the latest. Each
source window has one record here; the courier owns the boundary policy
(when a committed window may ship) and the interaction (what a boundary
is and how it merges), and tells the window side when a delivery landed.
A strong window whose face is pinned reads the same records: the
courier ships the neighbour's committed boundary to it and folds it in
(pin_strong_face, Bombin et al. 2303.04846 lines 775-788).
"""

import copy
import dataclasses
import functools
from typing import Callable, Optional

import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class HeldBoundary:
    """A boundary kept back until the window's result is final (Held)."""

    source_request_key: window_records.DecoderRequestKey
    operation: program_records.Operation
    boundary: object


class BoundaryCourier:
    """Delivers committed boundaries, versioned, held until final."""

    def __init__(
        self,
        planner,
        decoder_output,
        interaction,
        boundary_policy,
        on_boundary_received: Callable[[tuple, bool], None],
    ) -> None:
        self.planner = planner
        # the boundary leaves a decoder, so the decoder side sends it
        self.decoder_output = decoder_output
        self.interaction = interaction
        self.boundary_policy = boundary_policy
        # (source window key, is_unblocked): a delivery landed in a window;
        # True when it was the last boundary a shipped window owed
        self.on_boundary_received = on_boundary_received
        self.record_by_window: dict = {}

    # ---- what the committer asks

    def hand_on(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        is_final: bool,
    ) -> None:
        """Ship the decoder's boundary to dependent windows, or hold it."""
        boundary = self.interaction.boundary_from_result(result, None)
        if self.boundary_policy.on_commit(window, final=is_final):
            self.send(
                window, operation, boundary, source_request_key=request_key
            )
            return
        record = self._record(window.key)
        record.held = HeldBoundary(request_key, operation, boundary)

    def ship_held(
        self,
        window: window_records.Window,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """The window's result is final: a boundary held for it ships now."""
        record = self._record(window.key)
        held = record.held
        if held is None:
            return
        record.held = None
        boundary = self.interaction.boundary_from_result(result, held.boundary)
        self.send(
            window, held.operation, boundary, source_request_key=request_key
        )

    def has_committed(self, key: tuple) -> bool:
        """True once the window has shipped a boundary."""
        record = self.record_by_window.get(key)
        if record is None:
            return False
        return record.committed_request_key is not None

    def committed(self, key: tuple):
        """The boundary the window shipped, or None."""
        record = self.record_by_window.get(key)
        if record is None:
            return None
        return record.committed

    def invalidate(self, window: window_records.Window) -> None:
        """A window is about to be decoded again.

        A boundary already in transit belongs to the invalidated decode,
        so advancing the versions makes its scheduled delivery a no-op;
        its own shipped and held boundaries are gone.
        """
        record = self._record(window.key)
        record.version += 1
        for dependency in window.deps:
            source = self._record(dependency)
            delivery_version = source.delivery_version_by_dependent.get(
                window.key, 0
            )
            source.delivery_version_by_dependent[window.key] = (
                delivery_version + 1
            )
            source.released_dependents.discard(window.key)
        record.committed = None
        record.committed_request_key = None
        record.held = None

    # ---- delivery

    def send(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        boundary,
        *,
        source_request_key: window_records.DecoderRequestKey,
    ) -> None:
        """Ship a committed boundary to the dependents the interaction selects.

        A new version supersedes any delivery in flight.
        """
        source_key = window.key
        record = self._record(source_key)
        targets = self._targets(window)
        record.version += 1
        record.committed = boundary
        record.committed_request_key = source_request_key
        for dependent_key in targets:
            delivery_version = record.delivery_version_by_dependent.get(
                dependent_key, 0
            )
            delivery_version += 1
            record.delivery_version_by_dependent[dependent_key] = (
                delivery_version
            )
            self._send_one(
                window,
                operation,
                boundary,
                source_request_key,
                dependent_key,
                record.version,
                delivery_version,
            )

    def merge_available(
        self, source_key: tuple, destination: window_records.Window, boundary
    ) -> None:
        """Merge an already-delivered predecessor into a newly built window."""
        record = self._record(source_key)
        delivery_revision = record.delivery_version_by_dependent.get(
            destination.key, 0
        )
        source_round_count = self.planner.round_count_of(source_key[0])
        delivery = window_records.BoundaryDelivery(
            source_key=source_key,
            destination_key=destination.key,
            source_revision=record.version,
            delivery_revision=delivery_revision,
            latest_source_revision=record.version,
            latest_delivery_revision=delivery_revision,
            source_operation_round_count=source_round_count,
            dependency_released=True,
            payload=boundary,
        )
        update = self._propose_boundary_update(delivery, destination)
        if update.release_dependency:
            raise RuntimeError(
                f"window interaction released boundary dependency "
                f"{(source_key, destination.key)} more than once"
            )
        if update.accepted:
            destination.boundary_in = update.state

    # ---- a strong window's pinned face

    def pin_strong_face(
        self,
        source_key: tuple,
        destination: window_records.Window,
        model,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """Pin a strong window's face on what its neighbour committed.

        Bombin et al. 2303.04846 lines 775-788: the input to a later
        decoding task is the syndrome of the errors plus the corrections
        committed by the tasks before it, which updates the detectors of
        the one round layer where the two windows meet (Tan et al.
        2209.09219 lines 943-946). The strong window has its own error
        model, so the residual is intersected with its detectors and the
        message is priced against its own seam layer rather than the
        weak window's. The delivery rides decoder_to_decoder, where
        every boundary of this courier is charged; the fold is written
        into the strong window's boundary state here and the job's gate
        XORs it into the input when the decode starts.
        """
        record = self._record(source_key)
        if record.committed_request_key is None:
            raise RuntimeError(
                f"strong window {destination.key} pins a face on window "
                f"{source_key}, which has not committed"
            )
        positions = _row_positions(model)
        destination_info = window_records.WindowInfo.from_window(
            destination, detector_positions=positions
        )
        self._fold_pin(source_key, destination, destination_info, record)
        self._send_pin(
            source_key, destination, destination_info, operation, request_key
        )

    def _fold_pin(
        self,
        source_key: tuple,
        destination: window_records.Window,
        destination_info: window_records.WindowInfo,
        record: "_BoundaryRecord",
    ) -> None:
        """The committed boundary joins the strong window's boundary state.

        The strong window carries the escalated window's key and is not
        a dependent of the source, so the pin reads the delivery
        bookkeeping of the weak edge and advances none of it.
        """
        delivery_revision = record.delivery_version_by_dependent.get(
            destination.key, 0
        )
        source_round_count = self.planner.round_count_of(source_key[0])
        delivery = window_records.BoundaryDelivery(
            source_key=source_key,
            destination_key=destination.key,
            source_revision=record.version,
            delivery_revision=delivery_revision,
            latest_source_revision=record.version,
            latest_delivery_revision=delivery_revision,
            source_operation_round_count=source_round_count,
            dependency_released=True,
            payload=record.committed,
        )
        update = self._merged_update(delivery, destination, destination_info)
        if update.accepted:
            destination.boundary_in = update.state

    def _send_pin(
        self,
        source_key: tuple,
        destination: window_records.Window,
        destination_info: window_records.WindowInfo,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """One pinned face's message, priced on the strong window's seam.

        The bits are the strong window's own seam layer, so the transfer
        is attributed to that window and to the request that reads it;
        the relation names the neighbour's committed request, its window
        and the strong window it lands in. A weak delivery to the same
        window key is attributed to its source instead, so the two are
        told apart in the traffic.
        """
        record = self._record(source_key)
        delivery_revision = record.delivery_version_by_dependent.get(
            destination.key, 0
        )
        window_attribution = transfer_records.TransferAttribution.for_window(
            destination, operation, request_key
        )
        relation = transfer_records.BoundaryTransferRelation(
            record.committed_request_key,
            source_key,
            destination.key,
            record.version,
            delivery_revision,
        )
        attribution = dataclasses.replace(window_attribution, relation=relation)
        payload_bits = self.interaction.boundary_payload_bits(
            record.committed, destination_info
        )
        self.decoder_output.send_boundary(
            attribution, payload_bits, _pin_delivered
        )

    # ---- private

    def _record(self, key: tuple) -> "_BoundaryRecord":
        record = self.record_by_window.get(key)
        if record is None:
            record = _BoundaryRecord()
            self.record_by_window[key] = record
        return record

    def _targets(self, window: window_records.Window) -> tuple:
        """The dependents the interaction selects, absorbed ones left out."""
        window_info = window_records.WindowInfo.from_window(window)
        window_infos = self._window_infos()
        selected = self.interaction.boundary_targets(window_info, window_infos)
        targets = []
        for dependent_key in selected:
            dependent = self.planner.windows_by_key[dependent_key]
            if dependent.is_absorbed:
                continue
            targets.append(dependent_key)
        return tuple(targets)

    def _window_infos(self) -> dict:
        infos = {}
        for key, window in self.planner.windows_by_key.items():
            infos[key] = window_records.WindowInfo.from_window(window)
        return infos

    def _send_one(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        boundary,
        source_request_key: window_records.DecoderRequestKey,
        dependent_key: tuple,
        version: int,
        delivery_version: int,
    ) -> None:
        """One delivery over decoder_to_decoder, received at its landing."""
        window_attribution = transfer_records.TransferAttribution.for_window(
            window, operation, source_request_key
        )
        relation = transfer_records.BoundaryTransferRelation(
            source_request_key,
            window.key,
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
            window.key,
            version,
            delivery_version,
        )
        payload_bits = self._boundary_bits(boundary, dependent_key)
        self.decoder_output.send_boundary(attribution, payload_bits, receive)

    def _boundary_bits(self, boundary, dependent_key: tuple):
        """The bits this hand-off takes on the wire, or None for the card.

        The interaction owns what a boundary is, so it is the one that
        counts what crosses.
        """
        destination = self.planner.windows_by_key[dependent_key]
        destination_info = self._destination_info(destination)
        return self.interaction.boundary_payload_bits(
            boundary, destination_info
        )

    def _destination_info(
        self, destination: window_records.Window
    ) -> window_records.WindowInfo:
        """The destination as a policy reads it, with its detector layers."""
        model = self.planner.models.model_by_window.get(destination.key)
        detector_positions = None
        if model is not None:
            detector_positions = model.defect_positions
        return window_records.WindowInfo.from_window(
            destination, detector_positions=detector_positions
        )

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
        record = self._record(source_key)
        window = self.planner.windows_by_key[key]
        dependency_released = key in record.released_dependents
        latest_delivery_revision = record.delivery_version_by_dependent.get(
            key, 0
        )
        source_round_count = self.planner.round_count_of(source_operation_id)
        delivery = window_records.BoundaryDelivery(
            source_key=source_key,
            destination_key=key,
            source_revision=version,
            delivery_revision=delivery_version,
            latest_source_revision=record.version,
            latest_delivery_revision=latest_delivery_revision,
            source_operation_round_count=source_round_count,
            dependency_released=dependency_released,
            payload=defects,
        )
        update = self._propose_boundary_update(delivery, window)
        delivery_key = (source_key, key)
        self._check_update(update, window, delivery_key, dependency_released)
        if update.accepted:
            window.boundary_in = update.state
            if update.release_dependency:
                record.released_dependents.add(key)
                window.deps_remaining -= 1
        is_unblocked = self._is_unblocked_now(window)
        self.on_boundary_received(key, is_unblocked)

    @staticmethod
    def _is_unblocked_now(window: window_records.Window) -> bool:
        """The last boundary arrived for a shipped window still waiting."""
        if not window.queued or window.committed:
            return False
        return window.deps_remaining == 0

    def _check_update(
        self,
        update: window_records.BoundaryUpdate,
        window: window_records.Window,
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

    def _propose_boundary_update(
        self,
        delivery: window_records.BoundaryDelivery,
        destination: window_records.Window,
    ) -> window_records.BoundaryUpdate:
        """Let the interaction modify an isolated candidate boundary state."""
        destination_info = self._destination_info(destination)
        return self._merged_update(delivery, destination, destination_info)

    def _merged_update(
        self,
        delivery: window_records.BoundaryDelivery,
        destination: window_records.Window,
        destination_info: window_records.WindowInfo,
    ) -> window_records.BoundaryUpdate:
        """The interaction's merge, against the model the caller reads with.

        A weak window is read with the model the planner holds for it; a
        strong window is read with the model of its own re-decode, which
        the caller passes because both windows carry the same key.
        """
        try:
            candidate_state = copy.deepcopy(destination.boundary_in)
        except Exception as error:
            raise TypeError(
                f"boundary state for {delivery.destination_key} must support "
                "deep copying before merge_boundary"
            ) from error
        update = self.interaction.merge_boundary(
            delivery, destination_info, candidate_state
        )
        if not update.accepted and update.release_dependency:
            raise RuntimeError(
                f"rejected boundary {delivery.source_key}->"
                f"{delivery.destination_key} cannot release a dependency"
            )
        return update


def _row_positions(model) -> dict:
    """Where the model's own detector rows sit, and no other detector.

    A window's input is its own checks, so the residual is intersected
    with the rows the strong window decodes on (Bombin et al.
    2303.04846 lines 782-788, the input to task j is restricted to
    Sigma_j). The model also places the detectors its faults flip
    outside the window, which belong to a neighbour's input and not to
    this one's. A run whose windows carry no error model has no rows to
    intersect with, and the wire prices the message by its card.
    """
    if model is None:
        return None
    positions = {}
    for detector_id in model.detector_ids:
        positions[detector_id] = model.defect_positions[detector_id]
    return positions


def _pin_delivered(_transfer) -> None:
    """A pinned face's message landed.

    The pin is folded into the strong window when its job is built, so
    the delivery charges the wire and changes nothing on arrival.
    """


class _BoundaryRecord:
    """One source window's boundary: shipped, versioned, held, delivered."""

    def __init__(self) -> None:
        self.committed = None
        # the request that produced the shipped boundary; None until
        # the window has committed one, which is what has_committed asks
        self.committed_request_key = None
        self.version = 0
        self.delivery_version_by_dependent: dict = {}
        self.released_dependents: set = set()
        self.held: Optional[HeldBoundary] = None
