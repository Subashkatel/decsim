"""The window interaction: how adjacent or replaced windows relate.

An interaction decides what boundary a window starts with, what boundary
a result carries, which dependents receive it, how a delivery merges
into the destination's state, how the state masks a landed round, and
which strong region replaces an escalated window. It returns data and
immutable decisions; the window side owns event ordering, retention,
logical accounting and finality. The default interaction is decsim's
defect-mask boundary (qLDPC net_error, cudaq-x syndrome_mods) and its
double-window strong region.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional, Protocol, runtime_checkable

import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


@runtime_checkable
class WindowInteraction(Protocol):
    """Decisions relating adjacent or replaced windows."""

    def initial_boundary_state(self, window: window_records.WindowInfo) -> Any:
        """The boundary a window starts with."""

    def boundary_from_result(
        self, result: Optional[decoding_records.DecodeResult], fallback: Any
    ) -> Any:
        """The boundary a decode result carries, else the fallback."""

    def boundary_targets(
        self,
        source: window_records.WindowInfo,
        windows: Mapping[tuple, window_records.WindowInfo],
    ) -> list:
        """The unstarted destinations among the source's declared edges."""

    def merge_boundary(
        self,
        delivery: window_records.BoundaryDelivery,
        destination: window_records.WindowInfo,
        current_state: Any,
    ) -> window_records.BoundaryUpdate:
        """The destination's boundary after this delivery."""

    def apply_boundary(
        self,
        state: Any,
        window: window_records.WindowInfo,
        payload,
        round_key: int,
    ):
        """Fold one delivered boundary into a landed round of the window."""

    def plan_strong_region(
        self,
        weak_window: window_records.WindowInfo,
        later_windows: list,
        operation_round_count: int,
    ) -> Optional[window_records.StrongRegionPlan]:
        """The strong window that replaces a weak window, or None."""


class DefaultWindowInteraction:
    """decsim's defect-mask boundary and double-window strong region.

    The boundary is a mask per (round, patch) or per round, XORed into
    the landed rounds when the decode starts; a same-operation A/B
    delivery is mapped by stable detector identity.
    """

    def initial_boundary_state(self, _window):
        """An empty mask."""
        return _DefectBoundaryState()

    def boundary_from_result(self, result, fallback):
        """The result's residual, its boundary defects, or the fallback."""
        if result is None:
            return fallback
        if isinstance(result.boundary_data, window_records.DependencyResidual):
            return result.boundary_data
        if result.boundary_defects is not None:
            return result.boundary_defects
        if result.correction is not None:
            return result.boundary_defects
        return fallback

    def boundary_targets(self, source, _windows):
        """Every declared dependent of the source."""
        return list(source.dependents)

    def merge_boundary(self, delivery, destination, current_state):
        """A current delivery replaces its source's contribution to the mask."""
        if not delivery.is_current:
            return window_records.BoundaryUpdate(
                state=current_state,
                accepted=False,
                release_dependency=False,
            )
        prior_contributions = getattr(current_state, "contributions", {})
        contributions = dict(prior_contributions)
        contributions[delivery.source_key] = _map_defects(delivery, destination)
        combined = {}
        for contribution in contributions.values():
            for key, mask in contribution.items():
                previous = combined.get(key)
                combined[key] = _xor_mask(previous, mask)
        state = _DefectBoundaryState(combined, contributions)
        release_dependency = not delivery.dependency_released
        return window_records.BoundaryUpdate(
            state=state,
            accepted=True,
            release_dependency=release_dependency,
        )

    def apply_boundary(self, state, _window, payload, round_key):
        """XOR the round's mask into the payload's bits.

        A timing-only payload (no bits) takes the mask as its bits.
        """
        state = state or {}
        patch_mask = state.get((round_key, payload.patch_id))
        mask = patch_mask
        if mask is None:
            mask = state.get(round_key)
        if mask is None:
            return payload
        if payload.bits is None:
            bits = tuple(int(bit) for bit in mask)
        else:
            masked = _xor_mask(payload.bits, mask)
            bits = tuple(masked)
        return dataclasses.replace(payload, bits=bits)

    def plan_strong_region(
        self, weak_window, _later_windows, operation_round_count
    ):
        """The forward strong window: commit plus two buffers, one restart.

        The strong window commits from the weak window's commit start over
        commit + 2 buffer rounds (clamped at the operation's end) and reads
        one buffer of context on each side; a restart window past it
        re-reads one buffer into the strong region, whose seam faults the
        strong region owns.
        """
        commit_round_count = weak_window.commit_hi - weak_window.commit_lo + 1
        buffer_round_count = weak_window.buffer_hi - weak_window.commit_hi
        strong_round_count = commit_round_count + 2 * buffer_round_count
        commit_lo = weak_window.commit_lo
        strong_end = commit_lo + strong_round_count - 1
        commit_hi = min(strong_end, operation_round_count)
        trailing_span = weak_window.buffer_hi - weak_window.commit_hi
        trailing_rounds = max(0, trailing_span)
        context_start = commit_lo - trailing_rounds
        context_lo = max(1, context_start)
        context_end = commit_hi + trailing_rounds
        context_hi = min(context_end, operation_round_count)
        has_restart = commit_hi < operation_round_count
        restart_buffer_lo = None
        restart_seam_fault_owner = None
        if has_restart:
            restart_start = commit_hi - buffer_round_count + 1
            restart_buffer_lo = max(commit_lo, restart_start)
            restart_seam_fault_owner = (
                window_records.SeamFaultOwner.STRONG_REGION
            )
        return window_records.StrongRegionPlan(
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            context_lo=context_lo,
            context_hi=context_hi,
            restart_buffer_lo=restart_buffer_lo,
            restart_seam_fault_owner=restart_seam_fault_owner,
        )


class _DefectBoundaryState(dict):
    """Combined defect masks plus each source's replaceable contribution."""

    def __init__(self, combined=None, contributions=None):
        if combined is None:
            combined = {}
        if contributions is None:
            contributions = {}
        dict.__init__(self, combined)
        self.contributions = dict(contributions)


def _map_defects(
    delivery: window_records.BoundaryDelivery, destination
) -> dict:
    """The delivery's defects in the destination's round coordinates."""
    payload = delivery.payload
    if _is_same_operation_residual(delivery, destination):
        return _map_detector_identities(payload.detector_ids, destination)
    defects = payload
    if isinstance(payload, window_records.DependencyResidual):
        defects = payload.defects
    if not defects:
        return {}
    shift = 0
    if delivery.source_key[0] != destination.op_id:
        shift = -delivery.source_operation_round_count
    return _map_shifted_defects(defects, shift, destination)


def _is_same_operation_residual(delivery, destination) -> bool:
    """A same-operation A/B delivery with stable global detector identity."""
    payload = delivery.payload
    if not isinstance(payload, window_records.DependencyResidual):
        return False
    if not payload.detector_ids:
        return False
    if delivery.source_key[0] != destination.op_id:
        return False
    return destination.detector_positions is not None


def _map_detector_identities(detector_ids, destination) -> dict:
    """Intersect the residual's detectors with the destination model.

    That is the exact residual H_destination times the committed source
    correction.
    """
    positions = destination.detector_positions
    mapped = {}
    for detector_id in detector_ids:
        if detector_id not in positions:
            continue
        round_index, position = positions[detector_id]
        mask = mapped.setdefault(round_index, [])
        if len(mask) <= position:
            padding = position + 1 - len(mask)
            padding_bits = [0] * padding
            mask.extend(padding_bits)
        mask[position] ^= 1
    return mapped


def _map_shifted_defects(defects: dict, shift: int, destination) -> dict:
    """Round-keyed defects moved by the shift and clipped to the window."""
    mapped = {}
    for key, mask in defects.items():
        round_index = key
        patch = None
        if isinstance(key, tuple):
            round_index, patch = key
        round_index += shift
        if not destination.start_round <= round_index <= destination.buffer_hi:
            continue
        destination_key = round_index
        if patch is not None:
            destination_key = (round_index, patch)
        mapped[destination_key] = list(mask)
    return mapped


def _xor_mask(previous_mask, incoming_mask) -> list:
    """The XOR of two masks, the shorter one padded with zeros."""
    previous_bits = []
    if previous_mask is not None:
        previous_bits = [int(bit) for bit in previous_mask]
    incoming_bits = [int(bit) for bit in incoming_mask]
    if len(previous_bits) < len(incoming_bits):
        padding = len(incoming_bits) - len(previous_bits)
        previous_bits += [0] * padding
    for index, bit in enumerate(incoming_bits):
        previous_bits[index] ^= bit
    return previous_bits
