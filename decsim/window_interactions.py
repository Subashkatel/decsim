"""Shipped decisions for relationships between decode windows."""

from __future__ import annotations

from dataclasses import replace

from .message import BoundaryUpdate, SeamFaultOwner, StrongRegionPlan


class _DefectBoundaryState(dict):
    """Combined defect masks plus each source's replaceable contribution."""

    def __init__(self, combined=None, contributions=None):
        super().__init__(combined or {})
        self.contributions = dict(contributions or {})


class DefaultWindowInteraction:
    """Preserve decsim's defect-mask and double-window behavior.

    The runtime applies the decisions returned by this object. The
    implementation has no manager reference and cannot mutate lifecycle
    state.
    """

    def initial_boundary_state(self, window):
        return _DefectBoundaryState()

    def boundary_from_result(self, result, fallback):
        if result is None:
            return fallback
        if result.boundary_defects is not None or result.correction is not None:
            return result.boundary_defects
        return fallback

    @staticmethod
    def _canonical_boundary(boundary):
        if not boundary:
            return None
        return {
            key: tuple(int(bit) for bit in mask)
            for key, mask in boundary.items()
        }

    def boundaries_equal(self, left, right):
        return self._canonical_boundary(left) == self._canonical_boundary(right)

    def boundary_targets(self, source, windows):
        return list(source.dependents)

    def merge_boundary(self, delivery, destination, current_state):
        if not delivery.is_current:
            return BoundaryUpdate(
                state=current_state,
                accepted=False,
                release_dependency=False,
            )
        prior_contributions = getattr(current_state, "contributions", {})
        contributions = dict(prior_contributions)
        contributions[delivery.source_key] = self._map_defects(
            delivery, destination)
        combined = {}
        for contribution in contributions.values():
            for key, mask in contribution.items():
                combined[key] = self._xor_mask(combined.get(key), mask)
        state = _DefectBoundaryState(combined, contributions)
        return BoundaryUpdate(
            state=state,
            accepted=True,
            release_dependency=not delivery.dependency_released,
        )

    @staticmethod
    def _map_defects(delivery, destination):
        mapped = {}
        defects = delivery.payload
        if not defects:
            return mapped
        shift = 0
        if delivery.source_key[0] != destination.op_id:
            shift = -delivery.source_operation_round_count
        for key, mask in defects.items():
            round_index, patch = key if isinstance(key, tuple) else (key, None)
            round_index += shift
            if round_index < 1:
                continue
            destination_key = (
                (round_index, patch) if patch is not None else round_index
            )
            mapped[destination_key] = list(mask)
        return mapped

    def apply_boundary(self, state, window, payload, round_key):
        state = state or {}
        mask = state.get(
            (round_key, payload.patch_id),
            state.get(round_key),
        )
        if mask is None:
            return payload
        bits = (
            tuple(int(bit) for bit in mask)
            if payload.bits is None
            else tuple(self._xor_mask(payload.bits, mask))
        )
        return replace(payload, bits=bits)

    def invalidated_windows(self, source_key, windows):
        found = set()
        stack = list(windows[source_key].dependents)
        while stack:
            key = stack.pop()
            if key in found:
                continue
            found.add(key)
            stack.extend(windows[key].dependents)
        return sorted(found)

    def plan_strong_region(
        self, weak_window, later_windows, operation_round_count,
    ):
        commit_round_count = weak_window.commit_hi - weak_window.commit_lo + 1
        buffer_round_count = weak_window.buffer_hi - weak_window.commit_hi
        strong_round_count = commit_round_count + 2 * buffer_round_count
        commit_lo = weak_window.commit_lo
        commit_hi = min(
            commit_lo + strong_round_count - 1,
            operation_round_count,
        )
        trailing_rounds = max(
            0, weak_window.buffer_hi - weak_window.commit_hi)
        context_lo = max(1, commit_lo - trailing_rounds)
        context_hi = min(
            commit_hi + trailing_rounds,
            operation_round_count,
        )

        has_restart = any(
            later.commit_lo > commit_hi for later in later_windows
        )

        return StrongRegionPlan(
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            context_lo=context_lo,
            context_hi=context_hi,
            restart_buffer_lo=(
                max(commit_lo, commit_hi - buffer_round_count + 1)
                if has_restart else None
            ),
            restart_seam_fault_owner=(
                SeamFaultOwner.STRONG_REGION
                if has_restart else None
            ),
        )

    @staticmethod
    def _xor_mask(previous_mask, incoming_mask):
        previous_bits = (
            [int(bit) for bit in previous_mask]
            if previous_mask is not None else []
        )
        incoming_bits = [int(bit) for bit in incoming_mask]
        if len(previous_bits) < len(incoming_bits):
            previous_bits += [0] * (len(incoming_bits) - len(previous_bits))
        for index, bit in enumerate(incoming_bits):
            previous_bits[index] ^= bit
        return previous_bits
