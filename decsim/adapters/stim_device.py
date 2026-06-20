"""Stim-backed syndrome source for real-decoding runs."""

from __future__ import annotations

from typing import Callable, Optional

from ..message import Operation, SyndromePayload


class StimDevice:
    """Sample Stim circuits and stream detection events by syndrome round."""

    def __init__(self, seed: Optional[int] = None,
                 rounds_for: Optional[Callable[[Operation], int]] = None):
        self._seed = seed
        self._rounds_for = rounds_for
        self._samplers: dict = {}
        self._dets: dict = {}
        self._truth: dict = {}
        self._by_round: dict = {}

    @staticmethod
    def _key(op: Operation):
        """Sample key for a standalone operation or continuous stream."""
        return op.stream_id if op.stream_id is not None else op.id

    def begin_operation(self, op: Operation) -> None:
        """Sample one fresh shot, or reuse the stream shot for later segments."""
        key = self._key(op)
        if op.stream_id is not None and op.stream_offset:
            self._dets[op.id] = self._dets[key]
            self._truth[op.id] = self._truth[key]
            return
        sampler = self._samplers.get(key)
        if sampler is None:
            sampler = op.circuit.compile_detector_sampler(seed=self._seed) \
                if self._seed is not None else op.circuit.compile_detector_sampler()
            self._samplers[key] = sampler
        dets, obs = sampler.sample(shots=1, separate_observables=True)
        self._dets[key] = dets[0]
        self._truth[key] = obs[0]
        coords = op.circuit.get_detector_coordinates()
        max_time_coordinate = max((int(c[-1]) for c in coords.values()), default=0)
        round_count = self._rounds_for(op) if self._rounds_for is not None \
            else max_time_coordinate
        buckets: dict[int, list[int]] = {}
        for detector_index, coordinate in coords.items():
            detector_round = int(coordinate[-1]) + 1
            buckets.setdefault(
                min(detector_round, round_count), []).append(detector_index)
        for detector_ids in buckets.values():
            detector_ids.sort()
        self._by_round[key] = buckets
        self._dets[op.id] = self._dets[key]
        self._truth[op.id] = self._truth[key]

    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit this round's detection-event bits."""
        key = self._key(op)
        global_round = round_index + (op.stream_offset or 0)
        detector_indices = self._by_round[key].get(global_round, [])
        bits = self._dets[key][detector_indices]
        patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
        target = op.stream_id if op.stream_id is not None else op.id
        return SyndromePayload(target, patch, global_round, bits=bits)
