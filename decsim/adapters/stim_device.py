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
        self._stream_models: dict = {}

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

    def round_payloads(self, op: Operation, round_index: int) -> list[SyndromePayload]:
        """Emit this operation round as one Stim-backed payload."""
        key = self._key(op)
        global_round = round_index + (op.stream_offset or 0)
        detector_indices = self._by_round[key].get(global_round, [])
        bits = self._dets[key][detector_indices]
        patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
        target = op.stream_id if op.stream_id is not None else op.id
        return [SyndromePayload(target, patch, global_round, bits=bits)]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[SyndromePayload]:
        """Emit this feedback-idle stream round as one Stim-backed payload."""
        detector_indices = self._by_round.get(stream_id, {}).get(global_round, [])
        bits = self._dets[stream_id][detector_indices]
        return [SyndromePayload(stream_id, patch, global_round, bits=bits)]

    @staticmethod
    def _detector_rounds(circuit, round_count: int) -> dict:
        """Map each detector to a 1-based round, folding final detectors to the cap."""
        coords = circuit.get_detector_coordinates()
        return {
            detector_id: min(int(coordinate[-1]) + 1, round_count)
            for detector_id, coordinate in coords.items()
        }

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, belief_matching: bool = False):
        """Prepare the window model source for one finite Stim-backed stream."""
        if stream_op.circuit is None:
            return None

        from .window_error_models import WindowSlicer

        detector_rounds = self._detector_rounds(stream_op.circuit, round_count)
        self._stream_models[stream_op.id] = {
            "round_count": round_count,
            "slicer": WindowSlicer(
                stream_op.circuit,
                detector_rounds=detector_rounds,
                belief_matching=belief_matching),
        }
        return round_count

    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None:
        """Reject a Stim stream whose runtime length differs from its finite circuit."""
        stream_model = self._stream_models.get(stream_op.id)
        if stream_model is None:
            return

        finite_round_count = stream_model["round_count"]
        if stream_round_count == finite_round_count:
            return

        raise RuntimeError(
            f"{stream_op.name} sealed at {stream_round_count} rounds, but its Stim "
            f"circuit was registered for {finite_round_count} rounds. Real-syndrome "
            "live streams need an exact finite circuit. Use a timing-only stream for "
            "unknown feedback length, or build the Stim circuit after the stream length "
            "is known.")

    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int,
                                    *, belief_matching: bool = False) -> list:
        """Build detector error models for one finite Stim operation."""
        if op.circuit is None or not windows:
            return []

        from .window_error_models import build_window_error_models

        detector_rounds = self._detector_rounds(op.circuit, round_count)
        model_plan = [
            (window.start_round, window.commit_lo, window.commit_hi,
             min(window.buffer_hi, round_count))
            for window in windows
        ]
        return build_window_error_models(
            op.circuit,
            model_plan,
            detector_rounds=detector_rounds,
            belief_matching=belief_matching)

    def window_model_for_stream(self, stream_id, window, *, is_last: bool):
        """Build the detector error model for one dynamic stream window."""
        stream_model = self._stream_models.get(stream_id)
        if stream_model is None:
            return None

        buffer_lo = window.start_round
        return stream_model["slicer"].slice_window(
            buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi,
            is_last=is_last)

    def strong_window_model_for_operation(self, op: Operation, window, round_count: int,
                                          *, belief_matching: bool = False):
        """Build an independent two-sided context model for a strong re-decode."""
        if op.circuit is None:
            return None

        from .window_error_models import build_single_window_error_model

        detector_rounds = self._detector_rounds(op.circuit, round_count)
        return build_single_window_error_model(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            detector_rounds=detector_rounds,
            belief_matching=belief_matching)
