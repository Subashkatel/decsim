"""Stim-backed syndrome source for real-decoding runs."""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

from ..message import Operation, SyndromePayload


class StimDevice:
    """Sample Stim circuits and stream detection events by syndrome round."""

    def __init__(self, seed: Optional[int] = None,
                 rounds_for: Optional[Callable[[Operation], int]] = None,
                 detector_rounds: Optional[dict] = None):
        """detector_rounds: optional {op_or_stream_key: {detector: 1-based
        round}} override for circuits whose DETECTORs carry no coordinates
        (e.g. QLX-emitted circuits; the map comes from
        emit_decoder_params()['dem_detector_locs'] packet indices)."""
        self._seed = seed
        self._rounds_for = rounds_for
        self._detector_rounds_override = {
            key: dict(rounds_map)
            for key, rounds_map in (detector_rounds or {}).items()}
        self._samplers: dict = {}
        self._dets: dict = {}
        self._truth: dict = {}
        self._by_round: dict = {}
        self._stream_models: dict = {}

    @staticmethod
    def _key(op: Operation):
        """Sample key for a standalone operation or continuous stream."""
        return op.stream_id if op.stream_id is not None else op.id

    @staticmethod
    def _validate_sample_key(key) -> None:
        """Reject identities whose equality can alias a legal cache key."""
        sample_key_type = type(key)
        if sample_key_type not in (int, str):
            raise TypeError(
                f"stream_id must be an int or str so the run's sampling is "
                f"reproducible across processes; sample key {key!r} is a "
                f"{sample_key_type.__name__}")

    def _validated_root_seed(self) -> int:
        """Return the seed under Stim's public unsigned-64-bit contract."""
        if (not isinstance(self._seed, int)
                or not 0 <= self._seed < (1 << 64)):
            raise ValueError(
                f"seed must be None or a 64-bit unsigned integer; "
                f"got {self._seed!r}")
        return int(self._seed)

    def _sample_seed(self, key) -> int:
        """Derive a cross-process-stable substream from the shot-reuse identity."""
        self._validate_sample_key(key)
        root_seed = self._validated_root_seed()
        sample_key_type = type(key)
        key_type_tag = b"int" if sample_key_type is int else b"str"
        hash_input = b"\0".join((
            str(root_seed).encode(),
            b"stim_device",
            key_type_tag,
            str(key).encode(),
        ))
        digest = hashlib.blake2b(
            hash_input,
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big")

    def begin_operation(self, op: Operation) -> None:
        """Sample one fresh shot, or reuse the stream shot for later segments."""
        key = self._key(op)
        if self._seed is not None:
            self._validate_sample_key(key)
        if op.stream_id is not None and op.stream_offset:
            self._dets[op.id] = self._dets[key]
            self._truth[op.id] = self._truth[key]
            return
        sampler = self._samplers.get(key)
        if sampler is None:
            if self._seed is None:
                sampler = op.circuit.compile_detector_sampler()
            else:
                sample_seed = self._sample_seed(key)
                sampler = op.circuit.compile_detector_sampler(seed=sample_seed)
            self._samplers[key] = sampler
        dets, obs = sampler.sample(shots=1, separate_observables=True)
        self._dets[key] = dets[0]
        self._truth[key] = obs[0]
        override = self._detector_rounds_override.get(key)
        buckets: dict[int, list[int]] = {}
        if override is not None:
            max_round = max(override.values(), default=0)
            round_count = self._rounds_for(op) if self._rounds_for is not None \
                else max_round
            for detector_index, detector_round in override.items():
                buckets.setdefault(
                    min(detector_round, round_count), []).append(detector_index)
        else:
            coords = op.circuit.get_detector_coordinates()
            max_time_coordinate = max(
                (int(c[-1]) for c in coords.values()), default=0)
            round_count = self._rounds_for(op) if self._rounds_for is not None \
                else max_time_coordinate
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
        return [SyndromePayload(target, patch, global_round, bits=bits,
                                size_bits=len(bits))]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[SyndromePayload]:
        """Emit this feedback-idle stream round as one Stim-backed payload."""
        detector_indices = self._by_round.get(stream_id, {}).get(global_round, [])
        bits = self._dets[stream_id][detector_indices]
        return [SyndromePayload(stream_id, patch, global_round, bits=bits,
                                size_bits=len(bits))]

    @staticmethod
    def _detector_rounds(circuit, round_count: int) -> dict:
        """Map each detector to a 1-based round, folding final detectors to the cap."""
        coords = circuit.get_detector_coordinates()
        return {
            detector_id: min(int(coordinate[-1]) + 1, round_count)
            for detector_id, coordinate in coords.items()
        }

    def _detector_rounds_for_key(self, key, circuit, round_count: int) -> dict:
        """The detector->round map for one op/stream: explicit override when
        registered (coordinate-less circuits), else circuit coordinates."""
        override = self._detector_rounds_override.get(key)
        if override is not None:
            return {detector_id: min(detector_round, round_count)
                    for detector_id, detector_round in override.items()}
        return self._detector_rounds(circuit, round_count)

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, belief_matching: bool = False):
        """Prepare the window model source for one finite Stim-backed stream."""
        if stream_op.circuit is None:
            return None

        from ..detector_error_model import WindowSlicer

        detector_rounds = self._detector_rounds_for_key(
            stream_op.id, stream_op.circuit, round_count)
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

        from ..detector_error_model import build_window_error_models

        detector_rounds = self._detector_rounds_for_key(
            self._key(op), op.circuit, round_count)
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
                                          *, belief_matching: bool = False,
                                          exclude_faults_touching=None):
        """Build an independent two-sided context model for a strong re-decode
        with one optional inclusive range assigned to another seam side."""
        if op.circuit is None:
            return None

        from ..detector_error_model import build_single_window_error_model

        detector_rounds = self._detector_rounds_for_key(
            self._key(op), op.circuit, round_count)
        return build_single_window_error_model(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            detector_rounds=detector_rounds,
            belief_matching=belief_matching,
            exclude_faults_touching=exclude_faults_touching)

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        belief_matching: bool = False, fault_exclusion_ranges: tuple,
    ):
        """Build a strong model with multiple non-owned inclusive ranges."""
        if op.circuit is None:
            return None

        from ..detector_error_model import (
            build_single_window_error_model_with_exclusions,
        )

        detector_rounds = self._detector_rounds_for_key(
            self._key(op), op.circuit, round_count)
        return build_single_window_error_model_with_exclusions(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            detector_rounds=detector_rounds,
            belief_matching=belief_matching,
            fault_exclusion_ranges=fault_exclusion_ranges)
