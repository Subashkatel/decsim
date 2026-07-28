"""Stim-backed syndrome source for real-decoding runs."""

from __future__ import annotations

import hashlib
from numbers import Integral
import threading
from typing import Optional

from ..message import (
    Operation,
    RunSeedReservation,
    SyndromePayload,
)


class StimDevice:
    """Sample Stim circuits and stream detection events by syndrome round.

    ``seed=None`` delegates entropy selection to Stim and preserves legacy
    hashable operation/stream identities. A numeric seed may be any
    :class:`numbers.Integral` value in ``[0, 2**64)``; it enables stable
    per-identity substreams and therefore requires the selected operation id or
    stream id to be an exact built-in ``int`` or ``str``. Unsupported seeded
    identities raise before sampler-cache or stream-alias lookup.
    """

    operation_circuit_scope = "per_operation"

    @staticmethod
    def _manifest_identity(value):
        if type(value) is int:
            return {"kind": "integer", "value": str(value), "items": None}
        if type(value) is str:
            return {"kind": "string", "value": value, "items": None}
        if type(value) is tuple:
            return {
                "kind": "tuple",
                "value": None,
                "items": [
                    StimDevice._manifest_identity(item)
                    for item in value
                ],
            }
        raise TypeError(
            "StimDevice detector-round keys must use stable built-in "
            "int, str, or recursive tuple identities"
        )

    @staticmethod
    def _manifest_identity_bytes(value):
        if type(value) is int:
            encoded = str(value).encode("ascii")
            return b"I" + len(encoded).to_bytes(8, "big") + encoded
        if type(value) is str:
            encoded = value.encode("utf-8")
            return b"S" + len(encoded).to_bytes(8, "big") + encoded
        if type(value) is tuple:
            encoded_items = []
            for item in value:
                encoded = StimDevice._manifest_identity_bytes(item)
                encoded_items.append(
                    len(encoded).to_bytes(8, "big") + encoded
                )
            return (
                b"T"
                + len(encoded_items).to_bytes(8, "big")
                + b"".join(encoded_items)
            )
        raise TypeError(
            "StimDevice detector-round keys must use stable built-in "
            "int, str, or recursive tuple identities"
        )

    def run_manifest_config(self):
        rows = []
        for shot_key, detector_rounds in (
            self._detector_rounds_override.items()
        ):
            identity = self._manifest_identity(shot_key)
            rows.append((
                self._manifest_identity_bytes(shot_key),
                {
                    "shot_key": identity,
                    "detector_rounds": [
                        {"detector": detector, "round": round_index}
                        for detector, round_index in sorted(
                            detector_rounds.items()
                        )
                    ],
                },
            ))
        rows.sort(key=lambda item: item[0])
        return {
            "kind": "stim",
            "operation_circuit_scope": self.operation_circuit_scope,
            "detector_rounds": [row for _key, row in rows],
        }

    def __init__(
        self,
        seed: Optional[Integral] = None,
        detector_rounds: Optional[dict] = None,
    ):
        """Configure Stim sampling and optional detector-round overrides.

        ``detector_rounds`` maps an operation or stream identity to
        ``{detector: 1-based round}`` for circuits whose DETECTORs carry no
        coordinates (for example QLX-emitted circuits; the map comes from
        ``emit_decoder_params()['dem_detector_locs']`` packet indices).

        See the class contract for the root-seed domain and the conditional
        seeded identity restriction.
        """
        self._explicit_seed = seed
        self._seed = seed
        self._run_seed_lock = threading.Lock()
        self._pending_run_seed = None
        self._run_seed_claimed = False
        self._stochastic_use_started = False
        self._detector_rounds_override = {
            key: dict(rounds_map)
            for key, rounds_map in (detector_rounds or {}).items()}
        self._samplers: dict = {}
        self._dets: dict = {}
        self._truth: dict = {}
        self._by_round: dict = {}
        self._stream_models: dict = {}

    def reserve_run_seed(self, seed: Optional[int]) -> RunSeedReservation:
        """Prepare a run-root binding without changing active sampling state."""
        if seed is not None and (
            type(seed) is not int or not 0 <= seed < (1 << 64)
        ):
            raise TypeError(
                "StimDevice run root must be an unsigned 64-bit built-in "
                f"integer or None; got {seed!r}"
            )
        with self._run_seed_lock:
            if self._stochastic_use_started:
                raise ValueError(
                    "StimDevice was already used and cannot be rebound to a "
                    "run root; construct a fresh device"
                )
            if self._run_seed_claimed:
                raise ValueError(
                    "StimDevice is already claimed by a built run"
                )
            if self._pending_run_seed is not None:
                raise ValueError(
                    "StimDevice already has a pending run-seed reservation"
                )
            if seed is not None and self._explicit_seed is not None:
                raise ValueError(
                    "StimDevice has an explicit seed that conflicts with the "
                    f"numeric run root {seed}; move the seed to RunSpec"
                )

            if seed is not None:
                seed_source = "derived"
                effective_seed = seed
            elif self._explicit_seed is not None:
                seed_source = "explicit_local"
                effective_seed = self._validated_root_seed()
            else:
                seed_source = "entropy"
                effective_seed = None

            prepared_state = (
                effective_seed,
                {},
                {},
                {},
                {},
                {},
            )
            reservation = RunSeedReservation(
                proposed_seed_source=seed_source,
                proposed_seed=effective_seed,
                prepared_state=prepared_state,
            )
            self._pending_run_seed = reservation
            return reservation

    def cancel_run_seed(self, reservation: RunSeedReservation) -> None:
        """Release only the matching reversible run-seed reservation."""
        with self._run_seed_lock:
            if self._pending_run_seed is reservation:
                self._pending_run_seed = None

    def commit_run_seed(self, reservation: RunSeedReservation) -> None:
        """Install one prepared component seed after the root owns all leaves."""
        with self._run_seed_lock:
            if self._pending_run_seed is not reservation:
                raise ValueError(
                    "StimDevice can commit only its exact pending run-seed "
                    "reservation"
                )
            (
                self._seed,
                self._samplers,
                self._dets,
                self._truth,
                self._by_round,
                self._stream_models,
            ) = reservation.prepared_state
            self._pending_run_seed = None
            self._run_seed_claimed = True

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
        if not isinstance(self._seed, Integral):
            raise ValueError(
                f"seed must be None or a 64-bit unsigned integer; "
                f"got {self._seed!r}")
        root_seed = int(self._seed)
        if not 0 <= root_seed < (1 << 64):
            raise ValueError(
                f"seed must be None or a 64-bit unsigned integer; "
                f"got {self._seed!r}")
        return root_seed

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

    def begin_operation(
        self,
        op: Operation,
        resolved_round_count: int,
    ) -> None:
        """Sample one fresh shot, or reuse the stream shot for later segments."""
        if (
            type(resolved_round_count) is not int
            or resolved_round_count < 1
        ):
            raise ValueError(
                "resolved_round_count must be a positive built-in int"
            )
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
        with self._run_seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    "StimDevice cannot sample while a run-seed reservation "
                    "is pending"
                )
            self._stochastic_use_started = True
        dets, obs = sampler.sample(shots=1, separate_observables=True)
        self._dets[key] = dets[0]
        self._truth[key] = obs[0]
        override = self._detector_rounds_override.get(key)
        buckets: dict[int, list[int]] = {}
        if override is not None:
            max_round = max(override.values(), default=0)
            round_count = (
                max_round
                if op.stream_id is not None
                else resolved_round_count
            )
            for detector_index, detector_round in override.items():
                buckets.setdefault(
                    min(detector_round, round_count), []).append(detector_index)
        else:
            coords = op.circuit.get_detector_coordinates()
            max_time_coordinate = max(
                (int(c[-1]) for c in coords.values()), default=0)
            round_count = (
                max_time_coordinate
                if op.stream_id is not None
                else resolved_round_count
            )
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
