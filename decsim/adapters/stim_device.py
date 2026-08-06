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

    Numeric seeds create stable per-identity substreams; ``None`` lets Stim
    choose entropy.
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

    def __init__(
        self,
        seed: Optional[Integral] = None,
        detector_rounds: Optional[dict] = None,
        terminal_detector_ids: Optional[dict] = None,
        terminal_data_bits: Optional[dict] = None,
    ):
        """Configure sampling and explicit chronology or terminal metadata.

        Detector-round maps use one-based emitted rounds.
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
        self._terminal_detector_ids = {
            key: tuple(detector_ids)
            for key, detector_ids in (terminal_detector_ids or {}).items()
        }
        self._terminal_data_bits = dict(terminal_data_bits or {})
        self._samplers: dict = {}
        self._dets: dict = {}
        self._truth: dict = {}
        self._by_round: dict = {}
        self._stream_models: dict = {}
        self._source_bindings: dict = {}

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
                self._source_bindings,
            ) = reservation.prepared_state
            self._pending_run_seed = None
            self._run_seed_claimed = True

    @staticmethod
    def _key(op: Operation):
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
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Sample one fresh shot, or reuse the stream shot for later segments."""
        for name, value in (
            ("segment_round_count", segment_round_count),
            ("source_round_count", source_round_count),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive built-in int")
        if op.circuit is None:
            raise ValueError("StimDevice operations require a circuit")
        if op.stream_id is None:
            if op.stream_offset is not None or segment_round_count != source_round_count:
                raise ValueError("standalone duration must equal its source duration")
        else:
            if type(op.stream_offset) is not int or op.stream_offset < 0:
                raise ValueError("stream_offset must be a nonnegative built-in int")
            if op.stream_offset + segment_round_count > source_round_count:
                raise ValueError("stream segment extends beyond its finite source")
        key = self._key(op)
        if self._seed is not None:
            self._validate_sample_key(key)
        detector_rounds = self._source_rounds(
            key, op.circuit, source_round_count
        )
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
        terminal_ids = set(self._terminal_detector_ids.get(key, ()))
        buckets: dict[int, list[int]] = {}
        for detector_index, detector_round in detector_rounds.items():
            if detector_index not in terminal_ids:
                buckets.setdefault(detector_round, []).append(detector_index)
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

    def finalize_stream_round(
        self, op: Operation, source_round_count: int,
    ) -> list[SyndromePayload]:
        """Emit terminal detector events from the already sampled stream."""
        key = self._key(op)
        if key not in self._dets:
            raise RuntimeError("terminal finalizer requires a sampled stream")
        binding = self._source_bindings.get(key)
        if binding is None:
            raise RuntimeError("sampled stream has no source binding")
        circuit_text, bound_round_count, _ = binding
        if type(source_round_count) is not int or source_round_count < 1:
            raise ValueError("source_round_count must be a positive built-in int")
        if source_round_count != bound_round_count:
            raise ValueError("finalizer source duration differs from its binding")
        if op.circuit is None or str(op.circuit) != circuit_text:
            raise ValueError("finalizer circuit differs from its source binding")
        if type(op.stream_offset) is not int or op.stream_offset < 0:
            raise ValueError("finalizer offset must be a nonnegative built-in int")
        if op.stream_offset + 1 != source_round_count:
            raise ValueError("finalizer is not at the final source round")
        detector_ids = self._terminal_detector_ids.get(key)
        if detector_ids is None:
            raise ValueError("terminal finalizer has no declared detector ids")
        if key not in self._terminal_data_bits:
            raise ValueError("terminal finalizer has no raw data-bit size")
        patch = op.patches[0] if op.patches else op.qubits[0]
        return [SyndromePayload(
            key,
            patch,
            op.stream_offset + 1,
            bits=self._dets[key][list(detector_ids)],
            size_bits=self._terminal_data_bits[key],
        )]

    def idle_round_payloads(self, op: Operation, stream_id, global_round: int,
                            patch) -> list[SyndromePayload]:
        """Emit this feedback-idle stream round as one Stim-backed payload."""
        binding = self._source_bindings.get(stream_id)
        if stream_id not in self._dets or binding is None:
            raise RuntimeError("idle emission requires a sampled bound stream")
        if type(global_round) is not int or not 1 <= global_round <= binding[1]:
            raise ValueError("idle round is outside the finite source")
        detector_indices = self._by_round.get(stream_id, {}).get(global_round, [])
        bits = self._dets[stream_id][detector_indices]
        return [SyndromePayload(stream_id, patch, global_round, bits=bits,
                                size_bits=len(bits))]

    def _source_rounds(self, key, circuit, source_round_count: int) -> dict:
        """Bind one finite circuit, duration, and detector chronology per key."""
        from ..detector_error_model import resolve_detector_rounds

        resolved = resolve_detector_rounds(
            circuit,
            self._detector_rounds_override.get(key),
            source_round_count,
        )
        circuit_text = str(circuit)
        binding = self._source_bindings.get(key)
        if binding is None:
            private_rounds = dict(resolved)
            self._source_bindings[key] = (
                circuit_text, source_round_count, private_rounds
            )
            return private_rounds
        if binding[0] != circuit_text:
            raise ValueError("circuit differs from the bound finite source")
        if binding[1] != source_round_count:
            raise ValueError("source duration differs from the bound finite source")
        return binding[2]

    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, fault_model_requirement):
        """Prepare the window model source for one finite Stim-backed stream."""
        if stream_op.circuit is None:
            return None

        from ..detector_error_model import WindowSlicer

        detector_rounds = self._source_rounds(
            stream_op.id, stream_op.circuit, round_count)
        self._stream_models[stream_op.id] = {
            "round_count": round_count,
            "slicer": WindowSlicer(
                stream_op.circuit,
                round_count=round_count,
                detector_rounds=detector_rounds,
                fault_model_requirement=fault_model_requirement),
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
                                    round_count: int, *, fault_model_requirement,
                                    fault_exclusion_ranges: tuple) -> list:
        """Build detector error models for one finite Stim operation."""
        if op.circuit is None or not windows:
            return []

        from ..detector_error_model import build_window_error_models

        detector_rounds = self._source_rounds(
            self._key(op), op.circuit, round_count)
        model_plan = [
            (window.start_round, window.commit_lo, window.commit_hi,
             min(window.buffer_hi, round_count))
            for window in windows
        ]
        return build_window_error_models(
            op.circuit,
            model_plan,
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges)

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
                                          *, fault_model_requirement,
                                          exclude_faults_touching=None):
        """Build an independent two-sided context model for a strong re-decode
        with one optional inclusive range assigned to another seam side."""
        if op.circuit is None:
            return None

        from ..detector_error_model import build_single_window_error_model

        detector_rounds = self._source_rounds(
            self._key(op), op.circuit, round_count)
        return build_single_window_error_model(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            exclude_faults_touching=exclude_faults_touching)

    def strong_window_model_for_operation_with_exclusions(
        self, op: Operation, window, round_count: int, *,
        fault_model_requirement, fault_exclusion_ranges: tuple,
    ):
        """Build a strong model with multiple non-owned inclusive ranges."""
        if op.circuit is None:
            return None

        from ..detector_error_model import (
            build_single_window_error_model_with_exclusions,
        )

        detector_rounds = self._source_rounds(
            self._key(op), op.circuit, round_count)
        return build_single_window_error_model_with_exclusions(
            op.circuit,
            (window.start_round, window.commit_lo,
             window.commit_hi, min(window.buffer_hi, round_count)),
            round_count=round_count,
            detector_rounds=detector_rounds,
            fault_model_requirement=fault_model_requirement,
            fault_exclusion_ranges=fault_exclusion_ranges)
