from __future__ import annotations

from typing import Callable, Optional

from ..message import Operation, SyndromePayload
# =============================================================
# STIM DEVICE ADAPTER
# =============================================================

class StimDevice:
    """Samples each operation's stim.Circuit and streams detection events per round.

    Round alignment (the convention the whole real-decoding path shares): chip round
    r (1-based) carries the detectors with stim time coordinate t = r - 1, and every
    layer PAST the chip's last round folds into the last round's payload -- a memory
    circuit with R noisy rounds has R+1 detector layers (t = 0..R; layer R is the
    final data-measurement layer), and the chip only asks for R rounds. The folded
    rule is round = min(t + 1, R). WindowErrorModels for the same op must be built
    with the same folded mapping (the cluster does this) so that a window's
    concatenated payload bits line up with its model's rows exactly.
    """
    def __init__(self, seed: Optional[int] = None,
                 rounds_for: Optional[Callable[[Operation], int]] = None):
        """`seed` makes the sample stream deterministic (one stateful sampler per op,
        re-sampled on every begin_operation, so repeated runs draw successive shots).
        `rounds_for` overrides the chip rounds R used for folding; default R = the
        circuit's highest detector time coordinate (the memory-experiment shape)."""
        self._seed = seed
        self._rounds_for = rounds_for
        self._samplers: dict = {}
        self._dets: dict = {}
        # the TRUE observable values of each sample -- not consumed by the timing
        # pipeline; retained for accuracy studies (compare against the decoder's
        # logical_value, e.g. the LogicalErrorRate metric).
        self._truth: dict = {}
        self._by_round: dict = {}

    @staticmethod
    def _key(op: Operation):
        """Sample key: the STREAM id for a continuous-stream segment (one sample shared by
        every segment of the stream), else the op id (a standalone memory experiment)."""
        return op.stream_id if op.stream_id is not None else op.id

    def begin_operation(self, op: Operation) -> None:
        """Sample one fresh shot. For a CONTINUOUS STREAM (op.stream_id set) the shared circuit
        is sampled ONCE per shot -- when the first segment (stream_offset == 0) begins -- and the
        other segments reuse that one sample, so the stream decodes as a single continuous record
        with one observable. Across shots the offset-0 segment re-samples, giving fresh shots."""
        key = self._key(op)
        if op.stream_id is not None and op.stream_offset:
            # a later stream segment: reuse this shot's stream sample (already drawn by segment 0)
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
        max_t = max((int(c[-1]) for c in coords.values()), default=0)
        R = self._rounds_for(op) if self._rounds_for is not None else max_t
        buckets: dict[int, list[int]] = {}
        for det_index, c in coords.items():
            t = int(c[-1])                 # last coordinate = round, set via SHIFT_COORDS
            buckets.setdefault(min(t + 1, R), []).append(det_index)
        for idx in buckets.values():
            idx.sort()                     # ascending detector id within each round
        self._by_round[key] = buckets
        # mirror access by op id so device._dets[op_id] / _truth[op_id] work for any segment
        self._dets[op.id] = self._dets[key]
        self._truth[op.id] = self._truth[key]

    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit this round's REAL detection-event bits. For a stream segment, the op-local round
        maps to the continuous circuit's GLOBAL round (stream_offset + round), and the payload is
        TAGGED to the stream (operation_id = stream_id, round = global) so the cluster files it
        into the one continuous decode record. A standalone op tags itself with its own id and
        round, unchanged."""
        key = self._key(op)
        global_round = round_index + (op.stream_offset or 0)
        idx = self._by_round[key].get(global_round, [])
        bits = self._dets[key][idx]
        patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
        target = op.stream_id if op.stream_id is not None else op.id
        return SyndromePayload(target, patch, global_round, bits=bits)
