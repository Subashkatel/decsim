"""Decoder models and routing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import us
from .message import DecodeJob, DecodeResult

if TYPE_CHECKING:
    from .protocols import Decoder


class CodeRouter:
    """Route each job by code name, with a default decoder fallback."""

    def __init__(self, default, by_code: dict = None):
        self.default = default
        self.by_code = dict(by_code) if by_code else {}

    def route(self, job: DecodeJob):
        """Pick the decoder for this job (by code; default when unmapped)."""
        return self.by_code.get(job.code, self.default)


class LatencyModelDecoder:
    """DecLat monomial latency model. It returns timing only."""

    def __init__(self, d: int, alpha: float = 2.85e-10, beta: float = 1.2):
        self.d = d
        self.alpha = alpha
        self.beta = beta

    def latency(self, job: DecodeJob) -> int:
        """Decode time = alpha * nodes^beta per round, times the window's rounds (ticks)."""
        node_count = job.spatial_nodes if job.spatial_nodes else self.d * self.d
        per_round = self.alpha * (node_count ** self.beta)
        return us(job.n_rounds * per_round * 1e6)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return an empty timing-only result."""
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None)


class PresetLatencyDecoder:
    """Timing-only decoder with one fixed latency for every job."""

    def __init__(self, latency_us: float = 1.0):
        self._lat = us(latency_us)

    def latency(self, job: DecodeJob) -> int:
        """Always return the same decode time (independent of window size)."""
        return self._lat

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return an empty timing-only result."""
        return DecodeResult(job.op_id, job.window_id)


class PerRoundDecoder:
    """Timing-only decoder with linear cost per syndrome round."""

    def __init__(self, tau_us: float = 1.0):
        self.tau_us = tau_us

    def latency(self, job: DecodeJob) -> int:
        """Decode time = per-round time times the window's rounds (ticks)."""
        return us(job.n_rounds * self.tau_us)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return an empty timing-only result."""
        return DecodeResult(job.op_id, job.window_id)


class ParityDecoder:
    """Toy data-path decoder that returns the parity of payload bits."""

    def __init__(self, d: int = 3, alpha: float = 2.85e-10, beta: float = 1.2):
        self._lat = LatencyModelDecoder(d=d, alpha=alpha, beta=beta)
        self.payloads_seen = 0

    def latency(self, job: DecodeJob) -> int:
        """Same decode time as the latency-model decoder."""
        return self._lat.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return the parity of all payload bits."""
        bits: list = []
        for payload in job.payloads:
            if payload.bits:
                bits += list(payload.bits)
        self.payloads_seen += len(job.payloads)
        return DecodeResult(job.op_id, job.window_id, correction=None,
                            logical_value=(sum(bits) % 2))


class UnionFindDecoder:
    """Union-Find decoder adapter backed by decsim.uf_decoder."""

    def __init__(self, code_structure, detector_remap: dict, latency_model, channel: str = "x"):
        from decsim.uf_decoder import uf_original
        self._uf_original = uf_original
        self.code_structure = code_structure
        self.detector_remap = dict(detector_remap)
        self.latency_model = latency_model
        self.channel = channel

    @classmethod
    def for_toric(cls, L: int, circuit, latency_model, channel: str = "x") -> "UnionFindDecoder":
        """Build a toric-code Union-Find decoder from a Stim circuit."""
        from decsim.uf_decoder import CodeStructure
        from decsim.uf_decoder.codes import (toric_code_x_stabilisers, toric_code_z_stabilisers,
                                             toric_code_x_logicals, toric_code_z_logicals)
        parity = 0 if channel == "x" else 1
        code_structure = CodeStructure(
            toric_code_x_stabilisers(L),
            toric_code_z_stabilisers(L),
            toric_code_x_logicals(L),
            toric_code_z_logicals(L),
            L,
            repetitions=L,
        )
        detector_remap = {}
        for detector_id, coordinate in circuit.get_detector_coordinates().items():
            if int(coordinate[1]) % 2 == parity:
                detector_remap[int(detector_id)] = (
                    int(coordinate[1]) // 2,
                    int(coordinate[0]) // 2,
                    int(coordinate[2]),
                )
        return cls(code_structure, detector_remap, latency_model, channel)

    def latency(self, job: DecodeJob) -> int:
        """Decode time from the latency model (decode WORK is done in decode())."""
        return self.latency_model.latency(job)

    def _predict(self, syndrome_dict) -> "np.ndarray":
        """Run Union-Find and project the correction onto logical operators."""
        import numpy as np
        from collections import defaultdict
        logicals = (self.code_structure.logicals_x if self.channel == "x"
                    else self.code_structure.logicals_z)
        syndrome_by_vertex = defaultdict(int)
        for vertex, value in syndrome_dict.items():
            syndrome_by_vertex[vertex] = value
        if not any(syndrome_by_vertex.values()):
            return np.zeros(logicals.shape[0], dtype=int)
        correction, _ = self._uf_original(
            syndrome_by_vertex, self.code_structure, self.channel, grow_mode="parallel")
        if correction is not None and len(correction) > 0 and correction[0] is not None:
            return np.asarray((correction[0] @ logicals.T) % 2).ravel()
        return np.zeros(logicals.shape[0], dtype=int)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Assemble the syndrome, run Union-Find, and return the logical value."""
        import numpy as np
        from collections import defaultdict
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        if job.payloads:
            syndrome = np.concatenate([
                np.asarray(payload.bits, np.uint8)
                for payload in job.payloads
                if payload.bits is not None
            ])
        else:
            syndrome = np.zeros(0, np.uint8)
        syndrome_by_vertex = defaultdict(int)
        for detector_index, detector_id in enumerate(model.detector_ids):
            if syndrome[detector_index]:
                remapped_vertex = self.detector_remap.get(int(detector_id))
                if remapped_vertex is not None:
                    syndrome_by_vertex[remapped_vertex] = 1
        predicted_logicals = self._predict(syndrome_by_vertex)
        return DecodeResult(job.op_id, job.window_id,
                            logical_value=int(predicted_logicals[0]))


class SwitchingDecoder:
    """Timing-level weak/strong decoder switch model."""

    def __init__(self, weak: "Decoder", strong: "Decoder", gamma_switch: float,
                 handoff_us: float = 0.5, seed: int = 0,
                 t_comm_weak_us: float = 0.0):
        import random
        self.weak = weak
        self.strong = strong
        self.gamma_switch = gamma_switch
        self.handoff = us(handoff_us)
        self.t_comm_weak = us(t_comm_weak_us)
        self.rng = random.Random(seed)
        self.switches = 0                      # diagnostic: how many jobs escalated

    def latency(self, job: DecodeJob) -> int:
        """Weak latency, plus handoff and strong latency when a switch is sampled."""
        latency_ticks = self.t_comm_weak + self.weak.latency(job)
        if self.rng.random() < self.gamma_switch:
            job.hint = "strong"
            self.switches += 1
            latency_ticks += 2 * self.handoff + self.strong.latency(job)
        return latency_ticks

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Decode through the path sampled by latency()."""
        if job.hint == "strong":
            res = self.strong.decode(job)
            res.soft_output = 0.0
        else:
            res = self.weak.decode(job)
            res.soft_output = 1.0
        return res


class SwitchingRouter:
    """Route strong side jobs to the strong decoder and all other jobs to weak."""

    def __init__(self, weak: "Decoder", strong: "Decoder"):
        self.weak = weak
        self.strong = strong

    def route(self, job: DecodeJob):
        """Strong decoder for escalated jobs, weak decoder for everything else."""
        return self.strong if job.hint == "strong" else self.weak


class SampledSoftOutputDecoder:
    """Weak decoder wrapper that emits sampled soft output."""

    def __init__(self, inner: "Decoder", escalation_probability: float, seed: int = 0,
                 probability_for=None):
        import random
        self.inner = inner
        self.escalation_probability = escalation_probability
        self.probability_for = probability_for
        self.rng = random.Random(seed)

    def latency(self, job: DecodeJob) -> int:
        """The weak decode latency; the strong path is the cluster's separate parallel job."""
        return self.inner.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Weak-decode the window, then attach the sampled soft output."""
        res = self.inner.decode(job)
        escalation_probability = self.probability_for(job) if self.probability_for is not None \
            else self.escalation_probability
        res.soft_output = 0.0 if self.rng.random() < escalation_probability else 1.0
        return res


def switch_probability_per_round(gamma_switch: float, d: int):
    """Build the paper's size-dependent per-window switch probability."""

    def probability(job: DecodeJob) -> float:
        w = job.window
        commit = (w.commit_hi - w.commit_lo + 1) if w is not None else job.n_rounds
        return gamma_switch * commit / d
    return probability


class RelayBPDecoder:
    """Timing-only Relay-BP latency model for qLDPC studies."""

    def __init__(self, iterations: int = 40, t_iter_ns: float = 24.0):
        self.iterations = iterations
        self.t_iter_ns = t_iter_ns

    def latency(self, job: DecodeJob) -> int:
        """Decode time equals iterations times time per iteration."""
        return us(self.iterations * self.t_iter_ns / 1000.0)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return an empty timing-only result."""
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None)
