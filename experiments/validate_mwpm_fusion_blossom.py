"""Gate 3: decsim's inner MWPM (PyMatching on its window models) against Fusion
Blossom, an independent exact MWPM solver (pinned at tmp/references/code/
fusion-blossom, fa69060, Rust, checked upstream against Blossom V).

For every window model the loop builds and every recorded syndrome, both
solvers get the same graph: decsim's graphlike faults (check matrix, priors ->
log-odds weights) with the same integer weight scaling. Compared: minimum
matching weight, whether the correction satisfies the syndrome, and the
predicted logical class. Ties between equal-weight matchings may legitimately
give different physical corrections; only weight and logical class must agree.

Usage: python -m experiments.validate_mwpm_fusion_blossom [shots]
"""

from __future__ import annotations

import sys
from pathlib import Path

import fusion_blossom as fb
import numpy as np
import pymatching
import stim

from decsim.adapters.stim_device import RecordedStimDevice
from decsim.decoders import PresetLatencyDecoder
from decsim.detector_error_model.fault_model_contracts import FaultRepresentation
from decsim.message import Operation
from decsim.mwpm_decoder.decoder import PyMatchingDecoder
from decsim.mwpm_decoder.weights import matching_weights
from decsim.rounds import FixedRounds
from decsim.run_spec import RunSpec

REPORT = Path("experiments/results/validation/fusion_blossom_mwpm.md")
WEIGHT_SCALE = 1000          # Fusion Blossom takes integer weights


class _CapturingDecoder(PyMatchingDecoder):
    """PyMatchingDecoder that also records (faults, syndrome, weight, logical) per call."""

    def __init__(self):
        super().__init__(PresetLatencyDecoder(0.028))
        self.calls = []

    def decode(self, job):
        result = super().decode(job)
        model = job.dem
        if model is not None:
            from decsim.mwpm_decoder.decoder import payload_syndrome
            faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
            syndrome = payload_syndrome(job)
            matching = self._matching_for_model(faults)
            selected, weight = matching.decode(syndrome, return_weight=True)
            self.calls.append((faults, np.asarray(syndrome, dtype=np.uint8),
                               float(weight), np.asarray(selected, dtype=np.uint8)))
        return result


def fusion_blossom_solve(faults, syndrome):
    """Return (weight, logical bits, satisfies) from Fusion Blossom on the same graph."""
    check = np.asarray(faults.check.todense() if hasattr(faults.check, "todense") else faults.check)
    observables = np.asarray(faults.observables.todense()
                             if hasattr(faults.observables, "todense") else faults.observables)
    weights = matching_weights(faults.priors)
    detectors, num_faults = check.shape
    boundary = detectors                              # one virtual vertex for boundary edges
    # One edge per vertex pair: parallel faults keep the smallest weight, which is
    # PyMatching's from_check_matrix merge rule; Fusion Blossom expects a simple graph.
    best = {}
    for fault in range(num_faults):
        touched = np.flatnonzero(check[:, fault])
        w = max(1, int(round(weights[fault] * WEIGHT_SCALE)))
        w += w % 2                                    # Fusion Blossom wants even weights
        if len(touched) == 1:
            pair = (int(touched[0]), boundary)
        elif len(touched) == 2:
            pair = (int(touched[0]), int(touched[1]))
        else:
            raise ValueError("non-graphlike fault reached the matching gate")
        if pair not in best or w < best[pair][0]:
            best[pair] = (w, fault)
    edges = [(u, v, w) for (u, v), (w, _fault) in best.items()]
    edge_fault = [fault for (_u, _v), (_w, fault) in best.items()]
    initializer = fb.SolverInitializer(detectors + 1, edges, [boundary])
    solver = fb.SolverSerial(initializer)
    defects = [int(v) for v in np.flatnonzero(syndrome)]
    solver.solve(fb.SyndromePattern(defects))
    chosen = solver.subgraph()
    weight = sum(edges[e][2] for e in chosen) / WEIGHT_SCALE
    logical = np.zeros(observables.shape[0], dtype=np.uint8)
    flipped = np.zeros(detectors, dtype=np.uint8)
    for e in chosen:
        logical ^= observables[:, edge_fault[e]].astype(np.uint8)
        for v in (edges[e][0], edges[e][1]):
            if v != boundary:
                flipped[v] ^= 1
    solver.clear()
    return weight, logical, bool(np.array_equal(flipped, syndrome.astype(np.uint8)))


def main(argv) -> None:
    shots = int(argv[1]) if len(argv) > 1 else 200
    d, rounds, p = 3, 30, 0.01
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)
    dets, obs = circuit.compile_detector_sampler(seed=5).sample(shots, separate_observables=True)
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    decoder = _CapturingDecoder()
    for shot in range(shots):
        RunSpec(ops=[op], d=d, rounds_policy=FixedRounds(rounds),
                device=RecordedStimDevice(dets, obs, shot), decoder=decoder, seed=shot).build()
    calls = decoder.calls
    weight_agree = logical_agree = satisfied = disagree_with_different_weight = 0
    for faults, syndrome, py_weight, selected in calls:
        fb_weight, fb_logical, ok = fusion_blossom_solve(faults, syndrome)
        weights = matching_weights(faults.priors)
        py_scaled = sum(max(1, int(round(w * WEIGHT_SCALE))) + (max(1, int(round(w * WEIGHT_SCALE))) % 2)
                        for w, s in zip(weights, selected) if s) / WEIGHT_SCALE
        observables = np.asarray(faults.observables.todense()
                                 if hasattr(faults.observables, "todense") else faults.observables)
        py_logical = (observables.astype(np.uint8) @ selected.astype(np.uint8)) % 2
        # PyMatching optimizes float weights, Fusion Blossom the even-integer rounding of them:
        # optima may differ by the rounding of the selected edges, never more.
        tolerance = 2.0 / WEIGHT_SCALE * (int(selected.sum()) + 1)
        same_weight = abs(py_scaled - fb_weight) <= tolerance
        same_logical = bool(np.array_equal(py_logical, fb_logical))
        weight_agree += same_weight
        logical_agree += same_logical
        disagree_with_different_weight += (not same_logical) and (not same_weight)
        satisfied += ok
    lines = ["# Gate 3: decsim inner MWPM (PyMatching) vs Fusion Blossom on the same window graphs", "",
             f"Problem: rotated_memory_z d={d}, {rounds} rounds, p={p}, {shots} recorded shots through the loop; "
             f"{len(calls)} window decodes captured; identical graph (decsim graphlike faults, log-odds weights "
             f"scaled by {WEIGHT_SCALE} to even integers) handed to both solvers.", "",
             "| check | agree |", "|---|---|",
             f"| minimum matching weight equal | {weight_agree}/{len(calls)} |",
             f"| Fusion Blossom correction satisfies the syndrome | {satisfied}/{len(calls)} |",
             f"| predicted logical class equal | {logical_agree}/{len(calls)} |",
             f"| logical class differs AND weight differs (a real defect would show here) | {disagree_with_different_weight}/{len(calls)} |", "",
             "Equal-weight ties can select different physical corrections and, when parallel faults carry "
             "different observables, different logical classes; both are valid minimum-weight matchings. "
             "Only a logical disagreement at unequal weight would indicate a solver or graph defect."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
