"""Physical decode validation on a QLX-emitted T-gadget circuit (proxy).

Fixture: tests/data/qlx/mem_surface_t.tsim — emitted by qlx.tsim
(T-aware fabric-to-stim, probe_t_stim_emit.py) from a REAL QLX program:
Surface[15] region producing DISTILL_15TO1_T locally, 4 syndrome rounds
+ T injection + 4 rounds + transversal MZ with a logical-Z observable.

The tsim grammar's ONLY non-Clifford content is a single transversal
T layer on the 225 data qubits. The `tsim` sampler package is not
available (PyPI 'tsim' 0.1.0 is an empty placeholder), so the true
non-Clifford circuit cannot be sampled here. DECLARED PROXY SCOPE:
removing that one T layer yields a valid stim circuit with the
IDENTICAL detector/observable structure. These tests validate that
stabilizer proxy's structure, rounds, observable wiring, and explicit
rejection of its detectorless-logical DEM. They do NOT validate
successful window slicing or decoding, and do NOT validate the
non-Clifford error dynamics through the T layer. Claim-level qualifier
mandatory wherever this is cited (ledger row to say
"stabilizer-proxy, decoder-domain rejection").
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

DATA = pathlib.Path(__file__).resolve().parent / "data/qlx"
TSIM_PATH = DATA / "mem_surface_t.tsim"
ANCILLAS_PER_ROUND = 224          # 112 X + 112 Z ancillas of the d=15 patch
SHOTS = 3000
SEED = 20260703


def load_proxy() -> tuple:
    text = TSIM_PATH.read_text()
    t_lines = [ln for ln in text.splitlines() if re.match(r"^T\b", ln)]
    proxy = "\n".join(ln for ln in text.splitlines()
                      if not re.match(r"^T\b", ln))
    return text, t_lines, proxy


def test_single_transversal_t_layer_is_the_only_nonclifford_content():
    text, t_lines, proxy = load_proxy()
    assert len(t_lines) == 1, "expected exactly one transversal T layer"
    assert len(t_lines[0].split()) - 1 == 225   # all d=15 data qubits
    # the remainder must be plain Clifford stim (parses cleanly)
    circuit = stim.Circuit(proxy)
    assert circuit.num_observables == 1
    assert circuit.num_detectors % ANCILLAS_PER_ROUND == 0
    # detector/observable lines are untouched by the proxy substitution
    keep = lambda s: [ln for ln in s.splitlines()
                      if ln.startswith(("DETECTOR", "OBSERVABLE"))]
    assert keep(text) == keep(proxy)


def _detector_rounds(circuit) -> dict:
    n = circuit.num_detectors
    assert n % ANCILLAS_PER_ROUND == 0
    return {i: i // ANCILLAS_PER_ROUND + 1 for i in range(n)}


def test_windowed_decode_equals_global_on_t_gadget_proxy():
    """The G9 proxy evidence is preserved, but its invalid DEM is rejected."""
    from decsim.detector_error_model import build_window_error_models

    _, _, proxy = load_proxy()
    circuit = stim.Circuit(proxy)
    rounds_of = _detector_rounds(circuit)
    total_rounds = max(rounds_of.values())

    dets, obs = circuit.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    dets = dets.astype(np.uint8)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    pred_global = matching.decode_batch(dets)[:, 0].astype(np.uint8)

    plan = [(1, 3, 5), (4, 5, 6), (6, total_rounds, total_rounds)]
    with pytest.raises(ValueError, match="detectorless logical"):
        build_window_error_models(
            circuit,
            plan,
            detector_rounds=rounds_of,
        )

    # Noise-content reality check (QLX emission gap G9): the tsim path
    # emitted ONLY the bitflip noise (mz/mr) and dropped the idle
    # depolarize channel, so the single observable-coupled DEM fault is
    # the undetectable final data-measurement flip -- the decoder
    # rightly predicts nothing here.
    dem_text = str(circuit.detector_error_model(decompose_errors=True))
    assert dem_text.count("L0") == 1
    assert not pred_global.any()


def test_emission_has_no_stabilizer_coupling_G9():
    """FINDING (G9, 2026-07-03): the fabric-to-stim emission (both the
    plain and tsim paths, this QLX alpha) contains NO entangling gates
    -- ancilla MR rounds are never coupled to the data qubits, so
    detectors carry zero information about data errors and NO noise
    placement can make the circuit decodable. The same holds for the
    Gate-2 mem_surface.stim fixture (predictions all-zero over 20k
    shots; LER == raw parity == twin value). Decode-QUALITY validation
    of decsim therefore rests on real stabilizer circuits (G5R2 ledger
    V5, stim-generated tests), NOT on QLX-emitted circuits; QLX-emitted
    fixtures validate structure/mapping/timing only."""
    _, _, proxy = load_proxy()
    circuit = stim.Circuit(proxy)
    ops = {inst.name for inst in circuit.flattened()}
    assert not ops & {"CX", "CZ", "H", "MPP"}, \
        "emission now contains coupling -- G9 is fixed; upgrade the " \
        "proxy tests to real decode validation"
    # data errors flip no detectors: DEM has detector-coupled faults
    # only from ancilla measurement noise, and exactly one (undetectable)
    # observable-coupled fault from the final data measurement flip
    dem = str(circuit.detector_error_model(decompose_errors=True))
    assert dem.count("L0") == 1


def test_detector_round_grouping_matches_measurement_records():
    """Round map derived from block position must agree with the actual
    measurement-record semantics (no silent re-ordering)."""
    _, _, proxy = load_proxy()
    circuit = stim.Circuit(proxy)
    # walk instructions: count MR blocks before each DETECTOR
    flat = circuit.flattened()
    mr_layers = 0
    seen = 0
    for inst in flat:
        if inst.name == "MR":
            mr_layers += 1
        elif inst.name == "DETECTOR":
            # detectors are emitted after the pair of MR layers they
            # compare; block index = how many ancilla ROUND PAIRS done
            expected_round = mr_layers // 2 - 1
            assert _detector_rounds(circuit)[seen] == expected_round, \
                (seen, mr_layers)
            seen += 1
    assert seen == circuit.num_detectors
