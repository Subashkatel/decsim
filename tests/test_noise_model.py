"""NoiseModel (decsim.stimcircuits): physical noise carried on the circuit, with the
project's conventions written down once.

Acceptance (issue #1): the config-driven noise actually drives the physics -- detection-event
density and logical error rate both rise monotonically with p; the phenomenological 1.5x
convention is exact; and a NoiseModel circuit decodes through the full real-decoding engine
(StimDevice -> cluster windows -> PyMatching), not just standalone.

Requires stim + pymatching (skipped where unavailable)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.stimcircuits import NoiseModel


def _ler_offline(circuit, shots=20000, seed=0):
    """Global pymatching logical error rate of a memory circuit (offline, fast)."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    pred = matcher.decode_batch(dets)
    return float((pred[:, 0] != obs[:, 0]).mean())


def test_noiseless_has_no_detection_events():
    c = NoiseModel().circuit(distance=3, rounds=3)
    assert c.compile_detector_sampler().sample(32).sum() == 0


def test_phenomenological_convention_exact():
    nm = NoiseModel.phenomenological(0.02)
    assert nm.p_data == pytest.approx(0.03)     # the load-bearing 1.5x
    assert nm.p_meas == pytest.approx(0.02)
    assert nm.p_clifford == 0.0 and nm.p_reset == 0.0


def test_circuit_level_sets_all_channels():
    nm = NoiseModel.circuit_level(0.001)
    assert (nm.p_data, nm.p_meas, nm.p_clifford, nm.p_reset) == (0.001,) * 4


def test_detection_density_monotonic_in_p():
    dens = []
    for p in (0.0, 1e-3, 3e-3, 1e-2, 3e-2):
        c = NoiseModel.circuit_level(p).circuit(distance=5, rounds=5)
        dens.append(float(c.compile_detector_sampler().sample(4000).mean()))
    assert dens[0] == 0.0
    assert all(b >= a for a, b in zip(dens, dens[1:])), dens


def test_logical_error_rate_monotonic_in_p():
    lers = [_ler_offline(NoiseModel.circuit_level(p).circuit(distance=5, rounds=5))
            for p in (1e-3, 3e-3, 1e-2)]
    assert all(b >= a for a, b in zip(lers, lers[1:])), lers
    assert lers[0] < lers[-1]                   # genuinely grows across the range


def test_phenomenological_1p5x_is_applied_not_just_stored():
    """The 1.5x is the whole point of phenomenological(): with data depolarization at 1.5p
    the circuit must be measurably NOISIER than the bare-p footgun (data p, meas p) it
    replaces. Compared against a data+meas model only (NOT circuit_level, whose two-qubit
    gate noise would dominate and confound the comparison)."""
    p = 0.01
    bare_p = NoiseModel(p_data=p, p_meas=p)       # the mistake: bare p on the data channel
    d_phenom = float(NoiseModel.phenomenological(p).circuit(distance=5, rounds=5)
                     .compile_detector_sampler().sample(8000).mean())
    d_bare = float(bare_p.circuit(distance=5, rounds=5)
                   .compile_detector_sampler().sample(8000).mean())
    assert d_bare < d_phenom                       # 1.5x data noise is genuinely heavier


def test_runs_through_full_engine():
    """A NoiseModel circuit decodes through the real StimDevice -> cluster -> PyMatching path."""
    from decsim.message import Operation
    from decsim.wiring import build_and_run
    from decsim.adapters.stim_device import StimDevice
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.schemes import SlidingWindowScheme
    from decsim.codes import SurfaceCodeModel
    from decsim.planner import FixedRounds

    class _ZeroLatency:
        def latency(self, job):
            return 1

    D, R = 3, 12
    circ = NoiseModel.circuit_level(0.003).circuit(distance=D, rounds=R)
    op = Operation(id=1, name="mem", qubits=(0,), clifford=True, circuit=circ)
    res = build_and_run(ops=[op], num_units=4, d=D, rounds_policy=FixedRounds(R),
                        code=SurfaceCodeModel(d=D), scheme=SlidingWindowScheme(),
                        device=StimDevice(seed=1),
                        decoder=PyMatchingDecoder(_ZeroLatency()), verbose=False)
    assert 1 in res["cluster"].op_results        # produced a real logical value
