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
from decsim.run_spec import RunSpec, simulate


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


def test_high_p_saturates_at_coin_flip_and_windows_survive_dense_syndromes():
    """Far above threshold the decoder must be no better than a coin flip --
    LER saturates at 1/2 (a scale/sign bug would show up as e.g. 0.9 or 0.3).
    The same shots are also the dense-syndrome case (detection density ~0.5,
    merged DEM priors approaching the p=0.5 weight boundary): the windowed
    slicer must decode them and saturate identically, not crash or diverge."""
    from decsim.detector_error_model import build_window_error_models, decode_windowed
    from decsim.mwpm_decoder import matching_window_decoder
    from decsim.codes import SurfaceCodeModel
    from decsim.schemes import SlidingWindowScheme

    c = NoiseModel.circuit_level(0.15).circuit(distance=3, rounds=9)
    dets, obs = c.compile_detector_sampler(seed=5).sample(
        4000, separate_observables=True)
    assert float(dets.mean()) > 0.4               # genuinely dense
    matcher = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True))
    ler = float((matcher.decode_batch(dets)[:, 0] != obs[:, 0]).mean())
    assert 0.45 < ler < 0.55, f"saturation broken: global LER {ler}"

    n_layers = 1 + max(int(cc[-1])
                       for cc in c.get_detector_coordinates().values())
    plan = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0,
            n_layers,
            commit_round_count=3,
            buffer_round_count=3,
        ).windows
    ]
    models = build_window_error_models(c, plan)
    decode = matching_window_decoder()
    global_pred = matcher.decode_batch(dets)[:500, 0]
    windowed_pred = np.array([
        int(decode_windowed(models, dets[k], decode)[0]) for k in range(500)])
    w_fail = windowed_pred != obs[:500, 0]
    w_ler = float(w_fail.mean())
    assert 0.4 < w_ler < 0.6, f"windowed saturation broken: {w_ler}"
    # windowed must still TRACK global at ~50% density, not merely also
    # saturate (fixed seed; measured 13/500 disagreements, only_w/only_g
    # = 7/6 -- balanced, no systematic windowed bias)
    disagree = int((windowed_pred != global_pred).sum())
    assert disagree <= 40, \
        f"windowed drifted from global in dense regime: {disagree}/500"
    g_fail = global_pred != obs[:500, 0]
    only_w = int((w_fail & ~g_fail).sum())
    only_g = int((g_fail & ~w_fail).sum())
    assert abs(only_w - only_g) <= 10, \
        f"systematic windowed bias at saturation: {only_w} vs {only_g}"


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
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(R),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              device=StimDevice(),
              decoder=PyMatchingDecoder(_ZeroLatency()),
              seed=1,
          ), verbose=False)
    assert 1 in res.cluster.op_results        # produced a real logical value
