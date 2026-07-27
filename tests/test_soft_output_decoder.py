"""The data-path decoder emits a REAL soft output, not the 0/1 Bernoulli flag.

`SoftOutputDecoder` wraps any base decoder and attaches the real complementary
gap to `DecodeResult.soft_output`, computed on the SAME window decode graph the
hard decoder uses. Validated in naive (single-window) mode, where the window
carries the observable (Toshio et al. 2510.25222; see decsim.soft_output).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.message import Operation
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.schemes import NaiveOnlineScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import FixedRounds
from decsim.soft_output import SoftOutputDecoder, ComplementaryGapMetric
from decsim.run_spec import RunSpec, simulate


class _ZeroLatency:
    def latency(self, job):
        return 1


def _circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _naive_shot(circuit, device, decoder, d, rounds, *, seed):
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(rounds),
              code=SurfaceCodeModel(d=d),
              scheme=NaiveOnlineScheme(),
              device=device,
              decoder=decoder,
              seed=seed,
          ), verbose=False)
    return res["cluster"].op_results[1]


class _Capturing(SoftOutputDecoder):
    """Capture the soft output of the last decoded window for assertions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_soft = None

    def decode(self, job):
        result = super().decode(job)
        if result.soft_output is not None:
            self.last_soft = result.soft_output
        return result


def test_engine_emits_real_soft_output_not_flag():
    """Through the real engine, soft_output is a continuous gap, never the {0,1} flag."""
    d, rounds, p = 3, 6, 5e-3
    circuit = _circuit(d, rounds, p)
    decoder = _Capturing(PyMatchingDecoder(_ZeroLatency()), ComplementaryGapMetric)
    softs = []
    for shot in range(40):
        _naive_shot(
            circuit,
            StimDevice(),
            decoder,
            d,
            rounds,
            seed=4 + shot,
        )
        assert decoder.last_soft is not None
        assert decoder.last_soft >= 0.0
        softs.append(decoder.last_soft)
    assert set(np.round(softs, 6)) - {0.0, 1.0}          # not just the old path flag
    assert len(np.unique(np.round(softs, 6))) > 5


def test_soft_output_decoder_preserves_hard_decode():
    """Attaching soft output must not change the committed logical value."""
    d, rounds, p = 3, 6, 5e-3
    circuit = _circuit(d, rounds, p)
    plain = PyMatchingDecoder(_ZeroLatency())
    soft = SoftOutputDecoder(PyMatchingDecoder(_ZeroLatency()), ComplementaryGapMetric)
    for shot in range(30):
        a = _naive_shot(
            circuit,
            StimDevice(),
            plain,
            d,
            rounds,
            seed=8 + shot,
        )
        b = _naive_shot(
            circuit,
            StimDevice(),
            soft,
            d,
            rounds,
            seed=8 + shot,
        )
        assert a == b


def test_engine_soft_output_is_error_predictive():
    """corr(real soft output, logical error) < 0 through the engine (low g = error-prone)."""
    d, rounds, p = 3, 6, 8e-3
    circuit = _circuit(d, rounds, p)
    decoder = _Capturing(PyMatchingDecoder(_ZeroLatency()), ComplementaryGapMetric)
    gaps, errs = [], []
    shots = 600
    for shot in range(shots):
        device = StimDevice()
        decoded = _naive_shot(
            circuit,
            device,
            decoder,
            d,
            rounds,
            seed=15 + shot,
        )
        truth = int(device._truth[1][0])
        gaps.append(decoder.last_soft)
        errs.append(decoded[0] ^ truth)
    gaps, errs = np.asarray(gaps), np.asarray(errs)
    assert errs.sum() > 0
    assert np.corrcoef(gaps, errs)[0, 1] < 0
