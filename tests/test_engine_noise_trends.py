"""Full-ENGINE zero-noise and increasing-noise checks (Gate 2b §2).

test_noise_model.py covers these trends on the offline decode path; these
tests drive the complete simulator (chip -> controller -> windows ->
PyMatchingDecoder -> orchestrator) per shot. Fixed seeds make the shot
outcomes deterministic, so the assertions are frozen regressions, not
statistical gambles.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.adapters.stim_device import StimDevice
from decsim.codes import SurfaceCodeModel
from decsim.message import Operation
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.planner import FixedRounds
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate

D, ROUNDS = 3, 12


class _ZeroLatency:
    def latency(self, job):
        return 1


def _circuit(p: float):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=ROUNDS,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _engine_failures(p: float, shots: int, seed0: int) -> int:
    circuit = _circuit(p)
    failures = 0
    for shot in range(shots):
        device = StimDevice(seed=seed0 + shot)
        op = Operation(id=1, name="memory", qubits=(0,), clifford=True,
                       circuit=circuit)
        res = simulate(RunSpec(
                  ops=[op],
                  num_units=4,
                  rounds_policy=FixedRounds(ROUNDS),
                  code=SurfaceCodeModel(d=D),
                  scheme=SlidingWindowScheme(),
                  device=device,
                  decoder=PyMatchingDecoder(_ZeroLatency()),
              ), verbose=False)
        predicted = int(res["cluster"].op_results[1])
        failures += int(predicted != int(device._truth[1][0]))
    return failures


def test_zero_noise_full_engine_never_fails():
    """p=0: no detection events can fire, and the engine's logical result
    must match the sampled truth on EVERY shot."""
    assert _engine_failures(0.0, shots=30, seed0=500) == 0


def test_increasing_noise_full_engine_trend():
    """Logical failures rise with physical noise through the full engine
    (fixed seeds -> deterministic counts; low p chosen well below, high p
    near the sub-threshold/saturation crossover)."""
    low = _engine_failures(0.003, shots=60, seed0=700)
    high = _engine_failures(0.03, shots=60, seed0=700)
    assert low < high, f"failures did not increase: p=.003 -> {low}, p=.03 -> {high}"
    # zero-noise anchor from the other test pins the bottom of the trend
