"""Full-stack real decoding tests.

Stim syndromes flow through the simulator and must match the offline window
reference. Paper contract: docs/PAPER_MODEL_MAP.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.message import Operation
from decsim.controllers import ModularController, LinkModel
from decsim.frontends.circuit import CircuitFrontend
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder, matching_window_decoder
from decsim.detector_error_model import (build_window_error_models,
                                             decode_windowed)
from decsim.schemes import SlidingWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate


D, ROUNDS, P = 3, 12, 0.003


class _ZeroLatency:
    def latency(self, job):
        return 1


def _single_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    assert len(payloads) == 1
    return payloads[0]


def _zero_link_controller(engine):
    return ModularController(engine, links=LinkModel(qc=0, cd=0, dd=0, do=0, oc=0, cq=0), log_syndromes=False)


def _circuit():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=ROUNDS,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)


def _run_engine_shot(circuit, device, decoder):
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              d=D,
              rounds_policy=FixedRounds(ROUNDS),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              device=device,
              decoder=decoder,
          ), verbose=False)
    return res["cluster"].op_results[1]


def test_stim_device_round_alignment():
    """Round 1 carries layer t=0 (the init detectors -- the bits the old off-by-one
    dropped); the last round carries its own layer AND the folded closing layer."""
    circuit = _circuit()
    device = StimDevice(seed=3)
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    device.begin_operation(op)
    coords = circuit.get_detector_coordinates()
    layer = {}
    for det, c in coords.items():
        layer.setdefault(int(c[-1]), []).append(det)
    r1 = _single_payload(device, op, 1)
    assert len(r1.bits) == len(layer[0])
    last = _single_payload(device, op, ROUNDS)
    assert len(last.bits) == len(layer[ROUNDS - 1]) + len(layer[ROUNDS])
    # nothing beyond the chip's rounds, nothing at round 0
    assert len(_single_payload(device, op, ROUNDS + 1).bits) == 0
    assert len(_single_payload(device, op, 0).bits) == 0
    # every detector bit is emitted exactly once across rounds 1..R
    total = sum(len(_single_payload(device, op, r).bits)
                for r in range(1, ROUNDS + 1))
    assert total == circuit.num_detectors


def test_same_seed_double_run_is_bit_identical():
    """Seed reproducibility on the FULL physics path (validation-matrix row
    3): two complete engine runs with the same StimDevice seed must produce
    identical per-shot syndromes, identical decoded logical values, and
    identical completion times -- not just the same aggregate LER. A hidden
    ordering/caching nondeterminism anywhere chip -> controller -> windows ->
    decoder -> orchestrator would break the exact match."""
    circuit = _circuit()

    def one_run():
        results = []
        for shot in range(8):
            device = StimDevice(seed=100 + shot)
            op = Operation(id=1, name="memory", qubits=(0,), clifford=True,
                           circuit=circuit)
            res = simulate(RunSpec(
                      ops=[op],
                      num_units=4,
                      d=D,
                      rounds_policy=FixedRounds(ROUNDS),
                      code=SurfaceCodeModel(d=D),
                      scheme=SlidingWindowScheme(),
                      device=device,
                      decoder=PyMatchingDecoder(_ZeroLatency()),
                  ), verbose=False)
            results.append((int(res["cluster"].op_results[1]),
                            res["chip_done"], res["fully_done"],
                            device._dets[1].tobytes()))
        return results

    assert one_run() == one_run()


def test_engine_matches_offline_reference_and_global_exactly():
    """THE R3 gate: per shot, the engine's decoded logical value equals the offline
    decode_windowed reference (exact -- same construction end to end) and the global
    whole-history decode; LERs are equal by implication. Defects must really flow."""
    circuit = _circuit()
    # offline reference built with the engine's own folded round convention
    coords = circuit.get_detector_coordinates()
    folded = {det: min(int(c[-1]) + 1, ROUNDS) for det, c in coords.items()}
    plan = [(lo, hi, min(b, ROUNDS)) for lo, hi, b in
            SlidingWindowScheme().plan_windows(0, ROUNDS, SurfaceCodeModel(d=D))]
    ref_models = build_window_error_models(circuit, plan, detector_rounds=folded)
    ref_inner = matching_window_decoder()
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))

    defect_bits = 0

    class CountingDecoder(PyMatchingDecoder):
        def decode(self, job):
            nonlocal defect_bits
            r = super().decode(job)
            if r.boundary_defects:
                defect_bits += sum(sum(m) for m in r.boundary_defects.values())
            return r

    device = StimDevice(seed=11)
    shots = 150
    for s in range(shots):
        pred_engine = _run_engine_shot(circuit, device, CountingDecoder(_ZeroLatency()))
        shot = device._dets[1]
        pred_offline = int(decode_windowed(ref_models, shot, ref_inner)[0])
        pred_global = int(global_m.decode(shot)[0])
        assert pred_engine == pred_offline, f"shot {s}: engine != offline reference"
        assert pred_engine == pred_global, f"shot {s}: engine != global decode"
    assert defect_bits > 0, "no artificial defects ever crossed a commit boundary"


def test_engine_bposd_matches_offline_reference():
    """BP-OSD runs THROUGH the DES under the sliding scheme and equals the offline windowed
    BP-OSD reference per shot -- the runtime BPOSDDecoder drops into the same scheme machinery
    as PyMatchingDecoder, with no extra cluster wiring (it decodes the window check matrix)."""
    pytest.importorskip("ldpc")
    from decsim.bposd_decoder import BPOSDDecoder, bposd_window_decoder

    circuit = _circuit()
    coords = circuit.get_detector_coordinates()
    folded = {det: min(int(c[-1]) + 1, ROUNDS) for det, c in coords.items()}
    plan = [(lo, hi, min(b, ROUNDS)) for lo, hi, b in
            SlidingWindowScheme().plan_windows(0, ROUNDS, SurfaceCodeModel(d=D))]
    ref_models = build_window_error_models(circuit, plan, detector_rounds=folded)
    ref_inner = bposd_window_decoder()

    device = StimDevice(seed=29)
    for s in range(15):
        pred_engine = _run_engine_shot(circuit, device, BPOSDDecoder(_ZeroLatency()))
        pred_offline = int(decode_windowed(ref_models, device._dets[1], ref_inner)[0])
        assert pred_engine == pred_offline, f"shot {s}: engine != offline BP-OSD reference"


def test_blocked_successor_waits_for_real_pymatching_result():
    """A feedback-blocked successor is released only after the real PyMatching result
    for the blocking op has been computed and delivered through the orchestrator.

    The operations are T-labelled to exercise feedback, but each carries a memory
    stim circuit so the decode path is the same real StimDevice -> PyMatchingDecoder
    path certified above.
    """
    circuit = _circuit()
    window_count = len(SlidingWindowScheme().plan_windows(0, ROUNDS, SurfaceCodeModel(d=D)))

    class RecordingDecoder(PyMatchingDecoder):
        def __init__(self):
            super().__init__(_ZeroLatency())
            self.seen = []
            self.accumulated = {}

        def decode(self, job):
            r = super().decode(job)
            self.seen.append((job.op_id, job.window_id))
            if r.logical_value is not None:
                self.accumulated[job.op_id] = (
                    self.accumulated.get(job.op_id, 0) ^ int(r.logical_value))
            return r

    decoder = RecordingDecoder()

    ops = CircuitFrontend([
        Operation(0, "T0(memory)", (0,), clifford=False, blocked_by=None,
                  consumes_magic_state=False, circuit=circuit),
        Operation(1, "T1(memory)", (0,), clifford=False, blocked_by=0,
                  consumes_magic_state=False, circuit=circuit),
    ]).build()
    device = StimDevice(seed=23)
    res = simulate(RunSpec(
              ops=ops,
              num_units=4,
              d=D,
              rounds_policy=FixedRounds(ROUNDS),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              device=device,
              decoder=decoder,
              make_controller=_zero_link_controller,
          ), verbose=False)

    # every window was decoded before op0's result integrated, and the
    # integrated value is the XOR-accumulated per-window logical value
    assert set(decoder.seen) >= {(0, k) for k in range(window_count)}
    assert res["cluster"].op_results[0] == decoder.accumulated[0]

    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    assert res["cluster"].op_results[0] == int(global_m.decode(device._dets[0])[0])
    assert res["chip"].decode_release_time[1] == res["cluster"].windows[(0, window_count - 1)].t_done
    assert res["chip"].decode_release_time[1] <= res["chip"].body_done_time[1]


def test_timing_only_ops_still_run():
    """An op without a circuit keeps dem=None and decodes as a timing-only job.

    the real-decoding wiring must not break the timing pipeline."""
    op = Operation(id=1, name="timing", qubits=(0,), clifford=True)
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              d=D,
              rounds_policy=FixedRounds(ROUNDS),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              decoder=PyMatchingDecoder(_ZeroLatency()),
          ), verbose=False)
    assert res["cluster"].op_results == {}        # no logical value, but it completed
    assert len(res["cluster"].committed_windows) == res["cluster"].total_windows
