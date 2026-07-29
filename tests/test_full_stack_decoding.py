"""Full-stack real decoding tests.

Stim syndromes flow through the simulator and must match the offline window
reference. Paper contract: docs/PAPER_MODEL_MAP.md.
"""
import sys, pathlib
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from conftest import fixed_latency_link_config

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.message import Operation
from decsim.controllers import ModularController
from decsim.frontends.circuit import CircuitFrontend
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder, matching_window_decoder
from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    PlacedFaultModel,
    build_window_error_models,
    decode_windowed,
)
from decsim.adapters.window_decode_results import result_from_selected_faults
from decsim.schemes import ParallelWindowScheme, SlidingWindowScheme
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


def _zero_link_controller(engine, links):
    return ModularController(engine, links=links, log_syndromes=False)


def _circuit():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=ROUNDS,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)


def _run_engine_shot(circuit, device, decoder, *, seed):
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(ROUNDS),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              device=device,
              decoder=decoder,
              seed=seed,
          ), verbose=False)
    return res.window_manager.op_results[1]


def test_window_adapter_preserves_every_logical_observable_row():
    obs = np.zeros((12, 3), dtype=np.uint8)
    obs[9, 0] = 1
    obs[10, 1] = 1
    obs[11, 2] = 1
    placed_faults = PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=np.zeros((0, 3), dtype=np.uint8),
        priors=np.zeros(3),
        observables=obs,
        owned=np.ones(3, dtype=bool),
        future_flips={},
        source_fault_ids=(0, 1, 2),
    )
    model = SimpleNamespace(defect_positions={})
    job = SimpleNamespace(op_id=7, window_id=4)

    result = result_from_selected_faults(
        job,
        model,
        placed_faults,
        np.ones(3, dtype=np.uint8),
    )

    assert result.logical_observables == (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    )


def test_stim_device_round_alignment():
    """Round 1 carries layer t=0 (the init detectors -- the bits the old off-by-one
    dropped); the last round carries its own layer AND the folded closing layer."""
    circuit = _circuit()
    device = StimDevice(seed=3)
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    device.begin_operation(op, ROUNDS)
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
            device = StimDevice()
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
                      seed=100 + shot,
                  ), verbose=False)
            results.append((res.window_manager.op_results[1],
                            res.result.chip_done_ticks, res.result.fully_done_ticks,
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
    plan = [
        (window.commit_lo, window.commit_hi, min(window.buffer_hi, ROUNDS))
        for window in SlidingWindowScheme().plan_operation(
            0,
            ROUNDS,
            commit_round_count=D,
            buffer_round_count=D,
        ).windows
    ]
    ref_models = build_window_error_models(
        circuit,
        plan,
        detector_rounds=folded,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
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

    shots = 150
    for s in range(shots):
        device = StimDevice()
        pred_engine = _run_engine_shot(
            circuit,
            device,
            CountingDecoder(_ZeroLatency()),
            seed=11 + s,
        )
        shot = device._dets[1]
        pred_offline = (
            int(
                decode_windowed(
                    ref_models,
                    shot,
                    ref_inner,
                    selected_fault_representation=FaultRepresentation.GRAPHLIKE,
                )[0]
            ),
        )
        pred_global = (int(global_m.decode(shot)[0]),)
        assert pred_engine == pred_offline, f"shot {s}: engine != offline reference"
        assert pred_engine == pred_global, f"shot {s}: engine != global decode"
    assert defect_bits > 0, "no artificial defects ever crossed a commit boundary"


def test_parallel_engine_matches_global_decode_on_real_syndromes():
    """Parallel A/B windows preserve the global logical prediction.

    Geometry and dependency tests separately pin the A/B topology. This gate
    forces noisy detector data and nonzero seam defects through that topology.
    """
    circuit = _circuit()
    global_matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    boundary_defect_count = 0
    nonzero_syndrome_count = 0

    class CountingDecoder(PyMatchingDecoder):
        def decode(self, job):
            nonlocal boundary_defect_count
            result = super().decode(job)
            if result.boundary_defects:
                boundary_defect_count += sum(
                    sum(bits)
                    for bits in result.boundary_defects.values()
                )
            return result

    for shot_index in range(100):
        device = StimDevice()
        operation = Operation(
            id=1,
            name="parallel-memory",
            qubits=(0,),
            clifford=True,
            circuit=circuit,
        )
        result = simulate(
            RunSpec(
                ops=[operation],
                num_units=4,
                rounds_policy=FixedRounds(ROUNDS),
                code=SurfaceCodeModel(d=D),
                scheme=ParallelWindowScheme(),
                device=device,
                decoder=CountingDecoder(_ZeroLatency()),
                seed=20260728 + shot_index,
            ),
            verbose=False,
        )
        syndrome = device._dets[1]
        nonzero_syndrome_count += int(np.any(syndrome))
        assert result.window_manager.op_results[1] == (
            int(global_matching.decode(syndrome)[0]),
        )

    assert nonzero_syndrome_count > 0
    assert boundary_defect_count > 0


def test_engine_bposd_matches_offline_reference():
    """BP-OSD runs THROUGH the DES under the sliding scheme and equals the offline windowed
    BP-OSD reference per shot -- the runtime BPOSDDecoder drops into the same scheme machinery
    as PyMatchingDecoder, with no extra cluster wiring (it decodes the window check matrix)."""
    pytest.importorskip("ldpc")
    from decsim.bposd_decoder import BPOSDDecoder, bposd_window_decoder

    circuit = _circuit()
    coords = circuit.get_detector_coordinates()
    folded = {det: min(int(c[-1]) + 1, ROUNDS) for det, c in coords.items()}
    plan = [
        (window.commit_lo, window.commit_hi, min(window.buffer_hi, ROUNDS))
        for window in SlidingWindowScheme().plan_operation(
            0,
            ROUNDS,
            commit_round_count=D,
            buffer_round_count=D,
        ).windows
    ]
    ref_models = build_window_error_models(
        circuit,
        plan,
        detector_rounds=folded,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    ref_inner = bposd_window_decoder()

    for s in range(15):
        device = StimDevice()
        pred_engine = _run_engine_shot(
            circuit,
            device,
            BPOSDDecoder(_ZeroLatency()),
            seed=29 + s,
        )
        pred_offline = (
            int(
                decode_windowed(
                    ref_models,
                    device._dets[1],
                    ref_inner,
                    selected_fault_representation=FaultRepresentation.PHYSICAL,
                )[0]
            ),
        )
        assert pred_engine == pred_offline, f"shot {s}: engine != offline BP-OSD reference"


def test_blocked_successor_waits_for_real_pymatching_result():
    """A feedback-blocked successor is released only after the real PyMatching result
    for the blocking op has been computed and delivered through the orchestrator.

    The operations are T-labelled to exercise feedback, but each carries a memory
    stim circuit so the decode path is the same real StimDevice -> PyMatchingDecoder
    path certified above.
    """
    circuit = _circuit()
    window_count = len(SlidingWindowScheme().plan_operation(
        0,
        ROUNDS,
        commit_round_count=D,
        buffer_round_count=D,
    ).windows)

    class RecordingDecoder(PyMatchingDecoder):
        def __init__(self):
            super().__init__(_ZeroLatency())
            self.seen = []
            self.accumulated = {}

        def decode(self, job):
            r = super().decode(job)
            self.seen.append((job.op_id, job.window_id))
            if r.logical_observables is not None:
                previous = self.accumulated.get(
                    job.op_id,
                    (0,) * len(r.logical_observables),
                )
                self.accumulated[job.op_id] = tuple(
                    left ^ right
                    for left, right in zip(
                        previous,
                        r.logical_observables,
                    )
                )
            return r

    decoder = RecordingDecoder()

    ops = CircuitFrontend([
        Operation(0, "T0(memory)", (0,), clifford=False, blocked_by=None,
                  consumes_magic_state=False, circuit=circuit),
        Operation(1, "T1(memory)", (0,), clifford=False, blocked_by=0,
                  consumes_magic_state=False, circuit=circuit),
    ]).build()
    device = StimDevice()
    res = simulate(RunSpec(
              ops=ops,
              num_units=4,
              rounds_policy=FixedRounds(ROUNDS),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              device=device,
              decoder=decoder,
              links=fixed_latency_link_config(),
              make_controller=_zero_link_controller,
              seed=23,
          ), verbose=False)

    # every window was decoded before op0's result integrated, and the
    # integrated value is the XOR-accumulated per-window logical value
    assert set(decoder.seen) >= {(0, k) for k in range(window_count)}
    assert res.window_manager.op_results[0] == decoder.accumulated[0]

    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    assert res.window_manager.op_results[0] == (
        int(global_m.decode(device._dets[0])[0]),
    )
    assert res.chip.decode_release_time[1] == res.window_manager.windows[(0, window_count - 1)].t_done
    assert res.chip.decode_release_time[1] <= res.chip.body_done_time[1]


def test_timing_only_ops_still_run():
    """An op without a circuit keeps dem=None and decodes as a timing-only job.

    the real-decoding wiring must not break the timing pipeline."""
    op = Operation(id=1, name="timing", qubits=(0,), clifford=True)
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(ROUNDS),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              decoder=PyMatchingDecoder(_ZeroLatency()),
          ), verbose=False)
    assert res.window_manager.op_results == {}        # no logical value, but it completed
    assert len(res.window_manager.committed_windows) == res.window_manager.total_windows
