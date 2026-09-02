"""RecordedStimDevice replays recorded raw measurements through the loop."""

import pytest

from decsim.qpu.stim_device import RecordedStimDevice
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.message import Operation
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.qpu.round_policies import FixedRounds
from decsim.run_spec import RunSpec


@pytest.fixture(scope="module")
def recorded():
    stim = pytest.importorskip("stim")
    pymatching = pytest.importorskip("pymatching")
    p = 0.01
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=9, distance=3,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    measurements = circuit.compile_sampler(seed=5).sample(12)
    dets, obs = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    return circuit, measurements, dets, obs, matching


def test_replayed_shot_forms_the_recorded_truth_and_decodes_the_recorded_bits(recorded):
    circuit, measurements, dets, obs, matching = recorded
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    agree = 0
    for shot in range(len(dets)):
        result = RunSpec(ops=[op], d=3, rounds_policy=FixedRounds(9),
                         device=RecordedStimDevice(measurements, shot),
                         decoder=PyMatchingDecoder(PresetLatencyDecoder(0.028)),
                         seed=shot).build().result.operation_results[0]
        assert tuple(int(b) for b in obs[shot]) == tuple(result.observable_truth)
        agree += tuple(int(b) for b in matching.decode(dets[shot])) == result.logical_observables
    assert agree >= len(dets) - 1        # windowed vs whole-shot may differ rarely


def test_detector_rounds_from_concatenated_coordinates():
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit("""
        DETECTOR(1, 1, 0) rec[-1]
        DETECTOR(1, 1, 1, 1, 1, 0) rec[-1]
        DETECTOR(2, 2, 3, 2, 2, 2) rec[-1]
        DETECTOR(2, 2, 3, 3, 3, 3, 4, 4, 2) rec[-1]
    """)
    rounds = RecordedStimDevice.detector_rounds_from_coordinates(circuit, 3)
    assert rounds == {0: 1, 1: 2, 2: 3, 3: 3}


def test_sliding_windows_match_qldpc_shot_for_shot(recorded):
    """decsim's sliding windows and qLDPC's SlidingWindowDecoder are two
    implementations of the overlapping-recovery rule (Dennis et al.
    quant-ph/0110143; Skoric et al. 2209.08552). Fed the same shots, the
    same geometry (window = commit + buffer, stride = commit), the same round
    grouping and PyMatching with parallel faults merged as independent
    errors, they must predict identically on every shot."""
    qldpc_decoders = pytest.importorskip("qldpc.decoders")
    import numpy as np
    from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
    circuit, measurements, dets, obs, _ = recorded
    rounds, distance = 9, 3
    round_of_detector = resolve_detector_rounds(circuit, None, rounds)
    reference = qldpc_decoders.SlidingWindowDecoder(
        window_size=2 * distance, stride=distance,
        detector_to_time=lambda detector: int(round_of_detector[detector]),
        decompose_errors=True, with_MWPM=True, merge_strategy="independent",
    ).compile_decoder_for_dem(circuit.detector_error_model(decompose_errors=True))
    reference_predictions = reference.decode_shots(dets.astype(np.uint8))
    op = Operation(id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit)
    for shot in range(len(dets)):
        result = RunSpec(ops=[op], d=distance, rounds_policy=FixedRounds(rounds),
                         device=RecordedStimDevice(measurements, shot),
                         decoder=PyMatchingDecoder(PresetLatencyDecoder(0.028)),
                         seed=shot).build().result.operation_results[0]
        assert result.logical_observables == tuple(int(b) for b in reference_predictions[shot])
