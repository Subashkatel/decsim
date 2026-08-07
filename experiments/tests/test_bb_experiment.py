from dataclasses import asdict
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest
import stim

from experiments.bb import build_quits11_bb_memory
from decsim.bposd_decoder import bposd_window_decoder
from decsim.detector_error_model import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    decode_windowed,
)
from experiments.decoding import OfflineBatchDecoder
from experiments.harness import SamplePlan, sample_batch_sha256
from experiments.run_bb import (
    _run_parts,
    _sliding_window_entries,
    run_bb_configuration,
)


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="QUITS requires Python 3.10 or newer"
)
REPOSITORY = Path(__file__).parents[2]
BB_DEFINITIONS = {
    "bb-l6m6-a-x3_y1_y2-b-x1_x2_y3": (6, 6, (3,), (1, 2), (1, 2), (3,), 12),
    "bb-l15m3-a-x9_y1_y2-b-1_x2_x7": (15, 3, (9,), (1, 2), (0, 2, 7), (), 8),
    "bb-l3m21-a-1_y2_y10-b-x1_x2_y3": (3, 21, (), (0, 2, 10), (1, 2), (3,), 8),
    "bb-l12m6-a-x3_y1_y2-b-x1_x2_y3": (12, 6, (3,), (1, 2), (1, 2), (3,), 12),
    "bb-l3m27-a-1_y10_y14-b-x1_x2_y12": (3, 27, (), (0, 10, 14), (1, 2), (12,), 8),
    "bb-l12m12-a-x3_y2_y7-b-x1_x2_y3": (12, 12, (3,), (2, 7), (1, 2), (3,), 12),
    "bb-l28m14-a-x26_y6_y8-b-x9_x20_y7": (28, 14, (26,), (6, 8), (9, 20), (7,), 24),
}
BB72 = next(iter(BB_DEFINITIONS))
BB90 = next(definition for definition in BB_DEFINITIONS if "l15m3" in definition)


def _binary_rank(matrix):
    matrix = matrix.copy()
    pivot_row = 0
    for column in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[pivot_row:, column])
        if not len(candidates):
            continue
        selected_row = pivot_row + candidates[0]
        matrix[[pivot_row, selected_row]] = matrix[[selected_row, pivot_row]]
        for row in np.flatnonzero(matrix[:, column]):
            if row != pivot_row:
                matrix[row] ^= matrix[pivot_row]
        pivot_row += 1
    return pivot_row


def _paper_check_matrices(definition):
    l, m, a_x, a_y, b_x, b_y, _ = definition
    identity_l = np.eye(l, dtype=np.uint8)
    identity_m = np.eye(m, dtype=np.uint8)
    x_shift = np.kron(np.roll(identity_l, 1, axis=1), identity_m)
    y_shift = np.kron(identity_l, np.roll(identity_m, 1, axis=1))

    def polynomial(x_powers, y_powers):
        terms = [np.linalg.matrix_power(x_shift, power) for power in x_powers]
        terms += [np.linalg.matrix_power(y_shift, power) for power in y_powers]
        return np.bitwise_xor.reduce(terms)

    matrix_a = polynomial(a_x, a_y)
    matrix_b = polynomial(b_x, b_y)
    return np.hstack((matrix_a, matrix_b)), np.hstack((matrix_b.T, matrix_a.T))


def _configuration(**changes):
    configuration = {
        "definition_id": BB90,
        "basis": "X",
        "noise_profile": "reference-ionic-ratios",
        "circuit_model": "quits-1.1-custom",
        "physical_error_rate": 0.003,
        "syndrome_rounds": 10,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 7,
        "batch_shots": 4,
        "seed": 0,
        "workers": 1,
    }
    configuration.update(changes)
    return configuration


@pytest.mark.parametrize("changes", [
    {"definition_id": "bb126"},
    {"definition_id": "bb-l7m9-a-1_x1y1_x2y4-b-1_x6y4_x6y5"},
    {"basis": "x"},
    {"noise_profile": "standard"},
    {"circuit_model": "bravyi-table5"},
    {"decoder": "uf"},
    {"unknown_field": 1},
    {"physical_error_rate": True},
    {"physical_error_rate": float("nan")},
    {"syndrome_rounds": 0},
    {"syndrome_rounds": True},
])
def test_bb_runner_rejects_invalid_configuration_before_output(tmp_path, changes):
    output = tmp_path / "output"
    with pytest.raises((TypeError, ValueError)):
        run_bb_configuration(_configuration(**changes), output)
    assert not output.exists()


def test_bb_provider_rejects_the_wrong_quits_distribution(monkeypatch):
    monkeypatch.setattr("experiments.bb.package_version", lambda _: "9.9.9")

    with pytest.raises(RuntimeError, match="installed quits version 9.9.9"):
        build_quits11_bb_memory(
            definition_id=BB72,
            basis="Z",
            noise_profile="equal-rate",
            physical_error_rate=0.003,
            syndrome_round_count=2,
        )


@pytest.mark.parametrize("workers", [True, False, 0, -1, 1.5, "2"])
def test_bb_runner_rejects_invalid_worker_counts_before_output(
    tmp_path, workers
):
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="workers must be a positive integer"):
        run_bb_configuration(_configuration(workers=workers), output)

    assert not output.exists()


def test_bb72_provider_matches_the_frozen_quits_circuit():
    circuit, detector_rounds, detector_layer_count = build_quits11_bb_memory(
        definition_id=BB72,
        basis="Z",
        noise_profile="equal-rate",
        physical_error_rate=0.003,
        syndrome_round_count=10,
    )
    frozen = stim.Circuit.from_file(
        REPOSITORY / "tests/data/bb72_12_6_p003_r10.stim"
    )

    assert circuit == frozen
    assert hashlib.sha256(str(circuit).encode()).hexdigest() == (
        "6feb7de9b6f258fe092c20829920e67dcf38e66ab922a648126ba47d8fbcfc87"
    )
    assert (circuit.num_qubits, circuit.num_detectors,
            circuit.num_observables, circuit.num_ticks) == (144, 432, 12, 111)
    assert detector_layer_count == 12
    assert detector_rounds == {
        detector: detector // 36 + 1 for detector in range(432)
    }


@pytest.mark.parametrize("definition_id, definition", BB_DEFINITIONS.items())
def test_bb_catalog_matches_paper_definitions_and_quits(monkeypatch, definition_id, definition):
    import quits

    constructor_calls = []
    real_bb_code = quits.BbCode

    def recording_bb_code(**arguments):
        constructor_calls.append(arguments)
        return real_bb_code(**arguments)

    monkeypatch.setattr(quits, "BbCode", recording_bb_code)
    circuit, detector_rounds, layer_count = build_quits11_bb_memory(
        definition_id=definition_id,
        basis="X",
        noise_profile="equal-rate",
        physical_error_rate=0,
        syndrome_round_count=1,
    )
    l, m, a_x, a_y, b_x, b_y, logical_count = definition
    assert constructor_calls == [{
        "l": l, "m": m, "A_x_pows": list(a_x), "A_y_pows": list(a_y),
        "B_x_pows": list(b_x), "B_y_pows": list(b_y),
    }]
    check_x, check_z = _paper_check_matrices(definition)
    assert not np.any(check_x @ check_z.T % 2)
    assert 2 * l * m - _binary_rank(check_x) - _binary_rank(check_z) == logical_count
    assert (circuit.num_qubits, circuit.num_detectors,
            circuit.num_observables, circuit.num_ticks) == (
                4 * l * m, 3 * l * m, logical_count, 21)
    assert layer_count == 3
    assert set(detector_rounds.values()) == {1, 2, 3}


@pytest.mark.parametrize("profile, ratios", [
    ("equal-rate", (1, 1, 1, 1)), ("reference-ionic-ratios", (0.01, 0.1, 1, 0.1)),
])
def test_bb_noise_profiles_pass_exact_quits_rates(monkeypatch, profile, ratios):
    import quits

    error_model_calls = []
    real_error_model = quits.ErrorModel

    def recording_error_model(**arguments):
        error_model_calls.append(arguments)
        return real_error_model(**arguments)

    monkeypatch.setattr(quits, "ErrorModel", recording_error_model)
    circuit, _, _ = build_quits11_bb_memory(
        definition_id=BB72, basis="Z", noise_profile=profile,
        physical_error_rate=0.2, syndrome_round_count=1,
    )
    assert tuple(error_model_calls[0].values()) == tuple(
        0.2 * ratio for ratio in ratios
    )
    def has_noise(instruction_name, expected_rate):
        return any(
            instruction.name == instruction_name
            and instruction.gate_args_copy()[0] == pytest.approx(expected_rate)
            for instruction in circuit.flattened()
        )

    assert has_noise("DEPOLARIZE1", 0.2 * ratios[0])
    assert has_noise("DEPOLARIZE1", 0.2 * ratios[1])
    assert has_noise("DEPOLARIZE2", 0.2 * ratios[2])
    assert has_noise("X_ERROR", 0.2 * ratios[3])


def test_sample_identity_tracks_physics_but_not_decode_execution(monkeypatch):
    trajectories = []
    original_create = SamplePlan.create

    class RecordingSamplePlan:
        @classmethod
        def create(cls, experiment_id, experiment_seed, trajectory):
            trajectories.append(trajectory)
            return original_create(experiment_id, experiment_seed, trajectory)

    def fake_builder(**selection):
        return json.dumps(selection, sort_keys=True), {}, selection["syndrome_round_count"] + 2

    monkeypatch.setattr("experiments.run_bb.SamplePlan", RecordingSamplePlan)
    monkeypatch.setattr("experiments.run_bb.build_quits11_bb_memory", fake_builder)

    def run_parts(**changes):
        return _run_parts(_configuration(**changes))

    baseline_configuration, baseline_experiment, baseline, _ = run_parts()
    baseline_configuration_id = baseline_experiment.config_sha256(baseline_configuration)
    assert set(trajectories[0]) == {
        "definition_id", "circuit_model", "basis", "noise_profile",
        "physical_error_rate", "syndrome_rounds", "circuit_sha256",
        "shots", "batch_shots",
    }
    for changes in (
        {"definition_id": BB72}, {"basis": "Z"},
        {"noise_profile": "equal-rate"},
        {"physical_error_rate": 0.004}, {"syndrome_rounds": 9},
    ):
        assert run_parts(**changes)[2].sample_set_id != baseline.sample_set_id
    for changes in ({"commit_rounds": 2}, {"buffer_rounds": 2}):
        configuration, experiment, sample_plan, _ = run_parts(**changes)
        assert sample_plan.sample_set_id == baseline.sample_set_id
        assert experiment.config_sha256(configuration) != baseline_configuration_id
    configuration, experiment, sample_plan, _ = run_parts(workers=2)
    assert sample_plan.sample_set_id == baseline.sample_set_id
    assert experiment.config_sha256(configuration) == baseline_configuration_id


def test_bb_runner_uses_the_normal_four_window_plan_and_exact_batches(tmp_path):
    configuration = _configuration()
    assert _sliding_window_entries(configuration, 12) == (
        (1, 3, 6), (4, 6, 9), (7, 9, 12), (10, 12, 15)
    )
    result = run_bb_configuration(configuration, tmp_path)

    assert result.attempted_shots == 7
    assert result.accepted_shots == 7
    chunk_paths = sorted(tmp_path.glob("**/chunks/**/*.csv"))
    assert [path.name for path in chunk_paths] == ["0.csv", "1.csv"]


def test_bb_runner_resume_and_worker_count_preserve_chunk_bytes(tmp_path):
    one_worker = tmp_path / "one-worker"
    two_workers = tmp_path / "two-workers"
    first = run_bb_configuration(_configuration(workers=1), one_worker)
    first_chunks = {
        path.name: path.read_bytes() for path in sorted(one_worker.glob("**/*.csv"))
    }

    resumed = run_bb_configuration(_configuration(workers=1), one_worker)
    parallel = run_bb_configuration(_configuration(workers=2), two_workers)
    parallel_chunks = {
        path.name: path.read_bytes() for path in sorted(two_workers.glob("**/*.csv"))
    }

    assert resumed == first == parallel
    assert first_chunks == parallel_chunks


def test_bb_matched_windows_agree_with_quits_on_the_same_400_shots():
    from quits import BbCode, sliding_window_bposd_circuit_mem

    circuit, detector_rounds, detector_layer_count = build_quits11_bb_memory(
        definition_id=BB72,
        basis="Z",
        noise_profile="equal-rate",
        physical_error_rate=0.003,
        syndrome_round_count=10,
    )
    decoder = OfflineBatchDecoder.prepare(
        circuit,
        ((1, 3, 6), (4, 6, 9), (7, 12, 12)),
        bposd_window_decoder(),
        round_count=detector_layer_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
        fault_representation=FaultRepresentation.PHYSICAL,
    )
    archive = np.load(REPOSITORY / "tests/data/bb72_12_6_p003_r10.shots.npz")
    detectors, truth = archive["dets"], archive["obs"]
    digest_before = sample_batch_sha256(detectors, truth)

    decsim_predictions = np.asarray([
        decode_windowed(
            decoder.window_models,
            row,
            decoder.decode_window,
            selected_fault_representation=FaultRepresentation.PHYSICAL,
        )
        for row in detectors
    ])
    code = BbCode(
        l=6,
        m=6,
        A_x_pows=[3],
        A_y_pows=[1, 2],
        B_x_pows=[1, 2],
        B_y_pows=[3],
    )
    quits_predictions = sliding_window_bposd_circuit_mem(
        detectors,
        circuit,
        code.hz,
        code.lz,
        W=6,
        F=3,
        max_iter=2,
        osd_order=0,
        bp_method="product_sum",
        schedule="serial",
        osd_method="osd_cs",
    )

    assert sample_batch_sha256(detectors, truth) == digest_before
    assert decsim_predictions.shape == quits_predictions.shape == (400, 12)
    assert np.array_equal(decsim_predictions, quits_predictions)
    assert np.array_equal(
        np.any(decsim_predictions != truth, axis=1),
        np.any(quits_predictions != truth, axis=1),
    )
    assert int(np.any(decsim_predictions != truth, axis=1).sum()) == 60


def test_bb_cli_matches_the_direct_runner(tmp_path):
    configuration = _configuration()
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    direct = run_bb_configuration(configuration, tmp_path / "direct")
    command = [
        sys.executable,
        "-m",
        "experiments.run_bb",
        "--config",
        str(configuration_path),
        "--output",
        str(tmp_path / "cli"),
    ]

    completed = subprocess.run(
        command, cwd=REPOSITORY, text=True, capture_output=True
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == asdict(direct)
