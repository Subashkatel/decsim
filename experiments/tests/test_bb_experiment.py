from dataclasses import asdict
from importlib.metadata import PackageNotFoundError
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest
import stim

from experiments.bb import build_quits11_bb_memory
from decsim.adapters.window_decode_results import (
    BackendDecodeOutcome,
    BackendDecodeStatus,
    BackendFailureReason,
)
from decsim.bposd_decoder import bposd_window_decoder
from decsim.detector_error_model import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    decode_windowed,
)
from experiments.decoding import (
    OfflineBackendBatchDecoder, OfflineBatchDecoder, _chunk_row,
)
from experiments.harness import Batch, SamplePlan, sample_batch_sha256
from experiments.results import read_chunk_csv
from experiments.run_bb import (
    _BbDecoderFactory, _run_parts,
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


_FAILED_OUTCOMES = (
    (BackendDecodeStatus.LOW_CONFIDENCE,
     BackendFailureReason.SEARCH_LIMIT_EXHAUSTED, "backend_low_confidence"),
    (BackendDecodeStatus.NONCONVERGED,
     BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION, "backend_nonconverged"),
    (BackendDecodeStatus.INVALID_CORRECTION,
     BackendFailureReason.CORRECTION_NOT_BINARY, "backend_invalid_correction"),
    (BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE,
     BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS, "backend_empty_model_unsatisfiable"),
    (BackendDecodeStatus.BACKEND_ERROR,
     BackendFailureReason.UPSTREAM_EXCEPTION, "backend_error"),
)

_BACKEND_COUNTERS = tuple(case[2] for case in _FAILED_OUTCOMES)
_BACKEND_OUTCOMES = _FAILED_OUTCOMES + ((BackendDecodeStatus.SUCCEEDED, None, None),)


@pytest.mark.parametrize("status, reason, counter", _BACKEND_OUTCOMES)
def test_backend_batch_maps_each_terminal_status_exactly(
    monkeypatch, status, reason, counter
):
    class Sampler:
        def sample(self, *, shots, separate_observables):
            return np.zeros((shots, 2), dtype=np.uint8), np.zeros(
                (shots, 1), dtype=np.uint8
            )

    class Circuit:
        def compile_detector_sampler(self, *, seed):
            return Sampler()

    outcome = BackendDecodeOutcome(
        status=status,
        failure_reason=reason,
        physical_correction=() if status is BackendDecodeStatus.SUCCEEDED else None,
        component_correction=None,
        reconstructed_syndrome=() if status is BackendDecodeStatus.SUCCEEDED else None,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
        fault_model_fingerprint="1" * 64,
        decoder_configuration_fingerprint="2" * 64,
    )
    calls = 0

    def failed_walk(models, detectors, decode_window):
        nonlocal calls
        calls += 1
        return type("Walk", (), {
            "window_outcomes": (outcome,),
            "logical_prediction": (0,) if outcome.succeeded else None,
        })()

    monkeypatch.setattr(
        "experiments.decoding.decode_windowed_backend_outcomes", failed_walk
    )
    decoded = OfflineBackendBatchDecoder(
        Circuit(), (object(), object()), object(), FaultRepresentation.PHYSICAL
    ).run(Batch(index=0, first_shot=0, shots=1), sample_seed=5)

    row = _chunk_row(
        decoded, type("Experiment", (), {"experiment_id": "e"})(),
        type("Plan", (), {"sample_set_id": "s"})(), "3" * 64, "4" * 64,
    )
    counters = {name: getattr(row, name) for name in _BACKEND_COUNTERS}
    assert counters == {name: int(name == counter) for name in counters}
    assert (calls, decoded.window_attempts) == (1, 1)
    expected_failure = int(status is not BackendDecodeStatus.SUCCEEDED)
    assert (decoded.attempted_shots, decoded.primary_failures) == (1, expected_failure)
    assert (decoded.accepted_shots, decoded.accepted_logical_failures) == (
        1 - expected_failure, 0
    )


def _minimum_corrections(check, observables, priors, syndrome):
    fault_count = check.shape[1]
    assert fault_count <= 16
    assert check.shape[0] <= 8
    assert check.shape[0] < 65_535
    candidates = np.asarray([
        [(mask >> index) & 1 for index in range(fault_count)]
        for mask in range(1 << fault_count)
    ], dtype=np.uint8)
    feasible = np.all(candidates @ check.T % 2 == syndrome, axis=1)
    costs = candidates @ np.log((1 - priors) / priors)
    minimum = np.min(costs[feasible])
    corrections = candidates[feasible & np.isclose(costs, minimum, rtol=0.0, atol=1e-12)]
    correction_set = {tuple(int(bit) for bit in row) for row in corrections}
    logical_set = {
        tuple(int(bit) for bit in observables @ row % 2)
        for row in corrections
    }
    return correction_set, logical_set


@pytest.mark.parametrize("priors, expected_logicals", [
    (np.array([0.05, 0.2]), {(1,)}),
    (np.array([0.1, 0.1]), {(0,), (1,)}),
])
def test_exact_tesseract_result_belongs_to_complete_minimum_set(
    priors, expected_logicals
):
    import tesseract_decoder

    check = np.array([[1, 1]], dtype=np.uint8)
    observables = np.array([[0, 1]], dtype=np.uint8)
    syndrome = np.array([1], dtype=np.uint8)
    corrections, logicals = _minimum_corrections(
        check, observables, priors, syndrome
    )
    dem = stim.DetectorErrorModel()
    for column, probability in enumerate(priors):
        targets = [stim.DemTarget.relative_detector_id(0)]
        if observables[0, column]:
            targets.append(stim.DemTarget.logical_observable_id(0))
        dem.append(stim.DemInstruction("error", [probability], targets))
    configuration = tesseract_decoder.tesseract.TesseractConfig(
        dem=dem, det_beam=tesseract_decoder.tesseract.INF_DET_BEAM,
        beam_climbing=False, no_revisit_dets=False, verbose=False,
        merge_errors=False, pqlimit=int(np.iinfo(np.uintp).max),
        det_orders=[[0]], det_penalty=0.0, create_visualization=False,
        sparsify_errors=False, sparsify_base_degree=-1,
        sparsify_max_degree=-1, sparsify_reactivate_limit=-1,
    )
    backend = configuration.compile_decoder()
    selected = set(backend.decode_to_errors(syndrome.astype(bool)))
    correction = tuple(int(index in selected) for index in range(2))

    assert correction in corrections
    assert tuple(check @ correction % 2) == tuple(syndrome)
    assert tuple(observables @ correction % 2) in logicals
    assert logicals == expected_logicals


@pytest.mark.parametrize("changes", [
    {"definition_id": "bb126"},
    {"definition_id": "bb-l7m9-a-1_x1y1_x2y4-b-1_x6y4_x6y5"},
    {"basis": "x"},
    {"noise_profile": "standard"},
    {"circuit_model": "bravyi-table5"},
    {"decoder": "uf"},
    {"decoder": "relay_bp"},
    {"decoder": "tesseract", "decoder_seed": True},
    {"decoder": "bposd", "decoder_seed": 0},
    {"decoder": "relay_bp", "decoder_seed": 2**64},
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
    for decoder in ("relay_bp", "tesseract"):
        configuration, experiment, sample_plan, _ = run_parts(
            decoder=decoder, decoder_seed=17
        )
        assert sample_plan.sample_set_id == baseline.sample_set_id
        assert experiment.config_sha256(configuration) != baseline_configuration_id
        assert configuration["decoder_profile"]["name"] == decoder


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


@pytest.mark.parametrize("decoder", ["bposd", "relay_bp", "tesseract"])
def test_bb_runner_resume_and_worker_count_preserve_chunk_bytes(tmp_path, decoder):
    seed = {} if decoder == "bposd" else {"decoder_seed": 17}
    configuration = _configuration(
        decoder=decoder, **seed, syndrome_rounds=1, commit_rounds=1,
        buffer_rounds=1, shots=2, batch_shots=1,
    )
    one_worker = tmp_path / "one-worker"
    two_workers = tmp_path / "two-workers"
    first = run_bb_configuration(dict(configuration, workers=1), one_worker)
    first_chunks = {
        path.name: path.read_bytes() for path in sorted(one_worker.glob("**/*.csv"))
    }

    resumed = run_bb_configuration(dict(configuration, workers=1), one_worker)
    parallel = run_bb_configuration(dict(configuration, workers=2), two_workers)
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


def test_bb_tesseract_factory_disables_backend_output(monkeypatch):
    import tesseract_decoder

    captured = {}
    original = tesseract_decoder.tesseract.TesseractConfig

    def recording_configuration(**arguments):
        captured.update(arguments)
        return original(**arguments)

    monkeypatch.setattr(
        tesseract_decoder.tesseract, "TesseractConfig", recording_configuration
    )
    resolved = _run_parts(_configuration(
        decoder="tesseract", decoder_seed=17, syndrome_rounds=1,
        commit_rounds=1, buffer_rounds=1, shots=1, batch_shots=1,
    ))[0]
    decoder = _BbDecoderFactory(resolved)()
    model = decoder.window_models[0]
    assert model.physical_faults is not None and model.graphlike_faults is None
    decoder.decode_window(model, np.zeros(len(model.detector_ids), dtype=np.uint8))
    assert (captured["verbose"], captured["create_visualization"]) == (False, False)


@pytest.mark.parametrize("decoder", ["relay_bp", "tesseract"])
def test_optional_backend_dependency_fails_before_output(
    monkeypatch, tmp_path, decoder
):
    def missing(_):
        raise PackageNotFoundError

    monkeypatch.setattr("experiments.run_bb.package_version", missing)
    monkeypatch.setattr(
        "experiments.run_bb._run_parts",
        lambda _: ({"decoder": decoder}, None, None, None),
    )
    output = tmp_path / decoder
    with pytest.raises(ImportError, match="requires"):
        run_bb_configuration(
            _configuration(decoder=decoder, decoder_seed=3), output
        )
    assert not output.exists()


def test_real_optional_backends_share_one_bb_sample(tmp_path):
    results = []
    rows = []
    for decoder in ("bposd", "relay_bp", "tesseract"):
        output = tmp_path / decoder
        seed = {} if decoder == "bposd" else {"decoder_seed": 17}
        results.append(run_bb_configuration(_configuration(
            decoder=decoder, **seed, syndrome_rounds=1,
            commit_rounds=1, buffer_rounds=1, shots=1, batch_shots=1,
        ), output))
        rows.append(read_chunk_csv(next(output.glob("**/*.csv"))))

    assert len({result.sample_set_id for result in results}) == 1
    assert len({row.sample_batch_sha256 for row in rows}) == 1
    assert all(result.attempted_shots == 1 for result in results)
    assert len({result.config_id for result in results}) == 3

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
