from dataclasses import asdict
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest
import stim

from experiments.bb import build_bb72_memory_z
from decsim.bposd_decoder import bposd_window_decoder
from decsim.detector_error_model import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    decode_windowed,
)
from experiments.decoding import OfflineBatchDecoder
from experiments.harness import sample_batch_sha256
from experiments.run_bb import (
    _scientific_configuration,
    _sliding_window_entries,
    run_bb_configuration,
)


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="QUITS requires Python 3.10 or newer"
)
REPOSITORY = Path(__file__).parents[2]


def _configuration(**changes):
    configuration = {
        "physical_error_rate": 0.003,
        "syndrome_rounds": 10,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 7,
        "batch_shots": 4,
        "seed": 11,
        "workers": 1,
    }
    configuration.update(changes)
    return configuration


@pytest.mark.parametrize("changes", [
    {"code": "bb90-8-10"},
    {"basis": "X"},
    {"noise_model": "ionic"},
    {"decoder": "uf"},
    {"unknown_field": 1},
])
def test_bb_runner_rejects_unsupported_apparent_variation(changes):
    with pytest.raises(ValueError):
        _scientific_configuration(_configuration(**changes))


def test_bb_runner_accepts_seed_zero():
    assert _scientific_configuration(_configuration(seed=0))["seed"] == 0


def test_bb72_provider_matches_the_frozen_quits_circuit():
    circuit, detector_rounds, detector_layer_count = build_bb72_memory_z(
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


def test_bb72_provider_derives_detector_layers_from_requested_rounds():
    circuit, detector_rounds, detector_layer_count = build_bb72_memory_z(
        physical_error_rate=0.003,
        syndrome_round_count=2,
    )

    assert circuit.num_detectors == 144
    assert detector_layer_count == 4
    assert set(detector_rounds) == set(range(144))
    assert {round_index: tuple(detector_rounds.values()).count(round_index)
            for round_index in range(1, 5)} == {1: 36, 2: 36, 3: 36, 4: 36}


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

    circuit, detector_rounds, detector_layer_count = build_bb72_memory_z(
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
