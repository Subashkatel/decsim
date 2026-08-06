from dataclasses import replace
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from decsim.detector_error_model import FaultRepresentation, decode_windowed
import experiments.run_surface as surface_runner

from experiments.results import (
    ChunkResult,
    ShardConflictError,
    canonical_chunk_csv,
    publish_immutable,
    reduce_chunks,
    canonical_surface_plot_csv,
    surface_plot_row,
)
from experiments.plotting import (
    _binomial_interval,
    plot_logical_error_rate,
    write_logical_error_rate_card,
)
from experiments.run_surface import (
    _SurfaceMwpmFactory,
    _circuit,
    _new_output_directory,
    _sliding_window_entries,
    run_surface_configuration,
    write_surface_snapshot,
)
from experiments.harness import (
    Experiment,
    SamplePlan,
    exact_batches,
    offline_batch_seed,
    sample_batch_sha256,
)
from experiments.decoding import (
    DecodedBatch,
    OfflineBatchDecoder,
    load_layered_stim_input,
    read_stored_batch_result,
    run_offline_experiment,
    run_offline_parallel,
)


def _chunk(batch_index=0, first_shot_index=0, requested_shots=3, **changes):
    values = dict(
        schema_version=1,
        experiment_id="experiment",
        experiment_sha256="a" * 64,
        config_id="config",
        sample_set_id="samples",
        sample_batch_sha256=f"{batch_index + 1:064x}",
        batch_index=batch_index,
        first_shot_index=first_shot_index,
        requested_shots=requested_shots,
        attempted_shots=requested_shots,
        primary_failures=1,
        accepted_shots=requested_shots - 1,
        accepted_logical_failures=0,
        backend_low_confidence=1,
        backend_nonconverged=0,
        backend_invalid_correction=0,
        backend_empty_model_unsatisfiable=0,
        backend_error=0,
        window_attempts=requested_shots * 2,
    )
    values.update(changes)
    return ChunkResult(**values)


class _DeterministicOfflineDecoder:
    def run(self, batch, seed):
        return DecodedBatch(
            batch=batch,
            sample_batch_sha256=hashlib.sha256(
                f"{batch.index}:{seed}".encode()
            ).hexdigest(),
            attempted_shots=batch.shots,
            primary_failures=batch.index % 2,
            accepted_shots=batch.shots,
            accepted_logical_failures=batch.index % 2,
            window_attempts=batch.shots * 2,
        )


def _deterministic_offline_decoder():
    return _DeterministicOfflineDecoder()


_REPOSITORY = Path(__file__).parents[2]


def _surface_configuration(**changes):
    configuration = {
        "distance": 3,
        "rounds": 3,
        "physical_error_rate": 0.001,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 3,
        "batch_shots": 2,
        "seed": 53,
        "workers": 1,
    }
    configuration.update(changes)
    return configuration


def _run_surface_module(tmp_path, configuration, *, output_name="output"):
    configuration_path = tmp_path / f"{output_name}.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output_root = tmp_path / output_name
    command = [
        sys.executable,
        "-m",
        "experiments.run_surface",
        "--config",
        str(configuration_path),
        "--output",
        str(output_root),
    ]
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY,
        text=True,
        capture_output=True,
    )
    return completed, configuration_path, output_root


def _snapshot_directories(output_root):
    return sorted(output_root.glob("????-??-??/*"))


def test_layered_stim_input_is_bound_to_bytes_and_declared_rounds(tmp_path):
    source = b"M 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
    path = tmp_path / "layered.stim"
    path.write_bytes(source)

    circuit, detector_rounds, round_count = load_layered_stim_input(
        path, hashlib.sha256(source).hexdigest(), (1, 2)
    )

    assert circuit.num_detectors == 3
    assert detector_rounds == {0: 1, 1: 2, 2: 2}
    assert round_count == 2


@pytest.mark.parametrize(
    ("counts", "message"),
    [((), "nonempty tuple"), ([1, 2], "nonempty tuple"),
     ((1, 0), "positive built-in"), ((True, 2), "positive built-in")],
)
def test_layered_stim_input_rejects_invalid_layer_declarations(
    tmp_path, counts, message
):
    source = b"M 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
    path = tmp_path / "layered.stim"
    path.write_bytes(source)

    with pytest.raises((TypeError, ValueError), match=message):
        load_layered_stim_input(path, hashlib.sha256(source).hexdigest(), counts)


def test_layered_stim_input_rejects_a_detector_count_sum_mismatch(tmp_path):
    source = b"M 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
    path = tmp_path / "layered.stim"
    path.write_bytes(source)

    with pytest.raises(ValueError, match="declared detector counts"):
        load_layered_stim_input(
            path, hashlib.sha256(source).hexdigest(), (1, 1)
        )


def test_layered_stim_input_rejects_changed_bytes(tmp_path):
    path = tmp_path / "layered.stim"
    path.write_text("M 0\nDETECTOR rec[-1]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_layered_stim_input(path, "0" * 64, (1,))


_SELECTOR_FILES = {
    "configuration": {"configuration.json"},
    "invocation": {"invocation.json"},
    "runner_source": {"runner.py"},
    "circuit": {"circuit.stim"},
    "dem": {"detector_error_model.dem"},
    "detector_coordinates": {"detector_coordinates.json"},
    "detector_records": {"detector_records.jsonl"},
    "results": {"reduced_counts.json", "plot_data.csv"},
    "plot": {"logical_error_rate.png"},
    "plot_card": {"logical_error_rate.md"},
    "environment": {"environment.json"},
}


def test_chunk_csv_is_canonical_and_contains_no_operational_fields():
    first = _chunk()
    encoded = canonical_chunk_csv([first])

    assert encoded == canonical_chunk_csv([first])
    assert encoded.endswith(b"\n")
    assert b"wall" not in encoded
    assert b"hostname" not in encoded
    assert b"rss" not in encoded


def test_immutable_publication_is_idempotent_but_never_clobbers(tmp_path):
    destination = tmp_path / "chunk.csv"
    original = canonical_chunk_csv([_chunk()])
    conflicting = canonical_chunk_csv([replace(_chunk(), primary_failures=2)])

    assert publish_immutable(destination, original) is True
    assert publish_immutable(destination, original) is False
    with pytest.raises(ShardConflictError):
        publish_immutable(destination, conflicting)
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_reduction_sums_integer_counts_and_requires_exact_ranges():
    rows = [
        _chunk(),
        _chunk(
            batch_index=1,
            first_shot_index=3,
            requested_shots=2,
            attempted_shots=2,
            primary_failures=2,
            accepted_shots=1,
            accepted_logical_failures=1,
            backend_low_confidence=0,
            backend_nonconverged=1,
            window_attempts=5,
        ),
    ]

    summary = reduce_chunks(reversed(rows))

    assert summary.attempted_shots == 5
    assert summary.primary_failures == 3
    assert summary.accepted_shots == 3
    assert summary.accepted_logical_failures == 1
    assert summary.window_attempts == 11
    assert summary.primary_ler == 3 / 5

    with pytest.raises(ValueError, match="duplicate batch"):
        reduce_chunks([rows[0], rows[0]])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_chunks([rows[0], replace(rows[1], first_shot_index=4)])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_chunks([rows[0], replace(rows[1], first_shot_index=2)])


def test_surface_plot_row_uses_exact_reduced_counts():
    result = reduce_chunks([
        _chunk(),
        _chunk(
            batch_index=1,
            first_shot_index=3,
            requested_shots=2,
            attempted_shots=2,
            primary_failures=2,
        ),
    ])
    configuration = _surface_configuration()

    row = surface_plot_row(configuration, result)
    encoded = canonical_surface_plot_csv(row)
    parsed, = csv.DictReader(io.StringIO(encoded.decode("utf-8")))

    assert tuple(row) == (
        "schema_version", "experiment_id", "experiment_sha256", "config_id",
        "sample_set_id", "code_family", "distance", "rounds",
        "physical_error_rate", "basis", "decoder", "shots", "failures",
    )
    assert row["shots"] == 5
    assert row["failures"] == 3
    assert parsed["shots"] == "5"
    assert parsed["failures"] == "3"


def test_batches_cover_the_requested_shots_once_without_overshoot():
    assert [(batch.index, batch.first_shot, batch.shots) for batch in exact_batches(10, 4)] == [
        (0, 0, 4),
        (1, 4, 4),
        (2, 8, 2),
    ]


def test_sample_plan_is_stable_and_separates_experiment_seeds():
    trajectory = {
        "circuit_sha256": "c" * 64,
        "physical_error_rate": 0.001,
        "batch_shots": 100,
    }
    offline = SamplePlan.create("experiment", 7, trajectory)
    replay = SamplePlan.create("experiment", 7, trajectory)
    other_seed = SamplePlan.create("experiment", 8, trajectory)

    assert offline == replay
    expected_descriptor = json.dumps(
        trajectory,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert offline.input_sha256 == hashlib.sha256(
        expected_descriptor
    ).hexdigest()
    assert offline.sample_set_id != other_seed.sample_set_id


def test_offline_batch_seed_depends_on_batch_but_not_execution_order():
    plan = SamplePlan.create("experiment", 7, {"circuit": "memory"})

    forward = [offline_batch_seed(7, plan.sample_set_id, index) for index in range(3)]
    reversed_order = {
        index: offline_batch_seed(7, plan.sample_set_id, index)
        for index in reversed(range(3))
    }

    assert len(set(forward)) == 3
    assert forward == [reversed_order[index] for index in range(3)]


def test_sample_digest_covers_detector_truth_shape_dtype_and_bytes():
    np = pytest.importorskip("numpy")
    detectors = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    truth = np.array([[1], [0]], dtype=np.uint8)
    expected = sample_batch_sha256(detectors, truth)

    assert sample_batch_sha256(detectors.copy(), truth.copy()) == expected
    changed = detectors.copy()
    changed[0, 0] = 1
    assert sample_batch_sha256(changed, truth) != expected
    assert sample_batch_sha256(detectors.astype(np.int64), truth) != expected
    assert sample_batch_sha256(detectors.reshape(1, 4), truth) != expected
    changed_truth = truth.copy()
    changed_truth[0, 0] = 0
    assert sample_batch_sha256(detectors, changed_truth) != expected
    assert sample_batch_sha256(detectors, truth.astype(np.int64)) != expected
    assert sample_batch_sha256(detectors, truth.reshape(1, 2)) != expected


def test_experiment_bytes_and_hash_are_canonical_and_seed_sensitive():
    experiment = Experiment(
        experiment_id="experiment",
        experiment_seed=7,
        configurations=({"decoder": "mwpm", "distance": 3},),
        sampling={"batch_shots": 4, "max_shots": 10},
        stopping={"method": "fixed"},
    )

    assert experiment.canonical_bytes() == experiment.canonical_bytes()
    assert experiment.sha256() != replace(experiment, experiment_seed=8).sha256()


def test_offline_decoder_reuses_models_and_compiles_one_sampler_per_batch():
    stim = pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    from decsim.detector_error_model import (
        FaultRepresentation,
        GRAPHLIKE_FAULT_MODEL_REQUIRED,
        decode_windowed,
    )
    from decsim.mwpm_decoder import matching_window_decoder
    from decsim.schemes import SlidingWindowScheme

    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=6,
        after_clifford_depolarization=0.05,
    )
    source_round_count = 6
    windows = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0, source_round_count, commit_round_count=3, buffer_round_count=3
        ).windows
    ]

    class CountingCircuit:
        def __init__(self, inner):
            self.inner = inner
            self.sampler_compiles = 0

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def compile_detector_sampler(self, **kwargs):
            self.sampler_compiles += 1
            return self.inner.compile_detector_sampler(**kwargs)

    counted = CountingCircuit(circuit)
    inner = matching_window_decoder()
    model_ids = []

    def decode(model, syndrome):
        model_ids.append(id(model))
        return inner(model, syndrome)

    decoder = OfflineBatchDecoder.prepare(
        counted,
        windows,
        decode,
        round_count=source_round_count,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_representation=FaultRepresentation.GRAPHLIKE,
    )
    batches = exact_batches(103, 100)
    first = batches[0]
    seed = 19
    result = decoder.run(first, seed)

    detectors, truth = circuit.compile_detector_sampler(seed=seed).sample(
        shots=first.shots, separate_observables=True
    )
    reference_decode = matching_window_decoder()
    failures = sum(
        tuple(int(bit) for bit in decode_windowed(
            decoder.window_models,
            detectors[index],
            reference_decode,
            selected_fault_representation=FaultRepresentation.GRAPHLIKE,
        ))
        != tuple(int(bit) for bit in truth[index])
        for index in range(first.shots)
    )

    assert result.attempted_shots == first.shots
    assert result.primary_failures == failures
    assert result.sample_batch_sha256 == sample_batch_sha256(detectors, truth)
    assert counted.sampler_compiles == 1
    assert set(model_ids) == {id(model) for model in decoder.window_models}

    assert decoder.run(first, seed) == result
    decoder.run(batches[1], seed + 1)
    assert counted.sampler_compiles == 3
    assert set(model_ids) == {id(model) for model in decoder.window_models}


def test_offline_experiment_writes_chunks_and_resumes_without_redecoding(tmp_path):
    experiment = Experiment(
        experiment_id="offline-smoke",
        experiment_seed=7,
        configurations=({"decoder": "test"},),
        sampling={"batch_shots": 4, "max_shots": 7},
        stopping={"method": "fixed"},
    )
    plan = SamplePlan.create(
        experiment.experiment_id,
        experiment.experiment_seed,
        {"circuit_sha256": "c" * 64, "batch_shots": 4},
    )

    class FakeDecoder:
        def __init__(self):
            self.calls = []

        def run(self, batch, seed):
            self.calls.append((batch.index, seed))
            return DecodedBatch(
                batch=batch,
                sample_batch_sha256=f"{batch.index + 1:064x}",
                attempted_shots=batch.shots,
                primary_failures=batch.index,
                accepted_shots=batch.shots,
                accepted_logical_failures=batch.index,
                window_attempts=batch.shots * 2,
            )

    decoder = FakeDecoder()
    batches = exact_batches(7, 4)
    summary = run_offline_experiment(
        decoder,
        experiment,
        plan,
        {"decoder": "test"},
        batches,
        tmp_path,
    )

    assert summary.attempted_shots == 7
    assert summary.primary_failures == 1
    assert len(decoder.calls) == 2
    assert len(list(tmp_path.rglob("*.csv"))) == 2

    assert run_offline_experiment(
        decoder,
        experiment,
        plan,
        {"decoder": "test"},
        batches,
        tmp_path,
    ) == summary
    assert len(decoder.calls) == 2


def test_parallel_offline_run_matches_one_worker_byte_for_byte(tmp_path):
    experiment = Experiment(
        experiment_id="parallel-smoke",
        experiment_seed=11,
        configurations=({"decoder": "test"},),
        sampling={"batch_shots": 3, "max_shots": 10},
        stopping={"method": "fixed"},
    )
    plan = SamplePlan.create(
        experiment.experiment_id,
        experiment.experiment_seed,
        {"circuit_sha256": "d" * 64, "batch_shots": 3},
    )
    batches = exact_batches(10, 3)
    one = tmp_path / "one"
    two = tmp_path / "two"

    summary_one = run_offline_parallel(
        _deterministic_offline_decoder,
        experiment,
        plan,
        {"decoder": "test"},
        batches,
        one,
        workers=1,
    )
    summary_two = run_offline_parallel(
        _deterministic_offline_decoder,
        experiment,
        plan,
        {"decoder": "test"},
        reversed(batches),
        two,
        workers=2,
    )

    assert summary_one == summary_two
    one_chunks = sorted(path.read_bytes() for path in one.rglob("*.csv"))
    two_chunks = sorted(path.read_bytes() for path in two.rglob("*.csv"))
    assert one_chunks == two_chunks


def test_slurm_array_maps_one_index_to_one_configuration(tmp_path):
    configurations = tmp_path / "configurations.txt"
    configurations.write_text("first.json\nsecond.json\nthird.json\n")
    script = Path(__file__).parents[1] / "slurm_array.sh"
    environment = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": "1",
        "DECSIM_PYTHON": "/bin/echo",
    }

    completed = subprocess.run(
        ["bash", script, configurations, tmp_path / "output", "--ignored-option"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == (
        "-m experiments.run_surface --ignored-option "
        f"--config second.json --output {tmp_path / 'output'}"
    )
    assert "%" not in script.read_text()


def test_ler_plot_writes_figure_and_plain_language_card(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    output = tmp_path / "surface_ler.png"
    rows = [
        {"distance": 3, "physical_error_rate": 0.001, "failures": 3, "shots": 10_000},
        {"distance": 3, "physical_error_rate": 0.003, "failures": 20, "shots": 10_000},
        {"distance": 5, "physical_error_rate": 0.001, "failures": 0, "shots": 10_000},
        {"distance": 5, "physical_error_rate": 0.003, "failures": 4, "shots": 10_000},
    ]

    plot_logical_error_rate(rows, output, title="Offline weak-decoder baseline")
    assert not output.with_suffix(".md").exists()
    write_logical_error_rate_card(
        rows, output.with_suffix(".md"), title="Offline weak-decoder baseline"
    )

    assert output.stat().st_size > 1_000
    card = output.with_suffix(".md").read_text()
    assert "physical error rate" in card.lower()
    assert "logical error rate" in card.lower()
    assert "10,000" in card
    assert "offline" in card.lower()


def test_zero_failure_interval_matches_closed_form():
    lower, upper = _binomial_interval(0, 10_000)

    assert lower == 0
    assert upper == pytest.approx(1 - 0.025 ** (1 / 10_000))


def test_surface_runner_executes_real_mwpm_batches(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = {
        "distance": 3,
        "rounds": 6,
        "physical_error_rate": 0.001,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 7,
        "batch_shots": 4,
        "seed": 17,
        "workers": 1,
    }

    result = run_surface_configuration(configuration, tmp_path)

    assert result.attempted_shots == 7
    assert result.window_attempts > result.attempted_shots
    assert len(list(tmp_path.rglob("*.csv"))) == 2


def test_surface_runner_uses_requested_circuit_noise():
    stim = pytest.importorskip("stim")
    configuration = {
        "distance": 3,
        "rounds": 3,
        "physical_error_rate": 0.007,
    }
    circuit = _circuit(configuration)
    expected = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.007,
        after_reset_flip_probability=0.007,
        before_measure_flip_probability=0.007,
        before_round_data_depolarization=0.007,
    )

    assert circuit == expected


def test_surface_runner_uses_the_simulator_sliding_window_geometry():
    configuration = {
        "rounds": 7,
        "commit_rounds": 3,
        "buffer_rounds": 3,
    }

    assert _sliding_window_entries(configuration) == (
        (1, 3, 6),
        (4, 6, 9),
        (7, 7, 10),
    )
    assert _sliding_window_entries({**configuration, "rounds": 6}) == (
        (1, 3, 6),
        (4, 6, 9),
    )


def test_surface_runner_worker_count_does_not_change_scientific_identity(tmp_path):
    configuration = {
        "distance": 3,
        "rounds": 3,
        "physical_error_rate": 0.005,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 7,
        "batch_shots": 4,
        "seed": 29,
    }

    one = run_surface_configuration({**configuration, "workers": 1}, tmp_path / "one")
    two = run_surface_configuration({**configuration, "workers": 2}, tmp_path / "two")

    assert one == two
    assert sorted(path.read_bytes() for path in (tmp_path / "one").rglob("*.csv")) == sorted(
        path.read_bytes() for path in (tmp_path / "two").rglob("*.csv")
    )


def test_surface_runner_resolves_default_experiment_name_before_identity(tmp_path):
    configuration = {
        "distance": 3,
        "rounds": 3,
        "physical_error_rate": 0.005,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 3,
        "batch_shots": 2,
        "seed": 31,
        "workers": 1,
    }

    implicit = run_surface_configuration(configuration, tmp_path / "implicit")
    explicit = run_surface_configuration(
        {**configuration, "experiment_id": "surface-weak-mwpm"},
        tmp_path / "explicit",
    )

    assert implicit == explicit
    assert sorted(
        path.read_bytes() for path in (tmp_path / "implicit").rglob("*.csv")
    ) == sorted(
        path.read_bytes() for path in (tmp_path / "explicit").rglob("*.csv")
    )


def test_surface_runner_folds_terminal_detectors_into_the_final_window():
    configuration = {
        "distance": 3,
        "rounds": 6,
        "physical_error_rate": 0.008,
        "commit_rounds": 3,
        "buffer_rounds": 3,
    }
    decoder = _SurfaceMwpmFactory(configuration)()
    terminal_detector_ids = {
        detector_id
        for detector_id, coordinates in
        decoder.circuit.get_detector_coordinates().items()
        if int(coordinates[-1]) == configuration["rounds"]
    }

    assert terminal_detector_ids
    assert decoder.window_models[-1].commit_lo == 4
    assert decoder.window_models[-1].commit_hi == 6
    assert terminal_detector_ids <= set(decoder.window_models[-1].detector_ids)


@pytest.mark.parametrize(
    "snapshot_fields, expected_error",
    [
        ({"artifacts": ["configuration"]}, "run_name"),
        (
            {"artifacts": ["detector_records"], "run_name": "missing-shots"},
            "detector_record_shots",
        ),
        (
            {
                "artifacts": ["detector_records"],
                "run_name": "empty-shots",
                "detector_record_shots": [],
            },
            "detector_record_shots",
        ),
    ],
)
def test_surface_snapshot_requires_explicit_inputs_before_writing(
    tmp_path, snapshot_fields, expected_error
):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = {
        "distance": 3,
        "rounds": 3,
        "physical_error_rate": 0.001,
        "commit_rounds": 3,
        "buffer_rounds": 3,
        "shots": 3,
        "batch_shots": 2,
        "seed": 53,
        "workers": 1,
        **snapshot_fields,
    }
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    output_root = tmp_path / "output"

    completed = subprocess.run(
        [
            str(Path(__file__).parents[2] / ".venv" / "bin" / "python"),
            "-m",
            "experiments.run_surface",
            "--config",
            str(configuration_path),
            "--output",
            str(output_root),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not output_root.exists()


@pytest.mark.parametrize("selector", tuple(_SELECTOR_FILES))
def test_each_surface_snapshot_selector_writes_only_its_group(tmp_path, selector):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = _surface_configuration(
        artifacts=[selector],
        run_name=f"only-{selector}",
        detector_record_shots=[0, 2],
    )

    completed, _, output_root = _run_surface_module(tmp_path, configuration)

    assert completed.returncode == 0, completed.stderr
    snapshot, = _snapshot_directories(output_root)
    expected_names = _SELECTOR_FILES[selector] | {"SHA256SUMS"}
    assert {path.name for path in snapshot.iterdir()} == expected_names
    expected_manifest = "".join(
        f"{hashlib.sha256((snapshot / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(_SELECTOR_FILES[selector], key=str.encode)
    )
    assert (snapshot / "SHA256SUMS").read_text() == expected_manifest


@pytest.mark.parametrize(
    "artifacts, expected_error",
    [
        ([], "nonempty"),
        (["results", "results"], "duplicate"),
        (["all"], "unknown"),
        (["not-a-group"], "unknown"),
    ],
)
def test_surface_snapshot_rejects_invalid_selector_lists(
    tmp_path, artifacts, expected_error
):
    configuration = _surface_configuration(
        artifacts=artifacts,
        run_name="invalid-selection",
    )

    completed, _, output_root = _run_surface_module(tmp_path, configuration)

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not output_root.exists()


def test_omitting_artifacts_preserves_science_only_output(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    completed, _, output_root = _run_surface_module(
        tmp_path, _surface_configuration()
    )

    assert completed.returncode == 0, completed.stderr
    assert not _snapshot_directories(output_root)
    assert len(list(output_root.rglob("*.csv"))) == 2


def test_output_choices_do_not_change_scientific_identity_or_chunks(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    common = _surface_configuration(shots=7, batch_shots=4)
    first = {
        **common,
        "workers": 1,
        "artifacts": ["configuration"],
        "run_name": "first-choice",
    }
    second = {
        **common,
        "workers": 2,
        "artifacts": ["detector_records"],
        "run_name": "second-choice",
        "detector_record_shots": [1, 6],
    }

    first_run, _, first_root = _run_surface_module(
        tmp_path, first, output_name="first"
    )
    second_run, _, second_root = _run_surface_module(
        tmp_path, second, output_name="second"
    )

    assert first_run.returncode == second_run.returncode == 0
    assert json.loads(first_run.stdout) == json.loads(second_run.stdout)
    first_chunks = sorted(path.read_bytes() for path in first_root.rglob("*.csv"))
    second_chunks = sorted(path.read_bytes() for path in second_root.rglob("*.csv"))
    assert first_chunks == second_chunks


def test_surface_snapshot_artifacts_are_exact_and_honestly_scoped(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = _surface_configuration(
        artifacts=list(_SELECTOR_FILES),
        run_name="complete-snapshot",
        detector_record_shots=[0, 2],
    )

    completed, _, output_root = _run_surface_module(tmp_path, configuration)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    snapshot, = _snapshot_directories(output_root)
    saved_configuration = json.loads((snapshot / "configuration.json").read_text())
    invocation = json.loads((snapshot / "invocation.json").read_text())
    environment = json.loads((snapshot / "environment.json").read_text())
    plot_row, = csv.DictReader((snapshot / "plot_data.csv").open())

    assert saved_configuration["requested"] == configuration
    assert not ({"workers", "run_name", "artifacts", "detector_record_shots"}
                & saved_configuration["resolved"].keys())
    assert saved_configuration["batches"][-1] == {
        "index": 1, "first_shot": 2, "shots": 1
    }
    assert invocation["replay_argv"][:3] == [
        sys.executable, "-m", "experiments.run_surface"
    ]
    assert invocation["process_argv_observed"] != invocation["replay_argv"]
    assert Path(invocation["cwd_observed"]) == _REPOSITORY
    assert (snapshot / "runner.py").read_bytes() == (
        _REPOSITORY / "experiments" / "run_surface.py"
    ).read_bytes()
    assert (snapshot / "circuit.stim").read_text() == str(_circuit(configuration))
    assert "error(" in (snapshot / "detector_error_model.dem").read_text()
    coordinates = json.loads((snapshot / "detector_coordinates.json").read_text())
    assert [row["detector"] for row in coordinates] == sorted(
        row["detector"] for row in coordinates
    )
    assert json.loads((snapshot / "reduced_counts.json").read_text()) == result
    assert int(plot_row["shots"]) == result["attempted_shots"]
    assert int(plot_row["failures"]) == result["primary_failures"]
    assert (snapshot / "logical_error_rate.png").stat().st_size > 1_000
    assert "offline" in (snapshot / "logical_error_rate.md").read_text().lower()
    assert set(environment["direct_package_versions"]) == {
        "stim", "pymatching", "numpy", "scipy", "matplotlib"
    }
    assert "not a dependency closure" in environment["scope"]

    forbidden = {"latency_us", "traffic_bytes", "complete", "attempt"}

    def nested_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(nested_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    json_keys = set()
    for path in snapshot.glob("*.json"):
        json_keys |= nested_keys(json.loads(path.read_text()))
    for line in (snapshot / "detector_records.jsonl").read_text().splitlines():
        json_keys |= nested_keys(json.loads(line))
    csv_keys = set(plot_row)
    assert forbidden.isdisjoint(json_keys | csv_keys)
    assert not any(forbidden & set(path.name.split(".")) for path in snapshot.iterdir())


def test_detector_records_match_full_and_short_stored_batches_after_resume(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = _surface_configuration(
        artifacts=["detector_records"],
        run_name="first-records",
        detector_record_shots=[0, 2],
    )

    first_run, _, output_root = _run_surface_module(tmp_path, configuration)

    assert first_run.returncode == 0, first_run.stderr
    first_snapshot, = _snapshot_directories(output_root)
    first_records = [
        json.loads(line)
        for line in (first_snapshot / "detector_records.jsonl").read_text().splitlines()
    ]
    result = json.loads(first_run.stdout)
    decoder = _SurfaceMwpmFactory(_surface_configuration())()
    for record in first_records:
        batch = exact_batches(3, 2)[record["batch_index"]]
        seed = offline_batch_seed(53, result["sample_set_id"], batch.index)
        detectors, truth = decoder.circuit.compile_detector_sampler(seed=seed).sample(
            shots=batch.shots, separate_observables=True
        )
        local_index = record["shot_index"] - batch.first_shot
        prediction = decode_windowed(
            decoder.window_models,
            detectors[local_index],
            decoder.decode_window,
            selected_fault_representation=FaultRepresentation.GRAPHLIKE,
        )
        assert record["sample_batch_sha256"] == sample_batch_sha256(detectors, truth)
        assert record["fired_detector_indices"] == [
            index for index, fired in enumerate(detectors[local_index]) if fired
        ]
        assert record["observable_truth"] == [int(bit) for bit in truth[local_index]]
        assert record["prediction"] == [int(bit) for bit in prediction]

    chunk_bytes = [path.read_bytes() for path in sorted(output_root.rglob("*.csv"))]
    configuration["run_name"] = "resumed-records"
    resumed, _, _ = _run_surface_module(tmp_path, configuration)
    assert resumed.returncode == 0, resumed.stderr
    snapshots = _snapshot_directories(output_root)
    assert len(snapshots) == 2
    assert (snapshots[0] / "detector_records.jsonl").read_bytes() == (
        snapshots[1] / "detector_records.jsonl"
    ).read_bytes()
    assert chunk_bytes == [
        path.read_bytes() for path in sorted(output_root.rglob("*.csv"))
    ]


def test_detector_records_reject_a_mismatched_stored_batch_digest(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = _surface_configuration(
        artifacts=["detector_records"],
        run_name="valid-records",
        detector_record_shots=[0],
    )
    first_run, _, output_root = _run_surface_module(tmp_path, configuration)
    assert first_run.returncode == 0, first_run.stderr
    first_chunk = sorted(output_root.rglob("*.csv"))[0]
    stored = read_stored_batch_result(
        output_root,
        run_surface_configuration(configuration, output_root),
        0,
    )
    first_chunk.write_bytes(canonical_chunk_csv([
        replace(stored, sample_batch_sha256="0" * 64)
    ]))
    configuration["run_name"] = "mismatched-records"

    completed, _, _ = _run_surface_module(tmp_path, configuration)

    assert completed.returncode != 0
    assert "stored sample digest" in completed.stderr
    latest_snapshot = _snapshot_directories(output_root)[-1]
    assert not (latest_snapshot / "SHA256SUMS").exists()


def test_direct_slurm_and_replay_use_the_same_surface_module(tmp_path):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = _surface_configuration(
        artifacts=["invocation"],
        run_name="entrypoint-parity",
    )
    direct, configuration_path, direct_root = _run_surface_module(
        tmp_path, configuration, output_name="direct"
    )
    assert direct.returncode == 0, direct.stderr
    direct_snapshot, = _snapshot_directories(direct_root)
    replay_argv = json.loads(
        (direct_snapshot / "invocation.json").read_text()
    )["replay_argv"]

    replay = subprocess.run(
        replay_argv, cwd=_REPOSITORY, text=True, capture_output=True
    )
    configurations = tmp_path / "configurations.txt"
    configurations.write_text(f"{configuration_path}\n", encoding="utf-8")
    slurm_root = tmp_path / "slurm"
    environment = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": "0",
        "DECSIM_PYTHON": sys.executable,
    }
    slurm = subprocess.run(
        [
            "bash",
            _REPOSITORY / "experiments" / "slurm_array.sh",
            configurations,
            slurm_root,
        ],
        cwd=_REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert replay.returncode == slurm.returncode == 0
    assert json.loads(direct.stdout) == json.loads(replay.stdout) == json.loads(slurm.stdout)
    direct_chunks = sorted(path.read_bytes() for path in direct_root.rglob("*.csv"))
    slurm_chunks = sorted(path.read_bytes() for path in slurm_root.rglob("*.csv"))
    assert direct_chunks == slurm_chunks


def test_snapshot_write_failure_leaves_partial_output_and_untouched_chunks(
    tmp_path, monkeypatch
):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    configuration = _surface_configuration(
        artifacts=["configuration", "invocation"],
        run_name="write-failure",
    )
    output_root = tmp_path / "output"
    result = run_surface_configuration(configuration, output_root)
    chunks_before = [path.read_bytes() for path in sorted(output_root.rglob("*.csv"))]
    original_write_bytes = Path.write_bytes

    def fail_invocation(path, content):
        if path.name == "invocation.json":
            raise OSError("injected snapshot write failure")
        return original_write_bytes(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail_invocation)
        with pytest.raises(OSError, match="injected snapshot write failure"):
            write_surface_snapshot(
                configuration,
                output_root,
                result,
                tuple(configuration["artifacts"]),
                {"observed": "test"},
            )

    partial, = _snapshot_directories(output_root)
    assert {path.name for path in partial.iterdir()} == {"configuration.json"}
    assert chunks_before == [
        path.read_bytes() for path in sorted(output_root.rglob("*.csv"))
    ]
    completed = write_surface_snapshot(
        configuration,
        output_root,
        result,
        tuple(configuration["artifacts"]),
        {"observed": "test"},
    )
    assert completed != partial
    assert (completed / "SHA256SUMS").exists()


def test_snapshot_timestamp_collision_is_loud_and_not_retried(
    tmp_path, monkeypatch
):
    fixed_time = surface_runner.datetime(
        2026, 7, 30, 12, 34, 56, 789, tzinfo=surface_runner.timezone.utc
    )

    class FixedDateTime:
        @classmethod
        def now(cls, timezone):
            return fixed_time

    monkeypatch.setattr(surface_runner, "datetime", FixedDateTime)
    first = _new_output_directory(tmp_path, "collision", "c" * 64)

    with pytest.raises(FileExistsError):
        _new_output_directory(tmp_path, "collision", "c" * 64)

    assert first.exists()
