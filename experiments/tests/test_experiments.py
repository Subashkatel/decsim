from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from decsim.detector_error_model import FaultRepresentation, decode_windowed

from experiments.results import (
    ChunkResult,
    ShardConflictError,
    canonical_chunk_csv,
    publish_immutable,
    reduce_chunks,
)
from experiments.plotting import _binomial_interval, plot_logical_error_rate
from experiments.run_surface import (
    _SurfaceMwpmFactory,
    _circuit,
    _windows,
    run_surface_configuration,
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
    layers = 1 + max(
        int(coordinates[-1])
        for coordinates in circuit.get_detector_coordinates().values()
    )
    windows = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0, layers, commit_round_count=3, buffer_round_count=3
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
    environment = {"SLURM_ARRAY_TASK_ID": "1"}

    completed = subprocess.run(
        ["bash", script, configurations, "/bin/echo"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == "--config second.json"
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


def test_surface_runner_absorbs_terminal_layer_and_short_tail():
    configuration = {
        "rounds": 7,
        "commit_rounds": 3,
        "buffer_rounds": 3,
    }

    assert _windows(configuration) == ((1, 3, 6), (4, 7, 7))
    assert _windows({**configuration, "rounds": 6}) == (
        (1, 3, 6),
        (4, 6, 6),
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


def test_surface_runner_window_decode_matches_global_reference():
    pymatching = pytest.importorskip("pymatching")
    configuration = {
        "distance": 3,
        "rounds": 6,
        "physical_error_rate": 0.008,
        "commit_rounds": 3,
        "buffer_rounds": 3,
    }
    decoder = _SurfaceMwpmFactory(configuration)()
    detectors, _ = decoder.circuit.compile_detector_sampler(seed=43).sample(
        shots=1, separate_observables=True
    )
    prediction = decode_windowed(
        decoder.window_models,
        detectors[0],
        decoder.decode_window,
        selected_fault_representation=FaultRepresentation.GRAPHLIKE,
    )
    global_matching = pymatching.Matching.from_detector_error_model(
        decoder.circuit.detector_error_model(decompose_errors=True)
    )

    assert tuple(prediction) == tuple(global_matching.decode(detectors[0]))
