"""Tests for slicing global detector error models into window models.

Paper contract: docs/PAPER_MODEL_MAP.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from types import SimpleNamespace

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")

from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    LINKED_FAULT_MODELS_REQUIRED,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    PlacedFaultModel,
    WindowErrorModel,
    build_window_error_models,
    canonical_error_instructions,
    decode_windowed,
    decode_windowed_backend_outcomes,
    detector_error_model_to_faults,
    detector_error_model_to_faults_bm,
)
from decsim.adapters.window_decode_results import (
    BackendDecodeOutcome,
    BackendDecodeStatus,
    BackendFailureReason,
    DecoderAttemptFailed,
    OfflineShotRecord,
    decoder_configuration_fingerprint,
    empty_fault_model_outcome,
    fault_model_fingerprint,
    summarize_offline_shots,
    validate_backend_outcome,
)
from decsim.mwpm_decoder import matching_window_decoder
from decsim.codes import SurfaceCodeModel
from decsim.schemes import SlidingWindowScheme, ParallelWindowScheme


def _memory_circuit(d=3, rounds=12, p=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _plan(circuit, d=3):
    """The REAL scheme's window plan over the circuit's detector layers
    (layer t = round t+1), exactly as the cluster would plan it."""
    n_layers = 1 + max(int(c[-1]) for c in
                       circuit.get_detector_coordinates().values())
    return [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0,
            n_layers,
            commit_round_count=d,
            buffer_round_count=d,
        ).windows
    ]


def _placed(representation, check, priors, observables, *, owned=None,
            future_flips=None):
    check = np.asarray(check, dtype=np.uint8)
    return PlacedFaultModel(
        representation=representation,
        check=check,
        priors=np.asarray(priors, dtype=float),
        observables=np.asarray(observables, dtype=np.uint8),
        owned=(np.ones(check.shape[1], dtype=bool)
               if owned is None else np.asarray(owned, dtype=bool)),
        future_flips={} if future_flips is None else future_flips,
        source_fault_ids=tuple(range(check.shape[1])),
    )


def _window(*, detector_ids, graphlike=None, physical=None, link=None,
            commit_hi=1, defect_positions=None):
    return WindowErrorModel(
        detector_ids=tuple(detector_ids),
        detector_coordinates=None,
        commit_hi=commit_hi,
        defect_positions={} if defect_positions is None else defect_positions,
        graphlike_faults=graphlike,
        physical_faults=physical,
        physical_to_graphlike_detector_projection=link,
    )


def _adapter_physical_window():
    return _window(
        detector_ids=(0, 1),
        physical=_placed(
            FaultRepresentation.PHYSICAL,
            [[1, 0], [0, 1]],
            [0.1, 0.2],
            [[1, 0]],
        ),
    )


class _FakeTesseractDecoder:
    def __init__(self, selected_errors=(0,), *, low_confidence=False):
        self.selected_errors = selected_errors
        self.low_confidence_flag = low_confidence
        self.decode_calls = 0

    def decode_to_errors(self, syndrome):
        self.decode_calls += 1
        return self.selected_errors


def _install_fake_tesseract(monkeypatch, backend_decoder, observed_seeds):
    from decsim.tesseract_decoder import window_decoder as adapter_module

    def build_det_orders(dem, count, method, seed):
        observed_seeds.append(seed)
        return ("fixed-order",)

    fake_backend = SimpleNamespace(
        utils=SimpleNamespace(
            DetOrder=SimpleNamespace(
                DetIndex="index",
                DetBFS="breadth-first",
                DetCoordinate="coordinate",
            ),
            build_det_orders=build_det_orders,
        ),
        tesseract=SimpleNamespace(
            TesseractConfig=lambda **kwargs: SimpleNamespace(
                compile_decoder=lambda: backend_decoder
            )
        ),
    )
    monkeypatch.setattr(
        adapter_module,
        "_load_tesseract_backend",
        lambda: fake_backend,
    )


def _install_fake_relay(monkeypatch, detailed_result, observed):
    from decsim.relay_bp_decoder import RelayBpWindowDecoder

    class FakeRelayDecoder:
        def __init__(self, check, priors, **kwargs):
            observed["seeds"].append(kwargs["seed"])
            observed["gamma_tables"].append(kwargs["explicit_gammas"])

        def decode_detailed(self, syndrome):
            observed["calls"] += 1
            return detailed_result

    monkeypatch.setattr(
        RelayBpWindowDecoder,
        "_load_relay_decoder_type",
        staticmethod(lambda: FakeRelayDecoder),
    )


def _relay_result(*, success=True, decoded_detectors=(1, 0)):
    return SimpleNamespace(
        decoding=np.array([1, 0], dtype=np.uint8),
        decoded_detectors=np.array(decoded_detectors, dtype=np.uint8),
        iterations=4,
        max_iter=20,
        posterior_ratios=np.array([-2.0, 3.0]),
        success=success,
    )


def _backend_outcome(
    status,
    *,
    model_fingerprint,
    correction=None,
    reconstructed_syndrome=None,
    failure_reason=None,
    iterations=None,
    posterior_ratios=None,
):
    return BackendDecodeOutcome(
        status=status,
        failure_reason=failure_reason,
        physical_correction=correction,
        component_correction=None,
        reconstructed_syndrome=reconstructed_syndrome,
        iterations=iterations,
        iteration_limit=None,
        posterior_log_likelihood_ratios=posterior_ratios,
        fault_model_fingerprint=model_fingerprint,
        decoder_configuration_fingerprint=("2" * 64),
    )


@pytest.mark.parametrize(
    ("selected_errors", "low_confidence", "expected_status",
     "expected_reason"),
    [
        ((0,), False, BackendDecodeStatus.SUCCEEDED, None),
        (
            (0,),
            True,
            BackendDecodeStatus.LOW_CONFIDENCE,
            BackendFailureReason.SEARCH_LIMIT_EXHAUSTED,
        ),
        (
            (0, 0),
            False,
            BackendDecodeStatus.INVALID_CORRECTION,
            BackendFailureReason.CORRECTION_WRONG_ARITY,
        ),
    ],
)
def test_tesseract_adapter_calls_official_api_once_and_types_disposition(
    monkeypatch,
    selected_errors,
    low_confidence,
    expected_status,
    expected_reason,
):
    from decsim.tesseract_decoder import TesseractDecoderConfig
    from decsim.tesseract_decoder import TesseractWindowDecoder

    backend = _FakeTesseractDecoder(
        selected_errors,
        low_confidence=low_confidence,
    )
    _install_fake_tesseract(monkeypatch, backend, [])
    decoder = TesseractWindowDecoder(
        TesseractDecoderConfig(detector_order_seed=7)
    )

    outcome = decoder.decode(
        _adapter_physical_window(),
        np.array([1, 0], dtype=np.uint8),
    )

    assert backend.decode_calls == 1
    assert outcome.status is expected_status
    assert outcome.failure_reason is expected_reason
    if expected_status is BackendDecodeStatus.SUCCEEDED:
        assert outcome.physical_correction == (1, 0)
        assert outcome.reconstructed_syndrome == (1, 0)


@pytest.mark.parametrize(
    ("detailed_result", "expected_status", "expected_reason"),
    [
        (
            _relay_result(success=False),
            BackendDecodeStatus.NONCONVERGED,
            BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION,
        ),
        (
            _relay_result(decoded_detectors=(0, 1)),
            BackendDecodeStatus.INVALID_CORRECTION,
            BackendFailureReason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
        ),
    ],
)
def test_relay_adapter_calls_detailed_api_once_and_types_disposition(
    monkeypatch,
    detailed_result,
    expected_status,
    expected_reason,
):
    from decsim.relay_bp_decoder import RelayBpWindowDecoder

    observed = {"calls": 0, "seeds": [], "gamma_tables": []}
    _install_fake_relay(monkeypatch, detailed_result, observed)
    decoder = RelayBpWindowDecoder(
        relay_set_count=2,
        gamma_table_seed=11,
    )

    outcome = decoder.decode(
        _adapter_physical_window(),
        np.array([1, 0], dtype=np.uint8),
    )

    assert observed["calls"] == 1
    assert outcome.status is expected_status
    assert outcome.failure_reason is expected_reason
    assert outcome.iterations == 4
    assert outcome.iteration_limit == 20
    assert outcome.posterior_log_likelihood_ratios == (-2.0, 3.0)


def test_tesseract_none_seed_reservation_keeps_entropy_private():
    from decsim.tesseract_decoder import TesseractWindowDecoder

    decoder = TesseractWindowDecoder()
    reservation = decoder.reserve_run_seed(None)

    assert reservation.proposed_seed_source == "entropy"
    assert reservation.proposed_seed is None
    assert type(reservation.prepared_state) is int
    assert 0 <= reservation.prepared_state < 2**64

    decoder.commit_run_seed(reservation)


def test_physical_adapter_run_seed_binding_is_replayable_and_conflict_explicit(
    monkeypatch,
):
    from decsim.relay_bp_decoder import RelayBpWindowDecoder
    from decsim.tesseract_decoder import (
        TesseractDecoderConfig,
        TesseractWindowDecoder,
    )

    with pytest.raises(ValueError, match="explicit seed.*conflicts"):
        TesseractWindowDecoder(
            TesseractDecoderConfig(detector_order_seed=1)
        ).reserve_run_seed(2)
    with pytest.raises(ValueError, match="explicit gamma-table seed.*conflicts"):
        RelayBpWindowDecoder(gamma_table_seed=1).reserve_run_seed(2)

    tesseract_seeds = []
    _install_fake_tesseract(
        monkeypatch,
        _FakeTesseractDecoder(),
        observed_seeds=tesseract_seeds,
    )
    tesseract = TesseractWindowDecoder()
    reservation = tesseract.reserve_run_seed(91)
    tesseract.commit_run_seed(reservation)
    tesseract.decode(
        _adapter_physical_window(),
        np.array([1, 0], dtype=np.uint8),
    )
    assert tesseract_seeds == [91]

    relay_observed = {"calls": 0, "seeds": [], "gamma_tables": []}
    _install_fake_relay(
        monkeypatch,
        _relay_result(),
        relay_observed,
    )
    relay = RelayBpWindowDecoder(relay_set_count=2)
    reservation = relay.reserve_run_seed(91)
    relay.commit_run_seed(reservation)
    relay.decode(
        _adapter_physical_window(),
        np.array([1, 0], dtype=np.uint8),
    )
    expected_gammas = np.random.Generator(np.random.PCG64(91)).uniform(
        -0.24,
        0.66,
        size=(2, 2),
    )
    assert relay_observed["seeds"] == [91]
    assert np.array_equal(relay_observed["gamma_tables"][0], expected_gammas)


@pytest.mark.parametrize("decoder_family", ["tesseract", "relay"])
def test_physical_runtime_adapters_commit_correction_and_inject_latency(
    monkeypatch,
    decoder_family,
):
    from decsim.decoders import CodeRouter
    from decsim.message import DecodeJob, SyndromePayload

    class FixedLatency:
        def latency(self, job):
            return 17

    model = _adapter_physical_window()
    job = DecodeJob(
        op_id=5,
        window_id=2,
        n_rounds=1,
        dem=model,
        payloads=[
            SyndromePayload(
                operation_id=5,
                patch_id=0,
                round_index=1,
                bits=np.array([True, False]),
            )
        ],
    )
    if decoder_family == "tesseract":
        from decsim.tesseract_decoder import TesseractDecoder

        decoder = TesseractDecoder(FixedLatency())
    else:
        from decsim.relay_bp_decoder import RelayBpDecoder

        decoder = RelayBpDecoder(FixedLatency())
    faults = model.require_faults(FaultRepresentation.PHYSICAL)
    outcome = _backend_outcome(
        BackendDecodeStatus.SUCCEEDED,
        model_fingerprint=fault_model_fingerprint(faults),
        correction=(1, 0),
        reconstructed_syndrome=(1, 0),
    )
    monkeypatch.setattr(
        decoder.window_decoder,
        "decode",
        lambda received_model, syndrome: outcome,
    )
    router = CodeRouter(decoder)

    result = router.route(job).decode(job)

    assert (
        router.fault_model_requirement_for(job.code)
        is PHYSICAL_FAULT_MODEL_REQUIRED
    )
    assert decoder.latency(job) == 17
    assert result.logical_observables == (1,)


def test_backend_outcome_snapshots_mutable_backend_arrays():
    correction = np.array([1, 0], dtype=np.uint8)
    reconstructed = np.array([1], dtype=np.uint8)
    posterior_ratios = np.array([0.25, -0.75])

    outcome = _backend_outcome(
        BackendDecodeStatus.SUCCEEDED,
        model_fingerprint="1" * 64,
        correction=correction,
        reconstructed_syndrome=reconstructed,
        iterations=3,
        posterior_ratios=posterior_ratios,
    )
    correction[:] = 0
    reconstructed[:] = 0
    posterior_ratios[:] = 99

    assert outcome.physical_correction == (1, 0)
    assert outcome.reconstructed_syndrome == (1,)
    assert outcome.posterior_log_likelihood_ratios == (0.25, -0.75)
    assert outcome.succeeded


def test_backend_status_and_reason_must_describe_the_same_result():
    with pytest.raises(ValueError, match="successful.*failure reason"):
        _backend_outcome(
            BackendDecodeStatus.SUCCEEDED,
            model_fingerprint="1" * 64,
            correction=(),
            reconstructed_syndrome=(),
            failure_reason=BackendFailureReason.UPSTREAM_EXCEPTION,
        )
    with pytest.raises(ValueError, match="failed.*typed failure reason"):
        _backend_outcome(
            BackendDecodeStatus.NONCONVERGED,
            model_fingerprint="1" * 64,
        )


def test_decoder_attempt_failure_freezes_job_identity_with_its_outcome():
    from decsim.message import DecodeJob

    outcome = _backend_outcome(
        BackendDecodeStatus.NONCONVERGED,
        model_fingerprint="1" * 64,
        failure_reason=BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION,
    )
    job = DecodeJob(4, 7, 3, attempt=1)

    error = DecoderAttemptFailed(job, outcome)
    job.window_id = 99

    assert error.job_identity == (4, 7, 1)
    assert error.outcome is outcome


def test_fault_and_configuration_fingerprints_are_stable_sha256_values():
    faults = _placed(
        FaultRepresentation.PHYSICAL,
        [[1, 0]],
        [0.1, 0.2],
        [[0, 1]],
        future_flips={1: (4, 3)},
    )
    first = fault_model_fingerprint(faults)
    second = fault_model_fingerprint(faults)
    changed = fault_model_fingerprint(
        _placed(
            FaultRepresentation.PHYSICAL,
            [[1, 0]],
            [0.1, 0.3],
            [[0, 1]],
            future_flips={1: (4, 3)},
        )
    )

    assert len(first) == 64
    int(first, 16)
    assert first == second
    assert first != changed
    assert decoder_configuration_fingerprint(
        {"iterations": 20, "profile": "fixed"}
    ) == decoder_configuration_fingerprint(
        {"profile": "fixed", "iterations": 20}
    )


def test_empty_physical_model_has_exact_zero_and_nonzero_outcomes():
    faults = _placed(FaultRepresentation.PHYSICAL, np.zeros((2, 0)), [],
                     np.zeros((1, 0)))
    configuration_fingerprint = "2" * 64

    zero = empty_fault_model_outcome(
        faults,
        np.array([0, 0], dtype=np.uint8),
        decoder_configuration_fingerprint=configuration_fingerprint,
    )
    nonzero = empty_fault_model_outcome(
        faults,
        np.array([0, 1], dtype=np.uint8),
        decoder_configuration_fingerprint=configuration_fingerprint,
    )

    assert zero.status is BackendDecodeStatus.SUCCEEDED
    assert zero.physical_correction == ()
    assert zero.reconstructed_syndrome == (0, 0)
    assert nonzero.status is BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE
    assert nonzero.failure_reason is \
        BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS
    assert nonzero.physical_correction is None


def test_typed_backend_walk_uses_one_same_shot_outcome_per_window():
    first_faults = _placed(
        FaultRepresentation.PHYSICAL,
        [[1]],
        [0.1],
        [[0]],
        future_flips={0: (1,)},
    )
    second_faults = _placed(
        FaultRepresentation.PHYSICAL,
        [[1]],
        [0.1],
        [[1]],
    )
    models = [
        _window(
            detector_ids=(0,),
            physical=first_faults,
            commit_hi=1,
            defect_positions={1: (2, 0)},
        ),
        _window(detector_ids=(1,), physical=second_faults, commit_hi=2),
    ]
    calls = []

    def decode(model, syndrome):
        calls.append((model, tuple(int(bit) for bit in syndrome)))
        faults = model.require_faults(FaultRepresentation.PHYSICAL)
        return _backend_outcome(
            BackendDecodeStatus.SUCCEEDED,
            model_fingerprint=fault_model_fingerprint(faults),
            correction=np.array([1], dtype=np.uint8),
            reconstructed_syndrome=np.array([1], dtype=np.uint8),
        )

    result = decode_windowed_backend_outcomes(
        models,
        np.array([1, 0], dtype=np.uint8),
        decode,
    )

    assert [syndrome for _, syndrome in calls] == [(1,), (1,)]
    assert len(result.window_outcomes) == 2
    assert result.logical_prediction == (1,)


def test_component_correction_must_match_the_same_local_projection():
    physical = _placed(
        FaultRepresentation.PHYSICAL,
        [[1]],
        [0.1],
        [[0]],
    )
    graphlike = _placed(
        FaultRepresentation.GRAPHLIKE,
        [[1, 1]],
        [0.1, 0.1],
        [[0, 0]],
    )
    model = _window(
        detector_ids=(0,),
        physical=physical,
        graphlike=graphlike,
        link=np.array([[1], [0]], dtype=np.uint8),
    )
    outcome = BackendDecodeOutcome(
        status=BackendDecodeStatus.SUCCEEDED,
        failure_reason=None,
        physical_correction=(1,),
        component_correction=(0, 1),
        reconstructed_syndrome=(1,),
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
        fault_model_fingerprint=fault_model_fingerprint(physical),
        decoder_configuration_fingerprint="2" * 64,
    )

    with pytest.raises(ValueError, match="component correction.*projection"):
        validate_backend_outcome(
            outcome,
            model,
            physical,
            np.array([1], dtype=np.uint8),
        )


def test_typed_backend_walk_stops_at_first_failure_without_redecode():
    faults = _placed(FaultRepresentation.PHYSICAL, [[1]], [0.1], [[1]])
    models = [
        _window(detector_ids=(0,), physical=faults),
        _window(detector_ids=(1,), physical=faults),
    ]
    calls = 0

    def fail(model, syndrome):
        nonlocal calls
        calls += 1
        return _backend_outcome(
            BackendDecodeStatus.LOW_CONFIDENCE,
            model_fingerprint=fault_model_fingerprint(faults),
            failure_reason=BackendFailureReason.SEARCH_LIMIT_EXHAUSTED,
        )

    result = decode_windowed_backend_outcomes(
        models,
        np.array([1, 1], dtype=np.uint8),
        fail,
    )

    assert calls == 1
    assert len(result.window_outcomes) == 1
    assert result.logical_prediction is None


def test_attempted_shot_accounting_keeps_backend_failures_in_primary_ler():
    successful = _backend_outcome(
        BackendDecodeStatus.SUCCEEDED,
        model_fingerprint="1" * 64,
        correction=(),
        reconstructed_syndrome=(),
    )
    failed = _backend_outcome(
        BackendDecodeStatus.NONCONVERGED,
        model_fingerprint="1" * 64,
        failure_reason=BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION,
    )
    records = [
        OfflineShotRecord(2, (0,), (failed,), None),
        OfflineShotRecord(0, (0,), (successful,), (0,)),
        OfflineShotRecord(1, (0,), (successful,), (1,)),
    ]

    summary = summarize_offline_shots(records)

    assert summary.attempted_shots == 3
    assert summary.primary_failures == 2
    assert summary.primary_ler == pytest.approx(2 / 3)
    assert summary.accepted_shots == 2
    assert summary.acceptance_rate == pytest.approx(2 / 3)
    assert summary.accepted_logical_failures == 1
    assert summary.conditional_ler == pytest.approx(1 / 2)
    assert {
        record.shot_index: record.window_outcomes[0].status
        for record in records
    } == {
        0: BackendDecodeStatus.SUCCEEDED,
        1: BackendDecodeStatus.SUCCEEDED,
        2: BackendDecodeStatus.NONCONVERGED,
    }
    assert summarize_offline_shots(reversed(records)) == summary


def test_conditional_ler_is_absent_when_no_shot_is_accepted():
    failed = _backend_outcome(
        BackendDecodeStatus.LOW_CONFIDENCE,
        model_fingerprint="1" * 64,
        failure_reason=BackendFailureReason.SEARCH_LIMIT_EXHAUSTED,
    )

    summary = summarize_offline_shots([
        OfflineShotRecord(0, (0,), (failed,), None),
    ])

    assert summary.primary_ler == 1
    assert summary.acceptance_rate == 0
    assert summary.conditional_ler is None


def test_fault_conversion_merges_duplicates_with_the_standard_rule():
    """p (+) q = p(1-q) + q(1-p), the BeliefMatching/QUITS convention."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.2) D0 D1
        error(0.3) D1 D2 L0
    """)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    assert len(det_sets) == 2
    i = det_sets.index((0, 1))
    assert priors[i] == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)
    j = det_sets.index((1, 2))
    assert obs_sets[j] == (0,)


def test_composite_errors_split_into_matchable_components():
    """Stim's `^`-separated decompositions become separate <=2-detector faults, each
    carrying the parent probability (PyMatching's own convention)."""
    dem = stim.DetectorErrorModel("error(0.25) D0 D1 ^ D2 D3 L0")
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    assert sorted(det_sets) == [(0, 1), (2, 3)]
    assert priors == [0.25, 0.25]
    # and on the real circuit, EVERY column is matchable
    circuit = _memory_circuit()
    sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    assert max(len(s) for s in sets) <= 2


@pytest.mark.parametrize(
    ("instruction", "expected_detectors", "expected_logicals"),
    [
        ("error(0.25) D0 D0", (), ()),
        ("error(0.25) D0 ^ D0", (), ()),
        ("error(0.25) D0 L0 ^ D0 L0", (), ()),
        ("error(0.25) D0 D0 D1 L0", ((1,),), ((0,),)),
    ],
)
def test_canonical_error_instruction_uses_instruction_wide_xor_identity(
    instruction, expected_detectors, expected_logicals,
):
    records = canonical_error_instructions(
        stim.DetectorErrorModel(instruction)
    )

    assert tuple(
        component.detectors
        for record in records
        for component in record.components
    ) == expected_detectors
    assert tuple(
        component.logical_observables
        for record in records
        for component in record.components
    ) == expected_logicals


@pytest.mark.parametrize(
    "instruction",
    [
        "error(0.25) L0",
        "error(0.25) D0 D0 L0",
        "error(0.25) D0 L0 ^ D0",
    ],
)
def test_canonical_error_instruction_rejects_detectorless_logical_mechanism(
    instruction,
):
    dem = stim.DetectorErrorModel(instruction)

    with pytest.raises(
        ValueError,
        match="error 0.*detectorless logical",
    ):
        canonical_error_instructions(dem)


def test_every_raw_graph_consumer_shares_instruction_wide_validation():
    from decsim.soft_output.complementary import dem_to_matrices

    malformed = stim.DetectorErrorModel(
        "error(0.25) D0 L0 ^ D0"
    )
    consumers = (
        detector_error_model_to_faults,
        detector_error_model_to_faults_bm,
        dem_to_matrices,
    )

    for consume in consumers:
        with pytest.raises(ValueError, match="detectorless logical"):
            consume(malformed)


def test_every_placed_construction_path_rejects_detectorless_logical_fault():
    from decsim.belief_matching_decoder import BeliefMatchingDecoder
    from decsim.belief_matching_decoder import belief_matching_window_decoder
    from decsim.bposd_decoder import BPOSDDecoder
    from decsim.bposd_decoder import bposd_window_decoder
    from decsim.message import DecodeJob
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.mwpm_decoder import matching_window_decoder
    from decsim.soft_output import ComplementaryGapMetric
    from decsim.union_find_decoder import UnionFindDecoder

    class ZeroLatency:
        def latency(self, job):
            return 0

    malformed_graphlike = _placed(
        FaultRepresentation.GRAPHLIKE,
        np.zeros((0, 1), dtype=np.uint8),
        np.array([0.25]),
        np.ones((1, 1), dtype=np.uint8),
    )
    malformed_physical = _placed(
        FaultRepresentation.PHYSICAL,
        np.zeros((0, 1), dtype=np.uint8),
        np.array([0.25]),
        np.ones((1, 1), dtype=np.uint8),
    )
    model = _window(
        detector_ids=(),
        commit_hi=0,
        graphlike=malformed_graphlike,
        physical=malformed_physical,
        link=np.ones((1, 1), dtype=np.uint8),
    )
    job = DecodeJob(
        op_id=7,
        window_id=3,
        n_rounds=1,
        dem=model,
        label="malformed W3",
    )
    consumers = (
        lambda: PyMatchingDecoder(ZeroLatency()).decode(job),
        lambda: BPOSDDecoder(ZeroLatency()).decode(job),
        lambda: BeliefMatchingDecoder(ZeroLatency()).decode(job),
        lambda: UnionFindDecoder(ZeroLatency()).decode(job),
        lambda: ComplementaryGapMetric.from_window_model(model),
        lambda: matching_window_decoder()(model, np.zeros(0, dtype=np.uint8)),
        lambda: bposd_window_decoder()(model, np.zeros(0, dtype=np.uint8)),
        lambda: belief_matching_window_decoder()(
            model,
            np.zeros(0, dtype=np.uint8),
        ),
        lambda: ComplementaryGapMetric(
            malformed_graphlike.check,
            malformed_graphlike.observables,
            np.ones(1),
        ),
    )

    for consume in consumers:
        with pytest.raises(
            ValueError,
            match="(column 0|edge 0).*detectorless logical",
        ):
            consume()


def test_bposd_retains_general_placed_hyperedges():
    pytest.importorskip("ldpc")
    from decsim.bposd_decoder import bposd_window_decoder

    model = _window(
        detector_ids=(0, 1, 2),
        commit_hi=1,
        physical=_placed(
            FaultRepresentation.PHYSICAL,
            np.ones((3, 1), dtype=np.uint8),
            np.array([0.25]),
            np.zeros((1, 1), dtype=np.uint8),
        ),
    )

    selected = bposd_window_decoder()(
        model,
        np.zeros(3, dtype=np.uint8),
    )

    assert np.array_equal(selected, np.zeros(1, dtype=np.uint8))


def test_every_placed_graph_path_rejects_a_detector_hyperedge():
    from decsim.belief_matching_decoder import (
        BeliefMatchingDecoder,
        belief_matching_window_decoder,
    )
    from decsim.message import DecodeJob, SyndromePayload
    from decsim.mwpm_decoder import PyMatchingDecoder, matching_window_decoder
    from decsim.soft_output import ComplementaryGapMetric
    from decsim.union_find_decoder import UnionFindDecoder

    class ZeroLatency:
        def latency(self, job):
            return 0

    hyperedge_graphlike = _placed(
        FaultRepresentation.GRAPHLIKE,
        np.ones((3, 1), dtype=np.uint8),
        np.array([0.25]),
        np.zeros((1, 1), dtype=np.uint8),
    )
    physical = _placed(
        FaultRepresentation.PHYSICAL,
        np.ones((3, 1), dtype=np.uint8),
        np.array([0.25]),
        np.zeros((1, 1), dtype=np.uint8),
    )
    model = _window(
        detector_ids=(0, 1, 2),
        commit_hi=1,
        graphlike=hyperedge_graphlike,
        physical=physical,
        link=np.ones((1, 1), dtype=np.uint8),
    )
    job = DecodeJob(
        op_id=7,
        window_id=3,
        n_rounds=1,
        dem=model,
        payloads=[
            SyndromePayload(
                operation_id=7,
                patch_id=0,
                round_index=1,
                bits=np.zeros(3, dtype=np.uint8),
            )
        ],
        label="hyperedge W3",
    )
    consumers = (
        lambda: PyMatchingDecoder(ZeroLatency()).decode(job),
        lambda: BeliefMatchingDecoder(ZeroLatency()).decode(job),
        lambda: UnionFindDecoder(ZeroLatency()).decode(job),
        lambda: matching_window_decoder()(model, np.zeros(3, dtype=np.uint8)),
        lambda: belief_matching_window_decoder()(
            model,
            np.zeros(3, dtype=np.uint8),
        ),
        lambda: ComplementaryGapMetric(
            hyperedge_graphlike.check,
            hyperedge_graphlike.observables,
            np.ones(1),
        ),
    )

    for consume in consumers:
        with pytest.raises(
            ValueError,
            match="detector hyperedge.*graphlike decoder",
        ):
            consume()


def test_belief_placed_validation_separates_physical_and_component_domains():
    from decsim.detector_error_model import validate_belief_matching_matrices

    check = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    obs = np.zeros((1, 2), dtype=np.uint8)
    h_check = np.ones((3, 1), dtype=np.uint8)
    h2e = np.ones((2, 1), dtype=np.uint8)

    validate_belief_matching_matrices(
        check,
        obs,
        h_check,
        np.array([0.25]),
        h2e,
        location="valid physical hyperedge",
    )


def test_belief_placed_validation_rejects_detectorless_logical_aggregate():
    from decsim.detector_error_model import validate_belief_matching_matrices

    with pytest.raises(
        ValueError,
        match="physical column 0.*detectorless logical",
    ):
        validate_belief_matching_matrices(
            np.array([[1, 1]], dtype=np.uint8),
            np.array([[0, 1]], dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            np.array([0.25]),
            np.ones((2, 1), dtype=np.uint8),
            location="lost aggregate",
        )


@pytest.mark.parametrize("prior", [0.0, 0.5, 1.0])
def test_belief_inner_accepts_inclusive_physical_prior_range(prior):
    from decsim.belief_matching_decoder import belief_matching_window_decoder

    model = _window(
        detector_ids=(0,),
        commit_hi=1,
        graphlike=_placed(
            FaultRepresentation.GRAPHLIKE,
            np.ones((1, 1), dtype=np.uint8),
            np.array([0.25]),
            np.zeros((1, 1), dtype=np.uint8),
        ),
        physical=_placed(
            FaultRepresentation.PHYSICAL,
            np.ones((1, 1), dtype=np.uint8),
            np.array([prior]),
            np.zeros((1, 1), dtype=np.uint8),
        ),
        link=np.ones((1, 1), dtype=np.uint8),
    )

    selected = belief_matching_window_decoder()(
        model,
        np.zeros(1, dtype=np.uint8),
    )

    assert selected.shape == (1,)


@pytest.mark.parametrize(
    ("priors", "message"),
    [
        (np.array([], dtype=float), "0 physical priors for 1"),
        (np.array([-0.1]), "inclusive range"),
        (np.array([1.1]), "inclusive range"),
        (np.array([np.inf]), "must be finite"),
        (np.array([-np.inf]), "must be finite"),
        (np.array([np.nan]), "must be finite"),
    ],
)
def test_belief_inner_rejects_invalid_physical_priors(priors, message):
    from decsim.belief_matching_decoder import belief_matching_window_decoder

    model = _window(
        detector_ids=(0,),
        commit_hi=1,
        graphlike=_placed(
            FaultRepresentation.GRAPHLIKE,
            np.ones((1, 1), dtype=np.uint8),
            np.array([0.25]),
            np.zeros((1, 1), dtype=np.uint8),
        ),
        physical=_placed(
            FaultRepresentation.PHYSICAL,
            np.ones((1, 1), dtype=np.uint8),
            priors,
            np.zeros((1, 1), dtype=np.uint8),
        ),
        link=np.ones((1, 1), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match=message):
        belief_matching_window_decoder()(
            model,
            np.zeros(1, dtype=np.uint8),
        )


def _belief_converter_matrices(dem):
    (
        edge_detectors,
        edge_logicals,
        _edge_priors,
        physical_detectors,
        physical_priors,
        physical_to_edge,
    ) = detector_error_model_to_faults_bm(dem)
    check = np.zeros(
        (dem.num_detectors, len(edge_detectors)),
        dtype=np.uint8,
    )
    obs = np.zeros(
        (dem.num_observables, len(edge_logicals)),
        dtype=np.uint8,
    )
    h_check = np.zeros(
        (dem.num_detectors, len(physical_detectors)),
        dtype=np.uint8,
    )
    for column, detector_ids in enumerate(edge_detectors):
        check[list(detector_ids), column] = 1
    for column, logical_ids in enumerate(edge_logicals):
        obs[list(logical_ids), column] = 1
    for column, detector_ids in enumerate(physical_detectors):
        h_check[list(detector_ids), column] = 1
    return check, obs, h_check, np.asarray(physical_priors), physical_to_edge


def test_belief_first_decomposition_component_multiplicity_is_gf2():
    for instruction, expected_logical in (
        ("error(0.1) D0 ^ D0 ^ D1", []),
        ("error(0.1) D0 L0 ^ D0 ^ D1", [[1]]),
    ):
        dem = stim.DetectorErrorModel(instruction)
        check, obs, h_check, _priors, h2e = _belief_converter_matrices(dem)

        assert np.array_equal((check @ h2e) % 2, h_check)
        assert ((obs @ h2e) % 2).tolist() == expected_logical
        assert h_check.tolist() == [[0], [1]]


def test_belief_converter_matches_upstream_on_unambiguous_identities():
    from beliefmatching.belief_matching import (
        detector_error_model_to_check_matrices,
    )

    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.2) D1 D2 ^ D3 L0
        """
    )
    check, obs, h_check, priors, h2e = _belief_converter_matrices(dem)
    upstream = detector_error_model_to_check_matrices(dem)

    assert np.array_equal(check, upstream.edge_check_matrix.toarray())
    assert np.array_equal(obs, upstream.edge_observables_matrix.toarray())
    assert np.array_equal(h_check, upstream.check_matrix.toarray())
    assert np.array_equal(
        (obs @ h2e) % 2,
        upstream.observables_matrix.toarray(),
    )
    assert np.array_equal(priors, upstream.priors)
    assert np.array_equal(h2e, upstream.hyperedge_to_edge_matrix.toarray())


@pytest.mark.parametrize(
    "fixture_name",
    [
        "rsc-d3-r6-p0.005.stim",
        "rsc-d5-r10-p0.005.stim",
    ],
)
def test_belief_converter_matches_upstream_on_frozen_surface_fixtures(
    fixture_name,
):
    from beliefmatching.belief_matching import (
        detector_error_model_to_check_matrices,
    )

    fixture_path = pathlib.Path(__file__).parent / "data" / fixture_name
    circuit = stim.Circuit(fixture_path.read_text())
    dem = circuit.detector_error_model(decompose_errors=True)
    check, obs, h_check, priors, h2e = _belief_converter_matrices(dem)
    upstream = detector_error_model_to_check_matrices(dem)

    upstream_check = upstream.check_matrix.toarray()
    upstream_edge_check = upstream.edge_check_matrix.toarray()
    upstream_edge_observables = upstream.edge_observables_matrix.toarray()
    upstream_h2e = upstream.hyperedge_to_edge_matrix.toarray()
    assert np.array_equal(
        (upstream_edge_check @ upstream_h2e) % 2,
        upstream_check,
    )
    assert np.array_equal(check, upstream_edge_check)
    assert np.array_equal(obs, upstream_edge_observables)
    assert np.array_equal(h_check, upstream_check)
    assert np.array_equal(priors, upstream.priors)
    assert np.array_equal(h2e, upstream_h2e)
    assert np.array_equal(
        (obs @ h2e) % 2,
        upstream.observables_matrix.toarray(),
    )


@pytest.mark.parametrize(
    "instructions",
    [
        "error(0.1) D0\nerror(0.2) D0 L0",
        "error(0.2) D0 L0\nerror(0.1) D0",
    ],
)
def test_belief_physical_merge_key_includes_logical_identity(instructions):
    dem = stim.DetectorErrorModel(instructions)
    check, obs, h_check, priors, h2e = _belief_converter_matrices(dem)

    assert h_check.tolist() == [[1, 1]]
    assert sorted(priors.tolist()) == [0.1, 0.2]
    assert {
        tuple(column)
        for column in ((obs @ h2e) % 2).T.tolist()
    } == {(0,), (1,)}
    assert np.array_equal((check @ h2e) % 2, h_check)


def test_belief_repeated_physical_identity_keeps_first_decomposition():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D1
        error(0.2) D0 ^ D1
        """
    )
    edge_detectors, _, _, physical_detectors, priors, h2e = (
        detector_error_model_to_faults_bm(dem)
    )

    assert physical_detectors == [(0, 1)]
    assert priors == pytest.approx([0.26])
    selected_edges = {
        edge_detectors[index]
        for index in np.nonzero(h2e[:, 0])[0]
    }
    assert selected_edges == {(0, 1)}


def test_belief_window_projection_preserves_physical_component_identity():
    class DemBackedCircuit:
        num_observables = 0

        def detector_error_model(self, *, decompose_errors):
            return stim.DetectorErrorModel(
                "error(0.1) D0 D1 ^ D1 D2"
                if decompose_errors
                else "error(0.1) D0 D2"
            )

        def get_detector_coordinates(self):
            return {
                detector_id: [detector_id, 0, 0]
                for detector_id in range(3)
            }

    exact_model = build_window_error_models(
        DemBackedCircuit(),
        [(1, 1, 1)],
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
    )[0]
    graphlike = exact_model.graphlike_faults
    physical = exact_model.physical_faults
    assert physical.check.tolist() == [[1], [0], [1]]
    assert np.array_equal(
        (
            graphlike.check
            @ exact_model.physical_to_graphlike_detector_projection
        ) % 2,
        physical.check,
    )

    circuit = _memory_circuit(rounds=6)
    models = build_window_error_models(
        circuit,
        _plan(circuit),
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
    )

    for model in models:
        graphlike = model.graphlike_faults
        physical = model.physical_faults
        assert np.array_equal(
            (
                graphlike.check
                @ model.physical_to_graphlike_detector_projection
            ) % 2,
            physical.check,
        )

    first = models[0]
    graphlike = first.graphlike_faults
    physical = first.physical_faults
    projected_observables = (
        graphlike.observables
        @ first.physical_to_graphlike_detector_projection
    ) % 2
    observable_residuals = np.flatnonzero(
        np.any(projected_observables != physical.observables, axis=0)
    )

    # A real d=3 boundary contains five physical mechanisms whose omitted
    # beyond-window graph component carries a logical tag. The local link is
    # therefore detector-exact but cannot be observable-exact without adding
    # a detectorless logical graph column.
    assert observable_residuals.size == 5
    assert not np.any(
        (graphlike.check.sum(axis=0) == 0)
        & (graphlike.observables.sum(axis=0) != 0)
    )


def test_every_fault_is_owned_by_exactly_one_window():
    """The commit partition: each fault decided once, none lost (Skoric's rule)."""
    circuit = _memory_circuit()
    models = build_window_error_models(
        circuit, _plan(circuit),
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    det_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    owned_total = sum(
        int(model.graphlike_faults.owned.sum()) for model in models
    )
    assert owned_total == len(det_sets)


def test_interior_windows_have_open_time_boundaries():
    """A fault straddling a window's edge appears as a single-detector column -- the
    boundary edge Tan's imaginary detectors formalize."""
    circuit = _memory_circuit()
    models = build_window_error_models(
        circuit, _plan(circuit),
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    interior = models[1].graphlike_faults
    assert (interior.check.sum(axis=0) == 1).any()


def test_single_crossing_fault_round_trips_exactly():
    """THE mechanics test: a single fault that crosses a commit boundary must be
    committed by its owning window, hand its beyond-commit flips forward as
    artificial defects, and the windowed pass must reproduce the fault's observable
    flips exactly -- with every handed-forward defect consumed."""
    circuit = _memory_circuit()
    models = build_window_error_models(
        circuit, _plan(circuit),
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    det_sets, obs_sets, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    w0 = models[0]
    faults0 = w0.graphlike_faults
    crossing_cols = [
        column for column in faults0.future_flips if faults0.owned[column]
    ]
    assert crossing_cols, "no boundary-crossing fault found in window 0"
    decode = matching_window_decoder()
    n_dets = circuit.num_detectors
    checked = 0
    for col in crossing_cols[:5]:
        # rebuild the GLOBAL detection events of exactly this fault
        in_window = set(
            np.asarray(w0.detector_ids)[faults0.check[:, col] > 0]
        )
        beyond = set(faults0.future_flips[col])
        events = np.zeros(n_dets, dtype=np.uint8)
        for d in in_window | beyond:
            events[d] = 1
        predicted = decode_windowed(
            models, events, decode,
            selected_fault_representation=FaultRepresentation.GRAPHLIKE,
        )
        # which fault is this, globally? find it by its full detector set
        full = tuple(sorted(in_window | beyond))
        expected = np.zeros(circuit.num_observables, dtype=np.uint8)
        for o in obs_sets[det_sets.index(full)]:
            expected[o] = 1
        assert (predicted == expected).all(), f"fault {full} mis-roundtripped"
        checked += 1
    assert checked > 0


def test_windowed_accuracy_matches_global_decoding():
    """Skoric Appendix C, the published anchor: with buffer = d, sliding-window
    decoding shows 'no noticeable increase in logical error rate' over decoding the
    whole history at once. Fixed seed -> deterministic counts."""
    pymatching = pytest.importorskip("pymatching")
    circuit = _memory_circuit(d=3, rounds=12, p=0.003)
    models = build_window_error_models(
        circuit, _plan(circuit),
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    shots = 2000
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    global_pred = global_m.decode_batch(dets)
    decode = matching_window_decoder()
    windowed_pred = np.array([decode_windowed(
        models, dets[i], decode,
        selected_fault_representation=FaultRepresentation.GRAPHLIKE)
                              for i in range(shots)])
    agree = float((windowed_pred == global_pred).all(axis=1).mean())
    ler_global = float((global_pred != obs).any(axis=1).mean())
    ler_windowed = float((windowed_pred != obs).any(axis=1).mean())
    assert agree > 0.97, f"windowed disagrees with global too often: {agree}"
    # 'no noticeable increase': allow binomial wiggle on 2000 shots, nothing more
    assert ler_windowed <= ler_global + 2 * (ler_global / shots) ** 0.5 + 0.005, \
        f"windowed LER {ler_windowed} vs global {ler_global}"


def test_empty_syndrome_decodes_to_no_correction_and_no_flip():
    """Validation-matrix row 6, made explicit: an all-zero detection-event
    vector must produce a zero correction in every window, a zero observable
    from the windowed pipeline, and a zero prediction from global MWPM."""
    pymatching = pytest.importorskip("pymatching")
    circuit = _memory_circuit(d=3, rounds=9, p=0.003)
    models = build_window_error_models(
        circuit, _plan(circuit),
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    decode = matching_window_decoder()
    empty = np.zeros(circuit.num_detectors, dtype=np.uint8)
    assert int(decode_windowed(
        models, empty, decode,
        selected_fault_representation=FaultRepresentation.GRAPHLIKE,
    ).sum()) == 0
    for m in models:
        assert int(decode(m, np.zeros(len(m.detector_ids), dtype=np.uint8))
                   .sum()) == 0
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    assert int(global_m.decode(empty).sum()) == 0


def test_boundary_priors_are_clipped_not_infinite():
    """Regression (found 2026-07-02, validation-matrix row 24): a prior of
    exactly 1.0 -- which any deterministic injected fault produces -- made the
    edge weight -inf and crashed pymatching with 'maximum absolute edge weight
    exceeded'; p=0 silently produced +inf edges. The decoders now clip priors
    to [1e-12, 1-1e-12]. A p=1 fault must decode as 'always there': its own
    syndrome yields exactly its observable effect."""
    import dataclasses
    circuit = _memory_circuit(d=3, rounds=6, p=0.003)
    models = build_window_error_models(
        circuit,
        [(1, 3, 6), (4, 6, 6)],
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    m0 = models[0]
    faults0 = m0.graphlike_faults
    empty = np.zeros(len(m0.detector_ids), dtype=np.uint8)
    # p=0: an impossible fault is never selected
    priors = np.array(faults0.priors, dtype=float)
    priors[0] = 0.0
    selected = matching_window_decoder()(
        dataclasses.replace(
            m0,
            graphlike_faults=dataclasses.replace(faults0, priors=priors),
        ), empty)
    assert selected[0] == 0 and int(selected.sum()) == 0
    # p=1: negative weight -- on an EMPTY syndrome MWPM correctly infers the
    # fault happened anyway, compensated by partner faults (this used to be
    # the -inf crash; now it is well-defined negative-weight matching)
    priors = np.array(faults0.priors, dtype=float)
    priors[0] = 1.0
    selected = matching_window_decoder()(
        dataclasses.replace(
            m0,
            graphlike_faults=dataclasses.replace(faults0, priors=priors),
        ), empty)
    assert selected[0] == 1
    # the p=1 fault's own syndrome must select exactly that fault
    priors = np.array(faults0.priors, dtype=float)
    priors[0] = 1.0
    faults2 = dataclasses.replace(faults0, priors=priors)
    m2 = dataclasses.replace(m0, graphlike_faults=faults2)
    syndrome = np.zeros(len(m2.detector_ids), dtype=np.uint8)
    syndrome[np.flatnonzero(faults2.check[:, 0])] = 1
    selected = matching_window_decoder()(m2, syndrome)
    assert selected[0] == 1 and int(selected.sum()) == 1, \
        "the clipped p=1 fault must be selected to explain its own syndrome"
    # the runtime adapter's graph-construction site is protected identically
    from decsim.mwpm_decoder import PyMatchingDecoder

    class _NullLatency:
        def latency(self, job):
            return 1

    assert PyMatchingDecoder(_NullLatency())._matching_for_model(faults2) is not None
    # malformed priors are BUGS, not degenerate inputs: they must raise, not
    # be silently coerced into probabilities
    for bad in (-0.1, 1.5, float("nan")):
        priors = np.array(faults0.priors, dtype=float)
        priors[0] = bad
        broken_faults = dataclasses.replace(faults0, priors=priors)
        broken = dataclasses.replace(m0, graphlike_faults=broken_faults)
        with pytest.raises(ValueError, match="priors"):
            matching_window_decoder()(broken, empty)
        with pytest.raises(ValueError, match="priors"):
            PyMatchingDecoder(_NullLatency())._matching_for_model(
                broken_faults
            )


def test_sub_d_tail_window_measurably_degrades_accuracy():
    """Characterization pin for the 2026-07-02 finding (validation-matrix rows
    11/19): a final commit window shorter than d decodes with too little
    history and measurably degrades windowed accuracy, while absorbing the
    tail into the last full window tracks global decoding almost exactly.

    This is NOT the desired end state -- it documents a real, currently
    shipped behavior (SlidingWindowScheme docstring carries the caveat; the
    timing goldens pin its layout). If a change makes the tail plan match
    global, this test should fail and be UPDATED, loudly, not silently.

    Fixed seed -> deterministic counts; margins allow pymatching tie-breaking
    drift, not behavioral change."""
    pymatching = pytest.importorskip("pymatching")
    d, rounds, p, shots = 3, 10, 0.008, 4000
    circuit = _memory_circuit(d=d, rounds=rounds, p=p)

    def _plan(absorb):
        plan, lo = [], 1
        while (rounds - lo + 1 >= 2 * d) if absorb else (lo + d - 1 <= rounds):
            plan.append((lo, lo + d - 1, min(lo + d - 1 + d, rounds)))
            lo += d
        if lo <= rounds:
            plan.append((lo, rounds, rounds))   # absorbed tail / short tail
        return plan

    assert _plan(True)[-1] == (7, 10, 10)       # 4-round absorbed final commit
    assert _plan(False)[-1] == (10, 10, 10)     # 1-round sub-d tail window
    # and the SHIPPED scheme really does produce a sub-d tail here: for these
    # 11 detector layers its final window commits only 2 (< d) layers. If
    # this assertion fails, the shipped plan changed -- re-evaluate the whole
    # caveat, not just this test.
    shipped = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0,
            11,
            commit_round_count=d,
            buffer_round_count=d,
        ).windows
    ]
    assert shipped[-1] == (10, 11, 14)
    assert shipped[-1][1] - shipped[-1][0] + 1 < d

    dets, obs = circuit.compile_detector_sampler(seed=3).sample(
        shots, separate_observables=True)
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    g_fail = matcher.decode_batch(dets)[:, 0] != obs[:, 0]

    fails = {}
    for absorb in (True, False):
        models = build_window_error_models(
            circuit,
            _plan(absorb),
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
        )
        decode = matching_window_decoder()
        fails[absorb] = np.array([
            decode_windowed(
                models, dets[k], decode,
                selected_fault_representation=FaultRepresentation.GRAPHLIKE,
            )[0] != obs[k, 0]
            for k in range(shots)])

    # absorbed plan ~= global (measured 2 vs 1 discordant shots at this seed)
    assert int((fails[True] & ~g_fail).sum()) <= 5
    # sub-d tail is MEASURABLY worse than the absorbed plan on paired shots
    # (measured 16 vs 4); if this stops holding, the tail behavior changed
    worse = int((fails[False] & ~fails[True]).sum())
    better = int((fails[True] & ~fails[False]).sum())
    assert worse >= 2 * better + 4, (
        f"sub-d tail no longer degrades accuracy (worse={worse}, "
        f"better={better}) -- shipped-scheme caveat and docs need updating")


def test_parallel_two_sided_windows_match_global_decoding():
    """Two-sided parallel A/B windows (Skoric sec. III.C / Tan Eq. S10, w = s + 2b)
    carry a lookback buffer, so each window must decode its raw syndrome independently
    and commit only its core -- forward-passing artificial defects (the sliding-window
    technique) double-counts the boundary error and inflates the LER. This pins the
    two-sided path against global decoding for d=5, where the bug was glaring (windowed
    LER ~0.10 vs global ~0.06 before the fix). Fixed seed -> deterministic counts."""
    pymatching = pytest.importorskip("pymatching")
    d, rounds = 5, 20
    circuit = _memory_circuit(d=d, rounds=rounds, p=0.005)
    n_layers = 1 + max(int(c[-1]) for c in circuit.get_detector_coordinates().values())
    plan = [
        (
            window.buffer_lo,
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
        )
        for window in ParallelWindowScheme().plan_operation(
            0,
            n_layers,
            commit_round_count=d,
            buffer_round_count=d,
        ).windows
    ]
    assert any(len(w) == 4 and w[0] < w[1] for w in plan), "expected two-sided windows"
    models = build_window_error_models(
        circuit,
        plan,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    assert any(m.has_leading_buffer for m in models)
    shots = 2000
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    global_pred = global_m.decode_batch(dets)
    decode = matching_window_decoder()
    windowed_pred = np.array([decode_windowed(
        models, dets[i], decode,
        selected_fault_representation=FaultRepresentation.GRAPHLIKE)
                              for i in range(shots)])
    disagree = int((windowed_pred != global_pred).any(axis=1).sum())
    g_fail = (global_pred != obs).any(axis=1)
    w_fail = (windowed_pred != obs).any(axis=1)
    only_w = int((w_fail & ~g_fail).sum())
    only_g = int((g_fail & ~w_fail).sum())
    # paired-shot pins (fixed seed; measured 1 disagreement, only_w=1,
    # only_g=0 -- margins allow pymatching tie-break drift, not regression)
    assert disagree <= 5, \
        f"two-sided windowed drifted from global: {disagree}/{shots} shots differ"
    assert only_w - only_g <= 4, \
        f"windowed strictly worse on paired shots: only_w={only_w} only_g={only_g}"


# ---------------------------------------------------------------------------------
# Bivariate-bicycle code: the slicing must be code-agnostic, not surface-only.
# ---------------------------------------------------------------------------------

_BB_FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / \
    "bb72_12_6_p003_r10.stim"
_BB_CHECKS_PER_ROUND = 36     # the [[72,12,6]] code's Z checks, one detector layer each
# commit 3 / buffer 3 sliding plan over the fixture's 12 detector layers
# (10 noisy rounds + zeroth + final layer), scheme-style 1-based rounds
_BB_PLAN = [(1, 3, 6), (4, 6, 9), (7, 9, 12), (10, 12, 12)]


def _bb_circuit():
    return stim.Circuit.from_file(str(_BB_FIXTURE))


def _bb_models(circuit):
    """QUITS circuits attach no detector coordinates; detectors are emitted in time
    order, one layer of 36 per round, so round = id // 36 + 1."""
    rounds = {d: d // _BB_CHECKS_PER_ROUND + 1 for d in range(circuit.num_detectors)}
    return build_window_error_models(
        circuit,
        _BB_PLAN,
        detector_rounds=rounds,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )


def test_bb_circuit_without_coordinates_requires_explicit_rounds():
    """Coordinate-less detectors must fail loudly, not be silently mis-binned."""
    circuit = _bb_circuit()
    with pytest.raises(ValueError, match="detector_rounds"):
        build_window_error_models(
            circuit,
            _BB_PLAN,
            fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_bb_faults_are_not_matchable_and_partition_exactly():
    """The BB DEM is genuinely non-graphlike (faults flip up to 6 detectors), and the
    ownership partition still holds: every fault committed by exactly one window."""
    circuit = _bb_circuit()
    det_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=False))
    assert max(len(s) for s in det_sets) > 2          # matching would be unsound here
    models = _bb_models(circuit)
    assert sum(
        int(model.physical_faults.owned.sum()) for model in models
    ) == len(det_sets)
    # interior windows hand artificial defects forward; the last window closes
    assert all(
        len(model.physical_faults.future_flips) > 0
        for model in models[:-1]
    )
    assert models[-1].physical_faults.future_flips == {}


def test_bb_windowed_accuracy_matches_global_decoding():
    """The Skoric App C anchor, BB edition: windowed BP-OSD tracks whole-history
    BP-OSD. Unlike exact matching, BP-OSD is approximate, so windowed and global may
    legitimately differ on a few shots (QUITS reports the same character); we pin
    high agreement and LER within binomial wiggle. Fixed seed -> deterministic."""
    pytest.importorskip("ldpc")
    sp = pytest.importorskip("scipy.sparse")
    from ldpc import BpOsdDecoder
    from decsim.bposd_decoder import bposd_window_decoder

    circuit = _bb_circuit()
    models = _bb_models(circuit)
    dem = circuit.detector_error_model(decompose_errors=False)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    H = np.zeros((circuit.num_detectors, len(det_sets)), dtype=np.uint8)
    O = np.zeros((circuit.num_observables, len(det_sets)), dtype=np.uint8)
    for j, (ds, os_) in enumerate(zip(det_sets, obs_sets)):
        for d in ds:
            H[d, j] = 1
        for o in os_:
            O[o, j] = 1
    global_dec = BpOsdDecoder(sp.csr_matrix(H), error_channel=list(priors),
                              max_iter=2, bp_method="product_sum",
                              schedule="serial", osd_method="osd_cs", osd_order=0)
    shots = 300
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    inner = bposd_window_decoder()
    agree = ler_w = ler_g = 0
    for s in range(shots):
        predicted_w = decode_windowed(
            models, dets[s], inner,
            selected_fault_representation=FaultRepresentation.PHYSICAL,
        )
        predicted_g = (O @ global_dec.decode(dets[s].astype(np.uint8))) % 2
        actual = obs[s].astype(np.uint8)
        agree += int(np.array_equal(predicted_w, predicted_g))
        ler_w += int(not np.array_equal(predicted_w, actual))
        ler_g += int(not np.array_equal(predicted_g, actual))
    agree /= shots
    ler_w /= shots
    ler_g /= shots
    assert agree > 0.9, f"windowed disagrees with global too often: {agree}"
    assert ler_w <= ler_g + 2 * (ler_g * (1 - ler_g) / shots) ** 0.5 + 0.005, \
        f"windowed LER {ler_w} vs global {ler_g}"
