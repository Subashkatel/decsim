"""The relay_bp row against relay-bp's RelayDecoderF32 on the same model.

The reference is relay-bp 0.2.2 called directly, outside decsim, on the
window's physical check matrix and priors with the row's Settings as its
arguments and the gamma table the row draws from its seed. The wheel is
the bb-decoders extra; the reference tests skip until it is installed.
The yaml refusals need no wheel.
"""

import numpy
import pytest
import scipy.sparse

import decsim.config as config
import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SEED = 5
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
# the paper's surface code values, Relay-BP-5 (Mueller et al. 2506.01779
# lines 307, 332 and 343), with fewer legs so the test is quick
SURFACE = relay.RelayBeliefPropagationDecoder.Settings(
    gamma0=0.35,
    gamma_interval=(-0.254, 0.985),
    relay_set_count=20,
    converged_solution_count=5,
)


def _seeded_row(settings):
    """The row with its gamma table seeded as a run seeds it."""
    row = relay.RelayBeliefPropagationDecoder(settings=settings)
    reservation = row.window_decoder.reserve_run_seed(SEED)
    row.window_decoder.commit_run_seed(reservation)
    return row


def _reference_decoder(relay_bp, faults, settings):
    """relay-bp built directly with the settings' arguments."""
    column_count = faults.check.shape[1]
    bit_generator = numpy.random.PCG64(SEED)
    generator = numpy.random.Generator(bit_generator)
    low, high = settings.gamma_interval
    shape = (settings.relay_set_count, column_count)
    gamma_table = generator.uniform(low, high, size=shape)
    check = scipy.sparse.csr_matrix(faults.check)
    priors = numpy.asarray(faults.priors, dtype=numpy.float64)
    return relay_bp.RelayDecoderF32(
        check,
        priors,
        alpha=settings.alpha,
        alpha_iteration_scaling_factor=settings.alpha_iteration_scaling_factor,
        gamma0=settings.gamma0,
        pre_iter=settings.pre_iterations,
        num_sets=settings.relay_set_count,
        set_max_iter=settings.iterations_per_set,
        gamma_dist_interval=settings.gamma_interval,
        explicit_gammas=gamma_table,
        stop_nconv=settings.converged_solution_count,
        stopping_criterion="nconv",
        logging=False,
        seed=SEED,
    )


def test_the_row_returns_relay_bps_own_correction_property():
    relay_bp = pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 10, 3)
    row = _seeded_row(SURFACE)
    reference = _reference_decoder(relay_bp, physical, SURFACE)
    for shot in detection_events:
        syndrome = windows.row_syndrome(model, shot)
        detailed = reference.decode_detailed(syndrome)
        expected = list(detailed.decoding)
        job = windows.job_for(model, shot)
        result = row.decode(job)
        assert result.correction.tolist() == expected


def test_the_row_reports_relay_bps_own_iteration_count_property():
    """A measured device time law reads the count relay-bp itself reports.

    decode_detailed's `iterations` is the count the relay-bp package
    returns for one syndrome (relay-bp's DecodeResult), and the result
    carries it unchanged.
    """
    relay_bp = pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 10, 3)
    row = _seeded_row(SURFACE)
    reference = _reference_decoder(relay_bp, physical, SURFACE)
    for shot in detection_events:
        syndrome = windows.row_syndrome(model, shot)
        detailed = reference.decode_detailed(syndrome)
        job = windows.job_for(model, shot)
        result = row.decode(job)
        assert result.iterations == detailed.iterations


def test_the_tier_sections_keys_reach_the_rows_settings():
    section = {
        "kind": "relay_bp",
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
        "alpha": 0.5,
        "alpha_iteration_scaling_factor": 2,
        "gamma0": 0.35,
        "pre_iterations": 70,
        "relay_set_count": 600,
        "iterations_per_set": 50,
        "gamma_interval": [-0.254, 0.985],
        "converged_solution_count": 5,
    }
    clocks = config.ClockSettings({"decoder": 250.0})
    tier = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "weak_decoder"
    )
    assert tier.row_settings == relay.RelayBeliefPropagationDecoder.Settings(
        alpha=0.5,
        alpha_iteration_scaling_factor=2.0,
        gamma0=0.35,
        pre_iterations=70,
        relay_set_count=600,
        iterations_per_set=50,
        gamma_interval=(-0.254, 0.985),
        converged_solution_count=5,
    )


def test_a_section_with_no_keys_keeps_the_rows_old_profile():
    """The defaults are the values the row was hard-wired to before."""
    settings = relay.RelayBeliefPropagationDecoder.Settings.from_yaml(
        {}, None, "weak_decoder"
    )
    assert settings == relay.RelayBeliefPropagationDecoder.Settings(
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        gamma0=0.1,
        pre_iterations=80,
        relay_set_count=300,
        iterations_per_set=60,
        gamma_interval=(-0.24, 0.66),
        converged_solution_count=1,
    )


def test_the_first_relay_leg_must_run_at_least_once():
    """A zero first leg would hand back the previous window's answer.

    relay-bp's decode_inner runs its first leg for pre_iter iterations
    and leaves the previous call's decoding in place when that loop
    never runs (relay.rs in the relay-bp source), so a profile with no
    pre-iterations would report a stale correction for every window.
    One iteration is enough, so the boundary is exclusive at zero.
    """
    with pytest.raises(ValueError) as caught:
        relay.RelayBeliefPropagationDecoder.Settings.from_yaml(
            {"pre_iterations": 0}, None, "weak_decoder"
        )
    assert str(caught.value) == (
        "weak_decoder.pre_iterations must be a whole number of iterations, "
        "at least 1 (got 0)"
    )
    accepted = relay.RelayBeliefPropagationDecoder.Settings.from_yaml(
        {"pre_iterations": 1}, None, "weak_decoder"
    )
    assert accepted.pre_iterations == 1


def test_a_reversed_gamma_interval_is_refused():
    with pytest.raises(ValueError) as caught:
        relay.RelayBeliefPropagationDecoder.Settings.from_yaml(
            {"gamma_interval": [0.985, -0.254]}, None, "strong_decoder"
        )
    assert str(caught.value) == (
        "strong_decoder.gamma_interval must be [low, high], two finite real "
        "numbers with low at most high (got [0.985, -0.254])"
    )


def test_a_memory_strength_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError) as caught:
        relay.RelayBeliefPropagationDecoder.Settings.from_yaml(
            {"gamma0": "0.35"}, None, "weak_decoder"
        )
    assert str(caught.value) == (
        "weak_decoder.gamma0 must be a finite real number (got '0.35')"
    )
