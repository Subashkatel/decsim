"""The event_count burst detector against Stim's own samples.

The referent for a threshold is the tail Stim's detector sampler gives
on the same circuit: Stim's surface_code:rotated_memory_z at p = 1e-3,
the circuit decsim's memory_circuit generates. The counters are Q3DE's
(Suzuki et al. 2501.00331 lines 713-727) and Tan et al.'s one-round
Hamming weight widened to W rounds (2406.18897 lines 956-966); the
priors change is IonQ's (2608.25027 lines 334-340).
"""

import numpy
import pytest
import scipy.stats

import decsim.config as config
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.fault_model_contracts as fault_contracts
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.engine as engine_module
import decsim.escalation.burst_detectors as burst_detectors
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.windows as window_records

ROUNDS = 30
AFTER_LAST_ROUND = ROUNDS + 1
DISTANCE = 5
PHYSICAL_ERROR_PROBABILITY = 1e-3
CODE_TASK = "surface_code:rotated_memory_z"
# d = 5 rotated_memory_z: round one forms the 12 Z detectors against the
# prepared state, every later round 24 bulk detectors
FIRST_ROUND_QUIET = (0,) * 12
BULK_ROUND_QUIET = (0,) * 24
BULK_ROUND_LOUD = (1,) * 24
CLOCKS = config.ClockSettings({"fridge": 250.0})
GRAPHLIKE = fault_contracts.FaultRepresentation.GRAPHLIKE


def _circuit():
    return workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )


def _detector(settings, engine=None):
    if engine is None:
        engine = engine_module.Engine()
    circuit = _circuit()
    circuits = {1: (circuit, ROUNDS)}
    return burst_detectors.EventCountBurstDetector(settings, engine, circuits)


def _table():
    circuit = _circuit()
    return detector_formation.build_formation_table(circuit, ROUNDS)


def _rounds_of_events(sampled_events):
    """One shot's detection events cut into rounds, in formation order."""
    table = _table()
    rounds = []
    for round_index in range(1, AFTER_LAST_ROUND):
        recipes = table.detectors_of_round(round_index)
        values = []
        for recipe in recipes:
            value = sampled_events[recipe.detector_index]
            values.append(value)
        rounds.append(values)
    return rounds


def _sampled_rounds(circuit, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(1)
    return _rounds_of_events(shots[0])


def _whole_patch_burst(onset_round, probability):
    circuit = _circuit()
    table = _table()
    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=onset_round,
        burst_error_probability=probability,
    )
    return stim_device.burst_circuit(circuit, table, burst)


def _feed(detector, rounds):
    for index, events in enumerate(rounds):
        round_index = index + 1
        detector.observe_round(1, round_index, events)


def _sampled_bulk_round_counts(shot_count, seed):
    """Stim's detection events per bulk round, summed over the patch."""
    circuit = _circuit()
    sampler = circuit.compile_detector_sampler(seed=seed)
    samples = sampler.sample(shot_count)
    table = _table()
    bulk = detector_formation.LayerKind.BULK
    shape = (shot_count, ROUNDS)
    per_round = numpy.zeros(shape, dtype=numpy.int64)
    for recipe in table.detectors:
        if recipe.kind is bulk:
            column = recipe.round_index - 1
            per_round[:, column] += samples[:, recipe.detector_index]
    return per_round[:, 1:]


def _quiet_rounds(round_count):
    bulk_round_count = round_count - 1
    bulk_rounds = [BULK_ROUND_QUIET] * bulk_round_count
    return [FIRST_ROUND_QUIET, *bulk_rounds]


def _window(first_round, last_round):
    round_count = last_round - first_round + 1
    return window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=first_round,
        commit_hi=last_round,
        buffer_hi=last_round,
        round_count=round_count,
    )


def test_the_patch_count_tail_is_the_tail_stim_samples():
    """W = 1 at d = 5: the law's tail at its 1e-3 threshold, beside Stim's.

    Stim samples 20,000 shots, 29 bulk rounds each; the law's tail at
    the threshold must sit within five standard errors of the sampled
    frequency of the same count.
    """
    settings = burst_detectors.EventCountBurstDetector.Settings(
        patch_window_rounds=1, false_alarms_per_round=2e-3
    )
    detector = _detector(settings)
    counts = detector.counts_by_operation[1]
    law = counts.calibration.patch_law
    threshold = law.threshold(1e-3, 1.0)
    tails = law.tails(1.0)
    predicted = tails[threshold]
    bulk_rounds = _sampled_bulk_round_counts(20_000, seed=5)
    is_at_threshold = bulk_rounds >= threshold
    sampled = numpy.mean(is_at_threshold)
    variance = predicted / bulk_rounds.size
    standard_error = numpy.sqrt(variance)
    difference = predicted - sampled
    assert threshold == 6
    assert abs(difference) < 5 * standard_error


def test_a_whole_patch_burst_is_flagged_within_three_rounds():
    settings = burst_detectors.EventCountBurstDetector.Settings()
    detector = _detector(settings)
    burst = _whole_patch_burst(onset_round=12, probability=0.1)
    rounds = _sampled_rounds(burst, seed=3)
    _feed(detector, rounds[:14])
    window = _window(12, 14)
    assert detector.is_burst_window(window)


def test_a_burst_free_shot_is_not_flagged():
    settings = burst_detectors.EventCountBurstDetector.Settings()
    detector = _detector(settings)
    circuit = _circuit()
    rounds = _sampled_rounds(circuit, seed=3)
    _feed(detector, rounds)
    window = _window(1, ROUNDS)
    assert not detector.is_burst_window(window)


def test_a_flag_reaches_back_one_window_before_the_round_that_fired():
    """Q3DE 2501.00331 lines 727-728: the onset is read off the window."""
    settings = burst_detectors.EventCountBurstDetector.Settings(
        patch_window_rounds=4, detector_window_rounds=20
    )
    detector = _detector(settings)
    rounds = _quiet_rounds(11)
    _feed(detector, rounds)
    detector.observe_round(1, 12, BULK_ROUND_LOUD)
    first_flagged = _window(9, 9)
    before_it = _window(8, 8)
    assert detector.is_burst_window(first_flagged)
    assert not detector.is_burst_window(before_it)


def test_a_flag_is_not_seen_before_the_detector_publishes_it():
    """Five cycles of 1000 ticks a round, the rounds served in order."""
    clock = config.Clock(period_ticks=1000)
    settings = burst_detectors.EventCountBurstDetector.Settings(
        clock=clock, cycles_per_round=5
    )
    engine = engine_module.Engine()
    detector = _detector(settings, engine)
    detector.observe_round(1, 1, FIRST_ROUND_QUIET)
    detector.observe_round(1, 2, BULK_ROUND_LOUD)
    window = _window(2, 2)
    engine.now = 9_999
    assert not detector.is_burst_window(window)
    engine.now = 10_000
    assert detector.is_burst_window(window)


def test_the_row_none_takes_no_keys():
    section = {"kind": "none", "patch_window_rounds": 4}
    with pytest.raises(ValueError, match="burst_detector does not know"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_a_priced_count_needs_a_clock():
    section = {"kind": "event_count", "cycles_per_round": 3}
    with pytest.raises(ValueError, match="cycles_per_round needs a clock"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_a_false_alarm_rate_is_a_probability():
    section = {"kind": "event_count", "false_alarms_per_round": "1e-6"}
    with pytest.raises(ValueError, match="must be a probability"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_a_window_is_a_whole_number_of_rounds():
    section = {"kind": "event_count", "patch_window_rounds": 0}
    with pytest.raises(ValueError, match="whole number of rounds"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_raising_priors_is_true_or_false():
    section = {"kind": "event_count", "raise_strong_priors": "yes"}
    with pytest.raises(ValueError, match="must be true or false"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_burst_mode_ends_when_the_counts_return_to_their_usual_rate():
    """One loud round fires the patch count for W = 4 rounds, no longer."""
    settings = burst_detectors.EventCountBurstDetector.Settings(
        patch_window_rounds=4, detector_window_rounds=20
    )
    detector = _detector(settings)
    quiet_before = _quiet_rounds(11)
    quiet_after = [BULK_ROUND_QUIET] * 8
    rounds = [*quiet_before, BULK_ROUND_LOUD, *quiet_after]
    _feed(detector, rounds)
    last_firing = _window(15, 15)
    after_it = _window(16, 20)
    assert detector.is_burst_window(last_firing)
    assert not detector.is_burst_window(after_it)


def _window_model(first_round, last_round):
    """The window's model as the planner slices it: rows and columns."""
    circuit = _circuit()
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=ROUNDS,
        fault_model_requirement=fault_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )
    return slicer.slice_window(
        first_round, first_round, last_round, last_round, is_last=False
    )


def _flagged_detector(raise_strong_priors):
    """Every detector loud from round 12 to 17: the whole patch in burst."""
    settings = burst_detectors.EventCountBurstDetector.Settings(
        raise_strong_priors=raise_strong_priors
    )
    detector = _detector(settings)
    quiet_before = _quiet_rounds(11)
    loud = [BULK_ROUND_LOUD] * 6
    rounds = [*quiet_before, *loud]
    _feed(detector, rounds)
    return detector


def test_burst_priors_raise_the_priors_and_keep_the_graph():
    """IonQ 2608.25027 lines 334-340: only the prior vector changes.

    Every detector fires every round, a rate no prior below one half
    explains, so every column the flagged rows see sits at the cap.
    """
    detector = _flagged_detector(raise_strong_priors=True)
    window = _window(12, 17)
    model = _window_model(12, 17)
    raised = detector.with_burst_priors(window, model)
    faults = model.require_faults(GRAPHLIKE)
    raised_faults = raised.require_faults(GRAPHLIKE)
    changed_checks = faults.check != raised_faults.check
    assert changed_checks.nnz == 0
    assert raised.detector_ids == model.detector_ids
    assert raised_faults.source_fault_ids == faults.source_fault_ids
    is_capped = raised_faults.priors == 0.5
    was_below_cap = faults.priors < 0.5
    assert numpy.all(is_capped)
    assert numpy.all(was_below_cap)


def test_a_window_before_the_flag_keeps_its_model():
    detector = _flagged_detector(raise_strong_priors=True)
    window = _window(1, 6)
    model = _window_model(1, 6)
    kept = detector.with_burst_priors(window, model)
    assert kept is model


def test_without_burst_priors_a_flagged_window_keeps_its_model():
    detector = _flagged_detector(raise_strong_priors=False)
    window = _window(12, 17)
    model = _window_model(12, 17)
    kept = detector.with_burst_priors(window, model)
    assert kept is model


def test_a_flag_with_no_anomalous_position_keeps_the_model():
    """One loud round fires the patch count, no position's own count."""
    settings = burst_detectors.EventCountBurstDetector.Settings(
        raise_strong_priors=True
    )
    detector = _detector(settings)
    quiet_before = _quiet_rounds(11)
    rounds = [*quiet_before, BULK_ROUND_LOUD]
    _feed(detector, rounds)
    window = _window(12, 12)
    model = _window_model(12, 12)
    kept = detector.with_burst_priors(window, model)
    assert detector.is_burst_window(window)
    assert kept is model


def _noiseless_detector(settings):
    """The detector calibrated on the d = 5 memory circuit at p = 0."""
    circuit = workload_settings.memory_circuit(CODE_TASK, ROUNDS, DISTANCE, 0.0)
    circuits = {1: (circuit, ROUNDS)}
    engine = engine_module.Engine()
    return burst_detectors.EventCountBurstDetector(settings, engine, circuits)


@pytest.mark.filterwarnings("error")
def test_a_noiseless_background_fires_on_its_first_event():
    """No fault reaches a detector, so one event is rarer than any budget.

    The law of a count no fault reaches is a count that is always zero,
    and a region no fault reaches keeps its priors.
    """
    settings = burst_detectors.EventCountBurstDetector.Settings(
        raise_strong_priors=True
    )
    detector = _noiseless_detector(settings)
    quiet_before = _quiet_rounds(11)
    one_event = (1,) + BULK_ROUND_QUIET[1:]
    _feed(detector, quiet_before)
    quiet_window = _window(1, 11)
    was_quiet = detector.is_burst_window(quiet_window)
    detector.observe_round(1, 12, one_event)
    window = _window(12, 12)
    model = _window_model(12, 12)
    kept = detector.with_burst_priors(window, model)

    assert not was_quiet
    assert detector.is_burst_window(window)
    assert kept is model


def _exact_independent_tails(priors):
    """P(count >= k) for independent detectors: the Poisson binomial.

    Each fault flips its own detector, so the count's distribution is
    the convolution of the faults' Bernoulli terms.
    """
    distribution = numpy.array([1.0])
    for prior in priors:
        bernoulli = [1.0 - prior, prior]
        distribution = numpy.convolve(distribution, bernoulli)
    reversed_distribution = distribution[::-1]
    reversed_tails = numpy.cumsum(reversed_distribution)
    return reversed_tails[::-1]


def _exact_chain_tails(priors):
    """P(count >= k) when fault i flips detectors i and i + 1, enumerated.

    Two neighbouring faults cancel on the detector they share, as Stim
    XORs them, which is the case a sum of priors gets wrong.
    """
    fault_count = len(priors)
    detector_count = fault_count + 1
    incidence = _chain_incidence(fault_count)
    count_values = detector_count + 1
    distribution = numpy.zeros(count_values)
    mask_count = 1 << fault_count
    misses = 1.0 - priors
    for fired_mask in range(mask_count):
        fired = _fired_faults(fired_mask, fault_count)
        chances = numpy.where(fired, priors, misses)
        chance = numpy.prod(chances)
        flipped = incidence[fired].sum(axis=0) % 2
        flipped_count = flipped.sum()
        distribution[flipped_count] += chance
    reversed_distribution = distribution[::-1]
    reversed_tails = numpy.cumsum(reversed_distribution)
    return reversed_tails[::-1]


def _fired_faults(fired_mask: int, fault_count: int):
    shifts = numpy.arange(fault_count)
    shifted = fired_mask >> shifts
    bits = shifted & 1
    return bits.astype(bool)


def _chain_incidence(fault_count: int):
    shape = (fault_count, fault_count + 1)
    incidence = numpy.zeros(shape, dtype=numpy.uint8)
    for fault in range(fault_count):
        incidence[fault, fault] = 1
        incidence[fault, fault + 1] = 1
    return incidence


def _drawn_standard_errors(law):
    """Each tail's sampling error: the drawn conditional tails' spread."""
    fault_counts = numpy.arange(len(law.conditional_tails))
    weights = scipy.stats.poisson.pmf(fault_counts, law.fault_rate)
    conditional = law.conditional_tails
    spread = conditional * (1.0 - conditional)
    squared_weights = weights**2
    variance = squared_weights @ spread
    draws = burst_detectors.DRAWS_PER_FAULT_COUNT
    variance_of_mean = variance / draws
    return numpy.sqrt(variance_of_mean)


# The priors the exact grid runs, from a quiet chip to ten times Google's
# burst floor, and the fault counts each shape runs at.
GRID_PRIORS = [1e-3, 1e-2, 3e-2, 0.1]
INDEPENDENT_FAULT_COUNTS = [5, 12, 40]
CHAIN_FAULT_COUNTS = [5, 10]
# The smallest tail the grid compares, a tenth of the smallest budget a
# row is likely to ask for; the draws do not resolve smaller ones.
SMALLEST_COMPARED_TAIL = 1e-9
# The float rounding a tail of one carries, where no draw varies.
TAIL_ROUNDING = 1e-12


def _assert_the_law_is_the_exact_tail(priors, incidence, exact_tails):
    law = burst_detectors.tail_law(priors, incidence, 1e-8)
    tails = law.tails(1.0)
    standard_errors = _drawn_standard_errors(law)
    is_compared = exact_tails >= SMALLEST_COMPARED_TAIL
    compared_counts = numpy.flatnonzero(is_compared)
    law_tails = tails[compared_counts]
    exact = exact_tails[compared_counts]
    errors = standard_errors[compared_counts]
    signed_differences = law_tails - exact
    differences = numpy.abs(signed_differences)
    allowed = 5 * errors + TAIL_ROUNDING
    is_within = differences <= allowed
    assert numpy.all(is_within)


@pytest.mark.parametrize("prior", GRID_PRIORS)
@pytest.mark.parametrize("fault_count", INDEPENDENT_FAULT_COUNTS)
@pytest.mark.parametrize("is_mixed", [False, True])
def test_the_tail_law_is_the_poisson_binomial_tail(
    prior, fault_count, is_mixed
):
    """Independent faults on their own detectors, equal or mixed priors.

    Within five of the law's own sampling errors of the exact tail at
    every count whose tail is at least 1e-9; a law that fired each fault
    at a Poisson rate equal to its prior sat as much as 75 errors low.
    """
    priors = numpy.full(fault_count, prior)
    if is_mixed:
        generator = numpy.random.default_rng(fault_count)
        spread = generator.uniform(0.2, 1.8, fault_count)
        priors = priors * spread
    incidence = numpy.eye(fault_count, dtype=numpy.uint8)
    exact_tails = _exact_independent_tails(priors)

    _assert_the_law_is_the_exact_tail(priors, incidence, exact_tails)


@pytest.mark.parametrize("prior", GRID_PRIORS)
@pytest.mark.parametrize("fault_count", CHAIN_FAULT_COUNTS)
def test_the_tail_law_is_the_exact_tail_where_faults_cancel(prior, fault_count):
    priors = numpy.full(fault_count, prior)
    incidence = _chain_incidence(fault_count)
    exact_tails = _exact_chain_tails(priors)

    _assert_the_law_is_the_exact_tail(priors, incidence, exact_tails)


def test_a_budget_just_above_the_exact_tail_takes_the_exact_threshold():
    """20 faults at 1e-2: count 4 has tail 4.26e-5, over a 4.2e-5 budget."""
    priors = numpy.full(20, 1e-2)
    incidence = numpy.eye(20, dtype=numpy.uint8)
    exact_tails = _exact_independent_tails(priors)
    law = burst_detectors.tail_law(priors, incidence, 4.2e-5)

    threshold = law.threshold(4.2e-5, 1.0)

    assert exact_tails[4] > 4.2e-5
    assert threshold == 5


def test_a_count_no_fault_reaches_is_always_zero():
    priors = numpy.zeros(3)
    incidence = numpy.eye(3, dtype=numpy.uint8)
    law = burst_detectors.tail_law(priors, incidence, 1e-6)

    tails = law.tails(1.0)
    threshold = law.threshold(1e-6, 1.0)

    assert list(tails) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert threshold == 1
