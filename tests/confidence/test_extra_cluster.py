"""The extra-cluster gap beside the cluster gap and beside the paper.

The row under test is decsim/confidence/extra_cluster.py over
decsim/decoders/union_find/union_find.c (union_find_extra_growth). Its
laws are Kishi et al. 2602.03336's (lines 510 to 548 of the text):
Theorem 1, the gap is at most the cluster gap whenever it is defined,
here within the half weight step the whole-tick growth can add;
Theorem 2, it is defined whenever the cluster
gap is at or under the growth limit. The cluster gap is cluster_gap.c
through the same binding, checked in test_compiled_cluster_gap.py. The
join tick is also held against independent_extra_growth.py, a reading
of Algorithm 1 from the paper text whose join test is a breadth-first
two-colouring, on the decoder corpus's random small graphs and on
surface code windows. The four-detector chain below is traced tick by
tick and cycle by cycle.
"""

import math
import os
import random
import subprocess
import sys

import numpy
import pytest

import decsim.confidence.cluster as cluster
import decsim.confidence.extra_cluster as extra_cluster
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.escalation.settings as escalation_settings
import decsim.ports as ports
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
import tests.confidence.independent_extra_growth as independent_extra_growth
import tests.decoders.test_union_find_compiled_decoder as decoder_corpus
import tests.decoders.windows as windows

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
REQUIREMENT = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
# the paper's early-stopping threshold, 20 dB, in natural-log units
# (2602.03336 lines 696-699 and 774-776)
TWENTY_DECIBELS_NATS = 20.0 / (10.0 * math.log10(math.e))
SIXTY_DECIBELS_NATS = 3.0 * TWENTY_DECIBELS_NATS
CORPUS_SHOTS = 40
CIRCUIT_SEED = 11
RANDOM_GRAPHS = 60
RANDOM_GRAPH_SEED = 3
RANDOM_GRAPH_LIMIT_TICKS = 6
SANITIZED_GRAPHS = 40
CHILD_PROGRAM = (
    "import tests.confidence.test_extra_cluster as rows;"
    f"rows.compare_random_graphs({SANITIZED_GRAPHS}, {RANDOM_GRAPH_SEED})"
)

# The four-detector chain: b, 0, 1, 2, b with edge lengths 3, 2, 2, 3 ticks at
# a weight step of one natural-log unit, the logical row on the left
# boundary edge, and detectors 0 and 1 flipped
CHAIN_CHECK = ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1))
CHAIN_WEIGHT_STEP = 1.0
CHAIN_PRIORS = (
    1.0 / (1.0 + math.exp(3.0)),
    1.0 / (1.0 + math.exp(2.0)),
    1.0 / (1.0 + math.exp(2.0)),
    1.0 / (1.0 + math.exp(3.0)),
)
CHAIN_OBSERVABLES = ((1, 0, 0, 0),)
CHAIN_SYNDROME = (1, 1, 0)
CHAIN_LIMIT_TICKS = 10
CHAIN_LIMIT_NATS = CHAIN_LIMIT_TICKS * CHAIN_WEIGHT_STEP
TWO_TICKS_NATS = 2.0 * CHAIN_WEIGHT_STEP
# Helios's row: three registers between an element and the controller
HELIOS_DELAY_CYCLES = 3
CLOCK = config.Clock(10)


def chain_graph() -> evidence_records.UnionFindGraph:
    check = numpy.asarray(CHAIN_CHECK, dtype=numpy.uint8)
    priors = numpy.asarray(CHAIN_PRIORS, dtype=float)
    observables = numpy.asarray(CHAIN_OBSERVABLES, dtype=numpy.uint8)
    fault_count = check.shape[1]
    owned = numpy.ones(fault_count, dtype=bool)
    placed = fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(range(fault_count)),
        boundary_flips={},
    )
    return window_decoder.graph_from_model(
        placed, location="four-detector chain", weight_step=CHAIN_WEIGHT_STEP
    )


def chain_evidence() -> evidence_records.UnionFindHardEvidence:
    graph = chain_graph()
    syndrome = numpy.asarray(CHAIN_SYNDROME, dtype=numpy.uint8)
    return window_decoder.decode_graph(graph, syndrome)


def chain_result() -> decoding_records.DecodeResult:
    evidence = chain_evidence()
    return decoding_records.DecodeResult(
        evidence.selected_faults, 0, cluster_evidence=evidence
    )


def grown(evidence, growth_limit_ticks: int):
    return compiled_decoder.extra_growth(
        evidence.graph,
        evidence.edge_intervals,
        evidence.residual_syndrome,
        growth_limit_ticks,
    )


def test_the_row_declares_one_decode_and_the_growth_it_reads():
    """The signal's requirement and the Union-Find row's declaration meet."""
    signal = extra_cluster.ExtraClusterGap(TWENTY_DECIBELS_NATS)
    assert isinstance(signal, ports.ConfidenceSignal)
    assert signal.forced_logical_classes == ()
    required = signal.decoder_evidence_requirement
    assert required == decoding_records.CLUSTER_GROWTH_EVIDENCE
    row = union_find.UnionFindDecoder()
    assert not required - row.decoder_evidence
    assert signal.source.method == "extra_cluster_gap"


def test_the_growth_limit_is_the_whole_ticks_that_cover_the_threshold():
    """Half a step a tick per front: 20 dB at step 0.1 is 47 ticks."""
    assert extra_cluster.growth_limit_ticks(TWENTY_DECIBELS_NATS, 0.1) == 47
    assert extra_cluster.growth_limit_ticks(0.3, 0.1) == 3
    assert extra_cluster.growth_limit_ticks(0.0, 0.1) == 0


def test_a_growth_limit_that_is_not_a_weight_is_refused():
    with pytest.raises(ValueError) as refusal:
        extra_cluster.growth_limit_ticks(-1.0, 0.1)

    assert "finite nonnegative" in str(refusal.value)


def test_the_chain_decode_takes_one_step_and_leaves_the_ends_open():
    """The chain's decode: e1 closes at tick 2 and the pair goes even."""
    evidence = chain_evidence()
    step = evidence_records.GrowthStep(
        edge_count=3, hop_count=1, growth_ticks=2, fusion="parity"
    )
    assert evidence.growth_steps == (step,)
    intervals = evidence.edge_intervals
    assert intervals[1] == evidence_records.Closed()
    assert intervals[0].upper_tick - intervals[0].lower_tick == 4
    assert intervals[2].upper_tick - intervals[2].lower_tick == 2
    assert intervals[3].upper_tick - intervals[3].lower_tick == 6


def test_the_chain_joins_at_tick_three_over_three_events():
    """The chain tick by tick: e2, then e0, then e3 closes the walk."""
    evidence = chain_evidence()

    outcome = grown(evidence, CHAIN_LIMIT_TICKS)

    assert outcome.joined_at_tick == 3
    assert outcome.growth_steps == (
        evidence_records.GrowthStep(3, 2, 1, "roots"),
        evidence_records.GrowthStep(2, 2, 1, "roots"),
        evidence_records.GrowthStep(1, 0, 1, "none"),
    )


def test_the_chain_costs_twenty_two_cycles_on_helios_row():
    """(2 + 3 + 1) a tick for three ticks, plus two root floods two deep."""
    count = cycle_count_module.CycleCount(
        CLOCK, delay_cycles=HELIOS_DELAY_CYCLES
    )
    evidence = chain_evidence()
    outcome = grown(evidence, CHAIN_LIMIT_TICKS)

    assert count.extra_growth_cycles(outcome.growth_steps) == 22
    assert count.extra_growth_ticks(outcome.growth_steps) == 220


def test_a_growth_of_no_ticks_pays_one_join_test():
    count = cycle_count_module.CycleCount(CLOCK, delay_cycles=3)

    assert count.extra_growth_cycles(()) == cycle_count_module.JOIN_TEST_CYCLES


def test_the_chain_gap_is_three_steps_under_the_cluster_gap_of_six():
    """Theorem 1 on the chain: g_ec = w_max(P_c) = 3 < g_c = 6."""
    signal = extra_cluster.ExtraClusterGap(
        CHAIN_LIMIT_NATS, weight_step=CHAIN_WEIGHT_STEP
    )
    result = chain_result()

    computation = signal.compute((result,))

    assert computation.soft_output.gap == 3.0
    evidence = result.cluster_evidence
    half_ticks = compiled_decoder.cluster_gap(
        evidence.graph, evidence.edge_intervals
    )
    assert half_ticks == 12


def test_a_limit_below_the_join_leaves_the_gap_infinite():
    """The confident case: the limit comes first, nothing joins."""
    signal = extra_cluster.ExtraClusterGap(
        TWO_TICKS_NATS, weight_step=CHAIN_WEIGHT_STEP
    )
    result = chain_result()

    computation = signal.compute((result,))

    assert computation.soft_output.gap == math.inf


def test_the_limit_event_closes_nothing_and_is_charged_as_a_step():
    """Two ticks of a three-tick join: the last event stops at the limit."""
    evidence = chain_evidence()

    outcome = grown(evidence, 2)

    assert outcome.joined_at_tick is None
    assert outcome.growth_steps == (
        evidence_records.GrowthStep(3, 2, 1, "roots"),
        evidence_records.GrowthStep(2, 2, 1, "roots"),
    )


def test_the_cycle_count_prices_the_growth_on_the_unit():
    count = cycle_count_module.CycleCount(
        CLOCK, delay_cycles=HELIOS_DELAY_CYCLES
    )
    signal = extra_cluster.ExtraClusterGap(
        CHAIN_LIMIT_NATS,
        weight_step=CHAIN_WEIGHT_STEP,
        cycle_count=count,
    )
    result = chain_result()

    computation = signal.compute((result,))

    assert computation.ticks == 220


def test_a_card_prices_the_growth_instead():
    signal = extra_cluster.ExtraClusterGap(
        CHAIN_LIMIT_NATS,
        weight_step=CHAIN_WEIGHT_STEP,
        walk_microseconds=0.5,
    )
    result = chain_result()

    computation = signal.compute((result,))

    assert computation.ticks == config.microseconds_to_ticks(0.5)


def test_a_decode_without_growth_reports_no_gap():
    signal = extra_cluster.ExtraClusterGap(TWENTY_DECIBELS_NATS)
    solve = decoding_records.DecodeResult(1, 0)

    computation = signal.compute((solve,))

    assert computation.soft_output is None
    assert computation.ticks == 0


def test_the_row_builds_from_the_threshold_and_the_weak_row():
    count = cycle_count_module.CycleCount(CLOCK, delay_cycles=3)
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        confidence="extra_cluster_gap",
        gap_threshold_nats=TWENTY_DECIBELS_NATS,
    )
    weak = decoder_settings.DecoderSettings(
        kind="union_find", weight_step=0.2, cycle_count=count
    )

    signal = extra_cluster.ExtraClusterGap.from_settings(escalation, weak)

    assert signal.growth_limit_ticks == 24
    assert signal.weight_step == 0.2
    assert signal.cycle_count is count


def test_a_threshold_with_no_number_at_build_is_refused():
    """An online source has no epsilon_max to grow to."""
    escalation = escalation_settings.EscalationSettings(
        kind="switching",
        confidence="extra_cluster_gap",
        threshold_source="online",
    )
    weak = decoder_settings.DecoderSettings(kind="union_find")

    with pytest.raises(ValueError) as refusal:
        extra_cluster.ExtraClusterGap.from_settings(escalation, weak)

    assert "threshold_source online has no fixed number" in str(refusal.value)


def compare_random_graphs(count: int, seed: int) -> int:
    """One syndrome on each of count random graphs, C against the paper.

    The decoder corpus's random graphs carry zero-length edges, boundary
    to boundary columns and duplicates; a syndrome per graph is decoded
    and grown on, and the two readings must name one tick. Returns how
    many joined.
    """
    rng = random.Random(seed)
    joined = 0
    for _ in range(count):
        graph = decoder_corpus.random_graph(rng)
        bits = decoder_corpus.random_bits(rng, graph.detector_count)
        syndrome = numpy.asarray(bits, dtype=numpy.uint8)
        evidence = window_decoder.decode_graph(graph, syndrome)
        outcome = grown(evidence, RANDOM_GRAPH_LIMIT_TICKS)
        expected = independent_extra_growth.joined_at_tick(
            graph, evidence.edge_intervals, RANDOM_GRAPH_LIMIT_TICKS
        )
        assert outcome.joined_at_tick == expected, graph
        if expected is not None:
            joined += 1
    return joined


def test_the_random_graphs_join_on_the_tick_the_paper_reading_names():
    """Property: the C's union-find with parities against a two-colouring."""
    joined = compare_random_graphs(RANDOM_GRAPHS, RANDOM_GRAPH_SEED)

    assert 0 < joined < RANDOM_GRAPHS


@pytest.mark.skipif(
    decoder_corpus.SANITIZERS_MISSING,
    reason="no sanitizer runtime this interpreter can preload",
)
def test_the_extra_growth_stays_inside_its_buffers_on_the_random_graphs():
    """The random graphs again in a child under the sanitizers.

    The same child shape as the decoder's own sanitized run
    (test_union_find_compiled_decoder.py): the runtime preloaded, the
    sanitized library named, leaks unchecked because the interpreter
    holds allocations at exit.
    """
    checkout = decoder_corpus.CHECKOUT
    build_command = checkout / compiled_decoder.BUILD_COMMAND
    subprocess.run([str(build_command), "--sanitize"], cwd=checkout, check=True)
    library = checkout / decoder_corpus.SANITIZED_LIBRARY
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = decoder_corpus.SANITIZER_RUNTIME
    environment["ASAN_OPTIONS"] = "detect_leaks=0"
    environment["PYTHONPATH"] = str(checkout)
    environment[compiled_decoder.LIBRARY_VARIABLE] = str(library)
    child = subprocess.run(
        [sys.executable, "-c", CHILD_PROGRAM],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    assert "AddressSanitizer" not in child.stderr


@pytest.mark.parametrize("distance", (3, 5))
@pytest.mark.parametrize(
    "limit_nats", (TWENTY_DECIBELS_NATS, SIXTY_DECIBELS_NATS)
)
def test_the_windows_keep_kishi_theorems_against_the_cluster_gap(
    distance, limit_nats
):
    """Theorems 1 and 2 on surface code shots, at 20 dB and at 60 dB.

    Whenever the extra-cluster gap is defined it is at most the cluster
    gap plus half a step; whenever the cluster gap is at or under the
    limit the extra-cluster gap is defined. The join tick also agrees
    with the paper reading on every shot.
    """
    circuit = windows.memory_circuit(distance, distance, 0.005)
    model = windows.whole_circuit_window(circuit, distance, REQUIREMENT)
    row = union_find.UnionFindDecoder()
    sampled, _observables = windows.sampled_shots(
        circuit, CORPUS_SHOTS, CIRCUIT_SEED
    )
    step = evidence_records.DEFAULT_WEIGHT_STEP
    signal = extra_cluster.ExtraClusterGap(limit_nats, weight_step=step)
    for shot in sampled:
        job = windows.job_for(model, shot)
        result = row.decode(job)
        evidence = result.cluster_evidence
        cluster_gap = _cluster_gap_nats(evidence, step)
        computation = signal.compute((result,))
        gap = computation.soft_output.gap
        _check_theorems(gap, cluster_gap, limit_nats, step)
        _check_the_paper_reading(evidence, signal, gap, step)


def _cluster_gap_nats(evidence, step: float) -> float:
    half_ticks = compiled_decoder.cluster_gap(
        evidence.graph, evidence.edge_intervals
    )
    return cluster.gap_half_ticks_to_natural_log_weight(half_ticks, step)


def _check_theorems(gap, cluster_gap, limit_nats, step) -> None:
    rounding = step / 2.0
    if math.isfinite(gap):
        assert gap <= cluster_gap + rounding + 1e-9
    if cluster_gap <= limit_nats:
        assert math.isfinite(gap)


def _check_the_paper_reading(evidence, signal, gap, step) -> None:
    expected = independent_extra_growth.joined_at_tick(
        evidence.graph, evidence.edge_intervals, signal.growth_limit_ticks
    )
    if expected is None:
        assert gap == math.inf
        return
    expected_gap = expected * step
    assert gap == pytest.approx(expected_gap)
