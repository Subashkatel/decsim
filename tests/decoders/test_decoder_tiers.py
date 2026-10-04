"""The two decoder tiers as units: the mode picks one and it decodes.

The weak tier's real MWPM must reach whole-circuit PyMatching's answer
on the same events, and the strong tier's belief matching whole-circuit
beliefmatching's. Toshio arXiv 2510.25222: lightweight decoders
decode constantly, a separate accurate decoder is invoked on demand.
"""

import dataclasses

import beliefmatching
import numpy
import pymatching
import pytest

import decsim.config as config
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.machine as machine_module
import decsim.settings as machine_settings
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)


def _algorithm_records(machine) -> list:
    """The stage records of every decode's algorithm stage."""
    records = []
    for record in machine.observation.stages.records:
        if record.stage == staged_decoder.ALGORITHM_STAGE:
            records.append(record)
    return records


def _with_algorithm(settings, tier: str, algorithm):
    """The settings with one tier's decoder row replaced."""
    pool = getattr(settings, tier)
    replaced = dataclasses.replace(pool, algorithm=algorithm)
    return dataclasses.replace(settings, **{tier: replaced})


def _run(settings, seed: int = 0):
    """One shot of the settings: its machine and its result."""
    machine = machine_module.Machine.build(settings, seed)
    result = machine.run()
    return machine, result


def whole_circuit_predictions(machine, result) -> list:
    """Each operation's observables, PyMatching on its whole circuit.

    The reference a windowed loop is checked against: the decomposed
    detector error model of the circuit the source sampled, decoded in
    one piece on the events it drew, operation by operation in the
    result's order.
    """
    sampled = machine.observation.sampled_shots.shots_by_operation
    predictions = []
    for operation_result in result.operation_results:
        sampled_shot = sampled[operation_result.operation_id]
        model = sampled_shot.circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        events = numpy.asarray(sampled_shot.detection_events, dtype=bool)
        predicted = matching.decode(events)
        prediction = tuple(int(bit) for bit in predicted)
        predictions.append(prediction)
    return predictions


def loop_predictions(result) -> list:
    """Each operation's observables as the machine's loop decoded them."""
    predictions = []
    for operation_result in result.operation_results:
        observables = tuple(operation_result.logical_observables)
        predictions.append(observables)
    return predictions


@pytest.mark.parametrize("seed", range(3))
def test_weak_unit_loop_matches_direct_pymatching(seed):
    # The functional gate: the loop with the weak unit's real MWPM reaches
    # the same prediction as whole-circuit PyMatching on the same events.
    base = machine_settings.weak_decoder_baseline(3, 0.005, 1.0)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    settings = _with_algorithm(base, "weak_decoder", matching)
    machine, result = _run(settings, seed)
    windows = machine.observation.windows.windows
    loop = loop_predictions(result)
    assert len(windows) > 0
    assert loop == whole_circuit_predictions(machine, result)


def whole_circuit_belief_matching(machine, result) -> list:
    """Each operation's observables, beliefmatching on its whole circuit.

    BeliefMatching.decode (.pydeps/beliefmatching/belief_matching.py
    lines 344-358) at the strong base's own settings, 30 product-sum
    iterations, on the events the source drew, decoded offline in one
    piece.
    """
    sampled = machine.observation.sampled_shots.shots_by_operation
    predictions = []
    for operation_result in result.operation_results:
        sampled_shot = sampled[operation_result.operation_id]
        model = sampled_shot.circuit.detector_error_model(decompose_errors=True)
        offline = beliefmatching.BeliefMatching(
            model, max_bp_iters=30, bp_method="product_sum"
        )
        events = numpy.asarray(sampled_shot.detection_events, dtype=bool)
        predicted = offline.decode(events)
        prediction = tuple(int(bit) for bit in predicted)
        predictions.append(prediction)
    return predictions


@pytest.mark.parametrize("seed", range(12))
def test_strong_unit_loop_matches_offline_belief_matching(seed):
    """Strong-only predicts what its decoder predicts offline, shot for shot.

    The strong base's windowed loop, belief matching on the host, reaches
    the observables beliefmatching reaches on the whole circuit's events,
    so its accuracy is the offline decoder's on the same shots.
    """
    settings = machine_settings.strong_decoder_baseline(3, 0.01, 1.0)
    machine, result = _run(settings, seed)
    windows = machine.observation.windows.windows
    loop = loop_predictions(result)
    assert len(windows) > 0
    assert loop == whole_circuit_belief_matching(machine, result)


def test_a_union_find_tier_with_a_cycle_count_is_held_by_the_count():
    """Every algorithm stage ends on the count's clock edge, past its setup.

    A tier charged its host wall clock ends on no edge; one under a
    cycle count holds its unit for whole cycles of the named clock and
    never fewer than the eleven the quiet machine costs.
    """
    base = machine_settings.weak_decoder_baseline(3, 0.001, 1.0)
    one_hundred_megahertz = config.Clock.from_megahertz(100.0)
    cycle_count = cycle_count_module.CycleCount(
        clock=one_hundred_megahertz, delay_cycles=3
    )
    union_find_decoder = union_find.UnionFindDecoder.Settings(
        timing=cycle_count
    )
    settings = _with_algorithm(base, "weak_decoder", union_find_decoder)
    machine, _result = _run(settings)
    period_ticks = one_hundred_megahertz.period_ticks
    algorithm = _algorithm_records(machine)
    ticks_past_an_edge = {
        record.end_ticks % period_ticks for record in algorithm
    }
    held_ticks = [record.end_ticks - record.start_ticks for record in algorithm]
    assert len(algorithm) > 0
    assert ticks_past_an_edge == {0}
    assert min(held_ticks) >= 11 * period_ticks


def test_a_measured_table_tier_is_held_by_the_measured_line():
    """Every decode holds its unit 78.957 us plus 9.762 us an iteration.

    The gh200 and a100 rows differ, so the device the record names is
    the one the unit prices by: a100, whole, the 360-detector region
    nearest a d = 5 window (decoders/measured_table).
    """
    pytest.importorskip("relay_bp")
    base = machine_settings.strong_decoder_baseline(5, 0.001, 1.0)
    measured = measured_table.MeasuredTableDecoder.Settings(device="a100")
    settings = _with_algorithm(base, "strong_decoder", measured)
    machine, _result = _run(settings)
    algorithm = _algorithm_records(machine)
    intercept_ticks = 78_957_000
    ticks_per_iteration = 9_762_000
    held_ticks = [record.end_ticks - record.start_ticks for record in algorithm]
    iteration_ticks = [held - intercept_ticks for held in held_ticks]
    remainders = {ticks % ticks_per_iteration for ticks in iteration_ticks}
    assert len(algorithm) > 0
    assert min(iteration_ticks) >= 0
    assert remainders == {0}
