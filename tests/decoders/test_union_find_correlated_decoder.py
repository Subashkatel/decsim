"""The correlated_union_find row against Stim's decomposed error model.

Stim writes each error of circuit.detector_error_model(decompose_errors=
True) as its graphlike components split by ^ (Stim 1.15, the
decompose_errors flag). Correlated matching (Fowler 1310.0863) reads the
joint probability of two components from those errors, and a component's
prior is every error that holds it, each flipping it independently.
On a whole-circuit window the row's partner table must give each pair
the length that P(both) / P(selected) gives, capped at one half. The
second pass then changes some shots' answers and fails no more often
than one pass on the same shots.
"""

import collections
import itertools

import numpy

import decsim.decoders.union_find.correlated_decoder as correlated
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.basis_split as basis_split
import decsim.records.fault_model_contracts as fault_models
from tests.decoders import windows

HOST_TIME = cycle_count_module.HostMeasuredTime()
LINKED = fault_models.LINKED_FAULT_MODELS_REQUIRED
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
ROUNDS = 3


def test_each_partner_gets_the_length_stims_decomposition_gives_it():
    circuit = windows.memory_circuit(3, ROUNDS, 0.01)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    faults = model.require_faults(GRAPHLIKE)
    settings = correlated.CorrelatedUnionFindDecoder.Settings(timing=HOST_TIME)
    row = settings.build()
    backend = row.compile(faults, model)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    step = backend.graph.weight_step

    expected = stim_partner_lengths(detector_error_model, step)
    found = row_partner_lengths(backend, model, faults)

    assert found == expected


def test_a_shot_with_no_partnered_fault_selected_is_decoded_once():
    circuit = windows.memory_circuit(3, ROUNDS, 0.01)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    settings = correlated.CorrelatedUnionFindDecoder.Settings(timing=HOST_TIME)
    row = settings.build()
    quiet_shot = numpy.zeros(circuit.num_detectors, dtype=numpy.uint8)
    job = windows.job_for(model, quiet_shot)

    result = row.decode(job)

    assert result.cluster_evidence.earlier_passes == ()


def test_the_second_pass_changes_some_answers_and_fails_no_more_often():
    circuit = windows.memory_circuit(5, 5, 0.006)
    model = windows.whole_circuit_window(circuit, 5, LINKED)
    detection_events, observables = windows.sampled_shots(circuit, 2000, 7)
    one_pass_settings = union_find.UnionFindDecoder.Settings(timing=HOST_TIME)
    one_pass = one_pass_settings.build()
    two_pass_settings = correlated.CorrelatedUnionFindDecoder.Settings(
        timing=HOST_TIME
    )
    two_pass = two_pass_settings.build()
    one_pass_failures = 0
    two_pass_failures = 0
    changed_answers = 0
    for shot, truth in zip(detection_events, observables, strict=True):
        job = windows.job_for(model, shot)
        first = one_pass.decode(job)
        second = two_pass.decode(job)
        actual = (int(truth[0]),)
        one_pass_failures += first.logical_observables != actual
        two_pass_failures += second.logical_observables != actual
        changed_answers += (
            first.logical_observables != second.logical_observables
        )

    assert changed_answers > 0
    assert two_pass_failures < one_pass_failures


def stim_partner_lengths(detector_error_model, weight_step) -> dict:
    """(selected, partner) -> length, both named by their targets' text."""
    priors, joint = stim_priors_and_joint(detector_error_model)
    lengths = {}
    for (first, second), both in joint.items():
        conditional = both / priors[first]
        conditional = min(conditional, correlated.LARGEST_CONDITIONAL)
        if conditional > priors[second]:
            length = window_decoder.length_half_ticks(conditional, weight_step)
            lengths[(first, second)] = length
    return lengths


def stim_priors_and_joint(detector_error_model) -> tuple:
    """Each component's prior and each pair's joint probability.

    A component's errors flip it independently; a pair's joint is the
    sum of the errors that hold both.
    """
    priors = collections.defaultdict(float)
    joint = collections.defaultdict(float)
    for instruction in detector_error_model.flattened():
        if instruction.type != "error":
            continue
        arguments = instruction.args_copy()
        probability = arguments[0]
        components = stim_components(instruction)
        for component in components:
            prior = priors[component]
            priors[component] = prior + probability - 2 * prior * probability
        for pair in itertools.permutations(components, 2):
            joint[pair] += probability
    return priors, joint


def stim_components(instruction) -> list:
    """An error's components, each its sorted detectors and observables."""
    components = [[]]
    for target in instruction.targets_copy():
        if target.is_separator():
            components.append([])
        else:
            components[-1].append(str(target))
    return [tuple(sorted(component)) for component in components]


def row_partner_lengths(backend, model, faults) -> dict:
    """The row's table, each fault named as stim_components names it."""
    names = column_names(model, faults)
    lengths = {}
    for selected, partners in backend.partner_lengths.items():
        for edge_index, length in partners:
            edge = backend.graph.edges[edge_index]
            key = (names[selected], names[edge.fault_index])
            lengths[key] = length
    return lengths


def column_names(model, faults) -> list:
    """Each graphlike column's targets, as Stim writes them."""
    detector_rows = basis_split.every_column_rows(faults.check)
    observable_rows = basis_split.every_column_rows(faults.observables)
    names = []
    for rows, logicals in zip(detector_rows, observable_rows, strict=True):
        detectors = [f"D{model.detector_ids[row]}" for row in rows]
        logical_names = [f"L{logical}" for logical in logicals]
        targets = detectors + logical_names
        name = tuple(sorted(targets))
        names.append(name)
    return names
