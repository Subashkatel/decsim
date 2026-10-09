"""Draws a shot from a circuit's detector error model, with the errors it fired.

Stim's CompiledDemSampler.sample(return_errors=True) answers which
error mechanisms fired beside the detection events and observable
flips they make, numbering the errors as the flattened model lists its
error instructions. The model is Stim's undecomposed one with
approximate_disjoint_errors off, so Stim refuses a channel whose cases
it cannot make independent mechanisms (Circuit.detector_error_model)
and the events and flips keep the circuit sampler's distribution. Gong
et al. sample their sliding windows the same way
(SlidingWindowDecoder osd.py:124-125).
"""

import weakref
from typing import Optional

import numpy
import stim

import decsim.records.sampled_errors as sampled_error_records


class ErrorModelSampler:
    """One circuit's detector error model, read once for every shot.

    Each error's detectors and observables are kept as lists, so a
    shot's fired errors are read without walking the whole model.
    """

    def __init__(self, circuit: stim.Circuit) -> None:
        self.model = circuit.detector_error_model()
        self.detectors_by_error, self.observables_by_error = _error_targets(
            self.model
        )

    def draw(self, sample_seed: Optional[int]) -> tuple:
        """One shot: (detection events, observable flips, fired errors).

        The fired errors are the model's error indices, in order.
        """
        sampler = self.model.compile_sampler(seed=sample_seed)
        detection_events, observable_flips, errors = sampler.sample(
            shots=1, return_errors=True
        )
        shot_errors = errors[0]
        fired_errors = numpy.flatnonzero(shot_errors)
        return detection_events[0], observable_flips[0], fired_errors

    def fired_errors(
        self, error_indices: numpy.ndarray, detector_rounds: dict
    ) -> tuple[sampled_error_records.FiredError, ...]:
        """The fired errors as their rounds and observable bits.

        detector_rounds is each detector's round, the formation table's.
        """
        observable_count = self.model.num_observables
        fired = []
        for error_index in error_indices:
            detectors = self.detectors_by_error[error_index]
            observables = self.observables_by_error[error_index]
            rounds = [detector_rounds[detector] for detector in detectors]
            assert rounds, f"error {error_index} flips no detector"
            observable_bits = _observable_bits(observables, observable_count)
            error = sampled_error_records.FiredError(
                first_round=min(rounds), logical_observables=observable_bits
            )
            fired.append(error)
        return tuple(fired)


def sampler_of(circuit: stim.Circuit) -> ErrorModelSampler:
    """The circuit's sampler, built once and kept while the circuit lives.

    A task's shots share their workload's circuit, so the model is read
    once per task, as decoders/decoder.py keeps a backend per model. The
    entry lives exactly as long as the circuit, since CPython recycles
    id() values.
    """
    circuit_identity = id(circuit)
    entry = _SAMPLERS.get(circuit_identity)
    if entry is not None:
        reference, sampler = entry
        if reference() is circuit:
            return sampler
    sampler = ErrorModelSampler(circuit)

    def discard_dead_circuit(reference) -> None:
        current = _SAMPLERS.get(circuit_identity)
        if current is not None and current[0] is reference:
            del _SAMPLERS[circuit_identity]

    reference = weakref.ref(circuit, discard_dead_circuit)
    _SAMPLERS[circuit_identity] = (reference, sampler)
    return sampler


# circuit identity -> (circuit reference, sampler), in this process
_SAMPLERS: dict = {}


def _error_targets(model: stim.DetectorErrorModel) -> tuple[list, list]:
    """Each error's detector indices and observable indices, in order."""
    detectors_by_error = []
    observables_by_error = []
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = _targets_of(instruction)
        detectors_by_error.append(detectors)
        observables_by_error.append(observables)
    return detectors_by_error, observables_by_error


def _targets_of(instruction: stim.DemInstruction) -> tuple:
    """One error's detectors and observables, a repeated target cancelled.

    A `^` separator only marks a suggested decomposition; the mechanism
    flips the XOR of its targets.
    """
    detectors = set()
    observables = set()
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            detectors ^= {target.val}
        if target.is_logical_observable_id():
            observables ^= {target.val}
    return tuple(sorted(detectors)), tuple(sorted(observables))


def _observable_bits(observables: tuple, observable_count: int) -> tuple:
    """One bit per observable, set where the error flips it."""
    bits = [0] * observable_count
    for observable in observables:
        bits[observable] = 1
    return tuple(bits)
