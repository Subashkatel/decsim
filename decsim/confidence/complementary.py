"""The complementary gap: the confidence of one MWPM window decode.

g_comp = |complementary_class_weight - decoded_class_weight| (Toshio
et al. 2510.25222 Sec. III A; the
method of Gidney et al. 2312.04522): the minimum-weight matching gives
decoded_class_weight and the decoded class, and the same graph with one
virtual detector that pins the observable to the other class gives
complementary_class_weight. A small gap is a decoder unsure of its
class.
ComplementaryGap is the ConfidenceSignal row (decsim/ports.py): it
declares the source and builds one metric per window model for the
confidence wrappers (decoder.py).
"""

import dataclasses
import time
from typing import Optional

import numpy
import pymatching
import scipy.sparse

import decsim.decoders.minimum_weight_perfect_matching.weights as weights_module
import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.stim_fault_catalog as stim_fault_catalog
import decsim.records.decoding as decoding_records

COMPLEMENTARY_GAP_SOURCE = decoding_records.SoftOutputSource(
    method="complementary_gap",
    cluster_origin="mwpm_opposite_logical",
    growth_schedule="minimum_weight_matching",
    gap_units="log_likelihood_weight",
    correction="opposite_logical_constraint",
    weight_step_natural_log=None,
    references=("complementary-gap method",),
)
# PyMatching finishes its graph lazily on the first decode; this many
# single-fault syndromes warm both graphs at build
WARM_UP_SYNDROME_COUNT = 3


def detector_error_model_to_matrices(detector_error_model) -> tuple:
    """A decomposed stim model as (check, observables, weights).

    check is detectors by faults, observables is observables by faults,
    both 0/1; weights are ln((1 - p) / p) per fault.
    """
    detector_count = detector_error_model.num_detectors
    observable_count = detector_error_model.num_observables
    detector_sets, observable_sets, priors = (
        stim_fault_catalog.detector_error_model_to_faults(detector_error_model)
    )
    check_columns = []
    observable_columns = []
    fault_weights = []
    faults = zip(detector_sets, observable_sets, priors)
    for fault_index, (detectors, observables, probability) in enumerate(faults):
        location = f"fault {fault_index}"
        fault_identity.validate_graphlike_fault(
            detectors, observables, location=location
        )
        check_column = _indicator_column(detector_count, detectors)
        observable_column = _indicator_column(observable_count, observables)
        weight = weights_module.matching_weights([probability])
        check_columns.append(check_column)
        observable_columns.append(observable_column)
        fault_weights.append(float(weight[0]))
    check = _columns_as_matrix(check_columns, detector_count)
    observable_matrix = _columns_as_matrix(observable_columns, observable_count)
    weights = numpy.asarray(fault_weights, dtype=float)
    return check, observable_matrix, weights


class ComplementaryGapMetric:
    """The complementary gap of a single-observable decoding window."""

    name = "complementary_gap"

    def __init__(self, check, observables, weights) -> None:
        self.check_matrix = _as_uint8_csc(check)
        self.observable_matrix = _as_uint8_dense(observables)
        self.weights = numpy.asarray(weights, dtype=float)
        observable_count = self.observable_matrix.shape[0]
        if observable_count != 1:
            raise ValueError(
                "the complementary gap is defined for one observable; got "
                f"{observable_count}. Decode each logical operator with its "
                "own metric."
            )
        # parallel faults (a window restriction folds distinct faults onto
        # one edge) combine as independent errors, the convention of
        # PyMatching's detector-error-model loader and of the window
        # decoder, so decoded_class_weight here is the window decoder's
        # own minimum weight
        priors = 1 / (1 + numpy.exp(self.weights))
        self._matching = _matching_over(self.check_matrix, self.weights, priors)
        observable_row = self.observable_matrix[0:1, :]
        observable_check = scipy.sparse.csc_matrix(observable_row)
        stacked = scipy.sparse.vstack([self.check_matrix, observable_check])
        augmented_check = stacked.tocsc()
        self._augmented_matching = _matching_over(
            augmented_check, self.weights, priors
        )
        self._warm_matchings()

    @classmethod
    def from_detector_error_model(
        cls, detector_error_model
    ) -> "ComplementaryGapMetric":
        """The metric of a decomposed stim DetectorErrorModel."""
        check, observables, weights = detector_error_model_to_matrices(
            detector_error_model
        )
        return cls(check, observables, weights)

    @classmethod
    def from_window_model(cls, model) -> "ComplementaryGapMetric":
        """The metric of a decsim window model (check, observables, priors)."""
        faults = model.require_faults(
            fault_models.FaultRepresentation.GRAPHLIKE
        )
        weights = weights_module.matching_weights(faults.priors)
        return cls(faults.check, faults.observables, weights)

    def evaluate(self, syndrome) -> decoding_records.SoftOutput:
        """The soft output of one syndrome: the gap and the two weights.

        decoded_class_weight is the minimum-weight matching's weight,
        complementary_class_weight the weight of the matching forced
        into the other class (Toshio et al. 2510.25222 Sec. III A).
        """
        bits = _syndrome_bits(syndrome)
        correction, minimum_weight = self._matching.decode(
            bits, return_weight=True
        )
        predicted_class = self._class_of(correction)
        complementary_class = predicted_class ^ 1
        forced_bits = _with_virtual_detector(bits, complementary_class)
        _, complementary_weight = self._augmented_matching.decode(
            forced_bits, return_weight=True
        )
        decoded_class_weight = float(minimum_weight)
        complementary_class_weight = float(complementary_weight)
        difference = complementary_class_weight - decoded_class_weight
        gap = abs(difference)
        return decoding_records.SoftOutput(
            gap=gap,
            source=COMPLEMENTARY_GAP_SOURCE,
            decoded_class_weight=decoded_class_weight,
            complementary_class_weight=complementary_class_weight,
        )

    def forced_class_solve(self, syndrome, forced_class: int) -> tuple:
        """(weight, nanoseconds) of one forced-class solve.

        The augmented virtual-detector bit pins the observable's parity
        to forced_class; the solve is independent of the other class's
        solve, which is what lets the two run on different hardware.
        paired_evaluate is both solves on one host; the split engines
        call this once each.
        """
        bits = _syndrome_bits(syndrome)
        forced_bits = _with_virtual_detector(bits, forced_class)
        started = time.perf_counter_ns()
        _, weight = self._augmented_matching.decode(
            forced_bits, return_weight=True
        )
        finished = time.perf_counter_ns()
        elapsed_nanoseconds = finished - started
        return float(weight), elapsed_nanoseconds

    def paired_evaluate(self, syndrome) -> "PairedGapEvaluation":
        """The gap as two independent forced-class solves.

        Each solve constrains the observable to one class by setting the
        augmented virtual-detector bit, so neither depends on the other:
        this is the form two matching cores compute side by side. The
        unconstrained minimum weight decoded_class_weight equals the
        smaller forced weight, the prediction is the winning class, and
        the gap is the weight difference; evaluate computes the same
        object serially.
        Each solve carries its own wall-clock time so a caller can model
        the pair's latency as the slower core, not the sum.
        """
        forced_weights = []
        solve_nanoseconds = []
        for forced_class in (0, 1):
            weight, elapsed = self.forced_class_solve(syndrome, forced_class)
            forced_weights.append(weight)
            solve_nanoseconds.append(elapsed)
        is_class_one_lighter = forced_weights[1] < forced_weights[0]
        predicted_class = int(is_class_one_lighter)
        decoded_class_weight = forced_weights[predicted_class]
        complementary_class = 1 - predicted_class
        complementary_class_weight = forced_weights[complementary_class]
        difference = complementary_class_weight - decoded_class_weight
        gap = abs(difference)
        soft_output = decoding_records.SoftOutput(
            gap=gap,
            source=COMPLEMENTARY_GAP_SOURCE,
            decoded_class_weight=decoded_class_weight,
            complementary_class_weight=complementary_class_weight,
        )
        return PairedGapEvaluation(
            soft_output=soft_output,
            predicted_class=predicted_class,
            forced_solve_nanoseconds=tuple(solve_nanoseconds),
        )

    def _class_of(self, correction) -> int:
        """The observable's parity under the correction."""
        if self.observable_matrix.shape[1] == 0:
            return 0
        parity = (self.observable_matrix @ correction) % 2
        return int(parity[0])

    def _warm_matchings(self) -> None:
        """Decode a few single-fault syndromes on both graphs at build.

        Without this, every window's first timed solve silently pays
        the graph build (the base window decoder warms for the same
        reason). Each warm-up syndrome is one column's detector set,
        which that fault alone explains, so it is satisfiable; the
        augmented graph additionally gets that fault's observable bit.
        """
        warmed = 0
        column_count = self.check_matrix.shape[1]
        for column in range(column_count):
            rows = _column_rows(self.check_matrix, column)
            if rows.size == 0:
                continue
            syndrome = numpy.zeros(
                self.check_matrix.shape[0], dtype=numpy.uint8
            )
            syndrome[rows] = 1
            self._matching.decode(syndrome)
            observable_bit = int(self.observable_matrix[0, column])
            augmented_syndrome = _with_virtual_detector(
                syndrome, observable_bit
            )
            self._augmented_matching.decode(augmented_syndrome)
            warmed += 1
            if warmed == WARM_UP_SYNDROME_COUNT:
                break


@dataclasses.dataclass(frozen=True)
class PairedGapEvaluation:
    """One window's gap computed as two parallel forced-class solves."""

    soft_output: decoding_records.SoftOutput
    predicted_class: int
    forced_solve_nanoseconds: tuple


@dataclasses.dataclass(frozen=True)
class ComplementaryGap:
    """The signal row: the complementary gap, one metric per window model."""

    source = COMPLEMENTARY_GAP_SOURCE
    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def metric_for(self, model) -> Optional[ComplementaryGapMetric]:
        """The metric of one placed window model.

        None when the model has no single nonzero observable row: the
        gap pins the observable to the other class, so it needs one.
        """
        if not _has_one_observable(model):
            return None
        return ComplementaryGapMetric.from_window_model(model)


def _has_one_observable(model) -> bool:
    """Whether the model's one observable row is nonzero."""
    faults = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    observables = faults.observables.toarray()
    if observables.shape[0] != 1:
        return False
    has_nonzero_row = observables.any()
    return bool(has_nonzero_row)


def _indicator_column(length: int, indices) -> numpy.ndarray:
    column = numpy.zeros(length, dtype=numpy.uint8)
    column[list(indices)] = 1
    return column


def _columns_as_matrix(columns: list, row_count: int) -> numpy.ndarray:
    if not columns:
        return numpy.zeros((row_count, 0), numpy.uint8)
    stacked = numpy.array(columns, dtype=numpy.uint8)
    return stacked.T


def _as_uint8_csc(check):
    if scipy.sparse.issparse(check):
        sparse = check.tocsc()
    else:
        dense = numpy.asarray(check)
        sparse = scipy.sparse.csc_matrix(dense)
    return sparse.astype(numpy.uint8)


def _as_uint8_dense(observables) -> numpy.ndarray:
    if scipy.sparse.issparse(observables):
        dense = observables.toarray()
    else:
        dense = numpy.asarray(observables)
    return dense.astype(numpy.uint8)


def _matching_over(check, weights, priors) -> pymatching.Matching:
    copied = check.copy()
    return pymatching.Matching.from_check_matrix(
        copied,
        weights=weights,
        error_probabilities=priors,
        merge_strategy="independent",
    )


def _column_rows(check, column: int) -> numpy.ndarray:
    start = check.indptr[column]
    stop = check.indptr[column + 1]
    return check.indices[start:stop]


def _syndrome_bits(syndrome) -> numpy.ndarray:
    bits = numpy.asarray(syndrome, dtype=numpy.uint8)
    return bits.ravel()


def _with_virtual_detector(bits, observable_bit: int) -> numpy.ndarray:
    """The syndrome with the augmented graph's observable detector appended."""
    extended = numpy.concatenate([bits, [observable_bit]])
    return extended.astype(numpy.uint8)
