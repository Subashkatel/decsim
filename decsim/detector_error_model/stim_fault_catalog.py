"""The whole-circuit fault catalog, read off Stim's detector error model.

Every `error(p) D.. L.. ^ D.. L..` instruction (Stim,
doc/file_format_dem_detector_error_model.md) is reduced modulo two, as a
whole for the physical identity and per `^` component for the graphlike
ones, and mechanisms with one identity merge as independent errors,
p(1-q) + q(1-p) (PyMatching's merge_strategy="independent"). When a
decoder needs both catalogs, the link matrix says which graphlike
columns each physical column is made of.
"""

import dataclasses
import math
from typing import Optional

import numpy
import scipy.sparse
import stim

from decsim.detector_error_model import (
    fault_identity_validation,
    fault_model_contracts,
)


@dataclasses.dataclass(frozen=True)
class CanonicalErrorComponent:
    """One graphlike component of a Stim error, reduced modulo two."""

    component_ordinal: int
    detectors: tuple[int, ...]
    logical_observables: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class CanonicalErrorInstruction:
    """One Stim error: its whole identity, then its components."""

    error_ordinal: int
    probability: float
    aggregate_detectors: tuple[int, ...]
    aggregate_logical_observables: tuple[int, ...]
    components: tuple[CanonicalErrorComponent, ...]


def canonical_error_instructions(
    detector_error_model: stim.DetectorErrorModel,
) -> tuple[CanonicalErrorInstruction, ...]:
    """Every error of a flattened Stim model that flips a detector.

    The ordinal counts every error instruction, including the ones that
    flip nothing and are left out.
    """
    records: list[CanonicalErrorInstruction] = []
    error_ordinal = 0
    for instruction in detector_error_model.flattened():
        if instruction.type != "error":
            continue
        record = _canonical_error_instruction(instruction, error_ordinal)
        if record is not None:
            records.append(record)
        error_ordinal += 1
    return tuple(records)


def detector_error_model_to_faults(
    detector_error_model: stim.DetectorErrorModel,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[float]]:
    """The graphlike columns of a Stim model: detectors, observables, priors.

    A component that appears in several instructions is one column whose
    prior merges them as independent errors.
    """
    merged: dict = {}
    for record in canonical_error_instructions(detector_error_model):
        keys = _odd_component_keys(
            record, fault_identity_validation.validate_fault_identity
        )
        for key in keys:
            current_probability = merged.get(key, 0.0)
            merged[key] = _merge_probability(
                current_probability, record.probability
            )
    detector_sets = [key[0] for key in merged]
    observable_sets = [key[1] for key in merged]
    merged_priors = merged.values()
    priors = list(merged_priors)
    return detector_sets, observable_sets, priors


def prepare_fault_catalogs(
    circuit: stim.Circuit,
    requirement: fault_model_contracts.DecoderFaultModelRequirement,
) -> tuple[
    dict[
        fault_model_contracts.FaultRepresentation,
        fault_model_contracts.FaultCatalog,
    ],
    Optional[scipy.sparse.csc_matrix],
]:
    """The catalogs a requirement asks for, and the link when it asks for it.

    Returns (catalog by representation, link or None).
    """
    if requirement.require_physical_to_graphlike_link:
        return _linked_catalogs(circuit)
    catalogs: dict = {}
    graphlike = fault_model_contracts.FaultRepresentation.GRAPHLIKE
    if graphlike in requirement.representations:
        decomposed_model = circuit.detector_error_model(decompose_errors=True)
        catalogs[graphlike] = _catalog_from_detector_error_model(
            decomposed_model, graphlike
        )
    physical = fault_model_contracts.FaultRepresentation.PHYSICAL
    if physical in requirement.representations:
        undecomposed_model = circuit.detector_error_model(
            decompose_errors=False
        )
        catalogs[physical] = _catalog_from_detector_error_model(
            undecomposed_model, physical
        )
    return catalogs, None


def _merge_probability(current: float, incoming: float) -> float:
    """Independent faults combine as p (+) q = p(1-q) + q(1-p)."""
    return current * (1 - incoming) + incoming * (1 - current)


def _linked_catalogs(
    circuit: stim.Circuit,
) -> tuple[
    dict[
        fault_model_contracts.FaultRepresentation,
        fault_model_contracts.FaultCatalog,
    ],
    scipy.sparse.csc_matrix,
]:
    """Both catalogs and their link, keyed by representation."""
    decomposed_model = circuit.detector_error_model(decompose_errors=True)
    undecomposed_model = circuit.detector_error_model(decompose_errors=False)
    graphlike, physical, link = _prepare_linked_fault_catalogs(
        decomposed_model, undecomposed_model
    )
    catalogs = {
        fault_model_contracts.FaultRepresentation.GRAPHLIKE: graphlike,
        fault_model_contracts.FaultRepresentation.PHYSICAL: physical,
    }
    return catalogs, link


def _canonical_error_instruction(
    instruction, error_ordinal: int
) -> Optional[CanonicalErrorInstruction]:
    """One error reduced modulo two, or None when it flips no detector."""
    splitter = _TargetSplitter()
    for target in instruction.targets_copy():
        splitter.note(target)
    raw_components = splitter.finished_components()
    aggregate_detectors = fault_identity_validation.xor_target_ids(
        splitter.all_detectors
    )
    aggregate_logicals = fault_identity_validation.xor_target_ids(
        splitter.all_logicals
    )
    if not aggregate_detectors:
        if aggregate_logicals:
            raise ValueError(
                f"error {error_ordinal} is a detectorless logical "
                "mechanism after instruction-wide XOR reduction: "
                f"logical observables {aggregate_logicals}"
            )
        return None
    components = _canonical_components(raw_components, error_ordinal)
    arguments = instruction.args_copy()
    return CanonicalErrorInstruction(
        error_ordinal=error_ordinal,
        probability=float(arguments[0]),
        aggregate_detectors=aggregate_detectors,
        aggregate_logical_observables=aggregate_logicals,
        components=tuple(components),
    )


class _TargetSplitter:
    """Splits one error's targets at its `^` separators.

    Keeps each component's detectors and logicals as they arrive, and the
    whole instruction's lists beside them; nothing is reduced here.
    """

    def __init__(self):
        self.components: list = []
        self.component_detectors: list[int] = []
        self.component_logicals: list[int] = []
        self.all_detectors: list[int] = []
        self.all_logicals: list[int] = []

    def note(self, target: stim.DemTarget) -> None:
        """Place one target in the current component."""
        if target.is_separator():
            self._close_component()
            return
        if target.is_relative_detector_id():
            self.component_detectors.append(target.val)
            self.all_detectors.append(target.val)
            return
        if target.is_logical_observable_id():
            self.component_logicals.append(target.val)
            self.all_logicals.append(target.val)

    def finished_components(self) -> list:
        """Every component as a (detectors, logicals) pair."""
        self._close_component()
        return self.components

    def _close_component(self) -> None:
        self.components.append(
            (self.component_detectors, self.component_logicals)
        )
        self.component_detectors = []
        self.component_logicals = []


def _canonical_components(
    raw_components: list, error_ordinal: int
) -> list[CanonicalErrorComponent]:
    """Each component reduced modulo two; empty components are dropped."""
    components: list[CanonicalErrorComponent] = []
    for component_ordinal, raw_component in enumerate(raw_components):
        raw_detectors, raw_logicals = raw_component
        detectors = fault_identity_validation.xor_target_ids(raw_detectors)
        logicals = fault_identity_validation.xor_target_ids(raw_logicals)
        if not detectors:
            _refuse_detectorless(logicals, error_ordinal, component_ordinal)
            continue
        component = CanonicalErrorComponent(
            component_ordinal=component_ordinal,
            detectors=detectors,
            logical_observables=logicals,
        )
        components.append(component)
    return components


def _refuse_detectorless(
    logicals: tuple, error_ordinal: int, component_ordinal: int
) -> None:
    """A component that flips an observable but no detector is refused."""
    if logicals:
        raise ValueError(
            f"error {error_ordinal} component "
            f"{component_ordinal} is a detectorless logical "
            "mechanism after component XOR reduction: "
            f"logical observables {logicals}"
        )


def _odd_component_keys(record, validator) -> tuple:
    """The component identities that occur an odd number of times.

    Stim's `^` partitions one correlated mechanism; it does not list
    independent faults. Equal components therefore cancel modulo two
    before probabilities are merged across instructions.
    """
    odd: dict = {}
    for component in record.components:
        location = (
            f"error {record.error_ordinal} component "
            f"{component.component_ordinal}"
        )
        key = validator(
            component.detectors,
            component.logical_observables,
            location=location,
        )
        if key in odd:
            del odd[key]
        else:
            odd[key] = None
    return tuple(odd)


def _catalog_from_detector_error_model(
    detector_error_model,
    representation: fault_model_contracts.FaultRepresentation,
) -> fault_model_contracts.FaultCatalog:
    """One catalog from one Stim model, in the requested representation."""
    graphlike = fault_model_contracts.FaultRepresentation.GRAPHLIKE
    if representation is graphlike:
        detector_sets, observable_sets, priors = detector_error_model_to_faults(
            detector_error_model
        )
        _check_every_fault_is_graphlike(detector_sets, observable_sets)
    else:
        merged: dict = {}
        for record in canonical_error_instructions(detector_error_model):
            key = (
                record.aggregate_detectors,
                record.aggregate_logical_observables,
            )
            current_probability = merged.get(key, 0.0)
            merged[key] = _merge_probability(
                current_probability, record.probability
            )
        detector_sets = [key[0] for key in merged]
        observable_sets = [key[1] for key in merged]
        merged_priors = merged.values()
        priors = list(merged_priors)
    return fault_model_contracts.FaultCatalog(
        representation=representation,
        detector_sets=tuple(tuple(values) for values in detector_sets),
        observable_sets=tuple(tuple(values) for values in observable_sets),
        priors=tuple(float(value) for value in priors),
    )


def _check_every_fault_is_graphlike(
    detector_sets: list, observable_sets: list
) -> None:
    """A graphlike catalog holds only faults a matching decoder can take."""
    identities = zip(detector_sets, observable_sets)
    for fault_index, (detectors, observables) in enumerate(identities):
        fault_identity_validation.validate_graphlike_fault(
            detectors,
            observables,
            location=f"graphlike catalog fault {fault_index}",
        )


def _prepare_linked_fault_catalogs(
    decomposed_model: stim.DetectorErrorModel,
    undecomposed_model: stim.DetectorErrorModel,
) -> tuple[
    fault_model_contracts.FaultCatalog,
    fault_model_contracts.FaultCatalog,
    scipy.sparse.csc_matrix,
]:
    """Both catalogs and the graphlike-by-physical link between them.

    A physical mechanism is keyed by its whole identity and by its
    component set, so two mechanisms that flip the same detectors but
    decompose differently stay distinct columns. Every physical column is
    the parity of its linked components by construction: its identity is
    the whole instruction reduced modulo two, and its components partition
    that instruction. The window-local half of this link is the detector
    projection in window_placement.
    """
    graphlike_catalog = _catalog_from_detector_error_model(
        decomposed_model, fault_model_contracts.FaultRepresentation.GRAPHLIKE
    )
    column_by_identity = _column_by_identity(graphlike_catalog)
    mechanisms = _mechanisms_by_key(decomposed_model)
    physical_catalog, link = _physical_catalog_and_link(
        mechanisms, column_by_identity, len(graphlike_catalog.detector_sets)
    )
    undecomposed_catalog = _catalog_from_detector_error_model(
        undecomposed_model, fault_model_contracts.FaultRepresentation.PHYSICAL
    )
    _check_same_physical_faults(physical_catalog, undecomposed_catalog)
    return graphlike_catalog, physical_catalog, link


def _column_by_identity(catalog: fault_model_contracts.FaultCatalog) -> dict:
    """Each (detectors, observables) identity's column in the catalog."""
    identities = zip(catalog.detector_sets, catalog.observable_sets)
    return {identity: column for column, identity in enumerate(identities)}


def _mechanisms_by_key(decomposed_model) -> dict:
    """Merged prior of every (whole identity, component identities) pair."""
    mechanisms: dict = {}
    for record in canonical_error_instructions(decomposed_model):
        physical_key = (
            record.aggregate_detectors,
            record.aggregate_logical_observables,
        )
        component_keys = _odd_component_keys(
            record, fault_identity_validation.validate_graphlike_fault
        )
        mechanism_key = (physical_key, tuple(sorted(component_keys)))
        current_probability = mechanisms.get(mechanism_key, 0.0)
        mechanisms[mechanism_key] = _merge_probability(
            current_probability, record.probability
        )
    return mechanisms


def _physical_catalog_and_link(
    mechanisms: dict, column_by_identity: dict, graphlike_count: int
) -> tuple[fault_model_contracts.FaultCatalog, scipy.sparse.csc_matrix]:
    """The physical catalog and its link: a one per graphlike component."""
    physical_detector_sets = []
    physical_observable_sets = []
    physical_priors = []
    link_rows: list = []
    link_columns: list = []
    mechanism_items = mechanisms.items()
    for physical_column, (mechanism_key, prior) in enumerate(mechanism_items):
        physical_key, component_keys = mechanism_key
        physical_detector_sets.append(physical_key[0])
        physical_observable_sets.append(physical_key[1])
        physical_priors.append(prior)
        for component_key in component_keys:
            link_rows.append(column_by_identity[component_key])
            link_columns.append(physical_column)
    ones = numpy.ones(len(link_rows), dtype=numpy.uint8)
    link = scipy.sparse.csc_matrix(
        (ones, (link_rows, link_columns)),
        shape=(graphlike_count, len(mechanisms)),
    )
    physical_catalog = fault_model_contracts.FaultCatalog(
        representation=fault_model_contracts.FaultRepresentation.PHYSICAL,
        detector_sets=tuple(physical_detector_sets),
        observable_sets=tuple(physical_observable_sets),
        priors=tuple(physical_priors),
    )
    return physical_catalog, link


def _check_same_physical_faults(
    physical_catalog: fault_model_contracts.FaultCatalog,
    undecomposed_catalog: fault_model_contracts.FaultCatalog,
) -> None:
    """Stim's two models must agree on every physical fault and prior."""
    undecomposed = {}
    undecomposed_identities = zip(
        undecomposed_catalog.detector_sets, undecomposed_catalog.observable_sets
    )
    for key, prior in zip(undecomposed_identities, undecomposed_catalog.priors):
        undecomposed[key] = prior
    reconstructed: dict = {}
    identities = zip(
        physical_catalog.detector_sets, physical_catalog.observable_sets
    )
    for key, prior in zip(identities, physical_catalog.priors):
        current_probability = reconstructed.get(key, 0.0)
        reconstructed[key] = _merge_probability(current_probability, prior)
    if set(reconstructed) != set(undecomposed):
        raise ValueError(
            "decomposed and undecomposed Stim models disagree on "
            "physical faults"
        )
    for key, prior in reconstructed.items():
        if not math.isclose(prior, undecomposed[key], rel_tol=0, abs_tol=1e-15):
            raise ValueError(
                "decomposed and undecomposed Stim models disagree on "
                "physical faults"
            )
