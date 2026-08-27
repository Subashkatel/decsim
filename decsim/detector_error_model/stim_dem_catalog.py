"""Stim canonicalisation and global fault-catalog compilation.

Canonicalises Stim error instructions into physical and decomposed identities,
merges independent probabilities, compiles one global fault catalog per
requested domain, reconciles the linked graphlike/physical domains, and
dispatches the catalog set a DecoderFaultModelRequirement asks for.

Package-internal seams: _catalog_from_dem, _prepare_linked_fault_catalogs and
_prepare_fault_catalogs are package-private; _prepare_fault_catalogs is
imported by window_slicer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .fault_model_contracts import (
    DecoderFaultModelRequirement,
    FaultRepresentation,
    _FaultCatalog,
)
from .fault_identity_validation import (
    _xor_target_ids,
    validate_fault_identity,
    validate_graphlike_fault,
)


@dataclass(frozen=True)
class CanonicalErrorComponent:
    """One parity-reduced component of a physical Stim error instruction."""

    component_ordinal: int
    detectors: tuple[int, ...]
    logical_observables: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalErrorInstruction:
    """Both physical and decomposed identities of one Stim error mechanism."""

    error_ordinal: int
    probability: float
    aggregate_detectors: tuple[int, ...]
    aggregate_logical_observables: tuple[int, ...]
    components: tuple[CanonicalErrorComponent, ...]


def _merge_probability(current: float, incoming: float) -> float:
    """Merge independent faults with p (+) q = p(1-q) + q(1-p)."""
    return current * (1 - incoming) + incoming * (1 - current)


def _canonical_error_instruction(
    instruction,
    error_ordinal: int,
) -> Optional[CanonicalErrorInstruction]:
    """Parse one complete error before any consumer sees its components."""
    raw_components: list[tuple[list[int], list[int]]] = []
    component_detectors: list[int] = []
    component_logicals: list[int] = []
    aggregate_detectors: list[int] = []
    aggregate_logicals: list[int] = []

    for target in instruction.targets_copy():
        if target.is_separator():
            raw_components.append(
                (component_detectors, component_logicals)
            )
            component_detectors = []
            component_logicals = []
        elif target.is_relative_detector_id():
            component_detectors.append(target.val)
            aggregate_detectors.append(target.val)
        elif target.is_logical_observable_id():
            component_logicals.append(target.val)
            aggregate_logicals.append(target.val)
    raw_components.append((component_detectors, component_logicals))

    canonical_aggregate_detectors = _xor_target_ids(aggregate_detectors)
    canonical_aggregate_logicals = _xor_target_ids(aggregate_logicals)
    if not canonical_aggregate_detectors:
        if canonical_aggregate_logicals:
            raise ValueError(
                f"error {error_ordinal} is a detectorless logical "
                "mechanism after instruction-wide XOR reduction: "
                f"logical observables {canonical_aggregate_logicals}"
            )
        return None

    canonical_components: list[CanonicalErrorComponent] = []
    for component_ordinal, (
        raw_detectors,
        raw_logicals,
    ) in enumerate(raw_components):
        detectors = _xor_target_ids(raw_detectors)
        logical_observables = _xor_target_ids(raw_logicals)
        if not detectors:
            if logical_observables:
                raise ValueError(
                    f"error {error_ordinal} component "
                    f"{component_ordinal} is a detectorless logical "
                    "mechanism after component XOR reduction: "
                    f"logical observables {logical_observables}"
                )
            continue
        canonical_components.append(
            CanonicalErrorComponent(
                component_ordinal=component_ordinal,
                detectors=detectors,
                logical_observables=logical_observables,
            )
        )

    return CanonicalErrorInstruction(
        error_ordinal=error_ordinal,
        probability=float(instruction.args_copy()[0]),
        aggregate_detectors=canonical_aggregate_detectors,
        aggregate_logical_observables=canonical_aggregate_logicals,
        components=tuple(canonical_components),
    )


def canonical_error_instructions(dem) -> tuple[CanonicalErrorInstruction, ...]:
    """Return canonical physical errors from one flattened Stim DEM.

    Detector and logical identities are reduced across the complete
    instruction before its ``^``-separated components can be consumed.
    """
    records: list[CanonicalErrorInstruction] = []
    error_ordinal = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        record = _canonical_error_instruction(
            instruction,
            error_ordinal,
        )
        if record is not None:
            records.append(record)
        error_ordinal += 1
    return tuple(records)


def _odd_component_keys(record, validator) -> tuple:
    """Return component identities that occur oddly within one instruction.

    Stim's ``^`` separator partitions one correlated error mechanism; it does
    not create independent Bernoulli faults. Equal components therefore cancel
    modulo two before probabilities are merged across independent instructions.
    """
    odd: dict = {}
    for component in record.components:
        key = validator(
            component.detectors,
            component.logical_observables,
            location=(
                f"error {record.error_ordinal} component "
                f"{component.component_ordinal}"
            ),
        )
        if key in odd:
            del odd[key]
        else:
            odd[key] = None
    return tuple(odd)


def detector_error_model_to_faults(dem) -> tuple:
    """Convert a Stim detector error model into merged fault columns."""
    merged: dict = {}
    for record in canonical_error_instructions(dem):
        for key in _odd_component_keys(record, validate_fault_identity):
            current_probability = merged.get(key, 0.0)
            merged[key] = _merge_probability(
                current_probability,
                record.probability,
            )
    det_sets = [k[0] for k in merged]
    obs_sets = [k[1] for k in merged]
    priors = list(merged.values())
    return det_sets, obs_sets, priors


def _catalog_from_dem(
    dem,
    representation: FaultRepresentation,
) -> _FaultCatalog:
    """Build one independently sourced global fault catalog."""
    if representation is FaultRepresentation.GRAPHLIKE:
        detector_sets, observable_sets, priors = detector_error_model_to_faults(dem)
    else:
        merged = {}
        for record in canonical_error_instructions(dem):
            key = (
                record.aggregate_detectors,
                record.aggregate_logical_observables,
            )
            merged[key] = _merge_probability(
                merged.get(key, 0.0), record.probability)
        detector_sets = [key[0] for key in merged]
        observable_sets = [key[1] for key in merged]
        priors = list(merged.values())
    return _FaultCatalog(
        representation=representation,
        detector_sets=tuple(tuple(values) for values in detector_sets),
        observable_sets=tuple(tuple(values) for values in observable_sets),
        priors=tuple(float(value) for value in priors),
    )


def _prepare_linked_fault_catalogs(decomposed_dem, physical_dem):
    """Keep distinct physical mechanisms when their graph decompositions differ.

    Global half of the linked fault domain; the window-local half is
    window_placement._local_physical_to_graphlike_detector_projection.
    """
    import numpy as np

    graphlike_catalog = _catalog_from_dem(
        decomposed_dem,
        FaultRepresentation.GRAPHLIKE,
    )
    graphlike_index = {
        (detectors, observables): index
        for index, (detectors, observables) in enumerate(zip(
            graphlike_catalog.detector_sets,
            graphlike_catalog.observable_sets,
        ))
    }
    mechanisms: dict[
        tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        ],
        float,
    ] = {}
    for record in canonical_error_instructions(decomposed_dem):
        physical_key = (
            record.aggregate_detectors,
            record.aggregate_logical_observables,
        )
        component_keys = tuple(sorted(_odd_component_keys(
            record,
            validate_graphlike_fault,
        )))
        mechanism_key = (physical_key, component_keys)
        mechanisms[mechanism_key] = _merge_probability(
            mechanisms.get(mechanism_key, 0.0),
            record.probability,
        )

    physical_detector_sets = []
    physical_observable_sets = []
    physical_priors = []
    link_rows: list = []
    link_columns: list = []
    for physical_column, ((physical_key, component_keys), prior) in enumerate(
        mechanisms.items()
    ):
        physical_detector_sets.append(physical_key[0])
        physical_observable_sets.append(physical_key[1])
        physical_priors.append(prior)
        for component_key in component_keys:
            link_rows.append(graphlike_index[component_key])
            link_columns.append(physical_column)
    from scipy.sparse import csc_matrix
    link = csc_matrix(
        (np.ones(len(link_rows), dtype=np.uint8), (link_rows, link_columns)),
        shape=(len(graphlike_catalog.detector_sets), len(mechanisms)),
    )

    physical_catalog = _FaultCatalog(
        representation=FaultRepresentation.PHYSICAL,
        detector_sets=tuple(physical_detector_sets),
        observable_sets=tuple(physical_observable_sets),
        priors=tuple(physical_priors),
    )
    undecomposed_catalog = _catalog_from_dem(
        physical_dem,
        FaultRepresentation.PHYSICAL,
    )
    undecomposed = {
        (detectors, observables): prior
        for detectors, observables, prior in zip(
            undecomposed_catalog.detector_sets,
            undecomposed_catalog.observable_sets,
            undecomposed_catalog.priors,
        )
    }
    reconstructed: dict = {}
    for detectors, observables, prior in zip(
        physical_catalog.detector_sets,
        physical_catalog.observable_sets,
        physical_catalog.priors,
    ):
        key = (detectors, observables)
        reconstructed[key] = _merge_probability(
            reconstructed.get(key, 0.0), prior)
    if set(reconstructed) != set(undecomposed) or any(
        not math.isclose(reconstructed[key], undecomposed[key], rel_tol=0, abs_tol=1e-15)
        for key in reconstructed
    ):
        raise ValueError(
            "decomposed and undecomposed Stim models disagree on physical faults"
        )

    # The link audit works on sparse incidence matrices; a dense
    # target-by-fault matrix is quadratic in circuit length and cannot
    # exist at stress size (12 TiB at d=9, 10000 rounds), while the sparse
    # form is the reference decoders' own shape for this structure
    # (beliefmatching's check/hyperedge_to_edge matrices are csc_matrix).
    detector_row_count = max(
        (detector_id for detectors in graphlike_catalog.detector_sets
         for detector_id in detectors), default=-1) + 1
    graph_check = _incidence_matrix(
        graphlike_catalog.detector_sets, detector_row_count)
    derived_check = _incidence_matrix(
        physical_catalog.detector_sets, detector_row_count)
    if _parity_differs(graph_check @ link, derived_check):
        raise ValueError(
            "physical detector effects do not equal their graphlike components"
        )
    observable_row_count = max(
        (observable_id
         for observable_sets in (graphlike_catalog.observable_sets,
                                 physical_catalog.observable_sets)
         for observables in observable_sets
         for observable_id in observables), default=-1) + 1
    graph_observables = _incidence_matrix(
        graphlike_catalog.observable_sets, observable_row_count)
    physical_observables = _incidence_matrix(
        physical_catalog.observable_sets, observable_row_count)
    if _parity_differs(graph_observables @ link, physical_observables):
        raise ValueError(
            "physical logical effects do not equal their graphlike components"
        )
    return graphlike_catalog, physical_catalog, link


def _incidence_matrix(target_sets: tuple, row_count: int) -> csc_matrix:
    """Sparse target-by-fault incidence: entry (t, f) = 1 when fault f
    touches target t (a detector or an observable)."""
    import numpy as np
    from scipy.sparse import csc_matrix
    rows = [target_id for targets in target_sets for target_id in targets]
    columns = [column for column, targets in enumerate(target_sets)
               for _ in targets]
    return csc_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, columns)),
        shape=(row_count, len(target_sets)))


def _parity_differs(component_counts: csc_matrix, expected: csc_matrix) -> bool:
    """Whether the mod-2 reduction of a component-count matrix differs
    anywhere from the expected 0/1 incidence."""
    import numpy as np
    parity = component_counts.astype(np.int64)
    parity.data %= 2
    parity.eliminate_zeros()
    return (parity != expected).nnz != 0


def _prepare_fault_catalogs(
    circuit,
    requirement: DecoderFaultModelRequirement,
) -> tuple[dict[FaultRepresentation, _FaultCatalog], "object"]:
    """Build only the fault domains requested for this operation's code."""
    catalogs: dict[FaultRepresentation, _FaultCatalog] = {}
    if requirement.require_physical_to_graphlike_link:
        graphlike, physical, link = _prepare_linked_fault_catalogs(
            circuit.detector_error_model(decompose_errors=True),
            circuit.detector_error_model(decompose_errors=False),
        )
        catalogs[FaultRepresentation.GRAPHLIKE] = graphlike
        catalogs[FaultRepresentation.PHYSICAL] = physical
        return catalogs, link

    if FaultRepresentation.GRAPHLIKE in requirement.representations:
        catalogs[FaultRepresentation.GRAPHLIKE] = _catalog_from_dem(
            circuit.detector_error_model(decompose_errors=True),
            FaultRepresentation.GRAPHLIKE,
        )
    if FaultRepresentation.PHYSICAL in requirement.representations:
        catalogs[FaultRepresentation.PHYSICAL] = _catalog_from_dem(
            circuit.detector_error_model(decompose_errors=False),
            FaultRepresentation.PHYSICAL,
        )
    return catalogs, None
