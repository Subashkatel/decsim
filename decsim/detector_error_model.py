"""Slice a global Stim detector error model into per-window decoder inputs.

The measured bits change every shot; the sliced error models are compile-time
data shared across shots. Consumed by the window decoders (MWPM / BP+OSD /
belief matching) and built per op by adapters/stim_device.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FaultRepresentation(Enum):
    """Fault-column domain consumed and returned by a decoder."""

    GRAPHLIKE = "graphlike"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class DecoderFaultModelRequirement:
    """The exact fault views a decoder needs for one operation's code."""

    representations: frozenset[FaultRepresentation] = frozenset()
    require_physical_to_graphlike_link: bool = False

    def __post_init__(self) -> None:
        if type(self.representations) is not frozenset:
            raise TypeError("representations must be an exact frozenset")
        invalid = [
            representation
            for representation in self.representations
            if not isinstance(representation, FaultRepresentation)
        ]
        if invalid:
            raise TypeError(
                "representations must contain only FaultRepresentation values"
            )
        if type(self.require_physical_to_graphlike_link) is not bool:
            raise TypeError(
                "require_physical_to_graphlike_link must be an exact bool"
            )
        if self.require_physical_to_graphlike_link and self.representations != frozenset(
            {FaultRepresentation.GRAPHLIKE, FaultRepresentation.PHYSICAL}
        ):
            raise ValueError(
                "a physical-to-graphlike link requires both fault representations"
            )

    def joined(
        self,
        other: "DecoderFaultModelRequirement",
    ) -> "DecoderFaultModelRequirement":
        """Return the smallest requirement satisfying both consumers."""
        if not isinstance(other, DecoderFaultModelRequirement):
            raise TypeError("can join only DecoderFaultModelRequirement values")
        return DecoderFaultModelRequirement(
            self.representations | other.representations,
            self.require_physical_to_graphlike_link
            or other.require_physical_to_graphlike_link,
        )


NO_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement()
GRAPHLIKE_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(
    frozenset({FaultRepresentation.GRAPHLIKE})
)
PHYSICAL_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(
    frozenset({FaultRepresentation.PHYSICAL})
)
LINKED_FAULT_MODELS_REQUIRED = DecoderFaultModelRequirement(
    frozenset({FaultRepresentation.GRAPHLIKE, FaultRepresentation.PHYSICAL}),
    require_physical_to_graphlike_link=True,
)


@dataclass(frozen=True)
class PlacedFaultModel:
    """Column-aligned arrays for one explicitly named fault domain."""

    representation: FaultRepresentation
    check: "object"
    priors: "object"
    observables: "object"
    owned: "object"
    future_flips: dict
    source_fault_ids: tuple[int, ...]


@dataclass(frozen=True)
class _FaultCatalog:
    """One global fault domain before its columns are placed into windows."""

    representation: FaultRepresentation
    detector_sets: tuple[tuple[int, ...], ...]
    observable_sets: tuple[tuple[int, ...], ...]
    priors: tuple[float, ...]


@dataclass(frozen=True)
class WindowErrorModel:
    """One window's detector rows, fault columns, and boundary handoff data."""

    detector_ids: tuple
    detector_coordinates: Optional[tuple[tuple[float, ...], ...]]
    commit_hi: int
    defect_positions: dict
    graphlike_faults: Optional[PlacedFaultModel]
    physical_faults: Optional[PlacedFaultModel]
    physical_to_graphlike_detector_projection: "object" = None
    commit_lo: int = 0
    buffer_lo: int = 0

    def require_faults(
        self,
        representation: FaultRepresentation,
    ) -> PlacedFaultModel:
        """Return one requested view or fail at the consuming boundary."""
        if representation is FaultRepresentation.GRAPHLIKE:
            faults = self.graphlike_faults
        elif representation is FaultRepresentation.PHYSICAL:
            faults = self.physical_faults
        else:
            raise TypeError(
                "representation must be a FaultRepresentation value"
            )
        if faults is None:
            raise ValueError(
                f"window model does not contain {representation.value} faults"
            )
        return faults

    @property
    def has_leading_buffer(self) -> bool:
        """True when ``buffer_lo < commit_lo`` (two-sided parallel A/B window with lookback)."""
        # ref: Skoric 2209.08552 sec. III.C, Tan 2209.09219 Eq. S10 (w = s + 2b)
        return self.buffer_lo < self.commit_lo


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


def _xor_target_ids(target_ids) -> tuple[int, ...]:
    """Return the sorted ids with even multiplicities removed."""
    odd_ids: set[int] = set()
    for target_id in target_ids:
        if target_id in odd_ids:
            odd_ids.remove(target_id)
        else:
            odd_ids.add(target_id)
    return tuple(sorted(odd_ids))


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


def validate_fault_identity(
    detector_ids,
    logical_observable_ids,
    *,
    location: str,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Canonicalize one placed fault without changing its decoder domain."""
    detectors = _xor_target_ids(detector_ids)
    logical_observables = _xor_target_ids(logical_observable_ids)
    if not detectors:
        if logical_observables:
            raise ValueError(
                f"{location} is a detectorless logical fault: "
                f"logical observables {logical_observables}"
            )
        return None
    return detectors, logical_observables


def validate_graphlike_fault(
    detector_ids,
    logical_observable_ids,
    *,
    location: str,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Canonicalize one fault and require the one-/two-detector domain."""
    fault = validate_fault_identity(
        detector_ids,
        logical_observable_ids,
        location=location,
    )
    if fault is None:
        return None
    detectors, logical_observables = fault
    if len(detectors) > 2:
        raise ValueError(
            f"{location} is a detector hyperedge with detectors {detectors}; "
            "this graphlike decoder supports one or two detectors per fault"
        )
    return detectors, logical_observables


def _binary_matrix(value, *, location: str, name: str):
    """Return one rank-2 binary matrix without changing its identities."""
    import numpy as np

    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{location} {name} must be a rank-2 matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{location} {name} must contain only binary values")
    return matrix.astype(np.uint8, copy=False)


def _placed_matrix_faults(
    check,
    observables,
    *,
    location: str,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    """Return canonical ids for every placed matrix fault column."""
    import numpy as np

    check_matrix = _binary_matrix(
        check,
        location=location,
        name="check",
    )
    observable_matrix = _binary_matrix(
        observables,
        location=location,
        name="observable matrix",
    )
    if check_matrix.shape[1] != observable_matrix.shape[1]:
        raise ValueError(
            f"{location} check and observable matrices have different fault "
            f"counts: {check_matrix.shape[1]} and "
            f"{observable_matrix.shape[1]}"
        )
    faults = []
    for fault_index in range(check_matrix.shape[1]):
        detector_ids = np.nonzero(check_matrix[:, fault_index])[0]
        logical_observable_ids = np.nonzero(
            observable_matrix[:, fault_index]
        )[0]
        faults.append(
            (
                fault_index,
                tuple(int(value) for value in detector_ids),
                tuple(int(value) for value in logical_observable_ids),
            )
        )
    return tuple(faults)


def validate_placed_fault_matrices(
    check,
    observables,
    *,
    location: str,
) -> None:
    """Reject lost logical identity while preserving a decoder's degree domain."""
    for fault_index, detector_ids, logical_ids in _placed_matrix_faults(
        check,
        observables,
        location=location,
    ):
        validate_fault_identity(
            detector_ids,
            logical_ids,
            location=f"{location} column {fault_index}",
        )


def validate_graphlike_matrices(
    check,
    observables,
    *,
    location: str,
) -> None:
    """Validate every placed fault column at a graphlike consumer boundary."""
    for fault_index, detector_ids, logical_ids in _placed_matrix_faults(
        check,
        observables,
        location=location,
    ):
        validate_graphlike_fault(
            detector_ids,
            logical_ids,
            location=f"{location} column {fault_index}",
        )


def validate_belief_matching_matrices(
    check,
    observables,
    hyperedge_check,
    hyperedge_priors,
    hyperedge_to_edge,
    *,
    location: str,
) -> None:
    """Validate the linked physical and graphlike belief-matching domains."""
    import numpy as np

    check_matrix = _binary_matrix(
        check,
        location=location,
        name="component check",
    )
    observable_matrix = _binary_matrix(
        observables,
        location=location,
        name="component observable matrix",
    )
    hyperedge_check_matrix = _binary_matrix(
        hyperedge_check,
        location=location,
        name="physical check",
    )
    hyperedge_to_edge_matrix = _binary_matrix(
        hyperedge_to_edge,
        location=location,
        name="physical-to-component map",
    )
    if check_matrix.shape[1] != observable_matrix.shape[1]:
        raise ValueError(
            f"{location} component check and observable matrices have "
            "different fault counts"
        )
    if check_matrix.shape[0] != hyperedge_check_matrix.shape[0]:
        raise ValueError(
            f"{location} component and physical checks have different "
            "detector counts"
        )
    expected_map_shape = (
        check_matrix.shape[1],
        hyperedge_check_matrix.shape[1],
    )
    if hyperedge_to_edge_matrix.shape != expected_map_shape:
        raise ValueError(
            f"{location} physical-to-component map has shape "
            f"{hyperedge_to_edge_matrix.shape}; expected {expected_map_shape}"
        )

    try:
        priors = np.asarray(hyperedge_priors, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{location} physical priors must be real probabilities"
        ) from error
    if priors.ndim != 1:
        raise ValueError(f"{location} physical priors must be rank 1")
    if priors.shape[0] != hyperedge_check_matrix.shape[1]:
        raise ValueError(
            f"{location} has {priors.shape[0]} physical priors for "
            f"{hyperedge_check_matrix.shape[1]} physical fault columns"
        )
    if not np.all(np.isfinite(priors)):
        raise ValueError(f"{location} physical priors must be finite")
    if not np.all((0.0 <= priors) & (priors <= 1.0)):
        raise ValueError(
            f"{location} physical priors must lie in the inclusive range [0, 1]"
        )

    validate_graphlike_matrices(
        check_matrix,
        observable_matrix,
        location=f"{location} component graph",
    )
    for physical_index in range(hyperedge_check_matrix.shape[1]):
        component_indices = np.nonzero(
            hyperedge_to_edge_matrix[:, physical_index]
        )[0]
        if not component_indices.size:
            raise ValueError(
                f"{location} physical column {physical_index} has no "
                "suggested graph components"
            )
        derived_detector_bits = (
            check_matrix[:, component_indices].sum(axis=1) % 2
        )
        stored_detector_bits = hyperedge_check_matrix[:, physical_index]
        if not np.array_equal(derived_detector_bits, stored_detector_bits):
            raise ValueError(
                f"{location} physical column {physical_index} detector "
                "identity does not equal its component XOR"
            )
        derived_logical_bits = (
            observable_matrix[:, component_indices].sum(axis=1) % 2
        )
        identity = validate_fault_identity(
            np.nonzero(stored_detector_bits)[0],
            np.nonzero(derived_logical_bits)[0],
            location=f"{location} physical column {physical_index}",
        )
        if identity is None:
            raise ValueError(
                f"{location} physical column {physical_index} is inert and "
                "must be removed before belief matching"
            )


def detector_error_model_to_faults(dem) -> tuple:
    """Convert a Stim detector error model into merged fault columns."""
    merged: dict = {}
    for record in canonical_error_instructions(dem):
        for component in record.components:
            key = validate_fault_identity(
                component.detectors,
                component.logical_observables,
                location=(
                    f"error {record.error_ordinal} component "
                    f"{component.component_ordinal}"
                ),
            )
            assert key is not None
            current_probability = merged.get(key, 0.0)
            merged[key] = _merge_probability(
                current_probability,
                record.probability,
            )
    det_sets = [k[0] for k in merged]
    obs_sets = [k[1] for k in merged]
    priors = list(merged.values())
    return det_sets, obs_sets, priors


def detector_error_model_to_faults_bm(dem) -> tuple:
    """Return decomposed edge faults plus belief-matching hyperedge data."""
    import numpy as np

    edge_merged: dict = {}
    hyper_merged: dict = {}
    hyper_list: list = []
    h2e_pairs: set = set()

    for record in canonical_error_instructions(dem):
        hyperedge_key = (
            record.aggregate_detectors,
            record.aggregate_logical_observables,
        )

        hyperedge_index = hyper_merged.get(hyperedge_key)
        is_first_decomposition = hyperedge_index is None
        if is_first_decomposition:
            hyperedge_index = len(hyper_list)
            hyper_merged[hyperedge_key] = hyperedge_index
            hyper_list.append([hyperedge_key[0], hyperedge_key[1], 0.0])
        hyper_list[hyperedge_index][2] = _merge_probability(
            hyper_list[hyperedge_index][2],
            record.probability,
        )

        for component in record.components:
            component_key = validate_graphlike_fault(
                component.detectors,
                component.logical_observables,
                location=(
                    f"error {record.error_ordinal} component "
                    f"{component.component_ordinal}"
                ),
            )
            assert component_key is not None
            current = edge_merged.get(component_key, 0.0)
            edge_merged[component_key] = _merge_probability(
                current,
                record.probability,
            )
            if is_first_decomposition:
                pair = (hyperedge_index, component_key)
                if pair in h2e_pairs:
                    h2e_pairs.remove(pair)
                else:
                    h2e_pairs.add(pair)

    edge_keys = list(edge_merged)
    edge_index = {k: i for i, k in enumerate(edge_keys)}
    hyperedge_to_edge = np.zeros((len(edge_keys), len(hyper_list)), dtype=np.uint8)
    for hyperedge_index, component_key in h2e_pairs:
        hyperedge_to_edge[edge_index[component_key], hyperedge_index] = 1

    return ([k[0] for k in edge_keys], [k[1] for k in edge_keys],
            [edge_merged[k] for k in edge_keys],
            [h[0] for h in hyper_list], [h[2] for h in hyper_list],
            hyperedge_to_edge)


def _detector_rounds_from_circuit(circuit, detector_rounds: Optional[dict]) -> dict:
    """Return global detector id -> 1-based syndrome round."""
    if detector_rounds is not None:
        return dict(detector_rounds)

    coords = circuit.get_detector_coordinates()
    coordless = sum(1 for coord in coords.values() if not coord)
    if coordless:
        raise ValueError(
            f"{coordless} detectors carry no coordinates; pass detector_rounds "
            "(global detector id -> 1-based round) explicitly")
    return {detector_id: int(coord[-1]) + 1
            for detector_id, coord in coords.items()}


def _detector_position_in_round(round_of: dict) -> dict:
    """Return detector id -> index within its round, using Stim detector order."""
    detectors_by_round: dict = {}
    for detector_id in sorted(round_of):
        round_index = round_of[detector_id]
        detectors_by_round.setdefault(round_index, []).append(detector_id)
    return {detector_id: index
            for detectors in detectors_by_round.values()
            for index, detector_id in enumerate(detectors)}


def _parse_window_entry(window_entry: tuple) -> tuple[int, int, int, int]:
    """Normalize a 3-value or 4-value window plan entry."""
    if len(window_entry) == 4:
        return window_entry

    commit_lo, commit_hi, buffer_hi = window_entry
    buffer_lo = commit_lo
    return buffer_lo, commit_lo, commit_hi, buffer_hi


def _detectors_in_window(round_of: dict, buffer_lo: int, buffer_hi: int,
                         *, is_last: bool) -> list:
    """Choose the detector rows for this window."""
    if is_last:
        return sorted(detector_id
                      for detector_id, round_index in round_of.items()
                      if round_index >= buffer_lo)

    return sorted(detector_id
                  for detector_id, round_index in round_of.items()
                  if buffer_lo <= round_index <= buffer_hi)


def _fault_columns_for_window(det_sets: list, row_index: dict, lead_rows: set,
                              committed_elsewhere: set) -> list:
    """Choose the fault columns this window can use."""
    columns: list = []
    for fault_index, detectors in enumerate(det_sets):
        touches_window = any(detector_id in row_index for detector_id in detectors)
        if not touches_window:
            continue

        if fault_index not in committed_elsewhere:
            columns.append(fault_index)
            continue

        touches_leading_buffer = any(detector_id in lead_rows
                                     for detector_id in detectors)
        if touches_leading_buffer:
            columns.append(fault_index)
    return columns


def _catalog_from_dem(
    dem,
    representation: FaultRepresentation,
) -> _FaultCatalog:
    """Build one independently sourced global fault catalog."""
    detector_sets, observable_sets, priors = detector_error_model_to_faults(dem)
    return _FaultCatalog(
        representation=representation,
        detector_sets=tuple(tuple(values) for values in detector_sets),
        observable_sets=tuple(tuple(values) for values in observable_sets),
        priors=tuple(float(value) for value in priors),
    )


def _physical_to_graphlike_catalog_link(
    decomposed_dem,
    graphlike_catalog: _FaultCatalog,
    physical_catalog: _FaultCatalog,
):
    """Relate independent catalogs by complete canonical Stim identities."""
    import numpy as np

    graphlike_index = {
        (detectors, observables): index
        for index, (detectors, observables) in enumerate(zip(
            graphlike_catalog.detector_sets,
            graphlike_catalog.observable_sets,
        ))
    }
    physical_index = {
        (detectors, observables): index
        for index, (detectors, observables) in enumerate(zip(
            physical_catalog.detector_sets,
            physical_catalog.observable_sets,
        ))
    }
    link = np.zeros(
        (
            len(graphlike_catalog.detector_sets),
            len(physical_catalog.detector_sets),
        ),
        dtype=np.uint8,
    )
    decomposed_identities: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for record in canonical_error_instructions(decomposed_dem):
        physical_key = (
            record.aggregate_detectors,
            record.aggregate_logical_observables,
        )
        if physical_key in decomposed_identities:
            continue
        decomposed_identities.add(physical_key)
        try:
            target_column = physical_index[physical_key]
        except KeyError as error:
            raise ValueError(
                "decomposed Stim model contains a physical identity absent "
                "from the undecomposed model"
            ) from error
        for component in record.components:
            component_key = (
                component.detectors,
                component.logical_observables,
            )
            try:
                component_column = graphlike_index[component_key]
            except KeyError as error:
                raise ValueError(
                    "decomposed Stim component is absent from the graphlike "
                    "fault catalog"
                ) from error
            link[component_column, target_column] ^= 1
    if len(decomposed_identities) != len(physical_index):
        raise ValueError(
            "not every physical fault has a canonical graphlike decomposition"
        )
    for physical_key, target_column in physical_index.items():
        component_columns = np.nonzero(link[:, target_column])[0]
        derived_detectors = _xor_target_ids(
            detector_id
            for component_column in component_columns
            for detector_id in graphlike_catalog.detector_sets[component_column]
        )
        derived_observables = _xor_target_ids(
            observable_id
            for component_column in component_columns
            for observable_id in graphlike_catalog.observable_sets[component_column]
        )
        if (derived_detectors, derived_observables) != physical_key:
            raise ValueError(
                f"physical fault {target_column} does not equal the complete "
                "detector-and-observable XOR of its graphlike components"
            )
    return link


def _prepare_fault_catalogs(
    circuit,
    requirement: DecoderFaultModelRequirement,
) -> tuple[dict[FaultRepresentation, _FaultCatalog], "object"]:
    """Build only the fault domains requested for this operation's code."""
    if not isinstance(requirement, DecoderFaultModelRequirement):
        raise TypeError(
            "fault_model_requirement must be a DecoderFaultModelRequirement"
        )
    catalogs: dict[FaultRepresentation, _FaultCatalog] = {}
    decomposed_dem = None
    if FaultRepresentation.GRAPHLIKE in requirement.representations:
        decomposed_dem = circuit.detector_error_model(decompose_errors=True)
        catalogs[FaultRepresentation.GRAPHLIKE] = _catalog_from_dem(
            decomposed_dem,
            FaultRepresentation.GRAPHLIKE,
        )
    if FaultRepresentation.PHYSICAL in requirement.representations:
        physical_dem = circuit.detector_error_model(decompose_errors=False)
        catalogs[FaultRepresentation.PHYSICAL] = _catalog_from_dem(
            physical_dem,
            FaultRepresentation.PHYSICAL,
        )
    link = None
    if requirement.require_physical_to_graphlike_link:
        assert decomposed_dem is not None
        link = _physical_to_graphlike_catalog_link(
            decomposed_dem,
            catalogs[FaultRepresentation.GRAPHLIKE],
            catalogs[FaultRepresentation.PHYSICAL],
        )
    return catalogs, link


def _fault_owned_by_window(fault_index: int, fault_rounds: list,
                           committed_elsewhere: set, unowned_faults: set,
                           commit_lo: int,
                           commit_hi: int, *, is_last: bool) -> bool:
    """True when this window is responsible for committing the fault."""
    if (fault_index in committed_elsewhere
            or fault_index in unowned_faults):
        return False
    if is_last:
        return True
    return any(commit_lo <= round_index <= commit_hi
               for round_index in fault_rounds[fault_index])


def _fill_detector_and_observable_columns(check, obs, *, column_index: int,
                                          fault_index: int, det_sets: list,
                                          obs_sets: list, row_index: dict) -> None:
    """Fill the detector and observable entries for one fault column."""
    for detector_id in det_sets[fault_index]:
        if detector_id in row_index:
            check[row_index[detector_id], column_index] = 1

    for observable_id in obs_sets[fault_index]:
        obs[observable_id, column_index] = 1


def _future_flips_after_commit(det_sets: list, round_of: dict,
                               fault_index: int, commit_hi: int,
                               *, is_last: bool) -> tuple:
    """Return detector flips that must be handed to a later window."""
    if is_last:
        return ()
    return tuple(detector_id
                 for detector_id in det_sets[fault_index]
                 if round_of[detector_id] > commit_hi)


def _build_window_arrays(*, rows: list, columns: list, row_index: dict,
                         det_sets: list, obs_sets: list, n_obs: int,
                         round_of: dict, fault_rounds: list,
                         committed_elsewhere: set, unowned_faults: set,
                         commit_lo: int,
                         commit_hi: int, is_last: bool) -> tuple:
    """Build check, observable, ownership, and future-defect arrays."""
    import numpy as np

    check = np.zeros((len(rows), len(columns)), dtype=np.uint8)
    obs = np.zeros((n_obs, len(columns)), dtype=np.uint8)
    owned = np.zeros(len(columns), dtype=bool)
    future_flips: dict = {}

    for column_index, fault_index in enumerate(columns):
        _fill_detector_and_observable_columns(
            check, obs, column_index=column_index, fault_index=fault_index,
            det_sets=det_sets, obs_sets=obs_sets, row_index=row_index)

        owns_fault = _fault_owned_by_window(
            fault_index, fault_rounds, committed_elsewhere, unowned_faults,
            commit_lo, commit_hi, is_last=is_last)
        if not owns_fault:
            continue

        owned[column_index] = True
        committed_elsewhere.add(fault_index)
        beyond_commit = _future_flips_after_commit(
            det_sets, round_of, fault_index, commit_hi, is_last=is_last)
        if beyond_commit:
            future_flips[column_index] = beyond_commit

    return check, obs, owned, future_flips


def _defect_positions(future_flips: dict, round_of: dict, pos_of: dict) -> dict:
    """Return detector positions for artificial defects handed forward."""
    return {
        detector_id: (round_of[detector_id], pos_of[detector_id])
        for flips in future_flips.values()
        for detector_id in flips
    }


def _window_geometry(round_of: dict, det_sets: list, committed_elsewhere: set,
                     buffer_lo: int, commit_lo: int,
                     buffer_hi: int, *, is_last: bool) -> tuple:
    """Return rows, row index, and columns for one sliced window."""
    rows = _detectors_in_window(round_of, buffer_lo, buffer_hi, is_last=is_last)
    row_index = {
        detector_id: row_number
        for row_number, detector_id in enumerate(rows)
    }
    lead_rows = {
        detector_id
        for detector_id in rows
        if round_of[detector_id] < commit_lo
    }
    columns = _fault_columns_for_window(
        det_sets, row_index, lead_rows, committed_elsewhere)
    return rows, row_index, columns


def _validate_fault_exclusion_ranges(fault_exclusion_ranges: tuple) -> None:
    """Validate explicit inclusive round ranges without changing ownership."""
    for exclusion in fault_exclusion_ranges:
        try:
            exclude_lo, exclude_hi = exclusion
        except (TypeError, ValueError) as error:
            raise TypeError(
                "each fault-exclusion range must be an integer "
                f"(lo, hi) pair, got {exclusion!r}") from error
        if not all(isinstance(endpoint, int)
                   for endpoint in (exclude_lo, exclude_hi)):
            raise TypeError(
                "each fault-exclusion range must be an integer "
                f"(lo, hi) pair, got {exclusion!r}")
        if exclude_lo > exclude_hi:
            raise ValueError(
                f"fault-exclusion range {exclude_lo}-{exclude_hi} "
                f"is inverted")


def _unowned_faults(
    fault_rounds: tuple[tuple[int, ...], ...],
    fault_exclusion_ranges: tuple,
) -> set[int]:
    """Return source columns prevented from being committed by this slice."""
    return {
        fault_index
        for fault_index, rounds in enumerate(fault_rounds)
        if any(
            exclude_lo <= round_index <= exclude_hi
            for exclude_lo, exclude_hi in fault_exclusion_ranges
            for round_index in rounds
        )
    }


def _placed_faults_for_window(
    *,
    catalog: _FaultCatalog,
    rows: list[int],
    row_index: dict[int, int],
    lead_rows: set[int],
    n_obs: int,
    round_of: dict[int, int],
    committed_elsewhere: set[int],
    fault_exclusion_ranges: tuple,
    commit_lo: int,
    commit_hi: int,
    is_last: bool,
) -> PlacedFaultModel:
    """Place one requested catalog using the shared window geometry owner."""
    import numpy as np

    detector_sets = list(catalog.detector_sets)
    observable_sets = list(catalog.observable_sets)
    fault_rounds = tuple(
        tuple(round_of[detector_id] for detector_id in detectors)
        for detectors in catalog.detector_sets
    )
    columns = _fault_columns_for_window(
        detector_sets,
        row_index,
        lead_rows,
        committed_elsewhere,
    )
    check, observables, owned, future_flips = _build_window_arrays(
        rows=rows,
        columns=columns,
        row_index=row_index,
        det_sets=detector_sets,
        obs_sets=observable_sets,
        n_obs=n_obs,
        round_of=round_of,
        fault_rounds=list(fault_rounds),
        committed_elsewhere=committed_elsewhere,
        unowned_faults=_unowned_faults(
            fault_rounds,
            fault_exclusion_ranges,
        ),
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        is_last=is_last,
    )
    placed = PlacedFaultModel(
        representation=catalog.representation,
        check=check,
        priors=np.asarray(
            [catalog.priors[fault_index] for fault_index in columns],
            dtype=float,
        ),
        observables=observables,
        owned=owned,
        future_flips=future_flips,
        source_fault_ids=tuple(columns),
    )
    if catalog.representation is FaultRepresentation.GRAPHLIKE:
        validate_graphlike_matrices(
            placed.check,
            placed.observables,
            location="placed graphlike fault model",
        )
    else:
        validate_placed_fault_matrices(
            placed.check,
            placed.observables,
            location="placed physical fault model",
        )
    return placed


def _local_physical_to_graphlike_detector_projection(
    graphlike: PlacedFaultModel,
    physical: PlacedFaultModel,
    catalog_link,
):
    """Slice and exactly validate the link between the two local views."""
    import numpy as np

    local_link = np.asarray(catalog_link, dtype=np.uint8)[np.ix_(
        graphlike.source_fault_ids,
        physical.source_fault_ids,
    )]
    detector_identity = (
        np.asarray(graphlike.check, dtype=np.uint64)
        @ local_link.astype(np.uint64)
    ) % 2
    if not np.array_equal(detector_identity, physical.check):
        raise ValueError(
            "local physical detector identities do not equal their "
            "graphlike component XOR"
        )
    # A physical fault can decompose into a component whose detectors lie
    # wholly beyond this window while that component carries a logical tag.
    # The catalog-level check above preserves the complete observable identity;
    # the local link is deliberately only the detector-row projection consumed
    # by belief propagation.  Each decoder commits observables from its own
    # explicit placed view, never through this projected link.
    return local_link


def _coordinates_for_rows(circuit, rows: list[int]):
    """Return real circuit coordinates only when every local row has them."""
    coordinates = circuit.get_detector_coordinates()
    if any(not coordinates.get(detector_id) for detector_id in rows):
        return None
    return tuple(
        tuple(float(value) for value in coordinates[detector_id])
        for detector_id in rows
    )


class WindowSlicer:
    """Incremental single owner of window geometry and per-domain placement."""

    def __init__(
        self,
        circuit,
        num_observables: Optional[int] = None,
        *,
        detector_rounds: Optional[dict] = None,
        fault_model_requirement: DecoderFaultModelRequirement,
    ):
        self.circuit = circuit
        self.catalogs, self.catalog_link = _prepare_fault_catalogs(
            circuit,
            fault_model_requirement,
        )
        self.n_obs = (
            num_observables
            if num_observables is not None
            else circuit.num_observables
        )
        self.round_of = _detector_rounds_from_circuit(circuit, detector_rounds)
        self.pos_of = _detector_position_in_round(self.round_of)
        self.committed_elsewhere = {
            representation: set()
            for representation in self.catalogs
        }

    def slice_window(
        self,
        buffer_lo: int,
        commit_lo: int,
        commit_hi: int,
        buffer_hi: int,
        *,
        is_last: bool,
        fault_exclusion_ranges: tuple = (),
    ) -> WindowErrorModel:
        """Create one model and advance ownership in every requested domain."""
        _validate_fault_exclusion_ranges(fault_exclusion_ranges)
        rows = _detectors_in_window(
            self.round_of,
            buffer_lo,
            buffer_hi,
            is_last=is_last,
        )
        row_index = {
            detector_id: row_number
            for row_number, detector_id in enumerate(rows)
        }
        lead_rows = {
            detector_id
            for detector_id in rows
            if self.round_of[detector_id] < commit_lo
        }
        placed = {
            representation: _placed_faults_for_window(
                catalog=catalog,
                rows=rows,
                row_index=row_index,
                lead_rows=lead_rows,
                n_obs=self.n_obs,
                round_of=self.round_of,
                committed_elsewhere=self.committed_elsewhere[representation],
                fault_exclusion_ranges=fault_exclusion_ranges,
                commit_lo=commit_lo,
                commit_hi=commit_hi,
                is_last=is_last,
            )
            for representation, catalog in self.catalogs.items()
        }
        graphlike = placed.get(FaultRepresentation.GRAPHLIKE)
        physical = placed.get(FaultRepresentation.PHYSICAL)
        local_link = None
        if self.catalog_link is not None:
            assert graphlike is not None and physical is not None
            local_link = _local_physical_to_graphlike_detector_projection(
                graphlike,
                physical,
                self.catalog_link,
            )
        future_flips = {
            detector_id
            for fault_view in placed.values()
            for flips in fault_view.future_flips.values()
            for detector_id in flips
        }
        return WindowErrorModel(
            detector_ids=tuple(rows),
            detector_coordinates=_coordinates_for_rows(self.circuit, rows),
            commit_hi=commit_hi,
            defect_positions={
                detector_id: (
                    self.round_of[detector_id],
                    self.pos_of[detector_id],
                )
                for detector_id in future_flips
            },
            graphlike_faults=graphlike,
            physical_faults=physical,
            physical_to_graphlike_detector_projection=local_link,
            commit_lo=commit_lo,
            buffer_lo=buffer_lo,
        )


def build_window_error_models(
    circuit,
    plan: list,
    num_observables: Optional[int] = None,
    *,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: DecoderFaultModelRequirement,
) -> list:
    """Slice an operation circuit into one typed model per planned window."""
    slicer = WindowSlicer(
        circuit,
        num_observables,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    last_window = len(plan) - 1
    return [
        slicer.slice_window(
            *_parse_window_entry(window_entry),
            is_last=(window_index == last_window),
        )
        for window_index, window_entry in enumerate(plan)
    ]


def _build_single_window_error_model(
    circuit,
    window_entry: tuple,
    num_observables: Optional[int],
    *,
    detector_rounds: Optional[dict],
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
) -> WindowErrorModel:
    """Build an independent typed model with explicit non-owned ranges."""
    slicer = WindowSlicer(
        circuit,
        num_observables,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    return slicer.slice_window(
        *_parse_window_entry(window_entry),
        is_last=False,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def build_single_window_error_model(circuit, window_entry: tuple,
                                    num_observables: Optional[int] = None,
                                    *, detector_rounds: Optional[dict] = None,
                                    fault_model_requirement:
                                    DecoderFaultModelRequirement,
                                    exclude_faults_touching: Optional[tuple] = None
                                    ) -> WindowErrorModel:
    """Build one independent window model.

    ``exclude_faults_touching=(lo, hi)`` keeps faults touching that inclusive
    range available to explain the syndrome but prevents this window from
    committing them.
    """
    fault_exclusion_ranges = (
        () if exclude_faults_touching is None
        else (exclude_faults_touching,)
    )
    return _build_single_window_error_model(
        circuit, window_entry, num_observables,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def build_single_window_error_model_with_exclusions(
    circuit, window_entry: tuple, num_observables: Optional[int] = None, *,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
) -> WindowErrorModel:
    """Build one independent model with multiple non-owned inclusive ranges."""
    return _build_single_window_error_model(
        circuit, window_entry, num_observables,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def decode_windowed(
    window_models: list,
    detection_events,
    decode_window,
    *,
    selected_fault_representation: FaultRepresentation,
) -> "object":
    """Decode one shot by walking the committed windows in order.

    Two-sided/parallel A/B windows (any window with a leading buffer) decode
    independently and must NOT forward artificial defects -- that would double-count
    the boundary error; forward-only sliding windows push them forward instead.
    """
    # ref: Skoric 2209.08552, Tan 2209.09219
    logical_prediction, _ = _walk_windowed(
        window_models,
        detection_events,
        decode_window,
        selected_fault_representation,
        typed_backend_outcomes=False,
    )
    return logical_prediction


@dataclass(frozen=True)
class WindowedBackendDecode:
    """Same-shot backend outcomes and a prediction only after full success."""

    window_outcomes: tuple
    logical_prediction: Optional[tuple[int, ...]]


def decode_windowed_backend_outcomes(
    window_models: list,
    detection_events,
    decode_window,
) -> WindowedBackendDecode:
    """Walk physical windows once and preserve each exact backend outcome."""
    logical_prediction, outcomes = _walk_windowed(
        window_models,
        detection_events,
        decode_window,
        FaultRepresentation.PHYSICAL,
        typed_backend_outcomes=True,
    )
    return WindowedBackendDecode(
        window_outcomes=outcomes,
        logical_prediction=(
            None
            if logical_prediction is None
            else tuple(int(bit) for bit in logical_prediction)
        ),
    )


def _walk_windowed(
    window_models: list,
    detection_events,
    decode_window,
    selected_fault_representation: FaultRepresentation,
    *,
    typed_backend_outcomes: bool,
) -> tuple["object", tuple]:
    """Single owner of detector selection, commitment, and boundary forwarding."""
    import numpy as np

    if not window_models:
        raise ValueError("windowed decode requires at least one window model")
    two_sided = any(model.has_leading_buffer for model in window_models)
    pending: set = set()
    first_faults = window_models[0].require_faults(
        selected_fault_representation
    )
    total = np.zeros(first_faults.observables.shape[0], dtype=np.uint8)
    outcomes = []
    for model in window_models:
        faults = model.require_faults(selected_fault_representation)
        syndrome = detection_events[list(model.detector_ids)].astype(np.uint8).copy()
        if not two_sided:
            for detector_index, detector_id in enumerate(model.detector_ids):
                if detector_id in pending:
                    syndrome[detector_index] ^= 1
                    pending.discard(detector_id)

        decoded = decode_window(model, syndrome)
        if typed_backend_outcomes:
            from .adapters.window_decode_results import (
                validate_backend_outcome,
            )

            validate_backend_outcome(decoded, model, faults, syndrome)
            outcomes.append(decoded)
            if not decoded.succeeded:
                return None, tuple(outcomes)
            selected = np.asarray(
                decoded.physical_correction,
                dtype=np.uint8,
            )
        else:
            selected = np.asarray(decoded, dtype=np.uint8)
        if selected.shape != (faults.check.shape[1],):
            raise ValueError(
                "selected correction arity does not match the placed fault model"
            )
        committed = selected.astype(bool) & faults.owned
        total ^= (faults.observables @ committed.astype(np.uint8)) % 2
        if not two_sided:
            for column_index in np.nonzero(committed)[0]:
                for detector_id in faults.future_flips.get(int(column_index), ()):
                    pending.symmetric_difference_update({detector_id})
    if not two_sided and pending:
        raise RuntimeError(f"artificial defects were never consumed: {sorted(pending)}"
                           ". The plan does not cover the full detector stream.")
    return total, tuple(outcomes)
