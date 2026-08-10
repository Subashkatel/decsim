"""Slice a global Stim detector error model into per-window decoder inputs.

The measured bits change every shot; the sliced error models are compile-time
data shared across shots. Consumed by the window decoders (MWPM / BP+OSD /
belief matching) and built per op by adapters/stim_device.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Optional

from .message import WindowProtocol


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
    """One decoder matrix whose columns all describe the same fault domain.

    ``owned`` says which selected columns this window may commit.
    ``future_flips`` is the forward-only handoff used by Sliding decoding.
    ``boundary_flips`` stores each owned column's complete global detector
    effect for dependency-aware delivery in either time direction.
    ``source_fault_ids`` maps every local column back to the global catalog.
    """

    representation: FaultRepresentation
    check: "object"
    priors: "object"
    observables: "object"
    owned: "object"
    future_flips: dict
    source_fault_ids: tuple[int, ...]
    boundary_flips: dict

    def __post_init__(self) -> None:
        import numpy as np

        for field_name in ("check", "priors", "observables", "owned"):
            source = np.ascontiguousarray(getattr(self, field_name))
            frozen = np.frombuffer(
                source.tobytes(order="C"),
                dtype=source.dtype,
            ).reshape(source.shape)
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "source_fault_ids", tuple(self.source_fault_ids))
        for field_name in ("future_flips", "boundary_flips"):
            frozen_mapping = MappingProxyType({
                int(column): tuple(int(detector_id) for detector_id in detector_ids)
                for column, detector_ids in getattr(self, field_name).items()
            })
            object.__setattr__(self, field_name, frozen_mapping)


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
        """True when ``buffer_lo < commit_lo`` (a two-sided buffered window)."""
        # ref: Skoric 2209.08552 sec. I.C; Tan 2209.09219 supp. sec. S2.D
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
        assert key is not None
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



def resolve_detector_rounds(circuit, detector_rounds: Optional[dict],
                            round_count: int) -> dict[int, int]:
    """Resolve one finite source into decsim's one-based emitted rounds.

    Without a map, accept Stim repetition or decsim surface/toric coordinates.
    """
    if type(round_count) is not int:
        raise TypeError("round_count must be a built-in int")
    if round_count < 1:
        raise ValueError("round_count must be positive")
    detector_count = circuit.num_detectors
    if detector_count < 1:
        raise ValueError("a finite decoding source must contain detectors")

    if detector_rounds is None:
        coordinates = circuit.get_detector_coordinates()
        arities = {len(coordinates.get(detector_id, ()))
                   for detector_id in range(detector_count)}
        if len(arities) != 1:
            raise ValueError("finite-memory detector coordinates need one arity")
        coordinate_arity = next(iter(arities))
        if coordinate_arity != 2 and coordinate_arity < 3:
            raise ValueError(
                "finite-memory chronology requires supported coordinates or "
                "explicit detector_rounds"
            )
        raw_layers = {}
        for detector_id in range(detector_count):
            raw_value = coordinates[detector_id][-1]
            if not math.isfinite(raw_value) or raw_value != int(raw_value):
                raise ValueError("finite-memory detector layers must be finite integers")
            raw_layer = int(raw_value)
            if raw_layer < 0:
                raise ValueError("finite-memory detector layers must be nonnegative")
            raw_layers[detector_id] = raw_layer
        expected_layers = set(range(round_count + 1))
        if set(raw_layers.values()) != expected_layers:
            raise ValueError(
                "raw detector layers must equal the declared source duration"
            )
        resolved = {
            detector_id: (
                round_count if raw_layer == round_count else raw_layer + 1
            )
            for detector_id, raw_layer in raw_layers.items()
        }
    else:
        resolved = dict(detector_rounds)

    for detector_id, emitted_round in resolved.items():
        if type(detector_id) is not int or type(emitted_round) is not int:
            raise TypeError("detector ids and emitted rounds must be built-in ints")
    if set(resolved) != set(range(detector_count)):
        raise ValueError("detector-round map must cover every detector exactly")
    if any(not 1 <= value <= round_count for value in resolved.values()):
        raise ValueError("detector-round map contains an out-of-range round")
    if set(resolved.values()) != set(range(1, round_count + 1)):
        raise ValueError("detector-round map must fill every emitted round")
    return resolved


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
    """Normalize and validate a 3-value or 4-value window plan entry."""
    if type(window_entry) is not tuple or len(window_entry) not in (3, 4):
        raise TypeError("window entry must be an exact 3- or 4-int tuple")
    if any(type(bound) is not int for bound in window_entry):
        raise TypeError("window bounds must be built-in ints")
    if any(bound < 1 for bound in window_entry):
        raise ValueError("window bounds must be positive")
    if len(window_entry) == 4:
        buffer_lo, commit_lo, commit_hi, buffer_hi = window_entry
    else:
        commit_lo, commit_hi, buffer_hi = window_entry
        buffer_lo = commit_lo
    if not buffer_lo <= commit_lo <= commit_hi <= buffer_hi:
        raise ValueError("window geometry bounds are not ordered")
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


def _fault_columns_for_window(
    det_sets: list,
    row_index: dict,
    lead_rows: set,
    committed_elsewhere: set,
    *,
    include_committed_leading: bool = True,
) -> list:
    """Choose candidate columns, excluding causally prior committed faults."""
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
        if include_committed_leading and touches_leading_buffer:
            columns.append(fault_index)
    return columns


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
            validate_fault_identity(
                *key,
                location=f"physical Stim error {record.error_ordinal}",
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
    """Keep distinct physical mechanisms when their graph decompositions differ."""
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
    link = np.zeros(
        (len(graphlike_catalog.detector_sets), len(mechanisms)),
        dtype=np.uint8,
    )
    for physical_column, ((physical_key, component_keys), prior) in enumerate(
        mechanisms.items()
    ):
        physical_detector_sets.append(physical_key[0])
        physical_observable_sets.append(physical_key[1])
        physical_priors.append(prior)
        for component_key in component_keys:
            try:
                graphlike_column = graphlike_index[component_key]
            except KeyError as error:
                raise ValueError(
                    "decomposed Stim component is absent from the graphlike catalog"
                ) from error
            link[graphlike_column, physical_column] = 1

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

    derived_check = np.zeros(
        (max((detector_id for detectors in physical_catalog.detector_sets
              for detector_id in detectors), default=-1) + 1,
         len(physical_catalog.detector_sets)),
        dtype=np.uint8,
    )
    graph_check = np.zeros(
        (derived_check.shape[0], len(graphlike_catalog.detector_sets)),
        dtype=np.uint8,
    )
    for column, detectors in enumerate(graphlike_catalog.detector_sets):
        graph_check[list(detectors), column] = 1
    for column, detectors in enumerate(physical_catalog.detector_sets):
        derived_check[list(detectors), column] = 1
    if not np.array_equal((graph_check @ link) % 2, derived_check):
        raise ValueError(
            "physical detector effects do not equal their graphlike components"
        )
    for physical_column, observable_ids in enumerate(
        physical_catalog.observable_sets
    ):
        derived_observables = _xor_target_ids(
            observable_id
            for graphlike_column in np.nonzero(link[:, physical_column])[0]
            for observable_id in graphlike_catalog.observable_sets[graphlike_column]
        )
        if derived_observables != observable_ids:
            raise ValueError(
                "physical logical effects do not equal their graphlike components"
            )
    return graphlike_catalog, physical_catalog, link


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


def _fault_owned_by_window(
    fault_index: int,
    fault_rounds: list,
    committed_elsewhere: set,
    unowned_faults: set,
    explicitly_owned_faults: Optional[set],
    commit_lo: int,
    commit_hi: int,
    *,
    is_last: bool,
) -> bool:
    """True when this window is responsible for committing the fault."""
    if fault_index in unowned_faults:
        return False
    if explicitly_owned_faults is not None:
        return fault_index in explicitly_owned_faults
    if fault_index in committed_elsewhere:
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
                         explicitly_owned_faults: Optional[set],
                         commit_lo: int,
                         commit_hi: int, is_last: bool) -> tuple:
    """Build check, observable, ownership, and residual-defect arrays."""
    import numpy as np

    check = np.zeros((len(rows), len(columns)), dtype=np.uint8)
    obs = np.zeros((n_obs, len(columns)), dtype=np.uint8)
    owned = np.zeros(len(columns), dtype=bool)
    future_flips: dict = {}
    boundary_flips: dict = {}

    for column_index, fault_index in enumerate(columns):
        _fill_detector_and_observable_columns(
            check, obs, column_index=column_index, fault_index=fault_index,
            det_sets=det_sets, obs_sets=obs_sets, row_index=row_index)

        owns_fault = _fault_owned_by_window(
            fault_index, fault_rounds, committed_elsewhere, unowned_faults,
            explicitly_owned_faults, commit_lo, commit_hi, is_last=is_last)
        if not owns_fault:
            continue

        owned[column_index] = True
        if explicitly_owned_faults is None:
            committed_elsewhere.add(fault_index)
        beyond_commit = _future_flips_after_commit(
            det_sets, round_of, fault_index, commit_hi, is_last=is_last)
        if beyond_commit:
            future_flips[column_index] = beyond_commit
        # Keep the complete global detector effect. The destination intersects
        # it with its own rows, so the same correction can travel left or right.
        detector_effect = tuple(det_sets[fault_index])
        if detector_effect:
            boundary_flips[column_index] = detector_effect

    return check, obs, owned, future_flips, boundary_flips


def _validate_fault_exclusion_ranges(fault_exclusion_ranges: tuple) -> None:
    """Validate explicit inclusive round ranges without changing ownership."""
    for exclusion in fault_exclusion_ranges:
        try:
            exclude_lo, exclude_hi = exclusion
        except (TypeError, ValueError) as error:
            raise TypeError(
                "each fault-exclusion range must be an integer "
                f"(lo, hi) pair, got {exclusion!r}") from error
        if not all(type(endpoint) is int
                   for endpoint in (exclude_lo, exclude_hi)):
            raise TypeError(
                "each fault-exclusion range must use built-in integer "
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
    explicitly_owned_faults: Optional[set[int]],
    explicitly_prior_faults: Optional[set[int]],
    fault_exclusion_ranges: tuple,
    commit_lo: int,
    commit_hi: int,
    is_last: bool,
) -> PlacedFaultModel:
    """Build one window's local matrix from one global fault catalog."""
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
        (
            committed_elsewhere
            if explicitly_prior_faults is None
            else explicitly_prior_faults
        ),
        include_committed_leading=(explicitly_prior_faults is None),
    )
    check, observables, owned, future_flips, boundary_flips = (
        _build_window_arrays(
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
            explicitly_owned_faults=explicitly_owned_faults,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            is_last=is_last,
        )
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
        boundary_flips=boundary_flips,
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
    """Build local matrices from one global model in every requested domain.

    Direct callers may advance ownership window by window. Static planners can
    instead supply a precomputed owner set so construction order cannot change
    which window commits a fault.
    """

    def __init__(
        self,
        circuit,
        num_observables: Optional[int] = None,
        *,
        round_count: int,
        detector_rounds: Optional[dict] = None,
        fault_model_requirement: DecoderFaultModelRequirement,
    ):
        self.circuit = circuit
        self.catalogs, self.catalog_link = _prepare_fault_catalogs(
            circuit,
            fault_model_requirement,
        )
        if num_observables is not None:
            if type(num_observables) is not int:
                raise TypeError("num_observables must be a built-in int")
            if num_observables != circuit.num_observables:
                raise ValueError(
                    "num_observables must equal the circuit observable count"
                )
        self.n_obs = circuit.num_observables
        self.round_of = resolve_detector_rounds(
            circuit, detector_rounds, round_count
        )
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
        explicitly_owned_faults: Optional[dict] = None,
        explicitly_prior_faults: Optional[dict] = None,
    ) -> WindowErrorModel:
        """Create one local model and update incremental ownership if used."""
        explicit_values = (
            explicitly_owned_faults,
            explicitly_prior_faults,
        )
        if (explicit_values[0] is None) != (explicit_values[1] is None):
            raise ValueError(
                "explicit owner and predecessor fault maps must be supplied together"
            )
        if explicit_values[0] is not None:
            expected_representations = set(self.catalogs)
            if (
                set(explicitly_owned_faults) != expected_representations
                or set(explicitly_prior_faults) != expected_representations
            ):
                raise ValueError(
                    "explicit fault maps must exactly match the requested representations"
                )
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
                explicitly_owned_faults=(
                    None
                    if explicitly_owned_faults is None
                    else explicitly_owned_faults[representation]
                ),
                explicitly_prior_faults=(
                    None
                    if explicitly_prior_faults is None
                    else explicitly_prior_faults[representation]
                ),
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
        residual_rows = set(rows) | {
            detector_id
            for fault_view in placed.values()
            for flips in fault_view.boundary_flips.values()
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
                for detector_id in residual_rows
            },
            graphlike_faults=graphlike,
            physical_faults=physical,
            physical_to_graphlike_detector_projection=local_link,
            commit_lo=commit_lo,
            buffer_lo=buffer_lo,
        )


def _dependency_depths(window_count: int, dependency_edges: tuple) -> tuple[int, ...]:
    """Validate the dependency DAG and return each window's depth."""
    predecessors = [set() for _ in range(window_count)]
    seen = set()
    for edge in dependency_edges:
        if (
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(index) is not int for index in edge)
        ):
            raise TypeError("window dependencies must be exact (int, int) pairs")
        source, destination = edge
        if (
            source < 0
            or destination < 0
            or source >= window_count
            or destination >= window_count
            or source == destination
        ):
            raise ValueError("window dependency is out of range or self-directed")
        if edge in seen:
            raise ValueError("window dependencies must be unique")
        seen.add(edge)
        predecessors[destination].add(source)

    depths: list[Optional[int]] = [None] * window_count
    while any(depth is None for depth in depths):
        progressed = False
        for window_index, incoming in enumerate(predecessors):
            if depths[window_index] is not None:
                continue
            if any(depths[source] is None for source in incoming):
                continue
            depths[window_index] = (
                0
                if not incoming
                else 1 + max(depths[source] for source in incoming)
            )
            progressed = True
        if not progressed:
            raise ValueError("window dependencies must form an acyclic graph")
    return tuple(depth for depth in depths if depth is not None)


def _dependency_ancestors(
    window_count: int,
    dependency_edges: tuple,
    depths: tuple[int, ...],
) -> tuple[frozenset[int], ...]:
    """Return every direct and indirect predecessor of each window."""
    incoming = [set() for _ in range(window_count)]
    for source, destination in dependency_edges:
        incoming[destination].add(source)
    ancestors = [set() for _ in range(window_count)]
    for destination in sorted(range(window_count), key=depths.__getitem__):
        for source in incoming[destination]:
            ancestors[destination].add(source)
            ancestors[destination].update(ancestors[source])
    return tuple(frozenset(nodes) for nodes in ancestors)


def _explicit_prior_faults(
    ownership: tuple[dict[FaultRepresentation, set[int]], ...],
    ancestors: tuple[frozenset[int], ...],
) -> tuple[dict[FaultRepresentation, set[int]], ...]:
    """Collect the faults owned by each window's predecessors."""
    representations = tuple(ownership[0])
    return tuple(
        {
            representation: set().union(*(
                ownership[ancestor][representation]
                for ancestor in ancestor_indices
            ))
            for representation in representations
        }
        for ancestor_indices in ancestors
    )


def _explicit_fault_ownership(
    slicer: WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    depths: tuple[int, ...],
    *,
    round_count: int,
) -> tuple[dict[FaultRepresentation, set[int]], ...]:
    """Assign each fault to one shallowest commit window in the DAG.

    Equal-depth ambiguity is rejected. A plan that covers the full operation
    must assign every fault in every requested representation.
    """
    ownership = [
        {representation: set() for representation in slicer.catalogs}
        for _ in entries
    ]
    covers_full_operation = (
        entries[0][1] == 1 and entries[-1][2] == round_count
    )
    for representation, catalog in slicer.catalogs.items():
        for fault_index, detector_ids in enumerate(catalog.detector_sets):
            candidates = [
                window_index
                for window_index, (_, commit_lo, commit_hi, _) in enumerate(entries)
                if any(
                    commit_lo <= slicer.round_of[detector_id] <= commit_hi
                    for detector_id in detector_ids
                )
            ]
            if not candidates:
                if covers_full_operation:
                    raise ValueError(
                        f"{representation.value} fault {fault_index} touches no "
                        "window commit region"
                    )
                continue
            earliest_depth = min(depths[index] for index in candidates)
            earliest = [
                index for index in candidates
                if depths[index] == earliest_depth
            ]
            if len(earliest) != 1:
                raise ValueError(
                    f"{representation.value} fault {fault_index} straddles "
                    "independent commit regions without a causal owner"
                )
            ownership[earliest[0]][representation].add(fault_index)
    return tuple(ownership)


def _validate_closed_temporal_boundary_windows(
    slicer: WindowSlicer,
    models: list[WindowErrorModel],
    dependency_edges: Optional[tuple],
    closed_windows: tuple[int, ...],
) -> None:
    """Reject a declared closed time boundary if it cuts a global fault."""
    if type(closed_windows) is not tuple:
        raise TypeError(
            "closed_temporal_boundary_windows must be an exact tuple"
        )
    if (
        any(type(window_index) is not int for window_index in closed_windows)
        or len(set(closed_windows)) != len(closed_windows)
    ):
        raise TypeError(
            "closed temporal boundary window indices must be unique exact ints"
        )
    if not closed_windows:
        return
    if dependency_edges is None:
        raise ValueError(
            "closed temporal boundaries require explicit dependency edges"
        )
    destinations = {destination for _, destination in dependency_edges}
    for window_index in closed_windows:
        if window_index < 0 or window_index >= len(models):
            raise ValueError("closed temporal boundary window is out of range")
        if window_index not in destinations:
            raise ValueError(
                "closed temporal boundary window must be a dependency destination"
            )
        model = models[window_index]
        local_detector_ids = set(model.detector_ids)
        for representation, catalog in slicer.catalogs.items():
            faults = model.require_faults(representation)
            for source_fault_id in faults.source_fault_ids:
                global_detector_ids = set(
                    catalog.detector_sets[source_fault_id]
                )
                local_effect = global_detector_ids & local_detector_ids
                if local_effect and local_effect != global_detector_ids:
                    raise ValueError(
                        f"{representation.value} closed temporal boundary "
                        f"window {window_index} truncates global fault "
                        f"{source_fault_id}; a smooth B boundary cannot "
                        "contain an artificial boundary generator"
                    )


def _validate_window_protocol(
    entries: tuple[tuple[int, int, int, int], ...],
    window_protocol: WindowProtocol,
    dependency_edges: Optional[tuple],
    closed_windows: tuple[int, ...],
    fault_model_requirement: DecoderFaultModelRequirement,
) -> None:
    """Fail closed unless a Tan plan has the exact zero-seam contract."""
    if type(window_protocol) is not WindowProtocol:
        raise TypeError("window_protocol must be an exact WindowProtocol")
    if window_protocol is WindowProtocol.GENERIC:
        return
    if window_protocol is not WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE:
        raise ValueError("unsupported window protocol")

    seam_indices = tuple(range(1, len(entries), 2))
    if seam_indices and fault_model_requirement.representations != frozenset({
        FaultRepresentation.GRAPHLIKE
    }):
        raise ValueError(
            "Tan's validated zero-seam memory construction requires exactly the "
            "graphlike correction-edge representation"
        )
    if tuple(sorted(closed_windows)) != seam_indices:
        raise ValueError(
            "every Tan type-2 seam, and only a seam, must be temporally closed"
        )
    for seam_index in seam_indices:
        buffer_lo, commit_lo, commit_hi, buffer_hi = entries[seam_index]
        if not buffer_lo == commit_lo == commit_hi == buffer_hi:
            raise ValueError("a zero-offset Tan type-2 seam must be one detector layer")
    expected_edges = tuple(
        edge
        for seam_index in seam_indices
        for edge in ((seam_index - 1, seam_index),
                     (seam_index + 1, seam_index))
    )
    if (
        dependency_edges is None and expected_edges
        or dependency_edges is not None
        and (
            set(dependency_edges) != set(expected_edges)
            or len(dependency_edges) != len(expected_edges)
        )
    ):
        raise ValueError(
            "each Tan type-2 seam must depend on its two adjacent type-1 tasks"
        )


def build_window_error_models(
    circuit,
    plan: list,
    num_observables: Optional[int] = None,
    *,
    round_count: int,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
    dependency_edges: Optional[tuple] = None,
    closed_temporal_boundary_windows: tuple[int, ...] = (),
    window_protocol: WindowProtocol = WindowProtocol.GENERIC,
) -> list:
    """Slice one operation's global model into local decoder matrices.

    With ``dependency_edges``, ownership is compiled from the dependency DAG
    and predecessor-owned faults are removed from successor candidates. Without
    those edges, the slicer advances ownership in list order; that path is used
    by forward-only and dynamic Sliding construction. Indices listed in
    ``closed_temporal_boundary_windows`` are checked after slicing and rejected
    if any local column cuts a global fault into an artificial boundary edge.
    A contiguous partial plan is allowed for runtime suffix re-slicing. Its last
    window is terminal only when its commit region reaches ``round_count``;
    faults wholly outside the segment remain unowned.
    """
    entries = tuple(_parse_window_entry(window_entry) for window_entry in plan)
    if not entries:
        raise ValueError("window plan must contain at least one entry")
    _validate_window_protocol(
        entries,
        window_protocol,
        dependency_edges,
        closed_temporal_boundary_windows,
        fault_model_requirement,
    )
    next_commit_round = entries[0][1]
    for _, commit_lo, commit_hi, _ in entries:
        if commit_lo != next_commit_round:
            raise ValueError(
                "window commit regions must be contiguous in plan order "
                "without gaps or overlaps"
            )
        if commit_hi > round_count:
            raise ValueError("window commit region exceeds round_count")
        next_commit_round = commit_hi + 1

    slicer = WindowSlicer(
        circuit,
        num_observables,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    ownership = None
    prior_faults = None
    if dependency_edges is not None:
        depths = _dependency_depths(len(entries), dependency_edges)
        ancestors = _dependency_ancestors(
            len(entries), dependency_edges, depths)
        ownership = _explicit_fault_ownership(
            slicer,
            entries,
            depths,
            round_count=round_count,
        )
        prior_faults = _explicit_prior_faults(ownership, ancestors)
    last_window = len(entries) - 1
    models = [
        slicer.slice_window(
            *window_entry,
            is_last=(
                window_index == last_window
                and entries[-1][2] == round_count
            ),
            fault_exclusion_ranges=fault_exclusion_ranges,
            explicitly_owned_faults=(
                None if ownership is None else ownership[window_index]
            ),
            explicitly_prior_faults=(
                None if prior_faults is None else prior_faults[window_index]
            ),
        )
        for window_index, window_entry in enumerate(entries)
    ]
    _validate_closed_temporal_boundary_windows(
        slicer,
        models,
        dependency_edges,
        closed_temporal_boundary_windows,
    )
    if (
        ownership is not None
        and entries[0][1] == 1
        and entries[-1][2] == round_count
        and not fault_exclusion_ranges
    ):
        for representation, catalog in slicer.catalogs.items():
            owned = {
                source_fault_id
                for model in models
                for faults in [model.require_faults(representation)]
                for source_fault_id, is_owned in zip(
                    faults.source_fault_ids,
                    faults.owned,
                )
                if is_owned
            }
            if owned != set(range(len(catalog.detector_sets))):
                raise RuntimeError(
                    f"{representation.value} dependency ownership does not "
                    "partition the full fault catalog"
                )
    return models


def _build_single_window_error_model(
    circuit,
    window_entry: tuple,
    num_observables: Optional[int],
    *,
    round_count: int,
    detector_rounds: Optional[dict],
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
) -> WindowErrorModel:
    """Build an independent typed model with explicit non-owned ranges."""
    slicer = WindowSlicer(
        circuit,
        num_observables,
        round_count=round_count,
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
                                    *, round_count: int,
                                    detector_rounds: Optional[dict] = None,
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
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def build_single_window_error_model_with_exclusions(
    circuit, window_entry: tuple, num_observables: Optional[int] = None, *,
    round_count: int,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
) -> WindowErrorModel:
    """Build one independent model with multiple non-owned inclusive ranges."""
    return _build_single_window_error_model(
        circuit, window_entry, num_observables,
        round_count=round_count,
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
    """Decode one shot through a forward-only sliding-window chain.

    Parallel block A/B decoding needs a dependency-aware seam stage and
    residual-syndrome handoff. This list-ordered helper intentionally rejects
    leading-buffer models instead of approximating that different algorithm.
    """
    # ref: Skoric 2209.08552 sec. I.B; Tan 2209.09219 supp. sec. S2.C
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
    if any(model.has_leading_buffer for model in window_models):
        raise ValueError(
            "list-ordered decode_windowed supports only forward sliding windows; "
            "parallel A/B windows require dependency-aware seam reconciliation"
        )
    pending: set = set()
    last_window_for_detector = {
        detector_id: window_index
        for window_index, model in enumerate(window_models)
        for detector_id in model.detector_ids
    }
    first_faults = window_models[0].require_faults(
        selected_fault_representation
    )
    total = np.zeros(first_faults.observables.shape[0], dtype=np.uint8)
    outcomes = []
    for window_index, model in enumerate(window_models):
        faults = model.require_faults(selected_fault_representation)
        syndrome = detection_events[list(model.detector_ids)].astype(np.uint8).copy()
        for detector_index, detector_id in enumerate(model.detector_ids):
            if detector_id in pending:
                syndrome[detector_index] ^= 1
                if last_window_for_detector[detector_id] == window_index:
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
        for column_index in np.nonzero(committed)[0]:
            for detector_id in faults.future_flips.get(int(column_index), ()):
                pending.symmetric_difference_update({detector_id})
    if pending:
        raise RuntimeError(f"artificial defects were never consumed: {sorted(pending)}"
                           ". The plan does not cover the full detector stream.")
    return total, tuple(outcomes)
