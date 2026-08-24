"""Decoder-facing fault vocabulary, requirements and window products.

Holds the FaultRepresentation domain enum, the DecoderFaultModelRequirement
declaration with its four module-scope singletons, the two immutable products
handed to decoders (PlacedFaultModel, WindowErrorModel), and the private
_FaultCatalog record every upper layer of this package consumes.

This module imports nothing from this package and nothing at module scope
beyond the standard library; it is the whole contract an external decoder has
to read.

The fault-domain seam is CLOSED by design: FaultRepresentation has exactly two
members, require_faults dispatches on exactly those two, and
DecoderFaultModelRequirement.__post_init__ compares the representation set by
equality against both of them, so a third domain cannot be declared without
editing this file and its dispatch sites.

Package-internal seam: _FaultCatalog is imported by stim_dem_catalog and
window_placement; the leading underscore means package-private, not
module-private.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
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


def frozen_csc(value):
    """Return ``value`` as an immutable binary ``csc_matrix`` (uint8) with
    sorted indices; dense input is accepted and converted once."""
    from scipy.sparse import csc_matrix, issparse

    matrix = value.tocsc().copy() if issparse(value) else csc_matrix(value)
    matrix = matrix.astype("uint8", copy=False)
    matrix.sum_duplicates()
    matrix.sort_indices()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False
    return matrix


@dataclass(frozen=True)
class PlacedFaultModel:
    """One decoder matrix whose columns all describe the same fault domain.

    ``check`` and ``observables`` are scipy ``csc_matrix`` (uint8), the
    format qLDPC's DetectorErrorModelArrays and PyMatching's
    from_check_matrix use: one column per fault, stored entries only, so a
    window costs its ones and not its zeros (a whole-operation window at
    d=7 x 1000 rounds is 48k x 1.5M with 3M ones).
    ``owned`` says which selected columns this window may commit.
    ``boundary_flips`` stores each owned column's complete global detector
    effect; the destination window intersects it with its own rows, so one
    handoff serves forward and dependency-aware delivery alike.
    ``source_fault_ids`` maps every local column back to the global catalog.
    """

    representation: FaultRepresentation
    check: "object"
    priors: "object"
    observables: "object"
    owned: "object"
    source_fault_ids: tuple[int, ...]
    boundary_flips: dict

    def __post_init__(self) -> None:
        import numpy as np

        for field_name in ("priors", "owned"):
            source = np.asarray(getattr(self, field_name))
            frozen = np.frombuffer(
                source.tobytes(order="C"),
                dtype=source.dtype,
            ).reshape(source.shape)
            object.__setattr__(self, field_name, frozen)
        for field_name in ("check", "observables"):
            object.__setattr__(self, field_name, frozen_csc(getattr(self, field_name)))
        object.__setattr__(self, "source_fault_ids", tuple(self.source_fault_ids))
        frozen_mapping = MappingProxyType({
            int(column): tuple(int(detector_id) for detector_id in detector_ids)
            for column, detector_ids in self.boundary_flips.items()
        })
        object.__setattr__(self, "boundary_flips", frozen_mapping)


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
    defect_positions: dict
    graphlike_faults: Optional[PlacedFaultModel]
    physical_faults: Optional[PlacedFaultModel]
    physical_to_graphlike_detector_projection: "object" = None

    def __post_init__(self) -> None:
        projection = self.physical_to_graphlike_detector_projection
        if projection is not None:
            object.__setattr__(
                self,
                "physical_to_graphlike_detector_projection",
                frozen_csc(projection),
            )

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
