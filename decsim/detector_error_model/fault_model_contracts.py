"""What a decoder is handed: fault representations, requirements, windows.

A fault is one error mechanism of Stim's detector error model: the
detectors it flips, the logical observables it flips, and its probability.
A decoder reads faults in one of two representations. GRAPHLIKE faults
flip one or two detectors each; they are the components Stim writes with
decompose_errors=True and the edges PyMatching matches on. PHYSICAL faults
are the undecomposed mechanisms and may flip any number of detectors; they
are what Stim writes with decompose_errors=False and what belief
propagation and Tesseract read.

A decoder says which representations it needs with a
DecoderFaultModelRequirement. The slicer answers with one WindowErrorModel
per window: the window's detector rows, one PlacedFaultModel per requested
representation, and, when both are requested, the link that says which
graphlike columns each physical column is made of.

The two representations are the only ones. require_faults dispatches on
exactly these two members, so a third cannot be declared without editing
this file.

The matrices are what qLDPC's DetectorErrorModelArrays and PyMatching's
from_check_matrix read: one column per fault, a scipy csc_matrix of uint8,
stored entries only. A window at d=7 over 1000 rounds is 48k rows by 1.5M
columns with 3M ones; storing its zeros would not fit.

This module imports nothing from the package and no numeric library at
module scope, so a decoder that only reads the contract pays for neither.
"""

import dataclasses
import enum
import types
from typing import Optional


class FaultRepresentation(enum.Enum):
    """The fault domain a decoder consumes and returns columns in."""

    GRAPHLIKE = "graphlike"
    PHYSICAL = "physical"


_GRAPHLIKE_ONLY = frozenset({FaultRepresentation.GRAPHLIKE})
_PHYSICAL_ONLY = frozenset({FaultRepresentation.PHYSICAL})
_BOTH_REPRESENTATIONS = _GRAPHLIKE_ONLY | _PHYSICAL_ONLY


@dataclasses.dataclass(frozen=True)
class DecoderFaultModelRequirement:
    """The fault representations a decoder needs for one operation's code."""

    representations: frozenset = frozenset()
    require_physical_to_graphlike_link: bool = False

    def __post_init__(self) -> None:
        has_both = self.representations == _BOTH_REPRESENTATIONS
        if self.require_physical_to_graphlike_link and not has_both:
            raise ValueError(
                "a physical-to-graphlike link requires both fault "
                "representations"
            )

    def joined(
        self, other: "DecoderFaultModelRequirement"
    ) -> "DecoderFaultModelRequirement":
        """The smallest requirement that satisfies both decoders."""
        representations = self.representations | other.representations
        needs_link = (
            self.require_physical_to_graphlike_link
            or other.require_physical_to_graphlike_link
        )
        return DecoderFaultModelRequirement(representations, needs_link)


NO_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement()
GRAPHLIKE_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(_GRAPHLIKE_ONLY)
PHYSICAL_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(_PHYSICAL_ONLY)
LINKED_FAULT_MODELS_REQUIRED = DecoderFaultModelRequirement(
    _BOTH_REPRESENTATIONS, require_physical_to_graphlike_link=True
)


def frozen_sparse_columns(value):
    """`value` as a read-only uint8 csc_matrix with sorted indices.

    A dense matrix is converted once; a sparse one is copied so the
    caller's object is never frozen behind its back.
    """
    # Imported here so that reading the contract never loads scipy.
    import scipy.sparse

    if scipy.sparse.issparse(value):
        column_matrix = value.tocsc()
        matrix = column_matrix.copy()
    else:
        matrix = scipy.sparse.csc_matrix(value)
    matrix = matrix.astype("uint8", copy=False)
    matrix.sum_duplicates()
    matrix.sort_indices()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False
    return matrix


@dataclasses.dataclass(frozen=True)
class PlacedFaultModel:
    """One window's fault columns in one representation.

    `check` is detectors by faults and `observables` is logical observables
    by faults, both uint8 csc_matrix. `priors` holds one probability per
    column. `owned` marks the columns this window commits. `boundary_flips`
    maps each owned column to every detector it flips anywhere in the
    circuit; the window that receives the handoff intersects that with its
    own rows, so one record serves a forward handoff and a dependency
    handoff alike. `source_fault_ids` maps every column back to its
    position in the whole-circuit catalog.
    """

    representation: FaultRepresentation
    check: object
    priors: object
    observables: object
    owned: object
    source_fault_ids: tuple
    boundary_flips: dict

    def __post_init__(self) -> None:
        frozen_priors = _frozen_array(self.priors)
        object.__setattr__(self, "priors", frozen_priors)
        frozen_owned = _frozen_array(self.owned)
        object.__setattr__(self, "owned", frozen_owned)
        frozen_check = frozen_sparse_columns(self.check)
        object.__setattr__(self, "check", frozen_check)
        frozen_observables = frozen_sparse_columns(self.observables)
        object.__setattr__(self, "observables", frozen_observables)
        source_fault_ids = tuple(self.source_fault_ids)
        object.__setattr__(self, "source_fault_ids", source_fault_ids)
        flips_by_column = {}
        for column, detector_ids in self.boundary_flips.items():
            flips = tuple(int(detector_id) for detector_id in detector_ids)
            flips_by_column[int(column)] = flips
        frozen_flips = types.MappingProxyType(flips_by_column)
        object.__setattr__(self, "boundary_flips", frozen_flips)


@dataclasses.dataclass(frozen=True)
class FaultCatalog:
    """Every fault of one circuit in one representation, before slicing.

    Position i of the three tuples describes fault i: the detectors it
    flips, the logical observables it flips, and its probability.
    """

    representation: FaultRepresentation
    detector_sets: tuple
    observable_sets: tuple
    priors: tuple


@dataclasses.dataclass(frozen=True)
class WindowErrorModel:
    """One window's detector rows, fault columns and handoff data.

    `detector_ids` lists the window's rows in row order.
    `detector_coordinates` holds Stim's coordinates for those rows, or None
    when any row has none. `defect_positions` maps every detector the
    window can see or hand off to its (round, position in round).
    """

    detector_ids: tuple
    detector_coordinates: Optional[tuple]
    defect_positions: dict
    graphlike_faults: Optional[PlacedFaultModel]
    physical_faults: Optional[PlacedFaultModel]
    physical_to_graphlike_detector_projection: Optional[object] = None

    def __post_init__(self) -> None:
        projection = self.physical_to_graphlike_detector_projection
        if projection is None:
            return
        frozen_projection = frozen_sparse_columns(projection)
        object.__setattr__(
            self,
            "physical_to_graphlike_detector_projection",
            frozen_projection,
        )

    def require_faults(
        self, representation: FaultRepresentation
    ) -> PlacedFaultModel:
        """The requested view, or a refusal when the window lacks it."""
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


def _frozen_array(value):
    """A read-only copy of `value`, so no decoder edits a shared window."""
    # Imported here so that reading the contract never loads numpy.
    import numpy

    source = numpy.asarray(value)
    raw_bytes = source.tobytes(order="C")
    flat = numpy.frombuffer(raw_bytes, dtype=source.dtype)
    return flat.reshape(source.shape)
