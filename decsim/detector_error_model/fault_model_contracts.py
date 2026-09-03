"""What a decoder is handed: fault representations, requirements, windows.

A fault is one error mechanism of Stim's detector error model, read
either as GRAPHLIKE components of one or two detectors
(decompose_errors=True, the edges PyMatching matches on) or as the
PHYSICAL undecomposed mechanism (decompose_errors=False, what belief
propagation and Tesseract read); a decoder states which it needs, and
the slicer answers with one WindowErrorModel per window. The matrices
are one uint8 csc column per fault, what qLDPC's DetectorErrorModelArrays
and PyMatching's from_check_matrix read.
"""

import dataclasses
import enum
import types
from typing import Optional


class FaultRepresentation(enum.Enum):
    """The fault domain a decoder works in, for columns read or returned."""

    GRAPHLIKE = "graphlike"
    PHYSICAL = "physical"


# Public because a requirement and the protocol policy compare against
# the same set: window_protocol_policy holds a Tan plan to GRAPHLIKE_ONLY.
GRAPHLIKE_ONLY = frozenset({FaultRepresentation.GRAPHLIKE})


@dataclasses.dataclass(frozen=True)
class DecoderFaultModelRequirement:
    """The fault representations a decoder needs for one operation's code."""

    representations: frozenset[FaultRepresentation] = frozenset()
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


def frozen_sparse_columns(value: object) -> object:
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
    source_fault_ids: tuple[int, ...]
    boundary_flips: dict[int, tuple[int, ...]]

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
    detector_sets: tuple[tuple[int, ...], ...]
    observable_sets: tuple[tuple[int, ...], ...]
    priors: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class WindowErrorModel:
    """What a decoder is handed for one window.

    `detector_ids` lists the window's rows in row order.
    `detector_coordinates` holds Stim's coordinates for those rows, or None
    when any row has none. `defect_positions` maps every detector the
    window can see or hand off to its (round, position in round).
    """

    detector_ids: tuple[int, ...]
    detector_coordinates: Optional[tuple[tuple[float, ...], ...]]
    defect_positions: dict[int, tuple[int, int]]
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
        """The requested view; asking for one the window lacks is a bug."""
        if representation is FaultRepresentation.GRAPHLIKE:
            faults = self.graphlike_faults
        elif representation is FaultRepresentation.PHYSICAL:
            faults = self.physical_faults
        else:
            raise RuntimeError(
                "representation must be a FaultRepresentation value"
            )
        if faults is None:
            raise RuntimeError(
                f"window model does not contain {representation.value} faults"
            )
        return faults


_PHYSICAL_ONLY = frozenset({FaultRepresentation.PHYSICAL})
_BOTH_REPRESENTATIONS = GRAPHLIKE_ONLY | _PHYSICAL_ONLY

# The named requirements follow the sets they are built from.
NO_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement()
GRAPHLIKE_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(GRAPHLIKE_ONLY)
PHYSICAL_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(_PHYSICAL_ONLY)
LINKED_FAULT_MODELS_REQUIRED = DecoderFaultModelRequirement(
    _BOTH_REPRESENTATIONS, require_physical_to_graphlike_link=True
)


def _frozen_array(value: object) -> object:
    """A read-only copy of `value`, so no decoder edits a shared window."""
    # Imported here so that reading the contract never loads numpy.
    import numpy

    source = numpy.asarray(value)
    raw_bytes = source.tobytes(order="C")
    flat = numpy.frombuffer(raw_bytes, dtype=source.dtype)
    return flat.reshape(source.shape)
