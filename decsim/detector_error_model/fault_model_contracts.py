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
from collections.abc import Mapping
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
    """The fault representations a decoder needs for one operation's code.

    detector_bases asks for each detector's and observable's type, X or
    Z, on the window model, which a decoder that splits a region into
    its X and Z parts reads (basis_split.py).
    """

    representations: frozenset[FaultRepresentation] = frozenset()
    require_physical_to_graphlike_link: bool = False
    detector_bases: bool = False

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
        needs_bases = self.detector_bases or other.detector_bases
        return DecoderFaultModelRequirement(
            representations, needs_link, needs_bases
        )


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
    handoff alike. `source_fault_ids` maps every column to its stable identity
    in the source's fault catalog. Finite sources use catalog positions;
    growing sources preserve identities when their catalog order changes.
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
class FaultCatalogs:
    """One circuit's catalogs, keyed by representation, and their link.

    `link` is a graphlike-by-physical csc_matrix whose column j marks the
    graphlike faults physical fault j is made of, or None when the
    requirement does not ask for it.
    """

    by_representation: dict[FaultRepresentation, FaultCatalog]
    link: Optional[object]


@dataclasses.dataclass(frozen=True)
class WindowErrorModel:
    """What a decoder is handed for one window.

    `detector_ids` lists the window's rows in row order.
    `detector_coordinates` holds Stim's coordinates for those rows, or None
    when any row has none. `defect_positions` maps every detector the
    window can see or hand off to its (round, position in round).
    `first_commit_round` is the first round of the window's commit
    region, which says which of its columns reach behind it.
    `detector_bases` maps every detector of `defect_positions` to its
    type, X or Z, and `observable_bases` gives each logical observable's,
    when the decoder's requirement asks for them; None otherwise.
    """

    detector_ids: tuple[int, ...]
    detector_coordinates: Optional[tuple[tuple[float, ...], ...]]
    defect_positions: dict[int, tuple[int, int]]
    first_commit_round: int
    graphlike_faults: Optional[PlacedFaultModel]
    physical_faults: Optional[PlacedFaultModel]
    physical_to_graphlike_detector_projection: Optional[object] = None
    detector_bases: Optional[Mapping[int, str]] = None
    observable_bases: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.detector_bases is not None:
            bases = dict(self.detector_bases)
            frozen_bases = types.MappingProxyType(bases)
            object.__setattr__(self, "detector_bases", frozen_bases)
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
        faults = self._faults_or_none(representation)
        if faults is None:
            raise RuntimeError(
                f"window model does not contain {representation.value} faults"
            )
        return faults

    def owned_fault_ids(
        self,
    ) -> dict[FaultRepresentation, frozenset[int]]:
        """The catalog faults this window commits, per representation.

        A later window is handed these as its prior faults, so that the
        fault its neighbour has already decided is no column of its own
        model (Bombin et al. 2303.04846 lines 775-788: task j decodes
        over its own error generators). The ids index the circuit's
        fault catalog, which every model of one circuit and one decoder
        requirement shares.
        """
        owned_by_representation = {}
        for representation in FaultRepresentation:
            faults = self._faults_or_none(representation)
            if faults is None:
                continue
            owned_by_representation[representation] = _owned_ids(faults)
        return owned_by_representation

    def crossing_fault_ids(
        self,
    ) -> dict[FaultRepresentation, frozenset[int]]:
        """The faults it commits that reach behind its commit region.

        A window whose near seam no earlier owner closes commits the
        faults crossing it: the window that restarts the weak chain
        after a strong region owns the rounds it reads and the faults
        touching the region's last round with them
        (escalation/strong_regions.py, _fault_exclusions). Its own
        strong redo is pinned on that part of its correction rather than
        deciding those faults again (Toshio et al. 2510.25222 lines
        1248-1250).
        """
        crossing_by_representation = {}
        for representation in FaultRepresentation:
            faults = self._faults_or_none(representation)
            if faults is None:
                continue
            crossing_by_representation[representation] = _crossing_ids(
                self, faults
            )
        return crossing_by_representation

    def reaches_behind(self, detectors: tuple) -> bool:
        """Whether any of those detectors sits before the commit region."""
        for detector_id in detectors:
            position = self.defect_positions[detector_id]
            round_index = position[0]
            if round_index < self.first_commit_round:
                return True
        return False

    def _faults_or_none(
        self, representation: FaultRepresentation
    ) -> Optional[PlacedFaultModel]:
        if representation is FaultRepresentation.GRAPHLIKE:
            return self.graphlike_faults
        return self.physical_faults


_PHYSICAL_ONLY = frozenset({FaultRepresentation.PHYSICAL})
_BOTH_REPRESENTATIONS = GRAPHLIKE_ONLY | _PHYSICAL_ONLY

# The named requirements follow the sets they are built from.
NO_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement()
GRAPHLIKE_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(GRAPHLIKE_ONLY)
PHYSICAL_FAULT_MODEL_REQUIRED = DecoderFaultModelRequirement(_PHYSICAL_ONLY)
# joined onto a row's own requirement when it decodes X and Z apart
DETECTOR_BASES_REQUIRED = DecoderFaultModelRequirement(detector_bases=True)
LINKED_FAULT_MODELS_REQUIRED = DecoderFaultModelRequirement(
    _BOTH_REPRESENTATIONS, require_physical_to_graphlike_link=True
)


def _owned_ids(faults: PlacedFaultModel) -> frozenset[int]:
    """The catalog ids of the columns one placed view commits."""
    owned_ids = set()
    for column_index, fault_id in enumerate(faults.source_fault_ids):
        if faults.owned[column_index]:
            owned_ids.add(fault_id)
    return frozenset(owned_ids)


def _crossing_ids(
    model: WindowErrorModel, faults: PlacedFaultModel
) -> frozenset[int]:
    """The catalog ids of the columns it commits that flip an earlier round."""
    crossing_ids = set()
    for column_index, fault_id in enumerate(faults.source_fault_ids):
        if not faults.owned[column_index]:
            continue
        detectors = faults.boundary_flips[column_index]
        if model.reaches_behind(detectors):
            crossing_ids.add(fault_id)
    return frozenset(crossing_ids)


def _frozen_array(value: object) -> object:
    """A read-only copy of `value`, so no decoder edits a shared window."""
    # Imported here so that reading the contract never loads numpy.
    import numpy

    source = numpy.asarray(value)
    raw_bytes = source.tobytes(order="C")
    flat = numpy.frombuffer(raw_bytes, dtype=source.dtype)
    return flat.reshape(source.shape)
