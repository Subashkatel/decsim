"""The caller-facing builders.

Holds the plan builder that slices a whole window plan and the two
single-window builders that every production entry point uses.  Nothing inside
this package imports this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..message import WindowProtocol
from .window_slicer import WindowSlicer
from .window_placement import _parse_window_entry
from .window_ownership_dag import (
    _dependency_ancestors,
    _dependency_depths,
    _explicit_fault_ownership,
    _explicit_prior_faults,
)
from .window_protocol_policy import (
    _validate_closed_temporal_boundary_windows,
    _validate_window_protocol,
)

if TYPE_CHECKING:
    from .fault_model_contracts import (
        DecoderFaultModelRequirement,
        WindowErrorModel,
    )


def build_window_error_models(
    circuit,
    plan: list,
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
    Any contiguous partial segment is allowed; production uses suffix
    re-slicing. The last window is terminal only when its commit region
    reaches ``round_count``; faults wholly outside the segment remain unowned.
    """
    entries = tuple(_parse_window_entry(window_entry) for window_entry in plan)
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
    # the terminal window is the one whose commit region reaches the source's
    # last round; the contiguity check above makes it the last entry
    models = [
        slicer.slice_window(
            *window_entry,
            is_last=window_entry[2] == round_count,
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
    *,
    round_count: int,
    detector_rounds: Optional[dict],
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
) -> WindowErrorModel:
    """Build an independent typed model with explicit non-owned ranges."""
    slicer = WindowSlicer(
        circuit,
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
        circuit, window_entry,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def build_single_window_error_model_with_exclusions(
    circuit, window_entry: tuple, *,
    round_count: int,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: DecoderFaultModelRequirement,
    fault_exclusion_ranges: tuple,
) -> WindowErrorModel:
    """Build one independent model with multiple non-owned inclusive ranges."""
    return _build_single_window_error_model(
        circuit, window_entry,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )
