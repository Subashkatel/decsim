"""Slice a global Stim detector error model into per-window decoder inputs.

The measured bits change every shot. The sliced error models are compile-time
data shared across shots. See docs/PAPER_MODEL_MAP.md for the paper contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WindowErrorModel:
    """One window's detector rows, fault columns, and boundary handoff data."""

    detector_ids: tuple
    commit_hi: int
    check: "object"
    priors: "object"
    obs: "object"
    owned: "object"
    future_flips: dict
    defect_positions: dict = None
    h_check: "object" = None
    h_priors: "object" = None
    h2e: "object" = None


def _merge_probability(current: float, incoming: float) -> float:
    """Merge independent faults with p (+) q = p(1-q) + q(1-p)."""
    return current * (1 - incoming) + incoming * (1 - current)


def _error_components(inst) -> tuple:
    """Parse one Stim error instruction into decomposed components and one hyperedge."""
    probability = inst.args_copy()[0]
    components = []
    detectors = []
    observables = []
    all_detectors = []
    all_observables = []

    for target in inst.targets_copy():
        if target.is_separator():
            components.append((tuple(sorted(detectors)), tuple(sorted(observables))))
            detectors = []
            observables = []
        elif target.is_relative_detector_id():
            detectors.append(target.val)
            all_detectors.append(target.val)
        elif target.is_logical_observable_id():
            observables.append(target.val)
            all_observables.append(target.val)

    components.append((tuple(sorted(detectors)), tuple(sorted(observables))))
    hyperedge = (tuple(sorted(all_detectors)), tuple(sorted(all_observables)))
    return probability, components, hyperedge


def detector_error_model_to_faults(dem) -> tuple:
    """Convert a Stim detector error model into merged fault columns."""
    merged: dict = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probability, components, _hyperedge = _error_components(inst)
        for key in components:
            if not key[0]:
                continue
            current_probability = merged.get(key, 0.0)
            merged[key] = _merge_probability(current_probability, probability)
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

    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probability, components, hyperedge_key = _error_components(inst)
        if not hyperedge_key[0]:
            continue

        hyperedge_index = hyper_merged.get(hyperedge_key)
        if hyperedge_index is None:
            hyperedge_index = len(hyper_list)
            hyper_merged[hyperedge_key] = hyperedge_index
            hyper_list.append([hyperedge_key[0], hyperedge_key[1], 0.0])
        hyper_list[hyperedge_index][2] = _merge_probability(
            hyper_list[hyperedge_index][2], probability)

        for component_key in components:
            if not component_key[0]:
                continue
            current = edge_merged.get(component_key, 0.0)
            edge_merged[component_key] = _merge_probability(current, probability)
            h2e_pairs.add((hyperedge_index, component_key))

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


def _belief_matching_fields(columns: list, rows: list, row_index: dict,
                            h_det_sets: list, h_priors: list,
                            hyperedge_to_edge_map) -> dict:
    """Build the optional hyperedge fields used by belief matching."""
    import numpy as np

    hyperedge_columns = sorted({
        hyperedge_index
        for fault_index in columns
        for hyperedge_index in np.nonzero(hyperedge_to_edge_map[fault_index])[0]
    })
    hyperedge_index_by_id = {
        hyperedge_index: column_index
        for column_index, hyperedge_index in enumerate(hyperedge_columns)
    }

    hyperedge_check = np.zeros((len(rows), len(hyperedge_columns)), dtype=np.uint8)
    for hyperedge_index in hyperedge_columns:
        column_index = hyperedge_index_by_id[hyperedge_index]
        for detector_id in h_det_sets[hyperedge_index]:
            if detector_id in row_index:
                hyperedge_check[row_index[detector_id], column_index] = 1

    hyperedge_to_edge = np.zeros((len(columns), len(hyperedge_columns)),
                                 dtype=np.uint8)
    for edge_column, fault_index in enumerate(columns):
        for hyperedge_index in np.nonzero(hyperedge_to_edge_map[fault_index])[0]:
            hyperedge_to_edge[edge_column,
                              hyperedge_index_by_id[hyperedge_index]] = 1

    return {
        "h_check": hyperedge_check,
        "h_priors": np.array([h_priors[hyperedge_index]
                              for hyperedge_index in hyperedge_columns]),
        "h2e": hyperedge_to_edge,
    }


def _fault_owned_by_window(fault_index: int, fault_rounds: list,
                           committed_elsewhere: set, commit_lo: int,
                           commit_hi: int, *, is_last: bool) -> bool:
    """True when this window is responsible for committing the fault."""
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
                         committed_elsewhere: set, commit_lo: int,
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
            fault_index, fault_rounds, committed_elsewhere,
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


def _build_one_window_model(*, det_sets: list, obs_sets: list, priors: list,
                            n_obs: int, round_of: dict, fault_rounds: list,
                            pos_of: dict, committed_elsewhere: set,
                            buffer_lo: int, commit_lo: int, commit_hi: int,
                            buffer_hi: int, is_last: bool,
                            belief_matching: bool = False,
                            h_det_sets: Optional[list] = None,
                            h_priors: Optional[list] = None,
                            hyperedge_to_edge_map=None) -> WindowErrorModel:
    """Build one sliced window and update the fault ownership set."""
    import numpy as np

    rows, row_index, columns = _window_geometry(
        round_of, det_sets, committed_elsewhere,
        buffer_lo, commit_lo, buffer_hi, is_last=is_last)
    check, obs, owned, future_flips = _build_window_arrays(
        rows=rows, columns=columns, row_index=row_index,
        det_sets=det_sets, obs_sets=obs_sets, n_obs=n_obs,
        round_of=round_of, fault_rounds=fault_rounds,
        committed_elsewhere=committed_elsewhere,
        commit_lo=commit_lo, commit_hi=commit_hi, is_last=is_last)

    hyperedge_fields: dict = {}
    if belief_matching:
        hyperedge_fields = _belief_matching_fields(
            columns, rows, row_index, h_det_sets, h_priors,
            hyperedge_to_edge_map)

    return WindowErrorModel(
        detector_ids=tuple(rows),
        commit_hi=commit_hi,
        check=check,
        priors=np.array([priors[fault_index] for fault_index in columns]),
        obs=obs,
        owned=owned,
        future_flips=future_flips,
        defect_positions=_defect_positions(future_flips, round_of, pos_of),
        **hyperedge_fields)


def _fault_data_from_circuit(circuit, *, decompose_errors: bool,
                             belief_matching: bool) -> tuple:
    """Return fault lists and optional belief-matching hyperedge data."""
    dem = circuit.detector_error_model(decompose_errors=decompose_errors)
    if belief_matching:
        return detector_error_model_to_faults_bm(dem)

    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    return det_sets, obs_sets, priors, None, None, None


def _build_models_from_plan(*, plan: list, det_sets: list, obs_sets: list,
                            priors: list, n_obs: int, round_of: dict,
                            fault_rounds: list, pos_of: dict,
                            belief_matching: bool, h_det_sets, h_priors,
                            hyperedge_to_edge_map) -> list:
    """Build all window models for a normalized fault list and plan."""
    models: list = []
    committed_elsewhere: set = set()
    last_window = len(plan) - 1

    for window_index, window_entry in enumerate(plan):
        buffer_lo, commit_lo, commit_hi, buffer_hi = _parse_window_entry(window_entry)
        models.append(_build_one_window_model(
            det_sets=det_sets, obs_sets=obs_sets, priors=priors,
            n_obs=n_obs, round_of=round_of, fault_rounds=fault_rounds,
            pos_of=pos_of, committed_elsewhere=committed_elsewhere,
            buffer_lo=buffer_lo, commit_lo=commit_lo, commit_hi=commit_hi,
            buffer_hi=buffer_hi, is_last=(window_index == last_window),
            belief_matching=belief_matching, h_det_sets=h_det_sets,
            h_priors=h_priors, hyperedge_to_edge_map=hyperedge_to_edge_map))
    return models


def _prepare_model_inputs(circuit, num_observables: Optional[int],
                          decompose_errors: bool,
                          detector_rounds: Optional[dict],
                          belief_matching: bool) -> tuple:
    """Parse the circuit once into the data needed by window builders."""
    det_sets, obs_sets, priors, h_det_sets, h_priors, hyperedge_to_edge_map = \
        _fault_data_from_circuit(
            circuit,
            decompose_errors=decompose_errors,
            belief_matching=belief_matching)
    n_obs = num_observables if num_observables is not None else circuit.num_observables
    round_of = _detector_rounds_from_circuit(circuit, detector_rounds)
    fault_rounds = [tuple(round_of[d] for d in dets) for dets in det_sets]
    pos_of = _detector_position_in_round(round_of)
    return (det_sets, obs_sets, priors, n_obs, round_of, fault_rounds, pos_of,
            h_det_sets, h_priors, hyperedge_to_edge_map)


def build_window_error_models(circuit, plan: list, num_observables: Optional[int] = None,
                          *, decompose_errors: bool = True,
                          detector_rounds: Optional[dict] = None,
                          belief_matching: bool = False) -> list:
    """Slice an operation circuit into one WindowErrorModel per planned window."""
    (det_sets, obs_sets, priors, n_obs, round_of, fault_rounds, pos_of,
     h_det_sets, h_priors, hyperedge_to_edge_map) = _prepare_model_inputs(
         circuit, num_observables, decompose_errors, detector_rounds, belief_matching)
    return _build_models_from_plan(
        plan=plan, det_sets=det_sets, obs_sets=obs_sets, priors=priors,
        n_obs=n_obs, round_of=round_of, fault_rounds=fault_rounds,
        pos_of=pos_of, belief_matching=belief_matching,
        h_det_sets=h_det_sets, h_priors=h_priors,
        hyperedge_to_edge_map=hyperedge_to_edge_map)


def build_single_window_error_model(circuit, window_entry: tuple,
                                    num_observables: Optional[int] = None,
                                    *, decompose_errors: bool = True,
                                    detector_rounds: Optional[dict] = None,
                                    belief_matching: bool = False) -> WindowErrorModel:
    """Build one independent window model without changing a stream ownership cursor."""
    (det_sets, obs_sets, priors, n_obs, round_of, fault_rounds, pos_of,
     h_det_sets, h_priors, hyperedge_to_edge_map) = _prepare_model_inputs(
         circuit, num_observables, decompose_errors, detector_rounds, belief_matching)
    buffer_lo, commit_lo, commit_hi, buffer_hi = _parse_window_entry(window_entry)
    return _build_one_window_model(
        det_sets=det_sets, obs_sets=obs_sets, priors=priors,
        n_obs=n_obs, round_of=round_of, fault_rounds=fault_rounds,
        pos_of=pos_of, committed_elsewhere=set(),
        buffer_lo=buffer_lo, commit_lo=commit_lo, commit_hi=commit_hi,
        buffer_hi=buffer_hi, is_last=False,
        belief_matching=belief_matching, h_det_sets=h_det_sets,
        h_priors=h_priors, hyperedge_to_edge_map=hyperedge_to_edge_map)


class WindowSlicer:
    """Incremental slicer that owns each fault in exactly one window."""

    def __init__(self, circuit, num_observables: Optional[int] = None, *,
                 decompose_errors: bool = True, detector_rounds: Optional[dict] = None,
                 belief_matching: bool = False):
        self.belief_matching = belief_matching
        dem = circuit.detector_error_model(decompose_errors=decompose_errors)
        if belief_matching:
            (self.det_sets, self.obs_sets, self.priors,
             self.h_det_sets, self.h_priors,
             self.hyperedge_to_edge_map) = detector_error_model_to_faults_bm(dem)
        else:
            self.det_sets, self.obs_sets, self.priors = detector_error_model_to_faults(dem)
            self.h_det_sets = self.h_priors = self.hyperedge_to_edge_map = None
        self.n_obs = num_observables if num_observables is not None else circuit.num_observables
        round_of = _detector_rounds_from_circuit(circuit, detector_rounds)
        self.round_of = round_of
        self.fault_rounds = [tuple(round_of[d] for d in dets) for dets in self.det_sets]
        self.pos_of = _detector_position_in_round(round_of)
        self.committed_elsewhere: set = set()

    def slice_window(self, buffer_lo: int, commit_lo: int, commit_hi: int, buffer_hi: int,
                     *, is_last: bool) -> WindowErrorModel:
        """Create one WindowErrorModel and update the fault ownership state."""
        return _build_one_window_model(
            det_sets=self.det_sets, obs_sets=self.obs_sets, priors=self.priors,
            n_obs=self.n_obs, round_of=self.round_of,
            fault_rounds=self.fault_rounds, pos_of=self.pos_of,
            committed_elsewhere=self.committed_elsewhere,
            buffer_lo=buffer_lo, commit_lo=commit_lo, commit_hi=commit_hi,
            buffer_hi=buffer_hi, is_last=is_last,
            belief_matching=self.belief_matching, h_det_sets=self.h_det_sets,
            h_priors=self.h_priors,
            hyperedge_to_edge_map=self.hyperedge_to_edge_map)


def decode_windowed(window_models: list, detection_events, decode_window) -> "object":
    """Decode one shot by walking the committed windows in order."""
    import numpy as np
    pending: set = set()
    total = np.zeros(window_models[0].obs.shape[0], dtype=np.uint8)
    for model in window_models:
        syndrome = detection_events[list(model.detector_ids)].astype(np.uint8).copy()
        for detector_index, detector_id in enumerate(model.detector_ids):
            if detector_id in pending:
                syndrome[detector_index] ^= 1
                pending.discard(detector_id)
        selected = np.asarray(decode_window(model, syndrome), dtype=np.uint8)
        committed = selected.astype(bool) & model.owned
        total ^= (model.obs @ committed.astype(np.uint8)) % 2
        for column_index in np.nonzero(committed)[0]:
            for detector_id in model.future_flips.get(int(column_index), ()):
                pending.symmetric_difference_update({detector_id})
    if pending:
        raise RuntimeError(f"artificial defects were never consumed: {sorted(pending)}"
                           ". The plan does not cover the full detector stream.")
    return total
