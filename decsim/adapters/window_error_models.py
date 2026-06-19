from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# =====================================================================================
# PER-WINDOW DECODING PROBLEMS (gap #7)
#
# A decoder needs two inputs: the measured detector
# bits (which arrive shot by shot) and the ERROR MODEL -- the catalog of everything
# that can go wrong, listing for each fault which detectors it flips, how likely it
# is, and whether it flips the logical answer. stim computes that catalog ONCE for
# the whole circuit (the detector error model, DEM). Windowed decoding never builds
# per-window circuits; it slices the one global catalog into per-window pieces.
# A WindowErrorModel is one such piece: the prepared reference sheet a decoder needs
# to solve window k. It is pure compile-time data -- built before any shot exists,
# shared by every shot; only the detector bits change per shot.
#
# The slicing rules are multi-source verified (docs/DESIGN-real-window-decoding.md,
# incl. the 8-source architecture cross-check in its section 2b):
#   - row-slice the global DEM by detector rounds (Tan arXiv:2209.09219; QUITS
#     spacetime(); Huang & Puri arXiv:2311.03307)
#   - a fault is OWNED (committed) by the FIRST window whose commit region it touches
#     (Skoric arXiv:2209.08552's commit-the-crossing-edges rule; QUITS's advancing
#     column cursor; Bombin arXiv:2303.04846's disjoint commit-region partition);
#     the last window owns everything left (the experiment's true closing boundary).
#     Ownership is what guarantees every fault is decided exactly once.
#   - interior window time boundaries are OPEN: a fault whose other detector was cut
#     out of the slice becomes a single-detector column = a boundary edge (Tan's
#     imaginary detectors, mechanically free here)
#   - a committed fault's detector flips BEYOND the commit region are the artificial
#     defects handed forward (Skoric; Huang & Puri's sigma' = sigma + H*xi; QUITS's
#     window_update) -- the note telling the next window "already explained, ignore"
#
# The slice is CODE-AGNOSTIC: the same construction serves the surface code and
# qLDPC / bivariate-bicycle codes (QUITS validates it for qLDPC). Only the inner
# decoder differs: matching_window_decoder() (PyMatching) for matchable codes, built
# with decompose_errors=True; bposd_window_decoder() (ldpc BP-OSD) for BB/qLDPC,
# built with decompose_errors=False since their faults may flip >2 detectors.
# =====================================================================================


@dataclass(frozen=True)
class WindowErrorModel:
    """One window's decoding problem, sliced from the operation's global DEM.

    Rows are the window's detectors (sorted by global id, which stim orders by time);
    columns are the candidate faults this window may select: every fault touching its
    rows that no earlier window committed."""
    detector_ids: tuple        # global detector ids of the rows, in row order
    commit_hi: int             # last committed round (1-based; round = stim t + 1)
    check: "object"            # uint8 (n_rows, n_cols): fault -> in-window detector flips
    priors: "object"           # float (n_cols,): each fault's probability
    obs: "object"              # uint8 (n_obs, n_cols): fault -> logical observable flips
    owned: "object"            # bool (n_cols,): faults THIS window commits
    future_flips: dict         # owned col -> tuple of GLOBAL detector ids it flips
    #                            beyond commit_hi (the artificial defects handed on)
    defect_positions: dict = None  # global det id -> (round, index within that round's
    #                            detectors), for every id in future_flips: lets a decoder
    #                            turn handed-forward defects into the per-round bit masks
    #                            Window.boundary_in speaks (cluster XORs them into the
    #                            dependent window's payloads)
    # Belief-matching only (None unless build_window_error_models(belief_matching=True)):
    # the UNDECOMPOSED hyperedge graph for the BP pass, plus the edge<-hyperedge map, so a
    # belief-matching inner decoder can run BP on the hypergraph and reweight the matching
    # (edge) graph -- the `check`/`owned`/`future_flips` above stay the edge model.
    h_check: "object" = None       # uint8 (n_rows, n_hyperedges): hyperedge -> in-window detector flips
    h_priors: "object" = None      # float (n_hyperedges,): each hyperedge's probability
    h2e: "object" = None           # uint8 (n_edges, n_hyperedges): h2e[e,h]=1 iff edge e is a component of hyperedge h


def detector_error_model_to_faults(dem) -> tuple:
    """The standard DEM -> fault-list conversion (BeliefMatching / QUITS lineage).

    Composite errors (stim's `^`-separated suggested decompositions) are SPLIT into
    their components, each carrying the parent's probability -- the convention
    PyMatching itself applies, required for matchable (<= 2 detectors) columns.
    Identical (detectors, observables) faults merge with p (+) q = p(1-q) + q(1-p).

    Returns (det_sets, obs_sets, priors): parallel lists, one entry per fault."""
    merged: dict = {}                              # (dets, obs) -> prior
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        components, dets, obs = [], [], []
        for t in inst.targets_copy():
            if t.is_separator():
                components.append((tuple(sorted(dets)), tuple(sorted(obs))))
                dets, obs = [], []
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        components.append((tuple(sorted(dets)), tuple(sorted(obs))))
        for key in components:
            if not key[0]:
                continue                           # component with no detectors
            q = merged.get(key, 0.0)
            merged[key] = q * (1 - p) + p * (1 - q)
    det_sets = [k[0] for k in merged]
    obs_sets = [k[1] for k in merged]
    priors = list(merged.values())
    return det_sets, obs_sets, priors


def detector_error_model_to_faults_bm(dem) -> tuple:
    """Belief-matching variant of detector_error_model_to_faults: returns the SAME
    decomposed edge list (det_sets, obs_sets, priors -- byte-identical to the function
    above) PLUS the UNDECOMPOSED hyperedge list and the edge<-hyperedge map needed for the
    BP pass (Higgott & Gidney, arXiv:2203.04948: BP runs on the hyperedge graph, then the
    matching graph is reweighted by the BP posteriors).

    Returns (det_sets, obs_sets, priors, h_det_sets, h_priors, H2E) where h_det_sets /
    h_priors are the undecomposed mechanisms (a hyperedge = the union of an error's
    components, before stim's `^` split) and H2E is a uint8 (n_edges x n_hyperedges) map:
    H2E[e,h]=1 iff edge e is a decomposition component of hyperedge h. Identical-mechanism
    merging uses the same p (+) q = p(1-q)+q(1-p) rule as the edge list, so the edge
    columns come out in the same order as detector_error_model_to_faults."""
    import numpy as np
    edge_merged: dict = {}            # edge (dets,obs) -> prob  (same keying/order as above)
    hyper_merged: dict = {}           # hyper (dets,obs) -> index
    hyper_list: list = []             # [ [dets, obs, prob], ... ]
    h2e_pairs: set = set()            # (hyper_index, edge_key)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comps, dets, obs, all_dets, all_obs = [], [], [], [], []
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append((tuple(sorted(dets)), tuple(sorted(obs))))
                dets, obs = [], []
            elif t.is_relative_detector_id():
                dets.append(t.val); all_dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val); all_obs.append(t.val)
        comps.append((tuple(sorted(dets)), tuple(sorted(obs))))
        h_key = (tuple(sorted(all_dets)), tuple(sorted(all_obs)))
        if not h_key[0]:
            continue
        hi = hyper_merged.get(h_key)
        if hi is None:
            hi = len(hyper_list); hyper_merged[h_key] = hi
            hyper_list.append([h_key[0], h_key[1], 0.0])
        hyper_list[hi][2] = hyper_list[hi][2] * (1 - p) + p * (1 - hyper_list[hi][2])
        for ck in comps:
            if not ck[0]:
                continue
            q = edge_merged.get(ck, 0.0)
            edge_merged[ck] = q * (1 - p) + p * (1 - q)
            h2e_pairs.add((hi, ck))
    edge_keys = list(edge_merged)
    edge_index = {k: i for i, k in enumerate(edge_keys)}
    H2E = np.zeros((len(edge_keys), len(hyper_list)), dtype=np.uint8)
    for hi, ck in h2e_pairs:
        H2E[edge_index[ck], hi] = 1
    return ([k[0] for k in edge_keys], [k[1] for k in edge_keys],
            [edge_merged[k] for k in edge_keys],
            [h[0] for h in hyper_list], [h[2] for h in hyper_list], H2E)


def build_window_error_models(circuit, plan: list, num_observables: Optional[int] = None,
                          *, decompose_errors: bool = True,
                          detector_rounds: Optional[dict] = None,
                          belief_matching: bool = False) -> list:
    """Slice an operation's circuit into one WindowErrorModel per planned window.

    `plan` is scheme-style: [(commit_lo, commit_hi, buffer_hi), ...] in 1-based rounds,
    where round r covers the detectors with stim time coordinate t = r - 1. Detectors
    past the last window's buffer (the final data-measurement layer) join the LAST
    window -- the experiment's true closing time boundary (QUITS's special last
    window; Tan's closed final boundary).

    `decompose_errors` mirrors stim's flag: True (default) splits faults into the
    <= 2-detector components matching decoders require (surface code); False keeps
    whole faults for codes whose DEM is not graphlike (BB / qLDPC -- pair with
    bposd_window_decoder, since matching does not apply).

    `detector_rounds` maps global detector id -> 1-based round, for circuits whose
    detectors carry no time coordinates (e.g. QUITS-built BB circuits, where
    round = id // checks_per_round + 1). Default reads stim coordinates (t + 1).

    `belief_matching` (default False): also fill each window's h_check / h_priors / h2e
    (the undecomposed hyperedge graph + edge<-hyperedge map) so belief_matching_window_decoder
    can run BP on the hypergraph. The decomposed edge model is byte-identical either way;
    only the extra hyperedge fields are added, so default callers are unchanged."""
    import numpy as np
    dem = circuit.detector_error_model(decompose_errors=decompose_errors)
    if belief_matching:
        det_sets, obs_sets, priors, h_det_sets, h_priors, H2E = \
            detector_error_model_to_faults_bm(dem)
    else:
        det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    n_obs = num_observables if num_observables is not None else circuit.num_observables
    if detector_rounds is not None:
        round_of = dict(detector_rounds)
    else:
        coords = circuit.get_detector_coordinates()
        coordless = sum(1 for c in coords.values() if not c)
        if coordless:
            raise ValueError(
                f"{coordless} detectors carry no coordinates; pass detector_rounds "
                "(global detector id -> 1-based round) explicitly")
        round_of = {det: int(c[-1]) + 1 for det, c in coords.items()}
    fault_rounds = [tuple(round_of[d] for d in dets) for dets in det_sets]
    # position of each detector within its round (ascending id), for defect masks
    by_round: dict = {}
    for det in sorted(round_of):
        by_round.setdefault(round_of[det], []).append(det)
    pos_of = {det: i for dets in by_round.values() for i, det in enumerate(dets)}

    models: list = []
    committed_elsewhere: set = set()               # fault indices owned by past windows
    last = len(plan) - 1
    for k, win in enumerate(plan):
        # A plan entry is (commit_lo, commit_hi, buffer_hi) for a TRAILING-buffer-only window,
        # or (buffer_lo, commit_lo, commit_hi, buffer_hi) when the window ALSO has a LEADING
        # buffer -- look-ahead rounds BEFORE the commit region (the parallel A/B / two-sided
        # buffer scheme, Skoric arXiv:2209.08552 Sec I.C: an A window's past time boundary is
        # rough). buffer_lo defaults to commit_lo, so 3-tuple callers and every trailing-only
        # window are byte-identical to before.
        if len(win) == 4:
            buffer_lo, commit_lo, commit_hi, buffer_hi = win
        else:
            commit_lo, commit_hi, buffer_hi = win
            buffer_lo = commit_lo
        # rows: this window's detectors, from its first (leading-buffer or commit) round; the
        # last window keeps everything to the end (the experiment's true closing boundary).
        if k == last:
            rows = sorted(d for d, r in round_of.items() if r >= buffer_lo)
        else:
            rows = sorted(d for d, r in round_of.items()
                          if buffer_lo <= r <= buffer_hi)
        row_index = {d: i for i, d in enumerate(rows)}
        # leading-buffer rounds (strictly before the commit region): their detectors were
        # already committed by earlier windows, but a handed-forward artificial defect can land
        # on them, so they need an incident edge to match it to. Empty for a trailing-only window.
        lead_rows = {d for d in rows if round_of[d] < commit_lo}
        # columns: uncommitted faults touching the rows (as before), PLUS -- only for a window
        # with a leading buffer -- already-committed faults reaching a leading-buffer row,
        # included as UNOWNED boundary edges (the rough past boundary). No leading buffer => no
        # extra columns => identical matrices to before.
        cols: list = []
        for f in range(len(det_sets)):
            if not any(d in row_index for d in det_sets[f]):
                continue
            if f not in committed_elsewhere:
                cols.append(f)
            elif lead_rows and any(d in lead_rows for d in det_sets[f]):
                cols.append(f)                     # rough past-boundary edge (not owned here)
        check = np.zeros((len(rows), len(cols)), dtype=np.uint8)
        obs = np.zeros((n_obs, len(cols)), dtype=np.uint8)
        owned = np.zeros(len(cols), dtype=bool)
        future_flips: dict = {}
        for j, f in enumerate(cols):
            for d in det_sets[f]:
                if d in row_index:
                    check[row_index[d], j] = 1
            for o in obs_sets[f]:
                obs[o, j] = 1
            # ownership: a fault is committed by the FIRST window whose COMMIT REGION it
            # touches (the range commit_lo..commit_hi -- a leading buffer must NOT claim faults
            # belonging to an earlier commit), or by the last window (everything remaining).
            # Already-committed faults (the boundary edges above) are never re-owned -- every
            # fault is owned exactly once. For a trailing-only window this range test is
            # equivalent to the old prefix test (commit regions tile, so an in-cols fault always
            # has a detector at-or-after commit_lo).
            if f not in committed_elsewhere and (
                    k == last or any(commit_lo <= r <= commit_hi for r in fault_rounds[f])):
                owned[j] = True
                committed_elsewhere.add(f)
                beyond = tuple(d for d in det_sets[f] if round_of[d] > commit_hi)
                if beyond and k != last:
                    future_flips[j] = beyond
        defect_positions = {det: (round_of[det], pos_of[det])
                            for flips in future_flips.values() for det in flips}
        h_fields: dict = {}
        if belief_matching:
            # hyperedge slice aligned to this window's edge cols: every hyperedge that
            # parents an in-window edge joins the BP graph (so BP and matching agree).
            h_cols = sorted({h for f in cols for h in np.nonzero(H2E[f])[0]})
            h_index = {h: i for i, h in enumerate(h_cols)}
            h_check = np.zeros((len(rows), len(h_cols)), dtype=np.uint8)
            for h in h_cols:
                for d in h_det_sets[h]:
                    if d in row_index:
                        h_check[row_index[d], h_index[h]] = 1
            h2e = np.zeros((len(cols), len(h_cols)), dtype=np.uint8)
            for j, f in enumerate(cols):
                for h in np.nonzero(H2E[f])[0]:
                    h2e[j, h_index[h]] = 1
            h_fields = dict(h_check=h_check,
                            h_priors=np.array([h_priors[h] for h in h_cols]), h2e=h2e)
        models.append(WindowErrorModel(
            detector_ids=tuple(rows), commit_hi=commit_hi,
            check=check, priors=np.array([priors[f] for f in cols]),
            obs=obs, owned=owned, future_flips=future_flips,
            defect_positions=defect_positions, **h_fields))
    return models


class WindowSlicer:
    """Per-window slicing of ONE circuit's detector error model, one window AT A TIME. Built once
    from a circuit; slice_window() mints one WindowErrorModel per call, threading
    committed_elsewhere so every fault is owned by exactly one window -- regardless of whether the
    windows are sliced all at once (as build_window_error_models does) or INCREMENTALLY as a
    continuous syndrome stream grows. The incremental mode is the runtime round-driven WindowBuilder
    of SWIPER (arXiv:2412.05115 Sec 2.4 / Sec 5.1, Fig. 9): a window is cut and decoded as soon as
    its commit+buffer rounds exist, so an idle stretch of any (runtime-determined) length is
    absorbed by minting more windows -- no compile-time plan.

    slice_window's per-window math is byte-identical to the loop body of build_window_error_models
    (tests/test_window_slicer.py pins this), so the static and runtime builders agree exactly. The
    final, stream-closing window is sliced with is_last=True (Tan arXiv:2209.09219's closed final
    time boundary: it keeps every remaining detector); every interior window uses is_last=False
    (open boundary, defects handed forward)."""
    def __init__(self, circuit, num_observables: Optional[int] = None, *,
                 decompose_errors: bool = True, detector_rounds: Optional[dict] = None,
                 belief_matching: bool = False):
        """Precompute the circuit's fault list, per-fault rounds, and detector positions (the
        compile-time data shared by every window); start with no fault committed."""
        import numpy as np
        self.belief_matching = belief_matching
        dem = circuit.detector_error_model(decompose_errors=decompose_errors)
        if belief_matching:
            (self.det_sets, self.obs_sets, self.priors,
             self.h_det_sets, self.h_priors, self.H2E) = detector_error_model_to_faults_bm(dem)
        else:
            self.det_sets, self.obs_sets, self.priors = detector_error_model_to_faults(dem)
            self.h_det_sets = self.h_priors = self.H2E = None
        self.n_obs = num_observables if num_observables is not None else circuit.num_observables
        if detector_rounds is not None:
            round_of = dict(detector_rounds)
        else:
            coords = circuit.get_detector_coordinates()
            coordless = sum(1 for c in coords.values() if not c)
            if coordless:
                raise ValueError(
                    f"{coordless} detectors carry no coordinates; pass detector_rounds "
                    "(global detector id -> 1-based round) explicitly")
            round_of = {det: int(c[-1]) + 1 for det, c in coords.items()}
        self.round_of = round_of
        self.fault_rounds = [tuple(round_of[d] for d in dets) for dets in self.det_sets]
        by_round: dict = {}
        for det in sorted(round_of):
            by_round.setdefault(round_of[det], []).append(det)
        self.pos_of = {det: i for dets in by_round.values() for i, det in enumerate(dets)}
        self.committed_elsewhere: set = set()

    def slice_window(self, buffer_lo: int, commit_lo: int, commit_hi: int, buffer_hi: int,
                     *, is_last: bool) -> WindowErrorModel:
        """Mint one WindowErrorModel with this geometry (rounds 1-based). is_last keeps every
        remaining detector (the closed final boundary); otherwise rows span buffer_lo..buffer_hi
        with open interior boundaries. Mutates committed_elsewhere so each fault is owned once."""
        import numpy as np
        det_sets, obs_sets, priors = self.det_sets, self.obs_sets, self.priors
        round_of, fault_rounds, pos_of = self.round_of, self.fault_rounds, self.pos_of
        committed_elsewhere = self.committed_elsewhere
        if is_last:
            rows = sorted(d for d, r in round_of.items() if r >= buffer_lo)
        else:
            rows = sorted(d for d, r in round_of.items() if buffer_lo <= r <= buffer_hi)
        row_index = {d: i for i, d in enumerate(rows)}
        lead_rows = {d for d in rows if round_of[d] < commit_lo}
        cols: list = []
        for f in range(len(det_sets)):
            if not any(d in row_index for d in det_sets[f]):
                continue
            if f not in committed_elsewhere:
                cols.append(f)
            elif lead_rows and any(d in lead_rows for d in det_sets[f]):
                cols.append(f)
        check = np.zeros((len(rows), len(cols)), dtype=np.uint8)
        obs = np.zeros((self.n_obs, len(cols)), dtype=np.uint8)
        owned = np.zeros(len(cols), dtype=bool)
        future_flips: dict = {}
        for j, f in enumerate(cols):
            for d in det_sets[f]:
                if d in row_index:
                    check[row_index[d], j] = 1
            for o in obs_sets[f]:
                obs[o, j] = 1
            if f not in committed_elsewhere and (
                    is_last or any(commit_lo <= r <= commit_hi for r in fault_rounds[f])):
                owned[j] = True
                committed_elsewhere.add(f)
                beyond = tuple(d for d in det_sets[f] if round_of[d] > commit_hi)
                if beyond and not is_last:
                    future_flips[j] = beyond
        defect_positions = {det: (round_of[det], pos_of[det])
                            for flips in future_flips.values() for det in flips}
        h_fields: dict = {}
        if self.belief_matching:
            H2E, h_det_sets, h_priors = self.H2E, self.h_det_sets, self.h_priors
            h_cols = sorted({h for f in cols for h in np.nonzero(H2E[f])[0]})
            h_index = {h: i for i, h in enumerate(h_cols)}
            h_check = np.zeros((len(rows), len(h_cols)), dtype=np.uint8)
            for h in h_cols:
                for d in h_det_sets[h]:
                    if d in row_index:
                        h_check[row_index[d], h_index[h]] = 1
            h2e = np.zeros((len(cols), len(h_cols)), dtype=np.uint8)
            for j, f in enumerate(cols):
                for h in np.nonzero(H2E[f])[0]:
                    h2e[j, h_index[h]] = 1
            h_fields = dict(h_check=h_check,
                            h_priors=np.array([h_priors[h] for h in h_cols]), h2e=h2e)
        return WindowErrorModel(
            detector_ids=tuple(rows), commit_hi=commit_hi,
            check=check, priors=np.array([priors[f] for f in cols]),
            obs=obs, owned=owned, future_flips=future_flips,
            defect_positions=defect_positions, **h_fields)


def decode_windowed(window_models: list, detection_events, decode_window) -> "object":
    """The committed-window decoding pass over one shot (the offline reference; the
    cluster performs the same steps event-by-event at runtime).

    For each window in order: take its detectors' bits, XOR in the artificial defects
    handed forward by earlier commits, decode, keep only the OWNED faults, accumulate
    their observable flips, and hand THEIR beyond-commit flips forward. Returns the
    predicted observable flips (XOR over all windows -- the convention the cluster's
    op_results already uses)."""
    import numpy as np
    pending: set = set()                           # artificial defects, by global det id
    total = np.zeros(window_models[0].obs.shape[0], dtype=np.uint8)
    for model in window_models:
        syndrome = detection_events[list(model.detector_ids)].astype(np.uint8).copy()
        for i, det in enumerate(model.detector_ids):
            if det in pending:
                syndrome[i] ^= 1
                pending.discard(det)
        selected = np.asarray(decode_window(model, syndrome), dtype=np.uint8)
        committed = selected.astype(bool) & model.owned
        total ^= (model.obs @ committed.astype(np.uint8)) % 2
        for col in np.nonzero(committed)[0]:
            for det in model.future_flips.get(int(col), ()):
                pending.symmetric_difference_update({det})   # defects XOR (mod 2)
    if pending:
        raise RuntimeError(f"artificial defects were never consumed: {sorted(pending)}"
                           " -- the plan does not cover the full detector stream")
    return total


# INNER DECODERS live in their own per-algorithm packages (each is library-specific), NOT here:
#   * matching_window_decoder()        -> decsim.mwpm_decoder           (pymatching)
#   * bposd_window_decoder()           -> decsim.bposd_decoder          (ldpc BP-OSD)
#   * belief_matching_window_decoder() -> decsim.belief_matching_decoder (ldpc BP + pymatching)
# Only the CODE-AGNOSTIC windowing engine stays in this module: WindowErrorModel (incl. the
# belief-matching h_* fields), build_window_error_models (incl. its belief_matching flag),
# decode_windowed, and detector_error_model_to_faults / _bm. It is shared by every inner decoder.
