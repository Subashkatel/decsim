"""The doubtful box the chip packs, against the chip's own union-find growth.

The box holds every doubtful event and takes each cluster whole; union-
find's correction outside the box, with the box's clusters' corrections
added back, is union-find's whole correction. A margin wider than the
window takes every event. The packing reads each event twice, and the
wire names the box's events by row and the outside's corrections by
column. In a run, a box wide enough for every cluster answers as the
whole-window redo does, and the escalation carries the box and no round.

On parallel blocks (Skoric et al. 2209.08552) every joined correction,
of every shape, explains its window's syndrome; checked boundaries
commit what held ones do; and one stream's strong decodes can overlap,
which the sliding chain never lets them do.
"""

import dataclasses

import numpy
import pytest

import decsim
import decsim.confidence.cluster as cluster
import decsim.decoders.cluster_record as cluster_record
import decsim.decoders.minimum_weight_perfect_matching.box_decoder as box_decoder  # noqa: E501
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.minimum_weight_perfect_matching.detection_event_time as event_time  # noqa: E501
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.doubtful_box as doubtful_box
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.machine as machine_module
import decsim.presets.clocks as clocks
import decsim.presets.machines as machines
import decsim.records.fault_model_contracts as fault_models
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.parallel as parallel_scheme
import tests.escalation.all_walks_oracle as all_walks_oracle
import tests.escalation.test_switching_mode as switching_mode
from tests.decoders import windows

LINKED = fault_models.LINKED_FAULT_MODELS_REQUIRED
DISTANCE = 5
ROUNDS = 5
PROBABILITY = 0.006
SHOTS = 200
# wider than any window here, in rounds
WIDE_MARGIN = 100
CYCLE_COUNT = cycle_count.CycleCount(clock=clocks.FRIDGE_CLOCK)
UNION_FIND = union_find.UnionFindDecoder.Settings(timing=CYCLE_COUNT)


def packed_windows(margin: int) -> list:
    """(job, weak result, box) of each window that has a doubtful cluster."""
    circuit = windows.memory_circuit(DISTANCE, ROUNDS, PROBABILITY)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    weak = UNION_FIND.build()
    shape = doubtful_box.BoxShape(margin, margin)
    packer = doubtful_box.BoxPacker(shape)
    packed = []
    for shot in detection_events:
        job = windows.job_for(model, shot)
        result = weak.decode(job)
        record = cluster_record.cluster_record(result.cluster_evidence)
        if record.walk_cluster_count == 0:
            continue
        box = packer.pack(job, result)
        packed.append((job, result, box))
    return packed


def test_the_box_holds_every_doubtful_event_and_takes_clusters_whole():
    for margin in (0, 1, 2):
        for _job, result, box in packed_windows(margin):
            evidence = result.cluster_evidence
            roots = _roots(evidence)
            events = numpy.flatnonzero(evidence.syndrome)
            record = cluster_record.cluster_record(evidence)
            walk_rows = record.walk_detection_events
            box_rows = box.event_rows.tolist()
            assert set(walk_rows) <= set(box_rows)
            box_roots = roots[box.event_rows]
            chosen_list = box_roots.tolist()
            chosen = set(chosen_list)
            assert len(chosen) == box.cluster_count
            in_chosen = [row for row in events if roots[row] in chosen]
            assert sorted(in_chosen) == sorted(box_rows)


def test_the_outside_and_the_boxs_clusters_make_union_finds_correction():
    for _job, result, box in packed_windows(1):
        evidence = result.cluster_evidence
        weak = numpy.asarray(evidence.selected_faults, dtype=numpy.uint8)
        roots = _roots(evidence)
        chosen = numpy.unique(roots[box.event_rows])
        inside = doubtful_box._box_columns(evidence.graph, roots, chosen)
        overlap = box.outside & inside
        rejoined = box.outside | (weak & inside)
        assert not numpy.any(overlap)
        numpy.testing.assert_array_equal(rejoined, weak)


def test_a_margin_wider_than_the_window_takes_every_event():
    for _job, result, box in packed_windows(WIDE_MARGIN):
        events = numpy.flatnonzero(result.cluster_evidence.syndrome)
        numpy.testing.assert_array_equal(box.event_rows, events)
        assert not numpy.any(box.outside)


def test_the_packing_and_the_wire_are_counted_from_the_box():
    outside = numpy.zeros(300, dtype=numpy.uint8)
    outside[[3, 7, 299]] = 1
    event_rows = numpy.asarray([0, 5, 9, 40])
    box = doubtful_box.DoubtfulBox(
        event_rows=event_rows,
        outside=outside,
        cluster_count=2,
        window_event_count=13,
        detector_count=100,
    )
    # 100 rows take 7 bits a name, 300 columns 9
    expected_wire_bits = 4 * 7 + 3 * 9
    assert box.wire_bits() == expected_wire_bits
    assert box.packing_cycles() == 2 * 13
    syndrome = box.syndrome()
    assert syndrome.sum() == 4
    assert box.numbers() == {
        "box_detection_events": 4.0,
        "box_clusters": 2.0,
        "window_detection_events": 13.0,
        "box_wire_bits": float(expected_wire_bits),
        "box_outside_corrections": 3.0,
        "box_volume_detectors": 0.0,
        "box_doubtful_events": 0.0,
        "window_detectors": 100.0,
    }


@pytest.mark.parametrize("margin", [-1, 1.5, True, "2"])
def test_a_margin_is_a_whole_number_of_rounds(margin):
    with pytest.raises(ValueError, match="box_margin"):
        strong_window_shapes.RedoWindow.Settings(box_margin=margin)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_box_wide_enough_for_every_cluster_answers_as_the_whole_window_redo(
    seed,
):
    """Every cluster in the box: matching on the whole window, as the redo."""
    redo_card = _redo_card()
    box_card = _box_card(WIDE_MARGIN)
    _redo_machine, redo_result = switching_mode.run_shot(redo_card, seed)
    box_machine, box_result = switching_mode.run_shot(box_card, seed)
    assert switching_mode.predictions(box_result) == switching_mode.predictions(
        redo_result
    )
    assert _escalations(box_machine) > 0


def test_the_escalation_carries_the_box_and_no_round():
    """One message a redo, the selection with the box; no round moves.

    The whole-window redo carries its window's rounds the strong side
    lacks over the same hop, and its strong unit reads them.
    """
    seed = 0
    box_card = _box_card(2)
    redo_card = _redo_card()
    box_machine, box_result = switching_mode.run_shot(box_card, seed)
    redo_machine, redo_result = switching_mode.run_shot(redo_card, seed)

    box_hops = _escalation_hops(box_result)
    redo_hops = _escalation_hops(redo_result)

    escalations = _escalations(box_machine)
    assert escalations == _escalations(redo_machine) > 0
    assert len(box_hops["weak_to_strong"]) == escalations
    assert len(redo_hops["weak_to_strong"]) > escalations
    assert box_hops["strong_buffer_to_strong"]
    assert all(bits == 0 for bits in box_hops["strong_buffer_to_strong"])
    assert any(bits > 0 for bits in redo_hops["strong_buffer_to_strong"])


def test_a_box_redo_needs_a_row_that_reads_the_box():
    box_card = _box_card(2)
    strong = matching.CorrelatedPyMatchingDecoder.Settings()
    card = box_card.with_settings(machines.STRONG_POOL, algorithm=strong)
    with pytest.raises(ValueError, match="reads a window's rounds"):
        machine_module.Machine.build(card, 0)


def test_a_box_redo_refuses_a_strong_decode_started_before_the_verdict():
    box_card = _box_card(2)
    card = switching_mode.with_switching(box_card, run_both_at_once=True)
    with pytest.raises(ValueError, match="before it"):
        machine_module.Machine.build(card, 0)


# ---- helpers


def _roots(evidence) -> numpy.ndarray:
    root_list = cluster_record.detector_roots(
        evidence.graph, evidence.edge_intervals
    )
    return numpy.asarray(root_list)


def _union_find_card():
    """The gate's card at 20 dB, union-find and its cluster gap on the chip."""
    gate = switching_mode.switching_graph(20.0)
    on_union_find = gate.with_settings(machines.WEAK_POOL, algorithm=UNION_FIND)
    gap = cluster.ClusterGap.Settings(walk_microseconds=0.0)
    return switching_mode.with_switching(on_union_find, confidence=gap)


def _redo_card():
    """The whole-window redo, correlated matching on the host."""
    strong = matching.CorrelatedPyMatchingDecoder.Settings()
    union_find_card = _union_find_card()
    return union_find_card.with_settings(machines.STRONG_POOL, algorithm=strong)


def _box_card(margin: int, **redo_fields):
    """The box redo: the chip packs, the host's matching reads the box.

    redo_fields are the redo window's other settings: the box's shape
    and the boundary row.
    """
    law = event_time.DetectionEventTime(
        intercept_microseconds=5.0,
        coefficient_microseconds=0.5,
        exponent=1.2,
        scale=1.0,
        source="test",
    )
    strong = box_decoder.BoxCorrelatedPyMatchingDecoder.Settings(
        timing=law, join_microseconds=0.5
    )
    redo = strong_window_shapes.RedoWindow.Settings(
        box_margin=margin, **redo_fields
    )
    union_find_card = _union_find_card()
    card = switching_mode.with_switching(union_find_card, strong_window=redo)
    return card.with_settings(machines.STRONG_POOL, algorithm=strong)


def _escalation_hops(result) -> dict:
    """The payload bits of each transfer on the escalation's two hops."""
    hops = {"weak_to_strong": [], "strong_buffer_to_strong": []}
    for transfer in result.link_traffic["transfers"]:
        bits = hops.get(transfer["path"])
        if bits is not None:
            bits.append(transfer["payload_bits"])
    return hops


def _escalations(machine) -> int:
    """The windows the strong tier answered."""
    strong = 0
    for _window, frame in measure.decoded_windows(machine.observation):
        strong += frame.tier == machines.STRONG_POOL
    return strong


def test_every_window_of_a_box_run_has_its_final_answer_latency():
    """A box decode reads no round; its answer rests on the chip's rounds.

    The latency from a window's last round to its final answer reads the
    rounds the committing decode rests on (measure.py
    last_required_send_ticks), so every window, the boxed ones too, has
    one.
    """
    box_card = _box_card(2)
    task = decsim.Task("box_redo", box_card, {})
    shot = collect.run_shot(task, 0)
    measurement = measure.measure_shot(shot)
    latencies = measurement.samples["window_last_round_to_final_answer"]
    windows = measure.decoded_windows(shot.machine.observation)
    assert _escalations(shot.machine) > 0
    assert len(latencies) == len(windows)
    assert all(latency > 0 for latency in latencies)


# ---- the box's shapes (doubtful_box.BoxShape)


def shaped_windows(shape) -> list:
    """(weak result, box, cluster record) of each window with doubt."""
    circuit = windows.memory_circuit(DISTANCE, ROUNDS, PROBABILITY)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    weak = UNION_FIND.build()
    packer = doubtful_box.BoxPacker(shape)
    packed = []
    for shot in detection_events:
        job = windows.job_for(model, shot)
        result = weak.decode(job)
        record = cluster_record.cluster_record(result.cluster_evidence)
        if record.walk_cluster_count == 0:
            continue
        box = packer.pack(job, result)
        packed.append((result, box, record))
    return packed


def _chosen(result, box) -> set:
    roots = _roots(result.cluster_evidence)
    box_roots = roots[box.event_rows]
    chosen_list = box_roots.tolist()
    return set(chosen_list)


def test_the_shortest_odd_walk_found_again_is_the_gap():
    """odd_walks prices walks as cluster_gap.c does: the shortest is the gap."""
    shape = doubtful_box.BoxShape(1, 1)
    for result, _box, record in shaped_windows(shape):
        evidence = result.cluster_evidence
        walks = all_walks_oracle.odd_walks(
            evidence.graph, evidence.edge_intervals
        )
        assert min(length for length, _edges in walks) == record.walk_half_ticks


def test_all_walks_hold_the_shortest_walks_doubtful_events():
    nats = 40.0  # past every gap here, so every window has its walk kept
    shape = doubtful_box.BoxShape(0, 0, doubt="all_walks", doubt_nats=nats)
    for _result, box, record in shaped_windows(shape):
        box_rows = box.event_rows.tolist()
        assert set(record.walk_detection_events) <= set(box_rows)
        assert box.doubtful_event_count >= record.walk_event_count


def test_all_walks_mark_the_rows_the_moved_python_walk_marks():
    """The compiled walks under each limit are all_walks_oracle.py's.

    The limits run from below every gap here to past every gap, so the
    corpus holds windows with no walk kept, the shortest alone, and
    walks from several sources, ties included.
    """
    kept_more = 0
    for nats in (0.5, 2.0, 4.605170, 40.0):
        shape = doubtful_box.BoxShape(0, 0, doubt="all_walks", doubt_nats=nats)
        for result, _box, record in shaped_windows(shape):
            evidence = result.cluster_evidence
            roots = _roots(evidence)
            event_rows = numpy.flatnonzero(evidence.syndrome)
            limit_half_ticks = 2.0 * nats / evidence.graph.weight_step
            moved = all_walks_oracle.all_walk_rows(
                evidence, roots, event_rows, limit_half_ticks
            )
            root_list = roots.tolist()
            rows = doubtful_box.doubtful_rows(evidence, root_list, shape, False)
            assert rows == moved
            kept_more += len(rows) > record.walk_event_count
    assert kept_more > 0


@pytest.mark.parametrize("nats", [0.0, -1.0, float("nan")])
def test_doubt_nats_is_a_walk_weight_above_zero(nats):
    with pytest.raises(ValueError, match="above zero"):
        doubtful_box.BoxShape(0, 0, doubt="all_walks", doubt_nats=nats)


def test_a_graph_shape_of_no_edges_takes_the_doubtful_clusters_alone():
    shape = doubtful_box.BoxShape(0, 0, form="graph")
    for result, box, record in shaped_windows(shape):
        roots = _roots(result.cluster_evidence)
        walk_rows = list(record.walk_detection_events)
        walk_root_list = roots[walk_rows].tolist()
        assert _chosen(result, box) == set(walk_root_list)


def test_narrower_shapes_take_no_cluster_the_wider_box_leaves_out():
    wide_shape = doubtful_box.BoxShape(2, 2)
    wide = shaped_windows(wide_shape)
    for shape in (
        doubtful_box.BoxShape(1, 2),
        doubtful_box.BoxShape(2, 1),
        doubtful_box.BoxShape(2, 2, form="per_cluster"),
    ):
        narrow = shaped_windows(shape)
        pairs = zip(narrow, wide, strict=True)
        for (result, box, _), (_, wide_box, _) in pairs:
            assert _chosen(result, box) <= _chosen(result, wide_box)
            assert box.volume_detectors <= wide_box.volume_detectors


def test_a_shape_setting_needs_a_box_and_a_graph_one_margin():
    with pytest.raises(ValueError, match="box_margin"):
        strong_window_shapes.RedoWindow.Settings(box_form="graph")
    with pytest.raises(ValueError, match="equal"):
        strong_window_shapes.RedoWindow.Settings(
            box_margin=1, box_time_margin=2, box_form="graph"
        )
    with pytest.raises(ValueError, match="doubt_nats"):
        strong_window_shapes.RedoWindow.Settings(
            box_margin=1, box_doubt="all_walks"
        )
    redo = strong_window_shapes.RedoWindow.Settings(
        box_margin=2, box_time_margin=1
    )
    shape = redo.box_shape()
    assert (shape.time_margin, shape.space_margin) == (1, 2)


# ---- the box on parallel blocks (Skoric et al. 2209.08552)


def _on_parallel_blocks(card):
    scheme = parallel_scheme.ParallelWindowScheme.Settings()
    return card.with_settings(machines.WINDOWS, scheme=scheme)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_on_parallel_blocks_each_seam_commits_after_both_its_blocks(seed):
    """A box redo pins no face, so an A block can escalate.

    A B block reconciles the seam between two A blocks; its final
    correction rests on theirs, so it commits only once both have,
    their strong answers included.
    """
    box_card = _box_card(2)
    card = _on_parallel_blocks(box_card)
    machine, _result = switching_mode.run_shot(card, seed)
    commits = _commit_ticks_by_window(machine)
    seams = [index for index in commits if index % 2 == 1]
    early = _seams_committed_early(commits, seams)
    assert seams
    assert early == []


def test_parallel_blocks_escalate_to_the_box():
    box_card = _box_card(2)
    card = _on_parallel_blocks(box_card)
    escalations = 0
    for seed in range(4):
        machine, _result = switching_mode.run_shot(card, seed)
        escalations += _escalations(machine)
    assert escalations > 0


def test_a_whole_window_redo_on_parallel_blocks_is_still_refused():
    """It reads its commit and one buffer: an A block's past would be lost."""
    redo_card = _redo_card()
    card = _on_parallel_blocks(redo_card)
    with pytest.raises(NotImplementedError, match="pins its past face"):
        machine_module.Machine.build(card, 0)


def _commit_ticks_by_window(machine) -> dict:
    """Each decoded window's commit tick, by its window index."""
    commits = {}
    for window, frame in measure.decoded_windows(machine.observation):
        commits[window.window_index] = frame.committed_ticks
    return commits


def _seams_committed_early(commits: dict, seams: list) -> list:
    """(seam, block) of each seam that committed before a block beside it."""
    pairs = [(seam, block) for seam in seams for block in (seam - 1, seam + 1)]
    return [
        (seam, block)
        for seam, block in pairs
        if block in commits and commits[seam] < commits[block]
    ]


# the box's shapes the joined correction is checked for, as redo settings
SHAPES = {
    "margin 2": {},
    "margin 0": {"box_time_margin": 0, "box_space_margin": 0},
    "1 round, 2 columns": {"box_time_margin": 1},
    "per cluster": {"box_form": "per_cluster"},
    "graph, 1 edge": {
        "box_time_margin": 1,
        "box_space_margin": 1,
        "box_form": "graph",
    },
    "all walks": {"box_doubt": "all_walks", "box_doubt_nats": 4.605170},
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_on_parallel_blocks_every_joined_correction_explains_its_window(
    shape, monkeypatch
):
    """Each cluster's correction explains its own events, so the join does.

    The chip's window syndrome is kept for each box it packs, and each
    joined correction the host returns is checked against it. A window
    decoded again packs a new box, so each box is checked on its own.
    """
    syndromes = _kept_window_syndromes(monkeypatch)
    checked = _checked_joins(monkeypatch, syndromes)
    box_card = _box_card(2, **SHAPES[shape])
    card = _on_parallel_blocks(box_card)
    for seed in range(4):
        switching_mode.run_shot(card, seed)
    assert checked["joins"] > 0
    assert checked["unexplained"] == []


@pytest.mark.parametrize("seed", range(6))
def test_on_parallel_blocks_checked_boundaries_answer_as_held_ones(seed):
    """A block read on a provisional face is decoded again if it changes.

    Each B block reads its A blocks' faces as they are first answered
    and checks them when they are final (windows/input_checks.py), so
    its final answer is the one it gives when it waits for them.
    """
    held = _parallel_box_card(HELD)
    checked = _parallel_box_card(CHECKED)
    _held_machine, held_result = switching_mode.run_shot(held, seed)
    _checked_machine, checked_result = switching_mode.run_shot(checked, seed)
    assert switching_mode.predictions(
        checked_result
    ) == switching_mode.predictions(held_result)


def test_on_parallel_blocks_a_changed_face_is_decoded_again():
    """The check is not idle: a B block reads a face the strong side changes."""
    checked = _parallel_box_card(CHECKED)
    again = 0
    for seed in range(6):
        machine, _result = switching_mode.run_shot(checked, seed)
        again += _statistic(machine, "decoded_again")
    assert again > 0


def test_one_streams_strong_decodes_overlap_on_parallel_blocks_alone():
    """Two strong units: parallel blocks use both, the sliding chain one.

    On the sliding chain a window's escalation waits for the window
    before it to be final, so one stream has one strong decode at a
    time. On parallel blocks the A blocks rest on no other block, so
    their strong decodes can run at once.
    """
    box_card = _box_card(2)
    sliding = _with_boundary_policy(box_card, CHECKED)
    sliding = sliding.with_settings(machines.STRONG_POOL, unit_count=2)
    parallel = _on_parallel_blocks(sliding)
    sliding_overlaps = 0
    parallel_overlaps = 0
    for seed in range(6):
        sliding_machine, _ = switching_mode.run_shot(sliding, seed)
        parallel_machine, _ = switching_mode.run_shot(parallel, seed)
        sliding_overlaps += _overlapping_strong_decodes(sliding_machine)
        parallel_overlaps += _overlapping_strong_decodes(parallel_machine)
    assert sliding_overlaps == 0
    assert parallel_overlaps > 0


HELD = boundary_policies.Held.Settings()
CHECKED = boundary_policies.Checked.Settings()


def _parallel_box_card(policy):
    """The box redo of margin 2 on parallel blocks, on one boundary row."""
    box_card = _box_card(2)
    on_policy = _with_boundary_policy(box_card, policy)
    return _on_parallel_blocks(on_policy)


def _with_boundary_policy(card, policy):
    """The card with its windows and its redo window on one boundary row."""
    switching_part = card.part(machines.SWITCHING)
    switching = switching_part.settings
    redo = dataclasses.replace(switching.strong_window, boundary_policy=policy)
    card = switching_mode.with_switching(card, strong_window=redo)
    return card.with_settings(machines.WINDOWS, boundary_policy=policy)


def _statistic(machine, name: str) -> float:
    for row in machine.observation.statistics.rows():
        if row.statistic == name:
            return row.value
    raise KeyError(name)


def _kept_window_syndromes(monkeypatch) -> dict:
    """Each packed box's window syndrome, by the box's identity."""
    syndromes = {}
    pack = doubtful_box.BoxPacker.pack

    def keeping_pack(packer, weak_job, weak_result):
        box = pack(packer, weak_job, weak_result)
        syndrome = weak_result.cluster_evidence.syndrome
        syndromes[id(box)] = (box, numpy.asarray(syndrome, dtype=int))
        return box

    monkeypatch.setattr(doubtful_box.BoxPacker, "pack", keeping_pack)
    return syndromes


def _checked_joins(monkeypatch, syndromes: dict) -> dict:
    """Each joined correction against its window's syndrome."""
    checked = {"joins": 0, "unexplained": []}
    row = box_decoder.BoxCorrelatedPyMatchingDecoder
    answer = row.window_answer

    def checking_answer(decoder, job, backend, model, faults, syndrome):
        result = answer(decoder, job, backend, model, faults, syndrome)
        _box, window_syndrome = syndromes[id(job.request)]
        dense_check = faults.check.toarray()
        check = dense_check.astype(int)
        selected = numpy.asarray(result.selected_faults, dtype=int)
        explained = check @ selected % 2
        checked["joins"] += 1
        if not numpy.array_equal(explained, window_syndrome):
            checked["unexplained"].append(job.label)
        return result

    monkeypatch.setattr(row, "window_answer", checking_answer)
    return checked


def _overlapping_strong_decodes(machine) -> int:
    """Pairs of one stream's strong decodes of two windows that overlap."""
    spans = _strong_spans(machine)
    keys = sorted(spans)
    pairs = [
        (first, second)
        for index, first in enumerate(keys)
        for second in keys[index + 1 :]
        if first[0] == second[0]
    ]
    overlaps = 0
    for first, second in pairs:
        first_start, first_end = spans[first]
        second_start, second_end = spans[second]
        overlaps += first_start < second_end and second_start < first_end
    return overlaps


def _strong_spans(machine) -> dict:
    """(first start, last end) of each window's strong decodes, by window."""
    strong_unit = f"{machines.STRONG_POOL}#"
    spans = {}
    for record in machine.observation.stages.records:
        if not record.unit_name.startswith(strong_unit):
            continue
        key = (record.stream_id, record.window_id)
        start, end = spans.get(key, (record.start_ticks, record.end_ticks))
        spans[key] = (
            min(start, record.start_ticks),
            max(end, record.end_ticks),
        )
    return spans
