"""The doubtful box the chip packs, against the chip's own union-find growth.

The box holds every doubtful event and takes each cluster whole; union-
find's correction outside the box, with the box's clusters' corrections
added back, is union-find's whole correction. A margin wider than the
window takes every event. The packing reads each event twice, and the
wire names the box's events by row and the outside's corrections by
column. In a run, a box wide enough for every cluster answers as the
whole-window redo does, and the escalation carries the box and no round.
"""

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
    packer = doubtful_box.BoxPacker(margin)
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
    assert box.wire_bits() == 4 * 7 + 3 * 9
    assert box.packing_cycles() == 2 * 13
    syndrome = box.syndrome()
    assert syndrome.sum() == 4
    assert box.numbers() == {
        "box_detection_events": 4.0,
        "box_clusters": 2.0,
        "window_detection_events": 13.0,
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


def _box_card(margin: int):
    """The box redo: the chip packs, the host's matching reads the box."""
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
    redo = strong_window_shapes.RedoWindow.Settings(box_margin=margin)
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
