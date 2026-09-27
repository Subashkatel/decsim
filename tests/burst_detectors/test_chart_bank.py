"""The chart bank held against the method's rules, written out.

written_rules.py transcribes Page's CUSUM (1954) with Lucas's count
step (1985), Kulldorff's circular zones (1997) and Q3DE's mask
(2501.00331 lines 729-733) rule by rule, with none of the row's code;
the pairs are read off Stim's generated circuit.
"""

import numpy

import decsim.frontends.settings as workload_settings
import tests.burst_detectors.burst_rounds as burst_rounds
import tests.burst_detectors.written_rules as written_rules


def test_the_bank_scores_every_group_as_the_written_rules_do():
    """All 15 group scores of every round, on a burst that trips the mask.

    Only the order of the float sums differs from the transcription.
    """
    settings = burst_rounds.CUSUM.Settings(calibration_shot_count=200)
    detector = burst_rounds.cusum_detector(
        settings, rounds=burst_rounds.LONG_ROUNDS
    )
    positions, pairs, usual_rates = burst_rounds.bank_inputs(detector)
    circuit, sampled = burst_rounds.long_burst_shot(seed=4)
    rows = burst_rounds.bulk_rows(circuit, sampled, positions)

    scores = _bank_scores(detector, rows)
    reference, masked = written_rules.reference_scores(
        rows, positions, pairs, usual_rates
    )

    assert masked > 0
    numpy.testing.assert_allclose(scores, reference, rtol=1e-9, atol=1e-9)


def test_a_null_mask_count_scores_the_unmasked_regional_cusum():
    settings = burst_rounds.CUSUM.Settings(
        mask_count=None, calibration_shot_count=200
    )
    detector = burst_rounds.cusum_detector(
        settings, rounds=burst_rounds.LONG_ROUNDS
    )
    positions, pairs, usual_rates = burst_rounds.bank_inputs(detector)
    circuit, sampled = burst_rounds.long_burst_shot(seed=4)
    rows = burst_rounds.bulk_rows(circuit, sampled, positions)

    scores = _bank_scores(detector, rows)
    reference, masked = written_rules.reference_scores(
        rows, positions, pairs, usual_rates, mask_count=None
    )

    assert masked == 0
    numpy.testing.assert_allclose(scores, reference, rtol=1e-9, atol=1e-9)


def test_no_radii_leave_the_whole_patch_the_only_region():
    settings = burst_rounds.CUSUM.Settings(
        region_radii=(), calibration_shot_count=200
    )
    detector = burst_rounds.cusum_detector(settings)
    bank = detector.charts_by_operation[1].calibration.bank
    thresholds = detector.charts_by_operation[1].calibration.thresholds

    assert bank.incidence.shape == (24, 1)
    assert len(thresholds) == len(written_rules.REFERENCE_DESIGNS)


def test_the_pairs_are_the_checks_one_data_qubit_flip_fires_together():
    settings = burst_rounds.CUSUM.Settings(calibration_shot_count=200)
    detector = burst_rounds.cusum_detector(settings)
    positions, pairs, _ = burst_rounds.bank_inputs(detector)
    circuit = workload_settings.memory_circuit(
        burst_rounds.CODE_TASK,
        burst_rounds.ROUNDS,
        burst_rounds.DISTANCE,
        burst_rounds.PHYSICAL_ERROR_PROBABILITY,
    )

    bank_pairs = _position_pairs(positions, pairs)

    assert len(bank_pairs) == 30
    assert bank_pairs == written_rules.reference_pairs(circuit)


def _bank_scores(detector, rows):
    """The row's own bank scoring the rows from a cold state."""
    calibration = detector.charts_by_operation[1].calibration
    streams = rows[None, :, :]
    bank = calibration.bank
    state = bank.new_state(1, 2)
    group_scores = []
    for offset in range(rows.shape[0]):
        round_index = offset + 2
        round_rows = streams[:, offset]
        round_scores = bank.score_round(state, round_rows, round_index)
        group_scores.append(round_scores[0])
    return numpy.asarray(group_scores)


def _position_pairs(positions, pairs):
    """The bank's pairs of check indices as pairs of check positions."""
    position_pairs = set()
    for first, second in pairs:
        pair = frozenset([positions[first], positions[second]])
        position_pairs.add(pair)
    return position_pairs
