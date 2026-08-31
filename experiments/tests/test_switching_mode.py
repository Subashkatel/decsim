"""The switching mode: weak decode + complementary gap, conditional
strong escalation, one commit per window.

The contract under test is the paper's protocol (Toshio 2510.25222
Sec. III A, serial variant): every window decodes weak first and
carries a gap; a gap at or above the threshold keeps the weak result;
below it, the window crosses WSD, the strong decoder re-decodes the
strong-window extent from syndrome buffer 1 over SBD, and the strong result is the
window's only Pauli-frame write, riding DO home. The threshold's two
edges pin the plumbing: at 0 dB nothing escalates and the run is the
weak tier alone; at an unreachably high threshold everything escalates
and every window commits from the strong tier.
"""

import math

from experiments.experiment_config import load_experiment
from experiments.measure_shot import measure_shot

import pytest

from test_decoder_units import MINIMAL_CONFIG, strong_unit, write_config

NEAR_THRESHOLD_P = 0.008


def switching_config(tmp_path, gap_threshold_db: float,
                     rounds: int = 30):
    weak_unit = {"weak": {"algorithm": "pymatching", "units": 1,
                          "unit_buffer_size": None,
                          "engine": {"clock": "fridge",
                                     "fetch_cycles_per_round": 1,
                                     "release_cycles_per_job": 1}}}
    return write_config(tmp_path, {
        "mode": "switching",
        "rounds_per_shot": rounds,
        "switching": {"gap_threshold_db": gap_threshold_db},
        "decoder": {**weak_unit, **strong_unit("belief_matching")},
        "sweep": [{"physical_error_probability": [NEAR_THRESHOLD_P],
                   "distance": [3], "round_period_us": [1.0], "shots": 1}]})


def measured_shot(config, seed: int):
    return measure_shot(config, physical_error_probability=NEAR_THRESHOLD_P,
                        distance=3, round_period_us=1.0, seed=seed)


def test_switching_config_requires_both_tiers_and_the_card(tmp_path):
    with pytest.raises(ValueError, match="decoder.strong"):
        load_experiment(write_config(tmp_path, {
            "mode": "switching",
            "switching": {"gap_threshold_db": 20.0},
            "decoder": {"weak": MINIMAL_CONFIG["decoder"]["weak"]}}))
    with pytest.raises(ValueError, match="switching card"):
        load_experiment(write_config(tmp_path, {
            "mode": "switching",
            "decoder": {**MINIMAL_CONFIG["decoder"],
                        **strong_unit("belief_matching")}}))
    with pytest.raises(ValueError, match="never escalates"):
        load_experiment(write_config(tmp_path, {
            "switching": {"gap_threshold_db": 20.0}}))


def test_threshold_converts_decibels_to_natural_log_weight(tmp_path):
    config = load_experiment(switching_config(tmp_path, 20.0))
    assert config.switching.gap_threshold_db == 20.0
    assert math.isclose(config.switching.gap_threshold_nats,
                        2.0 * math.log(10.0))


def test_every_window_commits_once_across_both_output_links(tmp_path):
    """Escalations ride WSD then SBD then DO; kept windows ride WDO; the
    two output links together commit every window exactly once."""
    config = load_experiment(switching_config(tmp_path, 20.0))
    found_escalation = False
    for seed in range(6):
        measurement = measured_shot(config, seed)
        links = measurement.link_totals
        escalations = links["wsd"]["transfers"]
        assert links["sbd"]["transfers"] == escalations
        assert links["do"]["transfers"] == escalations
        assert (links["wdo"]["transfers"] + escalations
                == measurement.windows)
        found_escalation = found_escalation or escalations > 0
    assert found_escalation, ("no window escalated in 6 near-threshold "
                              "shots; raise p or the threshold")


def test_zero_threshold_never_escalates(tmp_path):
    config = load_experiment(switching_config(tmp_path, 0.0))
    measurement = measured_shot(config, seed=0)
    links = measurement.link_totals
    assert links["wsd"]["transfers"] == 0
    assert links["do"]["transfers"] == 0
    assert links["wdo"]["transfers"] == measurement.windows


def test_unreachable_threshold_escalates_every_window(tmp_path):
    config = load_experiment(switching_config(tmp_path, 1e6))
    measurement = measured_shot(config, seed=0)
    links = measurement.link_totals
    assert links["wsd"]["transfers"] == measurement.windows
    assert links["do"]["transfers"] == measurement.windows
    assert links["wdo"]["transfers"] == 0


def test_gap_records_decide_the_selected_tier(tmp_path):
    """Every weak decode's recorded gap sits on the escalation decision's
    dividing line: below the threshold the window's committed result is
    the strong tier's, at or above it the weak tier's. (Serial escalation
    replaces the prediction in place, so the window stays an
    ordinary_window either way; the selected request key names the tier
    that produced the committed result.)"""
    from decsim.message import DecoderTier
    from decsim.observe.run_views import switching_records_view
    from experiments.build_run import build_run

    config = load_experiment(switching_config(tmp_path, 20.0))
    threshold_nats = config.switching.gap_threshold_nats
    for seed in range(4):
        spec, _ = build_run(config,
                            physical_error_probability=NEAR_THRESHOLD_P,
                            distance=3, round_period_us=1.0, seed=seed)
        spec.record_switching_windows = True
        completed = spec.build()
        view = switching_records_view(completed.window_manager,
                                      completed.decoder_manager)
        gap_by_window = {}
        for record in view.requests:
            if record.request_key.tier is not DecoderTier.WEAK:
                continue
            assert record.soft_output is not None
            window_key = (record.request_key.operation_id,
                          record.request_key.window_id)
            gap_by_window[window_key] = record.soft_output.gap
        for row in view.windows:
            gap = gap_by_window[row.destination_key]
            selected_tier = row.selected_request_key.tier
            if gap >= threshold_nats:
                assert selected_tier is DecoderTier.WEAK
            else:
                assert selected_tier is DecoderTier.STRONG
