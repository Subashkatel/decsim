"""The switching mode: weak decode, gap, conditional strong escalation.

Every window commits once. The contract under test is the paper's
protocol (Toshio 2510.25222 Sec. III A, serial variant): every window
decodes weak first and carries a gap; a gap at or above the threshold
keeps the weak result; below it, the window crosses WSD, the strong
decoder re-decodes the strong-window extent from syndrome buffer 1 over
SBD, and the strong result is the window's only Pauli-frame write,
riding DO home. The threshold's two
edges pin the plumbing: at 0 dB nothing escalates and the run is the
weak tier alone; at an unreachably high threshold everything escalates
and every window commits from the strong tier.
"""

import math

import pytest

from decsim.front.experiment import load_experiment
from tests.front.yaml_configs import (
    MINIMAL_CONFIG,
    measure_point_shot,
    strong_unit,
    write_config,
)

NEAR_THRESHOLD_P = 0.008


def switching_config(tmp_path, gap_threshold_db: float, rounds: int = 30):
    weak_unit = {
        "weak_decoder": {
            "kind": "pymatching",
            "units": 1,
            "unit_memory_rounds": None,
            "engine": {
                "clock": "fridge",
                "fetch_cycles_per_round": 1,
                "release_cycles_per_job": 1,
            },
        }
    }
    return write_config(
        tmp_path,
        {
            "escalation": {
                "kind": "switching",
                "gap_threshold_db": gap_threshold_db,
            },
            "workload": {
                **MINIMAL_CONFIG["workload"],
                "rounds_per_shot": rounds,
            },
            **weak_unit,
            **strong_unit("belief_matching"),
            "sweep": [
                {
                    "physical_error_probability": [NEAR_THRESHOLD_P],
                    "distance": [3],
                    "round_period_us": [1.0],
                    "shots": 1,
                }
            ],
        },
    )


def measured_shot(config, seed: int):
    return measure_point_shot(
        config,
        physical_error_probability=NEAR_THRESHOLD_P,
        distance=3,
        round_period_us=1.0,
        seed=seed,
    )


def test_switching_config_requires_both_tiers_and_the_card(tmp_path):
    from decsim.machine import Machine

    weak_only = load_experiment(
        write_config(
            tmp_path,
            {"escalation": {"kind": "switching", "gap_threshold_db": 20.0}},
        )
    )
    with pytest.raises(ValueError, match="escalates to the strong_decoder"):
        Machine.build(
            weak_only.point_settings(
                physical_error_probability=NEAR_THRESHOLD_P,
                distance=3,
                round_period_us=1.0,
            )
        )
    with pytest.raises(ValueError, match="needs gap_threshold_db"):
        load_experiment(
            write_config(
                tmp_path,
                {
                    "escalation": {"kind": "switching"},
                    **strong_unit("belief_matching"),
                },
            )
        )
    with pytest.raises(ValueError, match="never escalates"):
        load_experiment(
            write_config(
                tmp_path,
                {
                    "escalation": {
                        "kind": "weak_baseline",
                        "gap_threshold_db": 20.0,
                    }
                },
            )
        )


def _restart_width_card(regions: int) -> dict:
    """The switching card with the double window and the re-read width."""
    escalation = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "double_window": True,
        "restart_reread_buffer_regions": regions,
    }
    strong_decoder = strong_unit("belief_matching")
    card = {"escalation": escalation}
    card.update(strong_decoder)
    return card


def test_both_restart_re_read_widths_load_from_the_escalation_section(
    tmp_path,
):
    """0 is Toshio 2510.25222 Sec. III C, 1 is decsim's forward window."""
    paper_card = _restart_width_card(0)
    paper_path = write_config(tmp_path, paper_card)
    paper = load_experiment(paper_path)
    one_region_card = _restart_width_card(1)
    one_region_path = write_config(tmp_path, one_region_card)
    one_region = load_experiment(one_region_path)

    assert paper.settings.escalation.restart_reread_buffer_regions == 0
    assert one_region.settings.escalation.restart_reread_buffer_regions == 1


def test_a_wider_restart_re_read_and_another_kind_are_refused(tmp_path):
    """Only the two widths have a referent, and only switching restarts."""
    wide_card = _restart_width_card(2)
    wide_path = write_config(tmp_path, wide_card)
    with pytest.raises(ValueError, match="must be 0, the paper's restart"):
        load_experiment(wide_path)
    weak_card = {
        "escalation": {
            "kind": "weak_baseline",
            "restart_reread_buffer_regions": 0,
        }
    }
    weak_path = write_config(tmp_path, weak_card)
    with pytest.raises(ValueError, match="never escalates"):
        load_experiment(weak_path)


def test_threshold_converts_decibels_to_natural_log_weight(tmp_path):
    config = load_experiment(switching_config(tmp_path, 20.0))
    assert config.settings.escalation.gap_threshold_decibels == 20.0
    assert math.isclose(
        config.settings.escalation.gap_threshold_nats, 2.0 * math.log(10.0)
    )


def test_every_window_commits_once_across_both_output_links(tmp_path):
    """Every window commits exactly once, over one of the output links.

    Escalations ride WSD then SBD then DO; kept windows ride WDO.
    """
    config = load_experiment(switching_config(tmp_path, 20.0))
    found_escalation = False
    for seed in range(6):
        measurement = measured_shot(config, seed)
        links = measurement.link_totals
        escalations = links["weak_decoder_to_strong_decoder"]["transfers"]
        assert (
            links["strong_buffer_to_strong_decoder"]["transfers"] == escalations
        )
        assert links["strong_decoder_to_frame"]["transfers"] == escalations
        assert (
            links["weak_decoder_to_frame"]["transfers"] + escalations
            == measurement.windows
        )
        found_escalation = found_escalation or escalations > 0
    assert found_escalation, (
        "no window escalated in 6 near-threshold "
        "shots; raise p or the threshold"
    )


def test_zero_threshold_never_escalates(tmp_path):
    config = load_experiment(switching_config(tmp_path, 0.0))
    measurement = measured_shot(config, seed=0)
    links = measurement.link_totals
    assert links["weak_decoder_to_strong_decoder"]["transfers"] == 0
    assert links["strong_decoder_to_frame"]["transfers"] == 0
    assert links["weak_decoder_to_frame"]["transfers"] == measurement.windows


def test_unreachable_threshold_escalates_every_window(tmp_path):
    config = load_experiment(switching_config(tmp_path, 1e6))
    measurement = measured_shot(config, seed=0)
    links = measurement.link_totals
    assert (
        links["weak_decoder_to_strong_decoder"]["transfers"]
        == measurement.windows
    )
    assert links["strong_decoder_to_frame"]["transfers"] == measurement.windows
    assert links["weak_decoder_to_frame"]["transfers"] == 0


def test_gap_records_decide_the_selected_tier(tmp_path):
    """Every recorded gap sits on the escalation decision's dividing line.

    Below the threshold the window's committed result is the strong
    tier's, at or above it the weak tier's. (Serial escalation
    replaces the prediction in place, so the window stays an
    ordinary_window either way; the selected request key names the tier
    that produced the committed result.)
    """
    from dataclasses import replace

    import decsim.records.windows as window_records
    from decsim.machine import Machine
    from decsim.observe.run_views import switching_records_view

    config = load_experiment(switching_config(tmp_path, 20.0))
    threshold_nats = config.settings.escalation.gap_threshold_nats
    for seed in range(4):
        settings = config.point_settings(
            physical_error_probability=NEAR_THRESHOLD_P,
            distance=3,
            round_period_us=1.0,
        )
        observation = replace(
            settings.observation, record_switching_windows=True
        )
        settings = replace(settings, observation=observation)
        completed = Machine.build(settings, seed)
        completed.run()
        view = switching_records_view(
            completed.observation.windows, completed.observation.decode_records
        )
        gap_by_window = {}
        for record in view.requests:
            if record.request_key.tier is not window_records.DecoderTier.WEAK:
                continue
            assert record.soft_output is not None
            window_key = (
                record.request_key.operation_id,
                record.request_key.window_id,
            )
            gap_by_window[window_key] = record.soft_output.gap
        for row in view.windows:
            gap = gap_by_window[row.destination_key]
            selected_tier = row.selected_request_key.tier
            if gap >= threshold_nats:
                assert selected_tier is window_records.DecoderTier.WEAK
            else:
                assert selected_tier is window_records.DecoderTier.STRONG
