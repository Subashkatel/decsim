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

The declared-tick runs at the end of the file pin the mode's timing and
its two variants: every hop of one serial escalation, and the parallel
variant's cancel and take (Sec. III A, Step 1). Their fabric is
tests/escalation/declared_fabric.py, so every tick asserted there is
arithmetic over declared latencies.
"""

import math

import pytest

import decsim.config as decsim_config
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
from decsim.front.experiment import load_experiment
from tests.front.yaml_configs import (
    MINIMAL_CONFIG,
    measure_point_shot,
    strong_unit,
    write_config,
)

NEAR_THRESHOLD_P = 0.008


def switching_config(tmp_path, gap_threshold_db: float, rounds: int = 30):
    escalation = {"kind": "switching", "gap_threshold_db": gap_threshold_db}
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
    workload = {**MINIMAL_CONFIG["workload"], "rounds_per_shot": rounds}
    sweep_point = {
        "physical_error_probability": [NEAR_THRESHOLD_P],
        "distance": [3],
        "round_period_us": [1.0],
        "shots": 1,
    }
    strong_decoder = strong_unit("belief_matching")
    card = {
        "escalation": escalation,
        "workload": workload,
        **weak_unit,
        **strong_decoder,
        "sweep": [sweep_point],
    }
    return write_config(tmp_path, card)


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

    weak_only_card = {
        "escalation": {"kind": "switching", "gap_threshold_db": 20.0}
    }
    weak_only_path = write_config(tmp_path, weak_only_card)
    weak_only = load_experiment(weak_only_path)
    with pytest.raises(ValueError, match="escalates to the strong_decoder"):
        weak_only_settings = weak_only.point_settings(
            physical_error_probability=NEAR_THRESHOLD_P,
            distance=3,
            round_period_us=1.0,
        )
        Machine.build(weak_only_settings)
    strong_decoder = strong_unit("belief_matching")
    no_threshold_card = {"escalation": {"kind": "switching"}}
    no_threshold_card.update(strong_decoder)
    no_threshold_path = write_config(tmp_path, no_threshold_card)
    with pytest.raises(ValueError, match="needs gap_threshold_db"):
        load_experiment(no_threshold_path)
    weak_baseline_card = {
        "escalation": {
            "kind": "weak_baseline",
            "gap_threshold_db": 20.0,
        }
    }
    weak_baseline_path = write_config(tmp_path, weak_baseline_card)
    with pytest.raises(ValueError, match="never escalates"):
        load_experiment(weak_baseline_path)


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
    flag_card = _restart_width_card(True)
    flag_path = write_config(tmp_path, flag_card)
    with pytest.raises(ValueError, match="must be 0, the paper's restart"):
        load_experiment(flag_path)
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
    config_path = switching_config(tmp_path, 20.0)
    config = load_experiment(config_path)
    log_of_ten = math.log(10.0)
    expected_nats = 2.0 * log_of_ten
    assert config.settings.escalation.gap_threshold_decibels == 20.0
    assert math.isclose(
        config.settings.escalation.gap_threshold_nats, expected_nats
    )


def test_every_window_commits_once_across_both_output_links(tmp_path):
    """Every window commits exactly once, over one of the output links.

    Escalations ride WSD then SBD then DO; kept windows ride WDO.
    """
    config_path = switching_config(tmp_path, 20.0)
    config = load_experiment(config_path)
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
    config_path = switching_config(tmp_path, 0.0)
    config = load_experiment(config_path)
    measurement = measured_shot(config, seed=0)
    links = measurement.link_totals
    assert links["weak_decoder_to_strong_decoder"]["transfers"] == 0
    assert links["strong_decoder_to_frame"]["transfers"] == 0
    assert links["weak_decoder_to_frame"]["transfers"] == measurement.windows


def test_unreachable_threshold_escalates_every_window(tmp_path):
    config_path = switching_config(tmp_path, 1e6)
    config = load_experiment(config_path)
    measurement = measured_shot(config, seed=0)
    links = measurement.link_totals
    assert (
        links["weak_decoder_to_strong_decoder"]["transfers"]
        == measurement.windows
    )
    assert links["strong_decoder_to_frame"]["transfers"] == measurement.windows
    assert links["weak_decoder_to_frame"]["transfers"] == 0


def gaps_by_window(view, weak_tier) -> dict:
    """Every weak request's gap, keyed by the window it decoded."""
    gaps = {}
    for record in view.requests:
        if record.request_key.tier is not weak_tier:
            continue
        assert record.soft_output is not None
        window_key = (
            record.request_key.operation_id,
            record.request_key.window_id,
        )
        gaps[window_key] = record.soft_output.gap
    return gaps


def expected_tier_of(gap, threshold_nats, weak_tier, strong_tier):
    """The tier the escalation decision selects for one recorded gap."""
    if gap >= threshold_nats:
        return weak_tier
    return strong_tier


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

    config_path = switching_config(tmp_path, 20.0)
    config = load_experiment(config_path)
    threshold_nats = config.settings.escalation.gap_threshold_nats
    weak_tier = window_records.DecoderTier.WEAK
    strong_tier = window_records.DecoderTier.STRONG
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
        gap_by_window = gaps_by_window(view, weak_tier)
        for row in view.windows:
            gap = gap_by_window[row.destination_key]
            selected_tier = row.selected_request_key.tier
            expected_tier = expected_tier_of(
                gap, threshold_nats, weak_tier, strong_tier
            )
            assert selected_tier is expected_tier


# ---- the declared-tick timeline of the two variants


def test_the_serial_escalation_timeline_is_exact():
    """Every hop of one escalation, in the order the serial mode runs them.

    Toshio arXiv:2510.25222 Sec. III A, serial variant: the strong
    re-decode cannot begin before the weak verdict has crossed
    weak_decoder_to_strong_decoder and the room-side context has
    crossed strong_buffer_to_strong_decoder, so its start is the later
    of the two; and the Held boundary keeps the next window's decode
    parked until the strong correction has committed, one
    decoder_to_decoder hop away.
    """
    machine = fabric.switching_machine(rounds=9, escalated_windows={0, 1, 2})
    machine.run()
    log_lines = machine.observation.log.lines
    weak_done = declared_run.log_tick(log_lines, "DECODE DONE mem1 W0")
    strong_start = declared_run.log_tick(
        log_lines, "START DECODE strong(mem1 W0)"
    )
    parked_start = declared_run.log_tick(log_lines, "START DECODE mem1 W1")
    snapshot = machine.pauli_frame.snapshot()
    first_record = snapshot.records[0]
    expected_weak_done = decsim_config.microseconds_to_ticks(30.0)
    expected_strong_start = decsim_config.microseconds_to_ticks(36.0)
    expected_accepted = decsim_config.microseconds_to_ticks(70.0)
    expected_committed = decsim_config.microseconds_to_ticks(71.0)
    expected_parked_start = decsim_config.microseconds_to_ticks(71.5)

    assert weak_done == expected_weak_done
    assert strong_start == expected_strong_start
    assert first_record.window_key == (1, 0)
    assert first_record.tier == "strong"
    assert first_record.accepted_ticks == expected_accepted
    assert first_record.committed_ticks == expected_committed
    assert parked_start == expected_parked_start


def test_the_parallel_mode_refuses_a_room_side_lag_beyond_the_margin():
    """The parallel strong sibling starts at once or the run stops.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: in the parallel variant
    the strong decoder is handed the window's context the moment the
    weak decode starts. On the declared card the room-side hop is 7 us
    against Buffer 0's 4 us, so the context is not yet in syndrome
    buffer 1 when the window becomes ready, and the shape refuses
    loudly rather than decode a window whose context it cannot read
    (decsim/escalation/strong_window_shapes.py).
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        strong_buffer_microseconds=7.0,
    )
    with pytest.raises(
        RuntimeError,
        match="controller_to_strong_buffer lag beyond the escalation margin",
    ):
        machine.run()


def test_every_strong_request_is_cancelled_when_the_weak_tier_is_confident():
    """A confident weak result cancels its sibling and frees its context.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: the parallel strong
    decode is speculative, so a confident weak result cancels it and no
    window commits from the strong tier. The cancel also owns the
    room-side hold the request took, and syndrome buffer 1 would report
    an unresolved hold at the end of the run if it did not release it.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        strong_buffer_microseconds=2.0,
    )
    machine.run()
    counts = machine.decoder_manager.strong_requests.counts
    tiers = fabric.frame_tiers(machine)

    assert tiers == [((1, 0), "weak"), ((1, 1), "weak"), ((1, 2), "weak")]
    assert counts.cancelled == 3
    assert counts.needed == 0
    machine.strong_round_writer.check_settled()


def test_every_window_takes_the_strong_result_when_the_weak_tier_is_not():
    """The other edge of the parallel variant: nothing is cancelled.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: when every weak result
    falls below the threshold the sibling that already ran is the
    window's answer, so every request is needed and the frame carries
    the strong tier alone.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows={0, 1, 2},
        run_both_at_once=True,
        strong_buffer_microseconds=2.0,
    )
    machine.run()
    counts = machine.decoder_manager.strong_requests.counts
    tiers = fabric.frame_tiers(machine)

    assert tiers == [((1, 0), "strong"), ((1, 1), "strong"), ((1, 2), "strong")]
    assert counts.needed == 3
    assert counts.cancelled == 0


def first_decrease_tick(timeline):
    """The tick at which an occupancy timeline first falls."""
    previous = timeline[0][1]
    for tick, occupancy in timeline[1:]:
        if occupancy < previous:
            return tick
        previous = occupancy
    raise AssertionError("the occupancy timeline never falls")


def test_the_strong_context_lives_until_the_escalated_window_commits():
    """Syndrome buffer 1 frees a context after the commit, not the read.

    Toshio arXiv:2510.25222 Sec. III A: the strong decoder re-decodes
    the escalated window's whole context, and its result is the
    window's only Pauli-frame write, so the context has to survive
    until that write lands. The escalated window reads all six rounds
    of the operation, and its strong input transfer lands long before
    the commit; a store that freed the rounds at the transfer would
    show a fall before the committed tick.
    """
    machine = fabric.switching_machine(rounds=6, escalated_windows={0})
    probe = declared_run.OccupancyProbe(machine.window_manager)
    machine.engine.action_done.connect(probe.observe)
    machine.run()
    timeline = probe.strong_timeline
    occupancies = [occupancy for _, occupancy in timeline]
    peak = max(occupancies)
    first_fall = first_decrease_tick(timeline)
    snapshot = machine.pauli_frame.snapshot()
    strong_record = snapshot.records[0]
    expected_committed = decsim_config.microseconds_to_ticks(71.0)

    assert strong_record.window_key == (1, 0)
    assert strong_record.tier == "strong"
    assert strong_record.committed_ticks == expected_committed
    assert peak == 6
    assert first_fall > expected_committed
