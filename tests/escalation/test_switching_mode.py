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

import json
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
    with pytest.raises(ValueError, match="decides on no confidence"):
        load_experiment(weak_baseline_path)


def _restart_width_card(regions: int) -> dict:
    """The switching card with the forward window and the re-read width."""
    escalation = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "forward",
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
    with pytest.raises(ValueError, match="decides on no confidence"):
        load_experiment(weak_path)


def _parallel_variant_card(run_both_at_once) -> dict:
    """The switching card with Sec. III A's Step 1 asked for, or not.

    Step 1 builds the strong job at weak readiness, so the room-side
    write must not lag the Buffer 0 publication path; the card wires the
    two store hops at one cycle each. On the reference numbers Buffer 1
    is 0.26 us behind Buffer 0 at 0.04 us and the build refuses.
    """
    escalation = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "run_both_at_once": run_both_at_once,
    }
    one_fridge_cycle = {
        "latency_cycles": 1,
        "clock": "fridge",
        "bits_per_cycle": None,
    }
    links = {
        "qpu_to_controller": one_fridge_cycle,
        "controller_to_weak_buffer": one_fridge_cycle,
        "controller_to_strong_buffer": one_fridge_cycle,
    }
    strong_decoder = strong_unit("belief_matching")
    card = {"escalation": escalation, "links": links}
    card.update(strong_decoder)
    return card


def _strong_request_counts(tmp_path, card: dict):
    """One shot of the switching card, and its strong-request counts."""
    from decsim.machine import Machine

    tmp_path.mkdir()
    config_path = write_config(tmp_path, card)
    config = load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=NEAR_THRESHOLD_P,
        distance=3,
        round_period_us=1.0,
    )
    machine = Machine.build(settings, 0)
    machine.run()
    return machine.decoder_manager.strong_requests.counts


def test_the_yaml_asks_for_the_papers_parallel_variant(tmp_path):
    """Toshio 2510.25222 Sec. III A: Step 1 runs both decoders at once.

    The default is the same section's on-demand variant (lines 631-640),
    where a confident window makes no strong request at all; under
    run_both_at_once every window's strong sibling starts and a
    confident window cancels it.
    """
    on_demand_card = _parallel_variant_card(False)
    parallel_card = _parallel_variant_card(True)
    on_demand_directory = tmp_path / "on_demand"
    parallel_directory = tmp_path / "parallel"
    on_demand = _strong_request_counts(on_demand_directory, on_demand_card)
    parallel = _strong_request_counts(parallel_directory, parallel_card)

    assert on_demand.cancelled == 0
    assert parallel.cancelled > 0


def test_a_run_both_at_once_that_is_not_a_flag_is_refused(tmp_path):
    card = _parallel_variant_card("yes")
    config_path = write_config(tmp_path, card)
    sentence = "escalation.run_both_at_once must be true or false, got 'yes'"
    with pytest.raises(ValueError, match=sentence):
        load_experiment(config_path)


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
    """Every window's gap, keyed by the window it decoded.

    A window's two forced-class requests are one attempt: the request
    that carries the window's answer carries the gap, its companion
    carries none.
    """
    gaps = {}
    for record in view.requests:
        if record.request_key.tier is not weak_tier:
            continue
        if record.soft_output is None:
            continue
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


def test_the_parallel_sibling_waits_for_the_context_it_reads():
    """The parallel strong sibling starts when its copy has landed.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: "a sequence of syndrome
    data sigma is simultaneously fed to both the weak and strong
    decoders" (lines 599-603). The paper prices no transport in Sec.
    III A and prices T_comm^strong at ten times T_comm^weak in Table I
    (lines 1943-1950), so in a model that prices transport Step 1 means
    the strong decoder starts when its copy has arrived. On the
    declared card the room-side hop is 7 us against Buffer 0's 4 us, so
    the first window's context is still crossing when the window
    becomes ready: the sibling is held, and it is submitted at the tick
    its last context round is stored in syndrome buffer 1.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        strong_buffer_microseconds=7.0,
    )
    machine.run()
    log_lines = machine.observation.log.lines
    held = declared_run.log_tick(
        log_lines, "strong(mem1 W0): strong start deferred"
    )
    last_context_round = declared_run.log_tick(
        log_lines, "SyndromeBuffer1: received round 6 of op 1"
    )
    submitted = declared_run.log_tick(
        log_lines,
        "strong(mem1 W0): strong context stored in syndrome buffer 1",
    )
    started = declared_run.log_tick(log_lines, "START DECODE strong(mem1 W0)")
    deferred_lines = fabric.log_lines_containing(
        machine, "strong start deferred until the context rounds"
    )
    assert "[(1, 4), (1, 5), (1, 6)]" in deferred_lines[0]
    assert held < submitted
    assert submitted == last_context_round
    assert started > submitted
    assert not machine.window_manager.strong_redecode.has_pending()


def test_a_deferred_strong_job_is_traced_from_its_hold_to_its_release(
    tmp_path,
):
    """The wait is a visible state on the strong tier's own lane.

    Note 14 section 3.4: a slice that opens when the row holds the job
    and closes when the condition fires, labelled with the strong
    request key, carrying the name of what it waits on, and stepping the
    window's flow into the job's dispatch. gem5 exposes a blocked port
    the same way, as state rather than as an exception
    (src/mem/port.hh:244-255).
    """
    trace_path = tmp_path / "parallel.trace.json"
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        strong_buffer_microseconds=7.0,
        trace_path=trace_path,
    )
    machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    text = trace_path.read_text()
    document = json.loads(text)
    lanes = _lane_names(document)
    held = _events_named(document, "X", "W0 strong window held")
    (slice_row,) = held
    args = slice_row["args"]
    expected_held = decsim_config.microseconds_to_ticks(15.0)
    expected_released = decsim_config.microseconds_to_ticks(18.0)

    assert lanes[slice_row["tid"]] == "Strong tier"
    assert args["request"] == "1:0:strong:1"
    assert args["waits_for"] == "stored rounds of operation 1"
    assert args["outcome"] == "strong context stored in syndrome buffer 1"
    assert args["rounds"] == 6
    assert args["tick"] == expected_held
    assert slice_row["dur"] == pytest.approx(3.0)

    arrows = _flows_of(document, lanes, "Strong tier", "window 1:0")
    (arrow,) = arrows
    assert arrow["args"]["tick"] == expected_released


def _lane_names(document) -> dict:
    """The name of every lane of the trace, by its thread id."""
    names = {}
    for row in document:
        if row.get("name") == "thread_name":
            names[row["tid"]] = row["args"]["name"]
    return names


def _events_named(document, phase: str, name: str) -> list:
    """Every event of one phase with one name."""
    rows = []
    for row in document:
        if row["ph"] == phase and row["name"] == name:
            rows.append(row)
    return rows


def _flows_of(document, lanes: dict, lane: str, flow_id: str) -> list:
    """Every flow event of one chain on one lane."""
    rows = []
    for row in document:
        if row["ph"] not in ("s", "t", "f"):
            continue
        if lanes[row["tid"]] != lane or row["id"] != flow_id:
            continue
        rows.append(row)
    return rows


def test_a_sibling_held_for_its_input_is_cancelled_by_a_confident_result():
    """Step 3 halts the strong computation a confident weak result spares.

    Toshio arXiv:2510.25222 Sec. III A, step 3 (lines 610-615). A
    sibling still waiting for its context has not started, so the
    confident weak result ends it where it waits; the room-side rounds
    it would have read are freed with the window's final commit, and
    the run settles with nothing held.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        strong_buffer_microseconds=40.0,
    )
    machine.run()
    cancelled = fabric.log_lines_containing(
        machine, "cancelled while held for its input"
    )
    assert len(cancelled) == 3
    assert not machine.window_manager.strong_redecode.has_pending()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
    ]


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
    show a fall before the committed tick. The declared fabric gives
    the release its exact tick as well, 84.5 us here: 71.0 for the
    commit and the declared hops back to the store after it.
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
    expected_released = decsim_config.microseconds_to_ticks(84.5)

    assert strong_record.window_key == (1, 0)
    assert strong_record.tier == "strong"
    assert strong_record.committed_ticks == expected_committed
    assert peak == 6
    assert first_fall > expected_committed
    assert first_fall == expected_released


def _walk_card(microseconds, weak_kind: str, confidence: str) -> dict:
    """The switching card that prices the confidence signal's computation."""
    escalation = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "confidence": confidence,
        "confidence_walk_microseconds": microseconds,
    }
    weak_decoder = {
        "weak_decoder": {
            "kind": weak_kind,
            "units": 1,
            "unit_memory_rounds": None,
            "engine": {
                "clock": "fridge",
                "fetch_cycles_per_round": 1,
                "release_cycles_per_job": 1,
            },
        }
    }
    observation = {"observation": {"record_switching_windows": True}}
    card = {"escalation": escalation}
    card.update(weak_decoder)
    card.update(observation)
    strong_decoder = strong_unit("belief_matching")
    card.update(strong_decoder)
    return card


def _confidence_charges(machine) -> list:
    """The ticks each CONFIDENCE line of the run charged, in order."""
    charged = []
    for line in machine.observation.log.lines:
        if "CONFIDENCE" not in line:
            continue
        parts = line.split(": ")
        charge_text = parts[-1]
        ticks_text = charge_text.replace(" ticks", "")
        charged.append(int(ticks_text))
    return charged


def _weak_services(machine) -> list:
    """The decode services of the weak pool, in the order they ended."""
    weak = []
    for service in machine.observation.decode_records.services:
        if service.pool == "default":
            weak.append(service)
    return weak


def _walk_card_machine(tmp_path, microseconds):
    """One d=3 shot of the walk card, built and run through the yaml."""
    from decsim.machine import Machine

    card = _walk_card(microseconds, "union_find", "cluster_gap")
    config_path = write_config(tmp_path, card)
    config = load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=NEAR_THRESHOLD_P,
        distance=3,
        round_period_us=1.0,
    )
    machine = Machine.build(settings, 0)
    machine.run()
    return machine


def test_a_priced_confidence_walk_charges_its_card_once_per_window(tmp_path):
    """escalation.confidence_walk_microseconds is a card, like a tier's.

    The cluster gap's walk is a Dijkstra over the decode's own edge
    intervals (Meister et al. 2405.07433 Algorithm 2), charged on the
    unit that grew them (decision D8). Left null it is measured on the
    host clock; given a number it is that number of microseconds per
    window, so a yaml experiment can price it the way a decoder tier is
    priced.
    """
    machine = _walk_card_machine(tmp_path, 12.0)
    expected_ticks = decsim_config.microseconds_to_ticks(12.0)
    charged = _confidence_charges(machine)
    weak_services = _weak_services(machine)
    assert charged
    assert len(charged) == len(weak_services)
    for ticks in charged:
        assert ticks == expected_ticks


def test_a_negative_confidence_walk_is_refused_by_name(tmp_path):
    card = _walk_card(-1.0, "union_find", "cluster_gap")
    config_path = write_config(tmp_path, card)
    with pytest.raises(
        ValueError,
        match="escalation.confidence_walk_microseconds must be finite",
    ):
        load_experiment(config_path)


def test_a_confidence_walk_that_is_not_a_number_is_refused_by_name(tmp_path):
    card = _walk_card("fast", "union_find", "cluster_gap")
    config_path = write_config(tmp_path, card)
    with pytest.raises(
        ValueError,
        match="escalation.confidence_walk_microseconds must be a number",
    ):
        load_experiment(config_path)
