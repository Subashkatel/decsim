"""The switching mode: weak decode, gap, conditional strong escalation.

Every window commits once. The contract under test is the paper's
protocol (Toshio 2510.25222 Sec. III A, serial variant): every window
decodes weak first and carries a gap; a gap at or above the threshold
keeps the weak result; below it, the window crosses WSD, the strong
decoder re-decodes the strong-window extent from the strong syndrome buffer over
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

import dataclasses
import json

import pytest

import decsim.confidence.cluster as cluster
import decsim.config as decsim_config
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.threshold_sources as threshold_sources
import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric
import tests.escalation.test_strong_window_shapes as shape_tests


def switching_settings(threshold_decibels: float, rounds_per_shot=None):
    """The gate's switching card at d=3, p 0.008, on its own threshold."""
    settings = shape_tests.gate_switching(rounds_per_shot=rounds_per_shot)
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=threshold_decibels
    )
    switching = dataclasses.replace(settings.switching, threshold=threshold)
    return dataclasses.replace(settings, switching=switching)


def run_shot(settings, seed: int):
    """One shot of the settings: its machine, run, and its result."""
    machine = machine_module.Machine.build(settings, seed)
    result = machine.run()
    return machine, result


def transfers_by_path(result) -> dict:
    """The run's transfer count on each path of the card."""
    transfers = {}
    for edge in result.link_traffic["semantic_edges"]:
        path = edge["path"]
        count = edge["counters"]["transfer_count"]
        transfers[path] = transfers.get(path, 0) + count
    return transfers


def test_every_window_commits_once_across_both_output_links_property():
    """Every window commits exactly once, over one of the output links.

    Escalations ride WSD then SBD then DO; kept windows ride WDO. WSD
    carries one selection per escalation and at most one region after
    it, so its transfers lie between one and two per escalation.
    """
    settings = switching_settings(20.0)
    found_escalation = False
    for seed in range(6):
        machine, result = run_shot(settings, seed)
        transfers = transfers_by_path(result)
        windows = len(machine.observation.windows.windows)
        escalations = transfers.get("strong_buffer_to_strong_decoder", 0)
        escalation_hops = transfers.get("weak_decoder_to_strong_decoder", 0)
        assert escalations <= escalation_hops <= 2 * escalations
        assert transfers.get("strong_decoder_to_frame", 0) == escalations
        kept = transfers["weak_decoder_to_frame"]
        assert kept + escalations == windows
        found_escalation = found_escalation or escalations > 0
    assert found_escalation, (
        "no window escalated in 6 near-threshold "
        "shots; raise p or the threshold"
    )


def through_weak_chip(settings):
    """The settings with every strong answer joined on the weak chip."""
    switching = dataclasses.replace(
        settings.switching, strong_answer_route="through_weak_chip"
    )
    return dataclasses.replace(settings, switching=switching)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_through_the_weak_chip_every_answer_reaches_the_frame_on_its_hop(
    seed,
):
    """A strong answer goes down to the chip, and the chip sends it home.

    Every window, kept or escalated, reaches the frame on
    weak_decoder_to_frame, and nothing rides strong_decoder_to_frame.
    """
    direct = switching_settings(20.0)
    settings = through_weak_chip(direct)
    machine, result = run_shot(settings, seed)
    transfers = transfers_by_path(result)
    windows = len(machine.observation.windows.windows)
    escalations = transfers["strong_buffer_to_strong_decoder"]
    assert escalations > 0
    assert transfers.get("strong_decoder_to_frame", 0) == 0
    assert transfers["strong_decoder_to_weak_decoder"] == escalations
    assert transfers["weak_decoder_to_frame"] == windows


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_both_strong_answer_routes_predict_the_same_shot_for_shot(seed):
    """The route moves where the join is priced, never what is joined."""
    direct = switching_settings(20.0)
    through = through_weak_chip(direct)
    _machine, direct_result = run_shot(direct, seed)
    _twin, through_result = run_shot(through, seed)
    assert predictions(through_result) == predictions(direct_result)
    assert failures(through_result) == failures(direct_result)


def failures(result) -> list:
    """Each operation's logical failure against the sampled truth."""
    failed = []
    for operation_result in result.operation_results:
        failed.append(operation_result.logical_failure)
    return failed


def never_escalating_settings():
    """The gate's card at 0 dB, its weak tier clocked and its gap free.

    Union-find on its cycle count prices every weak decode in whole
    cycles of the fridge clock, and the cluster gap walked in no time
    adds nothing to it, so a window costs what it costs the weak tier
    alone.
    """
    settings = switching_settings(0.0)
    cycle_count_law = cycle_count.CycleCount(clock=shape_tests.FRIDGE_CLOCK)
    union_find_decoder = union_find.UnionFindDecoder.Settings(
        timing=cycle_count_law
    )
    weak_decoder = dataclasses.replace(
        settings.weak_decoder, algorithm=union_find_decoder
    )
    cluster_gap = cluster.ClusterGap.Settings(walk_microseconds=0.0)
    switching = dataclasses.replace(settings.switching, confidence=cluster_gap)
    return dataclasses.replace(
        settings, weak_decoder=weak_decoder, switching=switching
    )


def predictions(result) -> list:
    """Each operation's predicted observables."""
    predicted = []
    for operation_result in result.operation_results:
        predicted.append(operation_result.logical_observables)
    return predicted


def weak_decode_stages(machine) -> list:
    """Every decode stage the run timed: window, stage, unit and ticks."""
    stages = []
    for record in machine.observation.stages.records:
        window_key = (record.operation_id, record.window_id)
        stages.append(
            (
                window_key,
                record.stage,
                record.unit_name,
                record.start_ticks,
                record.end_ticks,
            )
        )
    return stages


def test_a_run_that_never_escalates_is_the_weak_tier_alone_shot_for_shot():
    """At 0 dB nothing escalates and the run is weak-only, tick for tick.

    No gap is below 0 dB, so every window keeps its weak result (Toshio
    2510.25222 Sec. III A). The weak-only twin is the same machine with no
    switching slot and no strong tier, on the same windows, so both plan
    one set of windows; the switching run makes its predictions and times
    every weak decode stage exactly as the twin does.
    """
    settings = never_escalating_settings()
    weak_only = dataclasses.replace(
        settings, switching=None, strong_decoder=None
    )

    machine, result = run_shot(settings, 0)
    twin, twin_result = run_shot(weak_only, 0)

    transfers = transfers_by_path(result)
    windows = len(machine.observation.windows.windows)
    assert transfers.get("weak_decoder_to_strong_decoder", 0) == 0
    assert transfers.get("strong_decoder_to_frame", 0) == 0
    assert transfers["weak_decoder_to_frame"] == windows
    assert predictions(result) == predictions(twin_result)
    assert weak_decode_stages(machine) == weak_decode_stages(twin)


def test_unreachable_threshold_escalates_every_window():
    settings = switching_settings(1e6)
    machine, result = run_shot(settings, 0)
    transfers = transfers_by_path(result)
    windows = len(machine.observation.windows.windows)
    escalation_hops = transfers["weak_decoder_to_strong_decoder"]
    assert windows <= escalation_hops <= 2 * windows
    assert transfers["strong_buffer_to_strong_decoder"] == windows
    assert transfers["strong_decoder_to_frame"] == windows
    assert transfers.get("weak_decoder_to_frame", 0) == 0


def gaps_by_window(requests, weak_tier) -> dict:
    """Every window's gap, keyed by the window it decoded.

    A window's two forced-class requests are one attempt: the request
    that carries the window's answer carries the gap, its companion
    carries none.
    """
    gaps = {}
    for record in requests:
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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gap_records_decide_the_selected_tier(seed):
    """Every recorded gap sits on the escalation decision's dividing line.

    Below the threshold the window's committed result is the strong
    tier's, at or above it the weak tier's. (Serial escalation
    replaces the prediction in place, so the window stays an
    ordinary_window either way; the selected request key names the tier
    that produced the committed result.)
    """
    settings = switching_settings(20.0)
    threshold_nats = settings.switching.threshold.threshold_nats
    observation = dataclasses.replace(
        settings.observation, record_switching_windows=True
    )
    settings = dataclasses.replace(settings, observation=observation)
    weak_tier = window_records.DecoderTier.WEAK
    strong_tier = window_records.DecoderTier.STRONG
    completed, _result = run_shot(settings, seed)
    requests = completed.observation.decode_records.requests
    gap_by_window = gaps_by_window(requests, weak_tier)
    windows = completed.observation.windows.windows
    selected_tiers = {
        key: window.published_request_key.tier
        for key, window in windows.items()
    }
    expected_tiers = {
        key: expected_tier_of(
            gap_by_window[key], threshold_nats, weak_tier, strong_tier
        )
        for key in windows
    }
    assert selected_tiers == expected_tiers


# ---- the declared-tick timeline of the two variants


def test_the_serial_escalation_timeline_is_exact():
    """Every hop of one escalation, in the order the serial mode runs them.

    Toshio arXiv:2510.25222 Sec. III A, serial variant, with the
    region assigned at the switch (Sec. III C, lines 1247 to 1250): the
    weak verdict at 30 us sends the selection and the window's six
    rounds over weak_decoder_to_strong_decoder, 3 us; they land in the
    strong syndrome buffer at 33 us, the job is built and its input
    crosses strong_buffer_to_strong_decoder, 6 us, so the strong decode
    starts at 39 us; and the Held boundary keeps the next window's
    decode parked until the strong correction has committed, one
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
    snapshot = machine.control.pauli_frame.snapshot()
    first_record = snapshot.records[0]
    expected_weak_done = decsim_config.microseconds_to_ticks(30.0)
    expected_strong_start = decsim_config.microseconds_to_ticks(39.0)
    expected_accepted = decsim_config.microseconds_to_ticks(73.0)
    expected_committed = decsim_config.microseconds_to_ticks(74.0)
    expected_parked_start = decsim_config.microseconds_to_ticks(74.5)

    assert weak_done == expected_weak_done
    assert strong_start == expected_strong_start
    assert first_record.window_key == (1, 0)
    assert first_record.tier == "strong"
    assert first_record.accepted_ticks == expected_accepted
    assert first_record.committed_ticks == expected_committed
    assert parked_start == expected_parked_start


def test_through_the_weak_chip_the_frame_waits_a_hop_and_a_chip_cycle_more():
    """The weak chip's join is one cycle, then the weak tier's own hop home.

    Both strong answer hops are declared 4 us, so the answer lands on the
    chip when it would land on the frame; the commit edge, one 0.5 us
    cycle of the declared clock (Yang et al. 2605.04892 Table I prices
    the frame update as one cycle), and weak_decoder_to_frame, 2 us, are
    what the route adds.
    """
    direct = fabric.switching_machine(rounds=9, escalated_windows={0, 1, 2})
    switching = declared_run.declared_switching(
        strong_answer_route="through_weak_chip", clock=fabric.DECLARED_CLOCK
    )
    through = fabric.switching_machine(
        rounds=9, escalated_windows={0, 1, 2}, switching=switching
    )
    direct.run()
    through.run()
    direct_snapshot = direct.control.pauli_frame.snapshot()
    through_snapshot = through.control.pauli_frame.snapshot()
    direct_record = direct_snapshot.records[0]
    through_record = through_snapshot.records[0]
    extra_ticks = through_record.accepted_ticks - direct_record.accepted_ticks
    weak_leg_ticks = decsim_config.microseconds_to_ticks(
        fabric.DECLARED_MICROSECONDS["weak_decoder_to_frame"]
    )
    commit_edge_ticks = fabric.DECLARED_CLOCK.period_ticks

    assert through_record.window_key == direct_record.window_key
    assert through_record.tier == "strong"
    assert extra_ticks == weak_leg_ticks + commit_edge_ticks


def test_the_speculative_decode_waits_for_the_context_it_reads():
    """The speculative strong decode starts when its copy has landed.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: "a sequence of syndrome data
    sigma is simultaneously fed to both the weak and strong decoders" (lines
    598-601). The paper prices no transport in Sec. III A, and its simulations
    set T_comm^strong to ten times T_comm^weak (lines 1109-1114; Table I is a
    notation table and prices nothing), so in a model that prices transport Step
    1 means the strong decoder starts when its copy has arrived. The copy rides
    weak_decoder_to_strong_decoder at readiness, 3 us on the declared card, so
    the first window's six context rounds are all still on the chip when the
    window becomes ready: the speculative decode is held, and it is submitted at
    the tick its last context round is stored in the strong syndrome buffer.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
    )
    machine.run()
    log_lines = machine.observation.log.lines
    held = declared_run.log_tick(
        log_lines, "strong(mem1 W0): strong start deferred"
    )
    last_context_round = declared_run.log_tick(
        log_lines, "strong syndrome buffer: received round 6 of op 1"
    )
    submitted = declared_run.log_tick(
        log_lines,
        "strong(mem1 W0): strong context stored in strong syndrome buffer",
    )
    started = declared_run.log_tick(log_lines, "START DECODE strong(mem1 W0)")
    deferred_lines = fabric.log_lines_containing(
        machine, "strong start deferred until the context rounds"
    )
    assert (
        "[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]"
        in (deferred_lines[0])
    )
    assert held < submitted
    assert submitted == last_context_round
    assert started > submitted
    assert not machine.windows.window_manager.strong_redecode.has_pending()


def test_a_deferred_strong_job_is_traced_from_its_hold_to_its_release(
    tmp_path,
):
    """The wait is a visible state on the strong tier's own lane.

    It is a slice that opens when the row holds the job
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
    # the hold opens when the last context round leaves the QPU, 6 rounds
    # plus qpu_to_controller 2 plus readout_to_bits 3, and ends when
    # the escalated region lands over weak_decoder_to_strong_decoder, 3
    expected_held = decsim_config.microseconds_to_ticks(15.0)
    expected_released = decsim_config.microseconds_to_ticks(18.0)

    assert lanes[slice_row["tid"]] == "Strong tier"
    assert args["request"] == "1:0:strong:1"
    assert args["waits_for"] == "stored rounds of operation 1"
    assert args["outcome"] == "strong context stored in strong syndrome buffer"
    assert args["round_count"] == 6
    assert args["tick"] == expected_held
    assert slice_row["dur"] == pytest.approx(3.0)

    arrows = _flows_of(document, lanes, "Strong tier", "1:0")
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


def _flows_of(document, lanes: dict, lane: str, window_text: str) -> list:
    """Every flow event of one window's chain on one lane."""
    rows = []
    for row in document:
        if row["ph"] not in ("s", "t", "f"):
            continue
        if lanes[row["tid"]] != lane:
            continue
        if row["args"].get("window") != window_text:
            continue
        rows.append(row)
    return rows


def _input_residences(document) -> list:
    """Every span the trace draws for a decode input held in memory."""
    rows = []
    for row in document:
        if row["ph"] == "X" and row["name"].endswith(" input in memory"):
            rows.append(row)
    return rows


def test_a_held_speculative_decode_is_cancelled_by_a_confident_result():
    """Step 3 halts the strong computation a confident weak result spares.

    Toshio arXiv:2510.25222 Sec. III A, step 3 (lines 610-615). A speculative
    strong decode still waiting for its context has not started, so the
    confident weak result ends it where it waits; the room-side rounds it would
    have read are freed with the window's final commit, and the run settles with
    nothing held.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
        escalation_microseconds=40.0,
    )
    machine.run()
    cancelled = fabric.log_lines_containing(
        machine, "cancelled while held for its input"
    )
    assert len(cancelled) == 3
    assert not machine.windows.window_manager.strong_redecode.has_pending()
    assert fabric.frame_tiers(machine) == [
        ((1, 0), "weak"),
        ((1, 1), "weak"),
        ((1, 2), "weak"),
    ]


def test_every_strong_request_is_cancelled_when_the_weak_tier_is_confident():
    """A confident weak result cancels its speculative decode and its context.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: the parallel strong
    decode is speculative, so a confident weak result cancels it and no
    window commits from the strong tier. The cancel also owns the
    room-side hold the request took, and the strong syndrome buffer would report
    an unresolved hold at the end of the run if it did not release it.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows=set(),
        run_both_at_once=True,
    )
    machine.run()
    counts = machine.decoders.decoder_manager.strong_requests.counts
    tiers = fabric.frame_tiers(machine)

    assert tiers == [((1, 0), "weak"), ((1, 1), "weak"), ((1, 2), "weak")]
    assert counts.cancelled == 3
    assert counts.needed == 0
    machine.readout.strong_syndrome_round_receiver.check_settled()


def test_every_window_takes_the_strong_result_when_the_weak_tier_is_not():
    """The other edge of the parallel variant: nothing is cancelled.

    Toshio arXiv:2510.25222 Sec. III A, Step 1: when every weak result
    falls below the threshold the speculative decode that already ran is the
    window's answer, so every request is needed and the frame carries
    the strong tier alone.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows={0, 1, 2},
        run_both_at_once=True,
    )
    machine.run()
    counts = machine.decoders.decoder_manager.strong_requests.counts
    tiers = fabric.frame_tiers(machine)

    assert tiers == [((1, 0), "strong"), ((1, 1), "strong"), ((1, 2), "strong")]
    assert counts.needed == 3
    assert counts.cancelled == 0


def test_a_pinned_speculative_decode_is_planned_when_its_weak_job_unparks():
    """Step 1 on a pinned row starts the speculative decode at the unpark.

    Toshio arXiv:2510.25222 Sec. III A, Step 1 feeds both decoders the
    window at once (lines 598-601), and redo_window pins its past
    face on the earlier neighbour's final commit (Bombin arXiv:2303.04846
    lines 775-788). Under held boundaries that commit is what unparks
    the weak job, so the speculative decode is planned at that instant and every
    window's strong result pins on a committed neighbour.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows={0, 1, 2},
        strong_window=declared_run.REDO_WINDOW,
        run_both_at_once=True,
    )
    machine.run()
    counts = machine.decoders.decoder_manager.strong_requests.counts
    tiers = fabric.frame_tiers(machine)

    assert tiers == [((1, 0), "strong"), ((1, 1), "strong"), ((1, 2), "strong")]
    assert counts.needed == 3
    assert not machine.windows.window_manager.strong_redecode.has_pending()


def test_the_chips_manager_serves_no_strong_job_and_the_hosts_no_weak_one():
    """Each side's manager serves its own jobs and never the other's.

    LATTE 2509.03954 lines 705-720: the host's scheduler owns the strong
    queue and pool, and the local decoder beside the control electronics
    has no part in it.
    """
    machine = fabric.switching_machine(
        rounds=9,
        escalated_windows={0, 1, 2},
        run_both_at_once=False,
    )
    chip_kinds = set()
    host_kinds = set()

    def ended_on_chip(job, _result, _outcome, _ticks):
        chip_kinds.add(job.kind)

    def ended_on_host(job, _result, _outcome, _ticks):
        host_kinds.add(job.kind)

    chip = machine.decoders.decoder_manager
    host = machine.decoders.strong_decoder_manager
    chip.outcomes.trace.request_ended.connect(ended_on_chip)
    host.outcomes.trace.request_ended.connect(ended_on_host)
    machine.run()

    assert chip_kinds == {decoding_records.DecodeJobKind.WINDOW}
    assert host_kinds == {decoding_records.DecodeJobKind.STRONG_REDECODE}
    assert host.strong_requests.counts.needed == 3


def test_a_redone_windows_rounds_leave_the_strong_buffer_when_its_input_lands():
    """No reader is left once the strong unit holds the escalated input.

    The escalated window 0 commits rounds 1 to 3 and reads to round 6;
    window 1 commits 4 to 6. A strong redo reads no round behind its
    commit (Bombin et al. 2303.04846 lines 1456-1458), so window 1's
    potential strong read starts at round 4, and rounds 1 to 3 have no
    holder once window 0's strong input lands in the unit, 39.0 us here:
    the weak commit at 30.0, the escalation hop of 3.0 and the strong
    buffer's hop of 6.0. Rounds 4 to 6 stay for window 1 until its own
    commit releases them at 87.5 us, after the strong commit at 74.0.
    """
    machine = fabric.switching_machine(rounds=6, escalated_windows={0})
    probe = declared_run.OccupancyProbe(machine.windows.window_manager)
    machine.engine.action_done.connect(probe.observe)
    machine.run()
    timeline = probe.strong_timeline
    snapshot = machine.control.pauli_frame.snapshot()
    strong_record = snapshot.records[0]
    expected_committed = decsim_config.microseconds_to_ticks(74.0)
    expected_timeline = [
        (decsim_config.microseconds_to_ticks(0.0), 0),
        (decsim_config.microseconds_to_ticks(33.0), 6),
        (decsim_config.microseconds_to_ticks(39.0), 3),
        (decsim_config.microseconds_to_ticks(87.5), 0),
    ]

    assert strong_record.window_key == (1, 0)
    assert strong_record.tier == "strong"
    assert strong_record.committed_ticks == expected_committed
    assert timeline == expected_timeline


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


def _walk_card_machine(microseconds):
    """One d=3 shot of union-find's cluster gap, its walk priced."""
    settings = switching_settings(20.0)
    host_time = cycle_count.HostMeasuredTime()
    union_find_decoder = union_find.UnionFindDecoder.Settings(timing=host_time)
    weak_decoder = dataclasses.replace(
        settings.weak_decoder, algorithm=union_find_decoder
    )
    cluster_gap = cluster.ClusterGap.Settings(walk_microseconds=microseconds)
    switching = dataclasses.replace(settings.switching, confidence=cluster_gap)
    observation = dataclasses.replace(
        settings.observation, record_switching_windows=True
    )
    settings = dataclasses.replace(
        settings,
        weak_decoder=weak_decoder,
        switching=switching,
        observation=observation,
    )
    machine = machine_module.Machine.build(settings, 0)
    decodes = declared_run.FinishedDecodes()
    decodes.attach(machine)
    machine.run()
    return machine, decodes


def test_a_priced_confidence_walk_charges_its_card_once_per_window():
    """A confidence walk given a time is a card, like a tier's.

    The cluster gap's walk is a Dijkstra over the decode's own edge
    intervals (Meister et al. 2405.07433 Algorithm 2), charged on the
    unit that grew them. Left None it is measured on the
    host clock; given a number it is that number of microseconds per
    window, so a run can price it the way a decoder tier is priced.
    """
    machine, decodes = _walk_card_machine(12.0)
    expected_ticks = decsim_config.microseconds_to_ticks(12.0)
    charged = _confidence_charges(machine)
    weak_decodes = decodes.of_pool("default")
    decode_count = len(weak_decodes)
    assert charged
    assert charged == [expected_ticks] * decode_count


def test_a_windows_commit_instant_says_whether_its_result_is_provisional(
    tmp_path,
):
    """An escalated window commits its weak result provisionally at the verdict.

    The strong result replaces that result later (WindowCommitter
    .finish_strong) and the frame's own committed instant marks the
    answer landing, so the planner's instant says which of the two it
    is; without the word a reader takes the verdict for the answer,
    which Toshio's chain puts a strong decode later (2510.25222 lines
    1247 to 1250, the strong decode starts once the boundaries are
    determined).
    """
    trace_path = tmp_path / "serial.trace.json"
    machine = fabric.switching_machine(
        rounds=9, escalated_windows={0}, trace_path=trace_path
    )
    machine.run()
    machine.observation.trace_writer.write(str(trace_path))
    text = trace_path.read_text()
    document = json.loads(text)
    (escalated,) = _events_named(document, "i", "W0 committed")
    (kept,) = _events_named(document, "i", "W1 committed")

    assert escalated["args"]["result"] == "provisional"
    assert kept["args"]["result"] == "final"


def test_one_landed_input_is_one_residence_however_many_solves_read_it(
    tmp_path,
):
    """The forced-class solves of a window share one landed input.

    Decision D2 (decoder_memory.DecoderMemory.rewrite): the two solves
    read one resident input and the memory frees it when the last
    reader takes it. The trace's residence of that input is therefore
    one record per window, closed when the rounds are freed; a record
    per solve left the first one open to the end of the run.
    """
    trace_path = tmp_path / "gap.trace.json"
    settings = switching_settings(0.0, rounds_per_shot=9)
    observation = dataclasses.replace(
        settings.observation, trace=str(trace_path)
    )
    settings = dataclasses.replace(settings, observation=observation)
    machine, _result = run_shot(settings, 0)
    machine.observation.trace_writer.write(str(trace_path))
    text = trace_path.read_text()
    document = json.loads(text)
    residences = _input_residences(document)
    freed_reasons = [row["args"]["freed_reason"] for row in residences]
    windows = machine.observation.windows.windows
    window_count = len(windows)

    assert freed_reasons == ["decode done"] * window_count
