"""The latency points of one shot, against a hand derivation of its run.

Every number here is arithmetic over the declared cards of the configs
below: a 250 MHz fabric, a weak unit whose fetch is one cycle per round
and whose release is ten cycles (0.040 us), 30 one-microsecond rounds at
distance 3, and sliding windows that commit 3 rounds and buffer 3. The
chain is the one Skoric et al. 2209.08552 describe: each window's decode
takes its own time (tau_W, lines 429-435) and waits for the seam before
it starts (the artificial defects of the block before it, lines
268-275). The two must not be one number, so the tests read service and the park
apart, and the park itself is two points by cause: dep_block until the
boundary it waits for is in, compute_wait until the unit's compute is
free. The points of one window still sum to the window's whole reaction
time.

The two-tier config gives every hop of the strong path a card of its
own, so a point's value names the wire it was read from: the strong
store's hop is two cycles, the escalation hop five and the strong
decoder's way home three, against one cycle everywhere on the weak
path. Escalating every window or none is the threshold's doing, which
is Toshio et al. 2510.25222 Sec. III A step 3 driven to its two ends.

The last run here is the one the yaml cannot declare: the memory_circuit
row is one operation for the whole shot, so the several-stream workload
that tells a per-window point apart from a per-index one is built in
Python through the circuit_list row, on a fabric whose only priced hop
is the decoder-to-decoder seam.
"""

import collections
import dataclasses
import math
from typing import Optional

import pytest
import yaml

import decsim.collect as collect
import decsim.config as config_module
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.relay_belief_propagation.decoder as relay_decoder
import decsim.decoders.settings as decoder_settings
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.link_traffic as link_traffic
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings
from tests.experiments.yaml_configs import (
    CONFIGS_DIR,
    MINIMAL_CONFIG,
    measure_point_shot,
    memory_workload,
    write_config,
)

# the detection events of one round of the swept distance-three patch,
# and of its readout round, which compares the data qubits as well
BITS_PER_ROUND = 8
READOUT_ROUND_BITS = 12
# every hop of the weak-only fabric at one cycle of the fridge clock
ONE_TIER_LINKS = {
    "qpu_to_controller": {
        "latency_cycles": 1,
        "clock": "fridge",
        "bits_per_cycle": None,
    },
    "controller_to_weak_buffer": {
        "latency_cycles": 1,
        "clock": "fridge",
        "bits_per_cycle": None,
    },
    "weak_buffer_to_weak_decoder": {
        "latency_cycles": 1,
        "clock": "fridge",
        "bits_per_cycle": None,
    },
    "decoder_to_decoder": {
        "latency_cycles": 1,
        "clock": "fridge",
        "bits_per_cycle": None,
    },
    "weak_decoder_to_frame": {
        "latency_cycles": 1,
        "clock": "fridge",
        "bits_per_cycle": None,
    },
    "controller_to_strong_buffer": None,
    "strong_buffer_to_strong_decoder": None,
    "weak_decoder_to_strong_decoder": None,
    "frame_to_controller": None,
    "controller_to_qpu": None,
}

# the points that lie end to end between a window's data being complete
# in the weak syndrome buffer and its correction being committed in the frame
CHAIN = (
    "queue_wait",
    "weak_attempt",
    "input_link_per_window",
    "dep_block",
    "compute_wait",
    "service",
    "confidence",
    "output_link_per_window",
    "frame_commit",
)


def fridge_hop(cycles: int) -> dict:
    """One link card of that many cycles of the 250 MHz fridge clock."""
    return {
        "latency_cycles": cycles,
        "clock": "fridge",
        "bits_per_cycle": None,
    }


# the two-tier fabric, every strong hop on a card of its own
TWO_TIER_LINKS = {
    "qpu_to_controller": fridge_hop(1),
    "controller_to_weak_buffer": fridge_hop(1),
    "controller_to_strong_buffer": fridge_hop(1),
    "weak_buffer_to_weak_decoder": fridge_hop(1),
    "strong_buffer_to_strong_decoder": fridge_hop(2),
    "weak_decoder_to_strong_decoder": fridge_hop(5),
    "decoder_to_decoder": fridge_hop(1),
    "weak_decoder_to_frame": fridge_hop(1),
    "strong_decoder_to_frame": fridge_hop(3),
    "frame_to_controller": None,
    "controller_to_qpu": None,
}


def slow_unit_shot(tmp_path, units: int, card_microseconds: float = 5.0):
    """One shot of 30 rounds on `units` weak units of that card."""
    raw = dict(MINIMAL_CONFIG)
    workload = memory_workload(30)
    raw["workload"] = workload
    raw["links"] = ONE_TIER_LINKS
    raw["weak_decoder"] = {
        "kind": card_microseconds,
        "units": units,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 10,
            "release_cycles_per_round": 0,
        },
    }
    config_path = tmp_path / "slow_unit.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)
    return measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )


def switching_run(
    tmp_path,
    gap_threshold_db: float,
    run_both_at_once: bool = False,
    seed: int = 0,
    strong_units: int = 1,
    observation: tuple = (),
    patch_count: int = 1,
    commit_rounds: Optional[int] = None,
    sections: Optional[dict] = None,
):
    """One collected shot of a 1.0 us weak tier beside a 10.0 us one.

    A threshold far above every gap escalates every window and one far
    below escalates none, so the same two cards answer both branches.
    run_both_at_once starts the strong sibling with the weak job and
    cancels it when the weak result is kept, which is Toshio et al.
    2510.25222 Sec. III A step 1. observation names the section's flags
    the shot turns on; more than one patch runs the memory_patches maker;
    commit_rounds sets windows.commit_rounds, the code distance when
    None; sections replaces whole sections last.
    """
    raw = dict(MINIMAL_CONFIG)
    windows = dict(MINIMAL_CONFIG["windows"])
    windows["commit_rounds"] = commit_rounds
    raw["windows"] = windows
    raw["observation"] = dict.fromkeys(observation, True)
    workload = memory_workload(30)
    raw["workload"] = workload
    if patch_count > 1:
        workload["function"] = "decsim.producers:memory_patches"
        workload["arguments"]["patch_count"] = patch_count
    raw["links"] = TWO_TIER_LINKS
    raw["escalation"] = {
        "kind": "switching",
        "gap_threshold_db": gap_threshold_db,
        "strong_window": "near_seam_pinned",
        "run_both_at_once": run_both_at_once,
    }
    raw["weak_decoder"] = {
        "kind": 1.0,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 10,
            "release_cycles_per_round": 0,
        },
    }
    raw["strong_decoder"] = {
        "kind": 10.0,
        "units": strong_units,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 10,
            "release_cycles_per_round": 0,
        },
    }
    if sections is not None:
        raw.update(sections)
    config_path = tmp_path / "switching.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        1,
    )
    return collect.run_shot(task, seed)


def switching_shot(
    tmp_path,
    gap_threshold_db: float,
    run_both_at_once: bool = False,
    seed: int = 0,
):
    """That shot's measurement."""
    shot = switching_run(tmp_path, gap_threshold_db, run_both_at_once, seed)
    return measure.measure_shot(shot)


def bounded_store_shot(tmp_path):
    """One shot whose weak syndrome buffer holds six rounds, with a slow unit.

    Thirty rounds arrive a microsecond apart into a store of six
    rounds, which is five ordinary rounds and the wider readout round,
    and the 5.0 us unit frees three slots per window it reads, so the
    controller has to hold rounds it has already packed.
    """
    six_rounds_bits = 5 * BITS_PER_ROUND + READOUT_ROUND_BITS
    raw = dict(MINIMAL_CONFIG)
    workload = memory_workload(30)
    raw["workload"] = workload
    raw["links"] = ONE_TIER_LINKS
    raw["weak_syndrome_buffer"] = {"bits": six_rounds_bits}
    raw["weak_decoder"] = {
        "kind": 5.0,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 10,
            "release_cycles_per_round": 0,
        },
    }
    config_path = tmp_path / "bounded_store.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)
    return measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )


def ticks_of(microseconds: float) -> int:
    """One point's microseconds back in ticks, so a sum is exact."""
    ticks = microseconds * 1000000
    return round(ticks)


def chain_sum_ticks(measurement) -> list:
    """Every window's chain points added up in ticks, in window order."""
    totals = []
    window_count = len(measurement.samples["service"])
    for index in range(window_count):
        total = 0
        for point in CHAIN:
            sample = measurement.samples[point][index]
            total += ticks_of(sample)
        totals.append(total)
    return totals


def reaction_ticks(measurement) -> list:
    """Every window's data-complete to frame-committed span, in ticks."""
    totals = []
    for sample in measurement.samples["buffer0_ready_to_frame"]:
        ticks = ticks_of(sample)
        totals.append(ticks)
    return totals


def test_service_is_the_compute_and_the_park_is_its_own_point(tmp_path):
    """One 5.0 us unit at distance 3: the same compute on every window.

    The unit's compute is fetch 0.024 + algorithm 5.0 + release 0.040 =
    5.064 us, and it is the same on all nine windows however long they
    waited. What grows with the backlog is the wait: the ready-queue
    wait before a unit takes the job, and the park after the input has
    landed, which holds until the window before it hands over its
    boundary 0.004 us after its own decode ends. That park is all
    dependency: the one unit ended the predecessor's decode before it
    handed the boundary over, so its compute was already free when the
    window became startable and compute_wait is zero on every window.
    """
    measurement = slow_unit_shot(tmp_path, 1)

    samples = measurement.samples
    assert samples["service"] == [5.064] * 9
    assert samples["compute_wait"] == [0.0] * 9
    assert samples["dep_block"] == [
        0.0,
        2.068,
        4.136,
        5.068,
        5.068,
        5.068,
        5.068,
        5.068,
        5.068,
    ]
    assert samples["queue_wait"] == [
        0.0,
        0.0,
        0.0,
        1.136,
        3.204,
        5.272,
        7.340,
        9.408,
        11.476,
    ]
    assert samples["input_link_per_window"] == [0.004] * 9
    assert measurement.means["service"] == 5.064


def test_one_windows_points_sum_to_its_reaction_time(tmp_path):
    """Queue wait, input link, park, service, output link and commit.

    Those six are the whole path from the window's data being complete
    in the weak syndrome buffer to its correction committed in the frame, so
    they add up to buffer0_ready_to_frame on every window to the tick.
    """
    measurement = slow_unit_shot(tmp_path, 1)

    totals = chain_sum_ticks(measurement)
    assert totals == reaction_ticks(measurement)
    assert measurement.samples["buffer0_ready_to_frame"] == [
        5.076,
        7.144,
        9.212,
        11.280,
        13.348,
        15.416,
        17.484,
        19.552,
        21.620,
    ]


def test_load_is_the_units_occupancy_over_the_window_period(tmp_path):
    """Load is the compute plus the seam handoff over the period.

    Three one-microsecond rounds commit per window, so a 5.064 us
    compute plus the 0.004 us seam handoff on eight of the nine windows
    is a chain 1.689 times too slow, which is the backlog condition of
    Skoric et al. 2209.08552 lines 429-435 read as a ratio.
    """
    measurement = slow_unit_shot(tmp_path, 1)

    handoff_us = 8 * 0.004 / 9
    expected = (5.064 + handoff_us) / 3.0
    assert measurement.load == expected


def test_a_second_unit_moves_the_wait_and_leaves_service_alone(tmp_path):
    """Two units decode the same chain at the same ticks.

    The seam serialises the commits whatever the unit count, so every
    window commits at the tick it did on one unit and the reaction time
    does not move. What moves is where the wait is booked: the second
    unit takes each window as it arrives, so the ready-queue wait almost
    disappears and the same microseconds appear in the park instead.
    They appear in the dependency half of it and not in the structural
    half: the boundary a window waits for leaves after its
    predecessor's decode has ended, and by then the unit holding the
    window is idle, so compute_wait stays zero however many units the
    pool has. The two halves trade places where two decodes of one
    window share a unit, which is the forced-class pair of
    test_a_kept_weak_result_is_measured_on_the_weak_hops.
    """
    one_unit = slow_unit_shot(tmp_path, 1)
    two_units = slow_unit_shot(tmp_path, 2)

    assert two_units.samples["service"] == one_unit.samples["service"]
    assert two_units.load == one_unit.load
    assert (
        two_units.samples["buffer0_ready_to_frame"]
        == one_unit.samples["buffer0_ready_to_frame"]
    )
    assert two_units.samples["queue_wait"] == [0.0] * 8 + [1.340]
    assert two_units.samples["compute_wait"] == [0.0] * 9
    assert two_units.samples["dep_block"] == [
        0.0,
        2.068,
        4.136,
        6.204,
        8.272,
        10.340,
        12.408,
        14.476,
        15.204,
    ]


def test_the_pool_columns_read_each_tiers_own_queue_and_units(tmp_path):
    """A weak-only run's pool columns are the default pool's, the strong zero.

    The deepest the weak tier's own queue got is the deepest the ready
    queue got, since only that pool exists; a second unit halves the
    busy fraction exactly, because the same nine decodes of 5.064 us
    each run on two units over the same span
    (test_a_second_unit_moves_the_wait_and_leaves_service_alone), which
    is Triage's utilization rate read per tier (2605.04459 lines
    1024-1031).
    """
    one_unit = slow_unit_shot(tmp_path, 1)
    two_units = slow_unit_shot(tmp_path, 2)

    assert one_unit.weak_queue_max == one_unit.max_queued_windows == 3
    assert two_units.weak_queue_max == 1
    assert one_unit.strong_queue_max == 0
    assert one_unit.strong_busy_fraction == 0.0
    assert one_unit.escalated_windows == 0
    assert one_unit.strong_decoded_rounds == 0
    assert two_units.weak_busy_fraction == one_unit.weak_busy_fraction / 2


def test_a_strong_primary_runs_pool_columns_are_the_strong_tiers():
    """Under strong_only the default pool's numbers are the strong tier's.

    The plan's windows queue in the default pool whichever tier decodes
    them (decode_queue.POOL_BY_JOB_KIND), and under strong_only that
    tier is the strong one, so its queue peak and busy fraction belong
    in the strong columns and the weak columns read zero, the mirror of
    test_the_pool_columns_read_each_tiers_own_queue_and_units.
    """
    shot = shipped_shot("strong_decoder_baseline.yaml")
    measurement = measure.measure_shot(shot)

    assert measurement.weak_queue_max == 0
    assert measurement.weak_busy_fraction == 0.0
    assert measurement.strong_queue_max == measurement.max_queued_windows
    assert measurement.strong_busy_fraction > 0.0


def test_skorics_process_count_is_two_services_over_the_window_period(
    tmp_path,
):
    """N_par is ceil(2 tau_W / ((n_com + n_W) tau_rd)).

    Layer A commits n_com = 3 rounds and layer B its whole window,
    n_W = n_com + 2 n_buf = 9 (2209.08552 lines 388-390). A 5.064 us
    decode twice over those twelve one-microsecond rounds is 0.844
    processes, so one (lines 429-438).
    """
    measurement = slow_unit_shot(tmp_path, 1)

    assert measurement.parallel_processes_needed == 1


def test_skorics_process_count_is_above_one_when_a_decode_outlasts_the_layers(
    tmp_path,
):
    """The same twelve rounds against a 10 us card ask for two processes.

    N_par is ceil(2 tau_W / ((n_com + n_W) tau_rd)) with tau_W the
    shot's mean service and n_W = n_com + 2 n_buf (2209.08552 lines
    388-390 and 429-438). The window scheme's rounds are null here, so
    both sizes are the distance.
    """
    measurement = slow_unit_shot(tmp_path, 1, card_microseconds=10.0)

    commit_round_count = 3
    buffer_round_count = 3
    round_period_microseconds = 1.0
    both_buffers_round_count = 2 * buffer_round_count
    window_round_count = commit_round_count + both_buffers_round_count
    committed_round_count = commit_round_count + window_round_count
    committed_rounds_us = committed_round_count * round_period_microseconds
    both_layers_service_us = 2 * measurement.means["service"]
    processes = both_layers_service_us / committed_rounds_us
    expected = math.ceil(processes)
    assert expected > 1
    assert measurement.parallel_processes_needed == expected


def test_toshios_per_decode_bound_is_commit_time_over_escalated_share(
    tmp_path,
):
    """Theorem 1 per decode, read off a point where every window escalated.

    Ten windows escalated and their strong decodes read 57 rounds in
    all. Eq. (6) bounds the time per round by (1 / gamma)(d / r_strong)
    tau_gen = 3 / 5.7 us (2510.25222 lines 1206-1214), so one decode of
    r_strong = 5.7 rounds is bounded by 3 us: tau_gen r_com windows /
    escalated windows with r_com = d = 3. The strong service it bounds
    is the 10.0 us card plus its fetch and release, above it, as a card
    ten times the round time must be.
    """
    measurement = switching_shot(tmp_path, 1000000.0)
    record = report.record_of([measurement])
    rows = report.summarize(record.shots, record.window_samples)

    assert measurement.escalated_windows == 10
    assert measurement.strong_decoded_rounds == 57
    assert measurement.strong_service_mean_us == 10.0628
    assert rows[0]["strong_service_bound_us"] == 3.0
    assert rows[0]["escalated_windows"] == 10


def test_the_per_decode_bound_reads_r_com_off_the_commit_rounds_column(
    tmp_path,
):
    """Theorem 1 per decode at r_com = 2, not d = 3.

    With windows.commit_rounds 2 the window period is 2 tau_gen, so
    tau_gen r_com windows / escalated windows is 1.0 us x 2 x 15 / 15 =
    2 us over the fifteen windows of 30 rounds, every one escalated
    (Toshio 2510.25222 eq. (6), gamma_switch per d rounds, lines
    1254-1255). A bound read with r_com = d would say 3 us.
    """
    shot = switching_run(tmp_path, 1000000.0, commit_rounds=2)
    measurement = measure.measure_shot(shot)
    record = report.record_of([measurement])
    rows = report.summarize(record.shots, record.window_samples)

    assert record.shots[0]["commit_rounds"] == 2
    assert measurement.windows == 15
    assert measurement.escalated_windows == 15
    assert rows[0]["strong_service_bound_us"] == 2.0


def test_an_escalated_window_is_measured_on_the_strong_tiers_own_hops(
    tmp_path,
):
    """Every window escalates: the points are the strong decode's.

    The result the frame committed crossed the strong store's hop in
    (0.008 us) and the strong decoder's hop home (0.012 us), and the
    decode it describes is the 10.0 us one. The weak decode that did not
    commit is the weak_attempt point, and the escalation hop that
    carried the selection and then the window's rounds to the strong
    tier (0.020 us, the two transfers side by side on a latency-only
    card) is its own point, outside the sum. What it costs the decode
    is dep_block: the strong input hop starts when the rounds land, so
    the 0.020 us before it is the wait for the rounds. The last window's
    rounds were carried up with the window before it, so its input hop
    starts at the verdict beside the selection and waits 0.012 us for
    it after landing. The weak attempt starts where a unit took the
    window's first decode,
    so on window 2, whose complementary gap ran its two forced-class
    solves one after the other, it is both of them: 1.064 us of the
    first solve on top of the 12.240 us from the second one's dispatch,
    which waits on window 1's held boundary and so on window 1's strong
    decode, 0.008 us of input hop after its rounds landed.
    """
    measurement = switching_shot(tmp_path, 1000000.0)

    samples = measurement.samples
    assert samples["input_link_per_window"] == [0.008] * 10
    assert samples["output_link_per_window"] == [0.012] * 10
    assert samples["escalation_link_per_window"] == [0.020] * 10
    assert samples["dep_block"] == [0.020] * 9 + [0.012]
    assert samples["compute_wait"] == [0.0] * 10
    assert samples["algorithm"] == [10.0] * 10
    assert samples["weak_attempt"][2] == 13.304


def test_an_escalated_windows_points_sum_to_its_reaction_time(tmp_path):
    """The chain runs weak attempt first, then the strong decode.

    Toshio et al. 2510.25222 Sec. III A orders it: the weak decoder
    answers, the verdict sends the window and its rounds to the strong
    decoder, the strong decoder answers and that is what commits. Every
    point on that order adds up to the window's reaction time to the
    tick.
    """
    measurement = switching_shot(tmp_path, 1000000.0)

    totals = chain_sum_ticks(measurement)
    assert totals == reaction_ticks(measurement)


def test_a_kept_weak_result_is_measured_on_the_weak_hops(tmp_path):
    """No window escalates: the weak hops, and no escalation at all.

    The same config below the threshold keeps every weak result, so both
    link points read the weak path's one cycle and the two escalation
    points are zero on every window. On seed 50 window 3 is the
    exception that makes the rule plain: the result it committed came
    from the second of its two forced-class solves, which read the input
    the first solve had already brought into the unit's memory, so that
    decode crossed no link at all and its own hop is zero. The window's
    rounds still crossed once, and shot_links.csv still counts that
    crossing.

    Window 3 is where the park's two halves trade places. Its decode
    was dispatched with its sibling and waited 0.004 us for the input
    the sibling's transfer brought, which is the dependency it had, and
    then 1.064 us more for the unit's compute to finish that sibling's
    solve, which is the structural wait gem5 counts as fuBusy. Window 9
    is the plain case beside it: 1.064 us of dependency wait and no
    structural wait at all.
    """
    measurement = switching_shot(tmp_path, 0.0, seed=50)

    samples = measurement.samples
    weak_hops = [0.004] * 3 + [0.0] + [0.004] * 6
    assert samples["input_link_per_window"] == weak_hops
    assert samples["dep_block"] == [0.0] * 3 + [0.004] + [0.0] * 5 + [1.064]
    assert samples["compute_wait"] == [0.0] * 3 + [1.064] + [0.0] * 6
    assert samples["output_link_per_window"] == [0.004] * 10
    assert samples["escalation_link_per_window"] == [0.0] * 10
    assert samples["weak_attempt"] == [0.0] * 10
    assert samples["algorithm"] == [1.0] * 10


def shipped_shot(config_name: str):
    """One seeded shot of a config this repository ships.

    p 0.008, distance 3, one microsecond rounds, seed 0: the point the
    component validation reads, so what these assertions
    walk is a run folder a reader can build from the shipped yaml.
    """
    config_path = CONFIGS_DIR / config_name
    config = experiment.load_experiment(config_path)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        1,
    )
    return collect.run_shot(task, 0)


def decoded_window_ids(shot) -> list:
    """The windows the measurement's samples are in the order of."""
    observation = shot.machine.observation
    frames = measure.frame_records_by_window(observation)
    window_ids = []
    windows = observation.windows.windows.items()
    for key, window in sorted(windows):
        if key in frames and window.t_done is not None:
            window_ids.append(key[1])
    return window_ids


def chain_gap_ticks(shot, measurement) -> dict:
    """Window id -> its chain sum minus its reaction time, in ticks."""
    gaps = {}
    window_ids = decoded_window_ids(shot)
    sums = chain_sum_ticks(measurement)
    reactions = reaction_ticks(measurement)
    for index, window_id in enumerate(window_ids):
        gaps[window_id] = sums[index] - reactions[index]
    return gaps


def later_solve_ticks(shot, window_id: int) -> int:
    """What the window decoded after the decode that committed, in ticks.

    A complementary gap is two forced-class solves of one window
    (decision D2) and the window answers when both have, so the solve
    that did not commit can run after the one that did. That span is
    the window's confidence step, the point of the same name, so the
    tests read it here and expect it there.
    """
    observation = shot.machine.observation
    frames = measure.frame_records_by_window(observation)
    frame = frames[(1, window_id)]
    window = observation.windows.windows[(1, window_id)]
    ends = []
    for record in observation.stages.records_for(1, window_id):
        if frame.run_sequence in record.run_sequences:
            ends.append(record.end_ticks)
    committing_end = max(ends)
    later_ticks = window.t_done - committing_end
    return max(0, later_ticks)


def test_the_shipped_weak_baseline_sums_to_its_reaction_time():
    """The identity on a config the repository ships, not a test's own.

    configs/weak_decoder_baseline.yaml decodes each of its nine windows
    once and answers on that decode, so its confidence step is zero and
    every window's points are its whole path from its data being
    complete in the weak syndrome buffer to its correction committed in the
    frame.
    """
    shot = shipped_shot("weak_decoder_baseline.yaml")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    assert len(gaps) == 9
    assert measurement.samples["confidence"] == [0.0] * 9
    for window_id, gap in gaps.items():
        assert later_solve_ticks(shot, window_id) == 0
        assert gap == 0


def test_the_shipped_two_tier_config_sums_to_its_reaction_time():
    """The identity on every window of configs/two_tiers.yaml.

    Windows 0, 1, 8 and 9 escalated: their weak attempt, the strong
    decode that committed and the hops around it are the whole path, and
    their confidence step is zero because the attempt already runs to the
    verdict. The other six kept a weak result whose complementary gap
    ran a second forced-class solve after it, and that solve is their
    confidence step (decision D2, and Toshio et al. 2510.25222 Sec.
    III A steps 2 to 4 for the order of the escalated ones). Both sum
    to the tick.
    """
    shot = shipped_shot("two_tiers.yaml")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    escalated = [0, 1, 8, 9]
    with_a_later_solve = [2, 3, 4, 5, 6, 7]
    decoded = escalated + with_a_later_solve
    assert sorted(gaps) == sorted(decoded)
    for window_id in escalated:
        assert later_solve_ticks(shot, window_id) == 0
    for window_id in with_a_later_solve:
        assert later_solve_ticks(shot, window_id) > 0
    for gap in gaps.values():
        assert gap == 0


def test_the_shipped_pinned_config_sums_to_its_reaction_time():
    """The same law where the strong tier's time is measured, not declared.

    configs/seam_pinned_switching.yaml names a belief-matching strong
    tier, whose decode time is read off the host clock, so the values
    move from host to host and the identity does not: every window's
    points still add up to its reaction time to the tick, the solve it
    ran after the committing one being its confidence step.
    """
    shot = shipped_shot("seam_pinned_switching.yaml")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    assert gaps
    confidence = measurement.samples["confidence"]
    window_ids = decoded_window_ids(shot)
    for window_id, gap in gaps.items():
        assert gap == 0
        index = window_ids.index(window_id)
        later = later_solve_ticks(shot, window_id)
        assert ticks_of(confidence[index]) == later


def test_the_shipped_cluster_gap_config_sums_to_its_reaction_time():
    """The identity where the confidence step is a card of its own.

    configs/cluster_gap_switching.yaml walks a union find decode for its
    signal and prices that walk at 12.0 us on the weak unit (decision
    D8), so a window the weak tier answered carries the walk between its
    decode's end and its verdict. A window that escalated carries none:
    its weak attempt already runs to the verdict and the strong decode
    it commits answers after it. The union find decode itself is read
    off the host clock, so the assertions are in ticks and about the
    identity, never about a magnitude.
    """
    shot = shipped_shot("cluster_gap_switching.yaml")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    confidence = measurement.samples["confidence"]
    walk_us = 12.0
    kept = []
    for index, value in enumerate(confidence):
        if value:
            kept.append(index)
    assert kept
    assert len(kept) < len(confidence)
    for index, value in enumerate(confidence):
        if index in kept:
            assert value == walk_us
        else:
            assert value == 0.0
    for gap in gaps.values():
        assert gap == 0


def test_the_stage_points_are_the_committing_decodes_own_stages(tmp_path):
    """Fetch, algorithm and release add up to the service they are in.

    Every window of this run escalates, so the decode the frame took is
    the strong one and its three stages are the compute the service
    point measures: a six-round strong window, the commit region and
    the buffer ahead of it, fetches 0.024 us, decodes 10.0 and releases
    0.040, and the last one, whose buffer the stream's end clips to
    nothing, fetches 0.012.
    """
    measurement = switching_shot(tmp_path, 1000000.0)

    samples = measurement.samples
    stage_totals = []
    for index in range(len(samples["service"])):
        total = ticks_of(samples["fetch"][index])
        total += ticks_of(samples["algorithm"][index])
        total += ticks_of(samples["release"][index])
        stage_totals.append(total)
    service_ticks = []
    for sample in samples["service"]:
        ticks = ticks_of(sample)
        service_ticks.append(ticks)
    assert stage_totals == service_ticks
    assert samples["fetch"] == [0.024] * 9 + [0.012]


def test_a_cancelled_siblings_card_is_not_the_windows_algorithm(tmp_path):
    """A parallel run that escalates nothing reports the weak card.

    run_both_at_once starts a strong sibling on every window and cancels
    it at the verdict, and the sibling records its own stages under the
    same window key. The window's algorithm point is the decode that
    committed, which is the 1.0 us weak one on every window here.
    """
    measurement = switching_shot(tmp_path, 0.0, True)

    samples = measurement.samples
    assert samples["algorithm"] == [1.0] * 10
    assert measurement.means["algorithm"] == 1.0
    assert samples["service"] == [1.064] * 9 + [1.052]


def test_a_second_forced_solve_is_not_the_windows_algorithm(tmp_path):
    """The complementary gap runs two solves; one of them committed.

    Both are jobs of the same window and each is charged one card
    (decisions D2 and D7), so the window's algorithm point is one card's
    1.0 us and never their sum.
    """
    measurement = switching_shot(tmp_path, 0.0)

    assert measurement.samples["algorithm"] == [1.0] * 10


def test_a_cancelled_siblings_record_ends_at_the_cancel(tmp_path):
    """The strong sibling stops where the weak verdict stopped it.

    run_both_at_once starts the strong decoder on every window and the
    confident weak verdict halts it (Toshio et al. 2510.25222 Sec. III A
    steps 1 and 3). The engine cannot unschedule the card's timer, so
    the cancel is what closes the sibling's open stage: one cancelled
    record per window, each ending at that window's verdict rather than
    ten microseconds later, and the weak decode's own records untouched.
    """
    shot = switching_run(tmp_path, 0.0, True)

    stages = shot.machine.observation.stages
    windows = shot.machine.observation.windows.windows
    cancelled = []
    for record in stages.records:
        if record.cancelled:
            cancelled.append(record)
    ends = []
    verdicts = []
    for record in cancelled:
        ends.append(record.end_ticks)
        window = windows[(1, record.window_id)]
        verdicts.append(window.t_done)
    assert len(cancelled) == 10
    assert ends == verdicts
    for record in cancelled:
        assert record.stage == "algorithm"


def test_a_full_buffer_0_holds_rounds_and_the_wait_is_a_point(tmp_path):
    """The store's back-pressure on the controller, round by round.

    Rounds 1 to 15 find room. Round 16 is packed at 16.004 us into a
    full store and waits 0.144 us for the slot window 3's input frees at
    16.148; rounds 19 to 21 wait 2.212, 1.212 and 0.212 us for the three
    slots window 4 frees at 21.216, and the pattern repeats to round 30,
    the worst wait being 8.416 us on round 28. Thirteen rounds wait,
    51.912 us in all.
    """
    measurement = bounded_store_shot(tmp_path)

    stalls = measurement.samples["cwb_stall_per_round"]
    assert stalls[:15] == [0.0] * 15
    assert stalls[15] == 0.144
    assert stalls[18:21] == [2.212, 1.212, 0.212]
    assert stalls[27] == 8.416
    waiting = []
    for stall in stalls:
        if stall > 0.0:
            waiting.append(stall)
    total_ticks = 0
    for stall in stalls:
        total_ticks += ticks_of(stall)
    assert len(waiting) == 13
    assert total_ticks == 51912000


def test_the_store_wait_is_not_in_the_hop_the_round_then_crosses(tmp_path):
    """The round waits before the wire is asked for.

    The hop's own point is the card's 0.004 us on every one of the
    thirty rounds, held room or not, because the transfer is requested
    at the release; the wait is the store's and is reported as the
    store's.
    """
    measurement = bounded_store_shot(tmp_path)

    assert measurement.samples["cwb_per_round"] == [0.004] * 30
    assert measurement.means["cwb_stall_per_round"] == 51.912 / 30


def test_a_hops_setup_is_in_the_hop_and_not_in_the_wait_before_it(tmp_path):
    """A DMA's fixed delay is part of the latency its requester sees.

    gem5 adds a DMA's delay to the completion it schedules for the
    device (src/dev/dma_device.cc:116-118). The input hop here is one
    cycle of the 250 MHz fridge clock plus a five-cycle setup, 0.024 us,
    on every window, and a 0.1 us unit keeps pace with the rounds, so
    no window owes anything and none waits before its input hop.
    """
    raw = dict(MINIMAL_CONFIG)
    workload = memory_workload(30)
    raw["workload"] = workload
    links = dict(ONE_TIER_LINKS)
    input_hop = fridge_hop(1)
    input_hop["setup_cycles_per_transfer"] = 5
    links["weak_buffer_to_weak_decoder"] = input_hop
    raw["links"] = links
    raw["weak_decoder"] = {
        "kind": 0.1,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 10,
            "release_cycles_per_round": 0,
        },
    }
    config_path = tmp_path / "input_setup.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)
    measurement = measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )

    assert measurement.samples["input_link_per_window"] == [0.024] * 9
    assert measurement.samples["dep_block"] == [0.0] * 9


def seam_streams(stream_count: int) -> tuple:
    """That many memory streams, one patch and one qubit each.

    Every stream is the same 30-round distance-3 memory circuit, which
    is the circuit_list row of the workload table: the yaml's
    memory_circuit row plans one operation for the whole shot, so a run
    with more than one stream is declared here instead.
    """
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 30, 3, 0.001
    )
    operations = []
    for index in range(stream_count):
        operation_id = index + 1
        operation = program_records.Operation(
            id=operation_id,
            name=f"mem{operation_id}",
            qubits=(index,),
            patches=(index,),
            circuit=circuit,
        )
        operations.append(operation)
    return tuple(operations)


def seam_only_fabric():
    """The logical reference card with one hop priced: the seam.

    decoder_to_decoder is 125 cycles of the 250 MHz fridge clock, which
    is 0.5 us, and every other hop is free, so a window's outgoing
    boundary is the only thing a link charges for in the whole run.
    """
    clocks = config_module.ClockSettings({"fridge": 250.0, "room": 250.0})
    seam_card = {
        "latency_cycles": 125,
        "clock": "fridge",
        "bits_per_cycle": None,
    }
    free_card = {
        "latency_cycles": 0,
        "clock": "fridge",
        "bits_per_cycle": None,
    }
    fabric = {"kind": "logical_reference"}
    for path in transfer_records.LinkPath:
        fabric[path.value] = free_card
    fabric["decoder_to_decoder"] = seam_card
    return link_profiles.from_yaml(fabric, clocks, "one_card")


def seam_streams_shot(stream_count: int):
    """One shot of those streams on that fabric.

    Eight one-microsecond weak units decode them, so the streams run
    side by side and each plans nine sliding windows.
    """
    operations = seam_streams(stream_count)
    links = seam_only_fabric()
    rounds_policy = round_policies.FixedRounds(30)
    workload = workload_settings.WorkloadSettings(
        operations=operations,
        rounds_policy=rounds_policy,
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(
        distance=3, device=device, round_period_microseconds=1.0
    )
    engine_clock = config_module.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=engine_clock)
    weak_decoder = decoder_settings.DecoderSettings(
        kind=1.0,
        units=8,
        engine=engine,
    )
    manager = decoder_settings.DecoderManagerSettings(dispatch_cycles=0)
    windows = window_settings.WindowSettings(
        kind="sliding", terminal_policy="flush"
    )
    # a 1 GHz frame beside the 1 GHz decoder engine, so one write is the
    # 4 ns of Yang et al. 2605.04892 Fig. 1 and every correction reaches
    # the frame on one of its edges
    frame_clock = config_module.Clock(1000)
    frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=4, clock=frame_clock
    )
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        weak_decoder=weak_decoder,
        decoder_manager=manager,
        pauli_frame=frame,
        links=links,
    )
    task = collect.Task(settings, 1, {})
    return collect.run_shot(task, 0)


def test_a_windows_seam_delay_is_the_same_however_many_streams_run():
    """Two streams each have a window 3, and each paid one 0.5 us seam.

    dd_per_window is the decoder-to-decoder hop a window's own boundary
    rode. It is keyed by the window, which is its operation and its
    index, not by the index alone, which two streams share, so each of
    the eighteen windows reads the one card it crossed, and the last
    window of a stream reads nothing because no window follows it.
    """
    one_stream_run = seam_streams_shot(1)
    two_stream_run = seam_streams_shot(2)
    one_stream = measure.measure_shot(one_stream_run)
    two_streams = measure.measure_shot(two_stream_run)

    one_stream_seams = [0.5] * 8 + [0.0]
    assert one_stream.samples["dd_per_window"] == one_stream_seams
    assert two_streams.samples["dd_per_window"] == one_stream_seams * 2
    assert two_streams.load == one_stream.load


def test_throughput_counts_the_rounds_the_shot_read_out():
    """Two streams of thirty rounds each are sixty rounds, whatever the row.

    The rounds come from the QPU's EMITTED rows, one per operation round,
    so a Python-built workload with no rounds_per_shot key is measured on
    what it ran.
    """
    two_stream_run = seam_streams_shot(2)
    two_streams = measure.measure_shot(two_stream_run)
    decoded_windows = len(two_streams.samples["service"])

    rounds_per_window = (
        two_streams.throughput_rounds_per_us
        / two_streams.throughput_windows_per_us
    )

    assert decoded_windows == 18
    assert rounds_per_window * decoded_windows == pytest.approx(60)


def test_every_streams_window_is_measured_against_its_own_frame_record():
    """Two streams commit eighteen corrections and none is dropped.

    The frame writes one correction per window and a window is its
    operation and its index, so the two streams' nine windows each
    commit their own. Filed by the index alone, the second stream's
    record of every index replaced the first stream's and nine of the
    eighteen were lost, so the windows that lost theirs were measured
    against the other stream's decode. Filed by the whole key, every
    window reads the record it wrote, and each stream's frame commit is
    the one-stream run's 0.004 us on all nine of its windows.
    """
    one_stream_run = seam_streams_shot(1)
    two_stream_run = seam_streams_shot(2)
    observation = two_stream_run.machine.observation
    committed = observation.frame_corrections.committed
    frames = measure.frame_records_by_window(observation)
    one_stream = measure.measure_shot(one_stream_run)
    two_streams = measure.measure_shot(two_stream_run)

    filed_keys = sorted(frames)
    written_keys = sorted(record.window_key for record in committed)
    assert len(committed) == 18
    assert filed_keys == written_keys
    assert filed_keys[:9] == [(1, index) for index in range(9)]
    assert filed_keys[9:] == [(2, index) for index in range(9)]
    one_stream_commits = one_stream.samples["frame_commit"]
    assert one_stream_commits == [0.004] * 9
    assert two_streams.samples["frame_commit"] == one_stream_commits * 2


@pytest.mark.parametrize(
    "counter_name, metric_name, scale",
    [
        ("transfer_count", "transfers", 1),
        ("known_payload_bits", "payload_bits", 1),
        ("unknown_payload_transfer_count", "unknown_payload_transfers", 1),
        ("queue_wait_ticks", "queue_wait_us", 1_000_000),
        ("serialization_ticks", "serialization_us", 1_000_000),
        ("propagation_ticks", "propagation_us", 1_000_000),
    ],
)
def test_link_totals_sum_every_binding_of_one_semantic_path(
    counter_name: str, metric_name: str, scale: int
) -> None:
    first_value = 2 * scale
    second_value = 3 * scale
    first = link_traffic.TrafficCounters(**{counter_name: first_value})
    second = link_traffic.TrafficCounters(**{counter_name: second_value})
    first_counters = first.to_json_value()
    second_counters = second.to_json_value()
    traffic = {
        "semantic_edges": [
            {
                "path": "qpu_to_controller",
                "physical_alias": "left",
                "counters": first_counters,
            },
            {
                "path": "qpu_to_controller",
                "physical_alias": "right",
                "counters": second_counters,
            },
        ]
    }

    totals = measure.link_totals(traffic)

    assert totals["qpu_to_controller"][metric_name] == 5


def test_a_shot_fails_when_any_of_its_patches_reads_a_wrong_observable(
    tmp_path,
):
    """Two patches, a burst on the second: the first right, the shot wrong.

    The burst region, radius 4 about patch 1's centre (11, 3) on the
    patches' shared plane, misses patch 0 entirely; at seed 1 patch 1's
    observable comes out wrong.
    """
    raw = dict(MINIMAL_CONFIG)
    raw["workload"] = {
        "kind": "producer",
        "function": "decsim.producers:memory_patches",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": 6,
            "patch_count": 2,
            "distance": "${qpu.distance}",
        },
    }
    raw["qpu"] = {
        "kind": "burst_stim",
        "burst_radius": 4.0,
        "burst_center": [11.0, 3.0],
        "burst_error_probability": 0.3,
    }
    config_path = tmp_path / "two_patches.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)
    task = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
        2,
    )
    shot = collect.run_shot(task, 1)
    measurement = measure.measure_shot(shot)
    first, second = shot.result.operation_results

    assert first.logical_observables == first.observable_truth
    assert second.logical_observables != second.observable_truth
    assert measurement.logical_failure is True


LOAD_FLAGS = ("record_switching_windows", "backlog_trace")


def escalating_shot(tmp_path, strong_units: int):
    """Three patches' windows all escalate to 10 us strong cards.

    One patch's strong decodes wait on one another's boundaries and
    never queue; three patches' reach the strong units together.
    """
    shot = switching_run(
        tmp_path,
        1000000.0,
        strong_units=strong_units,
        observation=LOAD_FLAGS,
        patch_count=3,
    )
    return shot, measure.measure_shot(shot)


def lindley_waits_us(requests, stages) -> list:
    """One server's waits by Lindley's recursion, W' = max(0, W + S - A).

    Lindley 1952: a customer's service starts when it has arrived and the
    one before it has finished. The server is the strong unit's compute:
    a decode arrives at it an input hop after its enqueue, and is served
    for its compute span, both off the stage ledger, in arrival order.
    """
    strong = window_records.DecoderTier.STRONG
    strong_requests = [
        request for request in requests if request.request_key.tier is strong
    ]
    by_arrival = sorted(strong_requests, key=ready_ticks_of)
    waits = []
    free_at = 0
    for request in by_arrival:
        run_sequence = request.request_key.run_sequence
        records = [
            record
            for record in stages.records
            if run_sequence in record.run_sequences
        ]
        first = records[0]
        input_hop = first.ready_ticks - first.dispatch_ticks
        arrival = request.ready_ticks + input_hop
        starts = [record.start_ticks for record in records]
        ends = [record.end_ticks for record in records]
        duration = max(ends) - min(starts)
        start = max(arrival, free_at)
        wait_ticks = start - arrival
        wait = config_module.ticks_to_microseconds(wait_ticks)
        waits.append(wait)
        free_at = start + duration
    return waits


def ready_ticks_of(request) -> int:
    return request.ready_ticks


def test_a_strong_requests_wait_is_lindleys_one_server_wait(tmp_path):
    """Three patches' escalations into one strong unit queue as one server's."""
    shot, measurement = escalating_shot(tmp_path, 1)
    observation = shot.machine.observation
    requests = observation.decode_records.requests
    waits = lindley_waits_us(requests, observation.stages)

    assert measurement.strong_wait_max_us == pytest.approx(max(waits))
    mean_wait = sum(waits) / len(waits)
    assert measurement.strong_wait_mean_us == pytest.approx(mean_wait)
    assert measurement.strong_wait_max_us > 0


class StrongResidentsWaitingOnCompute:
    """Each tick's strong decodes held in a unit, off the units' own slots.

    A resident whose input landed, that is not parked on a boundary and
    has not started, is in the unit's memory waiting for its compute.
    The last sample of a tick is the depth the tick ends at.
    """

    def __init__(self, machine) -> None:
        self.units = []
        for manager in (
            machine.decoder_manager,
            machine.strong_decoder_manager,
        ):
            if manager is not None:
                pool_units = manager.service.pool.units()
                self.units.extend(pool_units)
        self.depth_by_tick = {}

    def sample(self, now: int) -> None:
        """Count the strong residents waiting for compute now."""
        waiting = 0
        for unit in self.units:
            waiting += _strong_residents_waiting(unit)
        self.depth_by_tick[now] = waiting


def _strong_residents_waiting(unit) -> int:
    waiting = 0
    for resident in unit.residents:
        if _is_strong_and_waiting(resident):
            waiting += 1
    return waiting


def _is_strong_and_waiting(resident) -> bool:
    """A strong decode landed, not parked, and not started."""
    strong = window_records.DecoderTier.STRONG
    if resident.request_key.tier is not strong:
        return False
    if resident.is_parked or resident.service_started:
        return False
    return resident.input_landed


def test_the_strong_decodes_held_in_units_are_the_units_own_residents(
    tmp_path, monkeypatch
):
    """A unit takes the next decode into its memory while it computes.

    The column reads the most strong decodes held at once off the stage
    ledger; the units' own slots, sampled after every engine action, give
    the same peak. One unit holds two residents, the decode it computes
    and the next one (decode_service.py, resident_capacity), so one is
    the most it can hold waiting.
    """
    samplers = []
    build = machine_module.Machine.build

    def build_and_sample(settings, seed):
        machine = build(settings, seed)
        sampler = StrongResidentsWaitingOnCompute(machine)
        machine.engine.action_done.connect(sampler.sample)
        samplers.append(sampler)
        return machine

    monkeypatch.setattr(machine_module.Machine, "build", build_and_sample)
    _, measurement = escalating_shot(tmp_path, 1)
    (sampler,) = samplers
    depths = sampler.depth_by_tick.values()

    assert measurement.strong_held_in_units_max == max(depths) == 1


def test_enough_strong_units_leave_the_same_windows_and_no_wait(tmp_path):
    """The paired run: the same seed, the real unit count and ten.

    The syndromes and the escalations are the same in both, so what the
    extra units remove, the strong wait and the rounds held, is the
    overload alone; what stays, the weak input's weight and service and
    the escalated windows, is the difficulty.
    """
    _, real = escalating_shot(tmp_path, 1)
    _, ample = escalating_shot(tmp_path, 10)

    assert ample.weak_syndrome_weight_mean == real.weak_syndrome_weight_mean
    assert ample.weak_syndrome_weight_max == real.weak_syndrome_weight_max
    assert ample.weak_service_mean_us == real.weak_service_mean_us
    assert ample.escalated_windows == real.escalated_windows == 30
    assert ample.strong_wait_max_us == 0.0
    assert real.strong_wait_max_us > 0.0
    assert ample.strong_held_in_units_max == 0
    assert ample.backlog_peak_rounds < real.backlog_peak_rounds


def test_a_shot_that_kept_no_records_writes_no_load_columns(tmp_path):
    """A column of zeros would say nothing waited; the shot writes none."""
    bare = switching_shot(tmp_path, 1000000.0)
    _, kept = escalating_shot(tmp_path, 1)
    bare_record = report.record_of([bare])
    kept_record = report.record_of([kept])
    bare_rows = report.summarize(bare_record.shots, bare_record.window_samples)
    kept_rows = report.summarize(kept_record.shots, kept_record.window_samples)

    assert "strong_wait_max_us" not in bare_record.shots[0]
    assert "escalated_fraction" not in bare_rows[0]
    assert kept_record.shots[0]["strong_wait_max_us"] > 0
    assert kept_rows[0]["escalated_fraction"] == 1.0
    assert kept_rows[0]["backlog_peak_rounds"] == kept.backlog_peak_rounds


def test_a_source_that_samples_no_shot_is_refused_with_a_sentence(tmp_path):
    """timing_only draws no shot, so the loop has no truth to be judged by."""
    raw = dict(MINIMAL_CONFIG)
    raw["qpu"] = {"kind": "timing_only"}
    config_path = tmp_path / "timing_only.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)

    with pytest.raises(
        refusal.RefusalError, match="sampled none for operation"
    ):
        measure_point_shot(
            config,
            physical_error_probability=0.001,
            distance=3,
            round_period_microseconds=1.0,
            seed=0,
        )


# A whole-patch burst from round 12 on the 30-round switching shot; the
# masked regional CUSUM, at its per-second budget over a 30 us shot,
# first fires on round 21 at seed 0, nine rounds after the onset
BURST_ONSET_ROUND = 12
FIRST_FLAG_ROUND = 21


def burst_detector_shot(
    tmp_path, burst_error_probability: float, catch_deadline_rounds=300
):
    """One switching shot with the masked regional CUSUM watching it."""
    qpu = {
        "kind": "burst_stim",
        "burst_onset_round": BURST_ONSET_ROUND,
        "burst_error_probability": burst_error_probability,
    }
    return detector_shot(tmp_path, qpu, catch_deadline_rounds)


def detector_shot(tmp_path, qpu: dict, catch_deadline_rounds=300):
    """That shot on any qpu section."""
    detector = {
        "kind": "masked_regional_cusum",
        "catch_deadline_rounds": catch_deadline_rounds,
    }
    sections = {"qpu": qpu, "burst_detector": detector}
    shot = switching_run(tmp_path, 0.0, sections=sections)
    return measure.measure_shot(shot)


def test_a_burst_shot_records_its_first_flag_and_a_catch_in_time(tmp_path):
    """Delay 9 is inside a 9-round deadline and outside an 8-round one.

    A catch within k is a detection delay of at most k, so the deadline
    is inclusive.
    """
    in_time = burst_detector_shot(tmp_path, 0.05, catch_deadline_rounds=9)
    late = burst_detector_shot(tmp_path, 0.05, catch_deadline_rounds=8)

    assert in_time.burst_first_flag_round == FIRST_FLAG_ROUND
    assert in_time.burst_caught_in_time is True
    assert late.burst_first_flag_round == FIRST_FLAG_ROUND
    assert late.burst_caught_in_time is False


@pytest.mark.parametrize(
    "qpu",
    [
        {"kind": "burst_stim", "burst_error_probability": 0.0},
        {"kind": "stim_device"},
    ],
)
def test_a_shot_with_no_burst_records_no_flag_and_no_catch(tmp_path, qpu):
    """A burst of probability 0 draws the operation's own circuit."""
    quiet = detector_shot(tmp_path, qpu)

    assert quiet.burst_first_flag_round == 0
    assert quiet.burst_caught_in_time is None


def test_the_point_holds_the_shares_flagged_and_caught_in_time(tmp_path):
    """Two shots, one caught in time, folded as one point's shots.

    The two deadlines are two settings, so two points; the late shot
    takes the caught shot's point id to be summed with it.
    """
    caught = burst_detector_shot(tmp_path, 0.05, catch_deadline_rounds=9)
    late_alone = burst_detector_shot(tmp_path, 0.05, catch_deadline_rounds=8)
    late = dataclasses.replace(late_alone, point_id=caught.point_id)
    quiet = burst_detector_shot(tmp_path, 0.0)
    burst_record = report.record_of([caught, late])
    quiet_record = report.record_of([quiet])

    burst_rows = report.summarize(burst_record.shots, [])
    quiet_rows = report.summarize(quiet_record.shots, [])

    assert burst_rows[0]["flagged_share"] == 1.0
    assert burst_rows[0]["caught_in_time_share"] == 0.5
    assert quiet_rows[0]["flagged_share"] == 0.0
    assert "caught_in_time_share" not in quiet_rows[0]


def test_a_run_without_a_detector_writes_no_burst_column(tmp_path):
    measurement = switching_shot(tmp_path, 1000000.0)
    record = report.record_of([measurement])
    rows = report.summarize(record.shots, record.window_samples)

    assert "burst_first_flag_round" not in record.shots[0]
    assert "burst_caught_in_time" not in record.shots[0]
    assert "flagged_share" not in rows[0]
    assert "caught_in_time_share" not in rows[0]


def recorded_relay_statuses(monkeypatch) -> list:
    """The statuses the Relay-BP row's decode_window returns, in order."""
    row = relay_decoder.RelayBeliefPropagationDecoder
    decode_window = row.decode_window
    returned = []

    def recording_decode_window(self, backend, model, faults, syndrome):
        answer = decode_window(self, backend, model, faults, syndrome)
        returned.append(answer.decode_status)
        return answer

    monkeypatch.setattr(row, "decode_window", recording_decode_window)
    return returned


def test_the_status_columns_count_the_statuses_the_decoder_returned(
    tmp_path, monkeypatch
):
    """One Relay-BP iteration leaves windows unconverged, and each counts.

    relay-bp's detailed API reports a decode that did not converge
    (Maurer et al. 2510.21600), and the row commits it as NONCONVERGED.
    The referent is the run itself: the statuses the row's decode_window
    returned, one per window, since a weak_baseline run decodes each
    window once.
    """
    pytest.importorskip("relay_bp")
    returned = recorded_relay_statuses(monkeypatch)
    weak_decoder = dict(
        MINIMAL_CONFIG["weak_decoder"],
        kind="relay_bp",
        pre_iterations=1,
        relay_set_count=0,
        iterations_per_set=1,
    )
    config_path = write_config(tmp_path, {"weak_decoder": weak_decoder})
    config = experiment.load_experiment(config_path)
    measurement = measure_point_shot(
        config,
        physical_error_probability=0.003,
        distance=3,
        round_period_microseconds=1.0,
        seed=0,
    )
    record = report.record_of([measurement])
    rows = report.summarize(record.shots, record.window_samples)
    expected = collections.Counter(
        f"{status.value}_windows" for status in returned if status is not None
    )
    statuses = measurement.window_statuses
    counted = {column: count for column, count in statuses.items() if count}
    nonconverged = counted["nonconverged_windows"]

    assert expected["nonconverged_windows"] > 0
    assert counted == expected
    assert record.shots[0]["nonconverged_windows"] == nonconverged
    assert rows[0]["nonconverged_windows"] == nonconverged


def decoder_row_shot(tmp_path, kind: str, seed: int):
    """One seeded shot of the minimal machine with its weak row named."""
    weak_decoder = dict(MINIMAL_CONFIG["weak_decoder"], kind=kind)
    config_path = write_config(tmp_path, {"weak_decoder": weak_decoder})
    config = experiment.load_experiment(config_path)
    return measure_point_shot(
        config,
        physical_error_probability=0.01,
        distance=3,
        round_period_microseconds=1.0,
        seed=seed,
    )


def test_the_sample_digest_names_the_draw_and_not_the_decoder(tmp_path):
    """Two decoder rows at one seed share a digest; two seeds do not.

    The device draws a shot from the run's seed alone, so PyMatching and
    Union-Find see the same detection events and truth at seed 0, and
    seed 1 is another draw.
    """
    matching = decoder_row_shot(tmp_path, "pymatching", 0)
    union_find = decoder_row_shot(tmp_path, "union_find", 0)
    next_seed = decoder_row_shot(tmp_path, "pymatching", 1)

    assert matching.algorithm != union_find.algorithm
    assert matching.sample_digest == union_find.sample_digest
    assert matching.sample_digest != next_seed.sample_digest


WEAK = window_records.DecoderTier.WEAK
STRONG = window_records.DecoderTier.STRONG
# more windows than any escalation shot here plans, so a set of every
# index names every window of a tier
EVERY_WINDOW = range(64)


def no_correction_answer(self, job, backend, model, faults, syndrome):
    """A backend that raised: no correction, the row's empty stand-in."""
    del self, job, backend, model, syndrome
    fault_count = faults.check.shape[1]
    return backend_outcome.no_correction_decode(
        decoding_records.BackendDecodeStatus.BACKEND_ERROR,
        decoding_records.BackendFailureReason.UPSTREAM_EXCEPTION,
        fault_count,
    )


def crashing_decodes(monkeypatch, crashing: set) -> dict:
    """Each decode named in crashing, (tier, window), answers no correction.

    Returns the syndrome every decode read, by (tier, window), the last
    solve's when a window is solved more than once; crashing may change
    between runs of one test.
    """
    decoder = decoder_module.WindowDecoderBase
    real_answer = decoder.window_answer
    syndromes = {}

    def answer(self, job, backend, model, faults, syndrome):
        key = (job.request_key.tier, job.request_key.window_id)
        syndromes[key] = syndrome.tolist()
        answers = dict.fromkeys(crashing, no_correction_answer)
        chosen = answers.get(key, real_answer)
        return chosen(self, job, backend, model, faults, syndrome)

    monkeypatch.setattr(decoder, "window_answer", answer)
    return syndromes


def test_a_strong_decode_with_no_correction_unscores_its_shot(
    tmp_path, monkeypatch
):
    """The strong answer is the window's final one, and so is its status.

    Every window escalates and every strong decode produces nothing, so
    each window's final decode is the strong one with no correction.
    """
    strong_windows = {(STRONG, index) for index in EVERY_WINDOW}
    crashing_decodes(monkeypatch, strong_windows)
    shot = switching_run(tmp_path, 1000000.0)
    measurement = measure.measure_shot(shot)
    statuses = measurement.window_statuses

    assert measurement.escalated_windows == measurement.windows
    assert measurement.is_scored is False
    assert measurement.unscored_reason == "upstream_exception"
    assert statuses["backend_error_windows"] == measurement.windows
    assert measurement.provisional_no_correction_windows == 0


def test_a_weak_decode_with_no_correction_unscores_its_shot_after_strong(
    tmp_path, monkeypatch
):
    """The replaced weak answer counts apart and still unscores the shot.

    The strong decodes all succeed, so no window's final status is an
    error; each weak answer was committed provisionally first, and what
    a provisional commit fed forward survives the strong answer.
    """
    weak_windows = {(WEAK, index) for index in EVERY_WINDOW}
    crashing_decodes(monkeypatch, weak_windows)
    shot = switching_run(tmp_path, 1000000.0)
    measurement = measure.measure_shot(shot)
    statuses = measurement.window_statuses

    assert measurement.escalated_windows == measurement.windows
    assert measurement.is_scored is False
    assert measurement.unscored_reason == "upstream_exception"
    assert statuses["backend_error_windows"] == 0
    assert measurement.provisional_no_correction_windows == (
        measurement.windows
    )


def test_a_strong_window_with_no_correction_moves_the_next_ones_input(
    tmp_path, monkeypatch
):
    """Window 0's empty strong answer reaches window 1, whose decode is fine.

    The held boundary ships from the strong result (window_boundaries.py
    ship_held), so window 1 reads a different syndrome than it does when
    window 0's strong decode answers, and the shot is unscored by window
    0 alone.
    """
    crashing = set()
    syndromes = crashing_decodes(monkeypatch, crashing)
    switching_run(tmp_path, 1000000.0)
    answered_input = syndromes[(WEAK, 1)]
    crashing.add((STRONG, 0))
    shot = switching_run(tmp_path, 1000000.0)
    measurement = measure.measure_shot(shot)
    crashed_input = syndromes[(WEAK, 1)]

    assert crashed_input != answered_input
    assert measurement.is_scored is False
    assert measurement.window_statuses["backend_error_windows"] == 1
