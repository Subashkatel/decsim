"""The latency points of one shot, against a hand derivation of its run.

Every number here is arithmetic over the declared cards of the configs
below: a 250 MHz fabric, a weak unit whose fetch is one cycle per round
and whose release is ten cycles (0.040 us), 30 one-microsecond rounds at
distance 3, and sliding windows that commit 3 rounds and buffer 3. The
chain is the one Skoric et al. 2209.08552 describe: each window's decode
takes its own time (tau_W, lines 429-435) and waits for the seam before
it starts (the artificial defects of the block before it, lines
268-275). The two must not be one number, so the tests read service and
dep_block apart and check that the points of one window still sum to
the window's whole reaction time.

The two-tier config gives every hop of the strong path a card of its
own, so a point's value names the wire it was read from: the strong
store's hop is two cycles, the escalation hop five and the strong
decoder's way home three, against one cycle everywhere on the weak
path. Escalating every window or none is the threshold's doing, which
is Toshio et al. 2510.25222 Sec. III A step 3 driven to its two ends.
"""

import yaml

import decsim.front.experiment as experiment
from tests.front.yaml_configs import MINIMAL_CONFIG, measure_point_shot

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
# in Buffer 0 and its correction being committed in the frame
CHAIN = (
    "queue_wait",
    "weak_attempt",
    "input_link_per_window",
    "dep_block",
    "service",
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


def slow_unit_shot(tmp_path, units: int):
    """One shot of 30 rounds on `units` five-microsecond weak units."""
    raw = dict(MINIMAL_CONFIG)
    workload = dict(MINIMAL_CONFIG["workload"])
    workload["rounds_per_shot"] = 30
    raw["workload"] = workload
    raw["links"] = ONE_TIER_LINKS
    raw["weak_decoder"] = {
        "kind": 5.0,
        "units": units,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 10,
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
        round_period_us=1.0,
        seed=0,
    )


def switching_shot(tmp_path, gap_threshold_db: float):
    """One shot of a 1.0 us weak tier that escalates to a 10.0 us one.

    A threshold far above every gap escalates every window and one far
    below escalates none, so the same two cards answer both branches.
    """
    raw = dict(MINIMAL_CONFIG)
    workload = dict(MINIMAL_CONFIG["workload"])
    workload["rounds_per_shot"] = 30
    raw["workload"] = workload
    raw["links"] = TWO_TIER_LINKS
    raw["escalation"] = {
        "kind": "switching",
        "gap_threshold_db": gap_threshold_db,
        "strong_window": "two_sided_context",
    }
    raw["weak_decoder"] = {
        "kind": 1.0,
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 10,
        },
    }
    raw["strong_decoder"] = {
        "kind": 10.0,
        "units": 1,
        "unit_memory_rounds": None,
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "release_cycles_per_job": 10,
        },
    }
    config_path = tmp_path / "switching.yaml"
    config_text = yaml.safe_dump(raw)
    config_path.write_text(config_text)
    config = experiment.load_experiment(config_path)
    return measure_point_shot(
        config,
        physical_error_probability=0.008,
        distance=3,
        round_period_us=1.0,
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
    boundary 0.004 us after its own decode ends.
    """
    measurement = slow_unit_shot(tmp_path, 1)

    samples = measurement.samples
    assert samples["service"] == [5.064] * 9
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
    in Buffer 0 to its correction committed in the frame, so they add up
    to buffer0_ready_to_frame on every window to the tick.
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


def test_an_escalated_window_is_measured_on_the_strong_tiers_own_hops(
    tmp_path,
):
    """Every window escalates: the points are the strong decode's.

    The result the frame committed crossed the strong store's hop in
    (0.008 us) and the strong decoder's hop home (0.012 us), and the
    decode it describes is the 10.0 us one. The weak decode that did not
    commit is the weak_attempt point, and the escalation hop that
    carried the selection to the strong tier (0.020 us) is its own
    point: it runs beside the strong input hop, so what it costs the
    decode is the 0.012 us the input waited for it, which is dep_block.
    """
    measurement = switching_shot(tmp_path, 1000000.0)

    samples = measurement.samples
    assert samples["input_link_per_window"] == [0.008] * 10
    assert samples["output_link_per_window"] == [0.012] * 10
    assert samples["escalation_link_per_window"] == [0.020] * 10
    assert samples["dep_block"] == [0.012] * 10
    assert samples["algorithm"] == [10.0] * 10
    assert samples["weak_attempt"][2] == 12.244


def test_an_escalated_windows_points_sum_to_its_reaction_time(tmp_path):
    """The chain runs weak attempt first, then the strong decode.

    Toshio et al. 2510.25222 Sec. III A orders it: the weak decoder
    answers, the verdict sends the window to the strong decoder, the
    strong decoder answers and that is what commits. Every point on that
    order adds up to the window's reaction time to the tick.
    """
    measurement = switching_shot(tmp_path, 1000000.0)

    totals = chain_sum_ticks(measurement)
    assert totals == reaction_ticks(measurement)


def test_a_kept_weak_result_is_measured_on_the_weak_hops(tmp_path):
    """No window escalates: the weak hops, and no escalation at all.

    The same config below the threshold keeps every weak result, so both
    link points read the weak path's one cycle and the two escalation
    points are zero on every window.
    """
    measurement = switching_shot(tmp_path, -1000000.0)

    samples = measurement.samples
    assert samples["input_link_per_window"] == [0.004] * 10
    assert samples["output_link_per_window"] == [0.004] * 10
    assert samples["escalation_link_per_window"] == [0.0] * 10
    assert samples["weak_attempt"] == [0.0] * 10
    assert samples["algorithm"] == [1.0] * 10
