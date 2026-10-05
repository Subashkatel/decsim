"""The latency points of one shot, against a hand derivation of its run.

Every number here is arithmetic over the declared cards of the machines
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

The two-tier machine gives every hop of the strong path a card of its
own, so a point's value names the wire it was read from: the strong
store's hop is two cycles, the escalation hop five and the strong
decoder's way home three, against one cycle everywhere on the weak
path. Escalating every window or none is the threshold's doing, which
is Toshio et al. 2510.25222 Sec. III A step 3 driven to its two ends.

The last run here is built from its operations: memory_circuit is one
operation for the whole shot, so the several-stream workload that tells
a per-window point apart from a per-index one is a list of streams, on
a fabric whose only priced hop is the decoder-to-decoder seam.
"""

import ast
import collections
import dataclasses
import functools
import math
import pathlib
import sys
from typing import Optional

import numpy
import pytest

import decsim.confidence.complementary as complementary
import decsim.config as config_module
import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.belief_propagation_osd.decoder as belief_propagation_osd
import decsim.decoders.decoder as decoder_module
import decsim.decoders.dispatch_steps.decoder as dispatch_steps
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.relay_belief_propagation.decoder as relay_decoder
import decsim.decoders.settings as decoder_settings
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.verify_windows as verify_windows
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import decsim.observe.decode_records as decode_records
import decsim.observe.link_traffic as link_traffic
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.producers as producers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings
import examples.live_memory_example as live_memory_example
import tests.declared_run as declared_run
import tests.experiments.run_files as run_files
import tests.qpu.memory_programs as memory_programs
from decsim.decoders.measured_table import decoder as measured_table_decoder
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

THIS_FILE = pathlib.Path(__file__)
THIS_PATH = THIS_FILE.resolve()
REPOSITORY = THIS_PATH.parents[2]
# the detection events of one round of the swept distance-three patch,
# and of its readout round, which compares the data qubits as well
BITS_PER_ROUND = 8
READOUT_ROUND_BITS = 12
# every hop of the weak-only fabric, in cycles of the fridge clock
ONE_TIER_HOPS = {
    "qpu_to_controller": 1,
    "controller_to_weak_buffer": 1,
    "weak_buffer_to_weak_decoder": 1,
    "decoder_to_decoder": 1,
    "weak_decoder_to_frame": 1,
}
# the two-tier fabric, every strong hop on a card of its own
TWO_TIER_HOPS = {
    "qpu_to_controller": 1,
    "controller_to_weak_buffer": 1,
    "controller_to_strong_buffer": 1,
    "weak_buffer_to_weak_decoder": 1,
    "strong_buffer_to_strong_decoder": 2,
    "weak_decoder_to_strong_decoder": 5,
    "decoder_to_decoder": 1,
    "weak_decoder_to_frame": 1,
    "strong_decoder_to_frame": 3,
}
HOPS_SOURCE = "a measurement test's hops, in fridge cycles"
FRIDGE_CLOCK = machine_settings.FRIDGE_CLOCK
# the task cells every hand derivation here is at, and the switching runs'
TASK_CELLS = {
    run_files.ERROR_RATE_PATH: 0.001,
    run_files.DISTANCE_PATH: 3,
    run_files.ROUND_PERIOD_PATH: 1.0,
}
SWITCHING_ERROR_PROBABILITY = 0.008
SHOT_ROUNDS = 30

# the points that lie end to end between a window's data being complete
# in the weak syndrome buffer and its correction being committed in the frame
CHAIN = (
    "admission_wait",
    "queue_wait",
    "weak_attempt",
    "input_link_per_window",
    "store_read",
    "dep_block",
    "compute_wait",
    "service",
    "confidence",
    "selection_wait",
    "output_link_per_window",
    "frame_commit",
)


def fridge_hops(hops: dict, setup_cycles: Optional[dict] = None):
    """The logical reference card, each named hop that many fridge cycles.

    Every named hop is an unbounded wire; setup_cycles gives a hop a
    per-transfer setup in the same cycles. A hop not named keeps the
    reference card's numbers.
    """
    setup_cycles = setup_cycles or {}
    links = link_profiles.logical_reference_profile()
    cards = {}
    for path_name, latency_cycles in hops.items():
        path_setup_cycles = setup_cycles.get(path_name, 0)
        cards[path_name] = link_profiles.path_card(
            links,
            path_name,
            clock=FRIDGE_CLOCK,
            latency_cycles=latency_cycles,
            bits_per_cycle=None,
            source=HOPS_SOURCE,
            setup_cycles_per_transfer=path_setup_cycles,
        )
    return dataclasses.replace(links, **cards)


def charged(microseconds: float):
    """PyMatching decoding for real, each decode charged that long."""
    return run_files.PYMATCHING.Settings(
        preset_latency_microseconds=microseconds
    )


def ten_cycle_release_pool(algorithm, unit_count: int = 1):
    """Units on the fridge clock: fetch a cycle a round, release ten a job."""
    pool = run_files.decoder_pool(algorithm, FRIDGE_CLOCK)
    engine = dataclasses.replace(pool.engine, release_cycles_per_job=10)
    return dataclasses.replace(pool, unit_count=unit_count, engine=engine)


def cells_at(probability: float) -> dict:
    """TASK_CELLS at another physical error probability."""
    cells = dict(TASK_CELLS)
    cells[run_files.ERROR_RATE_PATH] = probability
    return cells


def task_at(
    settings: machine_settings.MachineSettings, probability: float = 0.001
) -> collect.Task:
    """The task of the settings at TASK_CELLS, at that error probability.

    The probability is the one the settings' circuit was built at.
    """
    cells = cells_at(probability)
    return collect.Task("task", settings, cells)


def measured(task: collect.Task, seed: int = 0) -> measure.ShotMeasurement:
    """One seeded shot of the task, run and measured."""
    shot = collect.run_shot(task, seed)
    return measure.measure_shot(shot)


def one_tier_machine(
    weak_microseconds: float,
    units: int = 1,
    setup_cycles: Optional[dict] = None,
) -> machine_settings.MachineSettings:
    """30 rounds on the weak-only fabric, its units charged that long."""
    base = run_files.minimal_machine(TASK_CELLS, rounds_per_shot=SHOT_ROUNDS)
    links = fridge_hops(ONE_TIER_HOPS, setup_cycles)
    algorithm = charged(weak_microseconds)
    weak_decoder = ten_cycle_release_pool(algorithm, units)
    return dataclasses.replace(base, links=links, weak_decoder=weak_decoder)


def slow_unit_settings(
    units: int,
    card_microseconds: float = 5.0,
    decision_cycles: int = 0,
) -> machine_settings.MachineSettings:
    """One shot of 30 rounds on `units` weak units of that card.

    decision_cycles is the window side's decision before each request,
    on the fridge clock.
    """
    base = one_tier_machine(card_microseconds, units)
    windows = dataclasses.replace(
        base.windows, decision_cycles=decision_cycles, clock=FRIDGE_CLOCK
    )
    return dataclasses.replace(base, windows=windows)


def slow_unit_shot(units: int, **arguments) -> measure.ShotMeasurement:
    """That machine's shot at seed 0, measured."""
    settings = slow_unit_settings(units, **arguments)
    task = task_at(settings)
    return measured(task)


def switching_settings(
    gap_threshold_db: float,
    run_both_at_once: bool = False,
    strong_units: int = 1,
    observation: tuple = (),
    patch_count: int = 1,
    commit_rounds: Optional[int] = None,
    weak_microseconds: float = 1.0,
    strong_algorithm=None,
    strong_window=None,
) -> machine_settings.MachineSettings:
    """A 1.0 us weak tier beside a 10.0 us one, on the two-tier fabric.

    A threshold far above every gap escalates every window and one far
    below escalates none, so the same two cards answer both branches.
    run_both_at_once starts the speculative strong decode with the weak job and
    cancels it when the weak result is kept, which is Toshio et al.
    2510.25222 Sec. III A step 1. observation names the flags the shot
    turns on; more than one patch runs memory_patches; commit_rounds
    sets the sliding windows' commit, the code distance when None.
    weak_microseconds, strong_algorithm and strong_window replace the
    weak card, the strong row and the redo window.
    """
    cells = cells_at(SWITCHING_ERROR_PROBABILITY)
    base = run_files.minimal_machine(cells, rounds_per_shot=SHOT_ROUNDS)
    workload = base.workload
    if patch_count > 1:
        patches = producers.memory_patches(
            run_files.CODE_TASK,
            SHOT_ROUNDS,
            patch_count,
            3,
            SWITCHING_ERROR_PROBABILITY,
        )
        workload = workload_settings.WorkloadSettings.running(patches)
    windows = base.windows
    if commit_rounds is not None:
        scheme = dataclasses.replace(
            windows.scheme, commit_rounds=commit_rounds
        )
        windows = dataclasses.replace(windows, scheme=scheme)
    strong_window = strong_window or strong_window_shapes.RedoWindow.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=gap_threshold_db
    )
    confidence = complementary.ComplementaryGap.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence,
        threshold=threshold,
        run_both_at_once=run_both_at_once,
        strong_window=strong_window,
    )
    windows = window_settings.switching_windows(windows, strong_window)
    weak_algorithm = charged(weak_microseconds)
    weak_decoder = ten_cycle_release_pool(weak_algorithm)
    strong_algorithm = strong_algorithm or charged(10.0)
    strong_decoder = ten_cycle_release_pool(strong_algorithm, strong_units)
    flags = dict.fromkeys(observation, True)
    watched = dataclasses.replace(base.observation, **flags)
    links = fridge_hops(TWO_TIER_HOPS)
    return dataclasses.replace(
        base,
        links=links,
        workload=workload,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
        observation=watched,
    )


def switching_run(
    gap_threshold_db: float, seed: int = 0, **arguments
) -> collect.Shot:
    """One collected shot of switching_settings at p = 0.008."""
    settings = switching_settings(gap_threshold_db, **arguments)
    task = task_at(settings, SWITCHING_ERROR_PROBABILITY)
    return collect.run_shot(task, seed)


def switching_shot(
    gap_threshold_db: float, run_both_at_once: bool = False, seed: int = 0
) -> measure.ShotMeasurement:
    """That shot's measurement."""
    shot = switching_run(
        gap_threshold_db, seed, run_both_at_once=run_both_at_once
    )
    return measure.measure_shot(shot)


def bounded_store_settings() -> machine_settings.MachineSettings:
    """A weak syndrome buffer of six rounds, with a slow unit.

    Thirty rounds arrive a microsecond apart into a store of six
    rounds, which is five ordinary rounds and the wider readout round,
    and the 5.0 us unit frees three slots per window it reads, so the
    controller has to hold rounds it has already packed.
    """
    base = one_tier_machine(5.0)
    six_rounds_bits = 5 * BITS_PER_ROUND + READOUT_ROUND_BITS
    store = dataclasses.replace(base.weak_syndrome_buffer, bits=six_rounds_bits)
    return dataclasses.replace(base, weak_syndrome_buffer=store)


def bounded_store_shot() -> measure.ShotMeasurement:
    """That machine's shot at seed 0, measured."""
    settings = bounded_store_settings()
    task = task_at(settings)
    return measured(task)


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


def test_service_is_the_compute_and_the_park_is_its_own_point():
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
    measurement = slow_unit_shot(1)

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


def test_one_windows_points_sum_to_its_reaction_time():
    """Queue wait, input link, park, service, output link and commit.

    Those six are the whole path from the window's data being complete
    in the weak syndrome buffer to its correction committed in the frame, so
    they add up to buffer0_ready_to_frame on every window to the tick.
    """
    measurement = slow_unit_shot(1)

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


def test_a_windows_decision_time_is_its_own_point():
    """1000 cycles of the 250 MHz fridge clock: 4 us before every queue.

    windows.decision_cycles holds each window's requests until its clock
    edge before they enter the decode queue (windows/decode_requests.py
    request), so every window's admission_wait is 4.0 us and the
    points still add up to its reaction time. Ciw keeps a customer's
    pre-service wait as a field of its own record (ciw/data_record.py
    lines 3-21), and so does this span.
    """
    measurement = slow_unit_shot(1, decision_cycles=1000)

    assert measurement.samples["admission_wait"] == [4.0] * 9
    assert chain_sum_ticks(measurement) == reaction_ticks(measurement)


def test_a_strong_answer_held_for_its_selection_waits_in_its_own_point():
    """A speculative strong decode that ends first waits for its selection.

    Under run_both_at_once the strong decode starts with the weak one
    (Toshio et al. 2510.25222 lines 598-601) and its answer waits in its
    unit's output slot, as a sender keeps a packet until the far side
    takes it (gem5 port.hh:244-255), until the verdict's selection has
    crossed weak_decoder_to_strong_decoder. That crossing is
    selection_wait, and the points add up to every window's reaction.
    A 5.0 us weak card that runs both forced-class solves on one unit
    answers after the 10.0 us strong decode it started with.
    """
    shot = switching_run(20.0, run_both_at_once=True, weak_microseconds=5.0)
    measurement = measure.measure_shot(shot)

    assert measurement.samples["selection_wait"][0] == 0.02
    assert chain_sum_ticks(measurement) == reaction_ticks(measurement)


def test_a_gpu_decode_that_finds_the_dispatcher_busy_waits_in_its_own_point():
    """Two patches share one GPU's dispatcher, one decode on it at a time.

    Every window escalates to the measured_table row, whose decode holds
    the one dispatcher (decoders/measured_table). Patch 1's window 0
    finds it free and waits nothing; patch 2's window 0 arrives on the
    other strong unit at 10.320 us and waits until patch 1's ends at
    175.007 us, 164.687 us. The algorithm point holds that wait and the
    device's own time after it.
    """
    pytest.importorskip("relay_bp")
    measured_table = measured_table_decoder.MeasuredTableDecoder.Settings()
    shot = switching_run(
        1000000.0,
        patch_count=2,
        strong_algorithm=measured_table,
        strong_units=2,
    )
    measurement = measure.measure_shot(shot)
    waits = measurement.samples["backend_queue_wait"]
    algorithm = measurement.samples["algorithm"]

    assert waits[0] == 0.0
    assert waits[10] == 164.687
    assert algorithm[10] == 272.93


def test_a_withdrawn_request_waits_in_the_admission_point():
    """A restart window's first request is withdrawn and asked for again.

    double_window takes back the restart window's queued request
    when it re-slices it (escalation/strong_window_shapes.py
    _withdraw_stale_requests), and the fresh request enters the queue
    later while the window's data was complete all along. The time
    between is the window's admission_wait, as Ciw writes a customer
    that left the queue unserved its own record (ciw/node.py
    write_reneging_record) rather than folding it into the next wait,
    and window 3 of this seed spends 1.132 us there.
    """
    double_window = strong_window_shapes.DoubleWindow.Settings()
    shot = switching_run(
        20.0, weak_microseconds=5.0, strong_window=double_window
    )
    measurement = measure.measure_shot(shot)

    assert measurement.samples["admission_wait"] == [0.0, 1.132] + [0.0] * 5
    assert chain_sum_ticks(measurement) == reaction_ticks(measurement)


def test_load_is_the_units_occupancy_over_the_window_period():
    """Load is the compute plus the seam handoff over the period.

    Three one-microsecond rounds commit per window, so a 5.064 us
    compute plus the 0.004 us seam handoff on eight of the nine windows
    is a chain 1.689 times too slow, which is the backlog condition of
    Skoric et al. 2209.08552 lines 429-435 read as a ratio.
    """
    measurement = slow_unit_shot(1)

    handoff_us = 8 * 0.004 / 9
    expected = (5.064 + handoff_us) / 3.0
    assert measurement.load == expected


def test_a_second_unit_moves_the_wait_and_leaves_service_alone():
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
    one_unit = slow_unit_shot(1)
    two_units = slow_unit_shot(2)

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


def test_the_pool_columns_read_each_tiers_own_queue_and_units():
    """A weak-only run's pool columns are the default pool's, the strong zero.

    The deepest the weak tier's own queue got is the deepest the ready
    queue got, since only that pool exists; a second unit halves the
    busy fraction exactly, because the same nine decodes of 5.064 us
    each run on two units over the same span
    (test_a_second_unit_moves_the_wait_and_leaves_service_alone), which
    is Triage's utilization rate read per tier (2605.04459 lines
    1024-1031).
    """
    one_unit = slow_unit_shot(1)
    two_units = slow_unit_shot(2)

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
    them (build/decoders.py), and under strong_only that
    tier is the strong one, so its queue peak and busy fraction belong
    in the strong columns and the weak columns read zero, the mirror of
    test_the_pool_columns_read_each_tiers_own_queue_and_units.
    """
    shot = base_shot(machine_settings.strong_decoder_baseline)
    measurement = measure.measure_shot(shot)

    assert measurement.weak_queue_max == 0
    assert measurement.weak_busy_fraction == 0.0
    assert measurement.strong_queue_max == measurement.max_queued_windows
    assert measurement.strong_busy_fraction > 0.0


def test_patches_named_by_an_int_and_a_str_are_measured():
    """Windows are ordered by records/identity.py, not Python's comparison."""
    minimal = run_files.minimal_machine(TASK_CELLS)
    task = task_at(minimal)
    workload = producers.memory_patches(
        "surface_code:rotated_memory_z", 3, 2, 3, 0.001
    )
    first, second = workload.operations
    named = dataclasses.replace(second, id="b")
    round_counts = {1: 3, "b": 3}
    mixed = dataclasses.replace(
        workload, operations=(first, named), round_counts=round_counts
    )
    lowered = workload_settings.WorkloadSettings.running(mixed)
    settings = dataclasses.replace(task.machine, workload=lowered)
    mixed_task = collect.Task("mixed", settings, {})
    shot = collect.run_shot(mixed_task, 0)

    measured = measure.measure_shot(shot)

    assert measured.predictions == '{"1":"0","b":"0"}'
    assert measured.decoded_windows == 2


def test_the_window_sizes_and_period_are_the_card_the_qpu_ran():
    """The plan lays its windows out by the card; the qpu names no distance.

    The card commits 2 rounds, buffers 1 and keeps its own 2 us round,
    so a window arrives every 4 us, whatever the section's period says.
    """
    nine_rounds = run_files.minimal_machine(TASK_CELLS, rounds_per_shot=9)
    task = task_at(nine_rounds)
    card = code_geometry.SurfaceCodeModel(
        distance=3,
        round_microseconds=2.0,
        commit_rounds_override=2,
        buffer_rounds_override=1,
    )
    card_record = declared_run.GivenCard(card)
    qpu = dataclasses.replace(task.machine.qpu, code_card=card_record)
    settings = dataclasses.replace(task.machine, qpu=qpu)
    carded = collect.Task("carded", settings, {})
    shot = collect.run_shot(carded, 0)

    measured = measure.measure_shot(shot)

    assert measured.commit_rounds == 2
    assert measured.window_period_us == 4.0


def test_skorics_process_count_is_two_services_over_the_window_period():
    """N_par is ceil(2 tau_W / ((n_com + n_W) tau_rd)).

    Layer A commits n_com = 3 rounds and layer B its whole window,
    n_W = n_com + 2 n_buf = 9 (2209.08552 lines 388-390). A 5.064 us
    decode twice over those twelve one-microsecond rounds is 0.844
    processes, so one (lines 429-438).
    """
    measurement = slow_unit_shot(1)

    assert measurement.parallel_processes_needed == 1


def test_skorics_process_count_is_above_one_when_a_decode_outlasts_the_layers():
    """The same twelve rounds against a 10 us card ask for two processes.

    N_par is ceil(2 tau_W / ((n_com + n_W) tau_rd)) with tau_W the
    shot's mean service and n_W = n_com + 2 n_buf (2209.08552 lines
    388-390 and 429-438). The window scheme's rounds are null here, so
    both sizes are the distance.
    """
    measurement = slow_unit_shot(1, card_microseconds=10.0)

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
    """Theorem 1 per decode, read off a task where every window escalated.

    Ten windows escalated and their strong decodes read 57 rounds in
    all. Eq. (6) bounds the time per round by (1 / gamma)(d / r_strong)
    tau_gen = 3 / 5.7 us (2510.25222 lines 1206-1214), so one decode of
    r_strong = 5.7 rounds is bounded by 3 us: tau_gen r_com windows /
    escalated windows with r_com = d = 3. The strong service it bounds
    is the 10.0 us card plus its fetch and release, above it, as a card
    ten times the round time must be.
    """
    measurement = switching_shot(1000000.0)
    rows, _run_dir = run_files.folded_run(tmp_path, [measurement])

    assert measurement.escalated_windows == 10
    assert measurement.strong_decoded_rounds == 57
    assert rows[0]["strong_service_mean_us"] == 10.0628
    assert rows[0]["strong_service_bound_us"] == 3.0
    assert rows[0]["escalated_windows"] == 10


def test_the_per_decode_bound_counts_the_rounds_a_double_window_absorbed(
    tmp_path,
):
    """Theorem 1 per decode reads the generated rounds, not the windows.

    Every window escalates into a double window, which absorbs the
    windows it covers: four windows commit over the thirty rounds and all
    four escalated. The switching rate per d = 3 rounds is 4 / (30 / 3)
    = 0.4, so eq. (6) bounds one decode by d tau_gen / gamma = 3 / 0.4 =
    7.5 us (2510.25222 lines 1272-1304, 1335-1350). Four committed
    windows times r_com tau_gen would say 3 us.
    """
    double_window = strong_window_shapes.DoubleWindow.Settings()
    shot = switching_run(1000000.0, strong_window=double_window)
    measurement = measure.measure_shot(shot)
    rows, _run_dir = run_files.folded_run(tmp_path, [measurement])

    assert measurement.executed_rounds == 30
    assert measurement.decoded_windows == 4
    assert measurement.escalated_windows == 4
    assert rows[0]["strong_service_bound_us"] == 7.5


def test_a_tasks_strong_service_is_its_strong_decodes_mean(tmp_path):
    """The task divides its summed service by its strong decodes.

    At 5 dB seed 0 escalates no window and seed 1 escalates two, each a
    10.064 us strong decode. The task's mean service is 10.064 us, the
    sum over the count as gem5 forms avgMissLatency = missLatency /
    misses (src/mem/cache/base.cc:2188), not the 5.032 us a mean of the
    two shots' means would read.
    """
    quiet = switching_shot(5.0, seed=0)
    escalating = switching_shot(5.0, seed=1)
    rows, _run_dir = run_files.folded_run(tmp_path, [quiet, escalating])

    assert quiet.escalated_windows == 0
    assert escalating.escalated_windows == 2
    assert rows[0]["escalated_windows"] == 2
    assert rows[0]["strong_service_mean_us"] == 10.064


def test_a_windows_samples_are_counted_under_the_tier_that_committed_it():
    """Kept and escalated windows are told apart by the committing tier.

    At 5 dB seed 1 escalates two of the ten windows: their 10.064 us
    strong service is counted under strong and the eight weak decodes'
    under weak, as Ciw writes each record's customer class beside its
    wait (ciw node.py lines 856-862). A round's sample belongs to no
    window, so it names no tier.
    """
    escalating = switching_shot(5.0, seed=1)
    record = report.record_of([escalating])
    counts = _counts_by_name_tier_and_value(record.window_samples)

    assert counts[("service", "strong", 10.064)] == 2
    assert counts[("service", "weak", 1.064)] == 7
    assert counts[("service", "weak", 1.052)] == 1
    assert counts[("cwb_per_round", "", 0.004)] == 30


def _counts_by_name_tier_and_value(rows: list) -> dict:
    """window_samples.csv's counts, by latency point, tier and value."""
    counts = {}
    for row in rows:
        key = (row["name"], row["tier"], row["value_us"])
        counts[key] = row["count"]
    return counts


def test_the_formed_to_commit_median_and_p99_are_split_by_tier(tmp_path):
    """Kept and escalated windows each get their own median and p99.

    Seed 1 at 5 dB keeps eight windows and escalates two. Each tier's
    columns are numpy's nearest percentile of that tier's own samples,
    the rule the merged columns follow (percentile_of_counts).
    """
    escalating = switching_shot(5.0, seed=1)
    record = report.record_of([escalating])
    rows, _run_dir = run_files.folded_run(tmp_path, [escalating])
    kept = _samples_of(record.window_samples, "buffer0_ready_to_frame", "weak")
    escalated = _samples_of(
        record.window_samples, "buffer0_ready_to_frame", "strong"
    )

    assert len(kept) == 8
    assert len(escalated) == 2
    row = rows[0]
    kept_median = numpy.percentile(kept, 50, method="nearest")
    kept_p99 = numpy.percentile(kept, 99, method="nearest")
    escalated_median = numpy.percentile(escalated, 50, method="nearest")
    escalated_p99 = numpy.percentile(escalated, 99, method="nearest")
    assert row["buffer0_ready_to_frame_weak_median_us"] == kept_median
    assert row["buffer0_ready_to_frame_weak_p99_us"] == kept_p99
    assert row["buffer0_ready_to_frame_strong_median_us"] == escalated_median
    assert row["buffer0_ready_to_frame_strong_p99_us"] == escalated_p99


def test_a_tier_that_committed_no_window_has_no_latency_columns(tmp_path):
    """No escalated window, no strong percentile: the cell is empty."""
    quiet = switching_shot(5.0, seed=0)
    rows, _run_dir = run_files.folded_run(tmp_path, [quiet])

    assert rows[0]["buffer0_ready_to_frame_weak_median_us"] is not None
    assert "buffer0_ready_to_frame_strong_median_us" not in rows[0]


def _samples_of(rows: list, name: str, tier: str) -> list:
    """window_samples.csv's counts of one latency point and tier, expanded."""
    samples = []
    for row in rows:
        if (row["name"], row["tier"]) != (name, tier):
            continue
        repeated = [row["value_us"]] * row["count"]
        samples.extend(repeated)
    return samples


def test_a_task_with_no_strong_decode_has_no_strong_service(tmp_path):
    """No strong decode, no mean: the cell is empty, not zero.

    gem5 prints nothing for a ratio whose count is zero (the nonan flag,
    src/base/stats/text.cc:288-290); a zero would say the strong tier
    decoded in no time.
    """
    quiet = switching_shot(5.0, seed=0)
    rows, _run_dir = run_files.folded_run(tmp_path, [quiet])

    assert rows[0]["strong_service_mean_us"] is None


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
    shot = switching_run(1000000.0, commit_rounds=2)
    measurement = measure.measure_shot(shot)
    record = report.record_of([measurement])
    rows, _run_dir = run_files.folded_run(tmp_path, [measurement])

    assert record.shots[0]["commit_rounds"] == 2
    assert measurement.decoded_windows == 15
    assert measurement.escalated_windows == 15
    assert rows[0]["strong_service_bound_us"] == 2.0


def test_an_escalated_window_is_measured_on_the_strong_tiers_own_hops():
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
    measurement = switching_shot(1000000.0)

    samples = measurement.samples
    assert samples["input_link_per_window"] == [0.008] * 10
    assert samples["output_link_per_window"] == [0.012] * 10
    assert samples["escalation_link_per_window"] == [0.020] * 10
    assert samples["dep_block"] == [0.020] * 9 + [0.012]
    assert samples["compute_wait"] == [0.0] * 10
    assert samples["algorithm"] == [10.0] * 10
    assert samples["weak_attempt"][2] == 13.304


def test_an_escalated_windows_points_sum_to_its_reaction_time():
    """The chain runs weak attempt first, then the strong decode.

    Toshio et al. 2510.25222 Sec. III A orders it: the weak decoder
    answers, the verdict sends the window and its rounds to the strong
    decoder, the strong decoder answers and that is what commits. Every
    point on that order adds up to the window's reaction time to the
    tick.
    """
    measurement = switching_shot(1000000.0)

    totals = chain_sum_ticks(measurement)
    assert totals == reaction_ticks(measurement)


def test_a_kept_weak_result_is_measured_on_the_weak_hops():
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
    was dispatched with its other forced-class solve and waited 0.004 us
    for the input that solve's transfer brought, which is the dependency
    it had, and then 1.064 us more for the unit's compute to finish that
    solve, which is the structural wait gem5 counts as fuBusy. Window 9
    is the plain case beside it: 1.064 us of dependency wait and no
    structural wait at all.
    """
    measurement = switching_shot(0.0, seed=50)

    samples = measurement.samples
    weak_hops = [0.004] * 3 + [0.0] + [0.004] * 6
    assert samples["input_link_per_window"] == weak_hops
    assert samples["dep_block"] == [0.0] * 3 + [0.004] + [0.0] * 5 + [1.064]
    assert samples["compute_wait"] == [0.0] * 3 + [1.064] + [0.0] * 6
    assert samples["output_link_per_window"] == [0.004] * 10
    assert samples["escalation_link_per_window"] == [0.0] * 10
    assert samples["weak_attempt"] == [0.0] * 10
    assert samples["algorithm"] == [1.0] * 10


def shipped_shot(run_file: str, task_name: str):
    """Seed 0 of a task of a run file this repository ships.

    Each named task is at p 0.008, distance 3 and one microsecond
    rounds, the task the component validation reads, so what these
    assertions walk is a run folder a reader can build from the run file.
    """
    run_path = REPOSITORY / run_file
    study = experiment.load(run_path)
    (task,) = [task for task in study.tasks if task.name == task_name]
    return collect.run_shot(task, 0)


def base_shot(base) -> collect.Shot:
    """Seed 0 of a shipped base (decsim.settings) at that same task."""
    settings = base(3, SWITCHING_ERROR_PROBABILITY, 1.0)
    task = task_at(settings, SWITCHING_ERROR_PROBABILITY)
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
    """The identity on a machine the repository ships, not a test's own.

    decsim.settings.weak_decoder_baseline decodes each of its nine
    windows once and answers on that decode, so its confidence step is zero
    and every window's points are its whole path from its data being
    complete in the weak syndrome buffer to its correction committed in the
    frame.
    """
    shot = base_shot(machine_settings.weak_decoder_baseline)

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    later_solves = {
        window_id: later_solve_ticks(shot, window_id) for window_id in gaps
    }
    zero_by_window = dict.fromkeys(gaps, 0)
    assert len(gaps) == 9
    assert measurement.samples["confidence"] == [0.0] * 9
    assert later_solves == zero_by_window
    assert gaps == zero_by_window


def test_the_shipped_two_tier_config_sums_to_its_reaction_time():
    """The identity on every window of examples/two_tiers.py.

    Windows 0, 1, 8 and 9 escalated: their weak attempt, the strong
    decode that committed and the hops around it are the whole path, and
    their confidence step is zero because the attempt already runs to the
    verdict. The other six kept a weak result whose complementary gap
    ran a second forced-class solve after it, and that solve is their
    confidence step (decision D2, and Toshio et al. 2510.25222 Sec.
    III A steps 2 to 4 for the order of the escalated ones). Both sum
    to the tick.
    """
    shot = shipped_shot("examples/two_tiers.py", "d3")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    escalated = [0, 1, 8, 9]
    with_a_later_solve = [2, 3, 4, 5, 6, 7]
    decoded = escalated + with_a_later_solve
    escalated_later = [
        later_solve_ticks(shot, window_id) for window_id in escalated
    ]
    kept_later = [
        later_solve_ticks(shot, window_id) for window_id in with_a_later_solve
    ]
    zero_by_window = dict.fromkeys(decoded, 0)
    assert gaps == zero_by_window
    assert escalated_later == [0, 0, 0, 0]
    assert min(kept_later) > 0


def test_the_shipped_pinned_config_sums_to_its_reaction_time():
    """The same law where the strong tier's time is measured, not declared.

    experiments/switching/run.py's redo window task names a
    belief-matching strong tier, whose decode time is read off the host
    clock, so the values move from host to host and the identity does not:
    every window's points still add up to its reaction time to the tick, the
    solve it ran after the committing one being its confidence step.
    """
    shot = shipped_shot("experiments/switching/run.py", "redo_window_d3")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    confidence = measurement.samples["confidence"]
    window_ids = decoded_window_ids(shot)
    confidence_ticks = [ticks_of(value) for value in confidence]
    later_ticks = [
        later_solve_ticks(shot, window_id) for window_id in window_ids
    ]
    zero_by_window = dict.fromkeys(window_ids, 0)
    assert gaps
    assert gaps == zero_by_window
    assert confidence_ticks == later_ticks


def test_the_shipped_cluster_gap_config_sums_to_its_reaction_time():
    """The identity where the confidence step is a card of its own.

    experiments/switching/run.py's cluster gap task walks a union
    find decode for its signal and prices that walk at 12.0 us on the weak
    unit (decision D8), so a window the weak tier answered carries the walk
    between its decode's end and its verdict. A window that escalated
    carries none: its weak attempt already runs to the verdict and the
    strong decode it commits answers after it. The union find decode itself
    is read off the host clock, so the assertions are in ticks and about the
    identity, never about a magnitude.
    """
    shot = shipped_shot("experiments/switching/run.py", "cluster_gap_d3")

    measurement = measure.measure_shot(shot)
    gaps = chain_gap_ticks(shot, measurement)
    confidence = measurement.samples["confidence"]
    walk_us = 12.0
    confidence_values = set(confidence)
    gap_ticks = gaps.values()
    gap_values = set(gap_ticks)
    assert confidence_values == {0.0, walk_us}
    assert gap_values == {0}


def test_the_stage_points_are_the_committing_decodes_own_stages():
    """Fetch, algorithm and release add up to the service they are in.

    Every window of this run escalates, so the decode the frame took is
    the strong one and its three stages are the compute the service
    point measures: a six-round strong window, the commit region and
    the buffer ahead of it, fetches 0.024 us, decodes 10.0 and releases
    0.040, and the last one, whose buffer the stream's end clips to
    nothing, fetches 0.012.
    """
    measurement = switching_shot(1000000.0)

    samples = measurement.samples
    fetch_ticks = [ticks_of(value) for value in samples["fetch"]]
    algorithm_ticks = [ticks_of(value) for value in samples["algorithm"]]
    release_ticks = [ticks_of(value) for value in samples["release"]]
    service_ticks = [ticks_of(value) for value in samples["service"]]
    stages = zip(fetch_ticks, algorithm_ticks, release_ticks, strict=True)
    stage_totals = [sum(window_stages) for window_stages in stages]
    assert stage_totals == service_ticks
    assert samples["fetch"] == [0.024] * 9 + [0.012]


def test_a_cancelled_speculative_card_is_not_the_windows_algorithm():
    """A parallel run that escalates nothing reports the weak card.

    run_both_at_once starts a speculative strong decode on every window and
    cancels it at the verdict, and the speculative decode records its own stages
    under the same window key. The window's algorithm point is the decode that
    committed, which is the 1.0 us weak one on every window here.
    """
    measurement = switching_shot(0.0, True)

    samples = measurement.samples
    assert samples["algorithm"] == [1.0] * 10
    assert measurement.means["algorithm"] == 1.0
    assert samples["service"] == [1.064] * 9 + [1.052]


def test_a_second_forced_solve_is_not_the_windows_algorithm():
    """The complementary gap runs two solves; one of them committed.

    Both are jobs of the same window and each is charged one card
    (decisions D2 and D7), so the window's algorithm point is one card's
    1.0 us and never their sum.
    """
    measurement = switching_shot(0.0)

    assert measurement.samples["algorithm"] == [1.0] * 10


def cancelled_stage_records(stages) -> list:
    """The stage records a cancel closed, in record order."""
    cancelled = []
    for record in stages.records:
        if record.cancelled:
            cancelled.append(record)
    return cancelled


def test_a_cancelled_speculative_decode_ends_at_the_cancel():
    """The speculative strong decode stops where the weak verdict stopped it.

    run_both_at_once starts the strong decoder on every window and the
    confident weak verdict halts it (Toshio et al. 2510.25222 Sec. III A
    steps 1 and 3). The engine cannot unschedule the card's timer, so
    the cancel is what closes the speculative decode's open stage: one cancelled
    record per window, each ending at that window's verdict rather than
    ten microseconds later, and the weak decode's own records untouched.
    """
    shot = switching_run(0.0, run_both_at_once=True)

    stages = shot.machine.observation.stages
    windows = shot.machine.observation.windows.windows
    cancelled = cancelled_stage_records(stages)
    ends = [record.end_ticks for record in cancelled]
    verdicts = [windows[(1, record.window_id)].t_done for record in cancelled]
    cancelled_stages = [record.stage for record in cancelled]
    assert ends == verdicts
    assert cancelled_stages == ["algorithm"] * 10


def test_a_full_buffer_0_holds_rounds_and_the_wait_is_a_point():
    """The store's back-pressure on the controller, round by round.

    Rounds 1 to 15 find room. Round 16 is packed at 16.004 us into a
    full store and waits 0.144 us for the slot window 3's input frees at
    16.148; rounds 19 to 21 wait 2.212, 1.212 and 0.212 us for the three
    slots window 4 frees at 21.216, and the pattern repeats to round 30,
    the worst wait being 8.416 us on round 28. Thirteen rounds wait,
    51.912 us in all.
    """
    measurement = bounded_store_shot()

    stalls = measurement.samples["cwb_stall_per_round"]
    assert stalls[:15] == [0.0] * 15
    assert stalls[15] == 0.144
    assert stalls[18:21] == [2.212, 1.212, 0.212]
    assert stalls[27] == 8.416
    is_waiting = [stall > 0.0 for stall in stalls]
    waiting_count = sum(is_waiting)
    stall_ticks = [ticks_of(stall) for stall in stalls]
    total_ticks = sum(stall_ticks)
    assert waiting_count == 13
    assert total_ticks == 51912000


def test_the_store_wait_is_not_in_the_hop_the_round_then_crosses():
    """The round waits before the wire is asked for.

    The hop's own point is the card's 0.004 us on every one of the
    thirty rounds, held room or not, because the transfer is requested
    at the release; the wait is the store's and is reported as the
    store's.
    """
    measurement = bounded_store_shot()

    assert measurement.samples["cwb_per_round"] == [0.004] * 30
    assert measurement.means["cwb_stall_per_round"] == 51.912 / 30


def test_a_hops_setup_is_in_the_hop_and_not_in_the_wait_before_it():
    """A DMA's fixed delay is part of the latency its requester sees.

    gem5 adds a DMA's delay to the completion it schedules for the
    device (src/dev/dma_device.cc:116-118). The input hop here is one
    cycle of the 250 MHz fridge clock plus a five-cycle setup, 0.024 us,
    on every window, and a 0.1 us unit keeps pace with the rounds, so
    no window owes anything and none waits before its input hop.
    """
    settings = input_setup_settings()
    task = task_at(settings)
    measurement = measured(task)

    assert measurement.samples["input_link_per_window"] == [0.024] * 9
    assert measurement.samples["dep_block"] == [0.0] * 9


def test_a_stores_read_is_its_own_point_and_not_in_the_park():
    """A weak store whose read of a window costs 25 fridge cycles, 0.1 us.

    Each window's input read takes the store's 0.1 us after the dispatch
    and before the input hop, so store_read is 0.1 on all nine windows.
    The 0.1 us unit keeps pace with the rounds, so no window owes
    anything and dep_block is zero; the chain still adds up to each
    window's reaction time.
    """
    base = one_tier_machine(0.1)
    store = dataclasses.replace(
        base.weak_syndrome_buffer, read_cycles=25, clock=FRIDGE_CLOCK
    )
    settings = dataclasses.replace(base, weak_syndrome_buffer=store)
    task = task_at(settings)
    measurement = measured(task)

    assert measurement.samples["store_read"] == [0.1] * 9
    assert measurement.samples["dep_block"] == [0.0] * 9
    assert chain_sum_ticks(measurement) == reaction_ticks(measurement)


def input_setup_settings() -> machine_settings.MachineSettings:
    """A 0.1 us unit behind an input hop of one cycle and five of setup."""
    setup_cycles = {"weak_buffer_to_weak_decoder": 5}
    return one_tier_machine(0.1, setup_cycles=setup_cycles)


def seam_streams(stream_count: int, stagger_rounds: int = 0) -> tuple:
    """That many memory streams, one patch and one qubit each.

    Every stream is the same 30-round distance-3 memory circuit:
    memory_circuit plans one operation for the whole shot, so a run with
    more than one stream is declared here instead. Stream i starts
    i * stagger_rounds rounds into the shot.
    """
    circuit = workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", 30, 3, 0.001
    )
    operations = []
    for index in range(stream_count):
        operation_id = index + 1
        start_round = index * stagger_rounds
        operation = program_records.Operation(
            id=operation_id,
            name=f"mem{operation_id}",
            qubits=(index,),
            patches=(index,),
            circuit=circuit,
            scheduled_start_round=start_round,
        )
        operations.append(operation)
    return tuple(operations)


def seam_only_fabric():
    """The logical reference card with one hop priced: the seam.

    decoder_to_decoder is 125 cycles of the 250 MHz fridge clock, which
    is 0.5 us, and every other hop is free, so a window's outgoing
    boundary is the only thing a link charges for in the whole run.
    """
    hops = {path.value: 0 for path in transfer_records.LinkPath}
    hops["decoder_to_decoder"] = 125
    return fridge_hops(hops)


def seam_streams_shot(stream_count: int, stagger_rounds: int = 0):
    """One shot of those streams on that fabric."""
    settings = seam_streams_settings(stream_count, stagger_rounds)
    task = collect.Task("task", settings, {})
    return collect.run_shot(task, 0)


def seam_streams_settings(
    stream_count: int, stagger_rounds: int = 0
) -> machine_settings.MachineSettings:
    """The machine of those streams on that fabric.

    Eight one-microsecond weak units decode them, so the streams run
    side by side and each plans nine sliding windows.
    """
    operations = seam_streams(stream_count, stagger_rounds)
    links = seam_only_fabric()
    rounds_policy = round_policies.FixedRounds(30)
    workload = workload_settings.WorkloadSettings(
        operations=operations,
        rounds_policy=rounds_policy,
    )
    device = stim_device.StimDevice()
    source = declared_run.GivenSource(device)
    qpu = qpu_settings.QpuSettings(
        distance=3, source=source, round_period_microseconds=1.0
    )
    engine_clock = config_module.Clock(1000)
    engine = decoder_settings.EngineSettings(clock=engine_clock)
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings(
        preset_latency_microseconds=1.0
    )
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=matching,
        unit_count=8,
        engine=engine,
    )
    manager = decoder_settings.DecoderManagerSettings(dispatch_cycles=0)
    windows = window_settings.WindowSettings(terminal_policy="flush")
    # a 1 GHz frame beside the 1 GHz decoder engine, so one write is the
    # 4 ns of Yang et al. 2605.04892 Fig. 1 and every correction reaches
    # the frame on one of its edges
    frame_clock = config_module.Clock(1000)
    frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=4, clock=frame_clock
    )
    return machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        weak_decoder=weak_decoder,
        decoder_manager=manager,
        pauli_frame=frame,
        links=links,
    )


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


def test_a_late_streams_windows_count_from_their_own_rounds_sends():
    """A stream started 20 rounds late reads its own rounds' QPU sends.

    A round index counts within its operation's stream (records/
    windows.py Window), so the second stream's round 1 leaves the QPU
    20 us after the first stream's. Each window's first and last
    required rounds are its own operation's, found by operation and
    round as gem5 finds an instruction by thread and sequence number
    (src/cpu/o3/rob.hh:131-134), so both streams read the one-stream
    6.011 us and 1.011 us on every window, not 26.011 and 21.011.
    """
    one_stream_run = seam_streams_shot(1)
    staggered_run = seam_streams_shot(2, stagger_rounds=20)
    one_stream = measure.measure_shot(one_stream_run)
    staggered = measure.measure_shot(staggered_run)

    first_round = one_stream.samples["qpu_first_round_to_frame"]
    last_round = one_stream.samples["qpu_last_round_to_frame"]
    assert first_round == [6.011] * 9
    assert last_round == [1.011] * 9
    assert staggered.samples["qpu_first_round_to_frame"] == first_round * 2
    assert staggered.samples["qpu_last_round_to_frame"] == last_round * 2


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


def test_a_shot_with_no_pauli_frame_got_no_window_through():
    """A machine built with no frame commits nothing, and its rates are zero.

    Throughput runs from the first round to the last frame commit, and
    a Python-built machine whose pauli_frame is None commits no window,
    so no window and no round got through; the rounds it read out are
    still measured.
    """
    settings = seam_streams_settings(1)
    frameless = dataclasses.replace(settings, pauli_frame=None)
    task = collect.Task("frameless", frameless, {})
    shot = collect.run_shot(task, 0)
    measured = measure.measure_shot(shot)

    assert measured.decoded_windows == 0
    assert measured.throughput_windows_per_us == 0.0
    assert measured.throughput_rounds_per_us == 0.0
    assert measured.executed_rounds == 30


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


def test_a_shot_fails_when_any_of_its_patches_reads_a_wrong_observable():
    """Two patches, a burst on the second: the first right, the shot wrong.

    The burst region, radius 4 about patch 1's centre (11, 3) on the
    patches' shared plane, misses patch 0 entirely; at seed 1 patch 1's
    observable comes out wrong.
    """
    shot = _two_patch_burst_shot(seed=1)
    measurement = measure.measure_shot(shot)
    first, second = shot.result.operation_results

    assert first.logical_observables == first.observable_truth
    assert second.logical_observables != second.observable_truth
    assert measurement.logical_failure is True


def test_a_shot_records_each_operations_prediction():
    """The two patches' observables, each its own, not one failure flag.

    At seed 4 both truths are 0 and the loop predicts 1 on patch 1,
    under the burst, and 0 on patch 0, so operation 2's cell is 1 and
    operation 1's is 0. A csv reader that types a number would read the
    bits 01 as 1; the json cell reads back as the text it was written.
    """
    shot = _two_patch_burst_shot(seed=4)

    measurement = measure.measure_shot(shot)

    assert measurement.predictions == '{"1":"0","2":"1"}'
    assert fold.typed_value(measurement.predictions) == measurement.predictions


def two_patch_burst_settings() -> machine_settings.MachineSettings:
    """Two six-round memory patches under a burst on the second."""
    base = run_files.minimal_machine(TASK_CELLS)
    patches = producers.memory_patches(run_files.CODE_TASK, 6, 2, 3, 0.001)
    workload = workload_settings.WorkloadSettings.running(patches)
    burst = stim_device.BurstStimDevice.Settings(
        burst_radius=4.0,
        burst_center=(11.0, 3.0),
        burst_error_probability=0.3,
    )
    qpu = dataclasses.replace(base.qpu, source=burst)
    return dataclasses.replace(base, workload=workload, qpu=qpu)


def _two_patch_burst_shot(*, seed: int) -> collect.Shot:
    """One seed of two_patch_burst_settings."""
    settings = two_patch_burst_settings()
    task = task_at(settings)
    return collect.run_shot(task, seed)


LOAD_FLAGS = ("record_switching_windows", "backlog_trace")


def escalating_shot(strong_units: int):
    """Three patches' windows all escalate to 10 us strong cards.

    One patch's strong decodes wait on one another's boundaries and
    never queue; three patches' reach the strong units together.
    """
    shot = switching_run(
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
    by_arrival = strong_requests_by_arrival(requests)
    waits = []
    free_at = 0
    for request in by_arrival:
        records = stage_records_of(stages, request)
        first = records[0]
        input_hop = first.ready_ticks - first.dispatch_ticks
        arrival = request.ready_ticks + input_hop
        duration = service_ticks(records)
        start = max(arrival, free_at)
        wait_ticks = start - arrival
        wait = config_module.ticks_to_microseconds(wait_ticks)
        waits.append(wait)
        free_at = start + duration
    return waits


def strong_requests_by_arrival(requests) -> list:
    """The strong decode requests, in the order they became ready."""
    strong = window_records.DecoderTier.STRONG
    strong_requests = [
        request for request in requests if request.request_key.tier is strong
    ]
    return sorted(strong_requests, key=ready_ticks_of)


def stage_records_of(stages, request) -> list:
    """The stage ledger's records of one decode request's run."""
    run_sequence = request.request_key.run_sequence
    return [
        record
        for record in stages.records
        if run_sequence in record.run_sequences
    ]


def service_ticks(records) -> int:
    """A decode's service: its first stage's start to its last one's end."""
    starts = [record.start_ticks for record in records]
    ends = [record.end_ticks for record in records]
    return max(ends) - min(starts)


def ready_ticks_of(request) -> int:
    return request.ready_ticks


def test_a_strong_requests_wait_is_lindleys_one_server_wait():
    """Three patches' escalations into one strong unit queue as one server's."""
    shot, measurement = escalating_shot(1)
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
            machine.decoders.decoder_manager,
            machine.decoders.strong_decoder_manager,
        ):
            if manager is not None:
                self.units.extend(manager.service.pool.units)
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
    monkeypatch,
):
    """A unit takes the next decode into its memory while it computes.

    The column reads the most strong decodes held at once off the stage
    ledger; the units' own slots, sampled after every engine action, give
    the same peak. One unit holds two residents, the decode it computes
    and the next one (decoder_unit.py, INPUT_SLOT_COUNT), so one is
    the most it can hold waiting.
    """
    samplers = []
    build = machine_module.Machine.build

    def build_and_sample(
        settings, seed, built_models=None, online_threshold=None
    ):
        machine = build(settings, seed, built_models, online_threshold)
        sampler = StrongResidentsWaitingOnCompute(machine)
        machine.engine.action_done.connect(sampler.sample)
        samplers.append(sampler)
        return machine

    monkeypatch.setattr(machine_module.Machine, "build", build_and_sample)
    _, measurement = escalating_shot(1)
    (sampler,) = samplers
    depths = sampler.depth_by_tick.values()

    assert measurement.strong_held_in_units_max == max(depths) == 1


def test_enough_strong_units_leave_the_same_windows_and_no_wait():
    """The paired run: the same seed, the real unit count and ten.

    The syndromes and the escalations are the same in both, so what the
    extra units remove, the strong wait and the rounds held, is the
    overload alone; what stays, the weak input's weight and service and
    the escalated windows, is the difficulty.
    """
    _, real = escalating_shot(1)
    _, ample = escalating_shot(10)

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
    bare = switching_shot(1000000.0)
    _, kept = escalating_shot(1)
    bare_record = report.record_of([bare])
    kept_record = report.record_of([kept])
    bare_rows, _bare_dir = run_files.folded_run(tmp_path, [bare])
    kept_rows, _kept_dir = run_files.folded_run(tmp_path, [kept])

    assert "strong_wait_max_us" not in bare_record.shots[0]
    assert "escalated_fraction" not in bare_rows[0]
    assert kept_record.shots[0]["strong_wait_max_us"] > 0
    assert kept_rows[0]["escalated_fraction"] == 1.0
    assert kept_rows[0]["backlog_peak_rounds"] == kept.backlog_peak_rounds


def test_a_source_that_samples_no_shot_is_refused_with_a_sentence():
    """timing_only draws no shot, so the loop has no truth to be judged by."""
    settings = timing_only_settings()
    task = task_at(settings)

    with pytest.raises(refusal.RefusalError, match="sampled none"):
        measured(task)


def timing_only_settings() -> machine_settings.MachineSettings:
    """The minimal machine on a source that states each round's size."""
    base = run_files.minimal_machine(TASK_CELLS)
    timing_only = syndrome_devices.TimingOnlyDevice.Settings()
    qpu = dataclasses.replace(base.qpu, source=timing_only)
    return dataclasses.replace(base, qpu=qpu)


def test_a_live_stream_is_scored_through_its_owner_over_its_run_rounds():
    """The owner's own Stim circuit is the shot sinter would score.

    Its segments and the operations that hold and resume its patch have
    no truth of their own. Its horizon is its circuit's rounds, which
    include the rounds it idled through while the prefix's answer came
    back; a d=3 memory round holds 8 of Stim's detectors.
    """
    program = memory_programs.memory_program()
    live = live_memory_example.live_settings(
        program,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="memory-patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )
    frame = pauli_frame_module.PauliFrameConfig()
    settings = dataclasses.replace(live, pauli_frame=frame)
    task = collect.Task("task", settings, {})
    shot = collect.run_shot(task, 17)
    measured = measure.measure_shot(shot)
    sampled = shot.machine.observation.sampled_shots.shots_by_operation
    owner_shot = sampled[producers.LIVE_STREAM_ID]
    stim_rounds = owner_shot.circuit.num_detectors // 8

    assert list(sampled) == [producers.LIVE_STREAM_ID]
    assert measured.executed_rounds == stim_rounds
    assert stim_rounds > 3 + 1
    assert measured.logical_failure is False


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


def status_column_counts(statuses: list) -> collections.Counter:
    """How many windows returned each status, by its csv column."""
    counts = collections.Counter()
    for status in statuses:
        if status is None:
            continue
        column = f"{status.value}_windows"
        counts[column] += 1
    return counts


def nonzero_counts(counts: dict) -> dict:
    """The columns whose count is not zero."""
    found = {}
    for column, count in counts.items():
        if count:
            found[column] = count
    return found


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
    settings = one_iteration_relay_settings()
    task = task_at(settings, 0.003)
    measurement = measured(task)
    record = report.record_of([measurement])
    rows, _run_dir = run_files.folded_run(tmp_path, [measurement])
    expected = status_column_counts(returned)
    statuses = measurement.window_statuses
    counted = nonzero_counts(statuses)
    nonconverged = counted["nonconverged_windows"]

    assert expected["nonconverged_windows"] > 0
    assert counted == expected
    assert record.shots[0]["nonconverged_windows"] == nonconverged
    assert rows[0]["nonconverged_windows"] == nonconverged


def one_iteration_relay_settings() -> machine_settings.MachineSettings:
    """The minimal machine on Relay-BP held to one iteration, no relay."""
    cells = cells_at(0.003)
    base = run_files.minimal_machine(cells)
    relay = run_files.RELAY_BP.Settings(
        pre_iterations=1, relay_set_count=0, iterations_per_set=1
    )
    weak_decoder = run_files.decoder_pool(relay, FRIDGE_CLOCK)
    return dataclasses.replace(base, weak_decoder=weak_decoder)


def decoder_row_shot(kind: str, seed: int):
    """One seeded shot of the minimal machine with its weak row named."""
    cells = cells_at(0.01)
    settings = run_files.minimal_machine(cells, weak_decoder=kind)
    task = task_at(settings, 0.01)
    return measured(task, seed)


def test_the_sample_digest_names_the_draw_and_not_the_decoder():
    """Two decoder rows at one seed share a digest; two seeds do not.

    The device draws a shot from the run's seed alone, so PyMatching and
    Union-Find see the same detection events and truth at seed 0, and
    seed 1 is another draw.
    """
    matching = decoder_row_shot("pymatching", 0)
    union_find = decoder_row_shot("union_find", 0)
    next_seed = decoder_row_shot("pymatching", 1)

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


def test_a_strong_decode_with_no_correction_unscores_its_shot(monkeypatch):
    """The strong answer is the window's final one, and so is its status.

    Every window escalates and every strong decode produces nothing, so
    each window's final decode is the strong one with no correction.
    """
    strong_windows = {(STRONG, index) for index in EVERY_WINDOW}
    crashing_decodes(monkeypatch, strong_windows)
    shot = switching_run(1000000.0)
    measurement = measure.measure_shot(shot)
    statuses = measurement.window_statuses

    assert measurement.escalated_windows == measurement.decoded_windows
    assert measurement.is_scored is False
    assert measurement.unscored_reason == "upstream_exception"
    assert statuses["backend_error_windows"] == measurement.decoded_windows
    assert measurement.provisional_no_correction_windows == 0


def test_a_weak_decode_with_no_correction_unscores_its_shot_after_strong(
    monkeypatch,
):
    """The replaced weak answer counts apart and still unscores the shot.

    The strong decodes all succeed, so no window's final status is an
    error; each weak answer was committed provisionally first, and what
    a provisional commit fed forward survives the strong answer.
    """
    weak_windows = {(WEAK, index) for index in EVERY_WINDOW}
    crashing_decodes(monkeypatch, weak_windows)
    shot = switching_run(1000000.0)
    measurement = measure.measure_shot(shot)
    statuses = measurement.window_statuses

    assert measurement.escalated_windows == measurement.decoded_windows
    assert measurement.is_scored is False
    assert measurement.unscored_reason == "upstream_exception"
    assert statuses["backend_error_windows"] == 0
    assert measurement.provisional_no_correction_windows == (
        measurement.decoded_windows
    )


def test_a_strong_window_with_no_correction_moves_the_next_ones_input(
    monkeypatch,
):
    """Window 0's empty strong answer reaches window 1, whose decode is fine.

    The held boundary ships from the strong result (window_boundaries.py
    ship_held), so window 1 reads a different syndrome than it does when
    window 0's strong decode answers, and the shot is unscored by window
    0 alone.
    """
    crashing = set()
    syndromes = crashing_decodes(monkeypatch, crashing)
    switching_run(1000000.0)
    answered_input = syndromes[(WEAK, 1)]
    crashing.add((STRONG, 0))
    shot = switching_run(1000000.0)
    measurement = measure.measure_shot(shot)
    crashed_input = syndromes[(WEAK, 1)]

    assert crashed_input != answered_input
    assert measurement.is_scored is False
    assert measurement.window_statuses["backend_error_windows"] == 1


def refereed_settings() -> machine_settings.MachineSettings:
    """PyMatching weak decodes, each window decoded again by Tesseract."""
    cells = cells_at(0.01)
    base = run_files.minimal_machine(cells, weak_decoder="pymatching")
    checked = verify_windows.TesseractCheckedDecoder.Settings(
        inner=base.weak_decoder.algorithm
    )
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=checked)
    return dataclasses.replace(base, weak_decoder=weak_decoder)


def test_a_shot_counts_the_referees_checks_in_the_referee_columns(tmp_path):
    """The columns are the referee audit's own counts, whatever referees.

    The referee re-decodes every window the loop decoded
    (decsim/decoders/verify_windows.py), so its checks are the decoded
    windows, and the summary adds them up over the task's shots.
    """
    pytest.importorskip("tesseract_decoder")
    settings = refereed_settings()
    task = task_at(settings, 0.01)
    shot = collect.run_shot(task, 0)

    measurement = measure.measure_shot(shot)

    audit = shot.machine.observation.referee_audit
    (row,), _run_dir = run_files.folded_run(tmp_path, [measurement])
    assert audit.windows_checked == measurement.decoded_windows
    assert measurement.referee_windows_checked == audit.windows_checked
    assert (
        measurement.referee_window_disagreements == audit.window_disagreements
    )
    assert row["referee_windows_checked"] == audit.windows_checked
    assert row["referee_window_disagreements"] == audit.window_disagreements


def test_no_runner_module_names_a_row_of_the_decoder_table():
    """A new decoder is one table row and one axis value, no runner edit.

    The referent is the decoder rows themselves (DECODER_ROWS). The
    runner is every module below the experiments package. None imports
    a row's package, or a
    package only the rows import (a decoder backend), under any alias,
    and no string constant in it is a row's key (NOTE section 9 item
    10), so what it measures holds for every row.
    """
    runner_paths = _runner_paths()

    named = _rows_named_in(runner_paths)

    assert named == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "import decsim.decoders.minimum_weight_perfect_matching.decoder"
            " as plain\n",
            ["decsim.decoders.minimum_weight_perfect_matching.decoder"],
        ),
        (
            "from decsim.decoders import union_find as plain\n",
            ["decsim.decoders.union_find"],
        ),
        ("import pymatching as harmless\n", ["pymatching"]),
        ("from relay_bp import RelayDecoder\n", ["relay_bp.RelayDecoder"]),
        ('KIND = "tesseract"\n', ["tesseract"]),
        ('"""A tesseract has eight cells."""\n', []),
        ('SENTENCE = "a tesseract has eight cells"\n', []),
        ("import decsim.decoders.settings as decoder_settings\n", []),
    ],
)
def test_the_row_check_reads_imports_and_keys_and_not_prose(source, expected):
    """An aliased import is caught and a word in a sentence is not.

    A docstring is prose and is not read. A key joined from two
    literals is out of scope: the check catches coupling a reader
    writes, not every way to hide it.
    """
    assert _row_names_of(source) == expected


def _runner_paths() -> list:
    """Every module below the experiments package."""
    measure_file = pathlib.Path(measure.__file__)
    experiments_dir = measure_file.parent
    experiment_files = experiments_dir.rglob("*.py")
    return sorted(experiment_files)


def _rows_named_in(paths: list) -> list:
    """Each (file, name) where a module couples to a decoder row."""
    named = []
    for path in paths:
        source = path.read_text()
        names = _row_names_of(source)
        named.extend((path.name, name) for name in names)
    return named


def _row_names_of(source: str) -> list:
    """The row modules a source imports and the row keys it spells.

    A docstring, or any string standing alone as a statement, is prose.
    """
    keys, row_modules = _decoder_rows()
    tree = ast.parse(source)
    prose = _prose_strings(tree)
    names = []
    for node in ast.walk(tree):
        imported = _imported_names(node)
        coupled = _within(imported, row_modules)
        names.extend(coupled)
        is_string = isinstance(node, ast.Constant) and id(node) not in prose
        if is_string and node.value in keys:
            names.append(node.value)
    return names


# Every decoder row, by the name its reports carry (its Settings' name),
# as sinter's BUILT_IN_DECODERS lists its own
# (sinter/_decoding/_decoding_all_built_in_decoders.py:12).
DECODER_ROWS = {
    "pymatching": minimum_weight_perfect_matching.PyMatchingDecoder,
    "unweighted_pymatching": (
        minimum_weight_perfect_matching.UnweightedPyMatchingDecoder
    ),
    "belief_matching": belief_matching.BeliefMatchingDecoder,
    "union_find": union_find.UnionFindDecoder,
    "tesseract": tesseract.TesseractDecoder,
    "relay_bp": relay_decoder.RelayBeliefPropagationDecoder,
    "bposd": belief_propagation_osd.BeliefPropagationOsdDecoder,
    "measured_table": measured_table.MeasuredTableDecoder,
    "dispatch_steps": dispatch_steps.DispatchStepsDecoder,
}


@functools.cache
def _decoder_rows() -> tuple:
    """The row keys, and the modules only a row may import.

    Those are each row class's package, and every package the rows'
    modules import that no other decsim module does: the backends
    (pymatching, tesseract_decoder, relay_bp, ldpc). A package the rest
    of decsim shares, such as stim or numpy, is no row's.
    """
    rows = DECODER_ROWS
    row_packages = set()
    for row in rows.values():
        package, _, _ = row.__module__.rpartition(".")
        row_packages.add(package)
    row_imports = _packages_imported(row_packages, inside=True)
    other_imports = _packages_imported(row_packages, inside=False)
    shared = other_imports | sys.stdlib_module_names | {"decsim"}
    backends = row_imports - shared
    row_modules = row_packages | backends
    keys = frozenset(rows)
    return keys, frozenset(row_modules)


def _packages_imported(row_packages: set, *, inside: bool) -> set:
    """The top-level packages decsim's modules import, in or out of rows."""
    machine_file = pathlib.Path(machine_module.__file__)
    package_dir = machine_file.parent
    packages = set()
    for path in package_dir.rglob("*.py"):
        module = _module_name(path, package_dir.parent)
        if _is_within(module, row_packages) != inside:
            continue
        source = path.read_text()
        tree = ast.parse(source)
        for name in _all_imports(tree):
            package, _, _ = name.partition(".")
            packages.add(package)
    return packages


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    """The dotted module name of a source file below a root."""
    relative = path.relative_to(root)
    stem = relative.with_suffix("")
    return ".".join(stem.parts)


def _prose_strings(tree: ast.AST) -> set:
    """The ids of the strings that stand alone as statements."""
    prose = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            statement_id = id(node.value)
            prose.add(statement_id)
    return prose


def _imported_names(node: ast.AST) -> list:
    """The modules one import names, a from-import's names joined on."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [f"{node.module}.{alias.name}" for alias in node.names]
    return []


def _all_imports(tree: ast.AST) -> list:
    """Every module a module's imports name, at any depth."""
    names = []
    for node in ast.walk(tree):
        imported = _imported_names(node)
        names.extend(imported)
    return names


def _within(names: list, modules: frozenset) -> list:
    """The names that are one of the modules or below one."""
    return [name for name in names if _is_within(name, modules)]


def _is_within(name: str, modules) -> bool:
    """Whether a dotted name is one of the modules or below one."""
    for module in modules:
        if name == module or name.startswith(f"{module}."):
            return True
    return False


def test_a_windows_confidence_is_the_gap_its_verdict_request_ended_with():
    """Each verdict's gap and escalation, as the requests end them.

    A probe on the requests' end hears every request's soft output and
    terminal outcome, apart from the confidence ledger the rows are read
    from; the kept and the escalated weak requests are the windows whose
    confidence the verdict read (Toshio et al. 2510.25222 Sec. III A
    step 3).
    """
    cells = cells_at(0.003)
    switching = run_files.switching_machine(cells)
    task = task_at(switching, 0.003)
    settings = task.machine
    machine = machine_module.Machine.build(settings, 0)
    requests = declared_run.EndedRequests()
    requests.attach(machine)
    result = machine.run()
    shot = collect.Shot(task, 0, machine, result, 0.0)

    measurement = measure.measure_shot(shot)
    rows = report.window_confidence_rows([measurement])

    assert _verdicts_of_rows(rows) == _verdicts_of_requests(requests)
    assert {row["seed"] for row in rows} == {0}
    assert any(row["escalated"] for row in rows)


def _verdicts_of_rows(rows) -> dict:
    """Each row's window: its gap, and whether it escalated."""
    verdicts = {}
    for row in rows:
        key = (row["operation_id"], row["window_index"])
        verdicts[key] = (row["gap_nats"], row["escalated"])
    return verdicts


def _verdicts_of_requests(requests) -> dict:
    """Each verdict request's window: its gap, and whether it escalated."""
    escalated = decoding_records.RequestProcessingOutcome.WEAK_AWAITED_STRONG
    verdicts = {}
    for ended in requests.ended:
        if ended.outcome not in decode_records.VERDICT_OUTCOMES:
            continue
        request_key = ended.job.request_key
        key = (request_key.operation_id, request_key.window_id)
        gap = _gap_of(ended.result)
        verdicts[key] = (gap, ended.outcome is escalated)
    return verdicts


def _gap_of(result):
    """A request's gap, None when the signal gave none."""
    if result is None or result.soft_output is None:
        return None
    return result.soft_output.gap
