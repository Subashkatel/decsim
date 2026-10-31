"""Two T gates under the three ways a decision can meet the strong tier.

A T gate's conditional S waits for its merge's decoded outcome (SWIPER
2412.05115, device_manager.py; frontends/lattice_surgery.py). With
every window escalated, the outcome is a weak answer first and a strong
one later. Always waiting releases the S on the strong answer (the
final stage). Never waiting releases it on the weak one (the provisional
stage). Repairing releases it on the weak one and holds the next T
merge until the first one's answer is final, since a correction found
later turns the axis of the next non-Clifford rotation, which is fixed
when it starts (Chamberland, Iyer and Poulin 1704.06662 Sec. 2; Litinski
1808.02892 p. 5).
"""

import pathlib

import decsim.confidence.no_signal as no_signal
import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.frontends.lattice_surgery as lattice_surgery
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.qpu.settings as qpu_settings
import decsim.records.windows as window_records
import decsim.windows.schemes.region_sliding as region_sliding
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run

DISTANCE = 5
# two T gates on patch (0,0), each as SWIPER's compiler writes it
# (lattice_surgery_schedule.py inject_T, merge, S), the second after the
# first's conditional S
HERE = pathlib.Path(__file__)
DATA = HERE.parents[1] / "data"
TWO_T_GATES = DATA / "two_t_gates.lss"
FIRST_T_MERGE = 1
FIRST_S_MERGE = 3
SECOND_T_MERGE = 8
# a hold of 2 us, far below the strong tier's 20 us, and the round after
HOLD_LIMIT_MICROSECONDS = 2.0
ROUND_MICROSECONDS = 1.0
# (release stage, whether the gate holds decisions, its hold limit)
ALWAYS_WAIT = (window_records.FINAL_STAGE, False, None)
NEVER_WAIT = (window_records.PROVISIONAL_STAGE, False, None)
REPAIR = (window_records.PROVISIONAL_STAGE, True, None)
BOUNDED_REPAIR = (
    window_records.PROVISIONAL_STAGE,
    True,
    HOLD_LIMIT_MICROSECONDS,
)


def _pool(microseconds):
    algorithm = decoders.PresetLatencyDecoder.Settings(microseconds)
    return decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=declared_run.DECLARED_ENGINE, unit_count=8
    )


def _escalating_every_window():
    """Switching whose verdict sends every window to the strong tier."""
    confidence = no_signal.NoSignal.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    verdict = verdicts.Always.Settings()
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, verdict=verdict
    )


def run_two_t_gates(design):
    """The machine after its run, union-find 0.5 us and the strong 20 us."""
    release_stage, holds, limit = design
    workload = lattice_surgery.read_schedule(TWO_T_GATES, DISTANCE)
    settings = workload_settings.WorkloadSettings.running(workload)
    scheme = region_sliding.RegionSlidingScheme.Settings()
    windows = window_settings.WindowSettings(
        scheme=scheme, release_stage=release_stage
    )
    qpu = qpu_settings.QpuSettings(
        distance=DISTANCE, round_period_microseconds=ROUND_MICROSECONDS
    )
    controller = controller_settings.ControllerSettings(
        holds_decisions_for_final_results=holds,
        final_result_hold_limit_microseconds=limit,
    )
    base = machines.base(settings, windows, qpu, controller)
    weak_pool = _pool(0.5)
    with_weak = machines.with_pool(base, weak_pool)
    fabric = link_cards.strong_side(with_weak.fabric)
    strong_pool = _pool(20.0)
    switching = _escalating_every_window()
    graph = machines.with_strong_tier(with_weak, strong_pool, switching, fabric)
    machine = machine_module.Machine.build(graph, 0)
    machine.run()
    return machine


def started_at(machine, operation_id):
    """The tick the QPU started the operation's body."""
    stamps = machine.observation.runtime_stamps
    return stamps.op_start[operation_id]


def test_under_repair_each_decision_starts_after_the_finals_it_descends_from():
    machine = run_two_t_gates(REPAIR)

    gate = machine.control.final_result_gate
    assert gate.earlier_by_decision[SECOND_T_MERGE] == (FIRST_T_MERGE,)
    for decision_id, earlier_ids in gate.earlier_by_decision.items():
        decision_start = started_at(machine, decision_id)
        for earlier_id in earlier_ids:
            assert decision_start > gate.final_ticks[earlier_id]


def test_a_provisional_release_starts_the_conditional_s_before_waiting():
    """The S goes ahead on the weak answer; the gate never holds it."""
    waiting = run_two_t_gates(ALWAYS_WAIT)
    never_waiting = run_two_t_gates(NEVER_WAIT)
    repairing = run_two_t_gates(REPAIR)

    waited_start = started_at(waiting, FIRST_S_MERGE)
    provisional_start = started_at(never_waiting, FIRST_S_MERGE)
    repaired_start = started_at(repairing, FIRST_S_MERGE)
    assert provisional_start < waited_start
    assert repaired_start == provisional_start


def test_the_gate_holds_the_next_t_merge_that_never_waiting_starts():
    never_waiting = run_two_t_gates(NEVER_WAIT)
    repairing = run_two_t_gates(REPAIR)

    free_start = started_at(never_waiting, SECOND_T_MERGE)
    held_start = started_at(repairing, SECOND_T_MERGE)
    assert never_waiting.control.final_result_gate is None
    assert free_start < held_start


def test_a_bounded_hold_delays_a_decision_by_at_most_its_limit():
    """It waits 2 us and the boundary after, then starts on weak answers."""
    never_waiting = run_two_t_gates(NEVER_WAIT)
    bounded = run_two_t_gates(BOUNDED_REPAIR)

    free_start = started_at(never_waiting, SECOND_T_MERGE)
    bounded_start = started_at(bounded, SECOND_T_MERGE)
    gate = bounded.control.final_result_gate
    bound = HOLD_LIMIT_MICROSECONDS + ROUND_MICROSECONDS
    limit_and_a_round = config.microseconds_to_ticks(bound)
    assert free_start < bounded_start <= free_start + limit_and_a_round
    assert gate.expired_ids == {SECOND_T_MERGE}
