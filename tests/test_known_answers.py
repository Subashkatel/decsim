"""Known answers and invariants of whole runs, through the Python front end.

Each test builds its machine as a user does, from decsim.settings' bases on
Stim's rotated surface code memory at distance 3, and checks one law whose
answer is known without running the machine:

- zero noise draws no detection event (stim.Circuit.generated with every
  noise channel at 0), so no tier flips a logical observable;
- a latency added on one hop of the reaction path moves the reaction time,
  from a round's measurement to the instruction that acts on its correction
  (docs/reference/glossary.md), by exactly that latency;
- every round is committed by exactly one owner and every owner's
  correction enters the Pauli frame exactly once (Toshio et al. 2510.25222
  Theorem 1: two owners never claim one round).

Three more laws sit beside the parts they exercise: switching that never
escalates is the weak tier alone (tests/escalation/test_switching_mode.py),
strong-only predicts what its decoder predicts offline on the same events
(tests/decoders/test_decoder_tiers.py), and the frame is the XOR of its
committed corrections (tests/pauli_frame/test_pauli_frame.py).
"""

import collections
import dataclasses

import pytest

import decsim.collect as collect
import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.qpu.round_policies as round_policies
import decsim.records.program as program_records
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

DISTANCE = 3
ROUND_PERIOD_MICROSECONDS = 1.0
PHYSICAL_ERROR_PROBABILITY = 0.005
CODE_TASK = "surface_code:rotated_memory_z"
# no complementary gap reaches it, so every window escalates
UNREACHABLE_DECIBELS = 1e6
REDO_WINDOW = strong_window_shapes.RedoWindow.Settings()
DOUBLE_WINDOW = strong_window_shapes.DoubleWindow.Settings()
# every hop a round's correction crosses before the instruction that acts
# on it reaches the QPU, in the order it crosses them
REACTION_PATH = (
    "qpu_to_controller",
    "controller_to_weak_buffer",
    "weak_buffer_to_weak_decoder",
    "weak_decoder_to_frame",
    "frame_to_controller",
    "controller_to_qpu",
)
FEEDBACK_ROUNDS = 6


def run(settings, seed: int = 0) -> collect.Shot:
    """One seeded shot of the settings, run."""
    task = collect.Task(settings, {})
    return collect.run_shot(task, seed)


def weak_only(physical_error_probability: float):
    """The weak decoder baseline at distance 3 on 1 us rounds."""
    return machine_settings.weak_decoder_baseline(
        DISTANCE, physical_error_probability, ROUND_PERIOD_MICROSECONDS
    )


def strong_only(physical_error_probability: float):
    """The strong decoder baseline at distance 3 on 1 us rounds."""
    return machine_settings.strong_decoder_baseline(
        DISTANCE, physical_error_probability, ROUND_PERIOD_MICROSECONDS
    )


def switching(physical_error_probability: float, strong_window=REDO_WINDOW):
    """The weak base escalating every window to the strong base's tier.

    The strong side's four hops are one room cycle each.
    """
    base = weak_only(physical_error_probability)
    strong_base = strong_only(physical_error_probability)
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=UNREACHABLE_DECIBELS
    )
    slot = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, strong_window=strong_window
    )
    windows = window_settings.switching_windows(base.windows, strong_window)
    links = machine_settings.one_cycle_strong_side(base.links)
    return dataclasses.replace(
        base,
        links=links,
        windows=windows,
        strong_decoder=strong_base.strong_decoder,
        switching=slot,
    )


def switching_on_double_windows(physical_error_probability: float):
    """Every window escalating to a double window, which absorbs windows."""
    return switching(physical_error_probability, DOUBLE_WINDOW)


def logical_failures(shot: collect.Shot) -> list:
    """Each operation's logical failure: a predicted bit off the truth."""
    failures = []
    for operation_result in shot.result.operation_results:
        failures.append(operation_result.logical_failure)
    return failures


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "shape",
    (weak_only, strong_only, switching),
    ids=("weak_only", "strong_only", "switching"),
)
def test_a_noiseless_run_makes_no_logical_error(shape, seed):
    """Zero noise gives zero logical errors on every tier that commits."""
    settings = shape(0.0)

    shot = run(settings, seed)

    assert logical_failures(shot) == [False]


def feedback_settings():
    """The weak base running two memories, the second waiting on the first.

    A memory releases nothing, so the second operation, blocked on the
    first one's result, is the instruction that acts on its correction.
    """
    base = weak_only(0.001)
    circuit = workload_settings.memory_circuit(
        CODE_TASK, FEEDBACK_ROUNDS, DISTANCE, 0.001
    )
    first = program_records.Operation(
        id=1, name="mem0", qubits=(0,), patches=(0,), circuit=circuit
    )
    second = program_records.Operation(
        id=2,
        name="mem1",
        qubits=(1,),
        patches=(1,),
        circuit=circuit,
        blocked_by=1,
    )
    rounds_policy = round_policies.FixedRounds(FEEDBACK_ROUNDS)
    workload = workload_settings.WorkloadSettings(
        operations=(first, second), rounds_policy=rounds_policy
    )
    return dataclasses.replace(base, workload=workload)


def last_round_tick(shot: collect.Shot, operation_id: int) -> int:
    """The tick the operation's last round left the QPU."""
    ticks = []
    for event in shot.machine.observation.round_events.events:
        if event.kind != "EMITTED" or event.operation_id != operation_id:
            continue
        ticks.append(event.tick)
    return max(ticks)


def command_arrival_tick(shot: collect.Shot, operation_id: int) -> int:
    """The tick the operation's command reached the QPU."""
    arrivals = []
    for event in shot.machine.observation.command_events.events:
        operation = event.command.operation
        if event.kind != "ARRIVED" or operation.id != operation_id:
            continue
        arrivals.append(event.tick)
    (arrival,) = arrivals
    return arrival


def reaction_ticks(settings) -> int:
    """From the first memory's last round to the second one's command."""
    shot = run(settings)
    last_round = last_round_tick(shot, 1)
    arrival = command_arrival_tick(shot, 2)
    return arrival - last_round


def with_latency_added(settings, path_name: str, microseconds: float):
    """The settings with one path's latency longer by microseconds."""
    path = getattr(settings.links, path_name)
    latency_ticks = path.channel.propagation_latency_ticks
    latency = config.ticks_to_microseconds(latency_ticks)
    longer = latency + microseconds
    links = link_profiles.with_path_latency(settings.links, path_name, longer)
    return dataclasses.replace(settings, links=links)


@pytest.mark.parametrize("path_name", REACTION_PATH)
def test_a_latency_added_on_the_reaction_path_adds_exactly_that(path_name):
    """One microsecond more on any hop of the reaction path is one more.

    Each hop is one fridge cycle on the weak base, and one microsecond is
    a whole number of every clock's cycles, so no edge rounds it away.
    """
    settings = feedback_settings()
    slower = with_latency_added(settings, path_name, 1.0)

    reaction = reaction_ticks(settings)
    slower_reaction = reaction_ticks(slower)

    one_microsecond = config.microseconds_to_ticks(1.0)
    assert slower_reaction - reaction == one_microsecond


def committed_rounds(contributions: dict) -> list:
    """Every round some owner of the logical ledger commits, with repeats."""
    rounds = []
    for contribution in contributions.values():
        last = contribution.commit_hi + 1
        rounds.extend(range(contribution.commit_lo, last))
    return sorted(rounds)


def frame_commits_by_window(shot: collect.Shot) -> dict:
    """How many corrections the Pauli frame took from each window."""
    committed = shot.machine.observation.frame_corrections.committed
    window_keys = []
    for record in committed:
        window_keys.append(record.window_key)
    counts = collections.Counter(window_keys)
    return dict(counts)


@pytest.mark.parametrize(
    "shape",
    (weak_only, strong_only, switching_on_double_windows),
    ids=("weak_only", "strong_only", "switching_on_double_windows"),
)
def test_every_round_has_one_owner_and_the_frame_takes_each_owner_once(shape):
    """Every round is decoded once and every window committed once.

    The run's logical ledger tiles the shot's 30 rounds with its owners,
    none twice and none left out, and the frame takes one correction from
    each owner and none from any other window. On double windows every
    window escalates and its strong region absorbs its neighbours, so the
    owners are strong regions.
    """
    settings = shape(PHYSICAL_ERROR_PROBABILITY)

    shot = run(settings)

    contributions = shot.machine.windows.ledger.contributions
    owners_once = dict.fromkeys(contributions, 1)
    assert committed_rounds(contributions) == list(range(1, 31))
    assert frame_commits_by_window(shot) == owners_once
