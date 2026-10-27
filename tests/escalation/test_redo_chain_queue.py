"""The redo window's chain is one server: Lindley's recursion to the tick.

With held boundaries a window starts when its data is complete and the
window before it has committed, so its commit is

    F_k = max(F_{k-1} + b, A_k + h) + S_k

(Lindley 1952), A_k its data complete, h the hop into the weak unit, b
the boundary handoff and S_k its service: the weak decode, and on an
escalation the forward hop, the strong decode and the return. The
machine is Toshio et al.'s linear decoders (2510.25222 lines 968-971,
1109-1114) at tau_weak = 0.4 tau_gen, escalating i.i.d. at gamma = 0.2
(Sec. III C). A service depends on the window's tier and the one before
it (a strong predecessor's boundary is already on the host, one host
cycle from the strong buffer), never on the queue.
"""

import collections

import decsim
import decsim.confidence.no_signal as no_signal
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.topology.fabrics as fabrics

DISTANCE = 3
PROBABILITY = 0.001
ROUNDS_PER_DISTANCE = 40
GAMMA = 0.2
# tau_gen, the round period, and Toshio's decoder parameters in it
TAU_GEN_MICROSECONDS = 1.0
TAU_WEAK_MICROSECONDS = 0.4
TAU_STRONG_MICROSECONDS = 10.0
STRONG_COMMUNICATION_MICROSECONDS = 10.0
# a window reads its 3 commit and 3 buffer rounds
WINDOW_ROUNDS = 6
CYCLE_TICKS = clocks.FRIDGE_CLOCK.period_ticks
# seed 0 escalates after both tiers, and its windows wait on both terms
SEED = 0


def test_every_commit_is_lindleys_recursion():
    rows = _windows()
    first_window, first_frame = rows[0]
    data_waits = _data_waits(rows)
    chain_waits = _chain_waits(rows)
    hop = min(data_waits)
    handoff = min(chain_waits)
    services = _services(rows)

    finish = first_frame.committed_ticks
    predicted = [finish]
    chain_bound = 0
    for index in range(1, len(rows)):
        window, _frame = rows[index]
        from_chain = finish + handoff
        from_data = window.t_data_complete + hop
        if from_chain > from_data:
            chain_bound += 1
        start = max(from_chain, from_data)
        finish = start + services[_service_key(rows, index)]
        predicted.append(finish)

    committed = [frame.committed_ticks for _window, frame in rows]
    assert predicted == committed
    assert hop == CYCLE_TICKS
    assert handoff == 0
    assert 0 < chain_bound < len(rows) - 1


def test_a_service_is_toshios_linear_decode_and_fixed_hops():
    """Kept: tau_weak r and two cycles; escalated: T_comm, tau_strong r more."""
    rows = _windows()
    services = _services(rows)
    weak_microseconds = TAU_WEAK_MICROSECONDS * WINDOW_ROUNDS
    strong_microseconds = TAU_STRONG_MICROSECONDS * WINDOW_ROUNDS
    weak_ticks = _ticks(weak_microseconds)
    strong_ticks = _ticks(strong_microseconds)
    forward_ticks = _ticks(STRONG_COMMUNICATION_MICROSECONDS)
    kept = weak_ticks + 2 * CYCLE_TICKS
    escalated = kept + forward_ticks + strong_ticks + CYCLE_TICKS

    assert services[("weak", "weak", False)] == kept
    assert services[("strong", "weak", False)] == kept
    assert services[("weak", "strong", False)] == escalated
    assert services[("strong", "strong", False)] == escalated + CYCLE_TICKS


def _windows() -> list:
    """Every window with its frame record, in order, from one long shot."""
    machine = _machine()
    task = decsim.Task("redo_chain", machine, {})
    shot = collect.run_shot(task, SEED)
    return measure.decoded_windows(shot.machine.observation)


def _machine():
    """Toshio's linear decoders, the weak hop tau_gen, escalating at gamma."""
    bernoulli = verdicts.Bernoulli.Settings(rate=GAMMA)
    unread = threshold_sources.FixedThreshold.Settings(threshold_decibels=0.0)
    no_confidence = no_signal.NoSignal.Settings()
    switching = escalation_settings.SwitchingSettings(
        confidence=no_confidence,
        threshold=unread,
        verdict=bernoulli,
    )
    weak = decoder_settings.linear_decoder_pool(
        TAU_WEAK_MICROSECONDS, clocks.FRIDGE_CLOCK, solves_per_window=1
    )
    strong = decoder_settings.linear_decoder_pool(
        TAU_STRONG_MICROSECONDS, clocks.ROOM_CLOCK, solves_per_window=1
    )
    memory = producers.MemorySettings(
        PROBABILITY, rounds_per_distance=ROUNDS_PER_DISTANCE
    )
    base = machines.weak_baseline(DISTANCE, PROBABILITY, TAU_GEN_MICROSECONDS)
    long_shot = base.with_held("workload", memory)
    weak_tier = long_shot.with_record(machines.WEAK_POOL, weak)
    weak_link = fabrics.TimedLink(
        TAU_GEN_MICROSECONDS, excludes_receiver_processing=True
    )
    strong_link = fabrics.TimedLink(
        STRONG_COMMUNICATION_MICROSECONDS, excludes_receiver_processing=True
    )
    weak_side = fabrics.RoleLinks(
        write_written=weak_link, base=weak_tier.fabric
    )
    fabric = link_cards.strong_side(weak_side, strong_link)
    return machines.with_strong_tier(weak_tier, strong, switching, fabric)


def _data_waits(rows) -> list:
    """Each window's compute start after its data was complete."""
    return [
        window.t_compute_start - window.t_data_complete for window, _ in rows
    ]


def _chain_waits(rows) -> list:
    """Each window's compute start after the one before it committed."""
    waits = []
    for index in range(1, len(rows)):
        window, _frame = rows[index]
        _before, frame_before = rows[index - 1]
        wait = window.t_compute_start - frame_before.committed_ticks
        waits.append(wait)
    return waits


def _services(rows) -> dict:
    """Compute start to commit, by _service_key; each key has one value."""
    by_key = collections.defaultdict(set)
    for index in range(len(rows)):
        window, frame = rows[index]
        key = _service_key(rows, index)
        service = frame.committed_ticks - window.t_compute_start
        by_key[key].add(service)
    services = {}
    for key, values in by_key.items():
        assert len(values) == 1, (key, values)
        (services[key],) = values
    return services


def _service_key(rows, index: int) -> tuple:
    """(tier before, tier, is the last window): the last reads fewer rounds."""
    _window, frame = rows[index]
    before = None
    if index > 0:
        _before, frame_before = rows[index - 1]
        before = frame_before.tier
    is_last = index == len(rows) - 1
    return (before, frame.tier, is_last)


def _ticks(microseconds: float) -> int:
    return config.microseconds_to_ticks(microseconds)
