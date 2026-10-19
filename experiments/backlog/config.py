"""The backlog study's machine: Toshio et al.'s App. C on a memory stream.

    machine(0.4, window_rate(1e-3, 21), 21)

Both tiers are Toshio's linear decoder (2510.25222 Sec. III C): a decode
of r rounds costs tau_dec r, and a round reaches the weak decoder
T_weak_comm after it is measured. Each window commits r_com rounds and
reads d more (App. C's sliding windows, Fig. 19); the weak decoder
answers it, and an escalated window is redone on the strong decoder,
the next window waiting for its answer (strong_window_shapes.RedoWindow).
A window escalates at random (verdicts.Bernoulli), so the escalation
rate is set directly. The source is timing-only: no syndrome is sampled,
since no decision here reads one. Every number is in constants.yaml.

The queue's load is one window's mean service over the r_com tau_gen
between windows; the backlog grows without bound above load 1.
"""

import dataclasses
import math
import pathlib

import decsim.confidence.no_signal as no_signal
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.escalation.verdicts as verdicts
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.topology.constants as constants_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph

HERE = pathlib.Path(__file__)
CONSTANTS_PATH = HERE.with_name("constants.yaml")
constants = constants_module.load(CONSTANTS_PATH)
# a timing-only stream needs a probability to build its memory, never reads it
UNREAD_PROBABILITY = 0.001


def machine(
    weak_fraction: float, rate: float, commit_rounds: int
) -> graph.Machine:
    """Weak at tau_weak = weak_fraction tau_gen, escalating at rate a window."""
    period = constants.round_period_microseconds
    weak_per_round = weak_fraction * period
    weak = decoder_settings.linear_decoder_pool(
        weak_per_round, clocks.FRIDGE_CLOCK, solves_per_window=1
    )
    strong = decoder_settings.linear_decoder_pool(
        constants.strong_decode_microseconds_per_round,
        clocks.ROOM_CLOCK,
        solves_per_window=1,
    )
    stream = timing_stream(commit_rounds)
    weak_tier = stream.with_record(machines.WEAK_POOL, weak)
    switching = random_switching(rate)
    weak_link = fabrics.TimedLink(
        constants.weak_communication_microseconds,
        excludes_receiver_processing=True,
    )
    strong_link = fabrics.TimedLink(
        constants.strong_communication_microseconds,
        excludes_receiver_processing=True,
    )
    weak_side = fabrics.RoleLinks(
        write_written=weak_link, base=weak_tier.fabric
    )
    fabric = link_cards.strong_side(weak_side, strong_link)
    return machines.with_strong_tier(weak_tier, strong, switching, fabric)


def timing_stream(commit_rounds: int) -> graph.Machine:
    """The weak baseline on a timing-only stream of windows r_com + d."""
    distance = constants.distance
    period = constants.round_period_microseconds
    rounds_per_distance = constants.horizon_rounds // distance
    memory = producers.MemorySettings(
        UNREAD_PROBABILITY, rounds_per_distance=rounds_per_distance
    )
    timing_only = syndrome_devices.TimingOnlyDevice.Settings()
    base = machines.weak_baseline(distance, UNREAD_PROBABILITY, period)
    stream = base.with_held("workload", memory)
    stream = stream.with_settings("qpu", source=timing_only)
    windows = stream.held("windows")
    scheme = dataclasses.replace(
        windows.scheme, commit_rounds=commit_rounds, buffer_rounds=distance
    )
    return stream.with_settings("windows", scheme=scheme)


def random_switching(rate: float) -> escalation_settings.SwitchingSettings:
    """Escalate each window at rate; no gap is computed or read."""
    bernoulli = verdicts.Bernoulli.Settings(rate=rate)
    no_confidence = no_signal.NoSignal.Settings()
    unread = threshold_sources.FixedThreshold.Settings(threshold_decibels=0.0)
    return escalation_settings.SwitchingSettings(
        confidence=no_confidence, threshold=unread, verdict=bernoulli
    )


def commit_rounds(weak_fraction: float) -> int:
    """App. C's r_com: max(d, tau_weak r_buf / (0.95 tau_gen - tau_weak))."""
    distance = constants.distance
    room = constants.commit_margin - weak_fraction
    needed = weak_fraction * distance / room
    # a value a float puts a hair over a whole round is that round
    whole = round(needed, 9)
    rounded = math.ceil(whole)
    return max(distance, rounded)


def window_rate(gamma: float, commit: int) -> float:
    """A window's escalation chance at gamma per d rounds it reads.

    The weak decoder reads r_com + d rounds a window, so a window holds
    (r_com + d) / d of the d-round blocks gamma_switch is defined on
    (2510.25222 Sec. III C). Theorem 1's proof counts only the r_com
    committed rounds; this reading is the one Fig. 21's thresholds
    follow (README.md).
    """
    distance = constants.distance
    blocks = (commit + distance) / distance
    keep = 1.0 - gamma
    return 1.0 - keep**blocks


def load(weak_fraction: float, rate: float, commit: int) -> float:
    """A window's mean service over the time between windows.

    Kept, the weak decode of the r_com + d rounds; escalated, that, the
    hop up and the strong decode of the same rounds, as App. C's
    Fig. 19 redoes them.
    """
    period = constants.round_period_microseconds
    rounds = commit + constants.distance
    weak = weak_fraction * period * rounds
    strong_decode = constants.strong_decode_microseconds_per_round * rounds
    strong = constants.strong_communication_microseconds + strong_decode
    service = weak + rate * strong
    return service / (commit * period)
