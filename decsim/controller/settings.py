"""The controller's settings, and the idle policy it relays through.

The controller charges three per-round costs and bounds its packing
workspace; the idle policy says how an idle patch's rounds are charged.
"""

import dataclasses
import enum
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.controller.policies as policies
import decsim.ports as ports

# idle_policy.kind names one of these rows: what the controller does
# with the rounds of a patch that is idle.
IDLE_POLICIES = {
    "separate_decode_jobs": policies.SeparateDecodeJobs,
    "ignore": policies.Ignore,
    "extend_stream": policies.ExtendStream,
}

# controller.detection_events_formed_at names which width crosses the
# store and the tier's input link, and the row is whether the round
# leaves the controller sized by its detection events. At the
# controller, "inside the workstation, measurements are converted into
# detections and then streamed to the real-time decoding software via a
# shared memory buffer" (Google 2408.13687 lines 474-476). At the
# decoder, the controller writes the raw outcomes "sequentially to the
# decoder" and "the decoder computes the syndrome from measurement
# outcomes" (Caune et al. 2410.05202 lines 1252-1256), so the store and
# the tier's input link carry the wider raw round. The values are the
# device's under both rows, computed at the assembler because the
# formation table is stateful and reads every round once in round order,
# and no formation time is charged in either row: a decoder-side
# formation unit that forms each round on first demand is a component of
# its own and is not built.
DETECTION_EVENT_FORMATION = {"controller": True, "decoder": False}


class PackingOverflowPolicy(enum.Enum):
    """What the controller does with a finished round its store cannot take.

    STALL holds the round upstream of the store until a slot frees and
    writes it in order, the backpressure real-time systems apply to their
    source: the Rigetti sequencer polls the decoder's status register and
    stalls (Caune et al. 2410.05202), Helios's input port is valid/ready
    and asserts ready only when it can take data, QubiC's cores block in
    WAIT_MEAS, and credit-based flow control never loses a flit. LILLIPUT's
    readout buffer and QubiC's measurement register instead overwrite the
    latest value, a storage choice this policy does not model. DROP_ROUND
    drops it, ns-3's drop tail (point-to-point-net-device.cc Send:
    Enqueue false, the packet is dropped); it applies to the packing
    stage's bound too, which under STALL stops the run when full.
    """

    STALL = "stall"
    DROP_ROUND = "drop_round"


@dataclasses.dataclass(frozen=True)
class ControllerSettings:
    """The yaml's `controller` section, in microseconds.

    The three costs are charged per round on the way in (readout to bits,
    packing) and per decision on the way out (decision to pulse); zero
    means the work sits inside the round period, as Google's 921 ns cycle
    holds its 500 ns measurement (2207.06431). Points: 40 ns in-FPGA
    discrimination (Fermilab 2406.18807); 20 ns to compute a syndrome
    from the bit strings (Yang 2605.04892); a 42 ns conditional jump and
    a 52 ns next pulse on QICK (2110.00557 Table II), 125 ns at USTC
    (2110.07965), 155 ns root to leaf in Liu et al. (2603.16203).
    packing_rounds_in_flight bounds the rounds in flight through the
    packing stage at once, each from its first fragment until the windows
    hear of it (round_assembly.RoundsInFlight); None is unbounded.
    packing_overflow is what happens to a finished round the store cannot
    take: the yaml's stall or drop_round.
    detection_events_formed_at names a row of
    DETECTION_EVENT_FORMATION: which width the store and the decoder's
    input link carry, the round's detection events or its raw
    measurement outcomes. The values are computed at the assembler under
    both rows and no formation time is charged in either.
    """

    readout_to_bits_microseconds: float = 0.0
    packing_microseconds_per_round: float = 0.0
    decision_to_pulse_microseconds: float = 0.0
    packing_rounds_in_flight: Optional[int] = None
    packing_overflow: PackingOverflowPolicy = PackingOverflowPolicy.STALL
    detection_events_formed_at: str = "controller"

    def __post_init__(self) -> None:
        config.check_duration(
            "readout_to_bits_microseconds", self.readout_to_bits_microseconds
        )
        config.check_duration(
            "packing_microseconds_per_round",
            self.packing_microseconds_per_round,
        )
        config.check_duration(
            "decision_to_pulse_microseconds",
            self.decision_to_pulse_microseconds,
        )

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "ControllerSettings":
        """The `controller` section: cycles of its clock, resolved once."""
        clock = section["clock"]
        readout_cycles = section["readout_to_bits_cycles"]
        packing_cycles = section["packing_cycles_per_round"]
        decision_cycles = section["decision_to_pulse_cycles"]
        readout_microseconds = clocks.microseconds(readout_cycles, clock)
        packing_microseconds = clocks.microseconds(packing_cycles, clock)
        decision_microseconds = clocks.microseconds(decision_cycles, clock)
        packing_rounds_in_flight = section.get("packing_rounds_in_flight")
        packing_overflow = _packing_overflow(section)
        formed_at = section.get("detection_events_formed_at", "controller")
        return cls(
            readout_to_bits_microseconds=readout_microseconds,
            packing_microseconds_per_round=packing_microseconds,
            decision_to_pulse_microseconds=decision_microseconds,
            packing_rounds_in_flight=packing_rounds_in_flight,
            packing_overflow=packing_overflow,
            detection_events_formed_at=formed_at,
        )

    def readout_to_bits_ticks(self) -> int:
        """The readout classification cost in ticks."""
        return config.microseconds_to_ticks(self.readout_to_bits_microseconds)

    def packing_ticks(self) -> int:
        """The packet assembly cost per round in ticks."""
        return config.microseconds_to_ticks(self.packing_microseconds_per_round)

    def decision_to_pulse_ticks(self) -> int:
        """The decision to pulse cost in ticks."""
        return config.microseconds_to_ticks(self.decision_to_pulse_microseconds)


@dataclasses.dataclass(frozen=True)
class IdlePolicySettings:
    """The yaml's `idle_policy` value: how an idle patch's rounds are charged.

    Table rows (IDLE_POLICIES, above): separate_decode_jobs, ignore,
    extend_stream. Idle rounds are decoder workload, because the backlog
    bound counts every generated syndrome bit against the decoder's
    processing rate (Terhal 1302.3428 lines 3151-3159; Battistel et al.
    2303.00054 line 144), so separate_decode_jobs is the default; ignore
    is the optimistic card for active-path latency studies;
    extend_stream folds them into a live stream. A Python-built policy is
    used as it is.
    """

    kind: str = "separate_decode_jobs"
    policy: Optional[ports.IdlePolicy] = None


def _packing_overflow(section: Mapping) -> PackingOverflowPolicy:
    """The `packing_overflow` word, or the default backpressure."""
    default = PackingOverflowPolicy.STALL.value
    named = section.get("packing_overflow", default)
    for policy in PackingOverflowPolicy:
        if policy.value == named:
            return policy
    words = []
    for policy in PackingOverflowPolicy:
        words.append(policy.value)
    raise ValueError(
        f"controller.packing_overflow must be one of {tuple(words)}, got "
        f"{named!r}"
    )
