"""The controller's settings, and the idle policy it relays through.

The controller charges three per-round costs and bounds its packing
workspace; the idle policy says how an idle patch's rounds are charged.
"""

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any, Optional

import decsim.config as config
import decsim.controller.policies as policies
import decsim.ports as ports
import decsim.tables as tables
from decsim.detector_error_model import detection_event_formation

# idle_policy.kind names one of these rows: what the controller does
# with the rounds of a patch that is idle.
IDLE_POLICIES = {
    "separate_decode_jobs": policies.SeparateDecodeJobs,
    "ignore": policies.Ignore,
    "extend_stream": policies.ExtendStream,
}
# The keys every row of the idle_policy section shares; any other key is
# the row's own (its Settings, decsim/tables.py row_settings).
_IDLE_POLICY_KEYS = ("kind",)

# controller.detection_events_formed_at names one of these rows: where
# the machine turns a round's measurement outcomes into its detection
# events, and so which width crosses the store and the tier's input
# link. Each row is a component built with the run's former and the
# controller's own formation cost
# (detector_error_model/detection_event_formation.py).
DETECTION_EVENT_FORMATION = {
    "controller": detection_event_formation.ControllerSideFormation,
    "weak_syndrome_buffer": (
        detection_event_formation.WeakSyndromeBufferSideFormation
    ),
    "decoder": detection_event_formation.DecoderSideFormation,
}

# The controller section's keys.
_CONTROLLER_KEYS = (
    "clock",
    "readout_to_bits_cycles",
    "packing_cycles_per_round",
    "decision_to_pulse_cycles",
    "detection_event_cycles_per_round",
    "packing_rounds_in_flight",
    "packing_overflow",
    "detection_events_formed_at",
)
# The keys with no default: a controller card states its clock and its
# three per-round costs.
_REQUIRED_CONTROLLER_KEYS = (
    "clock",
    "readout_to_bits_cycles",
    "packing_cycles_per_round",
    "decision_to_pulse_cycles",
)


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
    """The yaml's `controller` section, in cycles of the clock it names.

    The three costs are charged per round on the way in (readout to bits,
    packing) and per decision on the way out (decision to pulse); zero
    means the work sits inside the round period, as Google's 921 ns cycle
    holds its 500 ns measurement (2207.06431). Points: 40 ns in-FPGA
    discrimination (Fermilab 2406.18807); 20 ns to compute a syndrome
    from the bit strings (Yang 2605.04892); 125 ns at USTC (2110.07965),
    155 ns root to leaf in Liu et al. (2603.16203). decision_to_pulse
    is the control processor's issue pipeline, the decision at the core
    to the pulse trigger: 8 cycles traced on QubiC's core (Fruitwala
    2404.15260 Sec. III and IV) with gem5's MinorCPU stage delays where
    the paper is silent, the result latched, the compare, the taken
    jump's redirect, the target fetched, decoded and executed, the pulse
    register written, the strobe; QICK measures 16 clocks for the
    conditional evaluation and the jump and 20 for the next pulse on its
    deeper tProcessor (2110.00557 lines 893-900). The reference yaml
    carries the trace.
    packing_rounds_in_flight bounds the rounds in flight through the
    packing stage at once, each from its first fragment until the windows
    hear of it (round_assembly.RoundsInFlight); None is unbounded.
    packing_overflow is what happens to a finished round the store cannot
    take: the yaml's stall or drop_round.
    detection_events_formed_at names a row of
    DETECTION_EVENT_FORMATION: where the round's outcomes become its
    detection events, and so which width the store and the tier's input
    link carry. detection_event_cycles_per_round is what that
    conversion costs the controller, charged once per round before the
    round leaves it and read by the controller row alone (the weak
    syndrome buffer's own charge is in its section); no paper
    publishes a controller-side figure, so it is zero by default. clock
    is the domain all four cycle counts are charged on; a cost of zero
    cycles is uncharged rather than rounded up to the next edge.
    """

    clock: Optional[config.Clock] = None
    readout_to_bits_cycles: int = 0
    packing_cycles_per_round: int = 0
    decision_to_pulse_cycles: int = 0
    detection_event_cycles_per_round: int = 0
    packing_rounds_in_flight: Optional[int] = None
    packing_overflow: PackingOverflowPolicy = PackingOverflowPolicy.STALL
    detection_events_formed_at: str = "controller"

    def __post_init__(self) -> None:
        config.check_cycles(
            "readout_to_bits_cycles", self.readout_to_bits_cycles
        )
        config.check_cycles(
            "packing_cycles_per_round", self.packing_cycles_per_round
        )
        config.check_cycles(
            "decision_to_pulse_cycles", self.decision_to_pulse_cycles
        )
        config.check_cycles(
            "detection_event_cycles_per_round",
            self.detection_event_cycles_per_round,
        )
        self._check_rounds_in_flight()
        self._check_clock()

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "ControllerSettings":
        """The `controller` section: its cycle counts, and its clock."""
        _check_section_keys(section)
        clock = clocks.clock(section["clock"])
        readout_cycles = section["readout_to_bits_cycles"]
        packing_cycles = section["packing_cycles_per_round"]
        decision_cycles = section["decision_to_pulse_cycles"]
        formation_cycles = _formation_cycles(section)
        packing_rounds_in_flight = section.get("packing_rounds_in_flight")
        packing_overflow = _packing_overflow(section)
        formed_at = section.get("detection_events_formed_at", "controller")
        tables.row(
            DETECTION_EVENT_FORMATION,
            "controller.detection_events_formed_at",
            formed_at,
        )
        return cls(
            clock=clock,
            readout_to_bits_cycles=readout_cycles,
            packing_cycles_per_round=packing_cycles,
            decision_to_pulse_cycles=decision_cycles,
            detection_event_cycles_per_round=formation_cycles,
            packing_rounds_in_flight=packing_rounds_in_flight,
            packing_overflow=packing_overflow,
            detection_events_formed_at=formed_at,
        )

    def _check_rounds_in_flight(self) -> None:
        """The packing stage's bound is a whole count of rounds, or null.

        A round enters the stage whole, so the bound counts whole rounds,
        and a bound below one admits no round and stops the run at the
        first fragment. gem5's integer parameters refuse a value outside
        their range where the configuration is read
        (src/python/m5/params/param_types.py:230-235, CheckedInt._check).
        """
        bound = self.packing_rounds_in_flight
        if bound is None:
            return
        if _is_round_count(bound):
            return
        raise ValueError(
            "controller.packing_rounds_in_flight must be a whole count of "
            f"rounds, at least one, or null for no bound (got {bound!r})"
        )

    def _check_clock(self) -> None:
        """A charged cost names the clock domain its cycles are counted on."""
        if self.clock is not None:
            return
        charged = self.readout_to_bits_cycles
        charged += self.packing_cycles_per_round
        charged += self.decision_to_pulse_cycles
        charged += self.detection_event_cycles_per_round
        if charged > 0:
            raise ValueError(
                "a charged controller cost needs the clock domain its "
                "cycles are counted on"
            )


@dataclasses.dataclass(frozen=True)
class IdlePolicySettings:
    """The yaml's `idle_policy` section: how an idle patch's rounds are charged.

    Table rows (IDLE_POLICIES, above): separate_decode_jobs, ignore,
    extend_stream. Idle rounds are decoder workload, because the backlog
    bound counts every generated syndrome bit against the decoder's
    processing rate (Terhal 1302.3428 lines 3151-3159; Battistel et al.
    2303.00054 line 144), so separate_decode_jobs is the default; ignore
    is the optimistic card for active-path latency studies;
    extend_stream folds them into a live stream. A Python-built policy is
    used as it is. row_settings is the row's own Settings, read from the
    section's keys other than kind, or None for a row that declares
    none.
    """

    kind: str = "separate_decode_jobs"
    policy: Optional[ports.IdlePolicy] = None
    # the row's own Settings record, opaque to the section
    row_settings: Optional[Any] = None

    @classmethod
    def from_yaml(cls, section: Mapping) -> "IdlePolicySettings":
        """The `idle_policy` section: a kind of the table, the row's keys.

        The section is a mapping with a kind key like every other, since
        a row owns its parameters the way a gem5 SimObject declares its
        own (src/mem/SimpleMemory.py:43-53).
        """
        kind = section.get("kind", "separate_decode_jobs")
        row = tables.row(IDLE_POLICIES, "idle_policy.kind", kind)
        row_settings = tables.row_settings(
            row, "idle_policy", section, _IDLE_POLICY_KEYS
        )
        return cls(kind=kind, row_settings=row_settings)


def _check_section_keys(section: Mapping) -> None:
    """The section names its required keys and no key it does not have.

    gem5 refuses a parameter its class does not declare
    (src/python/m5/SimObject.py:932-936), as the decoder_manager and
    escalation sections here do.
    """
    unknown = set(section) - set(_CONTROLLER_KEYS)
    if unknown:
        listed = sorted(unknown)
        raise ValueError(
            f"controller does not know {listed}; its keys are "
            f"{list(_CONTROLLER_KEYS)}"
        )
    missing = set(_REQUIRED_CONTROLLER_KEYS) - set(section)
    if missing:
        listed = sorted(missing)
        raise ValueError(
            f"controller needs the keys {listed}; configs/reference.yaml "
            "holds every key with its unit"
        )


def _is_round_count(value) -> bool:
    """A number of rounds a stage can hold: a whole count, never a flag."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, int):
        return False
    return value >= 1


def _formation_cycles(section: Mapping) -> int:
    """detection_event_cycles_per_round, the controller's own formation cost.

    null is the default and means the controller charges nothing for the
    conversion: Google's workstation converts measurements into
    detections (2408.13687 lines 474-476) and publishes no time for it.
    """
    cycles = section.get("detection_event_cycles_per_round")
    if cycles is None:
        return 0
    return cycles


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
