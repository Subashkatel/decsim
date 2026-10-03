"""The controller's settings, and the yaml's idle policy section.

The controller charges three per-round costs and bounds its packing
workspace; the idle policy says how an idle patch's rounds are charged.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Union

import decsim.config as config
import decsim.controller.policies as policies
import decsim.tables as tables

# idle_policy.kind names one of these records: what the controller does
# with the rounds of a patch that is idle.
IDLE_POLICIES = {
    "separate_decode_jobs": policies.SeparateDecodeJobsSettings,
    "ignore": policies.IgnoreSettings,
}
# The key the idle_policy section reads for itself; any other key is a
# field of the record its kind names.
_IDLE_POLICY_KEYS = ("kind",)

# The controller section's keys.
_CONTROLLER_KEYS = (
    "clock",
    "readout_to_bits_cycles",
    "packing_cycles_per_round",
    "decision_to_pulse_cycles",
    "packing_rounds_in_flight",
)
# The keys with no default: a controller card states its clock and its
# three per-round costs.
_REQUIRED_CONTROLLER_KEYS = (
    "clock",
    "readout_to_bits_cycles",
    "packing_cycles_per_round",
    "decision_to_pulse_cycles",
)


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
    packing stage at once, each from its emission, in emission order,
    until the windows hear of it (round_assembly.RoundsInFlight); None is
    unbounded.
    clock is the domain all three cycle counts are charged on; a cost of
    zero cycles is uncharged rather than rounded up to the next edge.
    clock None is the machine's clock.
    """

    clock: Optional[config.Clock] = None
    readout_to_bits_cycles: int = 0
    packing_cycles_per_round: int = 0
    decision_to_pulse_cycles: int = 0
    packing_rounds_in_flight: Optional[int] = None

    def __post_init__(self) -> None:
        config.check_cycles(
            "controller.readout_to_bits_cycles", self.readout_to_bits_cycles
        )
        config.check_cycles(
            "controller.packing_cycles_per_round",
            self.packing_cycles_per_round,
        )
        config.check_cycles(
            "controller.decision_to_pulse_cycles",
            self.decision_to_pulse_cycles,
        )
        self._check_rounds_in_flight()

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "ControllerSettings":
        """The `controller` section: its cycle counts, and its clock."""
        tables.refuse_unknown_keys("controller", section, _CONTROLLER_KEYS)
        tables.refuse_missing_keys(
            "controller", section, _REQUIRED_CONTROLLER_KEYS
        )
        clock = clocks.clock(section["clock"])
        readout_cycles = section["readout_to_bits_cycles"]
        packing_cycles = section["packing_cycles_per_round"]
        decision_cycles = section["decision_to_pulse_cycles"]
        packing_rounds_in_flight = section.get("packing_rounds_in_flight")
        return cls(
            clock=clock,
            readout_to_bits_cycles=readout_cycles,
            packing_cycles_per_round=packing_cycles,
            decision_to_pulse_cycles=decision_cycles,
            packing_rounds_in_flight=packing_rounds_in_flight,
        )

    def _check_rounds_in_flight(self) -> None:
        """The packing stage's bound is a whole count of rounds, or null.

        A round enters the stage whole, so the bound counts whole rounds,
        and a bound below one admits no round, so every round would wait
        for ever. gem5's integer parameters refuse a value outside
        their range where the configuration is read
        (src/python/m5/params/param_types.py:230-235, CheckedInt._check).
        """
        bound = self.packing_rounds_in_flight
        if bound is None:
            return
        if config.is_whole_count(bound):
            return
        raise ValueError(
            "controller.packing_rounds_in_flight must be a whole count of "
            f"rounds, at least one, or null for no bound (got {bound!r})"
        )


# decsim's one controller clock, 250 MHz: Yang et al.'s loop (2605.04892
# line 1063) and Caune et al.'s sequencer (2410.05202 lines 1021-1022).
_CLOCK_250_MEGAHERTZ = config.Clock.from_megahertz(250.0)
# QubiC's issue pipeline from the decision to the pulse trigger
# (2404.15260 lines 173-191), counted with gem5's MinorCPU stage delays
# where the paper gives none. The count is QubiC's; decsim runs it on its
# 250 MHz controller clock, while QubiC's cores run at 500 MHz (lines
# 716-717). Readout and packing sit inside the round.
QUBIC_ISSUE = ControllerSettings(
    clock=_CLOCK_250_MEGAHERTZ,  # estimate, QubiC runs 500 MHz
    readout_to_bits_cycles=0,
    packing_cycles_per_round=0,
    decision_to_pulse_cycles=8,  # estimate traced from the paper
)
# QICK's tProcessor: 16 clocks for the conditional and the jump and 20
# for the next pulse (2110.00557 lines 896-900), on its 384 MHz clock
# (line 624).
_QICK_CLOCK = config.Clock.from_megahertz(384.0)
QICK_ISSUE = ControllerSettings(
    clock=_QICK_CLOCK,
    readout_to_bits_cycles=0,
    packing_cycles_per_round=0,
    decision_to_pulse_cycles=36,
)
# A RISC-Q leaf (Liu et al. 2603.16203) issuing on QubiC's pipeline: the
# syndrome aggregator packs a round in 29 ns (lines 894-895), rounded up
# to 250 MHz cycles; RISC-Q gives no issue count. decsim runs the leaf on
# its 250 MHz controller clock, while RISC-Q's leaves run at 500 MHz
# (line 143).
RISC_Q_LEAF = ControllerSettings(
    clock=_CLOCK_250_MEGAHERTZ,  # estimate, RISC-Q leaves run 500 MHz
    readout_to_bits_cycles=0,  # in the readout hop, 2605.04892 lines 1053-1055
    packing_cycles_per_round=8,  # 29 ns, 2603.16203 lines 894-895
    decision_to_pulse_cycles=QUBIC_ISSUE.decision_to_pulse_cycles,
)


def idle_policy_from_yaml(
    section: Mapping,
) -> Union[policies.IgnoreSettings, policies.SeparateDecodeJobsSettings]:
    """The `idle_policy` section: the settings record its kind names.

    Idle rounds are decoder workload, because the backlog bound counts
    every generated syndrome bit against the decoder's processing rate
    (Terhal 1302.3428 lines 3151-3159; Battistel et al. 2303.00054 line
    144), so separate_decode_jobs is the default; ignore is the
    optimistic card for active-path latency studies.
    """
    kind = section.get("kind", "separate_decode_jobs")
    record = tables.row(IDLE_POLICIES, "idle_policy.kind", kind)
    values = tables.record_fields(
        record, "idle_policy", section, _IDLE_POLICY_KEYS
    )
    return record(**values)
