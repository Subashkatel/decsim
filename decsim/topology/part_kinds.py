"""The kinds of part a machine holds, and the ports each kind declares.

A part is a decsim settings record, and its kind is the slot of
MachineSettings whose type takes the record (parameters/kinds.py
admits). So a researcher's own record joins a kind by meeting the
slot's type, never by a list of classes here (STYLE.md rule 10).

A kind declares its ports by name, each with a direction and the signal
it carries, as a gem5 SimObject declares each Port with a role
(gem5 src/python/m5/params/port_params.py:316-341). A wire joins an
output to an input of the same signal, the pairing gem5's compat table
allows between a requestor and a responder (port_params.py:321-330,
362). A port takes one wire unless it takes many, as a gem5 VectorPort
does (port_params.py:284-300).
"""

import dataclasses
import functools
import typing
from typing import Any, Optional

import decsim.parameters.kinds as kinds
import decsim.settings as settings

OUTPUT = "output"
INPUT = "input"
# the signals a wire carries, from the QPU around the loop to its pulses
MEASUREMENTS = "measurements"
ROUNDS = "rounds"
# a pool's windows its edge's verdict escalates, and the re-decodes the
# edge asks of its target pool
ESCALATIONS = "escalations"
REDECODES = "redecodes"
CORRECTIONS = "corrections"
DECISIONS = "decisions"
COMMANDS = "commands"


@dataclasses.dataclass(frozen=True)
class Port:
    """One port of a kind of part: its direction and the signal it carries.

    is_many says it takes any number of wires, one otherwise; a required
    port must be wired before the machine is built.
    """

    name: str
    direction: str
    signal: str
    is_many: bool = False
    is_required: bool = True

    @property
    def role(self) -> str:
        """The port's role as a refusal names it, e.g. "rounds output"."""
        return f"{self.signal} {self.direction}"


@dataclasses.dataclass(frozen=True)
class PartKind:
    """A kind of part: the MachineSettings slots that take it, its ports."""

    name: str
    slots: tuple
    ports: tuple = ()

    def port(self, name: str) -> Optional[Port]:
        """The port of this name, None for none."""
        for port in self.ports:
            if port.name == name:
                return port
        return None

    def port_names(self) -> list:
        """Every port's name, in declaration order."""
        return [port.name for port in self.ports]


# The parts on the readout's path, each port in the order a round meets
# it. The controller sends rounds to every store, and the frame takes
# corrections from every pool. A pool's results reach a switching part,
# an escalate edge, whose verdict sends a window it escalates on to the
# re-decode of the pool it targets; a pool takes at most one edge in and
# one out, so the pools form a tree from the pool that decodes first.
_QPU_PORTS = (
    Port("readout", OUTPUT, MEASUREMENTS),
    Port("pulses", INPUT, COMMANDS),
)
_CONTROLLER_PORTS = (
    Port("readout", INPUT, MEASUREMENTS),
    Port("rounds", OUTPUT, ROUNDS, is_many=True),
    Port("decisions", INPUT, DECISIONS, is_required=False),
    Port("pulses", OUTPUT, COMMANDS),
)
_SYNDROME_BUFFER_PORTS = (
    Port("rounds", INPUT, ROUNDS),
    Port("reads", OUTPUT, ROUNDS, is_required=False),
)
_DECODER_POOL_PORTS = (
    Port("rounds", INPUT, ROUNDS),
    Port("corrections", OUTPUT, CORRECTIONS, is_required=False),
    Port("escalate", OUTPUT, ESCALATIONS, is_required=False),
    Port("escalated", INPUT, REDECODES, is_required=False),
)
# A switching record set on a run with no second pool escalates
# nowhere, so its ports are not required.
_SWITCHING_PORTS = (
    Port("results", INPUT, ESCALATIONS, is_required=False),
    Port("redecode", OUTPUT, REDECODES, is_required=False),
)
# A frame with no decoder to correct it holds the frame of a run that
# decodes nothing, and a pool with no frame to correct decodes for its
# timing alone.
_PAULI_FRAME_PORTS = (
    Port("corrections", INPUT, CORRECTIONS, is_many=True, is_required=False),
    Port("decisions", OUTPUT, DECISIONS),
)
# The parts off the path hold no port: their records set how the parts
# on it behave.
KINDS = (
    PartKind("qpu", ("qpu",), _QPU_PORTS),
    PartKind("controller", ("controller",), _CONTROLLER_PORTS),
    PartKind(
        "syndrome_buffer",
        ("weak_syndrome_buffer", "strong_syndrome_buffer"),
        _SYNDROME_BUFFER_PORTS,
    ),
    PartKind(
        "decoder_pool", ("weak_decoder", "strong_decoder"), _DECODER_POOL_PORTS
    ),
    PartKind("pauli_frame", ("pauli_frame",), _PAULI_FRAME_PORTS),
    PartKind("idle_policy", ("idle_policy",)),
    PartKind("detection_events", ("detection_events",)),
    PartKind("windows", ("windows",)),
    PartKind("decoder_manager", ("decoder_manager",)),
    PartKind("switching", ("switching",), _SWITCHING_PORTS),
    PartKind("workload", ("workload",)),
    PartKind("magic_state_factory", ("magic_state_factory",)),
)


@functools.cache
def kind_of(record_class: type) -> Optional[PartKind]:
    """The kind whose slot takes records of this class; None for none."""
    for kind in KINDS:
        hint = slot_hint(kind)
        if kinds.admits(hint, record_class):
            return kind
    return None


def slot_hint(kind: PartKind) -> Any:
    """The type of the kind's first slot, which every one of its slots meets.

    A part's record swapped by name is found among the records this type
    takes (parameters/tree.py).
    """
    hints = _machine_hints()
    first_slot = kind.slots[0]
    return hints[first_slot]


def run_file_fields(kind: PartKind) -> tuple:
    """The fields of the kind's record set in the run file only.

    MachineSettings names them by slot (qpu.distance); a part of the kind
    names them by its own name.
    """
    found = []
    for path in settings.MachineSettings.SET_IN_THE_RUN_FILE:
        slot, _, field = path.partition(".")
        if slot in kind.slots:
            found.append(field)
    return tuple(found)


@functools.cache
def _machine_hints() -> dict:
    return typing.get_type_hints(settings.MachineSettings)
