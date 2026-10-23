"""The kinds of part a machine holds, and the ports each kind declares.

A part is a decsim settings record, and its kind is the one whose
record type takes it (parameters/kinds.py admits). So a researcher's
own record joins a kind by meeting the kind's type, never by a list of
classes here (STYLE.md rule 10). A kind a machine holds once has a
default record, which a machine with no part of the kind runs on.

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
from typing import Any, Optional, Union

import decsim.controller.policies as idle_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.settings as decoder_settings
import decsim.detector_error_model.settings as detection_event_settings
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.parameters.kinds as kinds
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.ports as ports
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings

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
WORKLOAD = "workload"
QPU = "qpu"
SWITCHING = "switching"


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
    """A kind of part: the record type it takes, its ports, its default.

    hint is the type a part's record must meet, and a record swapped in
    by name is found among the records it takes (parameters/tree.py).
    is_many says a machine may hold any number of the kind; one held
    once runs on default when it holds none, and None is no such part.
    """

    name: str
    hint: Any
    ports: tuple = ()
    is_many: bool = False
    default: Any = None

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
# re-decode of the pool it targets. A pool sends to at most one edge,
# since no design sends one result to two decoders, but takes any number
# in: one strong pool may serve every weak pool, as Delfosse's decoding
# units serve every lazy decoder (2001.11427 Sec. V) and Toshio's single
# strong decoder its N_window weak ones (2510.25222 Sec. III C).
_QPU_PORTS = (
    Port("readout", OUTPUT, MEASUREMENTS),
    Port("pulses", INPUT, COMMANDS),
)
# A controller whose rounds reach no store runs a machine with no pool,
# which decodes nothing on one idle pool (graph_layout.py layout_of).
_CONTROLLER_PORTS = (
    Port("readout", INPUT, MEASUREMENTS),
    Port("rounds", OUTPUT, ROUNDS, is_many=True, is_required=False),
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
    Port("escalated", INPUT, REDECODES, is_many=True, is_required=False),
)
# A router sits between the entry store and the entry pools: the store's
# reads reach it, and its windows reach every pool it routes to.
_ROUTER_PORTS = (
    Port("rounds", INPUT, ROUNDS),
    Port("windows", OUTPUT, ROUNDS, is_many=True),
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
# on it behave. The defaults are a timing-only run of distance-three
# surface code patches with no frame.
_DEFAULT_QPU = qpu_settings.QpuSettings()
_DEFAULT_CONTROLLER = controller_settings.ControllerSettings()
_DEFAULT_IDLE_POLICY = idle_policies.SeparateDecodeJobsSettings()
_DEFAULT_DETECTION_EVENTS = detection_event_settings.DetectionEventSettings()
_DEFAULT_WINDOWS = window_settings.WindowSettings()
_DEFAULT_DECODER_MANAGER = decoder_settings.DecoderManagerSettings()
_DEFAULT_WORKLOAD = workload_settings.WorkloadSettings()
_DEFAULT_FACTORY = magic_state_factories.InfiniteFactory.Settings()
KINDS = (
    PartKind(
        QPU,
        qpu_settings.QpuSettings,
        _QPU_PORTS,
        default=_DEFAULT_QPU,
    ),
    PartKind(
        "controller",
        controller_settings.ControllerSettings,
        _CONTROLLER_PORTS,
        default=_DEFAULT_CONTROLLER,
    ),
    PartKind(
        "syndrome_buffer",
        ports.SyndromeBufferSettings,
        _SYNDROME_BUFFER_PORTS,
        is_many=True,
    ),
    PartKind(
        "decoder_pool",
        decoder_settings.DecoderPoolSettings,
        _DECODER_POOL_PORTS,
        is_many=True,
    ),
    PartKind(
        "pauli_frame", pauli_frame_module.PauliFrameConfig, _PAULI_FRAME_PORTS
    ),
    PartKind(
        "idle_policy",
        Union[
            idle_policies.IgnoreSettings,
            idle_policies.SeparateDecodeJobsSettings,
            ports.IdlePolicySettings,
        ],
        default=_DEFAULT_IDLE_POLICY,
    ),
    PartKind(
        "detection_events",
        detection_event_settings.DetectionEventSettings,
        default=_DEFAULT_DETECTION_EVENTS,
    ),
    PartKind(
        "windows",
        window_settings.WindowSettings,
        default=_DEFAULT_WINDOWS,
    ),
    PartKind(
        "decoder_manager",
        decoder_settings.DecoderManagerSettings,
        default=_DEFAULT_DECODER_MANAGER,
    ),
    PartKind(
        SWITCHING,
        escalation_settings.SwitchingSettings,
        _SWITCHING_PORTS,
        is_many=True,
    ),
    PartKind("router", routers.Router.Settings, _ROUTER_PORTS),
    PartKind(
        WORKLOAD,
        Union[workload_settings.WorkloadSettings, ports.WorkloadMakerSettings],
        default=_DEFAULT_WORKLOAD,
    ),
    PartKind(
        "magic_state_factory",
        Union[qpu_settings.FactorySettings, ports.MagicStateFactorySettings],
        default=_DEFAULT_FACTORY,
    ),
)
_KIND_BY_NAME = {kind.name: kind for kind in KINDS}


def kind_named(name: str) -> PartKind:
    """The kind of this name."""
    return _KIND_BY_NAME[name]


@functools.cache
def kind_of(record_class: type) -> Optional[PartKind]:
    """The kind whose type takes records of this class; None for none.

    A kind that names the class takes it before one whose Protocol it
    only fills, so a record with a build of the same shape as another
    kind's port stays its own kind.
    """
    for kind in KINDS:
        if kinds.names(kind.hint, record_class):
            return kind
    for kind in KINDS:
        if kinds.admits(kind.hint, record_class):
            return kind
    return None


def run_file_fields(kind: PartKind, workload: object) -> tuple:
    """The fields of the kind's record that the machine's workload fixes.

    A workload made at a distance of its own fixes the QPU's distance,
    since another would leave the circuit behind, a distance-5 machine
    running a distance-3 circuit; a lowered one fixes its lowered
    fields. A maker row that follows the QPU fixes nothing.
    """
    if kind.name == QPU and workload.fixes_the_distance:
        return ("distance",)
    if kind.name == WORKLOAD:
        fields = getattr(workload, "fields_set_in_the_run_file", ())
        return tuple(fields)
    return ()
