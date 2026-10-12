"""Today's MachineSettings as a graph.

graph_of writes a MachineSettings record as the graph it describes, so
a machine given as the record is built, and identified, by its graph
(topology/graph_layout.py). The weak tier sits in the fridge and the
strong tier on the host; the fabric is the record's own card.
"""

import decsim.settings as machine_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph

FRIDGE = fabrics.FRIDGE
HOST = fabrics.HOST
# graph_of's pools, by name and site, and each one's read and corrections
WEAK_POOL = ("weak", FRIDGE)
STRONG_POOL = ("strong", HOST)
WEAK_POOL_WIRES = (
    graph.Wire("weak_buffer.reads", "weak.rounds"),
    graph.Wire("weak.corrections", "frame.corrections"),
)
STRONG_POOL_WIRES = (
    graph.Wire("strong_buffer.reads", "strong.rounds"),
    graph.Wire("strong.corrections", "frame.corrections"),
)
STRONG_STORE_WRITE = graph.Wire("controller.rounds", "strong_buffer.rounds")
# the weak pool's escalations reach the switching part, whose re-decodes
# reach the strong pool
ESCALATION_WIRES = (
    graph.Wire("weak.escalate", "switching.results"),
    graph.Wire("switching.redecode", "strong.escalated"),
)
_DEFAULT_BUFFER = syndrome_buffer.SyndromeBufferSettings()


def graph_of(settings: machine_settings.MachineSettings) -> graph.Machine:
    """Today's settings as a graph, which resolves to them again.

    Each slot's record is the part of the slot's name, the weak tier in
    the fridge and the strong tier on the host, and the fabric is the
    settings' own card.
    """
    tier_parts, tier_wires = _tier_parts(settings)
    single_parts = _single_parts(settings)
    shared_wires = _shared_wires(settings)
    parts = (*single_parts, *tier_parts)
    wires = (*shared_wires, *tier_wires)
    fabric = fabrics.ReferenceFabric()
    readout_routes = ()
    if settings.links is not None:
        fabric = fabrics.CardFabric(settings.links)
        readout_routes = settings.links.readout_routes
    return graph.Machine(
        clock=settings.clock,
        parts=parts,
        wires=wires,
        fabric=fabric,
        readout_routes=readout_routes,
        observation=settings.observation,
    )


def _single_parts(settings: machine_settings.MachineSettings) -> tuple:
    """The parts every machine holds once, and the frame when it has one."""
    parts = [
        graph.Part("qpu", settings.qpu),
        graph.Part("controller", settings.controller),
        graph.Part("idle_policy", settings.idle_policy),
        graph.Part("detection_events", settings.detection_events),
        graph.Part("windows", settings.windows),
        graph.Part("decoder_manager", settings.decoder_manager),
        graph.Part("workload", settings.workload),
        graph.Part("magic_state_factory", settings.magic_state_factory),
    ]
    if settings.pauli_frame is not None:
        frame = graph.Part("frame", settings.pauli_frame)
        parts.append(frame)
    if settings.switching is not None:
        switching = graph.Part("switching", settings.switching)
        parts.append(switching)
    return tuple(parts)


def _shared_wires(settings: machine_settings.MachineSettings) -> tuple:
    """The QPU's loop through the controller, and the frame's decision."""
    wires = [
        graph.Wire("qpu.readout", "controller.readout"),
        graph.Wire("controller.pulses", "qpu.pulses"),
    ]
    if settings.pauli_frame is not None:
        decision = graph.Wire("frame.decisions", "controller.decisions")
        wires.append(decision)
    return tuple(wires)


def _tier_parts(settings: machine_settings.MachineSettings) -> tuple:
    """The stores and pools of today's settings as parts, and their wires.

    The weak store is always written; the strong one is a part when a
    strong pool reads it or it differs from the default, the record a
    task's id holds. Each pool reads its tier's store and corrects the
    frame, and the weak pool escalates to the strong one.
    """
    parts = [graph.Part("weak_buffer", settings.weak_syndrome_buffer, FRIDGE)]
    wires = [graph.Wire("controller.rounds", "weak_buffer.rounds")]
    if _has_strong_store(settings):
        strong_store = graph.Part(
            "strong_buffer", settings.strong_syndrome_buffer, HOST
        )
        parts.append(strong_store)
        wires.append(STRONG_STORE_WRITE)
    tiers = (
        (settings.weak_decoder, WEAK_POOL, WEAK_POOL_WIRES),
        (settings.strong_decoder, STRONG_POOL, STRONG_POOL_WIRES),
    )
    for pool_settings, (name, placed), pool_wires in tiers:
        if pool_settings is None:
            continue
        pool = graph.Part(name, pool_settings, placed)
        parts.append(pool)
        framed = _with_frame(pool_wires, settings)
        wires.extend(framed)
    if _has_two_pools(settings):
        wires.extend(ESCALATION_WIRES)
    return tuple(parts), tuple(wires)


def _has_strong_store(settings: machine_settings.MachineSettings) -> bool:
    has_strong_pool = settings.strong_decoder is not None
    return has_strong_pool or settings.strong_syndrome_buffer != _DEFAULT_BUFFER


def _has_two_pools(settings: machine_settings.MachineSettings) -> bool:
    has_weak_pool = settings.weak_decoder is not None
    return has_weak_pool and settings.strong_decoder is not None


def _with_frame(
    pool_wires: tuple, settings: machine_settings.MachineSettings
) -> tuple:
    """A pool's wires; its corrections reach the frame where there is one."""
    read, corrections = pool_wires
    if settings.pauli_frame is None:
        return (read,)
    return (read, corrections)
